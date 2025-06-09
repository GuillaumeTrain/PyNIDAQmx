import sys
import threading
import time
import numpy as np
import csv
import os
import pyqtgraph as pg

from queue import Queue
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QCheckBox, QLineEdit, QHeaderView, QAbstractItemView, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QTimer

try:
    import nidaqmx
    from nidaqmx.system import System
    REAL_DAQ = True
except ImportError as e:
    print("nidaqmx ImportError. Running in simulation mode.")
    print("Détail de l'erreur :", e)
    REAL_DAQ = False
except Exception as e:
    print("Erreur inattendue à l'import de nidaqmx !")
    print("Détail :", e)
    REAL_DAQ = False

CHANNEL_NAMES = {
    'ai0': 'Catenary Voltage',
    'ai1': 'Pantograph Current',
    'ai2': 'Speed',
    'ai3': 'Torque'
}

SAMPLE_RATE = 50000      # Default simulation params
ACQ_BUF_SIZE = 500
AGG_BUF_MULT = 20
DISP_BUF_MULT = 50

STOP_EVENT = threading.Event()


# Configuration tab for selecting device and channels
class ConfigTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        # Selection device
        dev_layout = QHBoxLayout()
        self.device_combo = QComboBox()
        self.refresh_btn = QPushButton("Rafraîchir")
        dev_layout.addWidget(QLabel("Matériel NI :"))
        dev_layout.addWidget(self.device_combo)
        dev_layout.addWidget(self.refresh_btn)
        dev_layout.addStretch()
        self.layout.addLayout(dev_layout)

        # Tableau channels
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Channel ID", "Nom", "Facteur", "Offset", "Activé"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        self.layout.addWidget(self.table)

        # Sampling rate
        self.sr_edit = QLineEdit()
        self.sr_edit.setPlaceholderText("50000")
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(QLabel("Sampling rate (Hz):"))
        sr_layout.addWidget(self.sr_edit)
        self.layout.addLayout(sr_layout)

        # Buffer size en ms
        self.buf_time_edit = QLineEdit()
        self.buf_time_edit.setPlaceholderText("100")
        self.buf_time_edit.textChanged.connect(self.update_samples_label)
        buf_layout = QHBoxLayout()
        buf_layout.addWidget(QLabel("Durée buffer acquisition (ms):"))
        buf_layout.addWidget(self.buf_time_edit)
        self.samples_label = QLabel("Nombre de samples : --")
        buf_layout.addWidget(self.samples_label)
        self.layout.addLayout(buf_layout)

        # Connect events
        self.refresh_btn.clicked.connect(self.refresh_devices)
        self.device_combo.currentIndexChanged.connect(self.device_changed)
        self.refresh_devices()

    def refresh_devices(self):
        self.device_combo.clear()
        self.table.setRowCount(0)
        if REAL_DAQ:
            system = System.local()
            devs = [d.name for d in system.devices]
        else:
            devs = []
        self.device_combo.addItems(devs)

    def device_changed(self, idx):
        self.table.setRowCount(0)
        devname = self.device_combo.currentText()
        if not devname:
            return
        if REAL_DAQ:
            try:
                system = System.local()
                dev = [d for d in system.devices if d.name == devname][0]
                ai_channels = dev.ai_physical_chans
                chans = [c.name for c in ai_channels]
            except Exception as e:
                print(e)
                chans = []
        else:
            chans = [f"{devname}/ai{i}" for i in range(4)]

        self.table.setRowCount(len(chans))
        for i, ch in enumerate(chans):
            ch_id = ch.split('/')[-1]
            item_id = QTableWidgetItem(ch_id)
            item_id.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, 0, item_id)
            default_name = CHANNEL_NAMES.get(ch_id.lower(), ch_id)
            item_name = QTableWidgetItem(default_name)
            self.table.setItem(i, 1, item_name)
            item_factor = QTableWidgetItem("1.0")
            self.table.setItem(i, 2, item_factor)
            item_offset = QTableWidgetItem("0.0")
            self.table.setItem(i, 3, item_offset)
            cb = QCheckBox()
            cb.setChecked(True)
            self.table.setCellWidget(i, 4, cb)

    def update_samples_label(self):
        try:
            rate = float(self.sr_edit.text() or 50000)
            buf_ms = float(self.buf_time_edit.text() or 100)
            samples = int(rate * buf_ms / 1000)
            self.samples_label.setText(f"Nombre de samples : {samples}")
        except Exception:
            self.samples_label.setText("Nombre de samples : --")

    def get_config(self):
        dev = self.device_combo.currentText()
        chans = []
        for i in range(self.table.rowCount()):
            ch_id = self.table.item(i, 0).text()
            name = self.table.item(i, 1).text()
            factor = float(self.table.item(i, 2).text())
            offset = float(self.table.item(i, 3).text())
            cb = self.table.cellWidget(i, 4)
            active = cb.isChecked()
            chans.append({
                'id': ch_id,
                'name': name,
                'factor': factor,
                'offset': offset,
                'active': active
            })
        try:
            sr = float(self.sr_edit.text())
        except Exception:
            sr = 50000
        try:
            buf_ms = float(self.buf_time_edit.text())
        except Exception:
            buf_ms = 100
        buf_samples = int(sr * buf_ms / 1000)
        return {
            'device': dev,
            'channels': chans,
            'sampling_rate': sr,
            'buffer_time_ms': buf_ms,
            'buffer_samples': buf_samples
        }


class AcquisitionThread(threading.Thread):
    def __init__(self, config, acq_queue, disp_queue):
        super().__init__()
        self.daemon = True
        self.config = config
        self.acq_queue = acq_queue
        self.disp_queue = disp_queue
        self.task = None
        self.is_real_daq = REAL_DAQ

        if self.is_real_daq:
            # Construction de la task
            from nidaqmx.constants import AcquisitionType
            active_ch = [ch for ch in config['channels'] if ch['active']]
            channel_ids = [f"{config['device']}/{ch['id']}" for ch in active_ch]
            self.channel_names = [ch['name'] for ch in active_ch]
            self.factors = np.array([ch['factor'] for ch in active_ch], dtype=np.float64)
            self.offsets = np.array([ch['offset'] for ch in active_ch], dtype=np.float64)
            self.channel_names = [ch['name'] for ch in active_ch]
            self.sampling_rate = config['sampling_rate']
            self.buffer_samples = config['buffer_samples']
            self.task = nidaqmx.Task()
            for ch in channel_ids:
                self.task.ai_channels.add_ai_voltage_chan(ch)
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sampling_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.buffer_samples
            )
            buffer_min_s = 0.25  # 250 ms
            buffer_min_samples = int(self.sampling_rate * buffer_min_s)
            self.task.in_stream.input_buf_size = max(buffer_min_samples, self.buffer_samples * 2)
        else:
            # Simulation: utilise la même logique pour factors/offsets
            active_ch = [ch for ch in config['channels'] if ch['active']]
            self.channel_names = [ch['name'] for ch in active_ch]
            self.factors = np.array([ch['factor'] for ch in active_ch], dtype=np.float64)
            self.offsets = np.array([ch['offset'] for ch in active_ch], dtype=np.float64)

        pass

    def run(self):
        if self.is_real_daq:
            self.run_daqmx()
        else:
            self.run_simulation()

    def run_daqmx(self):
        try:
            dt = 1.0 / self.sampling_rate
            t = 0
            n_channels = len(self.channel_names)
            buffer_min_s = 1.0
            buffer_min_samples = int(self.sampling_rate * buffer_min_s)
            self.task.in_stream.input_buf_size = max(buffer_min_samples, self.buffer_samples * 2)
            print(f"Buffer hardware: {self.task.in_stream.input_buf_size} samples")
            while not STOP_EVENT.is_set():
                try:
                    data = np.array(
                        self.task.read(number_of_samples_per_channel=self.buffer_samples, timeout=10.0)
                    )
                    if n_channels == 1:
                        data = data[np.newaxis, :]
                    # **Application des facteurs/offsets ici :**
                    data = data * self.factors[:, np.newaxis] + self.offsets[:, np.newaxis]
                except nidaqmx.errors.DaqReadError as err:
                    print("Erreur NI-DAQmx acquisition :\n", err)
                    # Tu peux ici stopper la tâche proprement ou prévenir l’utilisateur (signal Qt, etc.)
                    break
                if n_channels == 1:
                    data = data[np.newaxis, :]
                times = np.arange(self.buffer_samples) * dt + t
                t += self.buffer_samples * dt
                try:
                    self.acq_queue.put((times, data), timeout=0.2)
                    self.disp_queue.put((times, data), timeout=0.2)
                except:
                    pass
        finally:
            if self.task:
                self.task.close()
        try:
            dt = 1.0 / self.sampling_rate
            t = 0
            n_channels = len(self.channel_names)
            while not STOP_EVENT.is_set():
                # Cette lecture est BLOQUANTE : attend que le buffer soit plein
                data = np.array(
                    self.task.read(number_of_samples_per_channel=self.buffer_samples, timeout=10.0)
                )
                # data.shape = (n_channels, N) ou (N,) si 1 canal
                if n_channels == 1:
                    data = data[np.newaxis, :]
                times = np.arange(self.buffer_samples) * dt + t
                t += self.buffer_samples * dt
                try:
                    self.acq_queue.put((times, data), timeout=0.2)
                    self.disp_queue.put((times, data), timeout=0.2)
                except:
                    pass
                # Plus besoin de sleep ici !
        finally:
            if self.task:
                self.task.close()

    def run_simulation(self):
        sr = self.config['sampling_rate']
        N = self.config['buffer_samples']
        dt = 1.0 / sr
        t = 0
        active_ch = [ch for ch in self.config['channels'] if ch['active']]
        channel_ids = [f"{config['device']}/{ch['id']}" for ch in active_ch]
        self.channel_names = [ch['name'] for ch in active_ch]
        self.factors = np.array([ch['factor'] for ch in active_ch], dtype=np.float64)
        self.offsets = np.array([ch['offset'] for ch in active_ch], dtype=np.float64)
        n_channels = len(active_ch)
        freq_list = [10000 + 1000*i for i in range(n_channels)]
        while not STOP_EVENT.is_set():
            times = np.arange(N) * dt + t
            data = np.stack([
                np.sin(2 * np.pi * freq_list[k] * times)
                for k in range(n_channels)
            ])
            t += N * dt
            try:
                self.acq_queue.put((times, data), timeout=0.2)
                self.disp_queue.put((times, data), timeout=0.2)
            except:
                pass
            time.sleep(N / sr)


class AggregationThread(threading.Thread):
    def __init__(self, acq_queue, agg_queue, agg_buf_mult):
        super().__init__()
        self.daemon = True
        self.acq_queue = acq_queue
        self.agg_queue = agg_queue
        self.agg_buf_mult = agg_buf_mult
        self.times_buf = []
        self.data_buf = []

    def run(self):
        while not STOP_EVENT.is_set():
            try:
                times, data = self.acq_queue.get(timeout=0.2)
                self.times_buf.append(times)
                self.data_buf.append(data)
                if len(self.times_buf) >= self.agg_buf_mult:
                    all_times = np.concatenate(self.times_buf)
                    all_data = np.concatenate(self.data_buf, axis=1)
                    self.agg_queue.put((all_times, all_data))
                    self.times_buf.clear()
                    self.data_buf.clear()
            except:
                pass


class CsvWriterThread(threading.Thread):
    def __init__(self, agg_queue, filename, channel_names):
        super().__init__()
        self.daemon = True
        self.agg_queue = agg_queue
        self.filename = filename
        self.channel_names = channel_names
        self.header_written = os.path.exists(self.filename) and os.path.getsize(self.filename) > 0

    def run(self):
        while not STOP_EVENT.is_set():
            try:
                times, data = self.agg_queue.get(timeout=0.2)
                rows = np.column_stack((times, data.T))
                with open(self.filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not self.header_written:
                        header = ['time'] + self.channel_names
                        writer.writerow(header)
                        self.header_written = True
                    writer.writerows(rows)
            except:
                pass


class AcquisitionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.control_layout = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Nom du fichier d'enregistrement (.csv)")
        self.browse_btn = QPushButton("Choisir dossier/fichier")
        self.start_btn = QPushButton("Start acquisition")
        self.stop_btn = QPushButton("Stop acquisition")
        self.stop_btn.setEnabled(False)
        self.control_layout.addWidget(self.file_edit)
        self.control_layout.addWidget(self.browse_btn)
        self.control_layout.addWidget(self.start_btn)
        self.control_layout.addWidget(self.stop_btn)
        self.layout.addLayout(self.control_layout)
        self.graphs = []
        self.curves = []
        self.last_config = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.disp_queue = None
        self.csv_file = self.default_filename()

        self.file_edit.setText(self.csv_file)

        self.browse_btn.clicked.connect(self.choose_file)
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn.clicked.connect(self.on_stop)
        self.span_layout = QHBoxLayout()
        self.span_combo = QComboBox()
        # Valeurs classiques des oscillos (en ms/div)
        self.span_values = [
            0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000  # en ms/div
        ]
        for v in self.span_values:
            if v < 1000:
                txt = f"{v:.1f} ms/div" if v < 1 else f"{int(v)} ms/div"
            else:
                txt = f"{int(v / 1000)} s/div"
            self.span_combo.addItem(txt, v)
        self.span_combo.setCurrentIndex(9)  # par défaut 100 ms/div
        self.span_layout.addWidget(QLabel("Base de temps:"))
        self.span_layout.addWidget(self.span_combo)
        self.span_layout.addStretch()
        self.layout.addLayout(self.span_layout)

        self.span_combo.currentIndexChanged.connect(self.update_plot)

        self.start_callback = None  # sera branché par le MainWindow
        self.stop_callback = None

    def default_filename(self):
        return os.path.join(os.getcwd(), "acquisition.csv")

    def choose_file(self):
        dlg = QFileDialog(self)
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setNameFilter("Fichiers CSV (*.csv)")
        dlg.setDefaultSuffix("csv")
        filename, _ = dlg.getSaveFileName(
            self, "Choisir le fichier d'enregistrement", self.file_edit.text() or self.default_filename(), "Fichiers CSV (*.csv)")
        if filename:
            self.file_edit.setText(filename)
            self.csv_file = filename

    def setup_graphs(self, config, disp_queue):
        # Nettoyage
        for g in self.graphs:
            self.layout.removeWidget(g)
            g.deleteLater()
        self.graphs.clear()
        self.curves.clear()
        #mettre le fond blanc
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        # Garde config + disp_queue
        self.last_config = config
        self.disp_queue = disp_queue
        # Affichage des plots selon les canaux actifs
        active_channels = [ch for ch in config['channels'] if ch['active']]
        self.active_channel_names = [ch['name'] for ch in active_channels]
        self.n_channels = len(active_channels)
        if not active_channels:
            self.layout.addWidget(QLabel("Aucun canal activé dans la configuration."))
            return
        self.display_times = np.zeros(config['buffer_samples'] * DISP_BUF_MULT)
        self.display_data = np.zeros((self.n_channels, config['buffer_samples'] * DISP_BUF_MULT))
        for ch in active_channels:
            plot = pg.PlotWidget(title=ch['name'])
            curve = plot.plot(pen=pg.mkPen(color= (23, 167, 114) ,width=2))
            self.layout.addWidget(plot)
            self.graphs.append(plot)
            self.curves.append(curve)

    def on_start(self):
        filename = self.file_edit.text().strip()
        if not filename:
            QMessageBox.warning(self, "Erreur", "Veuillez indiquer un nom de fichier de sauvegarde.")
            return
        if os.path.exists(filename):
            reply = QMessageBox.question(self, "Fichier existe déjà", f"Le fichier {filename} existe déjà. L'écraser ?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
        self.csv_file = filename
        # Correction ici : utilise .window() pour accéder au MainWindow !
        mainwin = self.window()
        config = mainwin.tab_config.get_config()
        self.last_config = config
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        if self.start_callback:
            self.start_callback(config, self.csv_file)
        self.start_display()

    def on_stop(self):
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.stop_display()
        if self.stop_callback:
            self.stop_callback()

    def start_display(self):
        self.display_times[:] = 0
        self.display_data[:, :] = 0
        self.timer.start(30)

    def stop_display(self):
        self.timer.stop()

    def update_plot(self):
        if self.disp_queue is None:
            return
        updated = False
        try:
            while not self.disp_queue.empty():
                times, data = self.disp_queue.get_nowait()
                n = len(times)
                sz = self.display_times.size
                if n >= sz:
                    self.display_times[:] = times[-sz:]
                    self.display_data[:, :] = data[:, -sz:]
                else:
                    self.display_times = np.roll(self.display_times, -n)
                    self.display_data = np.roll(self.display_data, -n, axis=1)
                    self.display_times[-n:] = times
                    self.display_data[:, -n:] = data
                updated = True
        except Exception as e:
            print("Erreur update_plot:", e)
        # Ici, on gère le span comme un oscilloscope
        ms_per_div = self.span_combo.currentData()
        ndiv = 10
        span_s = (ms_per_div * ndiv) / 1000

        tmax = self.display_times.max()
        tmin_buf = self.display_times[self.display_times > 0].min() if np.any(self.display_times > 0) else 0
        span_s_max = tmax - tmin_buf
        if span_s > span_s_max:
            span_s = span_s_max
        tmin = tmax - span_s
        mask = self.display_times >= tmin
        for ch, curve in enumerate(self.curves):
            curve.setData(self.display_times[mask], self.display_data[ch][mask])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Acquisition DAQ NI - Configuration dynamique")
        self.tabs = QTabWidget()
        self.tab_config = ConfigTab()
        self.tab_acq = AcquisitionTab()
        self.tabs.addTab(self.tab_config, "Configuration")
        self.tabs.addTab(self.tab_acq, "Acquisition")
        self.tabs.addTab(QLabel("Tab 3 (à compléter)"), "Tab 3")
        self.setCentralWidget(self.tabs)

        self.tab_acq.start_callback = self.on_start_acquisition
        self.tab_acq.stop_callback = self.on_stop_acquisition

        self.config = None
        self.acq_queue = Queue(maxsize=10)
        self.agg_queue = Queue(maxsize=10)
        self.disp_queue = Queue(maxsize=10)
        self.threads = []

    def on_start_acquisition(self, config, filename):
        self.config = config
        STOP_EVENT.clear()
        self.tab_acq.setup_graphs(self.config, self.disp_queue)
        self.tab_acq.start_display()
        active_channels = [ch for ch in self.config['channels'] if ch['active']]
        channel_names = [ch['name'] for ch in active_channels]
        self.threads = [
            AcquisitionThread(self.config, self.acq_queue, self.disp_queue),
            AggregationThread(self.acq_queue, self.agg_queue, AGG_BUF_MULT),
            CsvWriterThread(self.agg_queue, filename, channel_names)
        ]
        for t in self.threads:
            t.start()

    def on_stop_acquisition(self):
        STOP_EVENT.set()
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2)
        self.threads = []
        self.tab_acq.stop_display()

    def closeEvent(self, event):
        STOP_EVENT.set()
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=2)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
