import sys
import nidaqmx
from nidaqmx.constants import AcquisitionType, LoggingMode, LoggingOperation, UnitsPreScaled, VoltageUnits
from nidaqmx.system import System
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTabWidget,
    QLineEdit, QLabel, QFormLayout, QMessageBox, QFileDialog, QHBoxLayout,
    QTableView, QHeaderView
)
from PySide6.QtCore import QTimer, Qt, QAbstractTableModel, QModelIndex
import pyqtgraph as pg
import numpy as np
from collections import deque
from threading import Thread, Event
from queue import Queue, Empty
import re
from nidaqmx.scale import Scale

CHANNEL_NAMES = {
    'ai0': 'Catenary Voltage',
    'ai1': 'Pantograph Current',
    'ai2': 'Speed',
    'ai3': 'Torque'
}

def clean_channel_name(name):
    # Remplace tout caractère non alphanumérique par "_"
    return re.sub(r'\W+', '_', name)

class ChannelTableModel(QAbstractTableModel):
    def __init__(self, df):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iloc[index.row(), index.column()]
        col = self._df.columns[index.column()]
        if role == Qt.DisplayRole or role == Qt.EditRole:
            # Pour la colonne active, évite d'afficher True/False
            return "" if col == "active" else str(value)
        if role == Qt.CheckStateRole and col == "active":
            return Qt.Checked if bool(value) else Qt.Unchecked
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid():
            return False
        col = self._df.columns[index.column()]
        row = index.row()
        if role == Qt.CheckStateRole and col == "active":
            # Fixe ici : on toggle explicitement le booléen
            current = bool(self._df.iloc[row, index.column()])
            self._df.iat[row, index.column()] = not current
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            return True
        if role == Qt.EditRole:
            if col in ["scaling_coeff", "offset"]:
                try:
                    value = float(value)
                except ValueError:
                    return False
            self._df.iat[row, index.column()] = value
            self.dataChanged.emit(index, index)
            return True
        return False

    def flags(self, index):
        col = self._df.columns[index.column()]
        if col == "active":
            return Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
        else:
            return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(self._df.index[section])
        return None


class DAQReader(Thread):
    def __init__(self, task, nb_samples_per_read, data_queue, stop_event):
        super().__init__()
        self.task = task
        self.nb_samples_per_read = nb_samples_per_read
        self.data_queue = data_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                data = self.task.read(number_of_samples_per_channel=self.nb_samples_per_read, timeout=2.0)
                self.data_queue.put(data)
            except nidaqmx.errors.DaqError as e:
                print("Thread DAQmx ERROR:", e)
                break


class ConfigTab(QWidget):
    def __init__(self, channel_df, general_config):
        super().__init__()
        self.channel_df = channel_df
        self.general_config = general_config  # dict-like

        main_layout = QVBoxLayout()

        # Ligne de contrôle générale (Sampling Rate + Fichier)
        form_layout = QHBoxLayout()
        self.rate_edit = QLineEdit(str(self.general_config["rate"]))
        form_layout.addWidget(QLabel("Frequence (Hz):"))
        form_layout.addWidget(self.rate_edit)

        self.file_edit = QLineEdit(self.general_config["tdms_file"])
        self.file_btn = QPushButton("Fichier TDMS…")
        self.file_btn.clicked.connect(self.select_file)
        form_layout.addWidget(QLabel("Fichier TDMS:"))
        form_layout.addWidget(self.file_edit)
        form_layout.addWidget(self.file_btn)

        main_layout.addLayout(form_layout)

        # Bouton de détection des channels
        self.refresh_btn = QPushButton("Détecter channels DAQ")
        self.refresh_btn.clicked.connect(self.refresh_channels)
        main_layout.addWidget(self.refresh_btn)

        # Table channels
        self.table_model = ChannelTableModel(self.channel_df)
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        main_layout.addWidget(self.table_view)

        # Enregistrer
        self.save_btn = QPushButton("Enregistrer config")
        self.save_btn.clicked.connect(self.save_config)
        main_layout.addWidget(self.save_btn)
        self.setLayout(main_layout)

    def select_file(self):
        file, _ = QFileDialog.getSaveFileName(self, "Sélectionner un fichier TDMS", filter="Fichiers TDMS (*.tdms)")
        if file:
            if not file.lower().endswith(".tdms"):
                file += ".tdms"
            self.file_edit.setText(file)

    def refresh_channels(self):
        system = System.local()
        rows = []

        for device in system.devices:
            print(f"Détection des channels pour le device: {device.name}")
            if not device.ai_physical_chans:
                print(f"Aucun canal analogique trouvé pour le device {device.name}.")
                continue
            else:
                for channel in device.ai_physical_chans:
                    print(f"  Canal détecté: {channel.name} )")
                    # Recherche aiX où X est un chiffre
                    match = re.search(r'ai(\d+)', channel.name)
                    if match:
                        ai_id = f'ai{match.group(1)}'
                        default_name = CHANNEL_NAMES.get(ai_id, channel.name)
                    else:
                        default_name = channel.name
                    rows.append({
                        "device": device.name,
                        "channel": channel.name,
                        "signal_name": default_name,
                        "scaling_coeff": 1.0,
                        "offset": 0.0,
                        "active": True
                    })
        if rows:
            self.channel_df.drop(self.channel_df.index, inplace=True)
            for row in rows:
                self.channel_df.loc[len(self.channel_df)] = row
            print(rows)
            print(self.channel_df)
            self.table_model.layoutChanged.emit()
        else:
            QMessageBox.warning(self, "Detection", "Aucun canal analogique trouvé.")



    def save_config(self):
        try:
            self.general_config["rate"] = int(self.rate_edit.text())
            self.general_config["tdms_file"] = self.file_edit.text()
            QMessageBox.information(self, "Config", "Paramètres généraux enregistrés.\nLa sélection des channels est automatiquement prise en compte.")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Paramètre général invalide : {e}")

class AcquisitionTab(QWidget):
    def __init__(self, channel_df, general_config):
        super().__init__()
        self.channel_df = channel_df
        self.general_config = general_config

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.start_button = QPushButton("Démarrer l'acquisition")
        self.start_button.clicked.connect(self.start_acquisition)
        self.layout.addWidget(self.start_button)

        self.plot_widgets = {}
        self.curves = {}
        self.time_buffer = {}
        self.data_buffer = {}

        self.timer = QTimer()
        self.timer.setInterval(500)  # 500 ms
        self.timer.timeout.connect(self.update_plot)

        self.task = None
        self.daq_thread = None
        self.data_queue = None
        self.stop_event = None

        self.colors = [
            (23, 167, 114),
            (210, 49, 80),
            (46, 134, 222),
            (245, 171, 53),
            (75, 123, 236)
        ]

    def start_acquisition(self):
        if self.task is not None:
            return

        # Nettoyer plots précédents
        for w in self.plot_widgets.values():
            self.layout.removeWidget(w)
            w.deleteLater()
        self.plot_widgets.clear()
        self.curves.clear()
        self.data_buffer.clear()
        self.time_buffer.clear()

        active_channels = self.channel_df[self.channel_df["active"] == True]
        if active_channels.empty:
            QMessageBox.warning(self, "Acquisition", "Aucun channel actif sélectionné.")
            return

        channels = [row["channel"] for _, row in active_channels.iterrows()]
        coeffs = [float(row["scaling_coeff"]) for _, row in active_channels.iterrows()]
        offsets = [float(row["offset"]) for _, row in active_channels.iterrows()]
        names = [str(row["signal_name"]) for _, row in active_channels.iterrows()]

        self.rate = int(self.general_config["rate"])
        self.nb_samples_per_read = int(self.rate // 10)  # 100 ms worth of data
        self.window_length = self.rate  # 1 seconde de points

        tdms_file = self.general_config["tdms_file"]

        self.task = nidaqmx.Task()
        for idx, ch in enumerate(channels):
            scale_coeff = coeffs[idx]
            offset = offsets[idx]
            scale_name = f"scale_{clean_channel_name(names[idx])}"

            # Crée le custom scale (supprime d'abord s'il existe déjà)
            try:
                s = Scale(scale_name)
                s.delete()
            except Exception:
                pass  # Scale n'existait pas

            scale = Scale(scale_name)
            scale.create_lin_scale(
                scale_name,
                slope=scale_coeff,
                y_intercept=offset,
                pre_scaled_units=UnitsPreScaled.VOLTS,
                scaled_units="Volts"
            )

            self.task.ai_channels.add_ai_voltage_chan(
                ch,
                min_val=-10.0 * scale_coeff + offset,
                max_val=10.0 * scale_coeff + offset,
                units=nidaqmx.constants.VoltageUnits.FROM_CUSTOM_SCALE,
                custom_scale_name=scale_name
            )

        #self.task.in_stream.input_buf_size = 20 * self.rate  # buffer DAQmx large !

        self.task.timing.cfg_samp_clk_timing(
            rate=self.rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.nb_samples_per_read
        )
        self.task.in_stream.configure_logging(
            tdms_file,
            LoggingMode.LOG_AND_READ,
            operation=LoggingOperation.CREATE_OR_REPLACE
        )

        for idx, name in enumerate(names):
            plot_widget = pg.PlotWidget(title=f"{name}")
            plot_widget.setBackground('w')
            color = self.colors[idx % len(self.colors)]
            curve = plot_widget.plot(pen=pg.mkPen(color=color, width=2), name=name)
            self.layout.addWidget(plot_widget)
            self.plot_widgets[name] = plot_widget
            self.curves[name] = curve
            self.data_buffer[name] = deque(maxlen=self.window_length)
            self.time_buffer[name] = deque(maxlen=self.window_length)

        self.coeffs = coeffs
        self.offsets = offsets
        self.names = names

        self.data_queue = Queue()
        self.stop_event = Event()
        self.daq_thread = DAQReader(self.task, self.nb_samples_per_read, self.data_queue, self.stop_event)
        self.task.start()
        self.daq_thread.start()
        self.timer.start()
        self.start_button.setEnabled(False)

    def update_plot(self):
        # Lire tous les paquets dispo dans la queue
        updated = False
        try:
            while True:
                data = self.data_queue.get_nowait()
                if isinstance(data, list) and isinstance(data[0], list):
                    for idx, name in enumerate(self.names):
                        samples = (np.array(data[idx]) )
                                   #* self.coeffs[idx] + self.offsets[idx])
                        n = len(samples)
                        if n > 0:
                            if len(self.time_buffer[name]) == 0:
                                tstart = 0
                            else:
                                tstart = self.time_buffer[name][-1] + 1 / self.rate
                            new_times = np.linspace(tstart, tstart + (n - 1) / self.rate, n)
                            self.time_buffer[name].extend(new_times)
                            self.data_buffer[name].extend(samples)
                            self.curves[name].setData(list(self.time_buffer[name]), list(self.data_buffer[name]))
                            updated = True
        except Empty:
            if not updated:
                # Affiche le dernier état, même si pas de nouvelle donnée
                for idx, name in enumerate(self.names):
                    self.curves[name].setData(list(self.time_buffer[name]), list(self.data_buffer[name]))
            pass
        except Exception as e:
            print("Erreur update_plot:", e)
            self.stop_acquisition()

    def stop_acquisition(self):
        self.timer.stop()
        if self.daq_thread is not None and self.daq_thread.is_alive():
            self.stop_event.set()
            self.daq_thread.join(timeout=2)
        if self.task is not None:
            self.task.stop()
            self.task.close()
            self.task = None
        self.start_button.setEnabled(True)

    def closeEvent(self, event):
        self.stop_acquisition()
        event.accept()




class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyDAQmx Acquisition Multi-channels")
        # DataFrame colonnes pour channels
        self.channel_df = pd.DataFrame(
            columns=["device", "channel", "signal_name", "scaling_coeff", "offset", "active"]
        )
        # Configuration générale
        self.general_config = {"rate": 1000, "tdms_file": "donnees.tdms"}

        self.acquisition_tab = AcquisitionTab(self.channel_df, self.general_config)
        self.config_tab = ConfigTab(self.channel_df, self.general_config)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.acquisition_tab, "Acquisition")
        self.tabs.addTab(self.config_tab, "Configuration")
        self.setCentralWidget(self.tabs)

    def closeEvent(self, event):
        self.acquisition_tab.closeEvent(event)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
