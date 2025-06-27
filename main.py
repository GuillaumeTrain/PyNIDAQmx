import os
import sys
import threading

import nidaqmx
from nidaqmx.constants import AcquisitionType, LoggingMode, LoggingOperation, UnitsPreScaled, VoltageUnits
from nidaqmx.system import System
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTabWidget,
    QLineEdit, QLabel, QFormLayout, QMessageBox, QFileDialog, QHBoxLayout,
    QTableView, QHeaderView, QComboBox
)
from PySide6.QtCore import QTimer, Qt, QAbstractTableModel, QModelIndex
import pyqtgraph as pg
import numpy as np
from collections import deque
from threading import Thread, Event
from queue import Queue, Empty
import re
from nidaqmx.scale import Scale
from nidaqmx.system._collections.device_collection import DeviceCollection
from pandas import DataFrame

CHANNEL_NAMES = {
    'ai0': 'Pantograph Current',
    'ai1': 'Catenary Voltage',
    'ai2': 'Speed',
    'ai3': 'Torque'
}


def get_unique_filename(base_filename):
    """
    Retourne un nom de fichier unique en ajoutant un suffixe _000, _001, etc. si nécessaire.
    Si base_filename = 'foo.tdms', retourne 'foo_000.tdms' si 'foo.tdms' existe, etc.
    Si 'foo_005.tdms' existe, retournera 'foo_006.tdms'.
    """
    root, ext = os.path.splitext(base_filename)

    # On veut repérer un suffixe déjà présent type _NNN AVANT l'extension
    suffix_pattern = re.compile(r'_(\d{3,})$')
    match = suffix_pattern.search(root)
    if match:
        root = root[:match.start()]  # Enlève le suffixe

    # Cherche tous les fichiers existants du même type
    directory = os.path.dirname(base_filename) or "."
    base = os.path.basename(root)
    candidates = []
    for f in os.listdir(directory):
        if f.startswith(base) and f.endswith(ext):
            candidates.append(f)
    # Cherche les suffixes existants
    existing_indices = []
    for fname in candidates:
        nmatch = re.search(r'_(\d{3,})' + re.escape(ext) + r'$', fname)
        if nmatch:
            existing_indices.append(int(nmatch.group(1)))
        elif fname == base + ext:
            existing_indices.append(-1)  # Le nom de base sans suffixe

    # S'il n'existe pas, on le prend
    candidate = root + ext
    if -1 not in existing_indices:
        if not os.path.exists(candidate):
            return candidate

    # Sinon, on incrémente
    next_index = 0
    if existing_indices:
        next_index = max(existing_indices) + 1
    while True:
        candidate = f"{root}_{next_index:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        next_index += 1

def clean_channel_name(device, module_id, signal_name):
    """
    Nettoie et formatte le nom du canal en remplaçant les caractères
    non alphanumériques par des underscores et en incluant le device
    et ID du module pour garantir l'unicité des noms de canaux.

    :param device: Le nom de l'appareil.
    :param module_id: L'ID du module. Peut être None.
    :param signal_name: Le nom du signal.
    :return: Nom de canal formaté.
    """
    # Nettoyer le nom du signal
    cleaned_signal_name = re.sub(r'\W+', '_', signal_name)

    # Éliminer l'ID du module s'il est None
    if module_id is not None:
        formatted_name = f"{device}_{module_id}_{cleaned_signal_name}"
    else:
        formatted_name = f"{device}_{cleaned_signal_name}"

    # Remplace les espaces et caractères spéciaux restants
    return re.sub(r'\W+', '_', formatted_name)

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


from threading import Thread

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
                data = self.task.read(
                    number_of_samples_per_channel=self.nb_samples_per_read,
                    timeout=2.0
                )
                # --- Correction NI : Toujours liste de listes ---
                if isinstance(data, list) and data and not isinstance(data[0], (list, np.ndarray)):
                    data = [data]
                self.data_queue.put(data)
            except nidaqmx.errors.DaqError as e:
                print(f"DAQmx ERROR: {e}")
            except Exception as e:
                print(f" Unexpected error: {e}")
                break

    def stop(self):
        """ Arrêt propre du thread """
        if not self.stop_event.is_set():
            self.stop_event.set()
        if self.is_alive():
            self.join(timeout=2)



class ConfigTab(QWidget):
    def __init__(self, channel_df, general_config):
        super().__init__()
        self.channel_df = channel_df
        self.general_config = general_config  # dict-like
        self.stop_event = threading.Event()

        # Déclaration de toutes les valeurs possibles de taux d'échantillonnage
        self.rate_choices = [50_000 // n for n in range(1, 32)]

        main_layout = QVBoxLayout()

        # Ligne de contrôle générale (Sampling Rate + Fichier)
        form_layout = QHBoxLayout()
        # Remplace QLineEdit par QComboBox
        self.rate_combo = QComboBox()
        self.rate_combo.addItems([str(rate) for rate in self.rate_choices])
        self.rate_combo.setCurrentText(str(self.general_config["rate"]))  # Définir la valeur actuelle à 50kHz
        self.rate_combo.currentTextChanged.connect(self.on_rate_change)  # Connecter un changement

        form_layout.addWidget(QLabel("Frequence (Hz):"))
        form_layout.addWidget(self.rate_combo)


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

    def on_rate_change(self, text):
        """Met à jour la fréquence lorsque l'utilisateur change la valeur dans le Combobox."""
        try:
            new_rate = int(text)
            self.general_config["rate"] = new_rate
        except ValueError:
            pass

    def select_file(self):
        file, _ = QFileDialog.getSaveFileName(self, "Sélectionner un fichier TDMS", filter="Fichiers TDMS (*.tdms)")
        if file:
            if not file.lower().endswith(".tdms"):
                file += ".tdms"
            self.file_edit.setText(file)

    def refresh_channels(self):
        self.channel_df : DataFrame
        system = System.local()
        rows = []
        master_device=""
        for device in system.devices:
            print(f"Détection des channels pour le device: {device.name}")
            print(f"{master_device}")
            if not device.ai_physical_chans:
                print(f"Aucun canal analogique trouvé pour le device {device.name}.")
                master_device=device.product_type
                continue
            for channel in device.ai_physical_chans:
                print(f"  Canal détecté: {channel.name} )")
                device_name = device.name.split('Mod')[0] if 'Mod' in device.name else device.name
                # Extraire le module depuis le nom de l'appareil
                match = re.search(r'Mod(\d+)', channel.name)
                module_id = "Mod"+match.group(1) if match else None
                match = re.search(r'ai(\d+)', channel.name)
                ai_id=""
                if match:
                    print("ai trouvé")
                    ai_id = f'ai{match.group(1)}'

                    default_name = CHANNEL_NAMES.get(ai_id, channel.name)

                else:
                    default_name = channel.name

                    # Vérifier la duplication de noms de signaux
                    # Créer un nom de signal unique
                unique_name = default_name
                counter = 1
                while any(row['signal_name'] == unique_name for row in rows):
                    unique_name = f"{default_name}_{counter}"
                    counter += 1
                print(f"default_name set to {default_name} ")

                rows.append({
                        "device": device_name,
                        "module_id": module_id,  # Rempli à partir de l'appareil/module
                        "channel": ai_id,
                        "signal_name":  unique_name,
                        "scaling_coeff": 1.0,
                        "offset": 0.0,
                        "active": True,
                        "M/N":master_device
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
            self.general_config["rate"] = int(self.rate_combo.currentText())
            self.general_config["tdms_file"] = self.file_edit.text()
            QMessageBox.information(self, "Config", "Paramètres généraux enregistrés.\nLa sélection des channels est automatiquement prise en compte.")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Paramètre général invalide : {e}")


class AcquisitionTab(QWidget):
    def __init__(self, channel_df, general_config , config_tab):
        super().__init__()
        self.last_global_time = 0.0
        self._master_plotwidget = None
        self.stop_event = config_tab.stop_event
        self.daq_threads = {}
        self.channel_df =channel_df
        self.general_config = general_config

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.filename_label = QLabel("en attente de l'acquisition...")
        self.layout.addWidget(self.filename_label)
        self.time_div_choices = [
            2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6,
            1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3
        ]
        self.time_div_combo = QComboBox()
        for tdiv in self.time_div_choices:
            label = f"{tdiv * 1e3:.3g} ms/div"
            self.time_div_combo.addItem(label, tdiv)
        self.time_div_combo.setCurrentIndex(10)  # 10 ms/div par défaut
        self.time_div_combo.currentIndexChanged.connect(self.force_update_plot)
        self.layout.addWidget(self.time_div_combo)

        # Ajouter le combo au layout avant les plots
        self.layout.addWidget(self.time_div_combo)
        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton("Démarrer l'acquisition")
        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button = QPushButton("Stoper l'acquisition")
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)
        self.layout.addLayout(self.button_layout)

        self.plot_widgets = {}
        self.curves = {}
        self.used_tdms_files = {}


        self.timer = QTimer()
        self.timer.setInterval(200)  # 500 ms
        self.timer.timeout.connect(self.update_plot)

        self.tasks = {}
        self.time_buffers = {}
        self.data_buffers = {}
        self.data_queues = {}
        self.daq_thread = None
        self.stop_event = None

        self.colors = [
            (23, 167, 114),
            (210, 49, 80),
            (46, 134, 222),
            (245, 171, 53),
            (75, 123, 236)
        ]

    def stop_acquisition(self):
        self.timer.stop()
        if hasattr(self, 'stop_event') and self.stop_event:
            self.stop_event.set()
            self.last_global_time = 0.0

        # Corrige l'appel à clear même si les attributs n'existent pas
        for attr in [
            "curves", "data_buffers", "time_buffers", "coeffs", "offsets", "device_channel_names"
        ]:
            if hasattr(self, attr):
                getattr(self, attr).clear()

        if hasattr(self, "data_queues"):
            self.data_queues.clear()
        if hasattr(self, "filename_label"):
            self.filename_label.setText("Aucun fichier TDMS sélectionné.")
        # ... le reste ne change pas ...
        # Arrêt des tâches NI
        for device, task in getattr(self, "tasks", {}).items():
            if task is not None:
                try:
                    task.stop()
                except nidaqmx.errors.DaqError as e:
                    print(f"[{device}] DAQmx ERROR lors de l'arrêt de la tâche: {e}")
                task.close()
                self.tasks[device] = None
        if hasattr(self, "tasks"):
            self.tasks.clear()

        # Arrêt threads d'acquisition
        for device, thread in getattr(self, "daq_threads", {}).items():
            if thread.is_alive():
                thread.stop_event.set()
                thread.join(timeout=5)
        if hasattr(self, "daq_threads"):
            self.daq_threads.clear()

        # Suppression correcte des widgets PyQtGraph
        for device in list(getattr(self, "plot_widgets", {}).keys()):
            for signal_name in list(self.plot_widgets[device].keys()):
                widget = self.plot_widgets[device][signal_name]
                self.layout.removeWidget(widget)
                widget.deleteLater()
            self.plot_widgets[device].clear()
        if hasattr(self, "plot_widgets"):
            self.plot_widgets.clear()
        self._master_plotwidget = None

        # Réactiver le bouton de démarrage après l'arrêt
        if hasattr(self, "start_button"):
            self.start_button.setEnabled(True)

        print("L'acquisition est arrêtée et tout est réinitialisé.")

    # Avis pour le débogage

    def start_acquisition(self):
        self.stop_acquisition()
        self.last_global_time = 0.0

        # Nettoyage complet
        for wdict in self.plot_widgets.values():
            for widget in wdict.values():
                self.layout.removeWidget(widget)
                widget.deleteLater()
        self.plot_widgets.clear()
        self.curves.clear()
        self.data_buffers.clear()
        self.time_buffers.clear()
        self.tasks.clear()
        self.data_queues.clear()
        self.daq_threads.clear()
        self.used_tdms_files.clear()

        # Ne conserve QUE les voies actives
        active_channels = self.channel_df[self.channel_df["active"] == True]
        if active_channels.empty:
            QMessageBox.warning(self, "Acquisition", "Aucun channel actif sélectionné.")
            return

        self.rate = int(self.general_config["rate"])
        self.nb_samples_per_read = int(self.rate)  # 1s worth (ajuste si besoin)
        self.window_length = self.nb_samples_per_read * 2

        self.device_channel_names = {}  # device -> [signal_name, ...] (ORDRE D'ACQUISITION)
        self.coeffs = {}
        self.offsets = {}
        self.data_buffers = {}
        self.time_buffers = {}
        self.curves = {}
        self.plot_widgets = {}
        self.data_queues = {}
        self.tasks = {}
        self.daq_threads = {}
        self.used_tdms_files = {}

        device_groups = active_channels.groupby("device")
        if len(device_groups) > 1:
            QMessageBox.warning(self, "Attention",
                                "Attention ! La synchronisation à l'échantillon près n'est pas supportée entre deux racks")

        color_idx = 0

        for device, group in device_groups:
            # Important : n'ajouter que les voies actives, et dans le même ordre partout
            signal_names = []
            self.coeffs[device] = {}
            self.offsets[device] = {}
            self.data_buffers[device] = {}
            self.time_buffers[device] = {}
            self.curves[device] = {}
            self.plot_widgets[device] = {}
            self.data_queues[device] = Queue()

            task = nidaqmx.Task()
            for idx, row in group.iterrows():
                # Ici, comme group ne contient déjà que les actifs, pas besoin de check "if not row['active']: continue"
                device_name = row['device']
                module_id_name = row['module_id']
                channel_name = row["channel"]
                scale_coeff = float(row["scaling_coeff"])
                offset = float(row["offset"])
                signal_name = str(row["signal_name"])
                signal_names.append(signal_name)

                self.data_buffers[device][signal_name] = deque(maxlen=self.window_length)
                self.time_buffers[device][signal_name] = deque(maxlen=self.window_length)
                self.coeffs[device][signal_name] = scale_coeff
                self.offsets[device][signal_name] = offset

                # Plot/courbe pour ce channel
                plot_widget = pg.PlotWidget(title=f"{signal_name}")
                plot_widget.setBackground('w')
                color = self.colors[color_idx % len(self.colors)]
                curve = plot_widget.plot(pen=pg.mkPen(color=color, width=2), name=signal_name)
                self.layout.addWidget(plot_widget)
                self.plot_widgets[device][signal_name] = plot_widget
                self.curves[device][signal_name] = curve
                color_idx += 1

                # Synchronise les axes X
                if self._master_plotwidget is None:
                    self._master_plotwidget = plot_widget
                else:
                    plot_widget.setXLink(self._master_plotwidget)

                # Custom scale NI
                signal_full_id = clean_channel_name(device_name, module_id_name, channel_name)
                scale_name = f"scale_{signal_full_id}"
                try:
                    s = Scale(scale_name)
                    s.delete()
                except Exception:
                    pass
                scale = Scale(scale_name)
                scale.create_lin_scale(
                    scale_name,
                    slope=scale_coeff,
                    y_intercept=offset,
                    pre_scaled_units=UnitsPreScaled.VOLTS,
                    scaled_units="Volts"
                )

                # Ajout du channel à la Task
                task.ai_channels.add_ai_voltage_chan(
                    f"{device}{module_id_name}/{channel_name}",
                    min_val=-10.0 * scale_coeff + offset,
                    max_val=10.0 * scale_coeff + offset,
                    units=nidaqmx.constants.VoltageUnits.FROM_CUSTOM_SCALE,
                    custom_scale_name=scale_name,
                    name_to_assign_to_channel=signal_full_id,
                )

            self.device_channel_names[device] = signal_names

            # Horloge d’échantillonnage
            task.timing.cfg_samp_clk_timing(
                rate=self.rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.nb_samples_per_read
            )
            base_tdms_file = self.general_config["tdms_file"].replace('.tdms', f'_{device}.tdms')
            tdms_file = get_unique_filename(base_tdms_file)
            self.used_tdms_files[device] = tdms_file

            task.in_stream.configure_logging(
                tdms_file,
                LoggingMode.LOG_AND_READ,
                operation=LoggingOperation.CREATE_OR_REPLACE
            )
            self.tasks[device] = task

        self.stop_event = Event()

        # Affiche les fichiers utilisés
        if self.used_tdms_files:
            text = "Fichiers TDMS utilisés :<br>" + "<br>".join(
                f"{dev} : {fname}" for dev, fname in self.used_tdms_files.items()
            )
            self.filename_label.setText(text)
        else:
            self.filename_label.setText("Aucun fichier TDMS sélectionné.")

        # Démarre les threads et tasks NI
        for device, task in self.tasks.items():
            self.daq_threads[device] = DAQReader(
                task, self.nb_samples_per_read, self.data_queues[device], self.stop_event
            )
            try:
                task.start()
            except nidaqmx.errors.DaqError as e:
                print(f"Erreur de configuration du déclencheur pour {device}: {e}")
            self.daq_threads[device].start()

        self.timer.start()
        self.start_button.setEnabled(False)

    def update_plot(self, force=False):
        try:
            data_by_device = {}
            nsamples_list = []
            # Synchro verticale : on attend une trame pour chaque device
            for device, queue in self.data_queues.items():
                try:
                    data = queue.get_nowait()
                    data_by_device[device] = data
                    nsamples_list.append(len(data[0]) if data else 0)
                except Empty:
                    if not force:  # On ne rafraîchit que si on a tout le monde, sauf si forcé par combo
                        return

            # Quand on force le redraw (via combo), il faut savoir sur quoi centrer. On prend la fin des buffers.
            MAX_POINTS_ON_SCREEN = 2000
            n = min(nsamples_list) if nsamples_list else 0
            dt = 1.0 / self.rate
            if not hasattr(self, 'last_global_time'):
                self.last_global_time = 0.0
            tstart = self.last_global_time
            if n > 0:
                new_times = np.linspace(tstart, tstart + (n - 1) * dt, n)
                self.last_global_time = tstart + n * dt
            else:
                new_times = None

            # Time/div
            tdiv = self.time_div_combo.currentData()
            ndiv = 5
            window = tdiv * ndiv

            # Pour chaque device/plot actif
            for device, data in data_by_device.items():
                active_channel_mask = (self.channel_df["active"] == True) & (self.channel_df["device"] == device)
                active_signals = self.channel_df[active_channel_mask]["signal_name"].tolist()
                for idx, signal_name in enumerate(active_signals):
                    if n > 0 and idx < len(data):
                        samples = np.array(data[idx][:n])

                        tb = self.time_buffers[device][signal_name]
                        tb.extend(new_times)
                        self.data_buffers[device][signal_name].extend(samples)
                    # Toujours afficher le contenu du buffer même si aucune data nouvelle
                    xdata = np.array(self.time_buffers[device][signal_name])
                    ydata = np.array(self.data_buffers[device][signal_name])
                    if len(xdata) == 0:
                        continue
                    curve = self.curves[device][signal_name]
                    # On centre la vue sur la fin du signal (oscillo style)
                    xmax = xdata[-1]
                    xmin = xmax - window
                    # Filtrer les points à l'affichage
                    mask = (xdata >= xmin) & (xdata <= xmax)
                    x_visible = xdata[mask]
                    y_visible = ydata[mask]
                    if len(x_visible) > MAX_POINTS_ON_SCREEN:
                        idxs = np.linspace(0, len(x_visible) - 1, MAX_POINTS_ON_SCREEN).astype(int)
                        x_visible = x_visible[idxs]
                        y_visible = y_visible[idxs]
                    curve.setData(x_visible, y_visible)
                    curve.getViewBox().setXRange(xmin, xmax, padding=0)

            # Affiche les fichiers TDMS utilisés
            self.filename_label.setText(
                "Fichiers TDMS utilisés:\n" +
                "\n".join(f"{dev}: {fname}" for dev, fname in self.used_tdms_files.items())
            )
        except Exception as e:
            print("Erreur update_plot:", e)

    def closeEvent(self, event):
        # Demande une confirmation à l'utilisateur avant de fermer
        reply = QMessageBox.question(self, 'Confirmation',
                                     "Êtes-vous sûr de vouloir quitter l'application?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                self.stop_acquisition()  # Arrêtez l'acquisition proprement
                print("Acquisition arrêtée. Fermeture de l'application.")
            except Exception as e:
                print(f"Erreur lors de l'arrêt de l'acquisition: {e}")
            event.accept()  # Accepter l'événement pour fermer l'application
        else:
            event.ignore()  # Ignorer l'événement pour garder l'application ouverte

    def force_update_plot(self):
        self.update_plot(force=True)

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyDAQmx Acquisition Multi-channels")
        # DataFrame colonnes pour channels
        self.channel_df = pd.DataFrame(
            columns=["device", "module_id", "channel", "signal_name", "scaling_coeff", "offset", "active","M/N"]
        )
        # Configuration générale
        self.general_config = {"rate": 50000, "tdms_file": "donnees.tdms"}


        self.config_tab = ConfigTab(self.channel_df, self.general_config)
        self.acquisition_tab = AcquisitionTab(self.channel_df, self.general_config, self.config_tab)
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