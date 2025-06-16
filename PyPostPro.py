import h5py
import numpy as np
import os
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QLineEdit, QHeaderView, QAbstractItemView, QFileDialog, QMessageBox, QTableView,
    QStyledItemDelegate, QSpinBox, QSlider
)
from PySide6.QtCore import Qt, QAbstractTableModel
from nptdms import TdmsFile
from scipy.signal import get_window, ShortTimeFFT
import pyqtgraph as pg

# ----- HDF5 FFT utils -----
def save_fft_stream_hdf5(filename, fft_data, freqs, times, channel_names, fft_params):
    with h5py.File(filename, "w") as f:
        f.create_dataset("Sx", data=fft_data, compression="gzip")
        f.create_dataset("frequencies", data=freqs)
        f.create_dataset("times", data=times)
        f.create_dataset("channel_names", data=hdf5_compatible(channel_names))
        grp = f.create_group("params")
        for k, v in fft_params.items():
            grp.attrs[k] = hdf5_compatible(v)


def hdf5_compatible(val):
    """
    Prépare val pour écriture HDF5 : convertit les types NumPy non supportés,
    en particulier les np.datetime64, en string (sans toucher aux données originales).
    """
    import numpy as np

    if isinstance(val, np.datetime64):
        return str(val)
    elif isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.datetime64):
        return val.astype(str)
    elif isinstance(val, (list, tuple)):
        # Traite récursivement les éléments
        return type(val)(hdf5_compatible(v) for v in val)
    elif isinstance(val, dict):
        return {k: hdf5_compatible(v) for k, v in val.items()}
    else:
        return val


# ----- Table & Delegates -----
class ChannelTableModel(QAbstractTableModel):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid(): return None
        value = self._df.iloc[index.row(), index.column()]
        if role in (Qt.DisplayRole, Qt.EditRole): return "" if pd.isna(value) else str(value)
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole: return False
        col, row = self._df.columns[index.column()], index.row()
        if col == "Nouveau nom":
            self._df.iloc[row, index.column()] = value
        elif col == "Unité":
            if value in ["Volts", "Ampère", "Vitesse", "Couple"]:
                self._df.iloc[row, index.column()] = value
            else:
                return False
        elif col == "Post-traitement":
            if value in ["Sommation", "FFT", "Monitoring", "Psophométrie", "Désactivé"]:
                self._df.iloc[row, index.column()] = value
            else:
                return False
        elif col in ["Coeff", "Offset"]:
            try:
                self._df.iloc[row, index.column()] = float(value)
            except ValueError:
                return False
        else:
            return False
        self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        if not index.isValid(): return Qt.NoItemFlags
        col = self._df.columns[index.column()]
        if col == "Signal d'origine":
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        elif col in ["Unité", "Post-traitement", "Nouveau nom", "Coeff", "Offset"]:
            return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return Qt.NoItemFlags

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(self._df.index[section])
        return None


class ComboBoxDelegate(QStyledItemDelegate):
    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.items = items

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.items)
        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.DisplayRole)
        if value in self.items: editor.setCurrentText(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)


# ----- MAIN Config Tab -----
class ConfigTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_path = None
        self.df = pd.DataFrame(
            columns=["Signal d'origine", "Nouveau nom", "Unité", "Coeff", "Offset", "Post-traitement"])
        self.layout = QVBoxLayout(self)
        # Fichier
        file_import_layout = QHBoxLayout()
        self.file_label = QLabel("Fichier TDMS:")
        self.file_path_line_edit = QLineEdit(self)
        self.file_path_line_edit.setReadOnly(True)
        self.file_dialog_button = QPushButton("Charger un fichier")
        file_import_layout.addWidget(self.file_label)
        file_import_layout.addWidget(self.file_path_line_edit)
        file_import_layout.addWidget(self.file_dialog_button)
        self.layout.addLayout(file_import_layout)
        self.file_dialog_button.clicked.connect(self.open_file)
        # Table
        self.table_model = ChannelTableModel(self.df)
        self.channels_table = QTableView()
        self.channels_table.setModel(self.table_model)
        self.channels_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.channels_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.channels_table)
        self.channels_table.setItemDelegateForColumn(2, ComboBoxDelegate(["Volts", "Ampère", "Vitesse", "Couple"],
                                                                         self.channels_table))
        self.channels_table.setItemDelegateForColumn(5, ComboBoxDelegate(
            ["Sommation", "FFT", "Monitoring", "Psophométrie", "Désactivé"], self.channels_table))
        # Sommation
        summ_layout = QHBoxLayout()
        self.output_channel_name_edit = QLineEdit(self)
        self.output_channel_name_edit.setPlaceholderText("Nom signal sommation (ex: SignalSomme)")
        summ_layout.addWidget(QLabel("Nom signal sommation :"))
        summ_layout.addWidget(self.output_channel_name_edit)
        self.layout.addLayout(summ_layout)
        # FFT params
        param_fft_layout = QHBoxLayout()
        self.fft_win_size_edit = QLineEdit()
        self.fft_win_size_edit.setPlaceholderText("Fenêtre (s)")
        self.fft_win_size_edit.setText("0.5")
        self.fft_window_type_combo = QComboBox()
        self.fft_window_type_combo.addItems(["hann", "gaussian", "rect", "blackman", "hamming"])
        self.fft_overlap_edit = QLineEdit()
        self.fft_overlap_edit.setPlaceholderText("Recouvrement (%)")
        self.fft_overlap_edit.setText("50")
        self.fft_average_spin = QSpinBox()
        self.fft_average_spin.setMinimum(1)
        self.fft_average_spin.setMaximum(100)
        self.fft_average_spin.setValue(6)
        self.fft_average_spin.setFixedWidth(60)
        param_fft_layout.addWidget(QLabel("Fenêtre :"))
        param_fft_layout.addWidget(self.fft_window_type_combo)
        param_fft_layout.addWidget(QLabel("Taille (s)"))
        param_fft_layout.addWidget(self.fft_win_size_edit)
        param_fft_layout.addWidget(QLabel("Recouvrement (%)"))
        param_fft_layout.addWidget(self.fft_overlap_edit)
        param_fft_layout.addWidget(QLabel("Nb fen. à moyenner"))
        param_fft_layout.addWidget(self.fft_average_spin)
        self.layout.addLayout(param_fft_layout)
        # ---- PostProcessing ----
        self.postproc_button = QPushButton("Lancer le post-processing", self)
        self.postproc_button.clicked.connect(self.postprocessing_all)
        self.layout.addWidget(self.postproc_button)
        self.config = {}

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner un fichier TDMS", "", "Fichiers TDMS (*.tdms)")
        if file_path:
            self.file_path = file_path
            self.file_path_line_edit.setText(file_path)
            try:
                with TdmsFile.open(self.file_path) as tdms_file:
                    all_groups = tdms_file.groups()
                    data = []
                    for group in all_groups:
                        for channel in group.channels():
                            data.append({
                                "Signal d'origine": channel.name,
                                "Nouveau nom": channel.name,
                                "Unité": "Volts",
                                "Coeff": 1.0,
                                "Offset": 0.0,
                                "Post-traitement": "Monitoring"
                            })
                    self.df = pd.DataFrame(data)
                    self.table_model._df = self.df
                    self.table_model.layoutChanged.emit()
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier TDMS:\n{e}")

    def get_fft_params(self, ref_fs=None):
        try:
            window_type = self.fft_window_type_combo.currentText()
            win_size_seconds = float(self.fft_win_size_edit.text())
            overlap_percent = float(self.fft_overlap_edit.text()) / 100.0
            n_average = int(self.fft_average_spin.value())
            # Fréquence d’échantillonnage (à partir du 1er signal FFT)
            fft_rows = self.df[self.df["Post-traitement"] == "FFT"]
            tdms_file_path = self.file_path
            if ref_fs is None:
                if fft_rows.empty: raise ValueError("Aucun canal FFT pour déduire fs")
                with TdmsFile.open(tdms_file_path) as tdms_file:
                    first_ch = fft_rows.iloc[0]["Signal d'origine"]
                    fs = None
                    for group in tdms_file.groups():
                        for channel in group.channels():
                            if channel.name == first_ch:
                                fs = 1.0 / channel.properties.get("wf_increment", 1.0)
                                break
                        if fs is not None: break
                if fs is None: raise ValueError("Impossible de récupérer la fréquence d'échantillonnage")
            else:
                fs = ref_fs
            win_size_samples = int(win_size_seconds * fs)
            hop = max(int(win_size_samples * (1 - overlap_percent)), 1)
            params = {
                "window_type": window_type, "win_size_seconds": win_size_seconds,
                "win_size_samples": win_size_samples, "overlap_percent": overlap_percent,
                "hop": hop, "n_average": n_average, "fs": fs
            }
            self.config["fft_params"] = params
            return params
        except Exception as e:
            QMessageBox.critical(self, "Erreur paramètres FFT", f"{e}")
            return None

    def postprocessing_all(self):
        try:
            if not self.file_path:
                QMessageBox.critical(self, "Erreur", "Fichier TDMS non chargé.")
                return

            df = self.df
            output_dir = os.path.join(os.getcwd(), "temp_files")
            os.makedirs(output_dir, exist_ok=True)

            tdms_file_path = self.file_path
            preprocessed_path = os.path.join(output_dir, "preprocessed.h5")

            # 1. Prétraitement HDF5 (renommage, scaling, offset)
            with TdmsFile.open(tdms_file_path) as tdms_file, h5py.File(preprocessed_path, "w") as h5file:
                channel_names = []
                meta_grp = h5file.create_group("metadata")
                for i, row in df.iterrows():
                    if row["Post-traitement"] == "Désactivé":
                        continue
                    ch_name = row["Signal d'origine"]
                    new_name = row["Nouveau nom"]
                    coeff, offset = float(row["Coeff"]), float(row["Offset"])
                    for group in tdms_file.groups():
                        for channel in group.channels():
                            if channel.name == ch_name:
                                sig = channel[:] * coeff + offset
                                h5file.create_dataset(new_name, data=sig, compression="gzip")
                                channel_names.append(new_name)
                                meta = {k: v for k, v in channel.properties.items()}
                                meta_grp.create_group(new_name)
                                for k, v in meta.items():
                                    meta_grp[new_name].attrs[k] = hdf5_compatible(v)
                h5file.create_dataset("channel_names", data=np.array(channel_names, dtype="S"))

            # 2. Sommation
            output_channel_name = None
            somm_rows = df[df["Post-traitement"] == "Sommation"]
            if not somm_rows.empty:
                with h5py.File(preprocessed_path, "a") as h5file, TdmsFile.open(tdms_file_path) as tdms_file:
                    channel_objects = {}
                    for group in tdms_file.groups():
                        for channel in group.channels():
                            channel_objects[channel.name] = channel
                    output_channel_name = self.output_channel_name_edit.text().strip() or "Somme"
                    n_samples = len(channel_objects[somm_rows.iloc[0]["Signal d'origine"]])
                    summed = np.zeros(n_samples)
                    for _, row in somm_rows.iterrows():
                        ch = row["Signal d'origine"]
                        coeff, offset = float(row["Coeff"]), float(row["Offset"])
                        summed += channel_objects[ch][:] * coeff + offset
                    h5file.create_dataset(output_channel_name, data=summed, compression="gzip")
                    # Met à jour la liste des channel_names et metadata
                    names = h5file["channel_names"][:].tolist() + [output_channel_name.encode()]
                    del h5file["channel_names"]
                    h5file.create_dataset("channel_names", data=names)
                    meta_grp = h5file["metadata"]
                    first_ch = somm_rows.iloc[0]["Signal d'origine"]
                    meta_grp.create_group(output_channel_name)
                    for k, v in channel_objects[first_ch].properties.items():
                        meta_grp[output_channel_name].attrs[k] = hdf5_compatible(v)

            # 3. FFT HDF5 du signal sommé ET des channels FFT
            fft_rows = df[df["Post-traitement"] == "FFT"]
            fft_channels = []
            fft_signals = []
            with h5py.File(preprocessed_path, "r") as h5file:
                # Ajoute le signal sommé s'il existe
                if output_channel_name and output_channel_name in h5file:
                    fft_channels.append(output_channel_name)
                    fft_signals.append(h5file[output_channel_name][:])
                for _, row in fft_rows.iterrows():
                    new_name = row["Nouveau nom"]
                    if new_name in h5file:
                        fft_channels.append(new_name)
                        fft_signals.append(h5file[new_name][:])

            if not fft_channels:
                QMessageBox.information(self, "Post-processing", "Aucun signal FFT à traiter.")
                return

            # Récupère les paramètres FFT
            fft_params = self.get_fft_params()
            if fft_params is None:
                return

            window_type = fft_params["window_type"]
            win_size_samples = fft_params["win_size_samples"]
            hop = fft_params["hop"]
            n_average = fft_params["n_average"]
            fs = fft_params["fs"]
            window = get_window(window_type, win_size_samples)
            all_sx, freqs, times = [], None, None

            # Pour chaque signal, calcule la magnitude RMS de la FFT (spectrogramme)
            for sig in fft_signals:
                SFT = ShortTimeFFT(window, hop, fs)
                Sx = SFT.stft(sig)
                mag = np.abs(Sx) / np.sqrt(win_size_samples)  # RMS par bin
                if n_average > 1:
                    from scipy.ndimage import uniform_filter1d
                    mag = uniform_filter1d(mag, size=n_average, axis=1, mode='nearest')
                all_sx.append(mag)
                if freqs is None:
                    freqs = SFT.f
                    times = SFT.t(len(sig))

            if not all_sx or freqs is None or times is None:
                QMessageBox.warning(self, "Post-processing",
                                    "Aucun résultat FFT n'a pu être calculé (pas de signal disponible ?).")
                return

            all_sx = np.stack(all_sx, axis=0)  # (n_channels, n_freqs, n_times)
            fft_h5_path = os.path.join(output_dir, "FFT_streams.h5")
            save_fft_stream_hdf5(fft_h5_path, all_sx, freqs, times, fft_channels, fft_params)

            QMessageBox.information(self, "Succès",
                                    f"Post-processing terminé.\nPréprocessing : {preprocessed_path}\nFFT : {fft_h5_path}")

        except Exception as e:
            QMessageBox.critical(self, "Erreur post-processing", f"{e}")


class TabVisualisation(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.h5file = None  # Instance h5py.File
        self.signals = {}   # nom: np.ndarray (attention à la taille !)
        self.channel_names = []
        self.sampling_rate = None
        self.times = None

        self.layout = QVBoxLayout(self)

        # Sélection du fichier HDF5 à charger
        file_layout = QHBoxLayout()
        self.file_label = QLabel("Fichier HDF5:")
        self.file_combo = QComboBox()  # ou un bouton pour ouvrir le fichier
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_combo)
        self.layout.addLayout(file_layout)

        # Zone graphique
        self.plot_widget = pg.PlotWidget(title="Monitor Signals")
        self.layout.addWidget(self.plot_widget)

        # Sélection de la div/zoom
        self.div_combo = QComboBox()
        for val, txt in [
            (1e-6, "1 µs"), (1e-5, "10 µs"), (1e-4, "100 µs"),
            (1e-3, "1 ms"), (1e-2, "10 ms"), (1e-1, "100 ms"),
            (1, "1 s")
        ]:
            self.div_combo.addItem(txt, val)
        self.div_combo.setCurrentIndex(4)  # Par défaut 1 ms/div
        self.layout.addWidget(self.div_combo)

        # Slider de navigation temporelle
        self.time_slider = QSlider(Qt.Horizontal)
        self.layout.addWidget(self.time_slider)

        # Zone curseurs de sélection période
        self.start_slider = QSlider(Qt.Horizontal)
        self.end_slider = QSlider(Qt.Horizontal)
        self.layout.addWidget(QLabel("Début période"))
        self.layout.addWidget(self.start_slider)
        self.layout.addWidget(QLabel("Fin période"))
        self.layout.addWidget(self.end_slider)

        # Connexions
        self.div_combo.currentIndexChanged.connect(self.on_div_changed)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        self.start_slider.valueChanged.connect(self.on_selection_changed)
        self.end_slider.valueChanged.connect(self.on_selection_changed)
        # TODO: Ajoute un bouton "Charger" et/ou fichier courant

        # Variables internes
        self.current_start = 0
        self.current_div = 1e-3  # 1 ms par div
        self.max_display_points = 5000

    def load_h5(self, filename):
        self.h5file = h5py.File(filename, "r")
        self.channel_names = [n.decode() for n in self.h5file["channel_names"][:]]
        # On charge uniquement la taille pour l'instant (pas les signaux entiers !)
        first = self.h5file[self.channel_names[0]]
        self.signal_length = len(first)
        self.sampling_rate = 1.0 / float(self.h5file["metadata"][self.channel_names[0]].attrs["wf_increment"])
        self.times = np.arange(self.signal_length) / self.sampling_rate
        self.time_slider.setMaximum(self.signal_length-1)
        self.start_slider.setMaximum(self.signal_length-1)
        self.end_slider.setMaximum(self.signal_length-1)
        self.end_slider.setValue(self.signal_length-1)
        self.update_plot()

    def on_div_changed(self, i):
        self.current_div = float(self.div_combo.currentData())
        self.update_plot()

    def on_time_slider_changed(self, val):
        self.current_start = val
        self.update_plot()

    def on_selection_changed(self):
        # À compléter selon les besoins (callback curseurs début/fin)
        self.update_plot()

    def update_plot(self):
        if not self.h5file: return
        self.plot_widget.clear()
        chunk_length = int(self.current_div * self.sampling_rate * 10)  # chunk = div * 10
        start = max(0, self.current_start - chunk_length//2)
        end = min(self.signal_length, start + chunk_length)
        if end > self.signal_length: start = self.signal_length - chunk_length
        if start < 0: start = 0

        for name in self.channel_names:
            data = self.h5file[name][start:end]
            # Downsampling si trop de points à afficher
            N = len(data)
            if N > self.max_display_points:
                # Downsampling dynamique (min-max preserve dynamics)
                n = self.max_display_points // 2
                stride = N // n
                min_points = data[:n*stride].reshape(n, stride).min(axis=1)
                max_points = data[:n*stride].reshape(n, stride).max(axis=1)
                plot_data = np.empty(n*2)
                plot_data[0::2] = min_points
                plot_data[1::2] = max_points
                time_axis = np.linspace(self.times[start], self.times[end-1], n*2)
            else:
                plot_data = data
                time_axis = self.times[start:end]
            self.plot_widget.plot(time_axis, plot_data, pen=None, symbol=None, name=name)

        # TODO : curseurs visuels sur le plot avec pg.LinearRegionItem et/ou InfiniteLine


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyNIDAQmx V1.0")
        self.tabs = QTabWidget()
        self.tab_config = ConfigTab()
        self.tab_visu = TabVisualisation()
        self.tabs.addTab(self.tab_config, "Configuration")
        self.tabs.addTab(self.tab_visu, "Visualisation")
        self.setCentralWidget(self.tabs)


def main():
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
