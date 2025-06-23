from numba import njit, prange
from scipy.signal import ShortTimeFFT
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QLineEdit, QHeaderView, QAbstractItemView, QFileDialog, QMessageBox, QTableView,
    QStyledItemDelegate, QSpinBox, QSlider, QCheckBox, QDockWidget, QGroupBox
)
from PySide6.QtCore import Qt, QAbstractTableModel
from nptdms import TdmsFile
import pyqtgraph as pg

from Data_Module import *
import scipy as sp
from scipy.signal import ShortTimeFFT, get_window
import time
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def detect_exceedings_numba_parallel(
    fft_matrix, timestamps, all_band_indices, all_band_limits, all_band_freqs, band_sizes
):
    n_windows, n_freqs = fft_matrix.shape
    n_bands, n_band_freqs = all_band_indices.shape
    # Taille max par fenêtre (en général, on ne dépasse jamais n_bands * n_band_freqs hits par fenêtre)
    max_hits_per_window = n_bands * n_band_freqs

    # Pré-allocate un buffer de résultats par fenêtre, taille [n_windows, max_hits_per_window, 4]
    temp_results = np.full((n_windows, max_hits_per_window, 4), -1.0)
    temp_counts = np.zeros(n_windows, dtype=np.int32)

    for i in prange(n_windows):
        hit_count = 0
        for bandidx in range(n_bands):
            size = band_sizes[bandidx]
            for f in range(size):
                idx = all_band_indices[bandidx, f]
                if idx < 0:
                    continue
                value = fft_matrix[i, int(idx)]
                limit = all_band_limits[bandidx, f]
                if value > limit:
                    # Stocke dans la "case locale" de ce thread (fenêtre)
                    temp_results[i, hit_count, 0] = timestamps[i]
                    temp_results[i, hit_count, 1] = bandidx
                    temp_results[i, hit_count, 2] = all_band_freqs[bandidx, f]
                    temp_results[i, hit_count, 3] = bandidx
                    hit_count += 1
        temp_counts[i] = hit_count

    # Maintenant, on "concatène" tous les résultats
    total_hits = np.sum(temp_counts)
    results = np.full((total_hits, 4), -1.0)
    pos = 0
    for i in range(n_windows):
        count = temp_counts[i]
        if count > 0:
            results[pos:pos+count] = temp_results[i, :count]
            pos += count

    return results

def pad(arr, fill, max_len=None):
    arr = np.asarray(arr)
    if max_len is None:
        max_len = len(arr)
    pad_width = max_len - len(arr)
    if pad_width > 0:
        return np.concatenate([arr, np.full(pad_width, fill, dtype=arr.dtype)])
    return arr

def prepare_limits_for_numba(limit_manager, filtered_frequencies):
    all_band_indices = []
    all_band_limits = []
    all_band_freqs = []
    band_sizes = []
    limit_name_list = []
    limit_band_idx_list = []  # <-- NOUVEAU
    for limit_name, limit_data in limit_manager.limits_by_name.items():
        if "interpolated_bands" not in limit_data:
            continue
        for band_idx, band in enumerate(limit_data["interpolated_bands"]):
            band_freqs = np.array(band["frequencies"])
            band_limits = np.array(band["limits"])
            indices = map_band_frequencies_to_indices(filtered_frequencies, band_freqs)
            all_band_indices.append(indices)
            all_band_limits.append(band_limits)
            all_band_freqs.append(band_freqs)
            band_sizes.append(len(band_freqs))
            limit_name_list.append(limit_name)
            limit_band_idx_list.append(band_idx)  # <-- NOUVEAU
    max_len = max(len(b) for b in all_band_indices)
    for arr in all_band_freqs:
        assert np.all(arr >= 0), "Erreur : une fréquence réelle vaut moins que 0 !"
    all_band_indices = np.stack([pad(arr, -1, max_len) for arr in all_band_indices])
    all_band_limits = np.stack([pad(arr, 1e20, max_len) for arr in all_band_limits])
    all_band_freqs = np.stack([pad(arr, -1, max_len) for arr in all_band_freqs])
    band_sizes = np.array(band_sizes)
    return (all_band_indices, all_band_limits, all_band_freqs,
            band_sizes, limit_name_list, limit_band_idx_list)

# Le Numba doit prendre band_sizes :
def map_band_frequencies_to_indices(filtered_frequencies, band_freqs, tol=1e-6):
    indices = []
    for freq in band_freqs:
        diffs = np.abs(filtered_frequencies - freq)
        idx = np.argmin(diffs)
        if diffs[idx] < tol:
            indices.append(idx)
        else:
            indices.append(-1)
    return np.array(indices)

def detect_fft_exceedings_numba(fft_stream, filtered_frequencies, limit_manager):
    fft_matrix = np.vstack(fft_stream.stream).astype(np.float64)
    timestamps = np.array(fft_stream.timestamps, dtype=np.float64)
    (all_band_indices, all_band_limits, all_band_freqs,
     band_sizes, limit_name_list, limit_band_idx_list) = prepare_limits_for_numba(limit_manager, filtered_frequencies)
    for limit_name, limit_data in limit_manager.limits_by_name.items():
        print(f"LIMIT: {limit_name}")
        for idx, band in enumerate(limit_data["interpolated_bands"]):
            print(f"  Band {idx}: freqs={band['frequencies']}, limits={band['limits']}")

    res = detect_exceedings_numba_parallel(
        fft_matrix, timestamps, all_band_indices, all_band_limits, all_band_freqs, band_sizes
    )
    exceeding_data = Exceedings_FFT()
    for row in res:
        timestamp, bandidx, freq, _ = row
        if freq < 0 or timestamp < 0:
            continue
        limit_name = limit_name_list[int(bandidx)]
        band_idx = limit_band_idx_list[int(bandidx)]
        print(f"AJOUT NUMBA: {limit_name} | band {band_idx} | t={timestamp} | f={freq}")
        exceeding_data.add_exceeding(limit_name, float(timestamp), [float(freq)])

    return exceeding_data

def group_exceeding_intervals(exceedings, tstep):
    """
    Regroupe les timestamps consécutifs (espacés de tstep) en intervalles continus.

    Paramètres :
    - exceedings : liste de tuples (timestamp, [frequencies])
    - tstep : pas temporel

    Retour :
    - liste de tuples (start_time, end_time)
    """
    if not exceedings:
        return []

    # Extraire et trier les timestamps uniques
    timestamps = sorted(set(ts for ts, _ in exceedings))
    grouped = []
    start = current = timestamps[0]

    for ts in timestamps[1:]:
        if np.isclose(ts, current + tstep, rtol=1e-6):  # tolérance numérique
            current = ts
        else:
            grouped.append((start, current + tstep))
            start = current = ts
    grouped.append((start, current + tstep))  # ajouter le dernier groupe
    print("grouped")
    return grouped

def calculate_fft_stream(catenary_signal, win_size_seconds, overlap_percent, n_average, window_type, freq_max_limit):
    temporal_signal_data = catenary_signal.data  # Données temporelles brutes
    temporal_signal_dt = float(round(catenary_signal.dt, 6))  # Pas temporel
    temporal_signal_length = catenary_signal.length  # Longueur des données
    # Vérifications
    if temporal_signal_data is None or temporal_signal_dt is None or temporal_signal_length is None:
        raise ValueError("Le signal 'Catenary Current' est incomplet.")

    fs = int(round(1.0 / temporal_signal_dt))  # Fréquence d'échantillonnage
    # Assurez que fft_size >= win_size_samples
    fft_size = int(round(1 / (temporal_signal_dt)))
    print(f"fft_size : {fft_size}")
    freq_resolution = fs / fft_size

    try:
        freq_max_limit = float(freq_max_limit)  # Convertir en float
        print(f"max freq text2: {freq_max_limit}")
        if freq_max_limit > fs / 2:
            freq_max_limit = fs / 2  # Limiter à fs / 2 si dépassement
            print(f"freq_max_limit ajustée à {freq_max_limit} Hz")
    except ValueError:
        print(f"max freq text (valeur par défaut utilisée) : {freq_max_limit}")
        freq_max_limit = fs / 2
    frequencies = np.linspace(0, fs / 2, fft_size // 2 + 1)  # Obtenir les fréquences associées aux bins FFT
    valid_indices = frequencies <= freq_max_limit
    filtered_frequencies = frequencies[valid_indices]
    filtered_fft_size = len(filtered_frequencies)  # Nouvelle taille après filtration

    # Initialisation de ShortTimeFFT
    # Taille de la fenêtre (en échantillons)
    win_size_samples = int(win_size_seconds * fs)
    hop_size = int(win_size_samples * (1 - overlap_percent))  # Décalage entre les fenêtres

    window = get_window(window_type, win_size_samples)
    short_time_fft = ShortTimeFFT(win=window, hop=hop_size, fs=fs, fft_mode='onesided', scale_to='magnitude')

    # Calcul de la FFT glissante
    S = short_time_fft.stft(temporal_signal_data)  # Matrice des FFTs calculées (complexe)

    # Calcul des timestamps
    timestamps = np.arange(0, len(temporal_signal_data) - win_size_samples + 1, hop_size) / fs
    fft_stream = FFT_Stream(
        name="Sliding_FFT_Catenary_Current",
        fft_size=filtered_fft_size,  # Utiliser la nouvelle taille FFT filtrée
        freq_resolution=freq_resolution,
        time_step=hop_size / fs,
    )
    # Ajouter les amplitudes FFT au flux FFT en excluant les fréquences au-delà de `freq_max_limit`
    for timestamp, fft_data in zip(timestamps, np.sqrt(2) * np.abs(S.T)):
        filtered_fft_data = fft_data[valid_indices]  # Filtrer les données FFT
        fft_stream.add_fft(filtered_fft_data, timestamp)
    return fft_stream

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
            if value in ["Pantograph Current", "Panthograph Voltage", "Monitoring", "Désactivé"]:
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


class ConfigTab(QWidget):
    def __init__(self, main_window):

        super().__init__(main_window)
        self.main_window = main_window

        self.visu_tab = None
        self.file_path = None
        self.channels_config = self.parent().channels_config
        self.temporal_signals = self.parent().temporal_signals
        self.fft_streams = self.parent().fft_streams
        self.layout = QVBoxLayout(self)

        # ----- Cadre pour la Partie "Signal" -----
        signal_group_box = QGroupBox("Configuration des Signaux")
        signal_layout = QVBoxLayout(signal_group_box)
        self.layout.addWidget(signal_group_box)

        # Fichier TDMS
        file_import_layout = QHBoxLayout()
        self.file_label = QLabel("Fichier TDMS:")
        self.file_path_line_edit = QLineEdit(self)
        self.file_path_line_edit.setReadOnly(True)
        self.file_dialog_button = QPushButton("Charger un fichier")
        file_import_layout.addWidget(self.file_label)
        file_import_layout.addWidget(self.file_path_line_edit)
        file_import_layout.addWidget(self.file_dialog_button)
        signal_layout.addLayout(file_import_layout)
        self.file_dialog_button.clicked.connect(self.open_file)

        # Table de signal
        self.table_model = ChannelTableModel(self.channels_config)
        self.channels_table = QTableView()
        self.channels_table.setModel(self.table_model)
        self.channels_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.channels_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        signal_layout.addWidget(self.channels_table)
        self.channels_table.setItemDelegateForColumn(
            2, ComboBoxDelegate(["Volts", "Ampère", "Vitesse", "Couple"], self.channels_table)
        )
        self.channels_table.setItemDelegateForColumn(
            5, ComboBoxDelegate(["Pantograph Current", "Panthograph Voltage", "Monitoring", "Désactivé"],
                                self.channels_table)
        )

        # Bouton pour valider les signaux
        self.preproc_button = QPushButton("Valider les signaux", self)
        self.preproc_button.clicked.connect(self.preprocess_all)
        signal_layout.addWidget(self.preproc_button)
        self.layout.addWidget(signal_group_box)

        # ----- Cadre pour la Partie "FFT et Limites" -----
        fft_limits_group_box = QGroupBox("Paramètres FFT & Limites")
        fft_limits_layout = QVBoxLayout(fft_limits_group_box)

        # FFT params
        param_fft_layout = QHBoxLayout()
        self.fft_win_size_edit = QLineEdit()
        self.fft_win_size_edit.setPlaceholderText("Fenêtre (s)")
        self.fft_win_size_edit.setText("1")
        self.fft_window_type_combo = QComboBox()
        self.fft_window_type_combo.addItems(["hann", "gaussian", "rect", "blackman", "hamming"])
        self.fft_overlap_edit = QLineEdit()
        self.fft_overlap_edit.setPlaceholderText("Recouvrement (%)")
        self.fft_overlap_edit.setText("50")
        self.fft_average_spin = QSpinBox()
        self.fft_average_spin.setMinimum(1)
        self.fft_average_spin.setMaximum(100)
        self.fft_average_spin.setValue(1)
        self.fft_average_spin.setFixedWidth(60)
        param_fft_layout.addWidget(QLabel("Fenêtre :"))
        param_fft_layout.addWidget(self.fft_window_type_combo)
        param_fft_layout.addWidget(QLabel("Taille (s)"))
        param_fft_layout.addWidget(self.fft_win_size_edit)
        param_fft_layout.addWidget(QLabel("Recouvrement (%)"))
        param_fft_layout.addWidget(self.fft_overlap_edit)
        param_fft_layout.addWidget(QLabel("Nb fen. à moyenner"))
        param_fft_layout.addWidget(self.fft_average_spin)
        fft_limits_layout.addLayout(param_fft_layout)

        # Section pour le dossier des limites FFT
        self.limit_manager = LimitManager()
        limites_fft_layout = QVBoxLayout()
        limites_fft_label = QLabel("Dossier des limites FFT :")
        self.limites_fft_dossier_edit = QLineEdit(self)
        self.limites_fft_dossier_edit.setReadOnly(True)
        self.limites_fft_dossier_button = QPushButton("Sélectionner un dossier")
        self.limites_fft_dossier_button.clicked.connect(self.fft_limit_folder_selection)
        limites_fft_layout.addWidget(limites_fft_label)
        limites_fft_layout.addWidget(self.limites_fft_dossier_edit)
        limites_fft_layout.addWidget(self.limites_fft_dossier_button)

        # Section fréquences min/max
        freq_layout = QHBoxLayout()
        self.keep_freq_values_checkbox = QCheckBox("Conserver les valeurs")
        self.keep_freq_values_checkbox.setChecked(False)

        self.freq_min_edit = QLineEdit(self)
        self.freq_min_edit.setPlaceholderText("Fréquence min")
        self.freq_max_edit = QLineEdit(self)
        self.freq_max_edit.setPlaceholderText("Fréquence max")
        freq_layout.addWidget(QLabel("Fréquence min :"))
        freq_layout.addWidget(self.freq_min_edit)
        freq_layout.addWidget(QLabel("Fréquence max :"))
        freq_layout.addWidget(self.freq_max_edit)
        freq_layout.addWidget(self.keep_freq_values_checkbox)
        limites_fft_layout.addLayout(freq_layout)
        fft_limits_layout.addLayout(limites_fft_layout)
        self.layout.addWidget(fft_limits_group_box)
        # Bouton pour lancer l'annalyse FFT
        self.process_FFT_button = QPushButton("Lancer l'analyse FFT", self)
        self.process_FFT_button.clicked.connect(self.process_FFT_analysis)
        fft_limits_layout.addWidget(self.process_FFT_button)

        self.layout.addWidget(fft_limits_group_box)

        # ----- Cadre pour la Partie "Psophométrie" -----
        psophometry_group_box = QGroupBox("Psophométrie")
        psophometry_layout = QVBoxLayout(psophometry_group_box)

        # File Selection for weights
        limit_layout = QHBoxLayout()
        self.psopho_limit_edit = QLineEdit()
        self.psopho_limit_edit.setText("1.5")
        limit_layout.addWidget(QLabel("Limite courant psophométrique (Arms):"))
        limit_layout.addWidget(self.psopho_limit_edit)
        psophometry_layout.addLayout(limit_layout)
        file_weights_layout = QHBoxLayout()
        self.file_weights_button = QPushButton("Sélectionner fichier des poids fréquentiels", self)
        self.file_weights_button.clicked.connect(self.load_weights_file)
        self.file_weights_line_edit = QLineEdit(self)
        self.file_weights_line_edit.setPlaceholderText("Fichier des poids")
        self.file_weights_line_edit.setReadOnly(True)
        file_weights_layout.addWidget(QLabel("Poids Fréquentiels:"))
        file_weights_layout.addWidget(self.file_weights_line_edit)
        file_weights_layout.addWidget(self.file_weights_button)
        psophometry_layout.addLayout(file_weights_layout)

        # FFT Window Size
        fft_psophometry_layout = QHBoxLayout()
        self.psophometry_win_size_edit = QLineEdit()
        self.psophometry_win_size_edit.setText("1")

        self.psophometry_overlap = QLineEdit()
        self.psophometry_overlap.setText("80")
        fft_psophometry_layout.addWidget(QLabel("Taille fenêtre FFT (s):"))
        fft_psophometry_layout.addWidget(self.psophometry_win_size_edit)
        fft_psophometry_layout.addWidget(QLabel("Overlap (%):"))
        fft_psophometry_layout.addWidget(self.psophometry_overlap)
        psophometry_layout.addLayout(fft_psophometry_layout)

        # Compute Button
        self.compute_psophometry_button = QPushButton("Calculer le courant psophométrique", self)
        self.compute_psophometry_button.clicked.connect(self.compute_psophometry)
        psophometry_layout.addWidget(self.compute_psophometry_button)
        self.layout.addWidget(psophometry_group_box)

        self.config = {}
        self.exceedings_fft = Exceedings_FFT()

    def get_max_analysis_frequency(self):
        # Calcul de la fréquence d'échantillonnage
        dt = self.get_catenary_signal().dt
        fs = int(round(1.0 / float(round(dt, 6))))  # Fréquence d'échantillonnage

        try:
            freq_max_limit = float(self.freq_max_edit.text().strip())  # Convertir en float
            print(f"max freq text2: {freq_max_limit}")
            if freq_max_limit > fs / 2:
                freq_max_limit = fs / 2  # Limiter à fs / 2 si dépassement
                print(f"freq_max_limit ajustée à {freq_max_limit} Hz")
        except ValueError:
            print(f"max freq text (valeur par défaut utilisée) : {freq_max_limit}")
            freq_max_limit = fs / 2
        return freq_max_limit

    def load_weights_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner le fichier des poids fréquentiels", "",
                                                   "Fichiers CSV (*.csv)")
        if file_path:
            self.file_weights_line_edit.setText(file_path)
            print(f"File loaded: {file_path}")

    def get_catenary_signal(self):
        """
        Récupère le signal 'Catenary Current' depuis les données stockées.
        """
        group_name = "Catenary Current"
        catenary_signals = list(self.temporal_signals.get_signals_by_group(group_name))
        if not catenary_signals:
            return None

        catenary_signal = catenary_signals[0]
        data = catenary_signal.data
        dt = float(round(catenary_signal.dt, 6))
        length = catenary_signal.length

        # Vérifications de validité
        if data is None or dt is None or length is None:
            raise ValueError("Le signal 'Catenary Current' est incomplet.")

        return catenary_signal

    def detect_fft_exceedings_fast(self,fft_stream, filtered_frequencies, limit_manager):
        """
        Wrapper : prépare la matrice FFT, indices, limites, appelle la fonction Numba puis convertit le résultat.
        Retourne : exceeding_data (ex: objet Exceedings_FFT ou dict selon ton infra)
        """
        fft_matrix = np.vstack(fft_stream.stream).astype(np.float64)
        timestamps = np.array(fft_stream.timestamps, dtype=np.float64)
        (all_band_indices, all_band_limits, all_band_freqs,
         band_sizes, limit_name_list, limit_band_idx_list) = prepare_limits_for_numba(limit_manager,
                                                                                      filtered_frequencies)
        for limit_name, limit_data in limit_manager.limits_by_name.items():
            print(f"LIMIT: {limit_name}")
            for idx, band in enumerate(limit_data["interpolated_bands"]):
                print(f"  Band {idx}: freqs={band['frequencies']}, limits={band['limits']}")

        res = detect_exceedings_numba_parallel(
        fft_matrix, timestamps, all_band_indices, all_band_limits, all_band_freqs, band_sizes
        )
        # Crée un vrai Exceedings_FFT
        exceeding_data = Exceedings_FFT()
        for row in res:
            timestamp, bandidx, freq, _ = row
            if freq < 0 or timestamp < 0:
                continue
            limit_name = limit_name_list[int(bandidx)]
            # note : band_idx_list n’est pas nécessaire ici, sauf si tu veux l’afficher
            exceeding_data.add_exceeding(limit_name, float(timestamp), [float(freq)])

        return exceeding_data

    def get_Sliding_FFT_parameters(self):
        """
        Récupère les paramètres de la FFT glissante à partir des champs d'entrée.
        """
        # Filtrer les fréquences au-delà de freq_max_limit
        freq_max_analysis = self.get_max_analysis_frequency()
        # récupérer les paramètres FFT
        win_size_seconds = float(self.fft_win_size_edit.text())  # Durée de la fenêtre FFT (en secondes)
        overlap_percent = float(self.fft_overlap_edit.text()) / 100.0  # Recouvrement entre fenêtres
        n_average = int(self.fft_average_spin.value())  # Nombre de fenêtres à moyenner
        window_type = self.fft_window_type_combo.currentText()  # Type de fenêtre FFT
        return (win_size_seconds, overlap_percent, n_average, window_type, freq_max_analysis)

    def compute_sliding_fft(self):
        """
        Calcule la FFT glissante à partir du signal 'Catenary Current' en utilisant `scipy.signal.ShortTimeFFT`.
        """
        catenary_signal = self.get_catenary_signal()
        if not catenary_signal:
            print(f"Aucun signal trouvé dans le groupe 'Catenary Current'.")
            return
        print("Calcul de la FFT glissante...")
        start_time = time.time()

        win_size_seconds, overlap_percent, n_average, window_type, freq_max_analysis = self.get_Sliding_FFT_parameters()
        fft_stream: FFT_Stream = calculate_fft_stream(catenary_signal, win_size_seconds, overlap_percent, n_average,
                                                      window_type, freq_max_analysis)
        filtered_frequencies = fft_stream.calculate_frequencies()
        timestamps = fft_stream.timestamps
        self.visu_tab.monitoring_tstep = timestamps[1]
        # Initialisation de la structure pour les dépassements
        self.exceedings_fft = Exceedings_FFT()
        # Interpoler les limites FFT
        print("Interpolation des limites FFT...")
        self.limit_manager.interpolate_limits(
            frequencies=filtered_frequencies  # Fréquences filtrées
        )
        stop_time = time.time()
        print(f"FFT glissante calculée en {stop_time - start_time:.2f} secondes.")
        print("Detection des dépassements...")
        start_time = time.time()
        self.exceedings_fft = self.detect_fft_exceedings_fast(fft_stream, filtered_frequencies, self.limit_manager
        )
        stop_time = time.time()
        print(f"Dépassements détectés en {stop_time - start_time:.2f} secondes.")
        #print(f"dépassements : {self.exceedings_fft.exceedings}")
        # Ajouter les dépassements à VisuTab
        print("Calcul du maxhold...")
        start_time = time.time()
        self.visu_tab.exceedings_fft = self.exceedings_fft

        fft_stream.calculate_maxhold()
        stop_time = time.time()
        print(f"maxhold claculé en {stop_time - start_time:.2f} secondes.")

        # Ajouter le FFT_Stream au groupe

        self.fft_streams.add_stream(fft_stream, "Catenary Current")

        # Confirmation
        print(f"FFT glissante calculée : {fft_stream.name}, Nombre de fenêtres : {len(fft_stream.timestamps)}")
        QMessageBox.information(self, "Success", f"FFT glissante calculée pour 'Catenary Current'.")

    def clear_limits(self):
        """
        Supprime toutes les limites FFT et réinitialise l'interface.
        """
        self.limit_manager.clear_limits()  # Nettoyage via LimitManager
        if self.keep_freq_values_checkbox.isChecked() == False:
            self.freq_min_edit.setText("")  # Réinitialiser l'affichage
            self.freq_max_edit.setText("")
        print("Toutes les limites FFT ont été réinitialisées.")

    def  compute_psophometry(self):
        """
        Calcule le courant psophométrique en appliquant une FFT glissante avec pondération des fréquences.
        """
        # ——— VIDE L’ONGLET PSOPHO AVANT TOUT ———
        tab = self.main_window.tab_psophometry
        tab.psophometric_signal = None
        tab.fft_psopho_stream = None
        tab.psopho_plot.clear()
        tab.weighted_fft_plot.clear()
        tab.fft_slider.setRange(0, 0)
        # Étape 1 : Récupération du signal "Catenary Current"
        group_name = "Catenary Current"
        catenary_signals = list(self.temporal_signals.get_signals_by_group(group_name))
        if not catenary_signals:
            QMessageBox.warning(self, "Erreur", f"Aucun signal trouvé dans le groupe {group_name}.")
            return

        catenary_signal = catenary_signals[0]
        data = catenary_signal.data  # Données temporelles brutes
        dt = float(round(catenary_signal.dt, 6))  # Pas temporel
        length = catenary_signal.length  # Longueur des données

        # Vérifications de la validité du signal
        if data is None or dt is None or length is None:
            raise ValueError("Le signal 'Catenary Current' est incomplet.")

        # Étape 2 : Définir les paramètres FFT
        window_type = "hann"
        win_size_seconds = float(self.psophometry_win_size_edit.text())  # Durée de la fenêtre FFT (en secondes)
        overlap_percent = float(self.psophometry_overlap.text()) / 100.0  # Recouvrement entre fenêtres
        fs = int(round(1.0 / dt))  # Fréquence d'échantillonnage

        # Taille de la fenêtre (en échantillons)
        win_size_samples = int(win_size_seconds * fs)
        hop_size = int(win_size_samples * (1 - overlap_percent))  # Décalage entre les fenêtres

        fft_size = win_size_samples
        freq_resolution = fs / fft_size

        # Étape 3 : Initialisation de ShortTimeFFT
        from scipy.signal import ShortTimeFFT
        window = sp.signal.get_window(window_type, win_size_samples)
        short_time_fft = ShortTimeFFT(win=window, hop=hop_size, fs=fs, fft_mode='onesided', scale_to='magnitude')

        # Étape 4 : Charger le fichier des poids fréquentiels
        weights_df = pd.read_csv(self.file_weights_line_edit.text(), sep=";", decimal=".")
        freqs = weights_df["Freq"].values  # Liste des fréquences issues du fichier CSV
        coeffs = weights_df["Coeff"].values  # Coefficients pour chaque fréquence
        freq_max_text = freqs[-1]  # Fréquence maximale définie dans le fichier CSV

        # Vérification et traitement de freq_max_text
        try:
            freq_max_limit = float(freq_max_text)
            if freq_max_limit > fs / 2:
                freq_max_limit = fs / 2  # Limiter à la fréquence de Nyquist (fs / 2)
        except ValueError:
            freq_max_limit = fs / 2  # Utiliser la limite par défaut (Nyquist)

        # Étape 5 : Calcul de la FFT glissante
        S = short_time_fft.stft(data)  # Matrice des FFT complexes glissantes
        timestamps = np.arange(0, len(data) - win_size_samples + 1, hop_size) / fs

        # Filtrer les fréquences au-delà de freq_max_limit
        frequencies = np.linspace(0, fs / 2, fft_size // 2 + 1)  # Frequencies associées aux bins FFT
        valid_indices = frequencies <= freq_max_limit
        filtered_frequencies = frequencies[valid_indices]

        # Interpolation des coefficients sur les fréquences disponibles
        from scipy.interpolate import interp1d
        interpolate_weights = interp1d(freqs, coeffs, kind='linear', bounds_error=False, fill_value=0)
        interpolated_coeffs = interpolate_weights(filtered_frequencies)

        # Étape 6 : Initialisation du FFT_Stream (pour filtrer et gérer les données FFT)
        fft_stream = FFT_Stream(
            name="Sliding_FFT_Psophometric",
            fft_size=len(filtered_frequencies),  # Taille après filtration
            freq_resolution=freq_resolution,
            time_step=hop_size / fs
        )

        # Ajouter les amplitudes FFT au flux FFT
        for timestamp, fft_data in zip(timestamps, np.sqrt(2) * np.abs(S.T)):
            filtered_fft_data = interpolated_coeffs*fft_data[valid_indices]  # Filtrer les fréquences au-delà de freq_max_limit
            fft_stream.add_fft(filtered_fft_data, timestamp)

        # Étape 7 : Calcul du courant psophométrique à partir du flux FFT
        psophometric_signal = Temporal_Signal(name="Psophometric Current", dt=hop_size / fs, length=len(timestamps))
        psophometric_values = []
        print()
        for fft_data in fft_stream.stream:
            weighted_fft = fft_data # Appliquer les poids interpolés
            #modifier le signal psophométrique
            squared_sum = np.sum(weighted_fft ** 2)  # Somme des carrés
            psophometric_value = np.sqrt(squared_sum)  # Racine de la valeur
            psophometric_values.append(psophometric_value)

        # Étape 8 : Stocker le signal psophométrique
        psophometric_signal.data = np.array(psophometric_values)
        self.temporal_signals.add_signal(psophometric_signal, "Psophometry")

        # Affichage dans le tab Psophometry
        self.main_window.tab_psophometry.fft_psopho_stream = fft_stream
        self.main_window.tab_psophometry.psophometric_signal = psophometric_signal
        self.main_window.tab_psophometry.plot_psophometric_current()

        try:
            psopho_limit = float(self.psopho_limit_edit.text())
        except ValueError:
            psopho_limit = None

        self.main_window.tab_psophometry.psopho_limit = psopho_limit

        print("Courant psophométrique calculé et affiché avec succès.")

    def fft_limit_folder_selection(self):
        """
        Sélectionne un dossier contenant des fichiers limites FFT et utilise LimitManager pour les charger.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier limite FFT")
        if folder_path:
            self.limites_fft_dossier_edit.setText(folder_path)  # Afficher le chemin dans le QLineEdit

            # Charger les limites avec LimitManager
            self.limit_manager.load_limits_from_folder(folder_path)

            # Mettre à jour les champs de fréquence min/max

            if self.limit_manager.limit_min_frequency is not None:
                if self.keep_freq_values_checkbox.isChecked() == False:
                    self.freq_min_edit.setText(str(self.limit_manager.limit_min_frequency))
            else:
                self.freq_min_edit.setText("Inconnu")

            if self.limit_manager.limit_max_frequency is not None:
                if self.keep_freq_values_checkbox.isChecked() == False:
                    self.freq_max_edit.setText(str(self.limit_manager.limit_max_frequency))
            else:
                self.freq_max_edit.setText("Inconnu")

    def open_file(self):
        self.clear_limits()
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
                    self.channels_config = pd.DataFrame(data)
                    self.table_model._df = self.channels_config
                    self.table_model.layoutChanged.emit()
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier TDMS:\n{e}")

    def preprocess_all(self):
        try:
            self.temporal_signals.clear_signals()
            self.fft_streams.clear_streams()
            if not self.file_path:
                QMessageBox.critical(self, "Erreur", "Fichier TDMS non chargé.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"{e}")

        # Réinitialiser la visualisation avant le traitement
        self.visu_tab.reset_visualization()
        channel_config = self.channels_config
        tdms_file_path = self.file_path
        catenary_voltage_signal_is_founded = False
        with TdmsFile.open(tdms_file_path) as tdms_file:
            channel_names = []
            panto_signals = Temporal_Signals()
            for i, row in channel_config.iterrows():
                if row["Post-traitement"] == "Désactivé":
                    continue
                if row["Post-traitement"] == "Monitoring":
                    ch_name = row["Signal d'origine"]
                    new_name = row["Nouveau nom"]
                    coeff, offset = float(row["Coeff"]), float(row["Offset"])
                    for group in tdms_file.groups():
                        for channel in group.channels():
                            if channel.name == ch_name:
                                if channel.name == ch_name:
                                    dt = round(channel.properties.get("wf_increment", 1.0), 6)
                                    if coeff != 1.0 or offset != 0.0:
                                        sig = channel[:] * coeff + offset
                                    else:
                                        sig = channel[:]
                                    length = len(sig)
                                    print(f"data name : {new_name} length = {length}; dt = {dt}")
                                    time_signal = Temporal_Signal(name=new_name, dt=dt, length=length, data=sig)
                                    print(f"time_signal : {time_signal}")
                                    self.temporal_signals.add_signal(time_signal, "Monitoring")
                                    print(f"signals : {self.temporal_signals}")
                if row["Post-traitement"] == "Pantograph Current":
                    ch_name = row["Signal d'origine"]
                    output_channel_name = "Catenary Current"
                    coeff, offset = float(row["Coeff"]), float(row["Offset"])
                    dt = None
                    length = None
                    panto_signals = []  # Initialisation de la liste des signaux temporels
                    catenary_current_signal = Temporal_Signal(name=output_channel_name)  # Signal de sortie initialisé

                    # Parcours des groupes et canaux TDMS
                    for group in tdms_file.groups():
                        for channel in group.channels():
                            if channel.name == ch_name:  # Vérification du nom du canal
                                dt = round(channel.properties.get("wf_increment", 1.0), 6)

                                # Application des coefficients et des offsets
                                if coeff != 1.0 or offset != 0.0:
                                    sig = channel[:] * coeff + offset
                                else:
                                    sig = channel[:]

                                # Création du signal temporel et ajout à la liste
                                length = len(sig)
                                panto_signals.append(Temporal_Signal(name=channel.name, dt=dt, length=length, data=sig))

                    catenary_current_signal.length = length
                    catenary_current_signal.dt = dt
                    # Ajout des données des signaux temporels dans le signal de courant caténaire
                    for signal in panto_signals:
                        if catenary_current_signal.data is None:
                            # Initialise les données si elles sont `None`
                            catenary_current_signal.data = signal.data.copy()
                        else:
                            # Cumul des données des différents signaux
                            catenary_current_signal.data += signal.data
                    self.temporal_signals.add_signal(catenary_current_signal, "Catenary Current")
                    self.temporal_signals.add_signal(catenary_current_signal, "Monitoring")

                    # (Optionnel : vérifier ou utiliser catenary_current_signal en dehors du bloc)
                if row["Post-traitement"] == "Pantograph Voltage":
                    if catenary_voltage_signal_is_founded == False:
                        ch_name = row["Signal d'origine"]
                        output_channel_name = "Catenary Voltage"
                        coeff, offset = float(row["Coeff"]), float(row["Offset"])
                        dt = None
                        length = None
                        catenary_voltage_signal = Temporal_Signal(
                            name=output_channel_name)  # Signal de sortie initialisé

                        # Parcours des groupes et canaux TDMS
                        for group in tdms_file.groups():
                            for channel in group.channels():
                                if channel.name == ch_name:  # Vérification du nom du canal
                                    dt = round(channel.properties.get("wf_increment", 1.0), 6)

                                    # Application des coefficients et des offsets
                                    if coeff != 1.0 or offset != 0.0:
                                        sig = channel[:] * coeff + offset
                                    else:
                                        sig = channel[:]

                                    # Création du signal temporel et ajout à la liste
                                    catenary_voltage_signal.length = len(sig)
                                    catenary_voltage_signal.dt = dt
                                    catenary_voltage_signal.data = signal.data.copy()
                                    catenary_voltage_signal_is_founded = True
                                    self.temporal_signals.add_signal(catenary_voltage_signal, "Pantograph Voltage")

                    # (Optionnel : vérifier ou utiliser catenary_current_signal en dehors du bloc)
            if catenary_voltage_signal_is_founded == True:
                print(f"Signal généré : {catenary_current_signal.name}, Length : {catenary_current_signal.length}")
            print(f"data : {catenary_current_signal.data}")
            self.temporal_signals.add_signal(catenary_current_signal, "Pantograph Current")
        print(self.temporal_signals.__dict__)
        self.visu_tab.plot_monitored_signals()

    def process_FFT_analysis(self):
        self.compute_sliding_fft()
        print("affichage des FFT glissantes et de leurs limites...")
        start_time = time.time()
        print("OKOKOK 1 Analyse FFT glissante terminée.")
        self.visu_tab.plot_limits()
        print("OKOKOK 2 afiichage des limites fait.")
        self.visu_tab.plot_fft_stream()
        stop_time = time.time()
        print(f"FFT glissante affichée et limite en {stop_time - start_time:.2f} secondes.")
        start_time = time.time()
        print("OKOKOK 3 Affichage des FFT.")
        self.visu_tab.add_exceeding_zones()
        stop_time = time.time()
        print(f"Zones de dépassement ajoutées en {stop_time - start_time:.2f} secondes.")
        print("OKOKOK 4 Affichage des dépassement fait.")

    def set_visu_tab(self, tab_visu):
        self.visu_tab = tab_visu


class VisuTab(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fft_streams = self.parent().fft_streams
        self.temporal_signals = self.parent().temporal_signals
        self.monitoring_tmin = None
        self.monitoring_tmax = None
        self.monitoring_tstep = None
        self.max_display_points = 5000  # Limite de points affichés
        self.downsampling_factor = None
        self.signal_sample_number = None
        self.plot_widgets = []  # Liste pour stocker les PlotWidget créés
        self.cursors = []  # Liste pour stocker les curseurs verticaux
        self.configtab = None
        self.exceedings_fft = None

        # Effacer la barre de menu, puisque ce n'est pas nécessaire ici
        self.menuBar().hide()

        # Création du dock pour les tracés "Monitoring"
        self.monitoring_dock = QDockWidget("Monitoring", self)
        self.monitoring_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.monitoring_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        monitoring_widget = QWidget(self.monitoring_dock)
        self.monitoring_layout = QVBoxLayout()
        monitoring_widget.setLayout(self.monitoring_layout)  # Intégrer le layout au widget
        self.monitoring_dock.setWidget(monitoring_widget)  # Ajouter le widget au dock
        self.addDockWidget(Qt.TopDockWidgetArea, self.monitoring_dock)  # Positionner Monitoring en bas

        # Création du dock pour le tracé FFT
        self.fft_dock = QDockWidget("FFT", self)
        self.fft_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.fft_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        fft_widget = QWidget(self.fft_dock)
        self.fft_layout = QVBoxLayout()
        fft_widget.setLayout(self.fft_layout)  # Intégrer le layout au widget
        self.fft_dock.setWidget(fft_widget)  # Ajouter le widget au dock
        self.addDockWidget(Qt.BottomDockWidgetArea, self.fft_dock)  # Positionner FFT en haut

        # Plot FFT
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setBackground("w")  # Fond blanc pour le graphe FFT
        self.fft_plot.setLabel('bottom', 'Fréquence', units='Hz')
        self.fft_plot.setLabel('left', 'Amplitude', units='')
        self.fft_layout.addWidget(self.fft_plot)

        # Slider FFT
        self.slider = QSlider(Qt.Horizontal)  # Slider horizontal pour le tracé FFT
        self.slider.valueChanged.connect(self.on_slider_change)
        self.fft_layout.addWidget(self.slider)

        # Références
        self.timestamps = None
        self.first_plot = None

    def add_exceeding_zones(self):
        """
        Ajoute des zones semi-transparentes aux graphes de monitoring pour indiquer les timestamps
        qui dépassent les limites FFT, en regroupant les dépassements consécutifs.
        """
        if not self.exceedings_fft or not self.exceedings_fft.get_all_exceedings():
            print("Aucun dépassement enregistré.")
            return

        if self.monitoring_tstep is None or self.monitoring_tmin is None or self.monitoring_tmax is None:
            print("Erreur : monitoring_tstep, monitoring_tmin ou monitoring_tmax n'a pas été initialisé.")
            return

        print("Ajout des zones de dépassement (regroupées)...")

        for limit_name, exceedings in self.exceedings_fft.get_all_exceedings().items():
            color = self.exceedings_fft.get_color(limit_name)
            grouped_intervals = group_exceeding_intervals(exceedings, self.monitoring_tstep)

            for rect_start, rect_end in grouped_intervals:
                if rect_end < self.monitoring_tmin or rect_start > self.monitoring_tmax:
                    continue  # zone hors du champ temporel affiché

                for plot_widget, _ in self.plot_widgets:
                    region = pg.LinearRegionItem([rect_start, rect_end], brush=pg.mkBrush(color))
                    region.setZValue(-10)
                    plot_widget.addItem(region)

    def plot_monitored_signals(self):
        """
        Affiche tous les signaux monitorés dans la zone de monitoring,
        et initialise les plages temporelles pour les zones de dépassements.
        """
        # Effacer les anciens tracés
        self.plot_widgets.clear()
        self.cursors.clear()
        while self.monitoring_layout.count() > 0:
            plot_item = self.monitoring_layout.takeAt(0)
            if plot_item.widget():
                plot_item.widget().deleteLater()

        self.first_plot = None

        # Ajouter un graphique par signal dans "Monitoring"
        for signal in self.temporal_signals.get_signals_by_group("Monitoring"):
            if signal:
                plot_widget = pg.PlotWidget()
                self.monitoring_layout.addWidget(plot_widget)

                # Ajouter le PlotWidget à la liste pour gestion ultérieure
                self.plot_widgets.append((plot_widget, signal.name))

                # Configuration pour afficher toutes les données
                time = signal.get_time_axis()
                data = signal.data

                if self.first_plot is None:
                    # Initialiser signal_dt en fonction du pas temporel du premier signal
                    self.signal_dt = signal.dt
                    # Initialiser les limites temporelles pour les zones
                    self.monitoring_tmin = time[0]
                    self.monitoring_tmax = time[-1]
                    self.signal_sample_number = len(data)
                    i = self.signal_sample_number
                    self.downsampling_factor = 1
                    while i >= self.max_display_points:
                        i = i // 2
                        self.downsampling_factor = self.downsampling_factor * 2

                time_ds = time[::self.downsampling_factor]
                data_ds = data[::self.downsampling_factor]

                # Configuration du fond blanc
                plot_widget.setBackground("w")  # Fond blanc
                plot_widget.plot(time_ds, data_ds, pen=pg.mkPen("#00a78e", width=2))  # Couleur courbe
                plot_widget.setLabel('bottom', 'Temps', units='s')
                plot_widget.setLabel('left', signal.name)
                plot_widget.showGrid(x=True, y=True)

                view_box = plot_widget.getViewBox()
                view_box.setLimits(xMin=self.monitoring_tmin, xMax=self.monitoring_tmax)

                if self.first_plot is None:
                    # Connecter la plage visible au premier plot
                    view_box.sigRangeChanged.connect(self.on_range_changed)
                    self.first_plot = plot_widget

                if self.first_plot:
                    # Synchroniser tous les autres graphs avec le premier
                    plot_widget.setXLink(self.first_plot)

        # Ajouter les curseurs verticaux synchronisés
        self.add_cursors()

    def add_cursors(self):
        """
        Ajoute 3 curseurs verticaux synchronisés à tous les graphiques.
        """
        if self.monitoring_tmin is None or self.monitoring_tmax is None:
            print("Erreur : self.monitoring_tmin ou self.monitoring_tmax n'a pas été défini.")
            return

        initial_position = self.monitoring_tmin + (self.monitoring_tmax - self.monitoring_tmin) / 3

        # Liste des curseurs par graphique
        self.cursors = []

        # Créer et ajouter les curseurs
        for plot_widget, _ in self.plot_widgets:
            cursor = pg.InfiniteLine(pos=initial_position, angle=90, movable=True,
                                     pen=pg.mkPen("#ff0000", width=3))  # Rouge, mobile
            plot_widget.addItem(cursor)
            self.cursors.append(cursor)  # Stocker le curseur

            # Connecter les curseurs pour synchronisation
            cursor.sigPositionChanged.connect(lambda c=cursor: self.sync_cursors(c))

    def sync_cursors(self, moved_cursor):
        """
        Synchronise les positions des curseurs lorsqu'un d'eux est déplacé.
        :param moved_cursor: Le curseur déplacé.
        """
        pos = moved_cursor.pos().x()  # Obtenir la nouvelle position X du curseur déplacé
        for cursor in self.cursors:
            if cursor != moved_cursor:  # Ne pas modifier le curseur en cours de déplacement
                cursor.setPos(pos)  # Synchroniser la position des autres curseurs

    def on_range_changed(self, view_box):
        """
        Méthode appelée lorsque la plage visible est modifiée (zoom ou déplacement).
        :param view_box: Le ViewBox qui a émis le signal.
        """
        view_range = view_box.viewRange()  # Retourne [[xmin, xmax], [ymin, ymax]]
        x_range = view_range[0]  # Plage X accessible
        xmin = x_range[0]
        xmax = x_range[1]
        length = (xmax - xmin) / self.signal_dt
        i = length
        self.downsampling_factor = 1
        while i >= self.max_display_points:
            i = i // 2
            self.downsampling_factor = self.downsampling_factor * 2
        print(f"new downsampling_factor : {self.downsampling_factor}")

        # Mettre à jour les données dans chaque signal
        for signal in self.temporal_signals.get_signals_by_group("Monitoring"):
            if signal:
                time = signal.get_time_axis()
                data = signal.data

                indices = (time >= xmin) & (time <= xmax)
                time_visible = time[indices]
                data_visible = data[indices]

                time_ds = time_visible[::self.downsampling_factor]
                data_ds = data_visible[::self.downsampling_factor]

                for plot_widget, plot_name in self.plot_widgets:
                    if plot_name == signal.name:
                        # Trouver le PlotDataItem correspondant au signal dans le PlotItem
                        plot_data_item = next(
                            (item for item in plot_widget.plotItem.items if isinstance(item, pg.PlotDataItem)),
                            None
                        )
                        if plot_data_item:
                            plot_data_item.setData(time_ds, data_ds)  # Mettre à jour les données visibles

    def plot_fft_stream(self):
        """
        Affiche ou met à jour le graphique FFT pour le timestamp sélectionné par le slider.
        """
        print("Plotting FFT Stream...")
        slider_value = self.slider.value()
        fft_streams = self.fft_streams.get_streams_by_group("Catenary Current")
        if fft_streams is None or len(fft_streams) == 0:
            print("Erreur : Aucun FFT_Stream trouvé dans le groupe 'Catenary Current'.")
            return

        fft_stream = next(iter(fft_streams))  # Récupère le premier FFT_Stream
        print("Plotting FFT Stream Step 2...")

        timestamps = fft_stream.timestamps
        frequencies = fft_stream.calculate_frequencies()
        magnitudes = fft_stream.stream  # Magnitudes FFT (amplitudes spectrales)
        # Configurer le slider si ce n'est pas déjà fait
        self.slider.setRange(0, len(timestamps) - 1)  # Ajuster la plage du slider
        self.slider.setTickInterval(1)  # Chaque tick correspond à un timestamp
        self.slider.setTickPosition(QSlider.TicksBelow)  # Position des ticks

        if timestamps.size == 0 or frequencies.size == 0 or magnitudes.size == 0:
            print("Erreur : FFT_Stream est vide ou incomplet.")
            return

        if slider_value < 0 or slider_value >= len(timestamps):
            print(f"Erreur : Valeur de slider ({slider_value}) hors limites : [0, {len(timestamps) - 1}]")
            return

        # Vérifications des données
        if timestamps.size == 0 or frequencies.size == 0 or magnitudes.size == 0:
            print("Erreur : FFT_Stream est vide ou incomplet.")
            return

        if slider_value < 0 or slider_value >= len(timestamps):
            print(f"Erreur : Valeur de slider ({slider_value}) hors limites : [0, {len(timestamps) - 1}]")
            return

        selected_timestamp = timestamps[slider_value]
        selected_magnitude = magnitudes[slider_value]  # Amplitudes spectrales associées au timestamp

        frequencies = frequencies[1:]
        selected_magnitude = selected_magnitude[1:]
        # Si la courbe n'existe pas encore, créez-la
        if not hasattr(self, 'fft_curve'):  # Vérifie si la courbe a été créée
            self.fft_curve = self.fft_plot.plot(frequencies, selected_magnitude, pen=pg.mkPen("#00a78e", width=2))

        else:
            self.fft_curve.setData(frequencies, selected_magnitude)  # Met à jour les fréquences et amplitudes
        print(f"Updating FFT plot for timestamp {selected_timestamp:.3f}s...")

        # Mettre à jour le titre du plot
        self.fft_plot.setTitle(f"FFT : Timestamp={selected_timestamp:.3f}s")
        # Configurer l'échelle logarithmique

        self.fft_plot.getPlotItem().showGrid(x=True, y=True)
        self.fft_plot.getPlotItem().setLogMode(True, True)  # Échelle logarithmique pour Y (amplitudes)

        # Vérifier si nous avons un FFT_Signal MaxHold et l'afficher
        maxhold_signal = fft_stream.get_maxhold()
        print(f"FFT MaxHold : {maxhold_signal.data}")
        if maxhold_signal is not None:
            maxhold_frequencies = maxhold_signal.get_frequency_axis()
            maxhold_data = maxhold_signal.data

            # Afficher la courbe MaxHold si elle n'existe pas encore
            if not hasattr(self, 'maxhold_curve'):  # Vérifie si la courbe MaxHold est créée
                self.maxhold_curve = self.fft_plot.plot(maxhold_frequencies, maxhold_data,
                                                        pen=pg.mkPen("#7fb3d5", width=1, style=Qt.DashLine))
            else:
                # Mettre à jour la courbe MaxHold existante
                self.maxhold_curve.setData(maxhold_frequencies, maxhold_data)

            print("Courbe MaxHold mise à jour.")
            # Limiter la hauteur de monitoring_layout
            self.update_monitoring_height()

    def update_monitoring_height(self):
        """
        Met à jour dynamiquement la hauteur de la zone de monitoring pour être limitée à la moitié de la hauteur de la fenêtre.
        """
        if self.parent() is not None and isinstance(self.parent(), QMainWindow):
            window_height = self.parent().height()

    def plot_limits(self):
        """
        Ajoute les bandes limites interpolées au graphique FFT en les regroupant par limit_name.
        Une seule entrée est ajoutée dans la légende pour chaque groupe. Préserve les courbes existantes (MaxHold).
        """
        if not self.configtab.limit_manager or not self.configtab.limit_manager.limits_by_name:
            print("Erreur : Aucun limite interpolée à afficher.")
            return

        # Vérifier si la légende existe, sinon la créer
        if self.fft_plot.plotItem.legend is None:
            self.fft_plot.plotItem.legend = pg.LegendItem(offset=(10, 10))  # Initialiser la légende
            self.fft_plot.plotItem.legend.setParentItem(self.fft_plot.plotItem)
        else:
            # On laisse les courbes existantes, comme MaxHold, intactes
            pass

        colors = ["r", "b", "g", "c", "m", "y"]  # Couleurs pour chaque groupe de limites
        color_index = 0

        for limit_name, limit_data in self.configtab.limit_manager.limits_by_name.items():
            if "interpolated_bands" not in limit_data:
                print(f"Aucune bande interpolée pour {limit_name}.")
                continue

            # Obtenir la couleur pour ce `limit_name`
            color = colors[color_index % len(colors)]
            color_index += 1

            # Ajouter une entrée dans la légende pour ce groupe
            legend_item = pg.PlotDataItem([], [], pen=pg.mkPen(color, width=2))
            self.fft_plot.addItem(legend_item)  # Ajouter l'item au graphique
            self.fft_plot.plotItem.legend.addItem(legend_item, limit_name)  # Ajouter MÊME groupe à la légende

            # Tracer chaque bande interpolée pour ce groupe
            for band in limit_data["interpolated_bands"]:
                band_frequencies = band["frequencies"]
                band_limits = band["limits"]

                self.fft_plot.plot(band_frequencies, band_limits, pen=pg.mkPen(color, width=2))

    def on_slider_change(self, value):
        """
        Méthode appelée lorsque le slider est déplacé.
        :param value: Position actuelle du slider (indice correspondant à un timestamp).
        """
        fft_streams = self.fft_streams.get_streams_by_group("Catenary Current")
        if fft_streams is None or len(fft_streams) == 0:
            print("Erreur : Aucun FFT_Stream trouvé dans le groupe 'Catenary Current'.")
            return

        fft_stream = next(iter(fft_streams))  # Récupère le premier FFT_Stream
        timestamps = fft_stream.timestamps
        hop = fft_stream.time_step  # Récupérer le pas temporel (`hop` en secondes)

        if value < 0 or value >= len(timestamps):
            print(f"Erreur : Valeur de slider ({value}) hors limites.")
            return

        # Calculer le timestamp correspondant
        position_in_seconds = hop * value  # Obtenir la position en secondes
        print(f"Position curseur slider : {position_in_seconds:.3f}s")

        # Déplacer les curseurs des plots monitoring
        for cursor in self.cursors:
            cursor.setPos(position_in_seconds)  # Met à jour la position des curseurs

        # Mettre à jour le graphique FFT
        self.plot_fft_stream()

    def update_slider(self, fft_stream):
        """
        Configure le slider pour utiliser les timestamps associés à un FFT_Stream.
        :param fft_stream: Instance de FFT_Stream.
        """
        self.timestamps = fft_stream.timestamps  # Enregistre les timestamps
        if self.timestamps is None or len(self.timestamps) == 0:
            print("Erreur : Aucun timestamp trouvé dans le FFT_Stream.")
            return

        self.slider.setRange(0, len(self.timestamps) - 1)  # Ajuster la plage du slider
        self.slider.setTickInterval(1)  # Chaque tick correspond à un timestamp

    def reset_visualization(self):
        """
        Réinitialise complètement tous les graphiques, curseurs, slider, et données FFT.
        """
        # Supprimer les plots de monitoring
        while self.monitoring_layout.count() > 0:  # Utiliser monitoring_layout au lieu de plots_layout
            plot_item = self.monitoring_layout.takeAt(0)
            if plot_item.widget():
                plot_item.widget().deleteLater()
        self.plot_widgets.clear()
        self.cursors.clear()

        # Réinitialiser le slider
        self.timestamps = None
        self.slider.setRange(0, 0)  # Réinitialise la plage
        self.slider.setValue(0)  # Remet le slider à 0

        # Supprimer les plots FFT
        self.fft_plot.clear()  # Supprimer la courbe FFT sur le graphique principal
        self.fft_plot.setTitle("")  # Réinitialiser le titre de la FFT
        if hasattr(self, 'fft_curve'):
            del self.fft_curve  # Supprimer la courbe FFT (si elle existe)
        if hasattr(self, 'maxhold_curve'):
            del self.maxhold_curve  # Supprimer la courbe de MaxHold (si elle existe)


class PsophometryTab(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.configtab :ConfigTab = None
        self.psopho_limit_line = None
        self.psopho_limit = None
        self.temporal_signals = self.parent().temporal_signals
        self.psophometric_signal = None
        self.fft_psopho_stream = None
        self.max_display_points = 1000
        self.color = "#00a78e"
        self.plot_widgets = []
        self.cursors = []  # curseurs synchronisés sur plots monitoring + psopho

        self.menuBar().hide()
        filler = QWidget(self)
        self.setCentralWidget(filler)

        # --- Dock Monitoring ---
        self.monitoring_dock = QDockWidget("Monitoring", self)
        self.monitoring_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.monitoring_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        mon_widget = QWidget()
        self.monitoring_layout = QVBoxLayout(mon_widget)
        self.monitoring_dock.setWidget(mon_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.monitoring_dock)

        # --- Dock Psophométrique ---
        self.psopho_dock = QDockWidget("Courant Psophométrique", self)
        self.psopho_dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.psopho_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        pso_widget = QWidget()
        self.psopho_layout = QVBoxLayout(pso_widget)
        self.psopho_dock.setWidget(pso_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.psopho_dock)
        self.splitDockWidget(self.monitoring_dock, self.psopho_dock, Qt.Vertical)

        # --- Dock FFT pondérée ---
        self.weighted_fft_dock = QDockWidget("FFT Pondérée Psophométrie", self)
        self.weighted_fft_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.weighted_fft_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        fft_widget = QWidget()
        self.weighted_fft_layout = QVBoxLayout(fft_widget)
        self.weighted_fft_dock.setWidget(fft_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.weighted_fft_dock)

        # --- Plots ---
        self.psopho_plot = pg.PlotWidget()
        self.psopho_plot.setBackground("w")
        self.psopho_plot.setLabel('bottom', 'Temps', units='s')
        self.psopho_plot.setLabel('left', 'Courant Psophométrique')
        self.psopho_plot.showGrid(x=True, y=True)
        self.psopho_layout.addWidget(self.psopho_plot)

        self.weighted_fft_plot = pg.PlotWidget()
        self.weighted_fft_plot.setBackground("w")
        self.weighted_fft_plot.setLabel('bottom', 'Fréquence', units='Hz')
        self.weighted_fft_plot.setLabel('left', 'Amplitude')
        self.weighted_fft_layout.addWidget(self.weighted_fft_plot)

        self.fft_slider = QSlider(Qt.Horizontal)
        self.fft_slider.valueChanged.connect(self._on_fft_slider_change)
        self.weighted_fft_layout.addWidget(self.fft_slider)

    def get_psopho_exceeding_zones(self):
        """
        Retourne une liste [(t1, t2), ...] des intervalles où le courant psophométrique dépasse la limite.
        """
        if self.psophometric_signal is None or self.psopho_limit is None:
            return []
        t = self.psophometric_signal.get_time_axis()
        d = self.psophometric_signal.data
        mask = d > self.psopho_limit
        if not np.any(mask):
            return []
        # Regroupe les zones consécutives
        from itertools import groupby
        from operator import itemgetter
        indices = np.where(mask)[0]
        zones = []
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            t1, t2 = t[group[0]], t[group[-1]]
            zones.append((t1, t2))
        return zones

    def plot_monitoring_signals(self):
        self.plot_widgets.clear()
        while self.monitoring_layout.count():
            w = self.monitoring_layout.takeAt(0).widget()
            if w: w.deleteLater()
        self.cursors.clear()
        self.psopho_cursor = None  # Reset to avoid NoneType bug

        # position initiale curseur = début psopho si dispo
        initial_cursor_pos = 0
        if self.psophometric_signal is not None:
            t = self.psophometric_signal.get_time_axis()
            if len(t):
                initial_cursor_pos = t[0]

        # Plots monitoring + curseurs
        for sig in self.temporal_signals.get_signals_by_group("Monitoring"):
            pw = pg.PlotWidget()
            pw.setBackground("w")
            pw.setLabel('bottom', 'Temps', units='s')
            pw.setLabel('left', sig.name)
            pw.showGrid(x=True, y=True)
            t = sig.get_time_axis()
            vb = pw.getViewBox()
            vb.setLimits(xMin=t[0], xMax=t[-1])
            data = sig.data
            step = max(1, len(data) // self.max_display_points)
            pw.plot(t[::step], data[::step], pen=pg.mkPen(color=self.color, width=2))
            pw.setXLink(self.psopho_plot)
            vb.sigXRangeChanged.connect(self._on_range_changed)

            # Curseur
            print(f"Adding cursor for signal: {sig.name} at initial position {initial_cursor_pos}")
            cursor = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen('#ff0000', width=2))
            pw.addItem(cursor)
            self.cursors.append(cursor)
            cursor.sigPositionChanged.connect(lambda _, c=cursor: self.sync_cursors(c))
            self.monitoring_layout.addWidget(pw)
            self.plot_widgets.append((pw, sig))

        # Curseur sur le plot psopho
        self.psopho_cursor = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen('#ff0000', width=2))
        self.psopho_plot.addItem(self.psopho_cursor)
        self.psopho_cursor.sigPositionChanged.connect(lambda: self.sync_cursors(self.psopho_cursor))
        self.cursors.append(self.psopho_cursor)

    def sync_cursors(self, moved_cursor):
        if moved_cursor is None:
            return
        pos = moved_cursor.pos().x()
        for cursor in self.cursors:
            if cursor is not moved_cursor:
                cursor.setPos(pos)

    def plot_psophometric_current(self):
        if not self.psophometric_signal:
            return
        self.psopho_plot.clear()
        t = self.psophometric_signal.get_time_axis()
        d = self.psophometric_signal.data
        step = max(1, len(d) // self.max_display_points)
        self.psopho_plot.plot(t[::step], d[::step], pen=pg.mkPen(color=self.color, width=2))
        # --- Ligne de limite ---
        if self.psopho_limit is not None:
            self.psopho_limit = float(self.psopho_limit)
            print(f"Plotting psophometric current with limit: {self.psopho_limit}")
        else:
            self.psopho_limit = float(self.configtab.psopho_limit_edit.text())
            # Supprimer l’ancienne limite si elle existe déjà
        if self.psopho_limit_line is not None:
            self.psopho_plot.removeItem(self.psopho_limit_line)
        # Ajouter la nouvelle ligne
        self.psopho_limit_line = pg.InfiniteLine(
            pos=self.psopho_limit, angle=0, movable=False,
            pen=pg.mkPen('#ff5050', width=2, style=Qt.DashLine)
        )
        self.psopho_plot.addItem(self.psopho_limit_line)

        self.plot_monitoring_signals()
        # --- Slider FFT ---
        if self.fft_psopho_stream:
            ts = self.fft_psopho_stream.timestamps
            self.fft_slider.setRange(0, len(ts)-1)
            self.fft_slider.setValue(0)
            self._on_fft_slider_change(0)
        if self.psopho_limit is not None:
            zones = self.get_psopho_exceeding_zones()
            for t1, t2 in zones:
                region = pg.LinearRegionItem([t1, t2], brush=pg.mkBrush(255, 0, 0, 70))  # Rouge clair transparent
                region.setZValue(-10)
                region.setMovable(False)
                self.psopho_plot.addItem(region)

    def _on_fft_slider_change(self, idx):
        if not self.fft_psopho_stream:
            return
        freqs = self.fft_psopho_stream.calculate_frequencies()
        mag = self.fft_psopho_stream.stream[idx]
        self.weighted_fft_plot.clear()
        self.weighted_fft_plot.plot(freqs, mag, pen=pg.mkPen(color=self.color, width=2))
        t = self.fft_psopho_stream.timestamps[idx]
        for line in self.cursors:
            line.setPos(t)

    def _on_range_changed(self, view_box, ranges):
        for pw, sig in self.plot_widgets:
            vb = pw.getViewBox()
            vr = vb.viewRange()[0]
            time = sig.get_time_axis()
            data = sig.data
            mask = (time >= vr[0]) & (time <= vr[1])
            sel_t = time[mask]
            sel_d = data[mask]
            step = max(1, len(sel_d) // self.max_display_points)
            # Create the curve only if it doesn't exist yet
            if not hasattr(pw, 'curve'):
                pw.curve = pw.plot(sel_t[::step], sel_d[::step], pen=pg.mkPen(self.color, width=2))
            else:
                pw.curve.setData(sel_t[::step], sel_d[::step])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.channels_config = pd.DataFrame(
            columns=["Signal d'origine", "Nouveau nom", "Unité", "Coeff", "Offset", "Post-traitement"])
        self.temporal_signals = Temporal_Signals()
        self.fft_streams = FFT_Streams()
        self.fft_signals = FFT_Signals()
        self.setWindowTitle("PyNIDAQmx V1.0")
        self.tabs = QTabWidget()

        self.tab_config = ConfigTab(main_window=self)
        self.tab_visu = VisuTab(self)
        self.tab_visu.configtab = self.tab_config
        self.tab_config.set_visu_tab(self.tab_visu)

        self.tab_psophometry = PsophometryTab(self)
        self.tab_psophometry.configtab = self.tab_config

        self.tabs.addTab(self.tab_config, "Configuration")
        self.tabs.addTab(self.tab_visu, "Visualization")
        self.tabs.addTab(self.tab_psophometry, "Psophometry")

        self.setCentralWidget(self.tabs)

    def resizeEvent(self, event):
        """
        Événement déclenché lors du redimensionnement de la fenêtre.
        """
        super().resizeEvent(event)
        if self.tab_visu is not None:
            self.tab_visu.update_monitoring_height()


def main():
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
