import os

import numpy as np
import pandas as pd


class Temporal_Signal:
    def __init__(self, name: str = None, dt: float = None, length: int = None, data: np.ndarray = None, parent=None):
        """
        Initialise un signal temporel avec les paramètres spécifiés.

        :param name: Nom du signal.
        :param dt: Intervalle de temps entre chaque sample.
        :param length: Nombre de samples dans le signal.
        :param data: Tableau numpy contenant les données du signal.
        :param parent: Parent du signal (optionnel).
        """
        self._name: str = name
        self._dt: float = dt
        self._length: int = length
        self._data: np.ndarray = None  # Déclaré comme None par défaut

        # Vérifications et affectations
        if data is not None:
            self.data = data  # Utilise le setter pour valider la taille du tableau

    # Getter and Setter for name
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    # Getter and Setter for dt
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value: float):
        self._dt = value

    # Getter and Setter for length
    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError("La propriété 'length' doit être un entier positif")
        self._length = value

    # Getter and Setter for data
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("La propriété 'data' doit être de type numpy.ndarray")
        if self._length is not None and value.size != self._length:
            raise ValueError(
                f"La taille des données ({value.size}) doit correspondre à 'length' ({self._length})."
            )
        self._data = value

    def get_time_axis(self):
        return np.linspace(0 , self._length*self.dt, self.length)


class Temporal_Signals:
    def __init__(self, parent=None):
        # Dictionnaire pour stocker les signaux, organisés par groupes
        self.signals = {}

    def add_group(self,group_name):
        if group_name not in self.signals:
            self.signals[group_name] = {}

    def remove_group(self, group_name):
        if group_name in self.signals:
            self.signals.pop(group_name)  # Supprime le groupe directement

    def add_signal(self, signal: Temporal_Signal, signal_group_name: str):
        """
        Ajoute un signal au dictionnaire des signaux sous un groupe spécifique.

        :param signal: Instance de Temporal_Signal à ajouter.
        :param signal_group_name: Nom du groupe auquel le signal appartient.
        """
        signal_name = signal.name
        if signal_group_name not in self.signals:
            self.add_group(signal_group_name)
        if signal_name not in self.signals[signal_group_name]:
            self.signals[signal_group_name][signal_name] = signal

    def get_signals_by_group(self, signal_group_name: str):
        """
        Récupère tous les signaux appartenant à un groupe spécifique.

        :param signal_group_name: Nom du groupe.
        :return: Liste des signaux appartenant au groupe (ou None si le groupe n'existe pas).
        """
        return self.signals[signal_group_name].values()

    def remove_signal(self, signal: Temporal_Signal, signal_group_name: str):
        """
        Supprime un signal spécifique d'un groupe.

        :param signal: Instance de Temporal_Signal à supprimer.
        :param signal_group_name: Nom du groupe où se trouve le signal.
        """
        if signal_group_name in self.signals:
            if signal in self.signals[signal_group_name]:
                self.signals[signal_group_name].pop(signal)

    def list_groups(self):
        """
        Liste tous les groupes de signaux.

        :return: Liste des noms des groupes.
        """
        return list(self.signals.keys())

    def count_signals_in_group(self, signal_group_name: str):
        """
        Compte le nombre de signaux dans un groupe spécifique.

        :param signal_group_name: Nom du groupe.
        :return: Nombre de signaux dans le groupe, 0 si le groupe n'existe pas.
        """
        return len(self.signals.get(signal_group_name, []))

    def clear_signals(self):
        """
        Supprime tous les signaux dans tous les groupes pour libérer complètement la mémoire.
        Cette fonction évite de modifier le dictionnaire pendant l'itération.
        """
        # Itérer sur une copie des noms des groupes pour éviter les changements de taille du dictionnaire
        for group_name in list(self.signals.keys()):
            # Itérer sur une copie des noms des signaux pour éviter les modifications sur le dictionnaire
            for signal_name in list(self.signals[group_name].keys()):
                signal = self.signals[group_name].pop(signal_name)  # Supprime le signal du groupe
                del signal  # Supprime explicitement l'objet s'il n'est plus utilisé

            # Supprime le groupe une fois qu'il est vide
            self.remove_group(group_name)


class FFT_Signal:
    def __init__(self, name: str = None, freq_resolution: float = None, fft_size: int = None, data: np.ndarray = None, parent=None):
        """
        Initialise un signal FFT avec les paramètres spécifiés.

        :param name: Nom du signal FFT.
        :param freq_resolution: Intervalle de fréquence entre chaque bin (ex. Hz).
        :param fft_size: Nombre de bins dans la FFT.
        :param data: Tableau numpy contenant les coefficients FFT (complexes ou réels).
        :param parent: Parent du signal FFT (optionnel).
        """
        self._name: str = name
        self._freq_resolution: float = freq_resolution
        self._fft_size: int = fft_size
        self._data: np.ndarray = None  # Déclaré comme None par défaut

        # Vérifications et affectations
        if data is not None:
            self.data = data  # Utilise le setter pour valider la taille du tableau

    # Getter and Setter for name
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    # Getter and Setter for freq_resolution
    @property
    def freq_resolution(self):
        return self._freq_resolution

    @freq_resolution.setter
    def freq_resolution(self, value: float):
        if not isinstance(value, (float, int)) or value <= 0:
            raise ValueError("La propriété 'freq_resolution' doit être un nombre positif.")
        self._freq_resolution = value

    # Getter and Setter for fft_size
    @property
    def fft_size(self):
        return self._fft_size

    @fft_size.setter
    def fft_size(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("La propriété 'fft_size' doit être un entier positif.")
        self._fft_size = value

    # Getter and Setter for data
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("La propriété 'data' doit être de type numpy.ndarray")
        if self._fft_size is not None and value.size != self._fft_size:
            raise ValueError(
                f"La taille des données FFT ({value.size}) doit correspondre à 'fft_size' ({self._fft_size})."
            )
        self._data = value

    def get_frequency_axis(self):
        """
        Calcule l'axe des fréquences correspondant aux bins FFT.
        :return: Tableau numpy contenant la liste des fréquences (en Hz).
        """
        if self._freq_resolution is None or self._fft_size is None:
            raise ValueError("Les propriétés 'freq_resolution' et 'fft_size' doivent être définies.")
        return np.linspace(0, self._freq_resolution * (self._fft_size - 1), self._fft_size)


class FFT_Signals:
    def __init__(self, parent=None):
        # Dictionnaire pour stocker les FFTs, organisées par groupes
        self.signals = {}

    def add_group(self, group_name):
        """
        Ajoute un groupe dans le dictionnaire des FFTs.

        :param group_name: Nom du groupe à ajouter.
        """
        if group_name not in self.signals:
            self.signals[group_name] = {}

    def remove_group(self, group_name):
        """
        Supprime un groupe du dictionnaire des FFTs.

        :param group_name: Nom du groupe à supprimer.
        """
        if group_name in self.signals:
            self.signals.pop(group_name)

    def add_fft(self, fft_signal: FFT_Signal, fft_group_name: str):
        """
        Ajoute une FFT au dictionnaire des FFTs sous un groupe spécifique.

        :param fft_signal: Instance de FFT_Signal à ajouter.
        :param fft_group_name: Nom du groupe auquel la FFT appartient.
        """
        if fft_group_name not in self.signals:
            self.add_group(fft_group_name)
        fft_name = fft_signal.name
        if fft_name not in self.signals[fft_group_name]:
            self.signals[fft_group_name][fft_name] = fft_signal

    def get_ffts_by_group(self, fft_group_name: str):
        """
        Récupère toutes les FFTs appartenant à un groupe spécifique.

        :param fft_group_name: Nom du groupe.
        :return: Liste des FFTs appartenant au groupe (ou None si le groupe n'existe pas).
        """
        return self.signals.get(fft_group_name, {}).values()

    def remove_fft(self, fft_signal: FFT_Signal, fft_group_name: str):
        """
        Supprime une FFT spécifique d'un groupe.

        :param fft_signal: Instance de FFT_Signal à supprimer.
        :param fft_group_name: Nom du groupe où se trouve la FFT.
        """
        if fft_group_name in self.signals:
            if fft_signal.name in self.signals[fft_group_name]:
                self.signals[fft_group_name].pop(fft_signal.name)

    def list_groups(self):
        """
        Liste tous les groupes de FFTs.

        :return: Liste des noms des groupes.
        """
        return list(self.signals.keys())

    def count_ffts_in_group(self, fft_group_name: str):
        """
        Compte le nombre de FFTs dans un groupe spécifique.

        :param fft_group_name: Nom du groupe.
        :return: Nombre de FFTs dans le groupe, 0 si le groupe n'existe pas.
        """
        return len(self.signals.get(fft_group_name, {}))

    def clear_signals(self):
        """
        Supprime toutes les FFTs dans tous les groupes pour libérer complètement la mémoire.
        Cette fonction évite de modifier le dictionnaire pendant l'itération.
        """
        for group_name in list(self.signals.keys()):
            for fft_name in list(self.signals[group_name].keys()):
                fft_signal = self.signals[group_name].pop(fft_name)  # Supprime la FFT du groupe
                del fft_signal  # Supprime explicitement l'objet FFT_Signal
            self.remove_group(group_name)


class FFT_Stream:
    def __init__(self, name: str, fft_size: int, freq_resolution: float, time_step: float):
        """
        Initialise une structure FFT_Stream pour gérer une FFT évolutive dans le temps.

        :param name: Nom du stream FFT.
        :param fft_size: Taille de chaque FFT (nombre de bins).
        :param freq_resolution: Résolution fréquentielle (en Hz par bin).
        :param time_step: Temps entre deux calculs de FFT (en secondes).
        """
        if not isinstance(fft_size, int) or fft_size <= 0:
            raise ValueError("La taille de la FFT doit être un entier positif.")
        if not isinstance(freq_resolution, (float, int)) or freq_resolution <= 0:
            raise ValueError("La résolution fréquentielle doit être un nombre positif.")
        if not isinstance(time_step, (float, int)) or time_step <= 0:
            raise ValueError("Le pas temporel doit être un nombre positif.")

        self._name: str = name
        self._fft_size: int = fft_size
        self._freq_resolution: float = freq_resolution
        self._time_step: float = time_step
        self._stream: list[np.ndarray] = []  # Liste pour stocker les FFTs glissantes
        self._timestamps: list[float] = []
        self._maxhold: np.ndarray = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def fft_size(self):
        return self._fft_size

    @property
    def freq_resolution(self):
        return self._freq_resolution

    @property
    def time_step(self):
        return self._time_step

    @property
    def stream(self):
        return np.array(self._stream)  # Retourne les FFTs sous forme de tableau numpy

    @property
    def timestamps(self):
        return np.array(self._timestamps)  # Retourne les timestamps sous forme de tableau numpy

    def add_fft(self, fft_data: np.ndarray, timestamp: float = None):
        """
        Ajoute une FFT au stream en spécifiant les données FFT et l'instant associé.

        :param fft_data: Tableau numpy contenant une FFT (de taille `fft_size`).
        :param timestamp: Temps en secondes associé à cette FFT (facultatif ; calculé automatiquement si None).
        """
        if not isinstance(fft_data, np.ndarray):
            raise ValueError("Les données FFT doivent être un tableau numpy.ndarray.")
        if fft_data.size != self._fft_size:
            raise ValueError(f"La taille de la FFT doit être {self._fft_size}.")

        # Si aucun timestamp n'est fourni, calculer automatiquement
        if timestamp is None:
            timestamp = self._time_step * len(self._timestamps)

        self._stream.append(fft_data)
        self._timestamps.append(timestamp)

    def calculate_maxhold(self):
        """
        Calcule la FFT MaxHold (maximum des magnitudes FFT sur tous les timestamps).
        """
        if len(self._stream) == 0:
            raise ValueError("Le stream ne contient aucune FFT.")

        maxhold_fft = np.max(self._stream, axis=0)  # Calcul du maximum pour chaque bin de fréquence
        self._maxhold = FFT_Signal(
            name=f"{self._name}_MaxHold",
            freq_resolution=self._freq_resolution,
            fft_size=self._fft_size,
            data=maxhold_fft,
        )
        print(f"MaxHold calculée pour le stream '{self._name}'.")

    def get_maxhold(self):
        return self._maxhold

    def get_magnitude(self, timestamp: float):
        """
        Récupère les amplitudes spectrales (magnitude FFT) correspondant au timestamp spécifié.

        :param timestamp: Temps pour lequel récupérer les amplitudes FFT.
        :return: np.ndarray représentant les amplitudes spectrales FFT (magnitude).
        """
        if not isinstance(timestamp, (float, int)):
            raise ValueError("Le timestamp doit être un nombre.")

        # Vérifier si le timestamp existe dans la liste
        if timestamp not in self._timestamps:
            raise ValueError(f"Le timestamp '{timestamp}' n'existe pas dans le stream.")

        # Trouver l'index du timestamp
        index = self._timestamps.index(timestamp)

        # Retourner les amplitudes FFT associées
        return self._stream[index]

    def calculate_frequencies(self):
        """
        Calcule l'axe des fréquences correspondant aux bins FFT.

        :return: Tableau numpy contenant les fréquences.
        """
        if self._freq_resolution is None or self._fft_size is None:
            raise ValueError("Les propriétés 'freq_resolution' et 'fft_size' doivent être définies.")
        return np.linspace(0, self._freq_resolution * (self._fft_size - 1), self._fft_size)

    def clear_stream(self):
        """
        Réinitialise le stream en supprimant toutes les FFTs et timestamps associés.
        """
        self._stream = []
        self._timestamps = []


class FFT_Streams:
    def __init__(self, parent=None):
        """
        Initialise une structure pour gérer plusieurs groupes de streams FFT.

        :param parent: Parent optionnel.
        """
        self.streams = {}  # Dictionnaire pour stocker les streams FFT par groupes

    def add_group(self, group_name: str):
        """
        Ajoute un nouveau groupe de streams FFT.

        :param group_name: Nom du groupe.
        """
        if group_name not in self.streams:
            self.streams[group_name] = {}

    def remove_group(self, group_name: str):
        """
        Supprime un groupe de streams FFT.

        :param group_name: Nom du groupe à supprimer.
        """
        if group_name in self.streams:
            self.streams.pop(group_name)

    def add_stream(self, stream: FFT_Stream, group_name: str):
        """
        Ajoute un stream FFT à un groupe spécifique.

        :param stream: Instance de FFT_Stream.
        :param group_name: Nom du groupe auquel le stream appartient.
        """
        if group_name not in self.streams:
            self.add_group(group_name)
        stream_name = stream.name
        if stream_name not in self.streams[group_name]:
            self.streams[group_name][stream_name] = stream

    def get_streams_by_group(self, group_name: str):
        """
        Récupère tous les streams FFT appartenant à un groupe spécifique.

        :param group_name: Nom du groupe.
        :return: Liste des streams FFT (ou None si le groupe n'existe pas).
        """
        return self.streams.get(group_name, {}).values()

    def remove_stream(self, stream: FFT_Stream, group_name: str):
        """
        Supprime un stream spécifique d'un groupe.

        :param stream: Instance de FFT_Stream à supprimer.
        :param group_name: Nom du groupe où se trouve le stream.
        """
        if group_name in self.streams:
            if stream.name in self.streams[group_name]:
                self.streams[group_name].pop(stream.name)

    def list_groups(self):
        """
        Liste tous les groupes existants.

        :return: Liste des noms des groupes.
        """
        return list(self.streams.keys())

    def count_streams_in_group(self, group_name: str):
        """
        Compte le nombre de streams FFT dans un groupe spécifique.

        :param group_name: Nom du groupe.
        :return: Nombre de streams dans le groupe, 0 si le groupe n'existe pas.
        """
        return len(self.streams.get(group_name, {}))

    def clear_streams(self):
        """
        Supprime tous les streams dans tous les groupes pour libérer la mémoire.
        """
        for group_name in list(self.streams.keys()):
            for stream_name in list(self.streams[group_name].keys()):
                stream = self.streams[group_name].pop(stream_name)
                del stream
            self.remove_group(group_name)


class LimitManager:
    def __init__(self):
        self.limits_by_name = {}  # Stocke les limites groupées par `limit_name`
        self.limit_max_frequency = None #Stoque la fréquence minimum pour l'ensemble des bandes limites
        self.limit_min_frequency = None #Stoque la fréquence maximum pour l'ensemble des bandes limites

    def load_limits_from_folder(self, folder_path):
        """
        Charge les limites FFT depuis un dossier contenant des fichiers CSV.
        :param folder_path: Chemin vers le dossier contenant les fichiers limites FFT.
        """
        try:
            # Récupérer tous les fichiers CSV du dossier
            limits_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            if not limits_files:
                print("Aucun fichier limite trouvé dans le dossier sélectionné.")
                return


            self.clear_limits()

            for file in limits_files:
                file_path = os.path.join(folder_path, file)
                result = self.load_limit_file(file_path)  # Charger un fichier unique
                if result:
                    limit_name = result["limit_name"]
                    if self.limit_max_frequency is None:
                        self.limit_max_frequency = result["limit_max_frequency"]
                    elif self.limit_max_frequency < result["limit_max_frequency"]:
                        self.limit_max_frequency = result["limit_max_frequency"]
                    if self.limit_min_frequency is None:
                        self.limit_min_frequency = result["limit_min_frequency"]
                    elif self.limit_min_frequency > result["limit_min_frequency"]:
                        self.limit_min_frequency = result["limit_min_frequency"]
                    self.limits_by_name[limit_name] = result  # Ajouter au dictionnaire

            print(f"min frequency = {self.limit_min_frequency} max frequency = {self.limit_max_frequency}")
            print(str(self.limits_by_name))
        except FileNotFoundError:
            print(f"Erreur : Impossible de trouver le dossier ou les fichiers dans le chemin {folder_path}.")
        except ValueError as ve:
             print(f"Erreur dans le traitement de {file_path} : {ve}.")
        except Exception as e:
            print(f"Erreur générale lors du chargement des fichiers limites : {e}.")

    def load_limit_file(self, file_path):
        """
        Charge un fichier limite contenant les bandes de fréquences FFT.
        :param file_path: Chemin du fichier CSV.
        :return: Dictionnaire contenant le type d'interpolation, le nom de la limite et la table des bandes.
        """
        try:
            interpolation = "log"  # Par défaut : logarithmique
            limit_name = None
            limit_table_start = None  # Ligne de début de la table limite

            # Lire le fichier ligne par ligne
            lines = []
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip()  # Retirer les espaces et les retours à la ligne

                # Si on détecte #limit_name, chercher la prochaine valeur non vide
                if line.startswith("#limit_name"):
                    for next_line in lines[i + 1:]:
                        next_line = next_line.strip()
                        if next_line:  # Si la ligne suivante n'est pas vide
                            limit_name = next_line
                            print(f"Nom de limite détecté : {limit_name}")
                            break

                # Si on détecte #interpolation, chercher la prochaine valeur non vide
                elif line.startswith("#interpolation"):
                    print("Interpolation détectée.")
                    for next_line in lines[i + 1:]:
                        next_line = next_line.strip()
                        if next_line:  # Si la ligne suivante n'est pas vide
                            interpolation = next_line.lower()  # Convertir en minuscule pour garantir homogénéité
                            print(f"Interpolation définie : {interpolation}")
                            break

                # Identifier le début de la table de limites
                elif line.startswith("#limit_table"):
                    limit_table_start = i + 1  # La table commence à partir de cette ligne
                    print(f"Début de la table des limites : Ligne {limit_table_start}")



            if limit_table_start is None:
                raise ValueError("Le fichier limite ne contient pas de table identifiable.")



            # Si aucun `limit_name`, dériver du fichier
            if limit_name is None:
                limit_name = os.path.basename(file_path).split(".")[0]

            # Charger la table limite avec pandas
            limit_df = pd.read_csv(file_path, sep=";", skiprows=limit_table_start, comment="#")

            required_columns = ['Fmin', 'Fmax', 'Fmin_Limit', 'Fmax_Limit']
            if not all(col in limit_df.columns for col in required_columns):
                raise ValueError(f"Colonnes manquantes dans {file_path}. Attendu : {required_columns}")

            # Calculer les fréquences min/max
            limit_min_freq = limit_df["Fmin"].min()
            limit_max_freq = limit_df["Fmax"].max()
            print( f"limite trouvée {limit_name} : {limit_min_freq} : {limit_max_freq} : {interpolation}")
            return {
                "interpolation": interpolation,
                "limit_name": limit_name,
                "limits": limit_df,
                "limit_min_frequency": limit_min_freq,
                "limit_max_frequency": limit_max_freq,
            }
        except Exception as e:
            print(f"Erreur lors du chargement du fichier limite : {file_path}. Exception : {e}")
            return None

    def interpolate_limits(self, frequencies=None, fmin=None, fmax=None, df=None):
        """
        Interpole uniquement les bandes FFT en fonction de l'axe des fréquences du FFT Stream,
        et stocke les bandes interpolées dans self.limits_by_name.

        :param frequencies: Axe des fréquences (np.linspace, par exemple).
        :param fmin: Fréquence minimale de l'axe des fréquences.
        :param fmax: Fréquence maximale de l'axe des fréquences.
        :param df: Intervalle entre les fréquences (résolution entre points).
        :param interpolation: Type d'interpolation ('log' ou 'lin').
        """
        if frequencies is None:
            if fmin is None or fmax is None or df is None:
                raise ValueError("Si 'frequencies' est absent, 'fmin', 'fmax', et 'df' doivent être fournis.")
            else:
                frequencies = np.linspace(fmin, fmax, (fmax - fmin + 1) // df)  # Génération automatique des fréquences

        try:
            # Parcourir chaque limite définie dans self.limits_by_name
            for limit_name, limit_data in self.limits_by_name.items():
                limit_df = limit_data["limits"]  # Récupérer le DataFrame des bandes
                interpolation = limit_data["interpolation"]
                interpolated_bands = []  # Liste pour stocker les bandes interpolées pour ce limit_name

                print(f"limit name {limit_name} Interpolation : {interpolation}")

                # Parcourir les bandes définies dans `limit_df`
                for _, row in limit_df.iterrows():
                    band_fmin = row["Fmin"]
                    band_fmax = row["Fmax"]
                    band_fmin_limit = row["Fmin_Limit"]
                    band_fmax_limit = row["Fmax_Limit"]

                    # Générer les fréquences pour cette bande
                    band_frequencies = frequencies[(frequencies >= band_fmin) & (frequencies <= band_fmax)]
                    if band_frequencies.size > 0:
                        # Interpoler les limites pour cette bande
                        if interpolation == "log":
                            print("limit interpoled logaritmicaly")
                            band_limits = np.logspace(
                                np.log10(band_fmin_limit), np.log10(band_fmax_limit), band_frequencies.size
                            )
                        elif interpolation == "lin":
                            print("limit interpoled lineary")
                            band_limits = np.linspace(
                                band_fmin_limit, band_fmax_limit, band_frequencies.size
                            )

                        # Ajouter la bande interpolée sous forme de tuple (frequencies, limits)
                        interpolated_bands.append({"frequencies": band_frequencies, "limits": band_limits})

                # Stocker les bandes interpolées dans self.limits_by_name
                limit_data["interpolated_bands"] = interpolated_bands

        except Exception as e:
            print(f"Erreur lors de l'interpolation des limites. Exception : {e}")

    def clear_limits(self):
        """
        Supprime toutes les limites FFT et réinitialise les fréquences min/max.
        """
        self.limits_by_name.clear()
        self.limit_min_frequency = None
        self.limit_max_frequency = None
        print("Toutes les limites FFT ont été réinitialisées.")

    def show_limits_summary(self):
        """
        Affiche un résumé des limites FFT chargées, incluant leur interpolation et les fréquences min/max.
        """
        if not self.limits_by_name:
            print("Aucune limite chargée.")
            return

        print("Résumé des limites FFT chargées :")
        for limit_name, limit_data in self.limits_by_name.items():
            print(f"Limite : {limit_name}")
            print(f"  Interpolation : {limit_data['interpolation']}")
            print(f"  Fréquence min : {limit_data['limit_min_freq']} Hz")
            print(f"  Fréquence max : {limit_data['limit_max_freq']} Hz")

class Exceedings_FFT:
    def __init__(self):
        """
        Initialise une structure pour stocker les dépassements FFT par limite et gérer les couleurs associées.
        """
        self.exceedings = {}  # Dictionnaire {limite_name: [(timestamp, [frequencies])]}
        self.limit_colors = [  # Liste de couleurs prédéfinies
            (255, 0, 0, 50),  # Rouge faiblement opaque
            (255, 255, 0, 50),  # Jaune faiblement opaque
            (0, 255, 255, 50),  # Cyan faiblement opaque
            (255, 0, 255, 50)  # Magenta faiblement opaque
        ]
        self.assigned_colors = {}  # Dictionnaire {limite_name: couleur}

    def add_exceeding(self, limit_name: str, timestamp: float, exceeding_frequencies: list):
        """
        Ajoute un dépassement pour une limite spécifique.
        :param limit_name: Nom de la limite.
        :param timestamp: Timestamp du dépassement.
        :param exceeding_frequencies: Liste des fréquences ayant dépassé la limite.
        """
        if limit_name not in self.exceedings:
            self.exceedings[limit_name] = []
        self.exceedings[limit_name].append((timestamp, exceeding_frequencies))

    def get_exceedings_by_limit(self, limit_name: str):
        """
        Récupère tous les dépassements pour une limite spécifique.
        :param limit_name: Nom de la limite.
        :return: Liste des dépassements [(timestamp, [frequencies])] pour la limite.
        """
        return self.exceedings.get(limit_name, [])

    def get_all_exceedings(self):
        """
        Retourne toutes les limites et leurs dépassements.
        :return: Dictionnaire complet des dépassements.
        """
        return self.exceedings

    def clear_exceedings(self):
        """
        Réinitialise tous les dépassements.
        """
        self.exceedings.clear()
        self.assigned_colors.clear()
        print("Tous les dépassements ont été réinitialisés.")

    def get_color(self, limit_name: str):
        """
        Retourne la couleur associée à une limite. Si aucune couleur n'est assignée, attribue une couleur.
        :param limit_name: Nom de la limite.
        :return: Code couleur RGBA associé à la limite.
        """
        if limit_name in self.assigned_colors:
            return self.assigned_colors[limit_name]

        # Assigner une couleur unique parmi les couleurs disponibles
        next_color_index = len(self.assigned_colors) % len(self.limit_colors)
        color = self.limit_colors[next_color_index]
        self.assigned_colors[limit_name] = color
        return color