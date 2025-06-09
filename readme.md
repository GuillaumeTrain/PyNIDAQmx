PyNIDAQmx — Interface PyQt pour acquisition et affichage de signaux NI-DAQmx
Résumé
PyNIDAQmx est une interface graphique Python (PySide6 + PyqtGraph) pour afficher et sauvegarder des signaux acquis en temps réel via une carte National Instruments (NI-DAQmx, par exemple NI 9239).

Affichage temps réel multi-canaux

Export CSV structuré

Facteurs d’échelle/offsets appliqués par canal

Mode simulation si NI-DAQmx non présent

Personnalisation facile du "span" d’affichage façon oscilloscope

Installation
Prérequis
Python 3.8+

PySide6

pyqtgraph

numpy

nidaqmx (optionnel pour le mode simulation)

bash
Copier
Modifier
pip install PySide6 pyqtgraph numpy
pip install nidaqmx  # (si tu utilises du matériel NI)
Dépendances (fichier requirements.txt)
text
Copier
Modifier
PySide6
pyqtgraph
numpy
nidaqmx ; sys_platform == 'win32'
Utilisation
bash
Copier
Modifier
python main.py
L’application se lance :

Onglet Configuration :

Sélectionne le matériel NI disponible.

Active les canaux à mesurer, attribue nom/facteur/offset.

Définit le taux d’échantillonnage et la taille du buffer d’acquisition.

Onglet Acquisition :

Choisis le nom et le dossier du fichier CSV à enregistrer.

Clique sur "Start acquisition" pour commencer l’affichage temps réel (plots verts sur fond blanc).

Clique sur "Stop acquisition" pour arrêter.

Le span d’affichage (zoom horizontal) se règle façon oscilloscope.

Tab 3 : Réservé pour extension.

Fonctionnalités principales
Affichage temps réel de chaque canal activé (1 graphe/courbe par canal)

Facteurs de redimensionnement et offsets appliqués à chaque canal

Acquisition et bufferisation multi-threadées pour ne jamais perdre d’échantillon

Enregistrement CSV : time, puis 1 colonne par canal (noms configurables)

Mode simulation : génère des sinusoïdes si pas de matériel NI détecté

Personnalisation du "span" (base de temps) comme sur un oscilloscope

Structure du code
ConfigTab : gestion de la configuration matérielle et des canaux

AcquisitionTab : interface d’acquisition et d’affichage des courbes

AcquisitionThread : thread acquisition NI ou simulation

AggregationThread : agrégation des data pour sauvegarde

CsvWriterThread : enregistre les data agrégées en CSV

MainWindow : gestion globale, centralisation des threads et UI

Capture d’écran
(à compléter avec une image de l’interface, une fois lancée)

Exemple de CSV généré
csv
Copier
Modifier
time,Catenary Voltage,Pantograph Current,Speed,Torque
0.000000, 0.01, 0.02, 0.03, 0.04
0.000020, 0.11, 0.12, 0.13, 0.14
...
