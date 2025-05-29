from truckscenes import TruckScenes
import numpy as np
from pypcd4 import PointCloud # github.com/MapIV/pypcd4
from box import Box
import os
import truck_utility as tru
from time import time

# Per dividere i dati in train e test e classificarli tramite un classificatore random forest
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.html
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Visualizzazione risultati, avanzamento progressi
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

start_time = time()

dir_path = "/home/marco/Documents/TAV project/dataset/man-truckscenes/"
trucksc = TruckScenes('v1.0-mini', dir_path, verbose=True)

first_sample_token = trucksc.scene[1]['first_sample_token']
my_sample = trucksc.get('sample', first_sample_token)

label_radar = [
    'RADAR_RIGHT_BACK',
    'RADAR_RIGHT_SIDE',
    'RADAR_RIGHT_FRONT',
    'RADAR_LEFT_FRONT',
    'RADAR_LEFT_SIDE',
    'RADAR_LEFT_BACK'
]

all_boxes = []

while True:
    # Per evitare di saltare l'ultimo frame
    if my_sample['next']=="":
        break

    # Lista di array con i dati da tutti i sensori di una campionamento della scena
    radar_points = []

    for lab in label_radar:
        radar_data = trucksc.get('sample_data', my_sample['data'][lab])
        full_path_data = os.path.join(dir_path, radar_data.get("filename"))

        pc: PointCloud = PointCloud.from_path(full_path_data)
        array: np.ndarray = pc.numpy()

        # Otteniamo i parametri per trasformare i punti dal sistema di riferimento locale (camion)
        # a quello inerziale (mondo)
        calib = trucksc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
        ego = trucksc.get('ego_pose', radar_data['ego_pose_token'])
        calib_t = np.array(calib['translation'])
        calib_r = np.array(calib['rotation'])
        ego_t = np.array(ego['translation'])
        ego_r = np.array(ego['rotation'])
        transformed_points = np.array([
            tru.transform_point_to_world(p[:7],
                                    calib_t, calib_r,
                                    ego_t, ego_r)
            for p in array
        ])

        radar_points.append(transformed_points)

    # Array con tutti i dati provenienti dai radar
    radar_unified_array = np.concatenate(radar_points, axis=0)
    # Punti visitati o meno
    labels = np.full(radar_unified_array.shape[0], False)

    # Vediamo ora se i punti sono in una box
    for ann_token in my_sample['anns']:
        ann_json = trucksc.get('sample_annotation',ann_token)
        box = Box(ann_json['translation'],ann_json['size'],ann_json['rotation'], ann_json['category_name'])
        
        for indx, p in enumerate(radar_unified_array):
            if not labels[indx] and box.contains(p[:3]):
                labels[indx] = True
                box.add_point(p)
        
        # Per evitare di aggiungere box vuote:
        if box.get_num_point() > 0:
            all_boxes.append(box)


    my_sample = trucksc.get('sample', my_sample['next'])
    # FINE RACCOLTA DATI

# Assegnazione dell'etichette e classificazione mediante random forest
X = [] # Array delle features associate a ogni box
Y = [] # Array delle etichette associate a ogni box 
for b in all_boxes:
    X.append(b.get_features_arr())
    Y.append(b.get_box_label())
assert(len(X)==len(Y)) # Controllo se entrambi i vettori hanno la stessa lunghezza

# ANALISI DATI
# Divisione in feature ed etichette per training e test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# numero alberi, lunghezza massima alberi, stato per avere riproducibilità
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train) # fit() ritorna uno stimatore adattato ai nostri dati di training

# Finita la fase di trainig inizio fase di predizione
y_pred = clf.predict(X_test)

# Accuratezza: percentuale di predizioni corrette
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza: {accuracy:.2f}")
# Matrice di confusione
print("Matrice di confusione:")
print(confusion_matrix(y_test, y_pred))

# FINE
end_time = time()
print("TEMPO DI ESECUZIONE: "+str(end_time - start_time))

# Grafici
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['statico', 'veicolo'], yticklabels=['statico', 'veicolo'])
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.title("Matrice di Confusione")
plt.show()