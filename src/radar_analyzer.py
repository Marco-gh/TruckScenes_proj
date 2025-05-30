from truckscenes import TruckScenes
import numpy as np
from pypcd4 import PointCloud  # github.com/MapIV/pypcd4
from box import Box
import os
import truck_utility as tru
from time import time

# Per dividere i dati in train e test e classificarli tramite un classificatore random forest
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.html
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels

# Visualizzazione risultati, avanzamento progressi
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def train_radar_classifier(trucksc, first_sample_token, dir_path, test_size_par: float):
    """
    Allena il classificatore Random Forest su una piccola parte dei frame della scena
    e restituisce lo stimatore allenato, pronto per essere riusato.
    """
    start_time = time()

    label_radar = [
        'RADAR_RIGHT_BACK',
        'RADAR_RIGHT_SIDE',
        'RADAR_RIGHT_FRONT',
        'RADAR_LEFT_FRONT',
        'RADAR_LEFT_SIDE',
        'RADAR_LEFT_BACK'
    ]

    # Precarica tutti i token dei frame della scena per usare tqdm
    # DA SPOSTARE NEL MAIN QUESTO PRECARICAMENTO?????
    # Preleva tutti i frame
    all_tokens = []
    current_token = first_sample_token
    while current_token:
        all_tokens.append(current_token)
        current_token = trucksc.get('sample', current_token)['next']

    # Dividiamo in "frame" di allenamento e di test (qui useremo solo quelli di training)
    train_tokens, _ = train_test_split(all_tokens, test_size=test_size_par, random_state=42)
    print(f"Frame usati per il training: {len(train_tokens)}")

    # Ciclo su tutti i frame con barra di avanzamento tqdm
    train_boxes = []

    for token in tqdm(train_tokens, desc="Training: elaborazione frame"):
        my_sample = trucksc.get('sample', token)

        # Lista di array con i dati da tutti i sensori di una campionamento della scena
        radar_points = []

        for lab in label_radar:
            radar_data = trucksc.get('sample_data', my_sample['data'][lab])
            full_path_data = os.path.join(dir_path, radar_data.get("filename"))

            pc: PointCloud = PointCloud.from_path(full_path_data)
            array: np.ndarray = pc.numpy()
            if array.size == 0:
                continue  # Salta sensore senza dati

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

        if len(radar_points) == 0:
            continue

        # Array con tutti i dati provenienti dai radar
        radar_unified_array = np.concatenate(radar_points, axis=0)
        # Punti visitati o meno
        point_mask = np.full(radar_unified_array.shape[0], False)

        # Vediamo ora se i punti sono in una box
        for ann_token in my_sample['anns']:
            ann_json = trucksc.get('sample_annotation', ann_token)
            box = Box(ann_json['translation'], ann_json['size'],
                      ann_json['rotation'], ann_json['category_name'])

            for indx, p in enumerate(radar_unified_array):
                if not point_mask[indx] and box.contains(p[:3]):
                    point_mask[indx] = True
                    box.add_point(p)

            # Per evitare di aggiungere box vuote:
            if box.get_num_point() > 0:
                train_boxes.append(box)

    #X = [] Array delle features associate a ogni box
    #Y = [] Array delle etichette associate a ogni box 
    X_train = [b.get_features_arr() for b in train_boxes]
    y_train = [b.get_box_label() for b in train_boxes]

    assert len(X_train) == len(y_train)

    # numero alberi, lunghezza massima alberi, stato per avere riproducibilità
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)  # fit() ritorna uno stimatore adattato ai nostri dati di training

    end_time = time()
    print(f"Training completato in {end_time - start_time:.1f} secondi")

    return clf


def radar_scene_analysis(trucksc, first_sample_token, dir_path, clf , test_size_par: float):
    start_time = time()

    label_radar = [
        'RADAR_RIGHT_BACK',
        'RADAR_RIGHT_SIDE',
        'RADAR_RIGHT_FRONT',
        'RADAR_LEFT_FRONT',
        'RADAR_LEFT_SIDE',
        'RADAR_LEFT_BACK'
    ]

    # Precarica tutti i token dei frame della scena per usare tqdm
    # DA SPOSTARE NEL MAIN QUESTO PRECARICAMENTO?????
    # Preleva tutti i frame
    all_tokens = []
    current_token = first_sample_token
    while current_token:
        all_tokens.append(current_token)
        current_token = trucksc.get('sample', current_token)['next']
    # Dividiamo in "frame" di allenamento e di test
    _, test_tokens = train_test_split(all_tokens, test_size=test_size_par, random_state=42)
    print(f"Frame di test: {len(test_tokens)}")

    # Ciclo su tutti i frame con barra di avanzamento tqdm
    test_boxes = []

    for token in tqdm(test_tokens, desc="Test: elaborazione frame"):
        my_sample = trucksc.get('sample', token)

        # Lista di array con i dati da tutti i sensori di una campionamento della scena
        radar_points = []

        for lab in label_radar:
            radar_data = trucksc.get('sample_data', my_sample['data'][lab])
            full_path_data = os.path.join(dir_path, radar_data.get("filename"))

            pc: PointCloud = PointCloud.from_path(full_path_data)
            array: np.ndarray = pc.numpy()
            if array.size == 0:
                continue

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

        if len(radar_points) == 0:
            continue

        # Array con tutti i dati provenienti dai radar
        radar_unified_array = np.concatenate(radar_points, axis=0)
        # Punti visitati o meno
        point_mask = np.full(radar_unified_array.shape[0], False)

        # Vediamo ora se i punti sono in una box
        for ann_token in my_sample['anns']:
            ann_json = trucksc.get('sample_annotation', ann_token)
            box = Box(ann_json['translation'], ann_json['size'],
                      ann_json['rotation'], ann_json['category_name'])

            for indx, p in enumerate(radar_unified_array):
                if not point_mask[indx] and box.contains(p[:3]):
                    point_mask[indx] = True
                    box.add_point(p)

            # Per evitare di aggiungere box vuote:
            if box.get_num_point() > 0:
                test_boxes.append(box)

    #X = [] Array delle features associate a ogni box
    #Y = [] Array delle etichette associate a ogni box 
    X_test = [b.get_features_arr() for b in test_boxes]
    y_test = [b.get_box_label() for b in test_boxes]
    assert len(X_test) == len(y_test)

    # Finita la fase di raccolta inizio fase di predizione
    y_pred = clf.predict(X_test)
    # Per ottenere le etichette presenti nei dati
    label_names = sorted(list(unique_labels(y_test, y_pred)))

    # FINE
    end_time = time()
    ex_time = end_time - start_time

    return X_test, y_test, y_pred, ex_time, label_names
