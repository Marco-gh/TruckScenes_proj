from time import time
from typing import List, Tuple
import os

import numpy as np
from tqdm import tqdm
from pypcd4 import PointCloud  # github.com/MapIV/pypcd4
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels

import truck_utility as tru
from box import Box

from scipy.spatial import cKDTree

from itertools import chain

from scipy.spatial import cKDTree

# -----------------------------------------------------------------------------
# Configurazioni
# -----------------------------------------------------------------------------
RADAR_LABELS = [
    'RADAR_RIGHT_BACK',
    'RADAR_RIGHT_SIDE',
    'RADAR_RIGHT_FRONT',
    'RADAR_LEFT_FRONT',
    'RADAR_LEFT_SIDE',
    'RADAR_LEFT_BACK'
]

LIDAR_LABELS = [
    'LIDAR_LEFT',
    'LIDAR_RIGHT',
    #'LIDAR_TOP_FRONT',
    #'LIDAR_TOP_LEFT',
    #'LIDAR_TOP_RIGHT',
    #'LIDAR_REAR'
]

DEFAULT_SENSOR_TYPE = 'lidar'

# Numero di colonne da considerare per ogni punto:
#   - Radar: (x, y, z, vx, vy, vz, rcs) => slice_len = 7
#   - Lidar: (x, y, z)                  => slice_len = 3
SENSOR_META = {
    'radar': {
        'labels': RADAR_LABELS,
        'slice_len': 7
    },
    'lidar': {
        'labels': LIDAR_LABELS,
        'slice_len': 3
    }
}

# -----------------------------------------------------------------------------
# Metodi private (helper)
# -----------------------------------------------------------------------------

def _collect_sample_tokens(trucksc, first_token: str) -> List[str]:
    """Raccoglie tutti i token dei sample della scena partendo dal primo."""
    tokens = []
    current = first_token
    while current:
        tokens.append(current)
        current = trucksc.get('sample', current)['next']
    return tokens


def _load_sensor_points(trucksc, sample, dir_path: str, sensor_labels: List[str], slice_len: int) -> List[np.ndarray]:
    """
    Carica e trasforma i punti (radar o lidar) da locale a mondo per ogni sensore nella lista.
    - Usa trasformazione batch per LIDAR.
    - Usa trasformazione punto-per-punto per RADAR.
    """
    sensor_points = []

    for lab in sensor_labels:
        data = trucksc.get('sample_data', sample['data'][lab])
        full_path = os.path.join(dir_path, data['filename'])

        pc = PointCloud.from_path(full_path)
        arr = pc.numpy()

        # In caso di Lidar sottocampioniamo (per evitare di far esplodere la ram senza perdere troppa informazione)
        if slice_len == 3:
            
            # Filtriamo i punti prima di campionarli
            #tree = cKDTree(arr) # Creazione di un kdtree
            #counts = tree.query_ball_point(arr, r=0.25, return_length=True) # Filtriamo i punti con almeno 2 vicini in un raggio di 0.25
            #mask = counts > 2 # Creazione maschera booleana
            #arr[mask] # Applicazione maschera
            
            #print(f"Numero punti lidar da sottocampionare: {len(arr)}")
            if arr.shape[0] > 10000:
                idx = np.random.choice(arr.shape[0], 5000, replace=False)  # Prendiamo 5000 punti invece dei ~80000 del lidar
                arr = arr[idx]
                #print(f"Numero punti lidar sottocampionato: {len(arr)}")
        
        if arr.size == 0:
            continue

        # Parametri di calibrazione e posa del veicolo
        calib = trucksc.get('calibrated_sensor', data['calibrated_sensor_token'])
        ego   = trucksc.get('ego_pose', data['ego_pose_token'])
        calib_t = np.array(calib['translation'])
        calib_r = np.array(calib['rotation'])
        ego_t   = np.array(ego['translation'])
        ego_r   = np.array(ego['rotation'])

        if slice_len == 3:
            # LIDAR: trasformazione batch veloce
            transformed = tru.transform_lidar_points_batch(arr[:, :3], calib_t, calib_r, ego_t, ego_r)
        else:
            # RADAR: trasformazione completa punto per punto
            transformed = np.array([
                tru.transform_point_to_world(p[:slice_len], calib_t, calib_r, ego_t, ego_r)
                for p in arr
            ])

        sensor_points.append(transformed)

    return sensor_points

def _build_boxes_for_sample(trucksc, sample, unified_points: np.ndarray, label_box: str) -> List[Box]:
    """Associa i punti del sample alle ann box e restituisce Box popolate"""
    boxes = []
    point_mask = np.full(unified_points.shape[0], False)

    tree = cKDTree(unified_points[:, :3])  # indicizzazione spaziale dei punti

    for ann_token in sample['anns']:
        ann = trucksc.get('sample_annotation', ann_token)
        box = Box(ann['translation'], ann['size'], ann['rotation'], ann['category_name'], label_box)

        center = np.array(box.center[:3])
        radius = np.linalg.norm(box.size) / 2  # stima: raggio che copre la box

        # ottieni gli indici dei punti vicini al centro della box
        nearby_indices = tree.query_ball_point(center, r=radius)

        # verifica precisa solo sui candidati vicini
        for idx in nearby_indices:
            if not point_mask[idx] and box.contains(unified_points[idx][:3]):
                point_mask[idx] = True
                box.add_point(unified_points[idx])

        if box.get_num_point() > 0:
            boxes.append(box)

    return boxes


def _process_frames(trucksc, tokens: List[str], dir_path: str, sensor_type: str):
    """Estrae le feature delle box popolate con i punti di tutti i frame passati"""
    meta = SENSOR_META[sensor_type]
    sensor_labels = meta['labels']
    slice_len = meta['slice_len']

    X_res = []
    y_res = []

    for token in tqdm(tokens, desc=f"Elaborazione frame {sensor_type}"):
        sample = trucksc.get('sample', token)
        sensor_pts = _load_sensor_points(trucksc, sample, dir_path, sensor_labels, slice_len)
        if not sensor_pts:
            continue
        unified = np.concatenate(sensor_pts, axis=0)
        boxes = _build_boxes_for_sample(trucksc, sample, unified, label_box=sensor_type)

        for b in boxes:
            X_res.append(b.get_features_arr())
            y_res.append(b.get_box_label())
            assert len(X_res) == len(y_res)

        # Forza la rimozione delle box con tutti i punti dentro
        boxes.clear()

    return X_res, y_res

# -----------------------------------------------------------------------------
# Funzioni di alto livello
# -----------------------------------------------------------------------------

def train_classifier(
    trucksc,
    arr_first_sample: list[str],
    dir_path: str,
    test_size_par: float = 1.0,
    sensor_type_par: str = DEFAULT_SENSOR_TYPE
) -> RandomForestClassifier:
    """Addestra un RandomForestClassifier per il sensore indicato (radar/lidar)."""
    assert sensor_type_par in SENSOR_META, f"Sensore non supportato: {sensor_type_par}"

    start = time()

    train_sample_tokens = []
    for first_sample_token in arr_first_sample:
        train_sample_tokens.append(_collect_sample_tokens(trucksc, first_sample_token))

    print(f"Scene usate per il training ({sensor_type_par}): {len(train_sample_tokens)}")
    # Serve "appiattire" la lista di token, ora una lista di liste
    train_sample_tokens = list(chain.from_iterable(train_sample_tokens))
    print(f"Frame totali usati per il training ({sensor_type_par}): {len(train_sample_tokens)}")

    
    X_train, y_train = _process_frames(trucksc, train_sample_tokens, dir_path, sensor_type=sensor_type_par)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Rimuove dalla ram i vettori di training:
    X_train.clear()
    y_train.clear()
    
    print(f"Training completato in {time() - start} s")
    return clf


def test_classifier(
    trucksc,
    first_sample_token: str,
    dir_path: str,
    clf: RandomForestClassifier,
    test_size,
    sensor_type_par: str = DEFAULT_SENSOR_TYPE
) -> Tuple[List, List, List, float, List[str]]:
    """Valuta il classificatore sul set di test del sensore indicato."""
    assert sensor_type_par in SENSOR_META, f"Sensore non supportato: {sensor_type_par}"

    start = time()
    tokens = _collect_sample_tokens(trucksc, first_sample_token)
    _, test_tokens = train_test_split(tokens, test_size=test_size, random_state=42)
    print(f"Frame di test ({sensor_type_par}): {len(test_tokens)}")

    X_test, y_test = _process_frames(trucksc, test_tokens, dir_path, sensor_type=sensor_type_par)
    assert len(X_test) == len(y_test)

    y_pred = clf.predict(X_test)
    label_names = sorted(list(unique_labels(y_test, y_pred)))
    exec_time = time() - start

    return X_test, y_test, y_pred, exec_time, label_names