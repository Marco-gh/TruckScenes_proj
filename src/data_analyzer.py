import os
from itertools import chain
from typing import Dict, List, Optional, Tuple

import numpy as np
from pypcd4 import PointCloud  # github.com/MapIV/pypcd4
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from tqdm.auto import tqdm

import truck_utility as tru
from box import Box

# -----------------------------------------------------------------------------
# Configurazioni costanti
# -----------------------------------------------------------------------------
RADAR_LABELS = [
    "RADAR_RIGHT_BACK",
    "RADAR_RIGHT_SIDE",
    "RADAR_RIGHT_FRONT",
    "RADAR_LEFT_FRONT",
    "RADAR_LEFT_SIDE",
    "RADAR_LEFT_BACK",
]

LIDAR_LABELS = [
    "LIDAR_LEFT",
    "LIDAR_RIGHT",
]

DEFAULT_SENSOR_TYPE = "lidar"

# slice_len descrive quante colonne estrarre da ogni punto
SENSOR_META: Dict[str, Dict] = {
    "radar": {"labels": RADAR_LABELS, "slice_len": 7},
    "lidar": {"labels": LIDAR_LABELS, "slice_len": 3},
}

# -----------------------------------------------------------------------------
# Funzioni di basso livello
# -----------------------------------------------------------------------------


def _collect_sample_tokens(trucksc, first_token: str) -> List[str]:
    """Ritorna tutti i sample-token consecutivi a partire dal primo."""
    tokens: List[str] = []
    current = first_token
    while current:
        tokens.append(current)
        current = trucksc.get("sample", current)["next"]
    return tokens


def _load_sensor_points(
    trucksc,
    sample,
    dir_path: str,
    sensor_labels: List[str],
    slice_len: int,
) -> List[np.ndarray]:
    """Carica i point-cloud dei sensori, li riporta nel world-frame e
    restituisce una lista di array NumPy (uno per sensore)."""
    
    sensor_points: List[np.ndarray] = []

    for lab in sensor_labels:
        # Lettura file point-cloud
        data = trucksc.get("sample_data", sample["data"][lab])
        pc_path = os.path.join(dir_path, data["filename"])
        arr = PointCloud.from_path(pc_path).numpy()   # shape = (N, 3 + …)

        # Riduzione punti lidar: da ~80 000 a 5 000 punti ----------
        if slice_len == 3 and arr.shape[0] > 10_000: # Lunghezza 3 = lidar
            # Scegliamo 10 000 punti casuali per non saturare la RAM
            idx_10k = np.random.choice(arr.shape[0], 10_000, replace=False)
            arr_10k = arr[idx_10k]

            # Costruiamo un KD-Tree sulle sole coordinate XYZ
            kdtree = cKDTree(arr_10k[:, :3])

            # Contiamo quanti vicini (r < 0.25 m) ha ogni punto
            neigh_count = kdtree.query_ball_point(
                arr_10k[:, :3], r=0.25, return_length=True
            )

            # Teniamo i punti con almeno 3 vicini totali
            dense_mask = neigh_count > 2
            arr_dense = arr_10k[dense_mask]

            # Per garantire sempre 5 000 punti finali
            source = arr_dense if arr_dense.shape[0] >= 5_000 else arr_10k
            idx_5k = np.random.choice(source.shape[0], 5_000, replace=False)
            arr = source[idx_5k]

        # Trasformazione nel sistema di riferimento inerziale
        calib = trucksc.get("calibrated_sensor", data["calibrated_sensor_token"])
        ego   = trucksc.get("ego_pose",        data["ego_pose_token"])

        if slice_len == 3: # LIDAR: batch veloce
            transformed = tru.transform_lidar_points_batch(
                arr[:, :3],
                calib["translation"], calib["rotation"],
                ego["translation"],   ego["rotation"],
            )
        else: # RADAR o altro: punto-per-punto
            transformed = np.array([
                tru.transform_point_to_world(
                    p[:slice_len],
                    calib["translation"], calib["rotation"],
                    ego["translation"],   ego["rotation"],
                )
                for p in arr
            ])

        if transformed.size: # Per evitare di inserire array vuoti
            sensor_points.append(transformed)

    return sensor_points


def _build_boxes_for_sample(
    trucksc,
    sample,
    unified_points: np.ndarray,
    label_box: str,
) -> List[Box]:
    """Popola Box con i punti assegnati tramite un KDTree rapido."""
    boxes: List[Box] = []
    point_mask = np.full(unified_points.shape[0], False, dtype=bool)
    tree = cKDTree(unified_points[:, :3])

    for ann_token in sample["anns"]:
        ann = trucksc.get("sample_annotation", ann_token)
        box = Box(
            ann["translation"],
            ann["size"],
            ann["rotation"],
            ann["category_name"],
            label_box,
        )

        center = np.asarray(box.center[:3])
        radius = np.linalg.norm(box.size) / 2.0
        for idx in tree.query_ball_point(center, r=radius):
            if not point_mask[idx] and box.contains(unified_points[idx][:3]):
                point_mask[idx] = True
                box.add_point(unified_points[idx])

        if box.get_num_point() > 0:
            boxes.append(box)

    return boxes


def _process_frames(
    trucksc,
    tokens: List[str],
    dir_path: str,
    sensor_type: str,
) -> Tuple[List[np.ndarray], List[str]]:
    """Estrae feature/label dalle box di tutte le frame tokens fornite."""
    meta = SENSOR_META[sensor_type]
    sensor_labels = meta["labels"]
    slice_len = meta["slice_len"]

    X_res: List[np.ndarray] = []
    y_res: List[str] = []

    for token in tqdm(tokens, desc=f"Frames {sensor_type}", unit="frame", leave=False):
        sample = trucksc.get("sample", token)
        sensor_pts = _load_sensor_points(
            trucksc, sample, dir_path, sensor_labels, slice_len
        )
        if not sensor_pts:
            continue

        try:
            unified = np.concatenate(sensor_pts, axis=0)
        except ValueError:  # tutti i sensori hanno restituito array vuoti
            continue

        boxes = _build_boxes_for_sample(
            trucksc, sample, unified, label_box=sensor_type
        )

        for b in boxes:
            X_res.append(b.get_features_arr())
            y_res.append(b.get_box_label())

    return X_res, y_res


# -----------------------------------------------------------------------------
# API di alto livello
# -----------------------------------------------------------------------------


def train_classifier(
    trucksc,
    arr_first_sample: List[str],
    dir_path: str,
    sensor_type_par: str = DEFAULT_SENSOR_TYPE,
) -> Tuple[RandomForestClassifier, GradientBoostingClassifier]:
    """Addestra sia RandomForest che GradientBoosting sul tipo di sensore indicato."""
    if sensor_type_par not in SENSOR_META:
        raise ValueError(f"Sensore non supportato: {sensor_type_par}")

    tokens_nested = [_collect_sample_tokens(trucksc, t) for t in arr_first_sample]
    print(f"Scene usate per il training ({sensor_type_par}): {len(tokens_nested)}")
    train_tokens = list(chain.from_iterable(tokens_nested))
    print(f"Frame totali usati per il training ({sensor_type_par}): {len(train_tokens)}")

    X_train, y_train = _process_frames(
        trucksc, train_tokens, dir_path, sensor_type=sensor_type_par
    )

    rfc = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    rfc.fit(X_train, y_train)

    gbc = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    gbc.fit(X_train, y_train)

    # libera RAM
    X_train.clear()
    y_train.clear()

    return rfc, gbc


def test_classifier(
    trucksc,
    first_sample_token: str,
    dir_path: str,
    rfc: RandomForestClassifier,
    gbc: GradientBoostingClassifier,
    test_size: Optional[float] = None,
    sensor_type_par: str = DEFAULT_SENSOR_TYPE,
):
    """Valuta entrambi i classificatori su una singola scena."""
    if sensor_type_par not in SENSOR_META:
        raise ValueError(f"Sensore non supportato: {sensor_type_par}")

    tokens = _collect_sample_tokens(trucksc, first_sample_token)

    if test_size is None or test_size >= 1.0:
        test_tokens = tokens  # usa tutte le frame
    else:
        _, test_tokens = train_test_split(tokens, test_size=test_size, random_state=42)
    print(f"Frame di test ({sensor_type_par}): {len(test_tokens)}")

    X_test, y_test = _process_frames(
        trucksc, test_tokens, dir_path, sensor_type=sensor_type_par
    )
    assert len(X_test) == len(y_test), "X_test e y_test hanno lunghezze diverse"

    y_pred_rfc = rfc.predict(X_test)
    y_pred_gbc = gbc.predict(X_test)

    label_names = sorted(list(unique_labels(y_test, y_pred_rfc)))

    # ritorna 0.0 come placeholder per eventuali metriche aggiuntive
    return X_test, y_test, y_pred_rfc, y_pred_gbc, 0.0, label_names
