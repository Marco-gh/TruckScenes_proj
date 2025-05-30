import numpy as np
from scipy.spatial.transform import Rotation as R
import json

# PARAMETRI GLOBALI
MOTION_THRESHOLD = 0.01  # [m/s] sotto il quale considero l’oggetto “fermo”

# MAPPING LABEL
ROOT_TO_CLASS = {
    'vehicle'        : 'dynamic',
    'ego_vehicle'    : 'dynamic',
    'static_object'  : 'static',
    'movable_object' : 'movable',
    'human'          : 'vulnerable',
    'animal'         : 'vulnerable',
}

def map_label_category(label: str) -> str:
    """Converte il label MAN TruckScenes in una delle 5 macro-categorie usate nel random forest"""
    root = label.split('.')[0]  # 'vehicle.car' → 'vehicle'
    return ROOT_TO_CLASS.get(root, 'unknown')

class Box:
    """
    Raggruppa i punti radar (o lidar) appartenenti a una bounding-box annotata
    e calcola statistiche/feature utili per la classificazione.
    """
    def __init__(self, center, size, quaternion, ann_cat_name):
        # posizione, dimensioni, orientamento
        self.center = np.asarray(center, dtype=np.float32)
        self.size   = np.asarray(size,   dtype=np.float32)          # w, l, h
        w, x, y, z  = quaternion
        self.rotation      = R.from_quat([x, y, z, w])
        self.inv_rotation  = self.rotation.inv()

        # categoria mappata
        self.label = map_label_category(str(ann_cat_name))

        # parametri pre-calcolati
        self.volume           = float(self.size.prod())
        self.elongation_ratio = float(self.size.max() / self.size.min())

        self.points_arr = []        # lista di punti completi (x,y,z,vx,vy,vz,rcs)
        self.rcs_values = []
        self.velocities = []        # vettori velocità (vx,vy,vz)

    def contains(self, point_xyz: np.ndarray) -> bool:
        """True se il punto (x,y,z) cade dentro la box (correttamente 
        orientata per cambio sistema di riferimento)."""
        rel_pt = self.inv_rotation.apply(point_xyz - self.center)
        return np.all(np.abs(rel_pt) <= self.size / 2)

    def add_point(self, point: np.ndarray):
        """Aggiunge un punto (array len≥7) alla box e aggiorna liste."""
        self.points_arr.append(point)
        self.rcs_values.append(point[6])
        self.velocities.append(point[3:6])

    def get_num_point(self) -> int:
        return len(self.points_arr)

    def get_box_label(self) -> str:
        return self.label

    def get_features_arr(self):
        """Restituisce il vettore di feature"""
        pts = self.get_num_point()
        pts_dens = pts / self.volume if self.volume else 0

        rcs = np.array(self.rcs_values, dtype=np.float32)
        avg_rcs   = float(rcs.mean()) if pts else 0
        sigma_rcs = float(rcs.std())  if pts else 0

        vel = np.array(self.velocities, dtype=np.float32)
        avg_vel_vec = vel.mean(axis=0) if pts else np.zeros(3, dtype=np.float32)
        sigma_vel   = vel.std(axis=0)  if pts else np.zeros(3, dtype=np.float32)

        motion_flag = int(np.linalg.norm(avg_vel_vec) > MOTION_THRESHOLD)
        vertical_pos = float(self.center[2])

        return [
            pts,
            self.volume,
            pts_dens,
            avg_rcs,
            sigma_rcs,
            float(np.linalg.norm(avg_vel_vec)),
            float(np.linalg.norm(sigma_vel)),
            self.elongation_ratio,
            motion_flag,
            vertical_pos,
        ]

    # FUNZIONE PER DEBUG
    def get_features_json(self):
        """Restituisce le feature in formato JSON (utile per debug/log)."""
        features = {
            "points_num"            : self.get_num_point(),
            "volume"                : self.volume,
            "points_density"        : self.get_num_point() / self.volume if self.volume else 0,
            "avg_rcs"               : np.mean(self.rcs_values) if self.rcs_values else 0,
            "sigma_rcs"             : np.std(self.rcs_values)  if self.rcs_values else 0,
            "avg_speed_magnitude"   : float(np.linalg.norm(np.mean(self.velocities, axis=0))) if self.velocities else 0,
            "sigma_speed_magnitude" : float(np.linalg.norm(np.std(self.velocities, axis=0)))  if self.velocities else 0,
            "elongation_ratio"      : self.elongation_ratio,
            "motion_flag"           : bool(np.linalg.norm(np.mean(self.velocities, axis=0)) > MOTION_THRESHOLD) if self.velocities else False,
            "vertical_position"     : float(self.center[2]),
            "label"                 : self.label,
        }
        return json.dumps(features, indent=2)
