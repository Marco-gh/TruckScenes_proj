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
        # Numero totale di punti radar dentro la bounding box
        points_num = len(self.points_arr)

        # Densità dei punti (numero di punti per unità di volume)
        points_dens = points_num / self.volume if self.volume > 0 else 0

        # Valore medio del Radar Cross Section (RCS) dei punti
        avg_rcs = np.mean(self.rcs_values) if self.rcs_values else 0

        # Deviazione standard del RCS (quanto varia la riflettività all'interno della box)
        sigma_rcs = np.std(self.rcs_values) if self.rcs_values else 0

        # Lista dei vettori velocità tridimensionali dei punti
        velocities = np.array(self.velocities)

        # Velocità media (vettoriale) dei punti all'interno della box
        avg_speed_direction = np.mean(velocities, axis=0) if len(velocities) > 0 else np.zeros(3)

        # Deviazione standard delle velocità (indica variabilità del movimento)
        sigma_vel = np.std(velocities, axis=0) if len(velocities) > 0 else np.zeros(3)

        # Flag binario che indica se l'oggetto si sta muovendo (modulo velocità media > soglia)
        motion_flag = np.linalg.norm(avg_speed_direction) > MOTION_THRESHOLD

        # Altezza assoluta del centro della box (utile per distinguere oggetti elevati o bassi)
        vertical_pos = self.center[2]

        # Feature aggiunte per migliorare distinzione tra categorie difficili:
        # Altezza fisica della bounding box
        height = self.size[2]

        # Indicatore di staticità
        percent_quasi_static = np.mean(np.linalg.norm(velocities, axis=1) < 0.02) if len(velocities) > 0 else 0

        return [
            points_num,   
            self.volume,    
            points_dens,    
            avg_rcs,        
            sigma_rcs,      
            np.linalg.norm(avg_speed_direction),  # Modulo velocità media
            np.linalg.norm(sigma_vel),            # Variabilità velocità
            self.elongation_ratio,       # Forma dell'oggetto (lungo/schiacciato)
            int(motion_flag),
            vertical_pos,
            height,                      # Dimensione verticale box
            percent_quasi_static         # % punti quasi fermi
        ]