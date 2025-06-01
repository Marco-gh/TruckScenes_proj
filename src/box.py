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
    def __init__(self, center, size, quaternion, ann_cat_name, label_box):
        assert(label_box=='radar' or label_box=='lidar')
        self.label_box = label_box

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

        self.points_arr = []        # lista di punti completi (x,y,z,...) radar o lidar
        if label_box=='radar':
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
        if self.label_box=='radar':
            self.rcs_values.append(point[6])
            self.velocities.append(point[3:6])

    def get_num_point(self) -> int:
        return len(self.points_arr)

    def get_box_label(self) -> str:
        return self.label

    def get_features_arr(self):
        # Altezza fisica della bounding box
        height = self.size[2]

        # Numero totale di punti radar/lidar dentro la bounding box
        points_num = len(self.points_arr)

        # Densità dei punti (numero di punti per unità di volume)
        points_dens = points_num / self.volume if self.volume > 0 else 0

        # Coordinate dei punti
        points = np.array(self.points_arr)
        positions = points[:, :3] if points.shape[0] > 0 else np.empty((0, 3))

        # Statistiche geometriche condivise tra radar e lidar
        if positions.shape[0] > 0:
            std_xyz = np.std(positions, axis=0)  # deviazione standard su x,y,z
            min_z = np.min(positions[:, 2])
            max_z = np.max(positions[:, 2])
            rel_height = max_z - min_z           # altezza effettiva dei punti
            perc_below_center = np.mean(positions[:, 2] < self.center[2])  # % punti sotto il centro
        else:
            std_xyz = np.zeros(3)
            rel_height = 0
            perc_below_center = 0

        # Feature comuni radar/lidar
        features = [
            points_num,   
            self.volume,    
            points_dens,    
            self.elongation_ratio,  # Forma dell'oggetto (lungo/schiacciato)
            height,                 # Dimensione verticale box
            std_xyz[0],             # Dispersione lungo X
            std_xyz[1],             # Dispersione lungo Y
            std_xyz[2],             # Dispersione lungo Z
            rel_height,             # Altezza effettiva dei punti nella box
            perc_below_center       # % punti sotto il centro della box
        ]

        if self.label_box=='radar':
            # Lista dei vettori velocità tridimensionali dei punti
            velocities = np.array(self.velocities)

            # Deviazione standard delle velocità (indica variabilità del movimento)
            sigma_vel = np.std(velocities, axis=0) if len(velocities) > 0 else np.zeros(3)

            # Valore medio del Radar Cross Section (RCS) dei punti
            avg_rcs = np.mean(self.rcs_values) if self.rcs_values else 0

            # Deviazione standard del RCS (quanto varia la riflettività all'interno della box)
            sigma_rcs = np.std(self.rcs_values) if self.rcs_values else 0

            # Velocità media (vettoriale) dei punti all'interno della box
            avg_speed_direction = np.mean(velocities, axis=0) if len(velocities) > 0 else np.zeros(3)

            # Flag binario che indica se l'oggetto si sta muovendo (modulo velocità media > soglia)
            motion_flag = np.linalg.norm(avg_speed_direction) > MOTION_THRESHOLD

            # Indicatore di staticità
            percent_quasi_static = np.mean(np.linalg.norm(velocities, axis=1) < 0.02) if len(velocities) > 0 else 0

            features += [
                avg_rcs,
                sigma_rcs,
                np.linalg.norm(avg_speed_direction),  # Modulo velocità media
                np.linalg.norm(sigma_vel),            # Variabilità velocità
                int(motion_flag),
                percent_quasi_static                  # % punti quasi fermi
            ]

        return features
