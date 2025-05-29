import numpy as np
from scipy.spatial.transform import Rotation as R
import json

motion_trashhold = 0.01

class Box:
    def __init__(self, center, size, quaternion, ann_cat_name):
        self.center = np.array(center)
        self.size = np.array(size)
        w, x, y, z = quaternion
        self.rotation = R.from_quat([x, y, z, w])
        self.inv_rotation = self.rotation.inv()
        self.points_arr = []
        self.label = self.set_label_type(str(ann_cat_name))

        self.volume = size[0] * size[1] * size[2]
        self.elongation_ratio = max(size) / min(size)

        self.rcs_values = []
        self.velocities = []

    def contains(self, point):
        rel_point = self.inv_rotation.apply(point - self.center)
        return np.all(np.abs(rel_point) <= self.size / 2)
    
    def add_point(self, point):
        self.points_arr.append(point)
        self.rcs_values.append(point[6])
        self.velocities.append(point[3:6])

    def get_features_arr(self):
        points_num = len(self.points_arr)
        points_dens = points_num / self.volume if self.volume > 0 else 0
        avg_rcs = np.mean(self.rcs_values) if self.rcs_values else 0
        sigma_rcs = np.std(self.rcs_values) if self.rcs_values else 0
        velocities = np.array(self.velocities)
        avg_speed_direction = np.mean(velocities, axis=0) if len(velocities) > 0 else np.zeros(3)
        sigma_vel = np.std(velocities, axis=0) if len(velocities) > 0 else np.zeros(3)
        motion_flag = np.linalg.norm(avg_speed_direction) > motion_trashhold
        vertical_pos = self.center[2]

        return [
            points_num,
            self.volume,
            points_dens,
            avg_rcs,
            sigma_rcs,
            np.linalg.norm(avg_speed_direction),
            np.linalg.norm(sigma_vel),
            self.elongation_ratio,
            int(motion_flag),
            vertical_pos
        ]

    def get_point(self):
        return np.array(self.points_arr)
    
    def get_features_json(self, motion_trashhold):
        """
            Utile per il debug forse da togliere
        """
        # Calcolo delle feature
        points_num = len(self.points_arr)
        volume = self.volume
        points_dens = points_num / volume if volume > 0 else 0

        avg_rcs = np.mean(self.rcs_values) if self.rcs_values else 0
        sigma_rcs = np.std(self.rcs_values) if self.rcs_values else 0

        velocities = np.array(self.velocities)
        if len(velocities) > 0:
            avg_speed_direction = np.mean(velocities, axis=0)
            sigma_vel = np.std(velocities, axis=0)
        else:
            avg_speed_direction = np.zeros(3)
            sigma_vel = np.zeros(3)

        motion_flag = bool(np.linalg.norm(avg_speed_direction) > motion_trashhold)
        vertical_pos = self.center[2]

        # Costruzione del dizionario delle feature
        features = {
            "points_num": points_num,
            "volume": volume,
            "points_density": points_dens,
            "avg_rcs": avg_rcs,
            "sigma_rcs": sigma_rcs,
            "avg_speed_magnitude": float(np.linalg.norm(avg_speed_direction)),
            "sigma_speed_magnitude": float(np.linalg.norm(sigma_vel)),
            "elongation_ratio": self.elongation_ratio,
            "motion_flag": motion_flag,
            "vertical_position": vertical_pos,
            "label": self.label
        }

        # Serializzazione in JSON
        return json.dumps(features)

    def set_label_type(self, label):
        # Oggetti statici
        static_labels = [
            'static_object.traffic_sign'
        ]
        # Veicoli
        vehicle_labels = [
            'vehicle.bicycle',
            'vehicle.bus.rigid',
            'vehicle.car',
            'vehicle.construction',
            'vehicle.ego_trailer',
            'vehicle.motorcycle',
            'vehicle.other',
            'vehicle.trailer',
            'vehicle.train',
            'vehicle.truck'
        ]
        # Animali e pedoni (soggetti vulnerabili)
        vulerable_labels = [
            'animal',
            'human.pedestrian.adult',
            'human.pedestrian.child',
            'human.pedestrian.constructi',
            'human.pedestrian.stroller'
        ]
        # Oggetti mobili non veicolari
        movable_labels = [
            'movable_object.barrier',
            'movable_object.trafficcone'
        ]

        if label in static_labels:
            return 'static'
        elif label in vehicle_labels:
            return 'dynamic'
        elif label in vulerable_labels:
            return 'vulnerable'
        elif label in movable_labels:
            return 'movable'
        else:
            return 'unknown'

    def get_box_label(self):
        return self.label
    
    def get_num_point(self):
        return len(self.points_arr)