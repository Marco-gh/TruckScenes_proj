import numpy as np
from scipy.spatial.transform import Rotation as R

motion_trashhold = 0.01

class Box:
    def __init__(self, center, size, quaternion):
        self.center = np.array(center)
        self.size = np.array(size)
        w, x, y, z = quaternion
        self.rotation = R.from_quat([x, y, z, w])
        self.inv_rotation = self.rotation.inv()
        self.points_arr = []

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
