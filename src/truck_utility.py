from scipy.spatial.transform import Rotation as R
import numpy as np

def transform_point_to_world(point, calib_translation, calib_rotation, ego_translation, ego_rotation):
    """
    Trasforma un punto (radar o lidar) dal frame sensore al frame mondo.
    Gestisce automaticamente:
    - punti LIDAR: [x, y, z]
    - punti RADAR: [x, y, z, vx, vy, vz, rcs]
    """
    # Calcola rotazioni come Rotation objects
    rot_calib = R.from_quat([calib_rotation[1], calib_rotation[2], calib_rotation[3], calib_rotation[0]])
    rot_ego   = R.from_quat([ego_rotation[1], ego_rotation[2], ego_rotation[3], ego_rotation[0]])

    # Trasforma posizione
    pos_in_vehicle = rot_calib.apply(point[:3]) + calib_translation
    pos_in_world   = rot_ego.apply(pos_in_vehicle) + ego_translation

    # Se disponibile, trasforma anche la velocità e RCS
    if len(point) >= 7:
        vrel_in_vehicle = rot_calib.apply(point[3:6])
        vrel_in_world   = rot_ego.apply(vrel_in_vehicle)
        rcs = point[6]
        return np.concatenate([pos_in_world, vrel_in_world, [rcs]])

    return pos_in_world


def transform_lidar_points_batch(points: np.ndarray,
                                  calib_translation: np.ndarray, calib_rotation: np.ndarray,
                                  ego_translation: np.ndarray, ego_rotation: np.ndarray) -> np.ndarray:
    """
    Trasforma in un colpo solo una nuvola di punti lidar dal sistema di riferimento sensore a quello mondo
    Parametri:
        - points: array (N, 3)
        - *_translation: array (3,)
        - *_rotation: quaternion (4,) in formato [w, x, y, z]
    """
    if points.shape[1] != 3:
        raise ValueError(f"I punti LIDAR devono essere (N, 3), ricevuto {points.shape}")

    # Rotazioni
    rot_calib = R.from_quat([
        calib_rotation[1], calib_rotation[2], calib_rotation[3], calib_rotation[0]
    ])
    rot_ego = R.from_quat([
        ego_rotation[1], ego_rotation[2], ego_rotation[3], ego_rotation[0]
    ])

    # Trasformazioni batch
    points_in_vehicle = rot_calib.apply(points) + calib_translation
    points_in_world = rot_ego.apply(points_in_vehicle) + ego_translation

    return points_in_world
