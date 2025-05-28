from scipy.spatial.transform import Rotation as R
import numpy as np

def transform_point_to_world(point, calib_translation, calib_rotation, ego_translation, ego_rotation):
    """
    Trasforma un punto radar completo (posizione, velocità relativa, rcs) 
    dal frame del sensore al world frame tramite quaternioni e le opportune
    traslazioni
    """
    # Sensore → Veicolo
    rot_calib = R.from_quat([calib_rotation[1], calib_rotation[2], calib_rotation[3], calib_rotation[0]])
    point_in_vehicle = rot_calib.apply(point[:3]) + calib_translation
    vrel_in_vehicle = rot_calib.apply(point[3:6])

    # Veicolo → Mondo
    rot_ego = R.from_quat([ego_rotation[1], ego_rotation[2], ego_rotation[3], ego_rotation[0]])
    point_in_world = rot_ego.apply(point_in_vehicle) + ego_translation
    vrel_in_world = rot_ego.apply(vrel_in_vehicle)

    return np.concatenate([point_in_world, vrel_in_world, [point[6]]])
