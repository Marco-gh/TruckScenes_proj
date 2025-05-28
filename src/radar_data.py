from truckscenes import TruckScenes
import numpy as np
from pypcd4 import PointCloud # github.com/MapIV/pypcd4
from box import Box
import os
import truck_utility as tru

dir_path = "/home/marco/Documents/TAV project/dataset/man-truckscenes/"
trucksc = TruckScenes('v1.0-mini', dir_path, verbose=True)

first_sample_token = trucksc.scene[0]['first_sample_token']
my_sample = trucksc.get('sample', first_sample_token)

label_radar = [
    'RADAR_RIGHT_BACK',
    'RADAR_RIGHT_SIDE',
    'RADAR_RIGHT_FRONT',
    'RADAR_LEFT_FRONT',
    'RADAR_LEFT_SIDE',
    'RADAR_LEFT_BACK'
]

# Lista di array con i dati da tutti i sensori
radar_points = []

for lab in label_radar:
    radar_data = trucksc.get('sample_data', my_sample['data'][lab])
    full_path_data = os.path.join(dir_path, radar_data.get("filename"))

    pc: PointCloud = PointCloud.from_path(full_path_data)
    array: np.ndarray = pc.numpy()

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

# Array con tutti i dati provenienti dai radar
radar_unified_array = np.concatenate(radar_points, axis=0)
# Punti visitati o meno
labels = np.full(radar_unified_array.shape[0], False)

# Vediamo ora se i punti sono in una box
boxes = []
for ann_token in my_sample['anns']:
    ann_json = trucksc.get('sample_annotation',ann_token)
    box = Box(ann_json['translation'],ann_json['size'],ann_json['rotation'])
    boxes.append(box)
    for indx, p in enumerate(radar_unified_array):
        if not labels[indx] and box.contains(p[:3]):
            labels[indx] = True
            box.add_point(p)

for b in boxes:
    a = b.get_features_arr()
    if a[0] > 1:
        print(a)
