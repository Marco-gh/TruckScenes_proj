from truckscenes import TruckScenes
import numpy as np
from pypcd4 import PointCloud # github.com/MapIV/pypcd4
from box import Box
import os
import truck_utility as tru
from time import time

# Per dividere i dati in train e test e classificarli tramite un classificatore random forest
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.html
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Visualizzazione risultati, avanzamento progressi
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def lidar_scene_analysis(trucksc, first_sample_token, dir_path):
    start_time = time()

    label_lidar = [
    'LIDAR_LEFT',
    'LIDAR_RIGHT',
    'LIDAR_TOP_FRONT',
    'LIDAR_TOP_LEFT',
    'LIDAR_TOP_RIGHT',
    'LIDAR_REAR'
    ]

    # Precarica tutti i token dei frame della scena per usare tqdm
    # DA SPOSTARE NEL MAIN QUESTO PRECARICAMENTO?????
    sample_tokens = []
    current_token = first_sample_token
    while current_token:
        sample_tokens.append(current_token)
        sample = trucksc.get('sample', current_token)
        current_token = sample['next'] if sample['next'] else None

    # Ciclo su tutti i frame con barra di avanzamento tqdm
    all_boxes = []