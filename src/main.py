import os
from time import time
from collections import Counter
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from truckscenes import TruckScenes

import data_analyzer as dtan  # Modulo con le funzioni train/test astratte
import random

# -----------------------------------------------------------------------------
# Helper di visualizzazione
# -----------------------------------------------------------------------------
def _print_report(y_true, y_pred, labels: list, sensor: str):
    """Stampa accuracy, confusion matrix e classification report per il sensore."""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"\n----- {sensor.upper()} RESULTS -----")
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def evaluate_scene(trucksc: TruckScenes, scene_idx: int, dir_path: str,
                   clf_radar, clf_lidar, test_size) -> Dict[str, Tuple]:
    """Esegue valutazione radar & lidar su una singola scena e ritorna i risultati."""
    first_token = trucksc.scene[scene_idx]['first_sample_token']

    results = {}
    for sensor, clf in [('radar', clf_radar), ('lidar', clf_lidar)]:
        print(f"\nEvaluating {sensor.upper()} on scene {scene_idx}…")
        X, y_true, y_pred, exec_time, labels = dtan.test_classifier(trucksc, first_token, dir_path, clf, test_size=test_size, sensor_type_par=sensor)
        _print_report(y_true, y_pred, labels, sensor)
        results[sensor] = (y_true, y_pred)
    return results


def main():
    # Per dataset mini
    #dir_path = "/home/marco/Documents/TAV project/dataset/man-truckscenes/"
    #trucksc = TruckScenes('v1.0-mini', dir_path, verbose=True)

    # Dataset più completo 
    dir_path = "/home/marco/Documents/TAV project/dataset/man-truckscenes"
    trucksc = TruckScenes('v1.0-trainval', dir_path, verbose=True)

    # Percentuale frame test per train
    train_pct = 1.0

    # 76 scene nel primo dataset scaricato (train dataset), scegliamo casualmente 61 scene per il training (80%)
    train_scenes = random.sample(range(77), 61)
    test_scenes = []
    for i in range(77):
        if i not in  train_scenes:
            test_scenes.append(i)

    # Raccolta dei primi frame delle scene
    first_train_token_arr = []
    for idx in train_scenes:
        first_train_token_arr.append(trucksc.scene[idx]['first_sample_token'])
    
    print("\n--- ALLENAMENTO CLASSIFICATORI ---")
    clf_radar = dtan.train_classifier(trucksc, first_train_token_arr, dir_path, test_size_par=train_pct, sensor_type_par='radar')
    clf_lidar = dtan.train_classifier(trucksc, first_train_token_arr, dir_path, test_size_par=train_pct, sensor_type_par='lidar')

    # Per valutazione aggregata di tutte le scene prese in considerazione
    aggregate_true_radar, aggregate_pred_radar = [], []
    aggregate_true_lidar, aggregate_pred_lidar = [], []

    # Test sul 20% delle scene
    for idx in test_scenes:
        print(f"\n============== SCENE {idx} ==============")
        scene_results = evaluate_scene(trucksc, idx, dir_path, clf_radar, clf_lidar, test_size=None)
        for sensor, (y_true, y_pred) in scene_results.items():
            # Aggrega per metriche globali (radar + lidar)
            if sensor == 'radar':
                aggregate_true_radar.extend(y_true)
                aggregate_pred_radar.extend(y_pred)
            elif sensor == 'lidar':
                aggregate_true_lidar.extend(y_true)
                aggregate_pred_lidar.extend(y_pred)
            
            #break # per valutare una scena sola (debug)

    # Report aggregato globale per radar
    print("\n==== METRICHE GLOBALI (RADAR) ====")
    print(f"Accuracy complessiva: {accuracy_score(aggregate_true_radar, aggregate_pred_radar):.2f}")
    print("Classification Report:")
    print(classification_report(aggregate_true_radar, aggregate_pred_radar, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(aggregate_true_radar, aggregate_pred_radar))

    # Report aggregato globale per lidar
    print("\n==== METRICHE GLOBALI (LIDAR) ====")
    print(f"Accuracy complessiva: {accuracy_score(aggregate_true_lidar, aggregate_pred_lidar):.2f}")
    print("Classification Report:")
    print(classification_report(aggregate_true_lidar, aggregate_pred_lidar, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(aggregate_true_lidar, aggregate_pred_lidar))

    print("\nDone.")


if __name__ == "__main__":
    main()
