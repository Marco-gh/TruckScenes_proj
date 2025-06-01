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

# -----------------------------------------------------------------------------
# Helper di visualizzazione
# -----------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray, labels: list, title: str):
    """Visualizza la matrice di confusione con seaborn."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


def _print_report(y_true, y_pred, labels: list, sensor: str):
    """Stampa accuracy, confusion matrix e classification report per il sensore."""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"\n----- {sensor.upper()} RESULTS -----")
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    #_plot_confusion_matrix(cm, labels, f"Confusion Matrix ({sensor})")
    return acc, cm


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def evaluate_scene(trucksc: TruckScenes, scene_idx: int, dir_path: str,
                   clf_radar, clf_lidar, test_size: float = 0.95) -> Dict[str, Tuple]:
    """Esegue valutazione radar & lidar su una singola scena e ritorna i risultati."""
    first_token = trucksc.scene[scene_idx]['first_sample_token']

    results = {}
    for sensor, clf in [('radar', clf_radar), ('lidar', clf_lidar)]:
        print(f"\nEvaluating {sensor.upper()} on scene {scene_idx}…")
        X, y_true, y_pred, exec_time, labels = dtan.test_classifier(trucksc, first_token, dir_path, clf, test_size_par=test_size, sensor_type_par=sensor)
        acc, cm = _print_report(y_true, y_pred, labels, sensor)
        results[sensor] = (acc, cm, exec_time, Counter(y_pred))
    return results


def main():
    dir_path = "/home/marco/Documents/TAV project/dataset/man-truckscenes/"
    trucksc = TruckScenes('v1.0-mini', dir_path, verbose=True)

    # Percentuale frame test per train
    train_pct = 0.1

    # Scena di riferimento per il training
    first_train_token = trucksc.scene[5]['first_sample_token']
    print("\n--- ALLENAMENTO CLASSIFICATORI ---")
    clf_radar = dtan.train_classifier(trucksc, first_train_token, dir_path, test_size_par=train_pct, sensor_type_par='radar')
    clf_lidar = dtan.train_classifier(trucksc, first_train_token, dir_path, test_size_par=train_pct, sensor_type_par='lidar')

    # Valutiamo tutte le scene
    aggregate_true, aggregate_pred = [], []
    summary = []

    # Analisi scena per scena
    for idx in range(len(trucksc.scene)):
        print(f"\n============== SCENE {idx} ==============")
        scene_results = evaluate_scene(trucksc, idx, dir_path, clf_radar, clf_lidar, test_size=0.95)
        for sensor, (acc, cm, t_exec, counter_pred) in scene_results.items():
            summary.append({
                'scene': idx,
                'sensor': sensor,
                'accuracy': acc,
                'exec_time': t_exec,
                'confusion_matrix': cm,
                'pred_counter': counter_pred
            })
        # Aggrega per metriche globali (radar + lidar insieme)
        aggregate_true.extend(scene_results['radar'][1].flatten())
        #break

    print("\n======== SUMMARY =========")
    for entry in summary:
        print(f"Scene {entry['scene']} | {entry['sensor']} → Acc: {entry['accuracy']:.2f} | Time: {entry['exec_time']:.1f}s")
        print("Predicted distribution:")
        for cat, count in entry['pred_counter'].items():
            print(f"  {cat}: {count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
