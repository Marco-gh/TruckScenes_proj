import random
from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm.auto import tqdm
from truckscenes import TruckScenes

import data_analyzer as dtan

# -----------------------------------------------------------------------------
# Funzione per visualizzazione
# -----------------------------------------------------------------------------

def _print_report(y_true: List[int], y_pred: List[int], labels: List[str], title: str) -> None:
    """Stampa accuracy, confusion matrix e classification report in modo compatto."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n----- {title.upper()} -----")
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels))


# -----------------------------------------------------------------------------
# Valutazione scena
# -----------------------------------------------------------------------------

def evaluate_scene(
    trucksc: TruckScenes,
    scene_idx: int,
    dir_path: str,
    clf_radar: Tuple,
    clf_lidar: Tuple,
    test_size: float | None,
) -> Dict[str, Dict[str, List]]:
    """Valuta radar & lidar su una singola scena e restituisce i risultati di RFC e GBC."""
    first_token = trucksc.scene[scene_idx]["first_sample_token"]

    results: Dict[str, Dict[str, List]] = {}
    for sensor, (rfc, gbc) in [("radar", clf_radar), ("lidar", clf_lidar)]:
        print(f"\nEvaluating {sensor.upper()} on scene {scene_idx}…")

        (_, y_test, y_pred_rfc, y_pred_gbc, _, label_names,) = dtan.test_classifier(
            trucksc,
            first_token,
            dir_path,
            rfc,
            gbc,
            test_size=test_size,
            sensor_type_par=sensor,
        )

        _print_report(y_test, y_pred_gbc, label_names, sensor)
        _print_report(y_test, y_pred_rfc, label_names, sensor)

        results[sensor] = {"y_true": y_test, "y_pred_rfc": y_pred_rfc, "y_pred_gbc": y_pred_gbc,}

    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    # Path al dataset TruckScenes
    dir_path = "/home/marco/Documents/TAV project/dataset/man-truckscenes"
    trucksc = TruckScenes("v1.0-trainval", dir_path, verbose=True)

    # Suddivisione delle 77 scene in 61 di training (~80 %) e 16 di test (~20 %)
    train_scenes = random.sample(range(77), 61)
    test_scenes = [i for i in range(77) if i not in train_scenes]

    # Token iniziali delle scene di training
    first_train_tokens = [trucksc.scene[idx]["first_sample_token"] for idx in train_scenes]

    print("\n--- ALLENAMENTO CLASSIFICATORI ---")
    rfc_radar, gbc_radar = dtan.train_classifier(trucksc, first_train_tokens, dir_path, sensor_type_par="radar")
    rfc_lidar, gbc_lidar = dtan.train_classifier(trucksc, first_train_tokens, dir_path, sensor_type_par="lidar")

    # Contenitori per le metriche aggregate
    aggregate: Dict[str, Dict[str, List]] = {
        "radar": {"y_true": [], "rfc": [], "gbc": []},
        "lidar": {"y_true": [], "rfc": [], "gbc": []},
    }

    # Valutazione su ogni scena di test
    for idx in tqdm(test_scenes, desc="Testing scenes", unit="scene"):
        scene_results = evaluate_scene(
            trucksc,
            idx,
            dir_path,
            (rfc_radar, gbc_radar),
            (rfc_lidar, gbc_lidar),
            test_size=None,  # Nessuna divisione tra frame -> usati tutti per il test
        )

        for sensor, res in scene_results.items():
            aggregate[sensor]["y_true"].extend(res["y_true"])
            aggregate[sensor]["rfc"].extend(res["y_pred_rfc"])
            aggregate[sensor]["gbc"].extend(res["y_pred_gbc"])

    # Report globali
    for sensor in ["radar", "lidar"]:
        print(f"\n==== GLOBAL METRICS {sensor.upper()} - RANDOM FOREST ====")
        _print_report(
            aggregate[sensor]["y_true"],
            aggregate[sensor]["rfc"],
            labels=sorted(set(aggregate[sensor]["y_true"])),
            title=f"{sensor} - Random Forest",
        )

        print(f"\n==== GLOBAL METRICS {sensor.upper()} - GRADIENT BOOSTING ====")
        _print_report(
            aggregate[sensor]["y_true"],
            aggregate[sensor]["gbc"],
            labels=sorted(set(aggregate[sensor]["y_true"])),
            title=f"{sensor} - Gradient Boosting",
        )

    print("\nFINE")


if __name__ == "__main__":
    main()