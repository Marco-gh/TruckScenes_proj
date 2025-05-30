from truckscenes import TruckScenes
import radar_analyzer as rd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

# Etichette semantiche
static_labels = {'traffic_cone', 'barrier', 'construction_cone', 'construction_barrel'}
vehicle_labels = {'car', 'truck', 'bus', 'trailer', 'motorcycle'}
vulnerable_labels = {'bicycle', 'pedestrian'}
movable_labels = {'pushable_pullable'}

def map_label_category(label):
    if label in static_labels:
        return 'static'
    elif label in vehicle_labels:
        return 'dynamic'
    elif label in vulnerable_labels:
        return 'vulnerable'
    elif label in movable_labels:
        return 'movable'
    else:
        return 'unknown'

def main():
    dir_path = "/home/marco/Documents/TAV project/dataset/man-truckscenes/"
    trucksc = TruckScenes('v1.0-mini', dir_path, verbose=True)

    # Percentuale di frame che rimane per il test
    #test_size = 0.7
    # Scena di allenamento
    first_train_sample_token = trucksc.scene[5]['first_sample_token']
    # Allenamento classificatore:
    clf = rd.train_radar_classifier(trucksc, first_train_sample_token, dir_path, 0.1)

    # Lista per riepilogo finale
    summary = []

    for sc in range(len(trucksc.scene)):
        print(f"\n===== Scena {sc} =====")
        # Scena di test
        first_test_sample_token = trucksc.scene[sc]['first_sample_token']

        # Uso classificatore
        X_test, y_test, y_pred, ex_time, labels = rd.radar_scene_analysis(trucksc, first_test_sample_token, dir_path, clf, 0.95)

        # Matrice di confusione
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        print("Matrice di confusione:\n", cm)

        # Accuratezza
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuratezza: {accuracy:.2f}")

        # Tempo di esecuzione
        print(f"Tempo analisi dati radar: {ex_time:.1f} s")

        # Distribuzione categorie predette
        predicted_categories = [map_label_category(label) for label in y_pred]
        category_count = Counter(predicted_categories)
        print("Distribuzione predizioni per categoria:")
        for cat, count in category_count.items():
            print(f"{cat}: {count}")

        # Report classificazione
        print("Report classificazione:")
        print(classification_report(y_test, y_pred, zero_division=0))

    print("\n===== RIEPILOGO =====")
    for entry in summary:
        print(f"\nScena {entry['scene']} → Accuracy: {entry['accuracy']:.2f}, Tempo: {entry['execution_time']:.1f}s")
        print("Matrice di confusione:")
        for row in entry['confusion_matrix']:
            print(row)
        print("Distribuzione predizioni per categoria:")
        for cat, count in entry['category_count'].items():
            print(f"{cat}: {count}")

if __name__ == "__main__":
    main()
