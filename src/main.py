from truckscenes import TruckScenes
import radar_analyzer as rd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

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
    # Liste aggregate per il test
    all_y_test = []
    all_y_pred = []

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

        # Report classificazione
        print("Report classificazione:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Per statistiche aggregate    
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        print("\n===== RIEPILOGO =====")
        for entry in summary:
            print(f"\nScena {entry['scene']} → Accuracy: {entry['accuracy']:.2f}, Tempo: {entry['execution_time']:.1f}s")
            print("Matrice di confusione:")
            for row in entry['confusion_matrix']:
                print(row)
            print("Distribuzione predizioni per categoria:")
            for cat, count in entry['category_count'].items():
                print(f"{cat}: {count}")

    print("\n===== STATISTICHE AGGREGATE =====")
    global_accuracy = accuracy_score(all_y_test, all_y_pred)
    global_cm = confusion_matrix(all_y_test, all_y_pred)
    global_report = classification_report(all_y_test, all_y_pred, zero_division=0)

    print(f"Accuratezza totale su tutte le scene: {global_accuracy:.2f}")
    print("Matrice di confusione aggregata:")
    print(global_cm)
    print("Report classificazione aggregato:")
    print(global_report)

if __name__ == "__main__":
    main()
