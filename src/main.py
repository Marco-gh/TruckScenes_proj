from truckscenes import TruckScenes
import radar_analyzer as rd
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    dir_path = "/home/marco/Documents/TAV project/dataset/man-truckscenes/"
    trucksc = TruckScenes('v1.0-mini', dir_path, verbose=True)

    first_sample_token = trucksc.scene[7]['first_sample_token']

    # Parametro per numero di frame da usare in fase di test
    test_size = 0.95
    # Allenamento classificatore:
    clf = rd.train_radar_classifier(trucksc, first_sample_token, dir_path, test_size)
    # Uso classificatore:
    X_test, y_test, y_pred, ex_time, labels = rd.radar_scene_analysis(trucksc, first_sample_token, dir_path, clf, test_size)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Matrice di confusione:\n", cm)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuratezza: {accuracy:.2f}")

    print(f"Tempo analisi dati radar: {ex_time:.1f} s")

if __name__ == "__main__":
    main()
