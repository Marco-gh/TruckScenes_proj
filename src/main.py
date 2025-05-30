from truckscenes import TruckScenes
import radar_analyzer as rd
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    dir_path = "/home/marco/Documents/TAV project/dataset/man-truckscenes/"
    trucksc = TruckScenes('v1.0-mini', dir_path, verbose=True)

    first_sample_token = trucksc.scene[0]['first_sample_token']

    X_test, X_train, y_test, y_train, y_pred, ex_time, labels = rd.radar_scene_analysis(trucksc, first_sample_token, dir_path)
    # Matrice di confusione
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"Matrice di confuzione:\n {cm}")
    # Accuratezza: percentuale di predizioni corrette
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuratezza: {accuracy:.2f}")
    # Tempo di esecuzione
    print(f"Tempo analisi dati radar: {ex_time}")

if __name__ == "__main__":
    main()