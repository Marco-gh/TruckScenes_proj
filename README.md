# TruckScenes Radar vs LiDAR — Supervised Classification (RF vs GBC)

University project that compares **Random Forest** and **Gradient Boosting** for **object classification** using **TruckScenes** data, running the pipeline **separately on Radar and LiDAR**. Labels are remapped into 4 macro-classes: **dynamic, static, movable, vulnerable**. :contentReference[oaicite:0]{index=0}  

## What it does
- Loads TruckScenes samples and 3D annotations (bounding boxes)
- Extracts features per box:
  - shared (geometry/statistics) + radar-only (e.g., relative speed, RCS)
- Trains and evaluates **RF** and **GBC**
- Reports **accuracy, confusion matrix, precision/recall/F1** :contentReference[oaicite:2]{index=2}

## Dataset
Uses TruckScenes (MAN). Due to size constraints, only a **subset of the training scenes** is used (e.g., first block ~77 scenes). :contentReference[oaicite:3]{index=3}

## Requirements
- Python 3.x
- `numpy`, `scikit-learn`, (TruckScenes devkit), plus common utilities

## Run (template)
```bash
pip install -r requirements.txt

python src/run_experiment.py --sensor radar --model rf
python src/run_experiment.py --sensor lidar --model gbc
