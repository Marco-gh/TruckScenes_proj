# TruckScenes Radar vs LiDAR — Supervised Classification (RF vs GBC)

University project that compares **Random Forest** and **Gradient Boosting** for **object classification** using **TruckScenes** data, running the pipeline **separately on Radar and LiDAR**. Labels are remapped into 4 macro-classes: **dynamic, static, movable, vulnerable**. 

## What it does
- Loads TruckScenes samples and 3D annotations (bounding boxes)
- Extracts features per box:
  - shared (geometry/statistics) + radar-only (e.g., relative speed, RCS)
- Trains and evaluates **RF** and **GBC**
- Reports **accuracy, confusion matrix, precision/recall/F1**

## Dataset
Uses TruckScenes (MAN). Due to size constraints, only a **subset of the training scenes** is used (e.g., first block ~77 scenes).

## Requirements
- Python 3.x
- `numpy`, `scikit-learn`, (TruckScenes devkit), plus common utilities

## Run (template)
```bash
pip install -r requirements.txt

python src/run_experiment.py --sensor radar --model rf
python src/run_experiment.py --sensor lidar --model gbc
```

## Useful links
https://brandportal.man/d/QSf8mPdU5Hgj
https://brandportal.man/d/QSf8mPdU5Hgj/devkit-tutorial#/-/dataset-schema
https://github.com/TUMFTM/truckscenes-devkit?tab=readme-ov-file
