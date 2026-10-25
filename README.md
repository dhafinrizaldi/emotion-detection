# Assignment 2: Detecting emotions with ML algorithms (Group 11)

Pedro Gómez Toja, Dhafin Praditya Rizaldi, Alvaro Huisman

Facial emotion recognition on FER2013 and KDEF with classical machine learning: a baseline comparison (Task 1), a subject-independent KDEF study (Task 2), cross-dataset transfer (Task 3), an explainable fuzzy classifier (Task 4), improvements to the FER2013 model (Task 5), and a real-time webcam demo (Task 6). The assignment is described in `Assignment_2_2026_instructions.pdf`.

## Final results

| Model | FER2013 PrivateTest accuracy | Macro-F1 |
|---|---|---|
| Task 1: HOG + Logistic Regression | 0.432 | 0.396 |
| Task 5: HOG + LBP + geometric features, balanced Random Forest | 0.515 | 0.489 |

The Task 5 model is the one used in the webcam demo.

## Repository layout

| Path | Contents |
|---|---|
| `emotion_detection_group11.ipynb` | Main notebook: Tasks 1–3, Task 4 feature extraction (4.1–4.2), Task 5 and Task 6 |
| `task4_fuzzy.ipynb` | Task 4 fuzzy classifier (pyFUME), run in a separate environment; see below |
| `demo.py` | Real-time webcam demo with the Task 5 model |
| `src/fer_features.py` | Task 5 feature extraction (HOG, LBP, geometric features), shared by the notebook and the demo |
| `src/preprocess.py` | Supplied KDEF preprocessing (face crop, grayscale, 48×48) |
| `models/task1` … `models/task5` | Trained models (`.pkl`) per task |
| `results/task1` … `results/task6` | Saved tables, predictions, confusion matrices and figures |
| `faceAndLandmarkModels/` | OpenCV Haar face detector and dlib 68-point landmark predictors |
| `auxFiles/kdef_subject_split.pkl` | Supplied 56/14 identity split for KDEF |
| `environment.yml`, `environment-fuzzy.yml` | Conda environments (see below) |
| `baseline_demo_fer2013.py`, `baseline_fer2013_landmarks.ipynb`, `z_load_*.ipynb`, `models/rf_fer_model_landmarks.pkl` | Supplied starter code and model, kept for reference |

## Why there are two notebooks

The fuzzy classifier in Task 4 uses **pyFUME**, which requires old library versions (numpy 1.24.4, pandas 1.5.3, scipy 1.10.1 and Python ≤ 3.11). These conflict with the versions used for everything else (numpy 2.x, pandas 3.x, Python 3.12), so the two cannot share one environment. The assignment also recommends a separate environment for pyFUME. The work is therefore split:

- **`emotion_detection_group11.ipynb`** (environment `emotion-detection`) contains all tasks, including the Task 4 landmark and geometric-feature extraction (sections 4.1–4.2). It writes the features to `results/task4/geometric_features.csv`.
- **`task4_fuzzy.ipynb`** (environment `emotion-fuzzy`) reads that file and trains and evaluates the pyFUME fuzzy classifier (sections 4.3–4.9).

## Setup

Create both environments with conda:

```bash
conda env create -f environment.yml          # main notebook and demo
conda env create -f environment-fuzzy.yml    # task4_fuzzy.ipynb only
```

In Jupyter or VS Code, select the kernel `emotion-detection` for the main notebook and `emotion-fuzzy` for `task4_fuzzy.ipynb`.

The datasets are not included in the repository. Place them as follows:

```
data/fer2013.csv                       # FER2013, Kaggle version as a single CSV
data/KDEF_and_AKDEF/KDEF/<subject>/    # extracted KDEF_and_AKDEF.zip
```

Extracted features are cached in `data/feature_cache/`, so later runs are faster.

## Running the notebooks

Run from the repository folder, in this order:

1. `emotion_detection_group11.ipynb`, Tasks 1–3 and sections 4.1–4.2.
2. `task4_fuzzy.ipynb` (needs the features from step 1).
3. `emotion_detection_group11.ipynb`, Tasks 5 and 6. Section 5.4 uses the geometric features defined in section 4.2, so run 4.2 in the same session first.

Section 5.5 (hyperparameter tuning) takes about 10 minutes; the other sections take a few minutes each.

## Webcam demo

```bash
conda activate emotion-detection
python demo.py                  # default webcam; press q to quit
python demo.py --camera 1       # another camera
python demo.py --no-smoothing   # classify every frame independently
```

The demo detects the largest face with the Haar cascade, crops it 10% tighter than the detection box to match FER2013's framing, and shows the predicted emotion with its probability. The probabilities are averaged over recent frames so the label does not flicker. Section 6 of the main notebook explains these choices.

## Reproducibility

All experiments use a fixed seed (42). Results are reproducible with the same software environment. On a different machine or operating system, floating-point differences can change a few predictions (in our runs, at most a few test images), so scores may differ slightly in the last decimal places.
