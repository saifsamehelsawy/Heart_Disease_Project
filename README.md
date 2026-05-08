# Heart Disease Predictor

A complete machine learning project for heart disease risk prediction, including:
- data analysis and model training notebook,
- saved trained artifacts,
- and a production-style Flask web interface with a modern UI.

---

## Project Overview

This project predicts cardiovascular disease risk from 13 clinical features (age, chest pain type, ECG indicators, etc.) using a trained classification model.

The repository covers the full workflow:
- data exploration and cleaning,
- feature preparation and model training,
- model selection/tuning,
- exporting model artifacts,
- deploying inference through a Flask app.

---

## Features

- End-to-end ML pipeline (EDA -> training -> evaluation -> deployment)
- Trained model artifacts ready for inference
- Web UI for entering patient data and getting instant prediction
- Probability output (when supported by the loaded model)
- Professional responsive interface for desktop and mobile

---

## Tech Stack

- Python
- Flask
- scikit-learn
- pandas, numpy
- joblib
- Bootstrap + custom CSS

---

## Repository Structure

```text
Heart_Disease_Project/
├── app.py                      # Flask app (inference + routes)
├── audit_retrain.py            # Retraining/audit script
├── requirements.txt            # Python dependencies
├── README.md
│
├── data/
│   └── heart_dataset.csv       # Dataset
│
├── models/
│   ├── pipeline.pkl            # Preferred artifact (loaded first)
│   ├── best_model.pkl          # Fallback model
│   └── scaler.pkl              # Optional scaler for model_only mode
│
├── notebook/
│   └── project.ipynb           # Full ML notebook
│
├── templates/
│   └── index.html              # Web page template
│
├── static/
│   ├── style.css               # UI styling
│   ├── logo.png
│   └── logo_clean.png
│
├── plots/                      # Saved EDA/training visuals
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_age_chol_histograms.png
│   ├── 04_boxplots_outliers.png
│   ├── 05_feature_vs_target.png
│   ├── 06_confusion_matrices.png
│   ├── 07_model_comparison.png
│   ├── 08_hyperparameter_tuning.png
│   └── 09_feature_importance.png
│
└── mlruns/                     # MLflow tracking artifacts
```

---

## Input Features (Model)

The Flask app expects these 13 features in this order:

1. `age`
2. `sex`
3. `cp`
4. `trestbps`
5. `chol`
6. `fbs`
7. `restecg`
8. `thalach`
9. `exang`
10. `oldpeak`
11. `slope`
12. `ca`
13. `thal`

---

## Prediction Output

The web app returns:
- predicted class label,
- and prediction probability (if `predict_proba` is available).

Current app interpretation:
- `1` -> **No Heart Disease**
- `0` -> **Heart Disease Detected**

---

## Setup and Run

### 1) Clone repository

```bash
git clone <your-repo-url>
cd Heart_Disease_Project
```

### 2) Create virtual environment (recommended)

```bash
python -m venv .venv
```

Activate:
- Windows (PowerShell):  
  ```bash
  .venv\Scripts\Activate.ps1
  ```
- Linux/macOS:  
  ```bash
  source .venv/bin/activate
  ```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run Flask app

```bash
python app.py
```

Open in browser:
- [http://localhost:5000](http://localhost:5000)

---

## Notebook / Training Workflow

To explore the full analysis and training process:

```bash
jupyter notebook notebook/project.ipynb
```

Optional retraining/audit script:

```bash
python audit_retrain.py
```

---

## Notes

- `app.py` tries loading `models/pipeline.pkl` first.
- If unavailable, it falls back to `models/best_model.pkl` (+ optional `models/scaler.pkl`).
- Ensure model files exist before running the web app.

---

## Future Improvements

- Add model explainability (SHAP/LIME)
- Add input validation and domain constraints
- Add unit/integration tests
- Dockerize the app for easier deployment
- Add CI pipeline for lint/test/build

---

## Author

Developed as a final machine learning project for heart disease prediction and deployment.
