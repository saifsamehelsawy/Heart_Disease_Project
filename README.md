# 🫀 Heart Disease Prediction using Machine Learning

A complete end-to-end classification project to predict whether a patient has heart disease based on medical attributes.

---

## 📁 Project Structure

```
Heart_Disease_Project/
│
├── data/
│   └── heart.csv               # Dataset (303 patients, 14 features)
│
├── notebook/
│   └── project.ipynb           # Main Jupyter notebook (fully executed)
│
├── plots/                      # All saved visualizations
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
├── mlruns/                     # MLflow tracking data (auto-generated)
├── requirements.txt
└── README.md
```

---

## 🎯 Project Goal

Binary classification: predict whether a patient **has heart disease (1)** or **does not (0)** based on 13 medical attributes.

---

## 📊 Dataset Features

| Feature     | Description                                      |
|-------------|--------------------------------------------------|
| age         | Age of the patient                               |
| sex         | Sex (1 = male, 0 = female)                       |
| cp          | Chest pain type (0–3)                            |
| trestbps    | Resting blood pressure (mm Hg)                   |
| chol        | Serum cholesterol (mg/dl)                        |
| fbs         | Fasting blood sugar > 120 mg/dl (1 = true)       |
| restecg     | Resting ECG results (0–2)                        |
| thalach     | Maximum heart rate achieved                      |
| exang       | Exercise-induced angina (1 = yes)                |
| oldpeak     | ST depression induced by exercise                |
| slope       | Slope of the peak exercise ST segment            |
| ca          | Number of major vessels colored by fluoroscopy   |
| thal        | Thalassemia type (0–3)                           |
| **target**  | **Heart disease (1 = yes, 0 = no)**              |

---

## 🤖 Models Trained

| Model              | Tuning         |
|--------------------|----------------|
| K-Nearest Neighbors | GridSearchCV  |
| Support Vector Machine | GridSearchCV |
| Decision Tree      | Default        |
| Random Forest      | GridSearchCV   |

---

## 📈 Evaluation Metrics

Each model is evaluated on:
- **Accuracy** — overall correct predictions
- **Precision** — true positives / predicted positives
- **Recall** — true positives / actual positives
- **F1-Score** — harmonic mean of precision & recall
- **Confusion Matrix** — visual breakdown of predictions

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter
```bash
jupyter notebook notebook/project.ipynb
```

### 3. Run All Cells
Use **Kernel → Restart & Run All** to execute the full pipeline.

### 4. View MLflow Dashboard
```bash
cd Heart_Disease_Project
mlflow ui
```
Then open: [http://localhost:5000](http://localhost:5000)

---

## 📦 Libraries Used

- **pandas** — data loading & manipulation
- **numpy** — numerical operations
- **matplotlib / seaborn** — data visualization
- **scikit-learn** — ML models, preprocessing, evaluation
- **mlflow** — experiment tracking & model registry

---

## 📌 Notebook Sections

1. Imports & Setup
2. Data Loading (`head`, `info`, `describe`)
3. Data Cleaning (missing values, duplicates, type validation)
4. Exploratory Data Analysis (5 visualizations)
5. Data Preprocessing (split + StandardScaler)
6. Model Training (KNN, SVM, Decision Tree, Random Forest)
7. Confusion Matrices
8. Model Comparison Table & Chart
9. Hyperparameter Tuning (GridSearchCV)
10. Feature Importance
11. MLflow Tracking & Best Run Query
12. Final Summary & Conclusion






py -m mlflow ui   