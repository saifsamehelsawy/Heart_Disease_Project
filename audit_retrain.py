from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


root = Path(r"d:\MNU_3Y_2\ML\Final_project\Heart_Disease_Project")
df = pd.read_csv(root / "data" / "heart_dataset.csv").drop_duplicates().copy()

print("=== 1) TARGET INSPECTION ===")
print("unique target values:", sorted(df["target"].unique().tolist()))
print("target distribution:")
print(df["target"].value_counts().sort_index())
print("\nSample rows with target:")
print(
    df[
        [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "thalach",
            "exang",
            "oldpeak",
            "ca",
            "thal",
            "target",
        ]
    ]
    .sample(8, random_state=42)
    .to_string(index=False)
)

print("\n=== 2) CORRELATIONS WITH TARGET ===")
key_feats = ["exang", "oldpeak", "ca", "thal", "cp", "thalach"]
corr = (
    df[key_feats + ["target"]]
    .corr(numeric_only=True)["target"]
    .drop("target")
    .sort_values(ascending=False)
)
print(corr.to_string())

print("\n=== 3) GROUPED MEANS BY TARGET ===")
print(df.groupby("target").mean(numeric_only=True).to_string())

print("\n=== 4) REAL ROW SAMPLES FROM EACH CLASS ===")
print("Rows where target=1:")
print(
    df[df["target"] == 1][
        [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "target",
        ]
    ]
    .head(5)
    .to_string(index=False)
)
print("\nRows where target=0:")
print(
    df[df["target"] == 0][
        [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "target",
        ]
    ]
    .head(5)
    .to_string(index=False)
)

df_train = df.copy()
df_train["target"] = 1 - df_train["target"]
print("\n=== 5) LABEL INVERSION (FORCED) ===")
print("Applied label inversion: True")

feats = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

X = df_train[feats].copy()
y = df_train["target"].copy()

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, class_weight="balanced", random_state=42)),
    ]
)

grid = GridSearchCV(
    pipe,
    {
        "svm__kernel": ["linear", "rbf", "poly"],
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": [0.001, 0.01, 0.1, 1],
    },
    scoring="f1",
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    n_jobs=-1,
    refit=True,
    verbose=0,
)

print("\n=== 6) RETRAINING ===")
grid.fit(Xtr, ytr)
best = grid.best_estimator_
pred = best.predict(Xte)

print("Best Params:", grid.best_params_)
print("Best CV F1:", round(grid.best_score_, 4))
print("Accuracy:", round(accuracy_score(yte, pred), 4))
print("Precision:", round(precision_score(yte, pred, zero_division=0), 4))
print("Recall:", round(recall_score(yte, pred, zero_division=0), 4))
print("F1:", round(f1_score(yte, pred, zero_division=0), 4))
print("Confusion Matrix:", confusion_matrix(yte, pred).tolist())
print("Classification Report:")
print(classification_report(yte, pred, digits=4, zero_division=0))

healthy = pd.DataFrame(
    [[24, 0, 0, 108, 135, 0, 0, 190, 0, 0, 2, 0, 1]], columns=feats
)
risky = pd.DataFrame(
    [[76, 1, 3, 200, 360, 1, 2, 80, 1, 5.2, 0, 4, 3]], columns=feats
)

ph = int(best.predict(healthy)[0])
pr = int(best.predict(risky)[0])
prob_h = float(best.predict_proba(healthy)[0][1])
prob_r = float(best.predict_proba(risky)[0][1])

print("\n=== 7) MANUAL CHECKS AFTER FIX ===")
print("Healthy -> pred:", ph, "prob(class=1):", round(prob_h, 6))
print("Risky   -> pred:", pr, "prob(class=1):", round(prob_r, 6))
print("Interpretation after fix: class 1 = No Heart Disease, class 0 = Heart Disease")

(root / "models").mkdir(parents=True, exist_ok=True)
joblib.dump(best, root / "pipeline.pkl")
joblib.dump(best, root / "best_model.pkl")
joblib.dump(best, root / "models" / "pipeline.pkl")
joblib.dump(best, root / "models" / "best_model.pkl")

print("\nSaved updated models to root and models/")
