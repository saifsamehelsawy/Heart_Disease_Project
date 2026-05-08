from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FEATURE_ORDER = [
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


def handle_outliers_iqr(X_df, zscore_threshold=None):
    """Keep compatibility with training-time FunctionTransformer."""
    X = X_df.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        X[col] = X[col].clip(lower=low, upper=high)

    if zscore_threshold is not None:
        for col in numeric_cols:
            mean = X[col].mean()
            std = X[col].std(ddof=0)
            if std == 0:
                continue
            z = (X[col] - mean) / std
            X.loc[z > zscore_threshold, col] = mean + zscore_threshold * std
            X.loc[z < -zscore_threshold, col] = mean - zscore_threshold * std

    return X


def add_derived_features(X_df):
    """Keep compatibility with training-time FunctionTransformer."""
    X = X_df.copy()

    if {"age", "chol", "thalach", "trestbps"}.issubset(X.columns):
        age_bins = pd.cut(
            X["age"],
            bins=[0, 40, 50, 60, 120],
            labels=[0, 1, 2, 3],
            include_lowest=True,
        ).astype(int)
        chol_bins = pd.cut(
            X["chol"],
            bins=[0, 200, 240, 1000],
            labels=[0, 1, 2],
            include_lowest=True,
        ).astype(int)
        max_theoretical_hr = (220 - X["age"]).clip(lower=1)
        hr_ratio = X["thalach"] / max_theoretical_hr

        X["age_bin"] = age_bins
        X["chol_cat"] = chol_bins
        X["hr_ratio"] = hr_ratio
        X["chol_age_ratio"] = X["chol"] / X["age"].clip(lower=1)
        X["bp_hr_ratio"] = X["trestbps"] / X["thalach"].clip(lower=1)

    return X


def load_artifacts():
    pipeline_path = MODELS_DIR / "pipeline.pkl"
    best_model_path = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"

    if pipeline_path.exists():
        try:
            return {
                "mode": "pipeline",
                "model": joblib.load(pipeline_path),
                "scaler": None,
                "source": "pipeline.pkl",
            }
        except Exception:
            # Fall back to best_model/scaler if pipeline was serialized with notebook-only state.
            pass

    if best_model_path.exists():
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        return {
            "mode": "model_only",
            "model": joblib.load(best_model_path),
            "scaler": scaler,
            "source": "best_model.pkl",
        }

    raise FileNotFoundError(
        "No model found. Expected models/pipeline.pkl or models/best_model.pkl."
    )


ARTIFACTS = load_artifacts()


def parse_input(form_data):
    values = []
    for feature in FEATURE_ORDER:
        raw_value = form_data.get(feature, "").strip()
        if raw_value == "":
            raise ValueError(f"Missing value for '{feature}'.")
        values.append(float(raw_value))
    return values


def predict_with_artifacts(features):
    x_df = pd.DataFrame([features], columns=FEATURE_ORDER)
    x_np = np.array([features], dtype=float)

    model = ARTIFACTS["model"]
    scaler = ARTIFACTS["scaler"]

    if ARTIFACTS["mode"] == "pipeline":
        prediction = int(model.predict(x_df)[0])
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(x_df)[0][1])
        else:
            probability = None
        return prediction, probability

    if scaler is not None:
        x_np = scaler.transform(x_np)

    prediction = int(model.predict(x_np)[0])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(x_np)[0][1])
    else:
        probability = None
    return prediction, probability


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", feature_order=FEATURE_ORDER, model_source=ARTIFACTS["source"])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = parse_input(request.form)
        prediction, probability = predict_with_artifacts(features)

        if prediction == 1:
            result = "✅ No Heart Disease"
            result_class = "result-safe"
        else:
            result = "❌ Heart Disease Detected"
            result_class = "result-danger"

        probability_text = None
        if probability is not None:
            probability_text = f"{probability * 100:.2f}%"

        return render_template(
            "index.html",
            feature_order=FEATURE_ORDER,
            model_source=ARTIFACTS["source"],
            result=result,
            result_class=result_class,
            probability=probability_text,
            form_data=request.form,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            feature_order=FEATURE_ORDER,
            model_source=ARTIFACTS["source"],
            error=f"Prediction failed: {exc}",
            form_data=request.form,
        ), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
