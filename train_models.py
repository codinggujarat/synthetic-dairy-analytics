# train_models.py
"""
Train ML models for milk yield prediction (regression)
and disease risk prediction (classification).
Saves trained models + preprocessor + label encoder into ./models/
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report

from utils import build_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_REG, TARGET_CLASS

MODEL_DIR = "models"
DATA_PATH = "data/cattle_synthetic.csv"


def load_data(path=DATA_PATH):
    """Load dataset and ensure correct dtypes."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run data_generator.py first to create synthetic data."
        )
    df = pd.read_csv(path)
    # Ensure correct type for vaccination flag
    if "vaccinations_up_to_date" in df.columns:
        df["vaccinations_up_to_date"] = df["vaccinations_up_to_date"].astype(int)
    return df


def train_and_save(n_estimators=150, random_state=42):
    """Train regressor + classifier, save artifacts, and print evaluation metrics."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()

    # Features & targets
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_reg = df[TARGET_REG]
    y_clf = df[TARGET_CLASS]

    # Preprocessing
    print("[INFO] Building and fitting preprocessor...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X)
    X_trans = preprocessor.transform(X)

    # Encode classification target
    le = LabelEncoder()
    y_clf_enc = le.fit_transform(y_clf)

    # Train/test split
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X_trans, y_reg, test_size=0.2, random_state=random_state
    )
    _, _, y_train_clf, y_test_clf = train_test_split(
        X_trans, y_clf_enc, test_size=0.2, random_state=random_state
    )

    # Train models
    print("[INFO] Training RandomForest Regressor...")
    rfr = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=-1, random_state=random_state
    )
    rfr.fit(X_train, y_train_reg)

    print("[INFO] Training RandomForest Classifier...")
    rfc = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )
    rfc.fit(X_train, y_train_clf)

    # Evaluate models
    print("\n[METRICS] Regression Performance:")
    y_pred_reg = rfr.predict(X_test)
    print(f"  MAE  : {mean_absolute_error(y_test_reg, y_pred_reg):.3f}")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.3f}")
    print(f"  R²   : {r2_score(y_test_reg, y_pred_reg):.3f}")

    print("\n[METRICS] Classification Report:")
    y_pred_clf = rfc.predict(X_test)
    print(classification_report(y_test_clf, y_pred_clf, target_names=le.classes_))

    # Save artifacts
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor_joblib.pkl"))
    joblib.dump(rfr, os.path.join(MODEL_DIR, "regressor_joblib.pkl"))
    joblib.dump(rfc, os.path.join(MODEL_DIR, "classifier_joblib.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "labelencoder_joblib.pkl"))
    print(f"[SUCCESS] Models and preprocessor saved to {MODEL_DIR}/")

    return {
        "preprocessor": preprocessor,
        "regressor": rfr,
        "classifier": rfc,
        "label_encoder": le,
    }


if __name__ == "__main__":
    train_and_save(n_estimators=150)
