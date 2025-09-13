# shap_utils.py
import numpy as np
import joblib
import shap
import pandas as pd
import os

MODEL_DIR = "models"
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "regressor.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
LABELENC_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    reg = joblib.load(REGRESSOR_PATH)
    clf = joblib.load(CLASSIFIER_PATH)
    le = joblib.load(LABELENC_PATH)
    return preprocessor, reg, clf, le

def shap_explain_regression(single_input: pd.DataFrame):
    """
    Returns a dictionary with SHAP values and base value for regression prediction.
    """
    preprocessor, reg, clf, le = load_artifacts()
    X = preprocessor.transform(single_input)
    # use TreeExplainer for RF
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer.shap_values(X)
    # Return as list of (feature_name, shap_value)
    # Need to reconstruct feature names after preprocessing
    # For simplicity return shap_values array and let caller map names if needed
    return {
        "shap_values": shap_values.tolist(),
        "base_value": float(explainer.expected_value)
    }

def shap_explain_classification(single_input: pd.DataFrame):
    preprocessor, reg, clf, le = load_artifacts()
    X = preprocessor.transform(single_input)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    classes = list(le.classes_)
    return {
        "shap_values": [sv.tolist() for sv in shap_values],
        "classes": classes,
        "base_values": [float(bv) for bv in explainer.expected_value] if isinstance(explainer.expected_value, (list, tuple, np.ndarray)) else [float(explainer.expected_value)]
    }
