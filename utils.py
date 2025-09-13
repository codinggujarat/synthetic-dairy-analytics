"""
Utility functions: feature lists, preprocessing pipelines, and helper functions
for the cattle monitoring project.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------
# Define features (MATCHING data_generator.py)
# ---------------------

NUMERICAL_FEATURES = [
    "age",
    "weight",
    "parity",
    "hist_yield_avg7",
    "feed_quality",
    "feed_qty_kg",
    "walking_km",
    "grazing_h",
    "rumination_h",
    "resting_h",
    "body_temp",
    "heart_rate",
    "ambient_temp",
    "humidity",
    "housing_score",
    "vaccinations_up_to_date",  # ✅ added to match dataset
    "disease_history_count",
]

CATEGORICAL_FEATURES = [
    "breed",
    "lactation_stage",
    "feed_type",
    "season",
]

TARGET_REG = "milk_yield"
TARGET_CLASS = "disease_label"


def build_preprocessor():
    """Build a ColumnTransformer that preprocesses numerical and categorical features."""
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def prepare_single_input(preprocessor, input_dict):
    """Convert a single input dictionary to a transformed feature array using fitted preprocessor."""
    df = pd.DataFrame([input_dict])
    return preprocessor.transform(df)


def disease_recommendations(disease_label: str) -> str:
    """Return actionable recommendations based on predicted disease risk."""
    recs = {
        "none": "No health issue detected. Maintain regular monitoring and proper feeding.",
        "mastitis": (
            "Mastitis risk detected. Improve udder hygiene, "
            "check milk for clots, consult vet for early treatment."
        ),
        "digestive": (
            "Digestive disorder risk detected. Check feed quality, "
            "avoid sudden diet changes, provide probiotics."
        ),
        "mineral_deficiency": (
            "Mineral deficiency risk detected. Supplement diet with Ca, P, and trace minerals. "
            "Ensure balanced ration formulation."
        ),
    }
    return recs.get(disease_label, "General health check recommended.")
