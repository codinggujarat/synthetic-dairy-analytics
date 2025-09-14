# app.py
import os
import json
import io
import base64
from flask import request, jsonify
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from flask_login import LoginManager, login_required, current_user
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import joblib
import matplotlib.pyplot as plt
import traceback
import google.generativeai as genai
import requests

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generate"

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"


# Extensions (must exist in your project)
from extensions import db, mail

# Auth blueprint and User (must exist)
from auth import auth_bp, User

# Models (must exist)
from models import AnimalRecord

# Utilities (must exist)
from shap_utils import shap_explain_regression, shap_explain_classification
from utils import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, build_preprocessor, disease_recommendations

# ML training import (must exist)
from train_models import train_and_save


# ------------------------------
# App Configuration
# ------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-secret-key")

basedir = os.path.abspath(os.path.dirname(__file__))
instance_dir = os.path.join(basedir, "instance")
os.makedirs(instance_dir, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_dir, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Mail config (optional)
app.config['MAIL_SERVER'] = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
app.config['MAIL_PORT'] = int(os.environ.get("MAIL_PORT", 587))
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get("MAIL_USERNAME")
app.config['MAIL_PASSWORD'] = os.environ.get("MAIL_PASSWORD")
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get("MAIL_DEFAULT_SENDER", app.config['MAIL_USERNAME'])
# Configure Gemini API (Set your API key as ENV variable or hardcode for testing)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ------------------------------
# Initialize extensions
# ------------------------------
db.init_app(app)
mail.init_app(app)

# ------------------------------
# Login Manager
# ------------------------------
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register auth blueprint
app.register_blueprint(auth_bp, url_prefix="/auth")

# ------------------------------
# Model Artifacts
# ------------------------------
MODEL_DIR = os.path.join(basedir, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')
REGRESSOR_PATH = os.path.join(MODEL_DIR, 'regressor.pkl')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'classifier.pkl')
LABELENC_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# ------------------------------
# Ensure tables exist
# ------------------------------
with app.app_context():
    db.create_all()

# ------------------------------
# Helpers
# ------------------------------
def load_artifacts():
    """
    Try to load model artifacts. Return dict with keys and 'loaded' flag.
    """
    artifacts = {'loaded': False}
    try:
        artifacts['preprocessor'] = joblib.load(PREPROCESSOR_PATH)
        artifacts['regressor'] = joblib.load(REGRESSOR_PATH)
        artifacts['classifier'] = joblib.load(CLASSIFIER_PATH)
        artifacts['label_encoder'] = joblib.load(LABELENC_PATH)
        artifacts['loaded'] = True
    except Exception as e:
        # Keep artifacts partial or none; log to console for debugging
        print("load_artifacts: could not load all artifacts:", e)
        # Optionally print stacktrace in debug only
        traceback.print_exc()
    return artifacts

artifacts = load_artifacts()

def plot_hist_vs_pred(hist_avg, pred):
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(['Hist Avg (7d)', 'Predicted'], [hist_avg, pred])
    ax.set_ylabel('L/day')
    ax.set_title('Milk Yield: Historical vs Predicted')
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def record_to_feature_dict(rec):
    """
    Safely convert an AnimalRecord instance into a dict with expected features.
    Uses NUMERICAL_FEATURES + CATEGORICAL_FEATURES to build row; missing attrs -> None/0.
    """
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    row = {}
    for c in feature_cols:
        # read attribute if exists, otherwise attempt key from to_series (if implemented)
        val = getattr(rec, c, None)
        if val is None:
            # try to pull from rec.to_series if defined
            try:
                series = rec.to_series()
                val = series.get(c, None)
            except Exception:
                val = None
        # numeric columns: ensure floats/ints where appropriate
        if c in NUMERICAL_FEATURES:
            try:
                row[c] = float(val) if val is not None else 0.0
            except Exception:
                row[c] = 0.0
        else:
            # categorical keep as string, but handle None
            row[c] = str(val) if val is not None else ""
    return row

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
@login_required
def index():
    # Fetch last 8 records from database
    records = AnimalRecord.query.order_by(AnimalRecord.id.desc()).limit(8).all()
    plot_data = None
    shap_values = None

    if records:
        last = records[0]
        plot_data = {
            "historical": getattr(last, 'hist_yield_avg7', 0) or 0,
            "predicted": getattr(last, 'predicted_milk_yield', 0) or 0
        }

        if hasattr(last, 'shap_values') and last.shap_values:
            shap_values = last.shap_values
            if not isinstance(shap_values, list):
                try:
                    shap_values = shap_values.tolist()
                except Exception:
                    shap_values = None

    # Convert records to serializable format
    records_json = []
    for r in records:
        created_at_str = ''
        if hasattr(r, 'created_at') and r.created_at:
            created_at_str = r.created_at.strftime('%Y-%m-%d %H:%M:%S') if not isinstance(r.created_at, str) else r.created_at

        records_json.append({
            'id': r.id,
            'breed': getattr(r, 'breed', 'Unknown'),
            'age': getattr(r, 'age', 0),
            'weight': getattr(r, 'weight', 0),
            'lactation_stage': getattr(r, 'lactation_stage', 'Unknown'),
            'parity': getattr(r, 'parity', 0),
            'feed_type': getattr(r, 'feed_type', 'Unknown'),
            'feed_quality': getattr(r, 'feed_quality', 'Unknown'),
            'feed_qty_kg': getattr(r, 'feed_qty_kg', 0),
            'rumination_h': getattr(r, 'rumination_h', 0),
            'resting_h': getattr(r, 'resting_h', 0),
            'health_score': getattr(r, 'health_score', 0),
            'vaccinations_up_to_date': getattr(r, 'vaccinations_up_to_date', False),
            'disease_label': getattr(r, 'disease_label', 'None'),
            'milk_yield': getattr(r, 'milk_yield', 0),
            'feed_efficiency': getattr(r, 'feed_efficiency', 0),
            'predicted_milk_yield': getattr(r, 'predicted_milk_yield', 0),
            'predicted_disease': getattr(r, 'predicted_disease', 'None'),
            'created_at': created_at_str
        })

    # --- CSV Data Section ---
    total_animals = avg_milk_yield = healthy_animals = unhealthy_animals = 0
    feed_efficiency = avg_weight = avg_temp = avg_heart_rate = vaccination_rate = 0
    breed_counts = {}

    try:
        data_path = os.path.join(basedir, "data", "cattle_synthetic.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)

            if not df.empty:
                total_animals = len(df)
                avg_milk_yield = round(df["milk_yield"].mean(), 2)
                healthy_animals = len(df[df["disease_label"] == 0])
                health_score = len(df[df["health_score"] == 0])
                unhealthy_animals = len(df[df["disease_label"] == 1])
                feed_efficiency = round(df["milk_yield"].sum() / df["feed_qty_kg"].sum(), 2)  # L/kg
                avg_weight = round(df["weight"].mean(), 2)
                # Optional: handle missing body_temp or heart_rate columns
                avg_temp = round(df["body_temp"].mean(), 2) if "body_temp" in df.columns else 0
                avg_heart_rate = round(df["heart_rate"].mean(), 2) if "heart_rate" in df.columns else 0
                vaccination_rate = round(df["vaccinations_up_to_date"].mean() * 100, 2) if "vaccinations_up_to_date" in df.columns else 0
                breed_counts = df["breed"].value_counts().to_dict()
        else:
            flash("No CSV found at data/cattle_synthetic.csv.", "warning")

    except Exception as e:
        traceback.print_exc()
        flash(f"Failed to load CSV: {e}", "danger")

    # --- Render Template ---
    return render_template(
        "index.html",
        total_animals=total_animals,
        avg_milk_yield=avg_milk_yield,
        healthy_animals=healthy_animals,
        unhealthy_animals=unhealthy_animals,
        feed_efficiency=feed_efficiency,
        avg_weight=avg_weight,
        avg_temp=avg_temp,
        health_score=health_score,
        avg_heart_rate=avg_heart_rate,
        vaccination_rate=vaccination_rate,
        breed_counts=breed_counts,
        records=records,
        records_json=records_json,
        plot_data=plot_data,
        shap_values=shap_values,
        artifacts_loaded=artifacts.get('loaded', False)
    )


@app.route('/predict-animal', methods=['POST'])
@login_required
def predict_animal():
    """Handles form submission, prediction, saving, and shows top 10 recent cow cards."""
    if not artifacts.get('loaded'):
        flash("Models not loaded, train manually first.", "warning")
        return redirect(url_for("input_form"))

    form = request.form
    data = {}
    for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        val = form.get(f, "")
        if f in NUMERICAL_FEATURES:
            try:
                data[f] = float(val)
            except Exception:
                data[f] = 0.0
        else:
            data[f] = val

    try:
        data['vaccinations_up_to_date'] = int(form.get('vaccinations_up_to_date', 1))
        data['disease_history_count'] = int(form.get('disease_history_count', 0))
    except Exception:
        pass

    df = pd.DataFrame([data])
    try:
        preproc = artifacts['preprocessor']
        reg = artifacts['regressor']
        clf = artifacts['classifier']
        le = artifacts['label_encoder']

        X_t = preproc.transform(df)
        milk_pred = float(reg.predict(X_t)[0])
        probs = clf.predict_proba(X_t)[0]
        classes = list(le.classes_)
        top_class = classes[int(np.argmax(probs))]
        proba_map = dict(zip(classes, probs))

        # Save record
        record = AnimalRecord(
            **data,
            predicted_milk_yield=milk_pred,
            predicted_disease=top_class,
            disease_probs=json.dumps(proba_map),
            created_at=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        )
        db.session.add(record)
        db.session.commit()

        flash(f"✅ Prediction saved! Yield: {milk_pred:.2f} L/day | Condition: {top_class}", "success")

    except Exception as e:
        db.session.rollback()
        flash(f"Prediction failed: {e}", "danger")
        traceback.print_exc()
        return redirect(url_for("recent_predictions"))

    # --- Fetch top 10 recent records to render cards ---
    recent_records = AnimalRecord.query.order_by(AnimalRecord.id.desc()).limit(10).all()
    records_json = []
    for r in recent_records:
        try:
            disease_probs = json.loads(r.disease_probs) if r.disease_probs else {}
        except Exception:
            disease_probs = {}
        records_json.append({
            "id": r.id,
            "breed": getattr(r, 'breed', 'Unknown'),
            "age": getattr(r, 'age', 0),
            "weight": getattr(r, 'weight', 0),
            "lactation_stage": getattr(r, 'lactation_stage', 'Unknown'),
            "parity": getattr(r, 'parity', 0),
            "predicted_milk_yield": getattr(r, 'predicted_milk_yield', 0),
            "predicted_disease": getattr(r, 'predicted_disease', 'Unknown'),
            "disease_probs": disease_probs,
            "created_at": getattr(r, 'created_at', '')
        })

    # Render template with top 10 cards
    return render_template("recent_predictions.html", records_json=records_json)

@app.route('/predict', methods=['POST'])
@login_required
def predict_route():
    if not artifacts.get('loaded'):
        flash("Models not loaded, train manually first.", "warning")
        return redirect(url_for('index'))

    form = request.form
    data = {}
    # Collect features safely
    for f in NUMERICAL_FEATURES + CATEGORICAL_FEATURES:
        val = form.get(f, "")
        if f in NUMERICAL_FEATURES:
            try:
                data[f] = float(val)
            except Exception:
                data[f] = 0.0
        else:
            data[f] = val

    # Cast special fields
    try:
        data['vaccinations_up_to_date'] = int(
            form.get('vaccinations_up_to_date', data.get('vaccinations_up_to_date', 1))
        )
    except Exception:
        data['vaccinations_up_to_date'] = 1

    try:
        data['disease_history_count'] = int(
            form.get('disease_history_count', data.get('disease_history_count', 0))
        )
    except Exception:
        data['disease_history_count'] = 0

    df = pd.DataFrame([data])

    try:
        preproc = artifacts['preprocessor']
        reg = artifacts['regressor']
        clf = artifacts['classifier']
        le = artifacts['label_encoder']
    except Exception:
        flash("Model artifacts not available. Train models first.", "warning")
        return redirect(url_for('index'))

    try:
        # Transform & predict
        X_t = preproc.transform(df)
        milk_pred = float(reg.predict(X_t)[0])
        probs = clf.predict_proba(X_t)[0]
        classes = list(le.classes_)
        proba_map = {cls: float(p) for cls, p in zip(classes, probs)}
        top_class = classes[int(np.argmax(probs))]

        # Optional: SHAP value calculation (if explainer is in artifacts)
        shap_values = None
        if artifacts.get("explainer"):
            try:
                explainer = artifacts["explainer"]
                shap_values_raw = explainer.shap_values(X_t)
                if isinstance(shap_values_raw, list):  # some explainers return list for multiclass
                    shap_values_raw = shap_values_raw[0]
                feature_names = preproc.get_feature_names_out()
                shap_values = list(zip(feature_names, shap_values_raw[0]))
            except Exception as e:
                print(f"[WARN] Could not compute SHAP values: {e}")

    except Exception as e:
        flash(f"Prediction failed: {e}", "danger")
        traceback.print_exc()
        return redirect(url_for('index'))

    # Save to DB
    try:
        record = AnimalRecord(
            **data,
            predicted_milk_yield=milk_pred,
            predicted_disease=top_class,
            disease_probs=json.dumps(proba_map),  # store as JSON
            created_at=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        )

        # If you added shap_values column in your DB, save it
        if shap_values:
            try:
                record.shap_values = json.dumps(shap_values)
            except Exception:
                pass

        db.session.add(record)
        db.session.commit()

        print(f"[DEBUG] Saved Record ID={record.id}, Disease={top_class}, Yield={milk_pred:.2f}")
    except Exception as e:
        db.session.rollback()
        flash(f"Failed to save prediction: {e}", "warning")
        traceback.print_exc()

    flash(f"Predicted milk yield: {milk_pred:.2f} L/day | Condition: {top_class}", "success")
    return redirect(url_for('index'))
# ------------------------------
# Training function (moved from train_models.py)
# ------------------------------
def train_and_save(df=None, n_estimators=100):
    if df is None:
        raise ValueError("No training data provided")

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import joblib

    # Separate features and targets
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    # ⚠️ Make sure these columns exist in your CSV
    if "milk_yield" not in df.columns:
        raise ValueError("CSV must contain 'milk_yield' column for regression target.")
    if "disease_label" not in df.columns:
        raise ValueError("CSV must contain 'disease_label' column for classification target.")

    y_reg = df["milk_yield"]
    y_clf = df["disease_label"]

    # Preprocessor
    preprocessor = build_preprocessor()
    X_t = preprocessor.fit_transform(X)

    # Train models
    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    reg.fit(X_t, y_reg)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    le = LabelEncoder()
    y_clf_enc = le.fit_transform(y_clf)
    clf.fit(X_t, y_clf_enc)

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(reg, REGRESSOR_PATH)
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(le, LABELENC_PATH)

# ------------------------------
# Manual training route
# ------------------------------
@app.route('/train')
@login_required
def train_manual():
    try:
        data_path = os.path.join(basedir, "data", "cattle_synthetic.csv")
        if not os.path.exists(data_path):
            flash("No CSV found at data/cattle_synthetic.csv. Please add it first.", "warning")
            return redirect(url_for('index'))

        df = pd.read_csv(data_path)

        # Verify columns match expected features
        expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        missing = [c for c in expected_cols if c not in df.columns]

        if missing:
            flash(f"Cannot train: Missing columns in CSV: {', '.join(missing)}", "danger")
            return redirect(url_for('index'))

        if len(df) < 2:
            flash("Not enough data to train. Need at least 2 rows in CSV.", "warning")
            return redirect(url_for('index'))

        train_and_save(df=df, n_estimators=150)

        global artifacts
        artifacts = load_artifacts()
        flash(f"Models trained successfully on {len(df)} records!", "success")

    except Exception as e:
        flash(f"Training failed: {e}", "danger")

    return redirect(url_for('index'))
# ------------------------------
# API Routes
# ------------------------------
@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "JSON body required"}), 400
    df = pd.DataFrame([payload])
    if not artifacts.get('loaded'):
        return jsonify({"error": "Models not loaded"}), 400

    try:
        preproc = artifacts['preprocessor']
        reg = artifacts['regressor']
        clf = artifacts['classifier']
        le = artifacts['label_encoder']
    except Exception:
        return jsonify({"error": "Model artifacts not available"}), 400

    try:
        X_t = preproc.transform(df)
        milk_pred = float(reg.predict(X_t)[0])
        probs = clf.predict_proba(X_t)[0]
        classes = list(le.classes_)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({
        "milk_yield": round(milk_pred, 2),
        "disease_prediction": classes[int(np.argmax(probs))],
        "disease_probabilities": dict(zip(classes, [float(p) for p in probs]))
    })

@app.route('/api/ingest', methods=['POST'])
def api_ingest():
    """
    Accepts CSV upload or JSON list of rows.
    If the request includes ?save=true then rows will be saved to the AnimalRecord table.
    Otherwise the endpoint validates and returns ingested_rows count.
    """
    save_to_db = request.args.get('save', 'false').lower() in ('1', 'true', 'yes')

    # Read dataframe
    try:
        if 'file' in request.files:
            df = pd.read_csv(request.files['file'])
        else:
            payload = request.get_json()
            if not payload:
                return jsonify({"error": "No data provided"}), 400
            # payload can be a list of dicts or single dict
            df = pd.DataFrame(payload)
    except Exception as e:
        return jsonify({"error": f"Failed to read input: {e}"}), 400

    # Validate columns
    expected = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    missing = expected - set(df.columns)
    if missing:
        return jsonify({"error": "missing columns", "missing": list(missing)}), 400

    # Optionally save to DB
    if save_to_db:
        inserted = 0
        try:
            for _, row in df.iterrows():
                row_data = {}
                for c in NUMERICAL_FEATURES:
                    # numeric conversion with fallback
                    try:
                        row_data[c] = float(row[c]) if pd.notna(row[c]) else 0.0
                    except Exception:
                        row_data[c] = 0.0
                for c in CATEGORICAL_FEATURES:
                    row_data[c] = row[c] if pd.notna(row[c]) else ""
                # build AnimalRecord
                rec = AnimalRecord(**row_data,
                                   predicted_milk_yield=row_data.get('hist_yield_avg7', 0.0),
                                   predicted_disease="Unknown",
                                   disease_probs="{}",
                                   created_at=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
                db.session.add(rec)
                inserted += 1
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Failed to save to DB: {e}"}), 500
        return jsonify({"ingested_rows": len(df), "inserted": inserted})

    return jsonify({"ingested_rows": len(df)})

# ------------------------------
# Export
# ------------------------------
@app.route('/export')
@login_required
def export_all():
    fmt = request.args.get('format', 'csv')
    records = AnimalRecord.query.order_by(AnimalRecord.id.desc()).all()
    if not records:
        flash('No records to export', 'warning')
        return redirect(url_for('index'))
    df = pd.DataFrame([record_to_feature_dict(r) for r in records])
    # optionally include id and predicted fields if present
    try:
        df['id'] = [r.id for r in records]
        df['predicted_milk_yield'] = [getattr(r, 'predicted_milk_yield', None) for r in records]
        df['predicted_disease'] = [getattr(r, 'predicted_disease', None) for r in records]
    except Exception:
        pass

    if fmt == 'csv':
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(io.BytesIO(buf.read().encode('utf-8')), mimetype='text/csv',
                         as_attachment=True, download_name='cattle_records.csv')
    elif fmt == 'excel':
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='records')
        buf.seek(0)
        return send_file(buf, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                         as_attachment=True, download_name='cattle_records.xlsx')
    else:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Cattle Records Report', ln=True, align='C')
        pdf.ln(4)
        pdf.set_font('Arial', size=10)
        for i, row in enumerate(records):
            pdf.cell(0, 6, txt=f"Record ID {row.id} | Breed: {getattr(row, 'breed', '')} | Predicted: {getattr(row, 'predicted_milk_yield', 0.0):.2f} L | Condition: {getattr(row, 'predicted_disease', '')}", ln=True)
            if i % 20 == 19:
                pdf.add_page()
        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return send_file(buf, mimetype='application/pdf',
                         as_attachment=True, download_name='cattle_records.pdf')

# ------------------------------
# Dev helper: seed route to add sample rows for testing
# ------------------------------
@app.route('/seed')
@login_required
def seed_data():
    from random import uniform, choice
    breeds = ["Holstein", "Jersey", "Brown Swiss"]
    lactation_stages = ["Early", "Mid", "Late"]
    feed_types = ["Hay", "Silage", "Concentrate"]
    seasons = ["Winter", "Summer", "Monsoon"]

    for i in range(3):  # Add 3 records
        record = AnimalRecord(
            age=uniform(2, 6),
            weight=uniform(350, 650),
            parity=int(uniform(1, 4)),
            hist_yield_avg7=uniform(8, 25),
            feed_quality=uniform(1, 10),
            feed_qty_kg=uniform(10, 20),
            walking_km=uniform(0.5, 5),
            grazing_h=uniform(2, 8),
            rumination_h=uniform(4, 9),
            resting_h=uniform(8, 12),
            body_temp=uniform(37, 39),
            heart_rate=uniform(50, 90),
            ambient_temp=uniform(15, 35),
            humidity=uniform(40, 80),
            housing_score=uniform(1, 10),
            vaccinations_up_to_date=1,
            disease_history_count=0,
            breed=choice(breeds),
            lactation_stage=choice(lactation_stages),
            feed_type=choice(feed_types),
            season=choice(seasons),
            predicted_milk_yield=uniform(10, 25),
            predicted_disease="Healthy",
            disease_probs="{}",
            created_at=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        )
        db.session.add(record)

    db.session.commit()
    flash("Seed data inserted successfully!", "success")
    return redirect(url_for('index'))

# ------------------------------
# Report generation route
# ------------------------------
@app.route('/report')
@login_required
def generate_report():
    """
    Generate farm-level report with milk yield trends, feed utilization,
    animal health records, and disease risk analysis.
    Exports CSV or PDF based on query param ?format=csv|pdf
    """
    fmt = request.args.get('format', 'csv').lower()

    # Step 1: Load data
    data_path = os.path.join(basedir, "data", "cattle_synthetic.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        records = AnimalRecord.query.order_by(AnimalRecord.id.asc()).all()
        if not records:
            flash("No data available for report generation.", "warning")
            return redirect(url_for('index'))
        df = pd.DataFrame([record_to_feature_dict(r) for r in records])

    # Step 1.1: Ensure expected columns exist
    required_cols = [
        "breed","age","weight","lactation_stage","parity","hist_yield_avg7",
        "feed_type","feed_quality","feed_qty_kg","walking_km","grazing_h","rumination_h",
        "resting_h","body_temp","heart_rate","ambient_temp","humidity","housing_score",
        "vaccinations_up_to_date","disease_history_count","season","health_score",
        "disease_label","milk_yield","feed_efficiency"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 if col not in ["breed","feed_type","season","disease_label"] else "Unknown"

    # Step 2: Summaries
    milk_summary = df.groupby('breed')['milk_yield'].agg(['mean','min','max']).reset_index()
    milk_summary.rename(columns={'mean':'avg_milk_yield','min':'min_milk_yield','max':'max_milk_yield'}, inplace=True)

    feed_summary = df.groupby('feed_type')['feed_qty_kg'].agg(['mean','sum']).reset_index()
    feed_summary.rename(columns={'mean':'avg_feed_qty','sum':'total_feed_qty'}, inplace=True)

    health_summary = df.groupby('disease_label').agg({
        'age': 'count',
        'milk_yield': ['mean','min','max']
    }).reset_index()
    health_summary.columns = ['disease','count','avg_milk_yield','min_milk_yield','max_milk_yield']

    # Step 3: Export
    if fmt == 'csv':
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.write("\n\nMilk Yield Summary\n")
        milk_summary.to_csv(buf, index=False)
        buf.write("\n\nFeed Summary\n")
        feed_summary.to_csv(buf, index=False)
        buf.write("\n\nAnimal Health Summary\n")
        health_summary.to_csv(buf, index=False)
        buf.seek(0)
        return send_file(io.BytesIO(buf.getvalue().encode('utf-8')),
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='farm_report.csv')

    elif fmt == 'pdf':
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Farm Report", ln=True, align='C')
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Milk Yield Summary", ln=True)
        pdf.set_font("Arial", '', 10)
        for _, row in milk_summary.iterrows():
            pdf.cell(0, 6, txt=f"{row['breed']} | Avg: {row['avg_milk_yield']:.2f} | Min: {row['min_milk_yield']:.2f} | Max: {row['max_milk_yield']:.2f}", ln=True)

        pdf.ln(4)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Feed Summary", ln=True)
        pdf.set_font("Arial", '', 10)
        for _, row in feed_summary.iterrows():
            pdf.cell(0, 6, txt=f"{row['feed_type']} | Avg Qty: {row['avg_feed_qty']:.2f} | Total Qty: {row['total_feed_qty']:.2f}", ln=True)

        pdf.ln(4)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Animal Health Summary", ln=True)
        pdf.set_font("Arial", '', 10)
        for _, row in health_summary.iterrows():
            pdf.cell(0, 6, txt=f"{row['disease']} | Count: {row['count']} | Avg Milk: {row['avg_milk_yield']:.2f}", ln=True)

        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return send_file(buf,
                         mimetype='application/pdf',
                         as_attachment=True,
                         download_name='farm_report.pdf')

    else:
        flash("Invalid report format requested.", "warning")
        return redirect(url_for('index'))

# ------------------------------
# Add Welcome Route
# ------------------------------
@app.route('/welcome')
@login_required
def welcome():
    """
    Simple welcome/landing page after login.
    Can show quick stats, buttons to navigate to dashboard (/), etc.
    """
    # Optionally load basic counts
    total_records = AnimalRecord.query.count()
    last_record = AnimalRecord.query.order_by(AnimalRecord.id.desc()).first()

    return render_template(
        'welcome.html',
        total_records=total_records,
        last_record=last_record
    )
@app.route("/chatbot", methods=["POST"])
@login_required
def chatbot():
    """
    Chatbot powered by Google Gemini LLM + pandas for answering questions about cattle_synthetic.csv.
    It does calculations for numeric queries (highest/lowest/average) before asking Gemini.
    """
    try:
        user_message = request.json.get("message", "").strip().lower()
        data_path = os.path.join(basedir, "data", "cattle_synthetic.csv")

        if not os.path.exists(data_path):
            return jsonify({"response": "No data available right now."})

        df = pd.read_csv(data_path)
        if df.empty:
            return jsonify({"response": "No records found in the dataset."})

        # --- Direct Calculation Responses ---
        if "highest" in user_message and "milk" in user_message:
            top_cow = df.loc[df["milk_yield"].idxmax()]
            reply = (
                f"The highest milk producer is a {top_cow['breed']} cow "
                f"producing {top_cow['milk_yield']} L/day. "
                f"She weighs {top_cow['weight']} kg and is in lactation stage {top_cow['lactation_stage']}."
            )
            return jsonify({"response": reply})

        if "lowest" in user_message and "milk" in user_message:
            low_cow = df.loc[df["milk_yield"].idxmin()]
            reply = (
                f"The lowest milk producer is a {low_cow['breed']} cow "
                f"producing only {low_cow['milk_yield']} L/day. "
                f"She weighs {low_cow['weight']} kg and is in lactation stage {low_cow['lactation_stage']}."
            )
            return jsonify({"response": reply})

        if "average" in user_message and "temperature" in user_message:
            reply = f"The average body temperature is {round(df['body_temp'].mean(), 2)}°C."
            return jsonify({"response": reply})

        if "highest" in user_message and "weight" in user_message:
            heavy_cow = df.loc[df["weight"].idxmax()]
            reply = (
                f"The heaviest cow is a {heavy_cow['breed']} weighing {heavy_cow['weight']} kg. "
                f"She produces {heavy_cow['milk_yield']} L/day of milk."
            )
            return jsonify({"response": reply})

        if "lowest" in user_message and "weight" in user_message:
            light_cow = df.loc[df["weight"].idxmin()]
            reply = (
                f"The lightest cow is a {light_cow['breed']} weighing {light_cow['weight']} kg. "
                f"She produces {light_cow['milk_yield']} L/day of milk."
            )
            return jsonify({"response": reply})

        # --- Dataset Summary for Gemini ---
        summary_context = {
            "total_animals": len(df),
            "avg_milk_yield": round(df["milk_yield"].mean(), 2),
            "avg_weight": round(df["weight"].mean(), 2),
            "avg_temp": round(df["body_temp"].mean(), 2),
            "healthy_animals": len(df[df["disease_label"] == 0]),
            "unhealthy_animals": len(df[df["disease_label"] == 1]),
            "feed_efficiency": round(df["milk_yield"].sum() / df["feed_qty_kg"].sum(), 2),
            "breed_counts": df["breed"].value_counts().to_dict(),
            "seasons": df["season"].value_counts().to_dict(),
        }

        prompt = f"""
You are a helpful farm data assistant.
Here is a dataset summary: {summary_context}.
Answer the user's question clearly, using numbers from the dataset whenever possible.
If the user asks for a recommendation, provide a short and practical tip.
User question: "{user_message}"
"""

        # Pick a working model dynamically
        available_models = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        chosen_model = "gemini-2.5-flash" if "models/gemini-2.5-flash" in available_models else available_models[0]

        model = genai.GenerativeModel(chosen_model)
        response = model.generate_content(prompt)

        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})
    
    
    
# --chat bot 2----------------------------
@app.route("/chatbotcm", methods=["POST"])
def chatbotcm():
    user_message = request.json.get("message", "").strip().lower()

    # Only allow cattle/cow related queries
    keywords = ["cow", "cattle", "milk", "farm", "dairy", "livestock", "cattlecare ai"]
    if not any(word in user_message for word in keywords):
        return jsonify({"reply": "Sorry, I only answer questions about cattle or cows."})

    if not API_KEY:
        return jsonify({"error": "GEMINI_API_KEY environment variable not set."}), 500

    payload = {
        "contents": [{"parts": [{"text": user_message}]}],
        "systemInstruction": {
            "parts": [{"text": "You are CattleCare AI, a friendly assistant specialized in cattle, cow, and farm insights. Answer only questions about cattle and farm management."}]
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        api_response = response.json()

        bot_reply = "I'm having trouble answering right now."
        try:
            bot_reply = api_response['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            pass

        return jsonify({"reply": bot_reply})

    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return jsonify({"error": "Could not connect to Gemini API."}), 503
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error."}), 500


# ------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
