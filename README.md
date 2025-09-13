# Cattle Monitoring — Flask + Python (Full Project)

## Setup
1. Create project folder and copy the files above in the structure you described.
2. Create & activate virtualenv:
   python -m venv venv
   # mac/linux
   source venv/bin/activate
   # windows
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Generate synthetic dataset (recommended):
   python data_generator.py

5. Train models:
   python train_models.py
   # or use the "Train / Retrain Models" button in the web UI

6. Run the app:
   python app.py
   # Open http://127.0.0.1:5000

Notes:
- The UI form posts to /predict and saved predictions are kept in SQLite at instance/app.db.
- Exports available via /export?format=csv|excel|pdf
- Models saved to ./models/. To re-train, run train_models.py or click Train in UI.
