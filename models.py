# models.py
import pandas as pd
from extensions import db

class AnimalRecord(db.Model):
    __tablename__ = 'animal_record'

    id = db.Column(db.Integer, primary_key=True)
    breed = db.Column(db.String(64))
    age = db.Column(db.Integer)
    weight = db.Column(db.Float)
    lactation_stage = db.Column(db.String(20))
    parity = db.Column(db.Integer)
    hist_yield_avg7 = db.Column(db.Float)
    feed_type = db.Column(db.String(32))
    feed_quality = db.Column(db.Float)
    feed_qty_kg = db.Column(db.Float)
    walking_km = db.Column(db.Float)
    grazing_h = db.Column(db.Float)
    rumination_h = db.Column(db.Float)
    resting_h = db.Column(db.Float)
    body_temp = db.Column(db.Float)
    heart_rate = db.Column(db.Integer)
    ambient_temp = db.Column(db.Float)
    humidity = db.Column(db.Float)
    housing_score = db.Column(db.Float)
    vaccinations_up_to_date = db.Column(db.Integer)
    disease_history_count = db.Column(db.Integer)
    season = db.Column(db.String(20))

    predicted_milk_yield = db.Column(db.Float)
    predicted_disease = db.Column(db.String(64))
    disease_probs = db.Column(db.Text)
    created_at = db.Column(db.String(64))

    def to_series(self):
        return pd.Series({c.name: getattr(self, c.name) for c in self.__table__.columns})
