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
    def to_dict(self):
        return {
            "id": self.id,
            "breed": self.breed,
            "age": self.age,
            "weight": self.weight,
            "lactation_stage": self.lactation_stage,
            "parity": self.parity,
            "hist_yield_avg7": self.hist_yield_avg7,
            "feed_type": self.feed_type,
            "feed_quality": self.feed_quality,
            "feed_qty_kg": self.feed_qty_kg,
            "walking_km": self.walking_km,
            "grazing_h": self.grazing_h,
            "rumination_h": self.rumination_h,
            "resting_h": self.resting_h,
            "body_temp": self.body_temp,
            "heart_rate": self.heart_rate,
            "ambient_temp": self.ambient_temp,
            "humidity": self.humidity,
            "housing_score": self.housing_score,
            "vaccinations_up_to_date": self.vaccinations_up_to_date,
            "disease_history_count": self.disease_history_count,
            "season": self.season,
            "predicted_milk_yield": self.predicted_milk_yield,
            "predicted_disease": self.predicted_disease,
            "disease_probs": self.disease_probs,
            "created_at": self.created_at
        }
