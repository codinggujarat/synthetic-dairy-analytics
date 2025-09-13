"""
Generates synthetic cattle dataset for demo and quick testing.
Produces: data/cattle_synthetic.csv
Includes computed fields for dashboard metrics.
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

BREEDS = ["Holstein", "Jersey", "Brown Swiss", "Buffalo"]
LACTATION_STAGES = ["early", "mid", "late"]
FEED_TYPES = ["green", "dry", "concentrate", "mixed"]
SEASONS = ["spring", "summer", "autumn", "winter"]

def breed_base_yield(breed):
    return {
        "Holstein": 25.0,
        "Jersey": 18.0,
        "Brown Swiss": 20.0,
        "Buffalo": 10.0
    }[breed]

def generate_row():
    breed = RNG.choice(BREEDS)
    age = int(RNG.integers(2, 15))
    weight = float(max(150, RNG.normal(600 if breed != "Buffalo" else 500, 50)))
    lactation_stage = RNG.choice(LACTATION_STAGES)
    parity = int(np.clip(RNG.poisson(2) + 1, 1, 8))
    hist_yield = float(np.clip(RNG.normal(breed_base_yield(breed), 3.0), 0.5, 60.0))
    feed_type = RNG.choice(FEED_TYPES)
    feed_quality = float(np.clip(RNG.normal(6.5, 1.8), 1, 10))
    feed_qty = float(np.clip(weight * RNG.normal(0.03, 0.007), 5, 40))
    walking_km = float(np.clip(RNG.normal(2.0, 1.5), 0.0, 10.0))
    grazing_h = float(np.clip(RNG.normal(3.0, 2.0), 0.0, 12.0))
    rumination_h = float(np.clip(RNG.normal(5.0, 1.5), 1.0, 8.0))
    resting_h = float(np.clip(RNG.normal(10.0, 2.0), 4.0, 20.0))
    body_temp = float(np.clip(RNG.normal(38.5, 0.6), 36.0, 41.5))
    heart_rate = int(np.clip(RNG.normal(65, 10), 40, 120))
    ambient_temp = float(np.clip(RNG.normal(25, 6), -5, 45))
    humidity = float(np.clip(RNG.normal(60, 20), 10, 100))
    housing_score = float(np.clip(RNG.normal(7.0, 2.0), 1, 10))
    vaccinations_up_to_date = int(RNG.choice([0, 1], p=[0.12, 0.88]))
    disease_history_count = int(np.clip(RNG.poisson(0.2), 0, 5))
    season = RNG.choice(SEASONS)

    # health score calculation
    health_score = (
        0.25 * (feed_quality / 10.0) +
        0.2 * (feed_qty / 40.0) +
        0.15 * (rumination_h / 8.0) +
        0.1 * (resting_h / 20.0) +
        0.1 * (housing_score / 10.0) +
        0.1 * vaccinations_up_to_date
    )
    temp_penalty = max(0, abs(ambient_temp - 22) - 5) / 20.0
    humidity_penalty = max(0, (humidity - 70) / 100.0)
    health_score -= (temp_penalty + humidity_penalty)
    health_score = float(np.clip(health_score, 0.0, 1.0))

    # milk yield
    base = breed_base_yield(breed)
    lact_mult = {"early": 1.15, "mid": 1.0, "late": 0.85}[lactation_stage]
    parity_adj = 1.0 + (parity - 2) * 0.02
    milk_yield = (
        base * lact_mult * parity_adj
        * (0.6 + 0.8 * health_score)
        * (1 + (feed_qty - 20) / 200.0)
        + RNG.normal(0.0, 1.8)
    )
    milk_yield = float(np.clip(milk_yield, 0.5, 60.0))

    # disease classification
    disease_label = "none" if health_score > 0.5 else "minor"

    # feed efficiency
    feed_efficiency = milk_yield / feed_qty if feed_qty > 0 else 0.0

    return {
        "breed": breed,
        "age": age,
        "weight": float(round(weight, 1)),
        "lactation_stage": lactation_stage,
        "parity": parity,
        "feed_type": feed_type,
        "feed_quality": float(round(feed_quality, 2)),
        "feed_qty_kg": float(round(feed_qty, 2)),
        "rumination_h": float(round(rumination_h, 2)),
        "resting_h": float(round(resting_h, 2)),
        "health_score": float(round(health_score)),
        "vaccinations_up_to_date": vaccinations_up_to_date,
        "disease_label": disease_label,
        "milk_yield": float(round(milk_yield, 2)),
        "feed_efficiency": float(round(feed_efficiency, 2)),
    }

def generate_dataset(n=3000, out_path="data/cattle_synthetic.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = [generate_row() for _ in range(n)]
    df = pd.DataFrame(rows)

    # Dashboard metrics (example, optional)
    total_animals = len(df)
    avg_milk_yield = df["milk_yield"].mean()
    healthy_count = (df["health_score"] > 0.5).sum()
    unhealthy_count = total_animals - healthy_count
    avg_feed_efficiency = df["feed_efficiency"].mean()
    avg_weight = df["weight"].mean()
    vaccination_rate = df["vaccinations_up_to_date"].mean() * 100

    print(f"Total Animals: {total_animals}")
    print(f"Average Milk Yield: {avg_milk_yield:.2f} L/day")
    print(f"Healthy Animals: {healthy_count}")
    print(f"Unhealthy Animals: {unhealthy_count}")
    print(f"Feed Efficiency: {avg_feed_efficiency:.2f} L/kg")
    print(f"Average Weight: {avg_weight:.2f} kg")
    print(f"Vaccination Rate: {vaccination_rate:.2f}%")

    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    return df

if __name__ == "__main__":
    generate_dataset(3000)
