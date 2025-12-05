import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# ============================================================
# 1️⃣ FLIGHT DELAY MODEL
# ============================================================
print("\n✈️ Training Flight Delay Model...")
np.random.seed(42)

delay_data = pd.DataFrame({
    'dep_hour': np.random.randint(0, 24, 500),
    'distance': np.random.randint(100, 3000, 500)
})

# Label: 1 if flight likely delayed, else 0
delay_data['delayed'] = (
    (delay_data['dep_hour'] > 18) | (delay_data['distance'] > 2000)
).astype(int)

X = delay_data[['dep_hour', 'distance']]
y = delay_data['delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

delay_model = LogisticRegression()
delay_model.fit(X_train, y_train)

print("  ✅ Accuracy:", round(accuracy_score(y_test, delay_model.predict(X_test)), 3))
joblib.dump(delay_model, "models/flight_delay_model.pkl")
print("  💾 Saved: flight_delay_model.pkl")

# ============================================================
# 2️⃣ PASSENGER SATISFACTION MODEL
# ============================================================
print("\n😊 Training Passenger Satisfaction Model...")

satisfaction_data = pd.DataFrame({
    'duration': np.random.uniform(0.5, 10, 500),
    'rating': np.random.randint(1, 6, 500)
})

# Label: Satisfied if shorter flights & better rating
satisfaction_data['satisfied'] = (
    (satisfaction_data['duration'] < 5) & (satisfaction_data['rating'] > 3)
).astype(int)

X = satisfaction_data[['duration', 'rating']]
y = satisfaction_data['satisfied']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

satisfaction_model = LogisticRegression()
satisfaction_model.fit(X_train, y_train)

print("  ✅ Accuracy:", round(accuracy_score(y_test, satisfaction_model.predict(X_test)), 3))
joblib.dump(satisfaction_model, "models/passenger_satisfaction_model.pkl")
print("  💾 Saved: passenger_satisfaction_model.pkl")

# ============================================================
# 3️⃣ PASSENGER DEMAND FORECAST MODEL
# ============================================================
print("\n📈 Training Passenger Demand Forecast Model...")

demand_data = pd.DataFrame({
    'year': np.random.randint(2015, 2025, 500),
    'month': np.random.randint(1, 13, 500),
    'previous_passengers': np.random.randint(10000, 200000, 500)
})

# Simulate future passenger volume
demand_data['future_demand'] = (
    demand_data['previous_passengers'] * np.random.uniform(0.95, 1.2, 500)
    + (demand_data['year'] - 2015) * 500
)

X = demand_data[['year', 'month', 'previous_passengers']]
y = demand_data['future_demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

demand_model = LinearRegression()
demand_model.fit(X_train, y_train)

print("  ✅ R²:", round(r2_score(y_test, demand_model.predict(X_test)), 3))
joblib.dump(demand_model, "models/passenger_demand_model.pkl")
print("  💾 Saved: passenger_demand_model.pkl")

# ============================================================
# 4️⃣ OPTIONAL: AIRPORT COORDINATE MAPPING (for route feature)
# ============================================================
print("\n🗺️ Generating Airport Coordinate Data (for route visualization)...")

airport_coords = {
    "DEL": (28.5562, 77.1000),
    "MAA": (12.9941, 80.1709),
    "BOM": (19.0896, 72.8656),
    "BLR": (13.1986, 77.7066),
    "HYD": (17.2403, 78.4294),
    "CCU": (22.6547, 88.4467)
}

# Save as a model for easy backend access (optional future use)
joblib.dump(airport_coords, "models/airport_coordinates.pkl")
print("  💾 Saved: airport_coordinates.pkl")

print("\n🎉 All models successfully generated inside 'models/' folder!")
