from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Models
model_dir = "models"

try:
    delay_model = joblib.load(os.path.join(model_dir, "flight_delay_model.pkl"))
    satisfaction_model = joblib.load(os.path.join(model_dir, "passenger_satisfaction_model.pkl"))
    demand_model = joblib.load(os.path.join(model_dir, "passenger_demand_model.pkl"))
except Exception as e:
    print(f"⚠️ Model loading error: {e}")
    delay_model = satisfaction_model = demand_model = None


# ---------------- FRONTEND ROUTE ----------------
@app.route('/')
def home():
    return render_template('AeroCommand_AI_Dashboard.html')


# ---------------- API ROUTES ----------------
@app.route('/predict_delay', methods=['POST'])
def predict_delay():
    try:
        data = request.get_json()
        features = np.array([[data['dep_hour'], data['distance']]])
        pred = delay_model.predict(features)[0]
        return jsonify({'prediction': int(pred)})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict_satisfaction', methods=['POST'])
def predict_satisfaction():
    try:
        data = request.get_json()
        features = np.array([[data['duration'], data['rating']]])
        pred = satisfaction_model.predict(features)[0]
        return jsonify({'prediction': int(pred)})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    try:
        data = request.get_json()
        features = np.array([[data['year'], data['month'], data['previous_passengers']]])
        pred = demand_model.predict(features)[0]
        return jsonify({'prediction': round(pred, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})


# ---------------- FLIGHT ROUTE TRACKER ----------------
@app.route('/get_flight_route', methods=['POST'])
def get_flight_route():
    try:
        data = request.get_json()
        origin = data.get('origin', '').upper()
        destination = data.get('destination', '').upper()

        # Sample airport coordinates (lat, lon)
        airport_coords = {
            "DEL": (28.5562, 77.1000),  # Delhi
            "MAA": (12.9941, 80.1709),  # Chennai
            "BOM": (19.0896, 72.8656),  # Mumbai
            "BLR": (13.1986, 77.7066),  # Bangalore
            "HYD": (17.2403, 78.4294),  # Hyderabad
            "CCU": (22.6547, 88.4467),  # Kolkata
        }

        if origin not in airport_coords or destination not in airport_coords:
            return jsonify({'error': 'Invalid airport code(s). Please use e.g. DEL, MAA, BOM, BLR, HYD, CCU'})

        route = {
            'origin': {'code': origin, 'coords': airport_coords[origin]},
            'destination': {'code': destination, 'coords': airport_coords[destination]}
        }
        return jsonify(route)

    except Exception as e:
        return jsonify({'error': str(e)})


# ---------------- RUN APP ----------------
if __name__ == '__main__':
    app.run(debug=True)
