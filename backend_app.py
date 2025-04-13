import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd # Required by XGBoost model implicitly sometimes
import json
import math
import os

print("--- Loading Model and Threshold (NO SCALER) ---")

# --- Configuration ---
MODEL_FILE = 'creditguard_xgb_best_randomsearch.joblib' # Correct Model file
THRESHOLD_FILE = 'optimal_threshold_randomsearch.json'
# SCALER_FILE = 'robust_scaler_randomsearch.joblib' # <-- REMOVED

# Demo assumptions
HOME_LAT = 40.71
HOME_LON = -74.00
DEFAULT_TIME_SINCE_LAST = 3600
DEFAULT_VELOCITY = 0

# --- Load Artifacts ---
try:
    model = joblib.load(MODEL_FILE)
    print(f"Model loaded successfully from '{MODEL_FILE}'")
except Exception as e:
    print(f"FATAL ERROR: Could not load model '{MODEL_FILE}': {e}")
    exit()

# --- Load Threshold ---
try:
    with open(THRESHOLD_FILE, 'r') as f:
        threshold_data = json.load(f)
    optimal_threshold = threshold_data.get('optimal_threshold', 0.5)
    print(f"Optimal threshold loaded: {optimal_threshold:.4f}")
except Exception as e:
    print(f"WARNING: Could not load threshold file '{THRESHOLD_FILE}': {e}. Using 0.5.")
    optimal_threshold = 0.5

# Expected feature order for the model
# IMPORTANT: This MUST match the order used during training!
FEATURE_ORDER = ['Amount', 'distance_from_home_km', 'time_since_last_tx_seconds', 'velocity_kmh']

# --- Helper Function (Haversine - unchanged) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    try: lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except (TypeError, ValueError): return 0.0
    dLat = math.radians(lat2 - lat1); dLon = math.radians(lon2 - lon1); lat1 = math.radians(lat1); lat2 = math.radians(lat2)
    a = math.sin(dLat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon/2)**2; c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)); return R * c

# --- Flask App ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "CreditGuard Prediction Server is running. Use the /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict_fraud():
    print("\nReceived /predict request")
    try:
        data = request.get_json()
        print("Request JSON:", data)

        required_keys = ["Amount", "MerchantLatitude", "MerchantLongitude"]
        if not data or not all(key in data for key in required_keys):
            print("Error: Missing required keys in request")
            return jsonify({"error": f"Missing required keys: {required_keys}"}), 400

        amount = float(data['Amount'])
        merch_lat = float(data['MerchantLatitude'])
        merch_lon = float(data['MerchantLongitude'])

        distance = haversine(HOME_LAT, HOME_LON, merch_lat, merch_lon)
        time_since_last = float(DEFAULT_TIME_SINCE_LAST)
        velocity = float(DEFAULT_VELOCITY)
        print(f"Calculated Distance: {distance:.2f} km")
        print(f"Using Defaults: TimeSinceLast={time_since_last}s, Velocity={velocity}km/h")

        # --- Prepare Feature Vector (Order matters!) ---
        # Create a DataFrame for consistent feature naming/ordering expected by XGBoost
        feature_values = [[amount, distance, time_since_last, velocity]]
        live_features_df = pd.DataFrame(feature_values, columns=FEATURE_ORDER)
        print("Feature DataFrame (unscaled):")
        print(live_features_df)


        # --- Predict (Using UNSCALED features) ---
        try:
            # predict_proba gives [[prob_class_0, prob_class_1]]
            probability_fraud = model.predict_proba(live_features_df)[0][1] # Use DataFrame
            print(f"Model Raw Probability (Fraud): {probability_fraud:.4f}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": f"Model prediction failed: {e}"}), 500

        # --- Apply Threshold ---
        is_fraud_prediction = bool(probability_fraud >= optimal_threshold)
        print(f"Threshold: {optimal_threshold:.4f}, Is Fraud Prediction: {is_fraud_prediction}")

        # --- Return Result ---
        result = {
            "is_fraud": is_fraud_prediction,
            "probability": round(float(probability_fraud), 4),
            "threshold_used": round(optimal_threshold, 4)
        }
        print("Sending response:", result)
        return jsonify(result), 200

    except Exception as e:
        print(f"Unhandled error in /predict: {e}")
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server (Scaler step REMOVED)...")
    app.run(debug=True)
