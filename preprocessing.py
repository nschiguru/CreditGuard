import pandas as pd
import numpy as np
import json
import math
from datetime import datetime, timezone, timedelta
# from sklearn.preprocessing import RobustScaler # SCALER IS REMOVED FROM THIS SCRIPT
# import joblib # No longer saving scaler here
import sys

print("--- Preprocessing Synthetic Data (Feature Engineering Only) ---")

# --- Configuration ---
INPUT_JSONL_FILE = 'simulated_transactions_harder.jsonl' # Input from enhanced simulator
OUTPUT_CSV_FILE = 'training_data_unscaled.csv'         # <--- CHANGED: Output UNscaled data
# SCALER_FILE = 'robust_scaler.joblib'         # REMOVED

# --- Helper Functions (unchanged) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except (TypeError, ValueError):
        return 0.0
    dLat = math.radians(lat2 - lat1); dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1); lat2 = math.radians(lat2)
    a = math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def parse_timestamp(timestamp_str):
    if not isinstance(timestamp_str, str): return None
    try:
        if timestamp_str.endswith('Z'): timestamp_str = timestamp_str[:-1] + '+00:00'
        ts = datetime.fromisoformat(timestamp_str)
        if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except (ValueError, TypeError) as e: print(f"Error parsing timestamp '{timestamp_str}': {e}"); return None

# --- 1. Load Data (unchanged) ---
print(f"Loading data from {INPUT_JSONL_FILE}...")
try:
    lines = []; # (Code to read JSONL is the same)
    with open(INPUT_JSONL_FILE, 'r') as f:
        for line in f:
            try: lines.append(json.loads(line))
            except json.JSONDecodeError: print(f"Skipping invalid JSON line: {line.strip()}")
    data = pd.DataFrame(lines)
    print(f"Data loaded. Shape: {data.shape}")
    if data.empty: print("Error: No data loaded."); sys.exit(1)
except FileNotFoundError: print(f"ERROR: Input file '{INPUT_JSONL_FILE}' not found."); sys.exit(1)
except Exception as e: print(f"Error loading data: {e}"); sys.exit(1)

# --- 2. Initial Data Preparation (unchanged) ---
print("Preparing data (parsing timestamps, sorting)...")
data['Timestamp_dt'] = data['Timestamp'].apply(parse_timestamp)
original_rows = len(data)
data.dropna(subset=['Timestamp_dt'], inplace=True)
if len(data) < original_rows: print(f"Warning: Dropped {original_rows - len(data)} rows.")
data.sort_values(by=['CardID', 'Timestamp_dt'], inplace=True)
data.reset_index(drop=True, inplace=True)

# --- 3. Feature Engineering (Now with warning fixes) ---
print("Engineering features (distance, time diff, velocity)...")
print(" - Calculating distance_from_home...")
data['distance_from_home_km'] = data.apply(lambda row: haversine(row['HomeLatitude'], row['HomeLongitude'], row['MerchantLatitude'], row['MerchantLongitude']), axis=1)
print(" - Calculating time_since_last_tx and velocity...")
data['LastTxTimestamp_dt'] = data.groupby('CardID')['Timestamp_dt'].shift(1)
data['LastTxLatitude'] = data.groupby('CardID')['MerchantLatitude'].shift(1)
data['LastTxLongitude'] = data.groupby('CardID')['MerchantLongitude'].shift(1)
time_diff = data['Timestamp_dt'] - data['LastTxTimestamp_dt']
data['time_since_last_tx_seconds'] = time_diff.dt.total_seconds().fillna(0.0)
data['time_since_last_tx_seconds'] = data['time_since_last_tx_seconds'].clip(lower=0)
data['distance_from_last_km'] = data.apply(lambda row: haversine(row['LastTxLatitude'], row['LastTxLongitude'], row['MerchantLatitude'], row['MerchantLongitude']), axis=1)
data['velocity_kmh'] = 0.0
mask = data['time_since_last_tx_seconds'] > 0
time_hours = data.loc[mask, 'time_since_last_tx_seconds'] / 3600.0
velocity_calc = data.loc[mask, 'distance_from_last_km'] / time_hours # Calculate on selection
data.loc[mask, 'velocity_kmh'] = velocity_calc # Assign back using .loc

# Handle potential infinite velocity & NaNs directly on the DataFrame column
data['velocity_kmh'] = data['velocity_kmh'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
data['velocity_kmh'] = data['velocity_kmh'].clip(lower=0) # Ensure non-negative


# --- 4. Select and Clean Final Features (unchanged) ---
print("Selecting and cleaning final features...")
feature_cols = [
    'Amount', 'distance_from_home_km', 'time_since_last_tx_seconds', 'velocity_kmh'
]
target_col = 'IsFraud'
# Keep relevant original + engineered features + target before scaling
final_data = data[feature_cols + [target_col]].copy()
for col in feature_cols:
    final_data[col] = pd.to_numeric(final_data[col], errors='coerce')
final_data.fillna(0.0, inplace=True)

# --- 5. REMOVED: Scaling Features ---
# print("Scaling features...") <-- REMOVED
# scaler = RobustScaler() # Initialize scaler <-- REMOVED
# final_data[feature_cols] = scaler.fit_transform(final_data[feature_cols]) <-- REMOVED
# print("Features scaled.") <-- REMOVED

# --- 6. REMOVED: Save the fitted scaler object ---
# try: <-- REMOVED
#     joblib.dump(scaler, SCALER_FILE) <-- REMOVED
#     print(f"Scaler saved to '{SCALER_FILE}'") <-- REMOVED
# except Exception as e: <-- REMOVED
#     print(f"Error saving scaler: {e}") <-- REMOVED


# --- 7. Save UNscaled Processed Data ---  <-- Renumbered Step
print(f"Saving UNscaled processed data to {OUTPUT_CSV_FILE}...")
try:
    final_data[target_col] = final_data[target_col].astype(int)
    # Output DF contains the unscaled features + target
    output_df = final_data[feature_cols + [target_col]]
    output_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Successfully saved UNscaled training data. Shape: {output_df.shape}")
    print("Head of the saved UNscaled data:")
    print(output_df.head())
except Exception as e:
    print(f"Error saving data to CSV: {e}")

print("\n--- Preprocessing (Feature Engineering Only) complete ---")
