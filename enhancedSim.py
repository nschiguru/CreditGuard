import json
import random
import time
import uuid
import datetime
import math
import os
import boto3
import google.generativeai as genai # Import Gemini
import re # Import regular expressions for parsing
from dotenv import load_dotenv

# --- Default Configuration (with Overlap Modifications) ---
DEFAULT_CONFIG = {
    "NUM_CARDS": 50,
    "NUM_TRANSACTIONS": 10000, # Keep larger size
    "OUTPUT_FILENAME": 'simulated_transactions_harder.jsonl', # New name
    "AVG_TRANSACTION_DELAY_SECONDS": 0.1,
    "COMPROMISE_CARD_PROBABILITY": 0.15,
    "FRAUD_INJECTION_PROBABILITY": 0.08, # Slightly increased
    "PERFORM_TEST_TXN_PROB": 0.15,    # Slightly less frequent test txns
    "ATTEMPT_NEAR_MISS_VELOCITY_PROB": 0.30,

    "FRAUD_TEST_AMOUNT_RANGE": (1.00, 50.00),      # Test transactions can overlap normal
    "FRAUD_MAIN_AMOUNT_RANGE": (50.00, 600.00),    # << MAIN FRAUD OVERLAPS NORMAL SIGNIFICANTLY
    "NORMAL_AMOUNT_RANGE": (5.00, 400.00),      # << NORMAL GOES HIGHER

    "LOCATION_VARIATION_KM_NORMAL_BASE": 75,     # << Base distance for normal (reduced from 200)
    "PROB_NORMAL_IS_FARTHER": 0.05,             # << 5% chance normal goes farther
    "LOCATION_VARIATION_KM_NORMAL_FARTHER": 300, # << How far 'farther normal' goes (overlaps fraud zone slightly)
    "PROB_FRAUD_IS_CLOSER": 0.20,               # << 20% chance main fraud is closer
    "DISTANCE_THRESHOLD_KM": 500,                # Detection rule assumption
    "VELOCITY_THRESHOLD_KMH": 800,               # Detection rule assumption
    "LOCATION_VARIATION_KM_FRAUD_MAX": 5000,     # Max distance for obvious fraud

    "SEND_TO_KINESIS": False,
    "KINESIS_STREAM_NAME": os.environ.get('KINESIS_STREAM_NAME', 'credit-guard-pipeline-dev-transaction-stream')
}

# Make a working copy of the config
config = DEFAULT_CONFIG.copy()

# --- Gemini API Key ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# --- Gemini Configuration & Function (UNCHANGED from previous working version) ---
gemini_model = None
if GEMINI_API_KEY:
    try:
        print("Configuring Gemini..."); genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        print("Gemini Configured.")
    except Exception as e: print(f"WARNING: Gemini config failed: {e}")
else: print("WARNING: GEMINI_API_KEY not found.")

def tune_parameters_with_gemini(current_config):
    # (Function code remains the same - attempts to tune probabilities)
    if not gemini_model: return current_config
    tunable_params = ["FRAUD_INJECTION_PROBABILITY", "PERFORM_TEST_TXN_PROB", "ATTEMPT_NEAR_MISS_VELOCITY_PROB", "PROB_NORMAL_IS_FARTHER", "PROB_FRAUD_IS_CLOSER"] # Added new tunable probs
    current_param_values = {k: current_config[k] for k in tunable_params if k in current_config}
    prompt = f"""
    Analyze the following credit card fraud simulation parameters:
    {json.dumps(current_param_values, indent=2)}
    Suggest *slightly modified* values (probabilities between 0.01 and 0.5) to simulate
    fraudsters being 'moderately sneaky but sometimes making mistakes'.
    Provide response ONLY as JSON: {{"PARAM1": VALUE1, ...}}""" # Condensed prompt
    print("\n--- Attempting to tune parameters with Gemini ---"); # print(f"Prompt:\n{prompt}")
    updated_config = current_config.copy()
    try:
        response = gemini_model.generate_content(prompt); # print(f"\nGemini Raw:\n{response.text}\n")
        cleaned_response = response.text.strip().strip('```json').strip('```').strip()
        suggested_params = json.loads(cleaned_response)
        print("Parsed Gemini Suggestion:", suggested_params)
        for key, value in suggested_params.items():
            if key in updated_config and isinstance(value, (int, float)):
                if key.endswith("_PROB"):
                    if 0.0 <= value <= 1.0: print(f"Updating {key} from {updated_config[key]:.2f} to {value:.2f}"); updated_config[key] = float(value)
                    else: print(f"WARNING: Invalid prob for {key}: {value}")
                # Add other type handling here if tuning ranges etc.
            else: print(f"WARNING: Ignoring Gemini suggestion for {key}")
    except Exception as e: print(f"WARNING: Gemini tuning failed: {e}. Using previous parameters.")
    print("--- Parameter tuning attempt complete ---")
    return updated_config

# --- Helper Functions (UNCHANGED - generate_location_around, haversine, generate_timestamp, send_to_kinesis) ---
def generate_location_around(lat, lon, max_distance_km): # (Same as before)
    radius_deg_lat = max_distance_km / 111.1;
    try: radius_deg_lon = max_distance_km / (111.1 * math.cos(math.radians(lat)))
    except ValueError: radius_deg_lon = max_distance_km / 111.1
    delta_lat = random.uniform(-radius_deg_lat, radius_deg_lat); delta_lon = random.uniform(-radius_deg_lon, radius_deg_lon)
    new_lat = max(-90.0, min(90.0, lat + delta_lat)); new_lon = ((lon + delta_lon + 180) % 360) - 180
    return round(new_lat, 6), round(new_lon, 6)
def haversine(lat1, lon1, lat2, lon2): # (Same as before)
    R = 6371
    try: dLat = math.radians(float(lat2) - float(lat1)); dLon = math.radians(float(lon2) - float(lon1)); lat1 = math.radians(float(lat1)); lat2 = math.radians(float(lat2)); a = math.sin(dLat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon/2)**2; c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)); return R * c
    except (TypeError, ValueError) as e: print(f"Haversine Error: {e}"); return 0.0
def generate_timestamp(start_time, increment_seconds): # (Same as before)
    safe_increment = max(0.01, increment_seconds); return start_time + datetime.timedelta(seconds=safe_increment)
def send_to_kinesis(data_record, stream_name, partition_key): # (Same as before)
    try: kinesis_client = boto3.client('kinesis'); kinesis_client.put_record(StreamName=stream_name, Data=json.dumps(data_record), PartitionKey=partition_key); return True
    except Exception as e: print(f"Kinesis Error: {e}"); return False

# --- Main Simulation ---

config = tune_parameters_with_gemini(config)

print("\n--- Starting Simulation with Final Parameters ---")
for key, value in config.items():
    if "KEY" not in key.upper() and "SECRET" not in key.upper(): print(f"Parameter: {key} = {value}")
print("---")

print("Initializing card data...")
cards_data = {} # (Initialization unchanged)
for i in range(config["NUM_CARDS"]): card_id = f"CARD_{i:04d}"; home_lat = random.uniform(25.0, 65.0); home_lon = random.uniform(-125.0, 40.0); cards_data[card_id] = {"HomeLatitude": round(home_lat, 6),"HomeLongitude": round(home_lon, 6),"IsCompromised": random.random() < config["COMPROMISE_CARD_PROBABILITY"],"LastTxTimestamp": None,"LastTxLatitude": None,"LastTxLongitude": None,}
print(f"Generated {len(cards_data)} cards.")
print(f"Starting simulation, writing to '{config['OUTPUT_FILENAME']}'...")

current_time = datetime.datetime.now(datetime.timezone.utc)
transactions_generated = 0; successful_sends = 0

with open(config['OUTPUT_FILENAME'], 'w') as outfile:
    while transactions_generated < config['NUM_TRANSACTIONS']:
        card_id = random.choice(list(cards_data.keys())); card_info = cards_data[card_id]
        is_fraud_scenario = card_info["IsCompromised"] and random.random() < config["FRAUD_INJECTION_PROBABILITY"]

        amount = 0.0; merchant_lat, merchant_lon = 0.0, 0.0; is_fraud_flag_for_output = False

        # -- FRAUD SCENARIO --
        if is_fraud_scenario:
            is_fraud_flag_for_output = True
            # Tactic 1: Small "Test" Transaction
            if random.random() < config["PERFORM_TEST_TXN_PROB"]:
                print(f"[Fraud Tactic] Simulating TEST transaction for {card_id}")
                amount = round(random.uniform(*config["FRAUD_TEST_AMOUNT_RANGE"]), 2)
                test_distance_km = random.uniform(config["DISTANCE_THRESHOLD_KM"] * 0.8, config["LOCATION_VARIATION_KM_FRAUD_MAX"] * 0.6) # Still quite far for test
                merchant_lat, merchant_lon = generate_location_around(card_info["HomeLatitude"], card_info["HomeLongitude"], test_distance_km)
                time_increment = random.expovariate(1.0 / (config["AVG_TRANSACTION_DELAY_SECONDS"] * 1.5))
                current_time = generate_timestamp(current_time, time_increment)

            # Tactic 2: Main Fraud Attempt (potentially closer or near-miss velocity)
            else:
                print(f"[Fraud Tactic] Simulating MAIN fraud attempt for {card_id}")
                amount = round(random.uniform(*config["FRAUD_MAIN_AMOUNT_RANGE"]), 2) # Overlaps normal

                # --- MODIFIED: Location logic for main fraud ---
                if random.random() < config["PROB_FRAUD_IS_CLOSER"]:
                     print("[Fraud Tactic Adjust] Simulating CLOSER main fraud.")
                     # Generate closer to home, in the harder-to-detect zone near the normal boundary
                     fraud_distance_km = random.uniform(
                         config["LOCATION_VARIATION_KM_NORMAL_BASE"] * 1.5, # Above normal base
                         config["DISTANCE_THRESHOLD_KM"] * 1.1               # But near/just over the threshold
                     )
                else: # Default: Far away main fraud
                    fraud_distance_km = random.uniform(
                        config["DISTANCE_THRESHOLD_KM"], config["LOCATION_VARIATION_KM_FRAUD_MAX"]
                    )
                merchant_lat, merchant_lon = generate_location_around(card_info["HomeLatitude"], card_info["HomeLongitude"], fraud_distance_km)
                # --- End Modified Location ---

                # Near-Miss Velocity Logic (remains mostly the same, applies to chosen location)
                if card_info["LastTxTimestamp"] and card_info["LastTxLatitude"] is not None and random.random() < config["ATTEMPT_NEAR_MISS_VELOCITY_PROB"]:
                    # ... (near-miss velocity time calculation same as before) ...
                    last_ts, last_lat, last_lon = card_info["LastTxTimestamp"], card_info["LastTxLatitude"], card_info["LastTxLongitude"]
                    distance_km = haversine(last_lat, last_lon, merchant_lat, merchant_lon)
                    target_velocity_kmh = config["VELOCITY_THRESHOLD_KMH"] * random.uniform(0.85, 0.99)
                    if distance_km > 0 and target_velocity_kmh > 0:
                        required_time_seconds = (distance_km / target_velocity_kmh) * 3600
                        simulated_delay_seconds = max(10.0, required_time_seconds * random.uniform(1.0, 1.15))
                        print(f"[Fraud Tactic] Attempting Near Miss Velocity ({target_velocity_kmh:.0f} km/h -> delay {simulated_delay_seconds:.0f}s) for {card_id}")
                        current_time = last_ts + datetime.timedelta(seconds=simulated_delay_seconds)
                    else: time_increment = random.expovariate(1.0 / config["AVG_TRANSACTION_DELAY_SECONDS"]); current_time = generate_timestamp(current_time, time_increment)
                else: # Standard time increment for main fraud
                    time_increment = random.expovariate(1.0 / config["AVG_TRANSACTION_DELAY_SECONDS"]); current_time = generate_timestamp(current_time, time_increment)

        # -- NORMAL TRANSACTION SCENARIO --
        else:
            is_fraud_flag_for_output = False
            amount = round(random.uniform(*config["NORMAL_AMOUNT_RANGE"]), 2) # Normal range overlaps fraud

            # --- MODIFIED: Location logic for normal transaction ---
            if random.random() < config["PROB_NORMAL_IS_FARTHER"]:
                 print("[Normal Tactic Adjust] Simulating FURTHER normal transaction.")
                 # Generate farther, potentially into the "closer fraud" zone
                 normal_distance_km = random.uniform(
                     config["LOCATION_VARIATION_KM_NORMAL_BASE"], # Start from base normal
                     config["LOCATION_VARIATION_KM_NORMAL_FARTHER"] # Go up to the farther limit
                 )
            else: # Default: Close normal
                normal_distance_km = random.uniform(0, config["LOCATION_VARIATION_KM_NORMAL_BASE"]) # Stay within base
            merchant_lat, merchant_lon = generate_location_around(card_info["HomeLatitude"], card_info["HomeLongitude"], normal_distance_km)
             # --- End Modified Location ---

            # Normal time progression
            time_increment = random.expovariate(1.0 / config["AVG_TRANSACTION_DELAY_SECONDS"]); current_time = generate_timestamp(current_time, time_increment)

        # --- Create Transaction Record (unchanged) ---
        transaction_id = str(uuid.uuid4()); merchant_id = f"MERCHANT_{random.randint(1000, 9999)}"
        transaction_data = {
            "TransactionID": transaction_id, "CardID": card_id, "Timestamp": current_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "Amount": amount, "MerchantID": merchant_id, "MerchantLatitude": merchant_lat, "MerchantLongitude": merchant_lon,
            "IsFraud": is_fraud_flag_for_output, "HomeLatitude": card_info["HomeLatitude"], "HomeLongitude": card_info["HomeLongitude"]
        }

        # --- Output / State Update (unchanged logic) ---
        json_output = json.dumps(transaction_data); outfile.write(json_output + '\n'); transactions_generated += 1
        if transactions_generated % 500 == 0: print(f"... generated {transactions_generated}/{config['NUM_TRANSACTIONS']} transactions") # Log less often for large runs
        if config['SEND_TO_KINESIS']:
            if send_to_kinesis(transaction_data, config['KINESIS_STREAM_NAME'], card_id): successful_sends += 1
            else: config['SEND_TO_KINESIS'] = False # Stop trying
        card_info["LastTxTimestamp"] = current_time; card_info["LastTxLatitude"] = merchant_lat; card_info["LastTxLongitude"] = merchant_lon

# --- End of Simulation ---
print(f"\nFinished generating {transactions_generated} MORE DIFFICULT auto-tuned transactions.")
print(f"Data saved to '{config['OUTPUT_FILENAME']}'")
if successful_sends > 0: print(f"Sent {successful_sends} records to Kinesis stream '{config['KINESIS_STREAM_NAME']}'.")
