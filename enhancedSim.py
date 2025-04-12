import json
import random
import time
import uuid
import datetime
import math
import os
import boto3
import google.generativeai as genai
import re
from dotenv import load_dotenv

# this is default values for the simulation parameters
# transaction from unusual location (ip address tracking)
# high value purchases
# out of character spending (sentiment analysis)
# rapid speed purchases
# multiple small transactions to test if the card words

DEFAULT_CONFIG = {
    # scale
    "NUM_CARDS": 50,
    "NUM_TRANSACTIONS": 1500,
    # generated data location
    "OUTPUT_FILENAME": 'simulated_transactions_auto_tuned.jsonl',
    # time between transactions
    "AVG_TRANSACTION_DELAY_SECONDS": 0.1,
    # probability that a card is "compromised" already
    "COMPROMISE_CARD_PROBABILITY": 0.15,
    # chance that next transaction is fraud when card is "compromised"
    "FRAUD_INJECTION_PROBABILITY": 0.05,
    # small test transactions
    "PERFORM_TEST_TXN_PROB": 0.20,
    # testing less suspicious speed cases of fraud
    "ATTEMPT_NEAR_MISS_VELOCITY_PROB": 0.30,
    # multiple small transactions
    "FRAUD_TEST_AMOUNT_RANGE": (1.00, 15.00),
    "FRAUD_MAIN_AMOUNT_RANGE": (150.00, 2500.00),
    "NORMAL_AMOUNT_RANGE": (5.00, 250.00),
    "LOCATION_VARIATION_KM_NORMAL": 50,
    "DISTANCE_THRESHOLD_KM": 500,
    "VELOCITY_THRESHOLD_KMH": 800,
    "LOCATION_VARIATION_KM_FRAUD": 5000,
    "SEND_TO_KINESIS": False,
    "KINESIS_STREAM_NAME": os.environ.get('KINESIS_STREAM_NAME', 'credit-guard-pipeline-dev-transaction-stream')
}

# Make a working copy of the config
config = DEFAULT_CONFIG.copy()

# --- Gemini API Key ---
# !! Store securely (e.g., environment variable, Secrets Manager) !!
# Replace with your method of getting the key
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE" # Or paste directly for hackathon testing ONLY

# --- Gemini Configuration & Function ---
gemini_model = None
if GEMINI_API_KEY:
    try:
        print("Configuring Gemini...")
        genai.configure(api_key=GEMINI_API_KEY)
        print("--- Listing Available Models (supporting generateContent) ---")
        try:
            count = 0
            for m in genai.list_models():
                # Check if the 'generateContent' method is supported
                if 'generateContent' in m.supported_generation_methods:
                    print(f"- {m.name}") # Print the EXACT name
                    count += 1
            if count == 0:
                print("  No models supporting 'generateContent' found for this API key.")
        except Exception as e:
            print(f"  Could not list models: {e}")
        print("--- End Model Listing ---")
        gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest') # Or 'gemini-1.0-pro' etc.
        print("Gemini Configured.")
    except Exception as e:
        print(f"WARNING: Failed to configure Gemini: {e}. Using default simulation parameters.")
else:
    print("WARNING: GEMINI_API_KEY not found. Using default simulation parameters.")

def tune_parameters_with_gemini(current_config):
    if not gemini_model:
        print("Gemini not available, skipping tuning.")
        return current_config # Return defaults if Gemini isn't working

    # Select parameters to tune
    tunable_params = [
        "FRAUD_INJECTION_PROBABILITY",
        "PERFORM_TEST_TXN_PROB",
        "ATTEMPT_NEAR_MISS_VELOCITY_PROB",
        # Could add amount ranges but parsing tuples/ranges is harder
    ]
    current_param_values = {k: current_config[k] for k in tunable_params}

    # Craft the prompt asking for structured output (attempt JSON)
    prompt = f"""
    Analyze the following credit card fraud simulation parameters:
    {json.dumps(current_param_values, indent=2)}

    Suggest *slightly modified* values for these parameters to simulate fraudsters being
    'more cautious and subtle'. Keep probabilities between 0.01 and 0.5.

    Provide your response ONLY as a valid JSON object containing the suggested
    values for these specific keys ({', '.join(tunable_params)}).
    Example Response Format (JUST THE JSON):
    {{
      "FRAUD_INJECTION_PROBABILITY": 0.04,
      "PERFORM_TEST_TXN_PROB": 0.25,
      "ATTEMPT_NEAR_MISS_VELOCITY_PROB": 0.35
    }}
    """

    print("\n--- Attempting to tune parameters with Gemini ---")
    print(f"Prompt sent to Gemini:\n{prompt}")

    updated_config = current_config.copy()
    try:
        response = gemini_model.generate_content(prompt)
        print(f"\nGemini Raw Response Text:\n{response.text}\n")

        # Attempt to parse the response as JSON
        # Clean potential markdown backticks if Gemini wraps JSON in them
        cleaned_response = response.text.strip().strip('```json').strip('```').strip()

        try:
            suggested_params = json.loads(cleaned_response)
            print("Successfully parsed JSON response from Gemini.")

            # Update the config dictionary safely
            for key, value in suggested_params.items():
                if key in updated_config and isinstance(value, (int, float)): # Basic type check
                     # Add bounds checks for probabilities
                     if key.endswith("_PROB"):
                         if 0.0 <= value <= 1.0:
                             print(f"Updating {key} from {updated_config[key]} to {value}")
                             updated_config[key] = float(value)
                         else:
                             print(f"WARNING: Gemini suggested invalid probability for {key}: {value}. Keeping default.")
                     # Add more checks for other types if tuning them (e.g., ranges)
                else:
                    print(f"WARNING: Ignoring invalid/unexpected key or value type from Gemini for key '{key}': {value}")

        except json.JSONDecodeError as e:
            print(f"WARNING: Could not parse Gemini response as JSON: {e}. Using default parameters.")
            # Fallback: Very basic regex attempt (less reliable)
            # for key in tunable_params:
            #    match = re.search(rf'"{key}"\s*:\s*([0-9.]+)', response.text)
            #    if match:
            #        try:
            #            val = float(match.group(1))
            #            if 0.0 <= val <= 1.0: # Probability check
            #                 print(f"Updating {key} via REGEX to {val}")
            #                 updated_config[key] = val
            #        except ValueError: continue # Ignore if conversion fails

    except Exception as e:
        print(f"WARNING: Error during Gemini API call: {e}. Using default parameters.")

    print("--- Parameter tuning attempt complete ---")
    return updated_config # Return the potentially updated config

# --- Helper Functions (generate_location_around, etc. - UNCHANGED) ---
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

# <<< *** GET TUNED PARAMETERS *** >>>
config = tune_parameters_with_gemini(config) # Update global config dictionary

# Use tuned parameters from the 'config' dictionary hereafter
print("\n--- Starting Simulation with Final Parameters ---")
for key, value in config.items():
    # Avoid printing sensitive info like API keys if they were accidentally added
    if "KEY" not in key.upper() and "SECRET" not in key.upper():
        print(f"Parameter: {key} = {value}")
print("---")


print("Initializing card data...")
cards_data = {}
for i in range(config["NUM_CARDS"]): # Use value from config
    card_id = f"CARD_{i:04d}"; home_lat = random.uniform(25.0, 65.0); home_lon = random.uniform(-125.0, 40.0)
    cards_data[card_id] = {"HomeLatitude": round(home_lat, 6),"HomeLongitude": round(home_lon, 6),"IsCompromised": random.random() < config["COMPROMISE_CARD_PROBABILITY"],"LastTxTimestamp": None,"LastTxLatitude": None,"LastTxLongitude": None,}
print(f"Generated {len(cards_data)} cards.")
print(f"Starting simulation, writing to '{config['OUTPUT_FILENAME']}'...")


current_time = datetime.datetime.now(datetime.timezone.utc)
transactions_generated = 0
successful_sends = 0

with open(config['OUTPUT_FILENAME'], 'w') as outfile:
    while transactions_generated < config['NUM_TRANSACTIONS']:
        card_id = random.choice(list(cards_data.keys()))
        card_info = cards_data[card_id]

        is_fraud_scenario = False
        if card_info["IsCompromised"] and random.random() < config["FRAUD_INJECTION_PROBABILITY"]: # Use config value
             is_fraud_scenario = True

        # --- Simulate Transaction ---
        # (Fraud logic now uses probabilities/ranges from the 'config' dictionary)
        amount = 0.0
        merchant_lat, merchant_lon = 0.0, 0.0
        is_fraud_flag_for_output = False

        if is_fraud_scenario:
            if random.random() < config["PERFORM_TEST_TXN_PROB"]: # Use config value
                print(f"[Fraud Tactic] Simulating TEST transaction for {card_id}")
                is_fraud_flag_for_output = True
                amount = round(random.uniform(*config["FRAUD_TEST_AMOUNT_RANGE"]), 2) # Use config value
                test_distance_km = random.uniform(config["DISTANCE_THRESHOLD_KM"] * 0.8, config["LOCATION_VARIATION_KM_FRAUD"] * 0.75)
                merchant_lat, merchant_lon = generate_location_around(card_info["HomeLatitude"], card_info["HomeLongitude"], test_distance_km)
                time_increment = random.expovariate(1.0 / (config["AVG_TRANSACTION_DELAY_SECONDS"] * 1.5))
                current_time = generate_timestamp(current_time, time_increment)
            else:
                print(f"[Fraud Tactic] Simulating MAIN fraud attempt for {card_id}")
                is_fraud_flag_for_output = True
                amount = round(random.uniform(*config["FRAUD_MAIN_AMOUNT_RANGE"]), 2) # Use config value
                fraud_distance_km = random.uniform(config["DISTANCE_THRESHOLD_KM"], config["LOCATION_VARIATION_KM_FRAUD"])
                merchant_lat, merchant_lon = generate_location_around(card_info["HomeLatitude"], card_info["HomeLongitude"], fraud_distance_km)

                if card_info["LastTxTimestamp"] and card_info["LastTxLatitude"] is not None and random.random() < config["ATTEMPT_NEAR_MISS_VELOCITY_PROB"]: # Use config value
                    last_ts, last_lat, last_lon = card_info["LastTxTimestamp"], card_info["LastTxLatitude"], card_info["LastTxLongitude"]
                    distance_km = haversine(last_lat, last_lon, merchant_lat, merchant_lon)
                    target_velocity_kmh = config["VELOCITY_THRESHOLD_KMH"] * random.uniform(0.85, 0.99)
                    if distance_km > 0 and target_velocity_kmh > 0:
                        required_time_seconds = (distance_km / target_velocity_kmh) * 3600
                        simulated_delay_seconds = max(10.0, required_time_seconds * random.uniform(1.0, 1.15))
                        print(f"[Fraud Tactic] Attempting Near Miss Velocity ({target_velocity_kmh:.0f} km/h -> delay {simulated_delay_seconds:.0f}s) for {card_id}")
                        current_time = last_ts + datetime.timedelta(seconds=simulated_delay_seconds)
                    else:
                        time_increment = random.expovariate(1.0 / config["AVG_TRANSACTION_DELAY_SECONDS"])
                        current_time = generate_timestamp(current_time, time_increment)
                else:
                    time_increment = random.expovariate(1.0 / config["AVG_TRANSACTION_DELAY_SECONDS"])
                    current_time = generate_timestamp(current_time, time_increment)
        else: # Normal Transaction
            is_fraud_flag_for_output = False
            amount = round(random.uniform(*config["NORMAL_AMOUNT_RANGE"]), 2) # Use config value
            normal_distance_km = random.uniform(0, config["LOCATION_VARIATION_KM_NORMAL"]) # Use config value
            merchant_lat, merchant_lon = generate_location_around(card_info["HomeLatitude"], card_info["HomeLongitude"], normal_distance_km)
            time_increment = random.expovariate(1.0 / config["AVG_TRANSACTION_DELAY_SECONDS"]) # Use config value
            current_time = generate_timestamp(current_time, time_increment)

        # --- Create Transaction Record (unchanged) ---
        transaction_id = str(uuid.uuid4()); merchant_id = f"MERCHANT_{random.randint(1000, 9999)}"
        transaction_data = {
            "TransactionID": transaction_id, "CardID": card_id, "Timestamp": current_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "Amount": amount, "MerchantID": merchant_id, "MerchantLatitude": merchant_lat, "MerchantLongitude": merchant_lon,
            "IsFraud": is_fraud_flag_for_output, "HomeLatitude": card_info["HomeLatitude"], "HomeLongitude": card_info["HomeLongitude"]
        }

        # --- Output / State Update (unchanged logic, uses updated current_time etc)---
        json_output = json.dumps(transaction_data); outfile.write(json_output + '\n'); transactions_generated += 1
        if transactions_generated % 100 == 0: print(f"... generated {transactions_generated}/{config['NUM_TRANSACTIONS']} transactions")
        if config['SEND_TO_KINESIS']:
            if send_to_kinesis(transaction_data, config['KINESIS_STREAM_NAME'], card_id): successful_sends += 1
            else: config['SEND_TO_KINESIS'] = False # Stop trying
        card_info["LastTxTimestamp"] = current_time; card_info["LastTxLatitude"] = merchant_lat; card_info["LastTxLongitude"] = merchant_lon

# --- End of Simulation ---
print(f"\nFinished generating {transactions_generated} auto-tuned transactions.")
print(f"Data saved to '{config['OUTPUT_FILENAME']}'")
if successful_sends > 0: print(f"Sent {successful_sends} records to Kinesis stream '{config['KINESIS_STREAM_NAME']}'.")