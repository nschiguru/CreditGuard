import json
import random
import uuid
import time
import datetime
import math
import boto3

# --- Configuration ---
NUM_CARDS = 50
NUM_TRANSACTIONS = 1000 # Adjust as needed
OUTPUT_FILENAME = 'simulated_transactions.jsonl' # Define the output file name
FRAUD_INJECTION_PROBABILITY = 0.03
COMPROMISE_CARD_PROBABILITY = 0.10
AVG_TRANSACTION_DELAY_SECONDS = 0.1 # Decrease for faster file generation
LOCATION_VARIATION_KM_NORMAL = 50
LOCATION_VARIATION_KM_FRAUD = 5000
NORMAL_AMOUNT_RANGE = (5.00, 250.00)
FRAUD_AMOUNT_RANGE = (100.00, 2000.00)
HIGH_VELOCITY_THRESHOLD_KMH = 800

# Optional: Kinesis Configuration
KINESIS_STREAM_NAME = 'your-fraud-detection-stream'
# ** SET TO FALSE IF YOU PRIMARILY WANT FILE OUTPUT **
SEND_TO_KINESIS = False

# --- Helper Functions ---
# (Keep generate_location_around, haversine, generate_timestamp, send_to_kinesis - unchanged)
def generate_location_around(lat, lon, max_distance_km):
    """Generates a random coordinate within a certain distance (approximation)."""
    radius_deg_lat = max_distance_km / 111.1
    radius_deg_lon = max_distance_km / (111.1 * math.cos(math.radians(lat)))
    delta_lat = random.uniform(-radius_deg_lat, radius_deg_lat)
    delta_lon = random.uniform(-radius_deg_lon, radius_deg_lon)
    new_lat = max(-90.0, min(90.0, lat + delta_lat))
    new_lon = ((lon + delta_lon + 180) % 360) - 180
    return round(new_lat, 6), round(new_lon, 6)

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the earth."""
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    a = math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def generate_timestamp(start_time, increment_seconds):
    """Generates an increasing timestamp."""
    return start_time + datetime.timedelta(seconds=increment_seconds)

def send_to_kinesis(data_record, stream_name, partition_key):
    """Sends a single data record to Kinesis."""
    try:
        kinesis_client = boto3.client('kinesis')
        response = kinesis_client.put_record(
            StreamName=stream_name,
            Data=json.dumps(data_record),
            PartitionKey=partition_key
        )
        return True
    except Exception as e:
        print(f"Error sending to Kinesis: {e}")
        return False

# --- Main Simulation ---

print("Initializing card data...")
cards_data = {}
for i in range(NUM_CARDS):
    card_id = f"CARD_{i:04d}"
    home_lat = random.uniform(25.0, 65.0)
    home_lon = random.uniform(-125.0, 40.0)
    cards_data[card_id] = {
        "HomeLatitude": round(home_lat, 6),
        "HomeLongitude": round(home_lon, 6),
        "IsCompromised": random.random() < COMPROMISE_CARD_PROBABILITY,
        "LastTxTimestamp": None,
        "LastTxLatitude": None,
        "LastTxLongitude": None,
    }

print(f"Generated {len(cards_data)} cards.")
print(f"Starting transaction simulation, writing to '{OUTPUT_FILENAME}'...")

current_time = datetime.datetime.now(datetime.timezone.utc)
transactions_generated = 0

# Open the output file in write mode ('w')
# Using 'with' ensures the file is properly closed even if errors occur
with open(OUTPUT_FILENAME, 'w') as outfile:
    for i in range(NUM_TRANSACTIONS):
        card_id = random.choice(list(cards_data.keys()))
        card_info = cards_data[card_id]

        is_fraud = False
        amount = 0.0
        merchant_lat = 0.0
        merchant_lon = 0.0

        # Determine if this transaction is fraudulent (Logic unchanged)
        if card_info["IsCompromised"] and random.random() < FRAUD_INJECTION_PROBABILITY:
            is_fraud = True
            amount = round(random.uniform(*FRAUD_AMOUNT_RANGE), 2)
            merchant_lat, merchant_lon = generate_location_around(
                card_info["HomeLatitude"],
                card_info["HomeLongitude"],
                random.uniform(LOCATION_VARIATION_KM_NORMAL * 2, LOCATION_VARIATION_KM_FRAUD)
            )
            # Optional velocity check/forcing (Logic unchanged)
            if card_info["LastTxTimestamp"] and card_info["LastTxLatitude"]:
                 time_delta_seconds = (current_time - card_info["LastTxTimestamp"]).total_seconds()
                 if time_delta_seconds > 0:
                     distance_km = haversine(
                         card_info["LastTxLatitude"], card_info["LastTxLongitude"],
                         merchant_lat, merchant_lon
                     )
                     velocity_kmh = (distance_km / time_delta_seconds) * 3600
                     if velocity_kmh < HIGH_VELOCITY_THRESHOLD_KMH and random.random() < 0.5:
                          required_time_seconds = (distance_km / HIGH_VELOCITY_THRESHOLD_KMH) * 3600
                          simulated_short_delay = max(1, random.uniform(required_time_seconds * 0.5, required_time_seconds * 0.9))
                          current_time = card_info["LastTxTimestamp"] + datetime.timedelta(seconds=simulated_short_delay)
                 else:
                     time_increment = random.expovariate(1.0 / AVG_TRANSACTION_DELAY_SECONDS)
                     current_time = generate_timestamp(current_time, time_increment)

        else: # Normal transaction (Logic unchanged)
            is_fraud = False
            amount = round(random.uniform(*NORMAL_AMOUNT_RANGE), 2)
            merchant_lat, merchant_lon = generate_location_around(
                card_info["HomeLatitude"],
                card_info["HomeLongitude"],
                random.uniform(0, LOCATION_VARIATION_KM_NORMAL)
            )
            time_increment = random.expovariate(1.0 / AVG_TRANSACTION_DELAY_SECONDS)
            current_time = generate_timestamp(current_time, time_increment)


        transaction_id = str(uuid.uuid4())
        merchant_id = f"MERCHANT_{random.randint(1000, 9999)}"

        # Create the transaction record (Schema unchanged)
        transaction_data = {
            "TransactionID": transaction_id,
            "CardID": card_id,
            "Timestamp": current_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "Amount": amount,
            "MerchantID": merchant_id,
            "MerchantLatitude": merchant_lat,
            "MerchantLongitude": merchant_lon,
            "IsFraud": is_fraud,
            "HomeLatitude": card_info["HomeLatitude"],
            "HomeLongitude": card_info["HomeLongitude"]
        }

        # --- Output ---
        # Convert the dictionary to a JSON string
        json_output = json.dumps(transaction_data)

        # Write the JSON string to the file, followed by a newline character
        outfile.write(json_output + '\n')

        transactions_generated += 1
        # Optional: Print progress to console
        if transactions_generated % 100 == 0:
            print(f"... generated {transactions_generated}/{NUM_TRANSACTIONS} transactions")


        # Optional: Send to Kinesis
        if SEND_TO_KINESIS:
            if not send_to_kinesis(transaction_data, KINESIS_STREAM_NAME, card_id):
                 print(f"Failed to send TXN {transaction_id} to Kinesis. Stopping.")
                 # break # Optional: Stop if Kinesis fails

        # --- Update card state --- (Logic unchanged)
        card_info["LastTxTimestamp"] = current_time
        card_info["LastTxLatitude"] = merchant_lat
        card_info["LastTxLongitude"] = merchant_lon

        # --- Simulate Delay --- (Optional, can be removed for faster file writing)
        # delay = max(0.01, random.gauss(AVG_TRANSACTION_DELAY_SECONDS, AVG_TRANSACTION_DELAY_SECONDS / 3))
        # time.sleep(delay)

# End of the 'with open...' block, file is automatically closed here

print(f"\nFinished generating {transactions_generated} transactions.")
print(f"Data saved to '{OUTPUT_FILENAME}'")