import json
import base64
import boto3
import os
import math
from datetime import datetime, timezone, timedelta

# --- Initialize AWS Clients (outside handler for reuse) ---
dynamodb = boto3.resource('dynamodb')
sns_client = boto3.client('sns')

# --- Read Environment Variables ---
# These MUST be set in the Lambda function's configuration (template.yaml)
TABLE_NAME = os.environ.get('DYNAMODB_TABLE_NAME')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')

if not TABLE_NAME:
    raise ValueError("Environment variable DYNAMODB_TABLE_NAME is not set")
if not SNS_TOPIC_ARN:
    raise ValueError("Environment variable SNS_TOPIC_ARN is not set")

# --- Configuration / Fraud Rules ---
DISTANCE_THRESHOLD_KM = 500    # Max distance from home considered potentially normal
VELOCITY_THRESHOLD_KMH = 800 # Max travel speed considered plausible
AMOUNT_THRESHOLD_HIGH = 1000 # High amount that might warrant suspicion regardless

# Get DynamoDB Table object
try:
    table = dynamodb.Table(TABLE_NAME)
except Exception as e:
    print(f"Error getting DynamoDB table object: {e}")
    # If we can't get the table, we probably can't proceed
    raise

# --- Helper Functions ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the earth (in km)."""
    R = 6371 # Earth radius in kilometers
    try:
        dLat = math.radians(float(lat2) - float(lat1))
        dLon = math.radians(float(lon2) - float(lon1))
        lat1 = math.radians(float(lat1))
        lat2 = math.radians(float(lat2))
        a = math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance
    except (TypeError, ValueError) as e:
        print(f"Error calculating Haversine distance ({lat1},{lon1} to {lat2},{lon2}): {e}")
        return None # Return None or handle error appropriately


def parse_timestamp(timestamp_str):
    """Parses ISO 8601 timestamp string (handles 'Z') into an aware datetime object."""
    try:
        # Handle Z timezone designator correctly
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        return datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError) as e:
        print(f"Error parsing timestamp '{timestamp_str}': {e}")
        return None


def lambda_handler(event, context):
    """
    Processes transaction records from Kinesis, checks for fraud based on
    location/velocity rules, updates state in DynamoDB, and sends alerts via SNS.
    """
    print(f"Received event with {len(event.get('Records', []))} records.")

    processed_count = 0
    fraud_detected_count = 0

    for record in event.get('Records', []):
        try:
            # 1. Decode and Parse Kinesis Record
            payload_decoded = base64.b64decode(record['kinesis']['data']).decode('utf-8')
            transaction = json.loads(payload_decoded)
            print(f"Processing TransactionID: {transaction.get('TransactionID')}")

            # Extract key fields
            card_id = transaction.get('CardID')
            tx_timestamp_str = transaction.get('Timestamp')
            tx_amount = transaction.get('Amount')
            tx_lat = transaction.get('MerchantLatitude')
            tx_lon = transaction.get('MerchantLongitude')
            home_lat = transaction.get('HomeLatitude') # Assuming this comes from simulator
            home_lon = transaction.get('HomeLongitude') # Assuming this comes from simulator

            # Validate necessary fields
            if not all([card_id, tx_timestamp_str, tx_amount, tx_lat, tx_lon, home_lat, home_lon]):
                print(f"Skipping record due to missing required fields: {transaction.get('TransactionID')}")
                continue

            # Parse timestamp string into datetime object
            tx_timestamp = parse_timestamp(tx_timestamp_str)
            if not tx_timestamp:
                print(f"Skipping record due to invalid timestamp: {transaction.get('TransactionID')}")
                continue

            # 2. Get Last State from DynamoDB
            last_state = None
            try:
                response = table.get_item(Key={'CardID': card_id})
                last_state = response.get('Item')
                if last_state:
                    print(f"Found previous state for CardID: {card_id}")
                else:
                    print(f"No previous state found for CardID: {card_id}")
            except Exception as e:
                print(f"Error getting state from DynamoDB for {card_id}: {e}")
                # Continue processing, but velocity check might be skipped

            # 3. Calculate Features (Distance, Velocity)
            distance_from_home = haversine(home_lat, home_lon, tx_lat, tx_lon)
            velocity_kmh = 0.0
            time_delta_seconds = 0.0

            if last_state:
                last_tx_lat = last_state.get('LastTxLatitude')
                last_tx_lon = last_state.get('LastTxLongitude')
                last_tx_timestamp_str = last_state.get('LastTxTimestamp')

                if all([last_tx_lat, last_tx_lon, last_tx_timestamp_str]):
                    last_tx_timestamp = parse_timestamp(last_tx_timestamp_str)

                    if last_tx_timestamp:
                        time_delta = tx_timestamp - last_tx_timestamp
                        time_delta_seconds = time_delta.total_seconds()

                        if time_delta_seconds > 0: # Avoid division by zero and moving back in time
                            distance_delta_km = haversine(last_tx_lat, last_tx_lon, tx_lat, tx_lon)
                            if distance_delta_km is not None:
                                velocity_kmh = (distance_delta_km / time_delta_seconds) * 3600 # km/h
                        else:
                             print(f"Warning: Current transaction timestamp is not after the last recorded timestamp for {card_id}. Cannot calculate velocity.")
                             # Optionally flag this as suspicious itself?


            print(f"CardID: {card_id}, DistFromHome: {distance_from_home:.2f} km, Velocity: {velocity_kmh:.2f} km/h")

            # 4. Apply Fraud Rules
            is_potentially_fraud = False
            fraud_reason = []

            if distance_from_home is not None and distance_from_home > DISTANCE_THRESHOLD_KM:
                is_potentially_fraud = True
                fraud_reason.append(f"Distance from home ({distance_from_home:.0f} km) exceeds threshold ({DISTANCE_THRESHOLD_KM} km).")

            if velocity_kmh > VELOCITY_THRESHOLD_KMH:
                is_potentially_fraud = True
                fraud_reason.append(f"Velocity ({velocity_kmh:.0f} km/h) exceeds threshold ({VELOCITY_THRESHOLD_KMH} km/h).")

            if float(tx_amount) > AMOUNT_THRESHOLD_HIGH:
                # Optionally add high amount as a factor, could make it a separate rule or increase suspicion
                if not is_potentially_fraud: # Only flag if not already flagged for other reasons (adjust logic as needed)
                     # is_potentially_fraud = True
                     # fraud_reason.append(f"High transaction amount ({tx_amount:.2f}) warrants review.")
                     pass # Keep it simple for now, focus on location/velocity
                print(f"Note: High transaction amount detected ({tx_amount:.2f})")


            # 5. Send Alert via SNS (if fraud detected)
            if is_potentially_fraud:
                fraud_detected_count += 1
                alert_message = f"Potential Fraud Detected for CardID: {card_id}\n"
                alert_message += f"Transaction ID: {transaction.get('TransactionID')}\n"
                alert_message += f"Timestamp: {tx_timestamp_str}\n"
                alert_message += f"Amount: {tx_amount:.2f}\n"
                alert_message += f"Merchant Location: ({tx_lat}, {tx_lon})\n"
                alert_message += f"Distance from Home: {distance_from_home:.0f} km\n"
                alert_message += f"Calculated Velocity: {velocity_kmh:.0f} km/h\n"
                alert_message += f"Reason(s): {' '.join(fraud_reason)}\n"

                print(f"ALERT: {alert_message}") # Log the alert

                try:
                    sns_client.publish(
                        TopicArn=SNS_TOPIC_ARN,
                        Message=alert_message,
                        Subject=f"Potential Fraud Alert - Card {card_id}"
                    )
                    print(f"Successfully published alert to SNS for {card_id}")
                except Exception as e:
                    print(f"Error publishing alert to SNS for {card_id}: {e}")
                    # Continue processing other records

            # 6. Update State in DynamoDB
            try:
                # Update the table with the current transaction details as the new 'last known state'
                table.put_item(
                    Item={
                        'CardID': card_id,
                        'LastTxTimestamp': tx_timestamp_str, # Store as string
                        'LastTxLatitude': str(tx_lat),      # Store as string for consistency
                        'LastTxLongitude': str(tx_lon),     # Store as string for consistency
                        'LastTxAmount': str(tx_amount)      # Optional: Store last amount
                        # Add other fields if needed
                    }
                )
                # print(f"Successfully updated state in DynamoDB for {card_id}")
            except Exception as e:
                print(f"Error updating state in DynamoDB for {card_id}: {e}")
                # Decide if this error should prevent further processing or just be logged


            processed_count += 1

        except json.JSONDecodeError as e:
            print(f"Error decoding/parsing JSON payload: {e}. Payload: {record.get('kinesis', {}).get('data')}")
        except Exception as e:
            # Catch-all for other unexpected errors in record processing
            print(f"Generic error processing record: {e}")
            # Consider adding more context here if possible (e.g., record data)

    print(f"Finished processing. Records processed: {processed_count}. Potential fraud detected: {fraud_detected_count}.")

    # Return value isn't typically used by Kinesis trigger on success,
    # but can be helpful for logging or debugging if needed.
    return {
        'statusCode': 200,
        'body': json.dumps(f'Successfully processed {processed_count} records. Detected {fraud_detected_count} potential fraud cases.')
    }