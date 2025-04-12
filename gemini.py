import os
import google.generativeai as genai
# from dotenv import load_dotenv # Optional: for loading API key from .env file

# --- Configuration ---
# Option 1: Load from environment variable (Recommended)
# load_dotenv() # Uncomment if using python-dotenv
# api_key = os.getenv("GEMINI_API_KEY")

# Option 2: Hardcode (Use only for quick testing, NOT recommended)
api_key = "AIzaSyC-kWSElvhl1gjNRNBfNgOL7u4EIusb-aw" # <-- PASTE YOUR KEY HERE

if not api_key:
    raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable or hardcode.")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    print("Gemini Configured.")
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    exit()

# --- Prompts to Ask Gemini ---

prompt_1 = """
Describe 3 plausible but potentially subtle ways stolen credit card data
(where the fraudster knows the cardholder's rough home location) could be
used to test the card or make initial fraudulent purchases *before* large-value
fraud. Focus on variations in transaction location, timing, and amount that
might evade basic distance-from-home or high-amount rules.
"""

prompt_2 = """
Imagine a credit card fraud detection system flags transactions based on:
1. Distance from cardholder's home > 500 km
2. Calculated velocity between consecutive transactions > 800 km/h

Suggest 2 distinct ways a fraudster might try to make multiple fraudulent
transactions while attempting to stay *just under* these specific thresholds
or making them less obvious.
"""

prompt_3 = """
Think about 'next-generation' credit card fraud tactics related to
Card-Not-Present transactions online. Describe 1-2 scenarios that combine
multiple techniques (e.g., location spoofing/proxies, slightly modified
shipping addresses, small initial purchases) to build trust or bypass
common checks before attempting a larger fraudulent purchase.
"""

# --- Function to Ask Gemini ---
def ask_gemini(prompt_text):
    print("-" * 30)
    print(f"Asking Gemini:\n{prompt_text}\n")
    try:
        response = model.generate_content(prompt_text)
        print(f"Gemini's Response:\n{response.text}")
        print("-" * 30 + "\n")
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("Requesting fraud pattern ideas from Gemini...\n")
    ideas_1 = ask_gemini(prompt_1)
    ideas_2 = ask_gemini(prompt_2)
    ideas_3 = ask_gemini(prompt_3)

    print("\nUse these ideas to manually enhance the simulation logic in")
    print("'simulate_transactions_enhanced.py'")