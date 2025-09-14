import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

user_message = "Hello, how are you today?"

payload = {
    "contents": [
        {"parts": [{"text": user_message}]}
    ],
    "systemInstruction": {
        "parts": [{"text": "You are a friendly chatbot named Moo-Bot."}]
    }
}

headers = {"Content-Type": "application/json"}

try:
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    api_response = response.json()

    # Safely extract response text
    bot_reply = api_response.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")
    print("Bot reply:", bot_reply)

except requests.exceptions.RequestException as e:
    print("API request error:", e)
except Exception as e:
    print("Unexpected error:", e)
