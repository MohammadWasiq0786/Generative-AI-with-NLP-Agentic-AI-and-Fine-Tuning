import requests
EURION_API_KEY = "your_eurion_api_key_here"  # Replace with your actual EURION API key
API_KEY = EURION_API_KEY 
BASE_URL = "https://api.euron.one/api/v1/euri/alpha"

def euri_chat_completion(messages, model="gpt-4.1-nano", temperature=0.7, max_tokens=1000):
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]