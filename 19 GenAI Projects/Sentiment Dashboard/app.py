import streamlit as st
import requests

# ==== Configuration ====
EURI_API_URL = "https://api.euron.one/api/v1/euri/alpha/chat/completions"
EURI_API_KEY =  "EURI_API_KEY" # Replace with your actual API key
MODEL = "gpt-4.1-nano"

# ==== Prompt Template ====
def build_sentiment_prompt(text):
    return f"""
You are a professional sentiment analysis system. Analyze the following text and classify its sentiment as one of the following:
- Positive
- Negative
- Neutral

Also, explain the reasoning briefly.

Text:
\"\"\"{text}\"\"\"

Respond strictly in the following JSON format:
{{
  "sentiment": "...",
  "reason": "..."
}}
"""

# ==== Function to Call EURI API ====
def get_sentiment_analysis(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a professional sentiment analysis bot."},
            {"role": "user", "content": build_sentiment_prompt(text)}
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }

    response = requests.post(EURI_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            result = response.json()["choices"][0]["message"]["content"]
            return result
        except Exception as e:
            return f"⚠️ Error: {e}"
    else:
        return f"❌ Failed: {response.status_code} - {response.text}"

# ==== Streamlit UI ====
st.set_page_config(page_title="📊 Sentiment Analysis Dashboard", layout="centered")
st.title("📊 Real-Time Sentiment Tracker")
st.markdown("Enter any text and get real-time sentiment analysis using **EURI AI**.")

text_input = st.text_area("📝 Enter your text here:")

if st.button("🔍 Analyze Sentiment"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            sentiment_output = get_sentiment_analysis(text_input)
            st.success("✅ Analysis Complete")
            st.json(sentiment_output)
    else:
        st.warning("⚠️ Please enter some text.")
