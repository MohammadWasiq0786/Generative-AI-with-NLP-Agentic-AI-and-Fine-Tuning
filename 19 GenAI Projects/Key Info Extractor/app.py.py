import streamlit as st
import requests
import json
import pandas as pd

# === Configuration ===
EURI_API_URL = "https://api.euron.one/api/v1/euri/alpha/chat/completions"
EURI_API_KEY =  EURI_API_KEY # Replace with your actual API key
MODEL = "gpt-4.1-nano"

# === Prompt Builder ===
def build_prompt(text):
    return f"""
You are an intelligent information extractor. Extract the key entities from the following text and return them in structured JSON format.

Text:
\"\"\"{text}\"\"\"

Extract the following fields if present:
- Full Name
- Email
- Phone Number
- Date
- Address
- Company
- Designation
- Skills
- Education
- Experience
- Any other useful metadata

Respond in clean JSON format only.
"""

# === API Request ===
def extract_info(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a professional key information extractor."},
            {"role": "user", "content": build_prompt(text)}
        ],
        "max_tokens": 1000,
        "temperature": 0.3
    }

    response = requests.post(EURI_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ Error parsing response: {e}"
    else:
        return f"❌ Request failed: {response.status_code}\n{response.text}"

# === Convert JSON to Table Row ===
def convert_to_row(json_data, fields):
    row = {}
    for field in fields:
        value = json_data.get(field, "")
        if isinstance(value, list):
            row[field] = ", ".join(map(str, value))
        else:
            row[field] = value
    return row

# === Streamlit App ===
st.set_page_config(page_title="🧠 Info Extractor", layout="centered")
st.title("📄 Key Information Extractor From Text")

input_text = st.text_area("Paste unstructured text here:", height=250)

if st.button("🚀 Extract Info"):
    if input_text.strip():
        with st.spinner("Extracting info..."):
            result = extract_info(input_text)

        try:
            parsed = json.loads(result)

            # Define the fields you want to show in table
            display_fields = [
                "Full Name", "Email", "Phone Number", "Company",
                "Designation", "Skills", "Education", "Experience", "Address"
            ]

            table_row = convert_to_row(parsed, display_fields)
            df = pd.DataFrame([table_row], columns=display_fields)

            st.success("✅ Extracted Summary")
            st.dataframe(df, use_container_width=True)

        except json.JSONDecodeError:
            st.error("⚠️ Failed to parse response into valid JSON.")
            st.code(result, language="json")
    else:
        st.warning("⚠️ Please enter some input text.")
