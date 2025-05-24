import streamlit as st
from sqlalchemy import create_engine
from config import DATABASE_URI
from utills import get_db_schema, call_euri_llm, execute_sql

import speech_recognition as sr
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="SQL Assistant", layout="wide")
st.title("🧠 SQL-Powered Data Retrieval Assistant")

# ----------------------------
# Step 1: Choose Input Language
# ----------------------------
st.subheader("🌐 Choose Language for Speech Recognition")
language_map = {
    "English (US)": "en-US",
    "Hindi (India)": "hi-IN",
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Chinese (Mandarin)": "zh-CN",
    "Arabic": "ar-SA",
    "Bengali": "bn-IN",
    "Japanese": "ja-JP",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Marathi": "mr-IN"
}
selected_language = st.selectbox("Choose a language", list(language_map.keys()))
language_code = language_map[selected_language]

# ----------------------------
# Step 2: Speech Input
# ----------------------------
st.subheader("🎙️ Speak Your SQL Query")
nl_query = ""
if st.button("🎤 Start Listening"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(f"Listening in {selected_language}... Speak now.")
        audio = recognizer.listen(source, timeout=6)

    try:
        nl_query = recognizer.recognize_google(audio, language=language_code)
        st.success(f"You said: {nl_query}")
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand the audio.")
    except sr.RequestError as e:
        st.error(f"Speech Recognition API error: {e}")
else:
    nl_query = st.text_input("Or type your question here:")

# ----------------------------
# Step 3: Process Natural Language Query
# ----------------------------
if nl_query:
    engine = create_engine(DATABASE_URI)
    schema = get_db_schema(engine)

    with open("prompt_template.txt") as f:
        template = f.read()
    prompt = template.format(schema=schema, question=nl_query)

    with st.spinner("🧠 Generating SQL using EURI LLM..."):
        sql_query = call_euri_llm(prompt)

    st.code(sql_query, language="sql")

    try:
        results, columns = execute_sql(engine, sql_query)
        if results:
            df = pd.DataFrame(results, columns=columns)
            st.subheader("📊 Query Results:")
            st.dataframe(df, use_container_width=True)

            # ----------------------------
            # Step 4: Auto Visualization
            # ----------------------------
            st.subheader("📈 Visualization")

            if "date" in df.columns[0].lower() or "time" in df.columns[0].lower():
                fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Line Chart")
                st.plotly_chart(fig, use_container_width=True)
            elif df.shape[1] == 2:
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Bar Chart")
                st.plotly_chart(fig, use_container_width=True)
            elif df.shape[1] == 3:
                fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=df.columns[2], title="Scatter Plot")
                st.plotly_chart(fig, use_container_width=True)
            elif df.shape[1] == 1:
                fig = px.histogram(df, x=df.columns[0], title="Histogram")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Visualization not auto-detected — please refine your query for 2–3 columns.")
        else:
            st.info("✅ Query executed successfully. No data returned.")
    except Exception as e:
        st.error(f"❌ Error running query: {e}")
