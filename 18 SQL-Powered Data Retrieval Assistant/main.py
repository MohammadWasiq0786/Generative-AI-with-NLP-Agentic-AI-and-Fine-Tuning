import streamlit as st
from sqlalchemy import create_engine
from config import DATABASE_URI
from utills import get_db_schema, call_euri_llm, execute_sql


st.set_page_config(page_title="SQL Assistant", layout="wide")
st.title("🧠 SQL-Powered Data Retrieval Assistant")

nl_query = st.text_input("Ask your question (in natural language):")

if nl_query:
    engine = create_engine(DATABASE_URI)
    schema = get_db_schema(engine)

    with open("prompt_template.txt") as f:
        template = f.read()
    prompt = template.format(schema=schema, question=nl_query)

    with st.spinner("Generating SQL using EURI LLM..."):
        sql_query = call_euri_llm(prompt)

    st.code(sql_query, language="sql")

    try:
        results, columns = execute_sql(engine, sql_query)
        if results:
            st.dataframe(results, use_container_width=True)
        else:
            st.info("Query executed successfully. No data returned.")
    except Exception as e:
        st.error(f"Error running query: {e}")
