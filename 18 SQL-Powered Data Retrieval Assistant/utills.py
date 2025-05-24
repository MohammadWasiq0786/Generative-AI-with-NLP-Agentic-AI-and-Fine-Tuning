import requests
from sqlalchemy import text,MetaData
from config import EURI_API_KEY,EURI_API_URL , MODEL_NAME


def get_db_schema(engine):
    meta = MetaData()
    meta.reflect(bind=engine)
    schema = ""
    for table in meta.tables.values():
        schema += f"\nTable: {table.name}\nColumns: {', '.join([col.name + ' (' + str(col.type) + ')' for col in table.columns])}\n"
    return schema.strip()



def call_euri_llm(prompt):
    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }

    response = requests.post(EURI_API_URL, headers=headers, json=payload)
    response.raise_for_status()  # raise exception on error
    return response.json()["choices"][0]["message"]["content"].strip()

def execute_sql(engine, query):
    with engine.connect() as conn:
        result = conn.execute(text(query))
        return result.fetchall(), result.keys()