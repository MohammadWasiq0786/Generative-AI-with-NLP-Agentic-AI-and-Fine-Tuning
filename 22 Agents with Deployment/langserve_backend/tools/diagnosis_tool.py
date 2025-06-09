from langchain.tools import tool
from utils.euri_client import euri_chat_completion

@tool
def ai_diagnos(symptom_description: str) -> str:
    """Use EURI AI model to diagnose based on symptoms."""
    messages = [
        {"role": "user", "content": f"A patient reports: {symptom_description}. What are the possible diagnoses and steps to take?"}
    ]
    return euri_chat_completion(messages)
