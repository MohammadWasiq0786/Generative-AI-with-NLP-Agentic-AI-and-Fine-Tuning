import os
from euriai import EuriaiClient
from dotenv import load_dotenv

load_dotenv()

EURI_CLIENT = EuriaiClient(
    
    api_key = os.getenv("EURI_API_KEY"),
    model = "gpt-4.1-nano"
    
)