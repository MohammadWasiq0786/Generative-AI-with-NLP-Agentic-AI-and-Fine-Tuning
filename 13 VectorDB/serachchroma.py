import os
import requests
from chromadb import HttpClient

from dotenv import load_dotenv
import numpy as np
load_dotenv()

EURI_API_KEY = os.getenv("EURI_API_KEY")
client = HttpClient(host="localhost",port = 8000)
collection = client.get_or_create_collection("sudh_euron_data")



# Using requests library for embeddings
import requests
import numpy as np

def generate_embeddings(text):
    url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    # Convert to numpy array for vector operations
    embedding = np.array(data['data'][0]['embedding'])
    
    print(f"Generated embedding with shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Example: Calculate vector norm
    norm = np.linalg.norm(embedding)
    print(f"Vector norm: {norm}")
    
    return embedding


def search_chroma(query_text):
    query_embed = generate_embeddings(query_text)
    
    result = collection.query(query_embeddings= [query_embed],n_results = 3 , include=["documents"])
    print(result)


search_chroma("sudhanshu has founded euron")

