from fastapi import FastAPI
from langserve import add_routes
from disgonostics_graph import build_graph

app = FastAPI()
graph = build_graph()

# langserve_backend/auth.py
from fastapi import Header, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "Authorization"
API_KEY = "secret-token-123"  # change this to a strong key or read from ENV

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Unauthorized access")
    return api_key


add_routes(app,graph,path="/diagnose",dependencies = [Depends(verify_api_key)])