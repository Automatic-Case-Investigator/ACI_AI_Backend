import chromadb
from chromadb.config import Settings

chromadb_client = chromadb.Client(
    Settings(
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
        chroma_server_host="chromadb",
        chroma_server_http_port=8000,
    )
)