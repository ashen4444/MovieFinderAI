import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

COLLECTION_NAME = "MovieFinderAI"


def load_environment():
    load_dotenv()

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY missing in .env")

    return url, api_key


def connect_qdrant(url, api_key):
    return QdrantClient(
        url=url,
        api_key=api_key
    )


def create_collection():
    # Load env + connect
    url, api_key = load_environment()
    client = connect_qdrant(url, api_key)

    # Check existing collections
    existing_collections = [
        col.name for col in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing_collections:
        print(f"Collection '{COLLECTION_NAME}' already exists ✅")
        return client  # ✅ IMPORTANT

    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
        )
    )

    print(f"Collection '{COLLECTION_NAME}' created successfully 🚀")

    return client  # ✅ IMPORTANT


def test_connection():
    url, api_key = load_environment()
    client = connect_qdrant(url, api_key)

    collections = client.get_collections()
    print("Connected to Qdrant successfully ✅")
    print("Existing collections:", collections)


if __name__ == "__main__":
    test_connection()