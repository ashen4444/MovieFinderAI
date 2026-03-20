import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient


def load_environment():
    load_dotenv()

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY missing in .env")

    return url, api_key


def connect_qdrant(url, api_key):
    client = QdrantClient(
        url=url,
        api_key=api_key
    )
    return client


def test_connection(client):
    collections = client.get_collections()
    print("Connected to Qdrant successfully ✅")
    print("Existing collections:", collections)


def main():
    print("Connecting to Qdrant Cloud...\n")

    url, api_key = load_environment()
    client = connect_qdrant(url, api_key)

    test_connection(client)


if __name__ == "__main__":
    main()