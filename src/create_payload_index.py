# create_payload_index.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient


# Load env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# Connect
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "MovieFinderAI"


# -------------------------------
# Create Indexes
# -------------------------------
def create_indexes():
    print("Creating payload indexes...")

    # vote_count index
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="vote_count",
        field_schema="integer"
    )

    # vote_average index
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="vote_average",
        field_schema="float"
    )

    print("✅ Indexes created successfully!")


if __name__ == "__main__":
    create_indexes()