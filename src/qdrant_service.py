# services/qdrant_service.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range

# -------------------------------
# 1. Load Environment Variables
# -------------------------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("Missing QDRANT credentials in .env file")

# -------------------------------
# 2. Initialize Clients
# -------------------------------
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

openai_client = OpenAI()

COLLECTION_NAME = "MovieFinderAI"

# -------------------------------
# 3. Generate Query Embedding
# -------------------------------
def get_query_embedding(query: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",   # 🔥 upgraded (recommended)
        input=query
    )
    return response.data[0].embedding


# -------------------------------
# 4. Search + Format (MERGED)
# -------------------------------
def search_movies(query: str, top_k=10):
    query_embedding = get_query_embedding(query)

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="vote_count",
                    range=Range(gte=1000)
                ),
                FieldCondition(
                    key="vote_average",
                    range=Range(gte=7.0)
                )
            ]
        )
    )

    movies = []
    for r in results.points:
        payload = r.payload

        movies.append({
            "title": payload.get("title"),
            "overview": payload.get("overview", ""),
            "vote_average": payload.get("vote_average"),
            "score": r.score
        })

    return movies