import os
import numpy as np
import pandas as pd
from qdrant_setup import create_collection  # your previous file
from qdrant_client.http.models import PointStruct

# --- CONFIG ---
BATCH_SIZE = 500  # Adjust based on free tier limits
EMBEDDINGS_FILE = "../data/embeddings/movie_embeddings.npy"
METADATA_FILE = "../data/embeddings/movie_embeddings_metadata.csv"
COLLECTION_NAME = "MovieFinderAI"

# --- LOAD DATA ---
embeddings = np.load(EMBEDDINGS_FILE)
metadata_df = pd.read_csv(METADATA_FILE)

print(f"Loaded embeddings: {embeddings.shape}")
print(f"Loaded metadata: {metadata_df.shape}")

# --- INIT CLIENT ---
client = create_collection()

# --- FUNCTION TO BATCH UPLOAD ---
def upload_batches():
    num_points = len(embeddings)
    print(f"Uploading {num_points} vectors in batches of {BATCH_SIZE}...")

    for start_idx in range(0, num_points, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_points)
        batch_embeddings = embeddings[start_idx:end_idx]
        batch_metadata = metadata_df.iloc[start_idx:end_idx]

        points = [
            PointStruct(
                id=int(idx),
                vector=batch_embeddings[i].tolist(),
                payload=batch_metadata.iloc[i].to_dict()
            )
            for i, idx in enumerate(range(start_idx, end_idx))
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print(f"Uploaded batch {start_idx} → {end_idx}")

    print("All embeddings uploaded successfully ✅")

# --- RUN UPLOAD ---
if __name__ == "__main__":
    upload_batches()