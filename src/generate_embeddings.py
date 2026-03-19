import os
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


# ----------------------------
# Configuration
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "embeddings" / "movies_for_embeddings.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"

EMBEDDINGS_FILE = OUTPUT_DIR / "movie_embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "movie_embeddings_metadata.csv"

MODEL_NAME = "text-embedding-3-small"

# Batch size = number of movie texts sent per API request
# Start conservatively for stability
BATCH_SIZE = 100

# Retry settings
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2  # seconds

# Save progress every N batches
SAVE_EVERY_N_BATCHES = 10


# ----------------------------
# Helper functions
# ----------------------------
def validate_input_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")


def create_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def load_environment() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    return api_key


def load_dataset(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    required_columns = ["id", "title", "combined_text"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["combined_text"] = df["combined_text"].fillna("").astype(str).str.strip()
    df = df[df["combined_text"] != ""].reset_index(drop=True)

    print(f"Loaded prepared dataset: {file_path}")
    print(f"Rows ready for embedding: {len(df)}")

    return df


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def chunk_list(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def request_embeddings_with_retry(
    client: OpenAI,
    texts: List[str],
    model: str,
) -> List[List[float]]:
    delay = INITIAL_RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=model,
                input=texts
            )

            # Important: preserve correct order by sorting on index
            data_sorted = sorted(response.data, key=lambda x: x.index)
            embeddings = [item.embedding for item in data_sorted]

            return embeddings

        except Exception as e:
            print(f"\nEmbedding request failed (attempt {attempt}/{MAX_RETRIES}): {e}")

            if attempt == MAX_RETRIES:
                raise RuntimeError("Max retries reached. Embedding generation failed.") from e

            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2

    raise RuntimeError("Unexpected retry failure.")


def save_embeddings(embeddings: np.ndarray, output_path: Path) -> None:
    np.save(output_path, embeddings)


def save_metadata(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8")


def save_progress(
    embeddings_list: List[List[float]],
    metadata_df: pd.DataFrame,
    embeddings_path: Path,
    metadata_path: Path
) -> None:
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    save_embeddings(embeddings_array, embeddings_path)
    save_metadata(metadata_df, metadata_path)


def generate_embeddings(df: pd.DataFrame, client: OpenAI) -> np.ndarray:
    texts = df["combined_text"].tolist()
    text_batches = chunk_list(texts, BATCH_SIZE)

    all_embeddings: List[List[float]] = []

    print(f"\nGenerating embeddings using model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total batches: {len(text_batches)}")

    for batch_index, batch_texts in enumerate(tqdm(text_batches, desc="Embedding batches"), start=1):
        batch_embeddings = request_embeddings_with_retry(
            client=client,
            texts=batch_texts,
            model=MODEL_NAME
        )

        all_embeddings.extend(batch_embeddings)

        if batch_index % SAVE_EVERY_N_BATCHES == 0:
            partial_df = df.iloc[:len(all_embeddings)].copy()
            save_progress(
                embeddings_list=all_embeddings,
                metadata_df=partial_df,
                embeddings_path=EMBEDDINGS_FILE,
                metadata_path=METADATA_FILE
            )
            print(f"\nPartial progress saved after batch {batch_index}")

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    return embeddings_array


def validate_final_output(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    if len(df) != len(embeddings):
        raise ValueError(
            f"Row count mismatch: dataset has {len(df)} rows but embeddings have {len(embeddings)} vectors"
        )

    print("\nValidation successful.")
    print(f"Movies: {len(df)}")
    print(f"Embeddings shape: {embeddings.shape}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    print("Starting MovieFinderAI embedding generation stage...\n")

    validate_input_file(INPUT_CSV)
    create_output_dir(OUTPUT_DIR)

    api_key = load_environment()
    client = get_client(api_key)

    df = load_dataset(INPUT_CSV)

    embeddings = generate_embeddings(df, client)

    validate_final_output(df, embeddings)

    save_embeddings(embeddings, EMBEDDINGS_FILE)
    save_metadata(df, METADATA_FILE)

    print(f"\nEmbeddings saved to: {EMBEDDINGS_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")

    print("\nStep 3 completed successfully.")
    print("Next step: loading saved embeddings and testing semantic similarity search.")


if __name__ == "__main__":
    main()