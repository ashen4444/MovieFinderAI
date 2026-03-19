import os
from pathlib import Path

import pandas as pd


# ----------------------------
# Configuration
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "cleaned_movies.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"

REQUIRED_COLUMNS = [
    "id",
    "title",
    "combined_text",
]

MIN_TEXT_LENGTH = 10   # discard very tiny/weak texts
MAX_CHAR_LENGTH = 12000  # temporary safety trim before token-based handling later


# ----------------------------
# Helper functions
# ----------------------------
def validate_input_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")


def create_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def load_dataset(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"Loaded dataset: {file_path}")
    print(f"Total rows before validation: {len(df)}")
    return df


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def clean_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert to string safely
    df["combined_text"] = df["combined_text"].fillna("").astype(str)

    # Strip surrounding whitespace
    df["combined_text"] = df["combined_text"].str.strip()

    # Remove rows with empty combined_text
    before_empty_filter = len(df)
    df = df[df["combined_text"] != ""]
    removed_empty = before_empty_filter - len(df)

    # Remove rows with too-short text
    before_short_filter = len(df)
    df = df[df["combined_text"].str.len() >= MIN_TEXT_LENGTH]
    removed_short = before_short_filter - len(df)

    # Temporary character-length trim
    # Later we can add token-aware truncation if needed
    df["combined_text"] = df["combined_text"].str.slice(0, MAX_CHAR_LENGTH)

    print(f"Removed empty combined_text rows: {removed_empty}")
    print(f"Removed too-short combined_text rows (< {MIN_TEXT_LENGTH} chars): {removed_short}")

    return df


def remove_duplicate_movies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    removed = before - len(df)

    print(f"Removed duplicate movie IDs: {removed}")
    return df


def show_summary(df: pd.DataFrame) -> None:
    print("\nValidation summary:")
    print(f"Final rows ready for embeddings: {len(df)}")
    print(f"Columns available: {list(df.columns)}")

    print("\nSample rows:")
    print(df[["id", "title", "combined_text"]].head(3).to_string(index=False))

    text_lengths = df["combined_text"].str.len()
    print("\ncombined_text length stats:")
    print(f"Min chars: {text_lengths.min()}")
    print(f"Max chars: {text_lengths.max()}")
    print(f"Avg chars: {text_lengths.mean():.2f}")


def save_prepared_dataset(df: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "movies_for_embeddings.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nPrepared dataset saved to: {output_path}")
    return output_path


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    print("Starting MovieFinderAI embedding preparation stage...\n")

    validate_input_file(INPUT_CSV)
    create_output_dir(OUTPUT_DIR)

    df = load_dataset(INPUT_CSV)
    validate_required_columns(df, REQUIRED_COLUMNS)

    df = clean_combined_text(df)
    df = remove_duplicate_movies(df)

    show_summary(df)
    save_prepared_dataset(df, OUTPUT_DIR)

    print("\nStep 2 completed successfully.")
    print("Next step: batching + OpenAI embedding generation.")


if __name__ == "__main__":
    main()