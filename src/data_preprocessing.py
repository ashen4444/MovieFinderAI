import pandas as pd

# =========================
# Step 1: Load Dataset
# =========================
df = pd.read_csv("../data/TMDB_all_movies.csv")

# =========================
# Step 2: Basic Cleaning
# =========================
# Remove nulls
df = df.dropna(subset=["title", "overview", "release_date"])

# Convert release_date to datetime
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df = df.dropna(subset=["release_date"])

# Extract year
df["year"] = df["release_date"].dt.year

# Filter year >= 2000
df = df[df["year"] >= 2000]

# Filter vote_count >= 30
df = df[df["vote_count"] >= 30]

# (Optional) English only
df = df[df["original_language"] == "en"]

# =========================
# Step 3: Select Columns
# =========================
final_columns = [
    "id",
    "title",
    "original_title",
    "overview",
    "genres",
    "tagline",
    "cast",
    "director",
    "writers",
    "release_date",
    "year",
    "vote_average",
    "vote_count",
    "popularity",
    "imdb_rating",
    "imdb_votes",
    "original_language",
    "runtime"
]

df = df[final_columns]

# =========================
# Step 4: Create Combined Text
# =========================
text_columns = ["title", "genres", "tagline", "overview", "cast", "director", "writers"]

for col in text_columns:
    df[col] = df[col].fillna("").astype(str).str.strip()

def build_text(row):
    parts = []

    if row["title"]:
        parts.append("Title: " + row["title"])

    if row["genres"]:
        parts.append("Genres: " + row["genres"])

    if row["tagline"]:
        parts.append("Tagline: " + row["tagline"])

    if row["overview"]:
        parts.append("Overview: " + row["overview"])

    if row["cast"]:
        parts.append("Cast: " + row["cast"])

    if row["director"]:
        parts.append("Director: " + row["director"])

    if row["writers"]:
        parts.append("Writers: " + row["writers"])

    return ". ".join(parts).replace("..", ".")

df["combined_text"] = df.apply(build_text, axis=1)
# =========================
# Step 5: Save Cleaned Dataset
# =========================
df.to_csv("../data/cleaned_movies.csv", index=False)

# =========================
# Step 6: Debug Output
# =========================
print("Cleaned dataset saved successfully.")
print("Final movie count:", len(df))
print("\nSample combined text:\n")
print(df["combined_text"].iloc[0])