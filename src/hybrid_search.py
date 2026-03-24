import json
from openai import OpenAI
from src.qdrant_service import search_movies

# -------------------------------
# 1. Initialize OpenAI
# -------------------------------
client = OpenAI()

# -------------------------------
# 2. Config
# -------------------------------
TOP_K = 10
FINAL_RESULTS = 5
SIMILARITY_THRESHOLD = 0.50

# -------------------------------
# 3. Safe JSON Parser
# -------------------------------
def safe_json_parse(content):
    try:
        return json.loads(content)
    except:
        content = content.strip()

        # Remove markdown formatting if present
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) > 1:
                content = parts[1]

        try:
            return json.loads(content)
        except:
            print("⚠️ LLM JSON parsing failed")
            print("RAW RESPONSE:", content)
            return []


# -------------------------------
# 4. Enrich Results with DB Metadata
# -------------------------------
def enrich_with_metadata(reranked, db_results):
    enriched = []

    for item in reranked:
        title = item.get("title")

        # Find matching movie in DB
        match = next((m for m in db_results if m["title"] == title), None)

        if match:
            enriched.append({
                "title": title,
                "reason": item.get("reason"),
                "vote_average": match.get("vote_average"),
                "overview": match.get("overview"),
                "score": match.get("score")
            })
        else:
            enriched.append(item)

    return enriched


# -------------------------------
# 5. LLM Reranking
# -------------------------------
def rerank_movies(query, movies):
    movie_list_text = "\n".join([
        f"{i+1}. {m['title']} - {m['overview'][:120]}"
        for i, m in enumerate(movies)
    ])

    prompt = f"""
You are a movie recommendation system.

User query:
"{query}"

Candidate movies:
{movie_list_text}

Task:
Select the BEST {FINAL_RESULTS} movies that match the query.

Return ONLY valid JSON (no markdown, no extra text):
[
  {{"title": "...", "reason": "..."}}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return safe_json_parse(response.choices[0].message.content)


# -------------------------------
# LLM Fallback (Final Version)
# -------------------------------
def llm_fallback(query):
    prompt = f"""
You are a movie recommendation system.

User query:
"{query}"

Return EXACTLY 5 movies.

Rules:
- Only valid JSON
- No markdown
- No extra text

Format:
[
  {{
    "title": "Movie Name",
    "overview": "Short summary of the movie",
    "vote_average": 0,
    "reason": "Why this movie matches the query"
  }}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    raw_results = safe_json_parse(response.choices[0].message.content)

    # -------------------------------
    # Normalize Output (IMPORTANT)
    # -------------------------------
    normalized_results = []

    for m in raw_results:
        normalized_results.append({
            "title": m.get("title"),
            "overview": m.get("overview", m.get("description", "")),
            "vote_average": m.get("vote_average", 0),
            "reason": m.get("reason", "Relevant to your query")
        })

    return normalized_results


# -------------------------------
# 7. Format DB Results (clean output)
# -------------------------------
def format_db_results(results):
    return [
        {
            "title": m["title"],
            "vote_average": m["vote_average"],
            "overview": m["overview"],
            "score": m["score"]
        }
        for m in results
    ]


# -------------------------------
# 8. Hybrid Search Pipeline
# -------------------------------
def hybrid_search(query):
    print("\n🔍 Searching movies...")

    results = search_movies(query, top_k=TOP_K)

    # No DB results
    if not results:
        print("❌ No DB results → LLM only")
        return {
            "source": "LLM_ONLY",
            "results": llm_fallback(query)
        }

    top_score = results[0]["score"]
    print(f"📊 Top Score: {top_score:.4f}")

    # -------------------------
    # Strong / Good Match
    # -------------------------
    if top_score >= 0.60:
        print("✅ Good match → Reranking")

        reranked = rerank_movies(query, results)
        enriched_results = enrich_with_metadata(reranked, results)

        return {
            "source": "HYBRID_RAG",
            "results": enriched_results
        }

    # -------------------------
    # Medium Match
    # -------------------------
    elif top_score >= 0.50:
        print("⚠️ Medium match → Using DB only")

        return {
            "source": "DB_ONLY",
            "results": format_db_results(results[:FINAL_RESULTS])
        }

    # -------------------------
    # Weak Match
    # -------------------------
    else:
        print("❌ Weak match → LLM fallback")

        return {
            "source": "LLM_FALLBACK",
            "results": llm_fallback(query)
        }


# -------------------------------
# 9. CLI Runner
# -------------------------------
if __name__ == "__main__":
    print("\n🎬 MovieFinder AI")
    print("------------------------")
    print("1. Search by Description")
    print("2. Search by Movie Title")

    choice = input("\n👉 Select option (1 or 2): ").strip()

    if choice == "1":
        query = input("\n📝 Enter movie description: ")
        output = hybrid_search(query)

    elif choice == "2":
        title = input("\n🎥 Enter movie title: ")

        # Force title-based behavior
        results = search_movies(title, top_k=TOP_K)

        exact_match = next(
            (m for m in results if m["title"].lower() == title.lower()),
            None
        )

        if exact_match:
            print("\n🎯 Finding similar movies...\n")

            similar_movies = [
                m for m in results
                if m["title"].lower() != title.lower()
            ]

            output = {
                "source": "SIMILAR_MOVIES",
                "input_movie": exact_match["title"],
                "results": format_db_results(similar_movies[:FINAL_RESULTS])
            }
        else:
            print("\n⚠️ Movie not found in DB → Using fallback\n")
            output = {
                "source": "LLM_FALLBACK",
                "results": llm_fallback(title)
            }

    else:
        print("❌ Invalid choice")
        exit()

    print("\n================ RESULT ================\n")
    print(json.dumps(output, indent=2))