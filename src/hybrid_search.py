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
SIMILARITY_THRESHOLD = 0.60


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
# 6. LLM Fallback
# -------------------------------
def llm_fallback(query):
    prompt = f"""
You are a movie recommendation system.

User query:
"{query}"

Return EXACTLY 5 highly relevant movies.

Rules:
- Only valid JSON
- No markdown
- No explanations outside JSON

Format:
[
  {{"title": "Movie Name", "description": "Short reason"}}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return safe_json_parse(response.choices[0].message.content)


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
    query = input("🎬 Enter movie description: ")

    output = hybrid_search(query)

    print("\n================ RESULT ================\n")
    print(json.dumps(output, indent=2))