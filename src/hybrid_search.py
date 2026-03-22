# hybrid_search.py

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
SIMILARITY_THRESHOLD = 0.75


# -------------------------------
# 3. LLM Reranking
# -------------------------------
def rerank_movies(query, movies):
    movie_list_text = "\n".join([
        f"{i+1}. {m['title']} - {m['overview'][:120]}"
        for i, m in enumerate(movies)
    ])

    prompt = f"""
User query:
"{query}"

Candidate movies:
{movie_list_text}

Task:
Select the BEST {FINAL_RESULTS} movies that match the query.

Return ONLY JSON:
[
  {{"title": "...", "reason": "..."}}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return []


# -------------------------------
# 4. LLM Fallback
# -------------------------------
def llm_fallback(query):
    prompt = f"""
User wants movie recommendations:

"{query}"

Suggest 5 highly relevant movies.

Return ONLY JSON:
[
  {{"title": "...", "description": "..."}}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return []


# -------------------------------
# 5. Hybrid Search Pipeline
# -------------------------------
def hybrid_search(query):
    print("\n🔍 Searching movies...")

    results = search_movies(query, top_k=TOP_K)

    # No DB results at all
    if not results:
        print("⚠️ No DB results → LLM fallback")
        return {
            "source": "LLM_ONLY",
            "results": llm_fallback(query)
        }

    top_score = results[0]["score"]
    print(f"📊 Top Score: {top_score:.4f}")

    # -------------------------
    # Strong Match → Rerank
    # -------------------------
    if top_score >= SIMILARITY_THRESHOLD:
        print("✅ Strong match → Reranking")

        reranked = rerank_movies(query, results)

        return {
            "source": "HYBRID_RAG",
            "results": reranked,
            "db_results": results[:FINAL_RESULTS]
        }

    # -------------------------
    # Weak Match → Fallback
    # -------------------------
    else:
        print("⚠️ Weak match → Using LLM fallback")

        return {
            "source": "LLM_FALLBACK",
            "db_results": results[:FINAL_RESULTS],
            "llm_results": llm_fallback(query)
        }


# -------------------------------
# 6. CLI Runner
# -------------------------------
if __name__ == "__main__":
    query = input("🎬 Enter movie description: ")

    output = hybrid_search(query)

    print("\n================ RESULT ================\n")
    print(json.dumps(output, indent=2))