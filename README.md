# 🎬 MovieFinderAI 

![Typing SVG](https://readme-typing-svg.herokuapp.com?color=00FFAA\&size=24\&center=true\&vCenter=true\&width=900\&lines=AI-Powered+Movie+Recommendation+System;Hybrid+RAG+%2B+Semantic+Search;Netflix-Level+Recommendations+with+LLMs)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![AI](https://img.shields.io/badge/AI-LLM%20%2B%20RAG-purple)
![Vector DB](https://img.shields.io/badge/Vector%20DB-Qdrant-red)
![Embeddings](https://img.shields.io/badge/Embeddings-OpenAI-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-Educational-lightgrey)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-orange)

---

## 🚀 Overview

**MovieFinderAI** is an intelligent movie recommendation system that uses
**Semantic Search + Hybrid RAG (Retrieval-Augmented Generation)** to deliver highly relevant movie suggestions from natural language queries.

Unlike traditional systems, it understands **meaning**, not just keywords.

---

## 🔥 Features

* 🔍 Semantic search using vector embeddings
* 🧠 Hybrid RAG architecture (Retrieval + LLM reasoning)
* 🎯 Dynamic filtering (vote average & popularity)
* 🤖 LLM reranking for high-quality results
* 🛟 Smart LLM fallback when no matches found
* 📊 Scalable dataset (18,000+ movies)

---

## 🏗️ System Architecture

```
User Query
     ↓
Embedding Generation (OpenAI)
     ↓
Vector Search (Qdrant)
     ↓
Filtering + Scoring
     ↓
LLM Reranking (Strong Matches)
     ↓
LLM Fallback (Weak Matches)
     ↓
Final Structured Response
```

---

## 🧰 Tech Stack

* **Language:** Python
* **Embeddings:** OpenAI (1536-dimension vectors)
* **Vector Database:** Qdrant
* **Dataset:** TMDB Movies Dataset
* **LLM:** GPT-based model

---

## 📂 Project Structure

```
MovieFinderAI/
│
├── data/
│   └── TMDB_all_movies.csv
│
├── src/
│   ├── create_payload_index.py
│   ├── data_preprocessing.py
│   ├── generate_embeddings.py
│   ├── hybrid_search.py
│   ├── qdrant_service.py
│   ├── qdrant_setup.py
│   └── qdrant_upload.py
│
├── .env
├── app.py
├── requirements.txt
└── README.md
```
**Note for users:**  
The `embeddings/` folder is generated locally when you run `embedding_generation.py` and is ignored from GitHub. You don’t need to download or commit it.

---

## ⚙️ Setup & Usage Instructions (All-in-One)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/ashen4444/MovieFinderAI.git
cd MovieFinderAI
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

---

## 🏃‍♂️ Step-by-Step Usage 

### 1️⃣ Data Preprocessing

```bash
python src/data_preprocessing.py
# Cleans and prepares the movie dataset (TMDB_all_movies.csv) for embeddings and indexing.
```

---
### 2️⃣ Generate Embeddings

```bash
python src/generate_embeddings.py
# Converts movies into vector embeddings.
# Creates the local 'embeddings/' folder (ignored from GitHub).
```
---
### 3️⃣ Qdrant Setup

```bash
python src/qdrant_setup.py
# Creates the Qdrant collection for storing embeddings.
```

---
### 4️⃣ Upload Embeddings to Qdrant

```bash
python src/qdrant_upload.py
# Uploads the generated embeddings to the Qdrant vector database.
```
---
### 5️⃣ (Optional) Create Payload Index

```bash
python src/create_payload_index.py
# Prepares additional metadata indexing in Qdrant for faster filtering.
```
---
### 6️⃣Run Hybrid Search (Streamlit Interface)

```bash
streamlit run app.py
# Launches the web interface for querying movies.
# Handles both vector search and LLM fallback.
```
---

## 💡 Example Input

```
A war sniper struggling with trauma after returning home
```

---

## 📤 Example Output

```json
{
  "title": "American Sniper",
  "overview": "A Navy SEAL sniper struggles to adjust to civilian life...",
  "vote_average": 7.4,
  "reason": "Matches war theme, sniper role, and post-war psychological struggle"
}
```

---

## 🧠 Key Concepts

### 🔹 Semantic Search

Transforms user input into embeddings and finds similar movies using vector similarity.

### 🔹 Hybrid Retrieval

Combines:

* Vector similarity (Qdrant)
* Metadata filtering
* LLM reasoning

### 🔹 Reranking

Improves top results using LLM to enhance relevance and accuracy.

### 🔹 LLM Fallback

Generates intelligent recommendations when database results are weak or unavailable.

---

## 📈 Current Progress

* ✅ 18,000+ movies indexed
* ✅ Embeddings generated
* ✅ Qdrant integration complete
* ✅ Hybrid search working
* ✅ LLM fallback implemented

---

## 🔮 Future Improvements

* 🎬 Top 5 recommendations instead of single output
* 🌐 Web UI (Streamlit / React)
* 🧠 Personalized recommendations
* ⚡ Faster reranking pipeline
* 📊 Analytics dashboard

---

## 🤝 Contribution

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## 📜 License

This project is for educational and research purposes.

---

## 💡 Author

**Ashen Wijethilaka**
AI & Deep Learning Enthusiast

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
