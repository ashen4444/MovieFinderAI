import streamlit as st
from src.hybrid_search import hybrid_search

st.set_page_config(page_title="MovieFinder AI", layout="wide")

st.title("🎬 MovieFinder AI")
st.write("Find movies (released after 2000) using description or title")

# -----------------------
# Mode Selection
# -----------------------
mode = st.radio(
    "Choose Search Mode:",
    ["Search by Description", "Search by Movie Title"]
)

# -----------------------
# User Input
# -----------------------
query = st.text_input("Enter your input:")

# -----------------------
# Search Button
# -----------------------
if st.button("🔍 Search"):

    if not query:
        st.warning("Please enter something")
    else:
        with st.spinner("Searching..."):

            result = hybrid_search(query)

        st.success(f"Source: {result['source']}")

        # -----------------------
        # Display Results
        # -----------------------
        results = result.get("results", [])

        if not results:
            st.error("No results found")
        else:
            for movie in results:
                st.markdown("---")
                st.subheader(movie.get("title", "Unknown"))

                if "vote_average" in movie:
                    st.write(f"⭐ Rating: {movie['vote_average']}")

                if "reason" in movie:
                    st.write(f"🧠 Reason: {movie['reason']}")

                if "overview" in movie:
                    st.write(f"📖 {movie['overview']}")

                if "description" in movie:
                    st.write(f"📖 {movie['description']}")