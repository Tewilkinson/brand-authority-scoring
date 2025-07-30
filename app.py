import os
import numpy as np
import openai
import pandas as pd
import streamlit as st

# Configuration: Set your OpenAI API key in the environment
oai_key = os.getenv('OPENAI_API_KEY')
if not oai_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()
openai.api_key = oai_key

class BrandKeywordRanker:
    """
    Computes a topical authority / ease-of-rank score (0-100)
    for a given brand and keyword/topic using semantic similarity and popularity signals.
    """
    def __init__(self, similarity_weight: float = 0.8, popularity_weight: float = 0.2):
        # Normalize weights to sum to 1
        total = similarity_weight + popularity_weight
        self.sim_w = similarity_weight / total
        self.pop_w = popularity_weight / total

    def _get_embedding(self, text: str) -> np.ndarray:
        resp = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(resp['data'][0]['embedding'])

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _fetch_popularity_score(self, keyword: str, brand: str) -> float:
        # TODO: integrate with a real SEO/popularity API (e.g., Google Trends, Ahrefs)
        return 1.0

    def score(self, brand: str, keyword: str) -> dict:
        emb_brand = self._get_embedding(brand)
        emb_kw = self._get_embedding(keyword)
        sim = self._cosine_similarity(emb_brand, emb_kw)
        pop = self._fetch_popularity_score(keyword, brand)
        combined = sim * self.sim_w + pop * self.pop_w
        final = min(combined, 1.0) * 100
        return {'similarity': sim, 'popularity': pop, 'score': final}

# ----- Streamlit UI ----- #
st.title("Brand vs. Keyword Topical Authority Scorer")

st.markdown("**Enter multiple brands and keywords (comma-separated):**")
brands_input = st.text_area("Brands", value="Nike, Adidas, Puma")
keywords_input = st.text_area("Keywords/Topics", value="new trainers, air max plus")

# Weights configuration
sim_w = st.slider("Similarity weight", 0.0, 1.0, 0.8)
pop_w = st.slider("Popularity weight", 0.0, 1.0, 0.2)

if st.button("Compute Scores"):
    brands = [b.strip() for b in brands_input.split(',') if b.strip()]
    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
    if not brands or not keywords:
        st.warning("Please enter at least one brand and one keyword.")
    else:
        ranker = BrandKeywordRanker(similarity_weight=sim_w, popularity_weight=pop_w)
        rows = []
        for brand in brands:
            for keyword in keywords:
                result = ranker.score(brand, keyword)
                rows.append({
                    'Brand': brand,
                    'Keyword': keyword,
                    'Semantic Similarity': f"{result['similarity']:.3f}",
                    'Popularity Signal': f"{result['popularity']:.3f}",
                    'Topical Authority Score': f"{result['score']:.1f}"
                })
        df = pd.DataFrame(rows)
        st.dataframe(df)

st.markdown("---")
st.markdown("Adjust weights so that similarity and popularity balance as you prefer. Weights will be normalized internally.")
