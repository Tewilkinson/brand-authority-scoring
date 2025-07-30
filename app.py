import os
import numpy as np
import openai
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
        self.sim_w = similarity_weight
        self.pop_w = popularity_weight

    def _get_embedding(self, text: str) -> np.ndarray:
        resp = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(resp['data'][0]['embedding'])

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _fetch_popularity_score(self, keyword: str, brand: str) -> float:
        # TODO: integrate with a real SEO/popularity API
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

brand = st.text_input("Brand name", value="Nike")
keyword = st.text_input("Keyword or topic", value="air max plus")
sim_w = st.slider("Similarity weight", 0.0, 1.0, 0.8)
pop_w = st.slider("Popularity weight", 0.0, 1.0, 0.2)

if st.button("Compute Score"):
    if not brand.strip() or not keyword.strip():
        st.warning("Please enter both a brand and a keyword.")
    else:
        ranker = BrandKeywordRanker(similarity_weight=sim_w, popularity_weight=pop_w)
        result = ranker.score(brand, keyword)
        st.metric("Semantic Similarity", f"{result['similarity']:.3f}")
        st.metric("Popularity Signal", f"{result['popularity']:.3f}")
        st.metric("Topical Authority Score", f"{result['score']:.1f}")

st.markdown("---")
st.markdown("Configure weights to balance pure semantic fit against external popularity metrics.")
