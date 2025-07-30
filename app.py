import os
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
import wikipedia  # pip install wikipedia

# Configuration: Set your OpenAI API key in the environment
oai_key = os.getenv('OPENAI_API_KEY')
if not oai_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize the OpenAI client
client = OpenAI(api_key=oai_key)

class BrandKeywordRanker:
    """
    Dynamically gathers brand context from Wikipedia (and optionally a Knowledge Graph),
    embeds both brand profile and keywords, and computes a robust topical authority score.
    """
    def __init__(self, similarity_weight: float = 0.8, popularity_weight: float = 0.2):
        total = similarity_weight + popularity_weight
        self.sim_w = similarity_weight / total
        self.pop_w = popularity_weight / total

    def _get_brand_profile(self, brand: str) -> str:
        """
        Fetch a brief summary of the brand from Wikipedia.
        Fallbacks to brand name if lookup fails.
        """
        try:
            return wikipedia.summary(brand, sentences=2)
        except Exception:
            return brand

    def _get_embedding(self, text: str) -> np.ndarray:
        resp = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(resp.data[0].embedding)

    def _combine_embeddings(self, embeddings: list) -> np.ndarray:
        """
        Averages a list of embeddings into a single vector.
        """
        return np.mean(np.vstack(embeddings), axis=0)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _fetch_popularity_score(self, keyword: str, brand: str) -> float:
        """
        Placeholder for search-volume or Trends-based popularity.
        TODO: integrate with real SEO API or Gemini predictions.
        """
        return 1.0

    def score(self, brand: str, keyword: str) -> dict:
        # Build brand profile
        profile = self._get_brand_profile(brand)
        # Embed brand name + profile summary
        emb_brand_name = self._get_embedding(brand)
        emb_brand_profile = self._get_embedding(profile)
        emb_brand = self._combine_embeddings([emb_brand_name, emb_brand_profile])

        # Embed keyword
        emb_kw = self._get_embedding(keyword)
        # Compute semantic similarity
        sim = self._cosine_similarity(emb_brand, emb_kw)
        # Popularity signal
        pop = self._fetch_popularity_score(keyword, brand)
        # Weighted combination
        combined = sim * self.sim_w + pop * self.pop_w
        final = min(combined, 1.0) * 100
        return {'similarity': sim, 'popularity': pop, 'score': final}

# ----- Streamlit UI ----- #
st.title("Dynamic Brand vs. Keyword Authority Scorer")

st.markdown("Enter brands & keywords (comma-separated). Scores use live Wikipedia context and OpenAI embeddings.")
brands_input = st.text_area("Brands", value="Nike, Adidas, Puma, Solero, PlayStation, Activision")
keywords_input = st.text_area("Keywords/Topics", value="new trainers, air max plus, ice creams, call of duty")

sim_w = st.slider("Similarity weight", 0.0, 1.0, 0.8)
pop_w = st.slider("Popularity weight", 0.0, 1.0, 0.2)

if st.button("Compute Scores"):
    brands = [b.strip() for b in brands_input.split(',') if b.strip()]
    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
    ranker = BrandKeywordRanker(similarity_weight=sim_w, popularity_weight=pop_w)

    results = []
    for brand in brands:
        for keyword in keywords:
            res = ranker.score(brand, keyword)
            results.append({
                'Brand': brand,
                'Keyword': keyword,
                'Semantic Similarity': f"{res['similarity']:.3f}",
                'Popularity Signal': f"{res['popularity']:.3f}",
                'Topical Authority Score': f"{res['score']:.1f}"
            })
    df = pd.DataFrame(results)
    st.dataframe(df)

st.markdown("---")
st.markdown(
    "**How it works:**\n"
    "- Pulls the first two sentences of each brand's Wikipedia page as a profile.\n"
    "- Embeds both brand name & profile, averaging them into a unified vector.\n"
    "- Embeds the keyword and measures cosine similarity.\n"
    "- (Optional) Layer in real search-volume signals via SEO/Gemini APIs.\n"
    "- Outputs a normalized 0–100 topical authority score."
)
