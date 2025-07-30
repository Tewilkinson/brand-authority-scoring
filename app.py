import os
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# Configuration: Set your OpenAI API key in the environment
oai_key = os.getenv('OPENAI_API_KEY')
if not oai_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize the OpenAI client
client = OpenAI(api_key=oai_key)

class BrandKeywordRanker:
    """
    Computes a topical authority / ease-of-rank score (0-100)
    for given brand(s) and keyword(s) using semantic similarity,
    popularity signals, brand-model heuristics, and category matching.
    """
    def __init__(self, similarity_weight: float = 0.8, popularity_weight: float = 0.2):
        # Normalize weights to sum to 1
        total = similarity_weight + popularity_weight
        self.sim_w = similarity_weight / total
        self.pop_w = popularity_weight / total

        # Known product lines per brand for heuristic penalization
        self.product_lines = {
            'nike': ['air max', 'air force', 'dunk', 'jersey', 'swoosh'],
            'adidas': ['ultraboost', 'stan smith', 'nmd', 'adilette'],
            'puma': ['suede', 'rs', 'basket'],
            'solero': ['ice cream', 'gelato', 'sorbet'],
        }
        # Map brands to their primary category
        self.brand_categories = {
            'nike': 'trainers',
            'adidas': 'trainers',
            'puma': 'trainers',
            'solero': 'ice creams',
        }
        # Simple keyword category detection by keyword tokens
        self.keyword_categories = {
            'trainers': ['trainer', 'trainers', 'air max', 'sneaker', 'sneakers', 'dunk'],
            'ice creams': ['ice cream', 'gelato', 'sorbet']
        }

    def _get_embedding(self, text: str) -> np.ndarray:
        resp = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(resp.data[0].embedding)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _detect_keyword_category(self, keyword: str) -> str:
        """
        Return the detected category for a keyword, or None if unknown.
        """
        kl = keyword.lower()
        for cat, tokens in self.keyword_categories.items():
            if any(tok in kl for tok in tokens):
                return cat
        return None

    def _fetch_popularity_score(self, keyword: str, brand: str) -> float:
        # TODO: integrate with a real SEO/popularity API
        return 1.0

    def score(self, brand: str, keyword: str) -> dict:
        # Normalize brand key
        bk = brand.lower().strip()
        # Detect categories
        brand_cat = self.brand_categories.get(bk)
        kw_cat = self._detect_keyword_category(keyword)
        # If either category is unknown or they don't match, return zeros
        if not brand_cat or not kw_cat or brand_cat != kw_cat:
            return {'similarity': 0.0, 'popularity': 0.0, 'score': 0.0}

        # Compute embeddings and base semantic similarity
        emb_brand = self._get_embedding(brand)
        emb_kw = self._get_embedding(keyword)
        sim = self._cosine_similarity(emb_brand, emb_kw)

        # Heuristic: penalize if keyword references a model not in the brand's lineup
        models = self.product_lines.get(bk, [])
        if models and not any(m in keyword.lower() for m in models):
            sim *= 0.5

        # Popularity signal
        pop = self._fetch_popularity_score(keyword, brand)

        # Weighted combination
        combined = sim * self.sim_w + pop * self.pop_w
        final = min(combined, 1.0) * 100
        return {'similarity': sim, 'popularity': pop, 'score': final}

# ----- Streamlit UI ----- #
st.title("Brand vs. Keyword Topical Authority Scorer")

st.markdown("**Enter multiple brands and keywords (comma-separated):**")
brands_input = st.text_area("Brands", value="Nike, Adidas, Puma, Solero")
keywords_input = st.text_area("Keywords/Topics", value="new trainers, air max plus, ice creams")

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
st.markdown("Now outputs 0 for brand-keyword pairs from mismatched categories (e.g. trainers vs. ice creams).")
