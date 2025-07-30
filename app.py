import os
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from typing import Optional

# Optional: pip install wikipedia
try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

# Configuration
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

class BrandKeywordRanker:
    """
    Computes a combined relevance score by blending:
      • Embedding similarity (via text-embedding-ada-002)
      • Optional popularity signal
      • LLM-driven relevance from GPT-4 and GPT-3.5-turbo
    """
    def __init__(
        self,
        embed_weight: float = 0.4,
        pop_weight: float = 0.1,
        llm_weight: float = 0.5,
        use_popularity: bool = False
    ):
        total = embed_weight + (pop_weight if use_popularity else 0) + llm_weight
        self.embed_w = embed_weight / total
        self.pop_w = (pop_weight / total) if use_popularity else 0
        self.llm_w = llm_weight / total
        self.use_popularity = use_popularity

    def _get_wiki(self, brand: str) -> str:
        if not WIKIPEDIA_AVAILABLE:
            return brand
        try:
            return wikipedia.summary(brand, sentences=2)
        except Exception:
            return brand

    def _embed(self, text: str) -> np.ndarray:
        """
        Compute embedding with Ada-002.
        """
        resp = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(resp.data[0].embedding)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def _popularity(self, brand: str, keyword: str) -> float:
        # Placeholder for real SEO/Gemini analytics
        return 1.0

    def _llm_score(self, model: str, brand: str, keyword: str) -> float:
        """
        Query LLM for numeric relevance 0–100. Fallback sequence on error.
        """
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a relevance scoring assistant."},
                    {"role": "user", "content": f"Rate relevance (0-100) of brand '{brand}' to topic '{keyword}'. Reply with just the integer."}
                ]
            )
            return float(resp.choices[0].message.content.strip())
        except Exception:
            return 0.0

    def score(self, brand: str, keyword: str) -> dict:
        # Build brand context
        profile = self._get_wiki(brand)
        # Embedding similarity
        emb_brand = np.mean([self._embed(brand), self._embed(profile)], axis=0)
        emb_kw = self._embed(keyword)
        sim_score = self._cosine(emb_brand, emb_kw) * 100
        # Popularity
        pop_score = self._popularity(brand, keyword) * 100 if self.use_popularity else 0
        # LLM relevance
        g4 = self._llm_score("gpt-4", brand, keyword)
        g35 = self._llm_score("gpt-3.5-turbo", brand, keyword)
        llm_score = (g4 + g35) / 2
        # Combine
        combined = sim_score * self.embed_w + pop_score * self.pop_w + llm_score * self.llm_w
        combined = max(0, min(combined, 100))
        return {
            'similarity': sim_score,
            'popularity': pop_score,
            'gpt4': g4,
            'gpt3.5': g35,
            'combined': combined
        }

# Streamlit UI
st.title("Brand vs. Topic Authority Scorer")

brands_input = st.text_input("Brands (comma-separated)", "Nike, Adidas, Puma")
keywords_input = st.text_input("Keywords (comma-separated)", "new trainers, air max plus")
use_pop = st.checkbox("Include popularity", False)
embed_w = st.slider("Embedding weight", 0.0, 1.0, 0.4)
pop_w = st.slider("Popularity weight", 0.0, 1.0, 0.1)
llm_w = st.slider("LLM weight", 0.0, 1.0, 0.5)

if st.button("Compute Scores"):
    brands = [b.strip() for b in brands_input.split(',') if b.strip()]
    kws = [k.strip() for k in keywords_input.split(',') if k.strip()]
    if not brands or not kws:
        st.warning("Enter at least one brand and one keyword.")
    else:
        rk = BrandKeywordRanker(embed_weight=embed_w, pop_weight=pop_w, llm_weight=llm_w, use_popularity=use_pop)
        rows = []
        for b in brands:
            for k in kws:
                res = rk.score(b, k)
                rows.append({
                    'Brand': b,
                    'Keyword': k,
                    'Similarity': f"{res['similarity']:.1f}",
                    'Popularity': f"{res['popularity']:.1f}",
                    'GPT-4': f"{res['gpt4']:.1f}",
                    'GPT-3.5': f"{res['gpt3.5']:.1f}",
                    'Combined': f"{res['combined']:.1f}"  
                })
        st.dataframe(pd.DataFrame(rows))
