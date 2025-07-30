import os
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from typing import Optional

# Optional installations:
# pip install wikipedia
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
    Computes a unified relevance score by combining:
      1) Embedding similarity from both Gemini and ChatGPT embeddings
      2) Optional popularity signal
      3) LLM-driven relevance from both Gemini-Pro and GPT-4o
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

    def _embed(self, model: str, text: str) -> np.ndarray:
        """
        Try embedding with specified model; fallback to ADA if unavailable.
        """
        for m in [model, "text-embedding-ada-002"]:
            try:
                resp = client.embeddings.create(model=m, input=text)
                return np.array(resp.data[0].embedding)
            except Exception as e:
                st.warning(f"Embedding model '{m}' failed: {e}")
        st.error("All embedding models failed.")
        return np.zeros((1536,))  # default dimension

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def _popularity(self, brand: str, keyword: str) -> float:
        # placeholder for real SEO/Gemini analytics
        return 1.0

    def _llm_score(self, model: str, brand: str, keyword: str) -> float:
        prompt = (
            f"Rate relevance 0–100 for brand '{brand}' to topic '{keyword}'. "
            "Answer with just the number."
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role":"system","content":"You are a helpful assistant."},
                    {"role":"user","content":prompt}
                ]
            )
            return float(resp.choices[0].message.content.strip())
        except Exception as e:
            st.warning(f"LLM model '{model}' failed: {e}")
            return 0.0

    def score(self, brand: str, keyword: str) -> dict:
        wiki_text = self._get_wiki(brand)
        # Embeddings: Gemini + ADA averaged
        emb_brand_g = np.mean([
            self._embed("gpt-4o-embedding-gecko", brand),
            self._embed("gpt-4o-embedding-gecko", wiki_text)
        ], axis=0)
        emb_brand_a = np.mean([
            self._embed("text-embedding-ada-002", brand),
            self._embed("text-embedding-ada-002", wiki_text)
        ], axis=0)
        emb_kw_g = self._embed("gpt-4o-embedding-gecko", keyword)
        emb_kw_a = self._embed("text-embedding-ada-002", keyword)
        sim_g = self._cosine(emb_brand_g, emb_kw_g)
        sim_a = self._cosine(emb_brand_a, emb_kw_a)
        sim_score = (sim_g + sim_a) / 2 * 100

        pop_score = self._popularity(brand, keyword) * 100 if self.use_popularity else 0

        llm_g = self._llm_score("gemini-pro", brand, keyword)
        llm_c = self._llm_score("gpt-4o", brand, keyword)
        llm_score = (llm_g + llm_c) / 2

        combined = sim_score * self.embed_w + pop_score * self.pop_w + llm_score * self.llm_w
        final = max(0, min(combined, 100))

        return {
            'embed_similarity': sim_score,
            'popularity': pop_score,
            'llm_gemini': llm_g,
            'llm_gpt4o': llm_c,
            'combined': final
        }

# Streamlit UI
st.title("Unified Brand–Topic Authority Scorer")

brands_input = st.text_input("Brands (comma-separated)", "Nike, Adidas, Puma", key="brands")
keywords_input = st.text_input("Keywords (comma-separated)", "new trainers, air max plus, ice creams, call of duty", key="keywords")
use_pop = st.checkbox("Include popularity signal", False)
embed_w = st.slider("Embedding weight", 0.0, 1.0, 0.4)
pop_w = st.slider("Popularity weight", 0.0, 1.0, 0.1)
llm_w = st.slider("LLM weight", 0.0, 1.0, 0.5)

if st.button("Score"):
    brands = [b.strip() for b in brands_input.split(',') if b.strip()]
    kws = [k.strip() for k in keywords_input.split(',') if k.strip()]
    if not brands or not kws:
        st.warning("Please enter at least one brand and one keyword.")
    else:
        rk = BrandKeywordRanker(embed_w, pop_w, llm_w, use_pop)
        rows = []
        for b in brands:
            for k in kws:
                r = rk.score(b, k)
                rows.append({
                    'Brand': b,
                    'Keyword': k,
                    'Sim(0-100)': f"{r['embed_similarity']:.1f}",
                    'Pop(0-100)': f"{r['popularity']:.1f}",
                    'GeminiLLM': f"{r['llm_gemini']:.1f}",
                    'GPT4oLLM': f"{r['llm_gpt4o']:.1f}",
                    'Final': f"{r['combined']:.1f}" })
        df = pd.DataFrame(rows)
        st.dataframe(df)
