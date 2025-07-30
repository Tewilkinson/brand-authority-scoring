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
    Robust relevance scorer combining:
      1) Embedding similarity (brand name + live Wikipedia summary) via configurable model
      2) Configurable popularity signal (stub for SEO/Gemini analytics)
      3) Optional LLM-based citation-driven relevance scoring via Gemini
    """
    def __init__(
        self,
        similarity_weight: float = 0.5,
        popularity_weight: float = 0.1,
        llm_weight: float = 0.4,
        use_llm: bool = True,
        embedding_model: str = "gpt-4o-embedding-gecko",
        llm_model: str = "gpt-4o",
        llm_temperature: float = 0.2
    ):
        total = similarity_weight + popularity_weight + (llm_weight if use_llm else 0)
        self.sim_w = similarity_weight / total
        self.pop_w = popularity_weight / total
        self.llm_w = (llm_weight / total) if use_llm else 0
        self.use_llm = use_llm
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

    def _fetch_wikipedia_summary(self, brand: str) -> str:
        if not WIKIPEDIA_AVAILABLE:
            return brand
        try:
            return wikipedia.summary(brand, sentences=2)
        except Exception:
            return brand

    def _get_embedding(self, text: str) -> np.ndarray:
        resp = client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(resp.data[0].embedding)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _popularity_score(self, brand: str, keyword: str) -> float:
        # Placeholder: integrate real SEO or Gemini analytics here
        return 1.0

    def _llm_relevance(self, brand: str, keyword: str) -> Optional[float]:
        if not self.use_llm:
            return None
        prompt = (
            f"You are an expert on brand–topic relevance. "
            f"On a scale from 0 to 100, how relevant is the brand '{brand}' "
            f"to the topic '{keyword}'?" 
            "Support your score with 2–3 short citations or real-world references. "
            "Respond in strict JSON: {\"score\": number, \"evidence\": [string,...]}."
        )
        response = client.chat.completions.create(
            model=self.llm_model,
            temperature=self.llm_temperature,
            messages=[
                {"role": "system", "content": "You are a data-driven evaluator."},
                {"role": "user", "content": prompt}
            ]
        )
        text = response.choices[0].message.content.strip()
        try:
            import json
            data = json.loads(text)
            return float(data.get('score', 0))
        except Exception:
            return None

    def score(self, brand: str, keyword: str) -> dict:
        # 1) Embedding-based semantic relevance
        summary = self._fetch_wikipedia_summary(brand)
        emb_brand = np.mean([
            self._get_embedding(brand),
            self._get_embedding(summary)
        ], axis=0)
        emb_kw = self._get_embedding(keyword)
        sim_score = self._cosine(emb_brand, emb_kw) * 100

        # 2) Popularity proxy
        pop_score = self._popularity_score(brand, keyword) * 100

        # 3) LLM-driven relevance (0-100)
        llm_score = self._llm_relevance(brand, keyword) if self.use_llm else None

        # Combine and clamp to [0,100]
        combined = sim_score * self.sim_w + pop_score * self.pop_w
        if llm_score is not None:
            combined += llm_score * self.llm_w
        final_score = max(0, min(combined, 100))

        return {
            'semantic': sim_score,
            'popularity': pop_score,
            'llm': llm_score,
            'combined': final_score
        }

# ----- Streamlit UI ----- #
st.title("Comprehensive Brand vs. Keyword Relevance Tool")

st.markdown("Use Gemini embeddings and LLM for robust, real-world relevance scoring.")
brands = [b.strip() for b in st.text_input("Brands (comma-separated)", "Nike, Adidas, Puma").split(',') if b.strip()]
keywords = [k.strip() for k in st.text_input("Keywords/Topics (comma-separated)", "new trainers, air max plus").split(',') if k.strip()]

sim_w = st.slider("Embedding similarity weight", 0.0, 1.0, 0.5)
pop_w = st.slider("Popularity weight", 0.0, 1.0, 0.1)
llm_w = st.slider("LLM relevance weight", 0.0, 1.0, 0.4)
use_llm = st.checkbox("Use LLM-based relevance scoring", True)
embed_model = st.selectbox(
    "Embedding model", 
    ["gpt-4o-embedding-gecko", "text-embedding-ada-002"],
    index=0
)
llm_model = st.selectbox(
    "LLM model", ["gpt-4o", "gemini-pro"], index=0
)
llm_temp = st.slider("LLM temperature", 0.0, 1.0, 0.2)

if st.button("Run Analysis") and brands and keywords:
    ranker = BrandKeywordRanker(
        similarity_weight=sim_w,
        popularity_weight=pop_w,
        llm_weight=llm_w,
        use_llm=use_llm,
        embedding_model=embed_model,
        llm_model=llm_model,
        llm_temperature=llm_temp
    )
    rows = []
    for brand in brands:
        for kw in keywords:
            res = ranker.score(brand, kw)
            rows.append({
                'Brand': brand,
                'Keyword': kw,
                'Semantic (0-100)': f"{res['semantic']:.1f}",
                'Popularity (0-100)': f"{res['popularity']:.1f}",
                'LLM (0-100)': f"{res['llm']:.1f}" if res['llm'] is not None else 'N/A',
                'Combined (0-100)': f"{res['combined']:.1f}"
            })
    df = pd.DataFrame(rows)
    st.dataframe(df)

st.markdown("---")
st.markdown(
    "**Notes:** \n"
    "- Embeddings: Gemini gecko vs. Ada (choose above).\n"
    "- LLM: Gemini-Pro or GPT-4o for citation-driven relevance.\n"
    "- Popularity: stub for your SEO/Gemini analytics integration.\n"
    "- Wikipedia summary used when available; falls back to brand name."
)
