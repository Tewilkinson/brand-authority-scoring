import os
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from typing import Optional

# Optional installations:
# pip install wikipedia
# For more robust search, consider using Google Knowledge Graph or SEO APIs
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
      1) Embedding similarity (brand name + live Wikipedia summary)
      2) Configurable popularity signal (stub)
      3) Optional LLM-based citation-driven relevance scoring
    """
    def __init__(self,
                 similarity_weight: float = 0.5,
                 popularity_weight: float = 0.1,
                 llm_weight: float = 0.4,
                 use_llm: bool = True,
                 llm_model: str = "gpt-4o",
                 llm_temperature: float = 0.3):
        total = similarity_weight + popularity_weight + (llm_weight if use_llm else 0)
        self.sim_w = similarity_weight / total
        self.pop_w = popularity_weight / total
        self.llm_w = (llm_weight / total) if use_llm else 0
        self.use_llm = use_llm
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
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(resp.data[0].embedding)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _popularity_score(self, brand: str, keyword: str) -> float:
        # Placeholder: integrate real SEO API here
        return 1.0

    def _llm_relevance(self, brand: str, keyword: str) -> Optional[float]:
        """
        Uses an LLM to score relevance 0-100 with citations.
        Expects model to output JSON: {"score": int, "evidence": "..."}
        """
        try:
            prompt = (
                f"You are an expert analyzing brand relevance. "
                f"On a scale of 0-100, how relevant is the brand '{brand}' "
                f"to the topic '{keyword}'? "
                "Provide a JSON response with keys 'score' (integer) and 'evidence' (a short citation list)."
            )
            response = client.chat.completions.create(
                model=self.llm_model,
                temperature=self.llm_temperature,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content.strip()
            # Parse naive JSON
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
        llm_score = None
        if self.use_llm:
            llm_score = self._llm_relevance(brand, keyword)

        # Combine
        total = sim_score * self.sim_w + pop_score * self.pop_w
        if llm_score is not None:
            total += llm_score * self.llm_w
        # Normalize to 0-100
        total = max(0, min(total, 100))

        return {
            'semantic': sim_score,
            'popularity': pop_score,
            'llm': llm_score,
            'combined_score': total
        }

# ----- Streamlit UI ----- #
st.title("Comprehensive Brand vs. Keyword Relevance Tool")

st.markdown(
    "Enter comma-separated brands & keywords. "
    "Enable LLM scoring for citation-driven relevance (may cost extra)."
)
brands = [b.strip() for b in st.text_input("Brands", "Nike, Adidas, Puma").split(',') if b.strip()]
keywords = [k.strip() for k in st.text_input("Keywords/Topics", "new trainers, air max plus").split(',') if k.strip()]

sim_w = st.slider("Embedding similarity weight", 0.0, 1.0, 0.5)
pop_w = st.slider("Popularity weight", 0.0, 1.0, 0.1)
llm_w = st.slider("LLM relevance weight", 0.0, 1.0, 0.4)
use_llm = st.checkbox("Use LLM-based relevance scoring", True)

if st.button("Run Analysis") and brands and keywords:
    ranker = BrandKeywordRanker(similarity_weight=sim_w,
                                 popularity_weight=pop_w,
                                 llm_weight=llm_w,
                                 use_llm=use_llm)
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
                'Final Score': f"{res['combined_score']:.1f}"
            })
    df = pd.DataFrame(rows)
    st.dataframe(df)

st.markdown("---")
st.markdown(
    "**Notes:** \n"
    "- Embedding uses brand name + Wikipedia summary.\n"
    "- Popularity is a stub; integrate your SEO/Gemini data pipeline.\n"
    "- LLM step can cite real-world evidence (enable for robust scoring)."
)
