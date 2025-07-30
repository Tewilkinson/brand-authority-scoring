import os
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from typing import Optional

# Optional dependencies:
# pip install wikipedia pytrends
try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except ImportError:
    TRENDS_AVAILABLE = False

# Configuration
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()
client = OpenAI(api_key=api_key)

class BrandKeywordRanker:
    """
    Computes relevance by combining:
      1) Semantic similarity (embedding of Wikipedia summary vs keyword)
      2) Boost if keyword occurs in summary text
      3) Optional Google Trends–based popularity
      4) LLM relevance from GPT-4 & GPT-3.5-turbo
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
        self.pop_w = pop_weight / total if use_popularity else 0.0
        self.llm_w = llm_weight / total
        self.use_popularity = use_popularity

    def _get_wiki_summary(self, brand: str) -> str:
        """
        Fetches a concise Wikipedia summary (2 sentences) or falls back to brand name.
        """
        if not WIKIPEDIA_AVAILABLE:
            return brand
        try:
            return wikipedia.summary(brand, sentences=2, auto_suggest=False)
        except Exception:
            return brand

    def _embed(self, text: str) -> np.ndarray:
        """
        Embeds text with Ada-002, splitting into chunks if too long.
        """
        max_len = 3000
        chunks = [text[i : i + max_len] for i in range(0, len(text), max_len)]
        embeds = []
        for chunk in chunks:
            resp = client.embeddings.create(model="text-embedding-ada-002", input=chunk)
            embeds.append(np.array(resp.data[0].embedding))
        return np.mean(np.vstack(embeds), axis=0)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def _popularity(self, brand: str, keyword: str) -> float:
        """
        Returns normalized Google Trends interest (0-1) for 'brand keyword'.
        """
        if not TRENDS_AVAILABLE or not self.use_popularity:
            return 0.0
        try:
            pytrends = TrendReq(hl='en-US', tz=0)
            term = f"{brand} {keyword}"
            pytrends.build_payload([term], timeframe='today 90-d')
            df = pytrends.interest_over_time()
            if df.empty:
                return 0.0
            return float(df[term].iloc[-1]) / 100.0
        except Exception:
            return 0.0

    def _llm_score(self, model: str, brand: str, keyword: str) -> float:
        """
        Queries LLM to rate relevance 0–100.
        """
        prompt = f"Rate relevance (0-100) of brand '{brand}' to topic '{keyword}'. Reply with the integer."
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a precise scoring assistant."},
                    {"role": "user", "content": prompt}
                ],
            )
            return float(resp.choices[0].message.content.strip())
        except Exception:
            return 0.0

    def score(self, brand: str, keyword: str) -> dict:
        # 1) Semantic similarity
        summary = self._get_wiki_summary(brand)
        emb_brand = self._embed(summary)
        emb_kw = self._embed(keyword)
        sim_score = self._cosine(emb_brand, emb_kw) * 100
        # 2) Boost if keyword explicitly in summary
        if keyword.lower() in summary.lower():
            sim_score = max(sim_score, 50)
        # 3) Popularity
        pop_score = self._popularity(brand, keyword) * 100
        # 4) LLM relevance
        g4 = self._llm_score("gpt-4", brand, keyword)
        g35 = self._llm_score("gpt-3.5-turbo", brand, keyword)
        llm_score = (g4 + g35) / 2
        # 5) Combined score
        combined = sim_score * self.embed_w + pop_score * self.pop_w + llm_score * self.llm_w
        combined = max(0, min(combined, 100))
        return {
            'semantic': sim_score,
            'popularity': pop_score,
            'gpt4': g4,
            'gpt35': g35,
            'combined': combined,
        }

# Streamlit UI
title = "Brand vs. Topic Authority Scorer"
st.title(title)
st.markdown("**Note:** Uses Wikipedia summary + optional Trends + GPT relevance.")

brands_input = st.text_input("Brands (comma-separated)", "Nike, Adidas, Puma")
keywords_input = st.text_input("Keywords (comma-separated)", "new trainers, ice cream, photography")
use_pop = st.checkbox("Include Google Trends popularity", False)
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
                r = rk.score(b, k)
                rows.append({
                    'Brand': b,
                    'Keyword': k,
                    'Semantic (0-100)': f"{r['semantic']:.1f}",
                    'Popularity (0-100)': f"{r['popularity']:.1f}",
                    'GPT-4': f"{r['gpt4']:.1f}",
                    'GPT-3.5': f"{r['gpt35']:.1f}",
                    'Combined (0-100)': f"{r['combined']:.1f}"  
                })
        st.dataframe(pd.DataFrame(rows))
