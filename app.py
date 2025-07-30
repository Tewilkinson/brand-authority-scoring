import os
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from typing import Optional

# Optional dependencies
# pip install wikipedia pytrends
try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

try:
    from pytrends.request import TrendReq  # For Google Trends
    TRENDS_AVAILABLE = True
except ImportError:
    TRENDS_AVAILABLE = False

# Configuration
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)

class BrandKeywordRanker:
    """
    Computes robust relevance by:
      1) Embedding similarity between keyword and full brand wiki content
      2) Boost if keyword string appears in the wiki text
      3) Optional Google Trends popularity
      4) LLM scoring from GPT-4 and GPT-3.5
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

    def _get_full_wiki(self, brand: str) -> str:
        """
        Fetches full Wikipedia page content for richer context.
        """
        if not WIKIPEDIA_AVAILABLE:
            return brand
        try:
            page = wikipedia.page(brand, auto_suggest=False)
            return page.content
        except Exception:
            return brand

    def _embed(self, text: str) -> np.ndarray:
        """
        Embeds text with Ada-002, splitting into chunks if too long.
        """
        # Ada embedding max ~8191 tokens; we chunk by characters as proxy
        max_chunk = 3000
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        embeds = []
        for chunk in chunks:
            resp = client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            embeds.append(np.array(resp.data[0].embedding))
        # average chunk embeddings
        return np.mean(np.vstack(embeds), axis=0)

.data[0].embedding)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def _popularity(self, brand: str, keyword: str) -> float:
        """
        Uses Google Trends (PyTrends) to fetch normalized interest (0-1) for 'brand keyword'.
        Requires PyTrends to be installed.
        """
        if not TRENDS_AVAILABLE or not self.use_popularity:
            return 0.0
        try:
            pytrends = TrendReq(hl='en-US', tz=0)
            term = f"{brand} {keyword}"
            pytrends.build_payload([term], timeframe='today 12-m')
            df = pytrends.interest_over_time()
            if df.empty:
                return 0.0
            return float(df[term].iloc[-1]) / 100.0
        except Exception:
            return 0.0
            return float(df[term].iloc[-1]) / 100.0
        except Exception:
            return 0.0

    def _llm_score(self, model: str, brand: str, keyword: str) -> float:
        """
        Queries LLM to rate 0-100 relevance of brand to keyword.
        """
        prompt = f"Rate relevance (0-100) of brand '{brand}' to topic '{keyword}'. Reply with just the integer." 
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role":"system","content":"You are a precise scoring assistant."},
                    {"role":"user","content":prompt}
                ]
            )
            return float(resp.choices[0].message.content.strip())
        except Exception:
            return 0.0

    def score(self, brand: str, keyword: str) -> dict:
        # 1) Full wiki context embedding
        wiki_text = self._get_full_wiki(brand)
        emb_brand = self._embed(wiki_text)
        emb_kw = self._embed(keyword)
        sim_score = self._cosine(emb_brand, emb_kw) * 100
        # Boost if exact keyword string appears in wiki
        if keyword.lower() in wiki_text.lower():
            sim_score = max(sim_score, 50)  # ensure at least a mid-level score

        # 2) Popularity
        pop_score = self._popularity(brand, keyword) * 100 if self.use_popularity else 0

        # 3) LLMs
        g4 = self._llm_score("gpt-4", brand, keyword)
        g35 = self._llm_score("gpt-3.5-turbo", brand, keyword)
        llm_score = (g4 + g35) / 2

        # 4) Combine
        combined = sim_score * self.embed_w + pop_score * self.pop_w + llm_score * self.llm_w
        combined = max(0, min(combined, 100))

        return {
            'semantic': sim_score,
            'popularity': pop_score,
            'gpt4': g4,
            'gpt35': g35,
            'combined': combined
        }

# Streamlit UI
st.title("Brand vs. Topic Authority Scorer")

st.markdown("**Note:** Relevance uses full Wikipedia content + embeddings; popularity via free Google Trends.")
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
                res = rk.score(b, k)
                rows.append({
                    'Brand': b,
                    'Keyword': k,
                    'Semantic (0-100)': f"{res['semantic']:.1f}",
                    'Popularity (0-100)': f"{res['popularity']:.1f}",
                    'GPT-4': f"{res['gpt4']:.1f}",
                    'GPT-3.5': f"{res['gpt35']:.1f}",
                    'Combined (0-100)': f"{res['combined']:.1f}"  
                })
        st.dataframe(pd.DataFrame(rows))
