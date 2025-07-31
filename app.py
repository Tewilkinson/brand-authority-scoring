import os
import pandas as pd
import streamlit as st
import plotly.express as px
from openai import OpenAI

# Optional: pip install pytrends
try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except ImportError:
    TRENDS_AVAILABLE = False

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE")

if not OPENAI_API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize OpenAI client for GPT
client = OpenAI(api_key=OPENAI_API_KEY)
# Initialize Gemini client if provided
gemini_client = (
    OpenAI(api_key=GEMINI_API_KEY, api_base=GEMINI_API_BASE)
    if GEMINI_API_KEY and GEMINI_API_BASE else client
)

class RelevanceScorer:
    """
    Blends Gemini-Pro, GPT-4, and Google Trends signals for relevance (0-100).
    """
    def __init__(self, gemini_w: float, gpt4_w: float, pop_w: float):
        total = gemini_w + gpt4_w + pop_w
        self.gemini_w = gemini_w / total
        self.gpt4_w   = gpt4_w   / total
        self.pop_w    = pop_w    / total

    def _llm_score(self, model: str, brand: str, keyword: str) -> float:
        prompt = (
            f"On a scale of 0 to 100, how relevant is the brand '{brand}'"
            f" to the topic '{keyword}'? Reply with only the integer score."
        )
        client_ = gemini_client if model == "gemini-pro" else client
        try:
            resp = client_.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role":"system","content":"You are an expert in brand-topic relevance scoring."},
                    {"role":"user","content":prompt}
                ]
            )
            return float(resp.choices[0].message.content.strip())
        except Exception:
            # fallback to GPT-4 for Gemini failures
            if model == "gemini-pro":
                try:
                    fb = client.chat.completions.create(
                        model="gpt-4", temperature=0.1,
                        messages=[
                            {"role":"system","content":"You are an expert in brand-topic relevance scoring."},
                            {"role":"user","content":prompt}
                        ]
                    )
                    return float(fb.choices[0].message.content.strip())
                except Exception:
                    return 0.0
            return 0.0

    def _popularity(self, brand: str, keyword: str) -> float:
        """
        Uses Google Trends (last 30 days) to return normalized 0-100 interest.
        """
        if not TRENDS_AVAILABLE:
            return 0.0
        try:
            tr = TrendReq(hl='en-US', tz=0)
            term = f"{brand} {keyword}"
            tr.build_payload([term], timeframe='today 30-d')
            df = tr.interest_over_time()
            if df.empty:
                return 0.0
            return float(df[term].iloc[-1])
        except Exception:
            return 0.0

    def score(self, brand: str, keyword: str) -> dict:
        g_score = self._llm_score('gemini-pro', brand, keyword)
        c_score = self._llm_score('gpt-4',    brand, keyword)
        p_score = self._popularity(brand, keyword)
        combined = (g_score * self.gemini_w + c_score * self.gpt4_w + p_score * self.pop_w)
        return {
            'Brand': brand,
            'Keyword': keyword,
            'Gemini-Pro': round(g_score,1),
            'GPT-4': round(c_score,1),
            'Trends (0-100)': round(p_score,1),
            'Combined': round(combined,1)
        }

# Streamlit App
st.title("Brand vs. Topic Authority: LLM + Trends")

brands_in   = st.text_input("Brands (comma-separated)",  "Nike, Adidas, Puma")
kws_in      = st.text_input("Keywords (comma-separated)", "new trainers, ice cream")

# Weights / Options
enable_trends = st.checkbox(
    "Include Google Trends data", value=True,
    help="Toggle to include Trends as popularity signal"
)
gem_w = st.slider(
    "Gemini-Pro weight", 0.0, 1.0, 0.4,
    help="Relative weight for Gemini‑Pro score"
)
gpt_w = st.slider(
    "GPT-4 weight", 0.0, 1.0, 0.4,
    help="Relative weight for GPT‑4 score"
)
pop_w = st.slider(
    "Trends weight", 0.0, 1.0, 0.2,
    help="Relative weight for Trends score"
)

if st.button("Compute Scores"):
    brands = [b.strip() for b in brands_in.split(',') if b.strip()]
    kws    = [k.strip() for k in kws_in.split(',') if k.strip()]
    scorer = RelevanceScorer(
        gem_w,
        gpt_w,
        pop_w if enable_trends else 0.0
    )
    rows = []
    for b in brands:
        for k in kws:
        for k in kws:
            rows.append(scorer.score(b, k))
    df = pd.DataFrame(rows)
    # Grouped bar chart
    fig = px.bar(df, x='Keyword', y='Combined', color='Brand', barmode='group',
                 labels={'Combined':'Score (0-100)'}, title='Combined Relevance Scores')
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed scores table
    st.subheader("Detailed Scores by Model and Combined")
    st.table(df.set_index(['Brand','Keyword']))
