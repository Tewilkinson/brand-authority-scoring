import os
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from openai import OpenAI

# Load API keys and credentials
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
GEMINI_API_BASE      = os.getenv("GEMINI_API_BASE")
DATAFORESEO_USERNAME = os.getenv("DATAFORESEO_USERNAME")
DATAFORESEO_PASSWORD = os.getenv("DATAFORESEO_PASSWORD")

if not OPENAI_API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize OpenAI client for GPT
client = OpenAI(api_key=OPENAI_API_KEY)
# Initialize Gemini client if provided
if GEMINI_API_KEY and GEMINI_API_BASE:
    gemini_client = OpenAI(api_key=GEMINI_API_KEY, api_base=GEMINI_API_BASE)
else:
    gemini_client = client

class RelevanceScorer:
    """
    Blends Gemini-Pro, GPT-4, and DataForSEO search volume share for relevance (0-100).
    """
    def __init__(self, gemini_w: float, gpt4_w: float, vol_w: float):
        total = gemini_w + gpt4_w + vol_w
        self.gemini_w = gemini_w / total if total else 0
        self.gpt4_w   = gpt4_w   / total if total else 0
        self.vol_w    = vol_w    / total if total else 0

    def _llm_score(self, model: str, brand: str, keyword: str) -> float:
        prompt = (
            f"On a scale of 0 to 100, how relevant is the brand '{brand}' "
            f"to the topic '{keyword}'? Reply with only the integer score."
        )
        client_ = gemini_client if model == "gemini-pro" else client
        try:
            resp = client_.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are an expert in brand-topic relevance scoring."},
                    {"role": "user",   "content": prompt}
                ]
            )
            return float(resp.choices[0].message.content.strip())
        except Exception:
            if model == "gemini-pro":
                try:
                    fb = client.chat.completions.create(
                        model="gpt-4",
                        temperature=0.1,
                        messages=[
                            {"role": "system", "content": "You are an expert in brand-topic relevance scoring."},
                            {"role": "user",   "content": prompt}
                        ]
                    )
                    return float(fb.choices[0].message.content.strip())
                except Exception:
                    return 0.0
            return 0.0

    def _get_volume(self, term: str) -> float:
        """
        Fetches raw search volume for a term from DataForSEO.
        """
        if not (DATAFORESEO_USERNAME and DATAFORESEO_PASSWORD):
            return 0.0
        url = "https://api.dataforseo.com/v3/keywords_data/google/search_volume/live"
        payload = [{"keyword": term, "language_name": "English", "location_code": 2840}]
        try:
            resp = requests.post(url, auth=(DATAFORESEO_USERNAME, DATAFORESEO_PASSWORD), json=payload)
            data = resp.json()
            tasks = data.get("tasks", [])
            if not tasks or not tasks[0].get("result"):
                return 0.0
            vol = tasks[0]["result"][0].get("search_volume", 0)
            return float(vol)
        except Exception:
            return 0.0

    def score(self, brand: str, keyword: str) -> dict:
        # LLM signals
        gem_score = self._llm_score('gemini-pro', brand, keyword)
        gpt_score = self._llm_score('gpt-4',      brand, keyword)
        # Volume signals: pure keyword vs brand+keyword
        kw_vol  = self._get_volume(keyword)
        bk_vol  = self._get_volume(f"{brand} {keyword}")
        # Compute share (brand-specific fraction of keyword volume)
        vol_share = (bk_vol / kw_vol * 100) if kw_vol > 0 else 0.0
        # Combined weighted score
        combined = (gem_score * self.gemini_w + gpt_score * self.gpt4_w + vol_share * self.vol_w)
        return {
            'Brand':              brand,
            'Keyword':            keyword,
            'Gemini-Pro':         round(gem_score, 1),
            'GPT-4':              round(gpt_score, 1),
            'Keyword Volume':     round(kw_vol, 1),
            'Brand+Keyword Vol':  round(bk_vol, 1),
            'Volume Share (%)':   round(vol_share, 1),
            'Combined Score':     round(combined, 1)
        }

# Streamlit App
st.title("Brand vs. Topic Relevance: LLM + Search Volume Share")

# Weight sliders
gem_w = st.slider("Gemini-Pro weight",       0.0, 1.0, 0.3)
gpt_w = st.slider("GPT-4 weight",             0.0, 1.0, 0.3)
vol_w = st.slider("Volume Share weight",      0.0, 1.0, 0.4)

# Inputs
brands_input = st.text_input("Brands (comma-separated)", "Nike, Adidas, Puma")
keywords_input = st.text_area(
    "Keywords (one per line)",
    """new trainers
ice cream
photography""",
    height=120
)

if st.button("Compute Relevance Scores"):
    brands   = [b.strip() for b in brands_input.split(',') if b.strip()]
    keywords = [k.strip() for k in keywords_input.splitlines() if k.strip()]
    scorer   = RelevanceScorer(gemini_w=gem_w, gpt4_w=gpt_w, vol_w=vol_w)
    rows     = []
    for brand in brands:
        for keyword in keywords:
            rows.append(scorer.score(brand, keyword))
    df = pd.DataFrame(rows)

    # Chart
    fig = px.bar(
        df,
        x='Keyword', y='Combined Score', color='Brand',
        barmode='group',
        title='Combined Relevance Scores by Keyword and Brand'
    )
    fig.update_layout(yaxis_title='Score (0-100)', xaxis_title='')
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Table
    st.subheader("Detailed Scores and Volume Data")
    st.table(df.set_index(['Brand', 'Keyword']))
