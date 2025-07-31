import os
import pandas as pd
import streamlit as st
import plotly.express as px
from openai import OpenAI

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE")

if not OPENAI_API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
if GEMINI_API_KEY and GEMINI_API_BASE:
    gemini_client = OpenAI(api_key=GEMINI_API_KEY, api_base=GEMINI_API_BASE)
else:
    gemini_client = client

class RelevanceScorer:
    """
    Blends Gemini-Pro and GPT-4 for relevance (0-100).
    """
    def __init__(self, gemini_w: float, gpt4_w: float):
        total = gemini_w + gpt4_w
        self.gemini_w = gemini_w / total if total else 0
        self.gpt4_w   = gpt4_w   / total if total else 0

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
            # Fallback to GPT-4 if Gemini-Pro fails
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

    def score(self, brand: str, keyword: str) -> dict:
        gemini_score = self._llm_score('gemini-pro', brand, keyword)
        gpt4_score  = self._llm_score('gpt-4',      brand, keyword)
        combined    = gemini_score * self.gemini_w + gpt4_score * self.gpt4_w
        return {
            'Brand':      brand,
            'Keyword':    keyword,
            'Gemini-Pro': round(gemini_score, 1),
            'GPT-4':      round(gpt4_score, 1),
            'Combined':   round(combined, 1)
        }

# Streamlit App
st.title("Brand vs. Topic Relevance: GPT-4 + Gemini-Pro")

# Input: brands comma-separated, keywords space-separated
brands_in = st.text_input("Brands (comma-separated)", "Nike, Adidas, Puma")
keywords_in = st.text_input("Keywords (space-separated)", "new trainers ice cream photography")

gem_w = st.slider("Gemini-Pro weight", 0.0, 1.0, 0.5)
gpt_w = st.slider("GPT-4 weight",       0.0, 1.0, 0.5)

if st.button("Compute Relevance Scores"):
    # Parse inputs
    brands  = [b.strip() for b in brands_in.split(',') if b.strip()]
    keywords = [k.strip() for k in keywords_in.split() if k.strip()]

    scorer = RelevanceScorer(gemini_w=gem_w, gpt4_w=gpt_w)
    rows = []
    for brand in brands:
        for keyword in keywords:
            rows.append(scorer.score(brand, keyword))
    df = pd.DataFrame(rows)

    # Grouped Bar Chart
    fig = px.bar(
        df,
        x='Keyword', y='Combined', color='Brand',
        barmode='group', title='Combined Relevance Scores by Keyword and Brand'
    )
    fig.update_layout(yaxis_title='Score (0-100)', xaxis_title='')
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Table
    st.subheader("Detailed Scores by Model and Combined")
    st.table(df.set_index(['Brand','Keyword']))
