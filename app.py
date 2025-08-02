import os
import pandas as pd
import streamlit as st
import plotly.express as px
from openai import OpenAI

# Load secrets
secrets = st.secrets
OPENAI_API_KEY  = secrets.get('OPENAI_API_KEY')
GEMINI_API_KEY  = secrets.get('GEMINI_API_KEY')
GEMINI_API_BASE = secrets.get('GEMINI_API_BASE')

if not OPENAI_API_KEY:
    st.error("Missing OpenAI API key in secrets.")
    st.stop()

# Initialize LLM clients
client = OpenAI(api_key=OPENAI_API_KEY)
if GEMINI_API_KEY and GEMINI_API_BASE:
    gemini_client = OpenAI(api_key=GEMINI_API_KEY, api_base=GEMINI_API_BASE)
else:
    gemini_client = client

class RelevanceScorer:
    """
    Blends Gemini-Pro and GPT-4 relevance scores (0-100).
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
            if model == "gemini-pro":
                return self._llm_score('gpt-4', brand, keyword)
            return 0.0

    def score(self, brand: str, keyword: str) -> dict:
        gem_score = self._llm_score('gemini-pro', brand, keyword)
        gpt_score = self._llm_score('gpt-4',      brand, keyword)
        combined  = gem_score * self.gemini_w + gpt_score * self.gpt4_w
        return {
            'Brand':      brand,
            'Keyword':    keyword,
            'Gemini-Pro': round(gem_score, 1),
            'GPT-4':      round(gpt_score, 1),
            'Combined':   round(combined, 1)
        }

# Streamlit App UI
st.title("Brand vs. Topic Relevance: GPT-4 + Gemini-Pro")

# Weight sliders in two columns
col1, col2 = st.columns(2)
with col1:
    gem_w = st.slider("Gemini-Pro weight", 0.0, 1.0, 0.5)
with col2:
    gpt_w = st.slider("GPT-4 weight",       0.0, 1.0, 0.5)

# Inputs
brands_input = st.text_input(
    "Brands (comma-separated)",
    "Nike, Adidas, Puma",
    help="Enter brands separated by commas"
)
keywords_input = st.text_area(
    "Keywords (one per line)",
    """new trainers
ice cream
photography""",
    height=150,
    help="Enter each keyword or topic on its own line"
)

if st.button("Compute Relevance Scores"):
    brands   = [b.strip() for b in brands_input.split(',') if b.strip()]
    keywords = [k.strip() for k in keywords_input.splitlines() if k.strip()]
    if not brands or not keywords:
        st.warning("Please enter at least one brand and one keyword.")
    else:
        scorer = RelevanceScorer(gemini_w=gem_w, gpt4_w=gpt_w)
        rows = []
        for brand in brands:
            for keyword in keywords:
                rows.append(scorer.score(brand, keyword))
        df = pd.DataFrame(rows)

        # Combined Scores Chart
        fig = px.bar(
            df,
            x='Keyword', y='Combined', color='Brand',
            barmode='group',
            title='Combined Relevance Scores by Keyword and Brand'
        )
        fig.update_layout(yaxis_title='Score (0-100)', xaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

        # Detailed Table
        st.subheader("Detailed LLM Scores and Combined Result")
        st.table(df.set_index(['Brand', 'Keyword']))
