import os
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from openai import OpenAI

# Load secrets
secrets = st.secrets
OPENAI_API_KEY       = secrets.get('OPENAI_API_KEY')
GEMINI_API_KEY       = secrets.get('GEMINI_API_KEY')
GEMINI_API_BASE      = secrets.get('GEMINI_API_BASE')
DATAFORESEO_USER     = secrets.get('DATAFORESEO_USERNAME')
DATAFORESEO_PASS     = secrets.get('DATAFORESEO_PASSWORD')

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
    def __init__(self, gem_w: float, gpt_w: float, vol_w: float):
        total = gem_w + gpt_w + vol_w
        self.gem_w = gem_w / total if total else 0
        self.gpt_w = gpt_w / total if total else 0
        self.vol_w = vol_w / total if total else 0

    def _llm_score(self, model, brand, keyword):
        prompt = f"On a scale of 0-100, how relevant is '{brand}' to '{keyword}'?"
        client_ = gemini_client if model == 'gemini-pro' else client
        try:
            resp = client_.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {'role':'system','content':'You are an expert scoring relevance.'},
                    {'role':'user','content':prompt}
                ]
            )
            return float(resp.choices[0].message.content.strip())
        except:
            if model=='gemini-pro': return self._llm_score('gpt-4', brand, keyword)
            return 0.0

    def _get_volume(self, term):
        # DataForSEO expects:
        # POST https://api.dataforseo.com/v3/keywords_data/google/search_volume/live
        # [ { "location_code":2840, "language_code":"en", "keywords":["term"] } ]
        if not DATAFORESEO_USER or not DATAFORESEO_PASS:
            return 0.0
        endpoint = 'https://api.dataforseo.com/v3/keywords_data/google/search_volume/live'
        payload = [{
            'location_code': 2840,
            'language_code': 'en',
            'keywords': [term]
        }]
        try:
            r = requests.post(
                endpoint,
                auth=(DATAFORESEO_USER, DATAFORESEO_PASS),
                json=payload,
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            # extract from tasks[0].result[0].search_volume
            vol = data['tasks'][0]['result'][0].get('search_volume', 0)
            return float(vol)
        except Exception as e:
            st.error(f"DataForSEO error for '{term}': {e}")
            return 0.0

    def score(self, brand, keyword):
        gem = self._llm_score('gemini-pro', brand, keyword)
        gpt = self._llm_score('gpt-4', brand, keyword)
        # volumes
        v_kw = self._get_volume(keyword)
        v_bk = self._get_volume(f"{brand} {keyword}")
        share = (v_bk / v_kw * 100) if v_kw>0 else 0.0
        combined = gem*self.gem_w + gpt*self.gpt_w + share*self.vol_w
        return {
            'Brand': brand,
            'Keyword': keyword,
            'Gemini-Pro': round(gem,1),
            'GPT-4': round(gpt,1),
            'Volume KW': round(v_kw,1),
            'Volume BK': round(v_bk,1),
            'Share %': round(share,1),
            'Combined': round(combined,1)
        }

# Streamlit App UI
st.title("Brand vs. Topic Relevance: LLM + Search Volume Share")

# Weight sliders
col1, col2, col3 = st.columns(3)
with col1:
    gem_w = st.slider("Gemini-Pro weight", 0.0, 1.0, 0.3)
with col2:
    gpt_w = st.slider("GPT-4 weight", 0.0, 1.0, 0.3)
with col3:
    vol_w = st.slider("Volume Share weight", 0.0, 1.0, 0.4)

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
    brands = [b.strip() for b in brands_input.split(',') if b.strip()]
    keywords = [k.strip() for k in keywords_input.splitlines() if k.strip()]
    if not brands or not keywords:
        st.warning("Please enter at least one brand and one keyword.")
    else:
        scorer = RelevanceScorer(gem_w, gpt_w, vol_w)
        rows = []
        for brand in brands:
            for keyword in keywords:
                rows.append(scorer.score(brand, keyword))
        df = pd.DataFrame(rows)

        # Combined Scores Chart
        fig = px.bar(
            df,
            x='Keyword',
            y='Combined',
            color='Brand',
            barmode='group',
            title='Combined Relevance Scores by Keyword and Brand'
        )
        fig.update_layout(yaxis_title='Score (0-100)', xaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

        # Detailed Table
        st.subheader("Detailed Scores and Volume Metrics")
        st.table(df.set_index(['Brand', 'Keyword']))
