import os
import pandas as pd
import streamlit as st
import plotly.express as px
from openai import OpenAI

# Import pytrends for Google Trends
from pytrends.request import TrendReq
import time # For rate limiting if needed

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
    gemini_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_API_BASE) # Use base_url for custom API endpoints
else:
    gemini_client = client

# Initialize TrendReq globally or pass it
# pytrends = TrendReq(hl='en-US', tz=360) # Initialize once

class RelevanceScorer:
    """
    Blends Gemini-Pro, GPT-4, and Google Trends for relevance (0-100).
    """
    def __init__(self, gemini_w: float, gpt4_w: float, pop_w: float):
        total = gemini_w + gpt4_w + pop_w
        self.gemini_w = gemini_w / total if total else 0
        self.gpt4_w    = gpt4_w    / total if total else 0
        self.pop_w     = pop_w     / total if total else 0
        # Initialize pytrends here to ensure it's available for _popularity
        self.pytrends = TrendReq(hl='en-US', tz=360) # hl for host language, tz for timezone offset

    def _llm_score(self, model: str, brand: str, keyword: str) -> float:
        prompt = (
            f"On a scale of 0 to 100, how relevant is the brand '{brand}' "
            f"to the topic '{keyword}'? Reply with only the integer score."
        )
        # Choose the correct client based on the model
        client_to_use = gemini_client if "gemini" in model else client
        try:
            resp = client_to_use.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are an expert in brand-topic relevance scoring."},
                    {"role": "user", "content": prompt}
                ]
            )
            return float(resp.choices[0].message.content.strip())
        except Exception as e:
            st.warning(f"Error with {model} for '{brand}' and '{keyword}': {e}")
            # Fallback for Gemini if it fails, try GPT-4 (assuming client is GPT-4 by default)
            if "gemini" in model and model != "gpt-4":
                try:
                    fb = client.chat.completions.create( # Use 'client' which is GPT-4
                        model="gpt-4",
                        temperature=0.1,
                        messages=[
                            {"role": "system", "content": "You are an expert in brand-topic relevance scoring."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return float(fb.choices[0].message.content.strip())
                except Exception as fb_e:
                    st.error(f"Fallback GPT-4 also failed: {fb_e}")
                    return 0.0
            return 0.0

    def _popularity(self, brand: str, keyword: str) -> float:
        """
        Uses Google Trends (via pytrends) to fetch search interest for 'brand keyword'.
        Returns a normalized 0-100 score based on trend data.
        """
        term = f"{brand} {keyword}"
        try:
            # Build payload for the term
            # timeframe='today 3-m' for last 3 months, 'today 12-m' for last 12 months
            self.pytrends.build_payload([term], cat=0, timeframe='today 3-m', geo='')
            
            # Get interest over time
            df_trend = self.pytrends.interest_over_time()

            if df_trend.empty or term not in df_trend.columns:
                return 0.0
            
            # The last value in the 'interest over time' series usually represents the most recent interest.
            # Normalize to 0-100 (Google Trends already returns values on a 0-100 scale,
            # where 100 is the peak popularity for the given region and time frame).
            # We'll just take the most recent value directly.
            score = float(df_trend[term].iloc[-1])
            return score
        except Exception as e:
            st.error(f"[Google Trends] Error fetching '{term}': {e}")
            # Google Trends can rate limit, adding a small delay might help in production
            time.sleep(1) 
            return 0.0

    def score(self, brand: str, keyword: str) -> dict:
        g_score = self._llm_score('gemini-pro', brand, keyword)
        c_score = self._llm_score('gpt-4',      brand, keyword)
        p_score = self._popularity(brand, keyword) # Now uses Google Trends
        combined = g_score * self.gemini_w + c_score * self.gpt4_w + p_score * self.pop_w
        return {
            'Brand': brand,
            'Keyword': keyword,
            'Gemini-Pro': round(g_score,1),
            'GPT-4':      round(c_score,1),
            'Trends':     round(p_score,1), # Renamed from 'Popularity' to 'Trends' for clarity
            'Combined':   round(combined,1)
        }

# Streamlit App
st.title("Brand vs. Topic Authority: LLM + Trends")

brands_in = st.text_input("Brands (comma-separated)",  "Nike, Adidas, Puma")
kws_in    = st.text_input("Keywords (comma-separated)", "new trainers, ice cream")

enable_trends = st.checkbox("Include Google Trends data", True)
gem_w = st.slider("Gemini-Pro weight", 0.0, 1.0, 0.4)
gpt_w = st.slider("GPT-4 weight",      0.0, 1.0, 0.4)
pop_w = st.slider("Trends weight",     0.0, 1.0, 0.2)

if st.button("Compute Scores"):
    brands = [b.strip() for b in brands_in.split(',') if b.strip()]
    kws    = [k.strip() for k in kws_in.split(',')    if k.strip()]
    
    if not brands or not kws:
        st.warning("Please enter at least one brand and one keyword.")
    else:
        scorer = RelevanceScorer(
            gemini_w=gem_w,
            gpt4_w=gpt_w,
            pop_w=(pop_w if enable_trends else 0.0)
        )
        
        rows = []
        progress_bar = st.progress(0)
        total_iterations = len(brands) * len(kws)
        current_iteration = 0

        for brand in brands:
            for kw in kws:
                rows.append(scorer.score(brand, kw))
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)
                # Add a small delay between Google Trends calls to avoid rate limiting
                time.sleep(0.5) 

        df = pd.DataFrame(rows)
        
        # Grouped Bar Chart
        fig = px.bar(
            df,
            x='Keyword', y='Combined', color='Brand',
            barmode='group', title='Combined Relevance Scores'
        )
        fig.update_layout(yaxis_title='Score (0-100)', xaxis_title='')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Table
        st.subheader("Detailed Scores by Model and Combined")
        st.table(df.set_index(['Brand','Keyword']))
