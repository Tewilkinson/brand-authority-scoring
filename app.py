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

# Streamlit UI omitted for brevity
