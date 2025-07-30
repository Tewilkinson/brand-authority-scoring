import os
import pandas as pd
import streamlit as st
from openai import OpenAI

# Configuration
oai_key = os.getenv('OPENAI_API_KEY')
if not oai_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()
client = OpenAI(api_key=oai_key)

class LLMRelevanceScorer:
    """
    Scores brand relevance by combining ChatGPT (GPT-4) and Gemini-Pro outputs.
    """
    def __init__(self, gemini_weight: float = 0.5, gpt4_weight: float = 0.5):
        # Normalize weights
        total = gemini_weight + gpt4_weight
        self.gemini_w = gemini_weight / total
        self.gpt4_w = gpt4_weight / total

    def _llm_score(self, model: str, brand: str, keyword: str) -> float:
        prompt = (
            f"On a scale of 0 to 100, how relevant is the brand '{brand}' "
            f"to the topic '{keyword}'? Reply with only the integer score."
        )
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You are an expert at scoring brand-topic relevance."},
                    {"role": "user", "content": prompt}
                ]
            )
            return float(response.choices[0].message.content.strip())
        except Exception:
            return 0.0

    def score(self, brand: str, keyword: str) -> dict:
        # Gemini-Pro relevance
        gemini_score = self._llm_score("gemini-pro", brand, keyword)
        # GPT-4 relevance
        gpt4_score = self._llm_score("gpt-4", brand, keyword)
        # Combined weighted score
        combined = gemini_score * self.gemini_w + gpt4_score * self.gpt4_w
        return {
            'Brand': brand,
            'Keyword': keyword,
            'Gemini-Pro (0-100)': gemini_score,
            'GPT-4 (0-100)': gpt4_score,
            'Combined (0-100)': round(combined, 1)
        }

# Streamlit UI
st.title("Brand vs. Topic Relevance: GPT-4 + Gemini-Pro")
st.markdown("### Combined LLM-based relevance scoring (0-100)")

brands = st.text_input("Brands (comma-separated)", "Nike, Adidas, Puma").split(',')
keywords = st.text_input("Topics/Keywords (comma-separated)", "new trainers, ice cream").split(',')

gemini_w = st.slider("Gemini-Pro weight", 0.0, 1.0, 0.5)
gpt4_w = st.slider("GPT-4 weight", 0.0, 1.0, 0.5)

if st.button("Compute Relevance Scores"):
    scorer = LLMRelevanceScorer(gemini_weight=gemini_w, gpt4_weight=gpt4_w)
    results = []
    for brand in [b.strip() for b in brands if b.strip()]:
        for keyword in [k.strip() for k in keywords if k.strip()]:
            results.append(scorer.score(brand, keyword))
    df = pd.DataFrame(results)
    st.dataframe(df)
