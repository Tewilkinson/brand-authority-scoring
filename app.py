import os
import pandas as pd
import streamlit as st
from openai import OpenAI

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE")

if not OPENAI_API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize OpenAI client for GPT models
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Gemini client if credentials provided, else fallback to OpenAI client
if GEMINI_API_KEY and GEMINI_API_BASE:
    gemini_client = OpenAI(api_key=GEMINI_API_KEY, api_base=GEMINI_API_BASE)
else:
    gemini_client = client

class LLMRelevanceScorer:
    """
    Scores brand relevance by combining Gemini-Pro and GPT-4 outputs.
    """
    def __init__(self, gemini_weight: float = 0.5, gpt4_weight: float = 0.5):
        total = gemini_weight + gpt4_weight
        self.gemini_w = gemini_weight / total
        self.gpt4_w = gpt4_weight / total

    def _llm_score(self, model: str, brand: str, keyword: str) -> float:
        prompt = (
            f"On a scale of 0 to 100, how relevant is the brand '{brand}' "
            f"to the topic '{keyword}'? Reply with only the integer score."
        )
        try:
            # Choose client based on model
            chat_client = gemini_client if model == "gemini-pro" else client
            response = chat_client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You are an expert in scoring brand-topic relevance."},
                    {"role": "user", "content": prompt}
                ]
            )
            return float(response.choices[0].message.content.strip())
        except Exception:
            # Fallback: if Gemini-Pro fails, retry with GPT-4
            if model == "gemini-pro":
                try:
                    fallback = client.chat.completions.create(
                        model="gpt-4",
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": "You are an expert in scoring brand-topic relevance."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return float(fallback.choices[0].message.content.strip())
                except Exception:
                    return 0.0
            return 0.0

    def score(self, brand: str, keyword: str) -> dict:
        # Get Gemini-Pro and GPT-4 scores
        gemini_score = self._llm_score("gemini-pro", brand, keyword)
        gpt4_score = self._llm_score("gpt-4", brand, keyword)
        # Compute combined weighted score
        combined_score = gemini_score * self.gemini_w + gpt4_score * self.gpt4_w
        return {
            "Brand": brand,
            "Keyword": keyword,
            "Gemini-Pro (0-100)": gemini_score,
            "GPT-4 (0-100)": gpt4_score,
            "Combined (0-100)": round(combined_score, 1)
        }

# Streamlit UI
st.title("Brand vs. Topic Relevance: GPT-4 + Gemini-Pro")

brands_input = st.text_input("Brands (comma-separated)", "Nike, Adidas, Puma")
keywords_input = st.text_input("Keywords (comma-separated)", "new trainers, ice cream")

gemini_w = st.slider("Gemini-Pro weight", 0.0, 1.0, 0.5, help="Relative weight for Gemini-Pro score")
gpt4_w = st.slider("GPT-4 weight", 0.0, 1.0, 0.5, help="Relative weight for GPT-4 score")

if st.button("Compute Relevance Scores"):
    brands = [b.strip() for b in brands_input.split(',') if b.strip()]
    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
    if not brands or not keywords:
        st.warning("Enter at least one brand and one keyword.")
    else:
        scorer = LLMRelevanceScorer(gemini_weight=gemini_w, gpt4_weight=gpt4_w)
        results = []
        for brand in brands:
            for keyword in keywords:
                results.append(scorer.score(brand, keyword))
        df = pd.DataFrame(results)
        st.dataframe(df)
