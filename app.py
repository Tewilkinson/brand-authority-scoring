import os
import pandas as pd
import streamlit as st
import plotly.express as px
from openai import OpenAI
from pytrends.request import TrendReq
import time # For rate limiting and exponential backoff

# --- Configuration for API Keys ---
# Set these as environment variables for security and ease of deployment
# Example:
# export OPENAI_API_KEY="sk-..."
# export GEMINI_API_KEY="AIza..."
# export GEMINI_API_BASE="https://generativelanguage.googleapis.com/v1beta" # For Google AI Studio

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta") # Default for AI Studio

if not OPENAI_API_KEY:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize OpenAI client for GPT models
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Gemini client using the base_url
gemini_client = None
if GEMINI_API_KEY:
    try:
        # Use base_url for non-OpenAI endpoints when using the OpenAI library
        gemini_client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_API_BASE)
        
        # --- IMPORTANT: VERIFY THIS GEMINI MODEL NAME ---
        # Go to Google AI Studio or Vertex AI and check the exact deployed model name.
        # Common options: "gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-1.5-flash-002", "gemini-1.5-pro", "gemini-1.5-pro-001"
        TEST_GEMINI_MODEL = "gemini-1.5-flash" # <--- **ADJUST THIS BASED ON YOUR ACCESS**
        
        # Attempt to list models to confirm connection and access
        # This is a good health check
        try:
            available_models = gemini_client.models.list()
            found_gemini_model = False
            for model_obj in available_models:
                if model_obj.id == TEST_GEMINI_MODEL:
                    found_gemini_model = True
                    break
            
            if found_gemini_model:
                st.success(f"Successfully initialized and verified access to Gemini model: {TEST_GEMINI_MODEL}")
            else:
                st.warning(f"Connected to Gemini API but model '{TEST_GEMINI_MODEL}' not found or accessible. Please check model name and permissions.")
                st.warning("Falling back to GPT-4 for Gemini scores.")
                gemini_client = client # Fallback to OpenAI client
        except Exception as e:
            st.warning(f"Could not verify Gemini model access (error: {e}). Falling back to GPT-4 for Gemini scores. Check GEMINI_API_BASE and API Key.")
            gemini_client = client # Fallback if model listing fails
            
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}. Ensure GEMINI_API_KEY and GEMINI_API_BASE are correct.")
        gemini_client = client # Fallback if initialization itself fails
else:
    st.warning("GEMINI_API_KEY not set. Gemini scores will use GPT-4 as a fallback.")
    gemini_client = client


class RelevanceScorer:
    """
    Blends Gemini-Pro, GPT-4, and Google Trends for relevance (0-100).
    """
    def __init__(self, gemini_w: float, gpt4_w: float, pop_w: float):
        total = gemini_w + gpt4_w + pop_w
        self.gemini_w = gemini_w / total if total else 0
        self.gpt4_w    = gpt4_w    / total if total else 0
        self.pop_w     = pop_w     / total if total else 0
        
        # Initialize pytrends once per scorer instance
        # tz=360 for UTC-6. Adjust hl (host language) and tz (timezone) as needed.
        # Removing retries and backoff_factor from here as we're doing custom handling
        self.pytrends = TrendReq(hl='en-US', tz=360) 


    def _llm_score(self, model_alias: str, brand: str, keyword: str) -> float:
        prompt = (
            f"On a scale of 0 to 100, how relevant is the brand '{brand}' "
            f"to the topic '{keyword}'? Reply with only the integer score."
        )
        
        client_to_use = None
        actual_model_name = None

        if "gemini" in model_alias.lower():
            client_to_use = gemini_client
            # --- IMPORTANT: ENSURE THIS IS THE CORRECT, ACCESSIBLE GEMINI MODEL ---
            actual_model_name = "gemini-1.5-flash" # <--- **ADJUST THIS IF NEEDED**
        elif "gpt-4" in model_alias.lower():
            client_to_use = client # This is your OpenAI client for GPT-4
            actual_model_name = "gpt-4o" # Or "gpt-4-turbo", "gpt-4" - **ADJUST AS PER YOUR OPENAI ACCESS**
        else:
            st.error(f"Unsupported LLM model alias: {model_alias}")
            return 0.0

        if client_to_use is None:
            return 0.0 # This should be caught by the initialization health check

        try:
            resp = client_to_use.chat.completions.create(
                model=actual_model_name,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are an expert in brand-topic relevance scoring."},
                    {"role": "user", "content": prompt}
                ]
            )
            score = float(resp.choices[0].message.content.strip())
            return max(0.0, min(100.0, score)) # Ensure score is within 0-100
        except Exception as e:
            st.warning(f"Error with {model_alias} ({actual_model_name}) for '{brand}' and '{keyword}': {e}")
            return 0.0 # Return 0 if LLM call fails


    def _popularity(self, brand: str, keyword: str) -> float:
        """
        Uses Google Trends (via pytrends) to fetch search interest for 'brand keyword'.
        Returns a normalized 0-100 score based on trend data.
        Implements exponential backoff for rate limiting (429 errors).
        """
        term = f"{brand} {keyword}"
        max_retries = 5
        base_delay = 2 # seconds, starting delay for pytrends

        for attempt in range(max_retries):
            try:
                # Build payload for the term
                # timeframe='today 3-m' for last 3 months, 'today 12-m' for last 12 months
                # cat=0 means 'All Categories', geo='' means 'Worldwide'
                self.pytrends.build_payload([term], cat=0, timeframe='today 3-m', geo='')
                
                # Get interest over time
                df_trend = self.pytrends.interest_over_time()

                if df_trend.empty or term not in df_trend.columns:
                    # If no data, it means low or no search interest for the term
                    return 0.0
                
                # Google Trends already returns values on a 0-100 scale.
                score = float(df_trend[term].iloc[-1])
                return score
            except Exception as e:
                error_message = str(e)
                # Check for rate limit error (HTTP 429 or similar indicators)
                if "429" in error_message or "Too Many Requests" in error_message or "ResponseError: The request failed: Google returned a response with code 429" in error_message:
                    delay = base_delay * (2 ** attempt) # Exponential backoff
                    st.warning(f"[Google Trends] Rate limit hit for '{term}'. Retrying in {delay:.1f} seconds (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                elif "method_whitelist" in error_message:
                    st.error(f"[Google Trends] Compatibility error with urllib3 for '{term}'. "
                             f"Please run `pip install \"urllib3<2\"` in your terminal and restart the app. Error: {error_message}")
                    return 0.0 # Stop trying for this specific error
                else:
                    st.error(f"[Google Trends] Unhandled error fetching '{term}': {error_message}")
                    return 0.0 # Return 0 for other errors
        
        st.error(f"[Google Trends] Failed to fetch '{term}' after {max_retries} attempts due to persistent rate limiting.")
        return 0.0 # Return 0 if all retries fail


    def score(self, brand: str, keyword: str) -> dict:
        # Use a specific Gemini model name alias for the call
        g_score = self._llm_score('gemini-llm', brand, keyword) 
        # Use a specific GPT-4 model name alias for the call
        c_score = self._llm_score('gpt-4-llm', brand, keyword)
        p_score = self._popularity(brand, keyword) 
        
        combined = g_score * self.gemini_w + c_score * self.gpt4_w + p_score * self.pop_w
        return {
            'Brand': brand,
            'Keyword': keyword,
            'Gemini LLM Score': round(g_score,1), # More descriptive name
            'GPT-4 LLM Score':  round(c_score,1), # More descriptive name
            'Google Trends Search Interest': round(p_score,1), 
            'Combined Score':   round(combined,1)
        }

# Streamlit App
st.set_page_config(page_title="Brand vs. Topic Authority", layout="wide")
st.title("Brand vs. Topic Authority: LLM + Google Trends Powered")
st.markdown("This tool combines scores from Google's Gemini LLM, OpenAI's GPT-4, and Google Trends search interest to evaluate brand relevance and authority for specific keywords.")

# Input fields
brands_in = st.text_input("Enter Brands (comma-separated)", "Nike, Adidas, Puma")
kws_in    = st.text_input("Enter Keywords (comma-separated)", "new trainers, ice cream")

st.subheader("Scoring Weights")
col1, col2, col3 = st.columns(3)
with col1:
    gem_w = st.slider("Gemini LLM Weight", 0.0, 1.0, 0.4, help="Weight for Google Gemini's relevance assessment.")
with col2:
    gpt_w = st.slider("GPT-4 LLM Weight", 0.0, 1.0, 0.4, help="Weight for OpenAI GPT-4's relevance assessment.")
with col3:
    pop_w = st.slider("Google Trends Weight", 0.0, 1.0, 0.2, help="Weight for Google Trends search interest data. This reflects public popularity.")

# Ensure weights sum to 1 (or allow slight deviation and normalize internally)
total_weights = gem_w + gpt_w + pop_w
if total_weights == 0:
    st.warning("All weights are zero. Please adjust to get meaningful scores.")
elif total_weights != 1.0:
    st.info(f"Weights sum to {total_weights:.1f}. They will be normalized internally.")

enable_trends = st.checkbox("Include Google Trends data", True, help="Uncheck to exclude Google Trends from the combined score calculation.")

if st.button("Compute Scores"):
    brands = [b.strip() for b in brands_in.split(',') if b.strip()]
    kws    = [k.strip() for k in kws_in.split(',')    if k.strip()]
    
    if not brands or not kws:
        st.warning("Please enter at least one brand and one keyword to compute scores.")
    else:
        scorer = RelevanceScorer(
            gemini_w=gem_w,
            gpt4_w=gpt_w,
            pop_w=(pop_w if enable_trends else 0.0)
        )
        
        rows = []
        progress_text = "Computing scores... Please wait. This may take a while for many combinations due to API calls."
        progress_bar = st.progress(0, text=progress_text)
        total_iterations = len(brands) * len(kws)
        current_iteration = 0

        for brand in brands:
            for kw in kws:
                rows.append(scorer.score(brand, kw))
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations, text=f"Processing: Brand '{brand}', Keyword '{kw}' ({current_iteration}/{total_iterations})")
                
                # Small delay after each pair to mitigate overall rate limits, not just pytrends specific
                time.sleep(0.1) 
        
        progress_bar.empty() # Remove progress bar once done
        st.success("Score computation complete!")

        df = pd.DataFrame(rows)
        
        # Display Results
        st.subheader("Combined Relevance Scores Overview")
        # Ensure column names match for plotting
        fig = px.bar(
            df,
            x='Keyword', y='Combined Score', color='Brand',
            barmode='group', title='Combined Relevance Scores by Brand and Keyword'
        )
        fig.update_layout(yaxis_title='Score (0-100)', xaxis_title='Keyword')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detailed Scores")
        # Rename columns for better display in the table
        df_display = df.rename(columns={
            'Gemini LLM Score': 'Gemini LLM',
            'GPT-4 LLM Score': 'GPT-4 LLM',
            'Google Trends Search Interest': 'Google Trends'
        })
        st.dataframe(df_display.set_index(['Brand', 'Keyword']))
