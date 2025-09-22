import os
import requests
import pandas as pd
import streamlit as st
from transformers import pipeline

# ----------------------------------------
# Load API Key (from Streamlit secrets)
# ----------------------------------------
API_KEY = os.getenv("NEWS_API_KEY") or st.secrets["NEWS_API_KEY"]

CATEGORIES = [
    "business",
    "entertainment",
    "general",
    "health",
    "science",
    "sports",
    "technology"
]

# ----------------------------------------
# Function to Fetch News
# ----------------------------------------
def fetch_news(category=None, limit=10):
    if category:  # If user picks a category
        url = f"https://newsapi.org/v2/top-headlines?country=us&category={category}&apiKey={API_KEY}"
    else:  # Default = all categories
        url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
    
    response = requests.get(url).json()
    articles = response.get("articles", [])[:limit]
    
    data = []
    for a in articles:
        text = (a.get("description") or "") + " " + (a.get("content") or "")
        data.append({
            "title": a.get("title"),
            "content": text,
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt")
        })
    return pd.DataFrame(data)

# ----------------------------------------
# Summarizer
# ----------------------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_article(text):
    if not text or len(text.split()) < 20:
        return "Not enough text to summarize."
    
    input_len = len(text.split())
    max_len = min(120, int(input_len * 0.6))   # ~60% of input
    min_len = max(30, int(input_len * 0.3))    # ~30% of input
    
    summary = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )
    return summary[0]['summary_text']

# ----------------------------------------
# Streamlit App UI
# ----------------------------------------
st.title("News Summarizer Bot")
st.write("Get the latest headlines and short summaries!")

category = st.selectbox("Choose a category:", ["All"] + CATEGORIES)
limit = st.slider("Number of articles", 1, 10, 5)

if st.button("Fetch News"):
    df = fetch_news(None if category == "All" else category, limit=limit)
    df["summary"] = df["content"].apply(summarize_article)
    
    for i, row in df.iterrows():
        st.subheader(row["title"])
        st.caption(f"Published: {row['publishedAt']}")
        st.markdown(f"[Read full article]({row['url']})")
        st.write(row["summary"])
        st.markdown("---")
