import os
import re
import requests
import pandas as pd
import streamlit as st
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ========================
# CONFIG & MODELS
# ========================
API_KEY = st.secrets["NEWS_API_KEY"]

CATEGORIES = [
    "business", "entertainment", "general",
    "health", "science", "sports", "technology"
]

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

summarizer = load_summarizer()
sentiment_model = load_sentiment()
embedder = load_embedder()

# ========================
# HELPERS
# ========================
def fetch_news(category=None, limit=10, source=None, keywords=None):
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
    if category:
        url += f"&category={category}"
    if source:
        url += f"&sources={source}"

    response = requests.get(url).json()
    articles = response.get("articles", [])[:limit]

    data = []
    for a in articles:
        text = (a.get("description") or "") + " " + (a.get("content") or "")
        if keywords:
            if not any(k.lower() in text.lower() for k in keywords):
                continue
        data.append({
            "title": a.get("title"),
            "content": text,
            "url": a.get("url"),
            "publishedAt": a.get("publishedAt"),
            "source": a["source"]["name"] if a.get("source") else "Unknown"
        })
    return pd.DataFrame(data)

def summarize_article(text):
    if not text or len(text.split()) < 20:
        return "Not enough text to summarize."
    
    input_len = len(text.split())
    max_len = min(250, int(input_len * 0.8))
    min_len = max(60, int(input_len * 0.3))

    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
    points = summary.split(". ")
    bullet_summary = "\n".join([f"- {p.strip()}" for p in points if len(p) > 20])
    return bullet_summary

def extract_stats(text):
    if not text:
        return "No stats found."
    stats = re.findall(r'\b[\d,.]+(?:%| million| billion| trillion| cases| votes| points| USD| dollars)?\b', text)
    return ", ".join(stats) if stats else "No stats found."

def get_sentiment(text):
    if not text:
        return "Neutral"
    result = sentiment_model(text[:512])[0]  # limit to 512 chars
    return result["label"]

def generate_wordcloud(texts):
    combined = " ".join(texts)
    wc = WordCloud(width=800, height=400, background_color="white").generate(combined)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

def cluster_topics(texts, n_clusters=5):
    if len(texts) < n_clusters:
        return {}
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    clusters = {}
    for i, label in enumerate(km.labels_):
        clusters.setdefault(label, []).append(texts[i])
    return clusters

def build_faiss_index(texts):
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# ========================
# STREAMLIT UI
# ========================
st.title("📰 Ultimate News Summarizer & Insights Dashboard")

# Sidebar
category = st.sidebar.selectbox("Choose a category", ["All"] + CATEGORIES)
source = st.sidebar.text_input("Source filter (optional)")
keywords = st.sidebar.text_input("Custom keywords (comma separated)")
limit = st.sidebar.slider("Number of articles", 1, 20, 5)
fetch = st.sidebar.button("Fetch News")

if fetch:
    df = fetch_news(None if category=="All" else category,
                    limit=limit,
                    source=source if source else None,
                    keywords=[k.strip() for k in keywords.split(",")] if keywords else None)

    if df.empty:
        st.warning("No articles found. Try different filters.")
    else:
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Summaries", "Sentiment", "Hot Topics", "Keywords", "By Source", "Word Cloud", "Research Assistant"
        ])

        # --- Tab 1: Summaries ---
        with tab1:
            st.header("Hybrid Summaries")
            for _, row in df.iterrows():
                st.subheader(row["title"])
                st.caption(f"{row['source']} | {row['publishedAt']}")
                st.markdown(f"[Read full article]({row['url']})")
                st.info(f"📊 Stats: {extract_stats(row['content'])}")
                st.write(summarize_article(row["content"]))
                st.markdown("---")

        # --- Tab 2: Sentiment ---
        with tab2:
            st.header("Sentiment Analysis")
            df["sentiment"] = df["content"].apply(get_sentiment)
            st.dataframe(df[["title", "sentiment"]])
            sentiment_counts = df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

        # --- Tab 3: Hot Topics ---
        with tab3:
            st.header("Hot Topics Today")
            clusters = cluster_topics(df["title"].tolist(), n_clusters=3)
            for label, titles in clusters.items():
                st.subheader(f"Topic {label+1}")
                for t in titles:
                    st.write(f"- {t}")

        # --- Tab 4: Keywords ---
        with tab4:
            st.header("Personalized Keywords")
            if keywords:
                for _, row in df.iterrows():
                    st.write(f"**{row['title']}**")
                    st.write(summarize_article(row["content"]))
            else:
                st.write("Enter keywords in the sidebar to filter.")

        # --- Tab 5: By Source ---
        with tab5:
            st.header("Summarize by Source")
            for src in df["source"].unique():
                st.subheader(src)
                subset = df[df["source"] == src]
                for _, row in subset.iterrows():
                    st.write(f"- {row['title']}")

        # --- Tab 6: Word Cloud ---
        with tab6:
            st.header("Trending Word Cloud")
            fig = generate_wordcloud(df["title"].tolist())
            st.pyplot(fig)

        # --- Tab 7: Research Assistant ---
        with tab7:
            st.header("Ask the News Assistant")
            query = st.text_input("Ask about today's news...")
            if query:
                index, embeddings = build_faiss_index(df["content"].tolist())
                q_emb = embedder.encode([query])
                D, I = index.search(q_emb, k=1)
                match_idx = I[0][0]
                st.write(f"Closest article: {df.iloc[match_idx]['title']}")
                st.write("Summary:")
                st.write(summarize_article(df.iloc[match_idx]["content"]))
