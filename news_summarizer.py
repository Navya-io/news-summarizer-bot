# Databricks notebook source
import requests
import pandas as pd

# 🔑 Replace with your NewsAPI key
API_KEY = os.getenv("NEWS_API_KEY")

# Example: Get top US headlines
url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"

response = requests.get(url).json()

# Extract relevant fields
articles = response.get("articles", [])
df = pd.DataFrame([{
    "title": a.get("title"),
    "author": a.get("author"),
    "publishedAt": a.get("publishedAt"),
    "content": a.get("content"),
    "url": a.get("url")
} for a in articles])

print(df.head())


# COMMAND ----------

CATEGORIES = [
    "business",
    "entertainment",
    "general",
    "health",
    "science",
    "sports",
    "technology"
]

# COMMAND ----------

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

# COMMAND ----------

pip install transformers


# COMMAND ----------

# MAGIC %pip install torch
# MAGIC

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Load summarizer model (make sure torch is installed!)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# COMMAND ----------

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

# COMMAND ----------

# Step 3: Run with User Choice
# -------------------------------
# Example: user picks category
user_choice = "business"   # can be any from CATEGORIES, or None for all
limit = 5

df = fetch_news(user_choice if user_choice != "All" else None, limit=limit)
df["summary"] = df["content"].apply(summarize_article)

# Display results
for i, row in df.iterrows():
    print(f"\n {row['title']}")
    print(f" Published: {row['publishedAt']}")
    print(f" {row['url']}")
    print(f" Summary: {row['summary']}")
    print("-" * 80)