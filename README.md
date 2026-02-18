# 📰 News Summarizer & Analytics Dashboard

> Real-time news summarization pipeline using Transformer models + semantic search powered by ChromaDB — reducing content volume by 80% while preserving key context.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 What This App Does

| Feature | Details |
|---|---|
| **Live News Fetch** | Pulls real-time articles from NewsAPI across 7 categories |
| **AI Summarization** | BART / DistilBART / Pegasus summarization — 80% content reduction |
| **Keyword Extraction** | TF-IDF extracts top keywords per article automatically |
| **Semantic Search** | ChromaDB + sentence-transformers for natural language retrieval |
| **Analytics Dashboard** | Source breakdown, trends over time, keyword frequency charts |

---

## 🖥️ App Screenshots

> *(Run the app and add your screenshots here)*

| Summaries View | Semantic Search | Analytics |
|---|---|---|
| ![s1](visuals/tab_summaries.png) | ![s2](visuals/tab_search.png) | ![s3](visuals/tab_analytics.png) |

---

## 🚀 Quickstart

### 1 — Clone the repo
```bash
git clone https://github.com/navya-manjunatha/news-summarizer-app.git
cd news-summarizer-app
```

### 2 — Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Get a free NewsAPI key
1. Go to [newsapi.org](https://newsapi.org/register)
2. Sign up for free (takes 30 seconds)
3. Copy your API key

### 5 — Run the app
```bash
streamlit run app.py
```
The app opens at **http://localhost:8501**

### 6 — Use the app
1. Paste your NewsAPI key in the sidebar
2. Enter a topic (e.g. "machine learning", "climate change")
3. Click **Fetch & Summarize**
4. Explore the three tabs!

---

## 📁 Project Structure

```
news-summarizer-app/
│
├── app.py                  ← Streamlit dashboard (main entry point)
├── requirements.txt
├── .gitignore
├── .env.example            ← Template for your API key
│
├── src/
│   ├── __init__.py
│   ├── news_fetcher.py     ← NewsAPI integration
│   ├── summarizer.py       ← Transformer summarization + TF-IDF keywords
│   ├── vector_store.py     ← ChromaDB vector storage + semantic search
│   └── utils.py            ← Shared helpers
│
├── visuals/                ← Screenshots for README
└── .chromadb/              ← Persistent ChromaDB storage (auto-created, gitignored)
```

---

## 🧠 Pipeline Architecture

```
NewsAPI (live feed)
       │
       ▼
news_fetcher.py  ──→  Raw articles (title, content, url, source)
       │
       ▼
summarizer.py  ──→  BART / DistilBART / Pegasus
       │               • 80% content reduction
       │               • TF-IDF keyword extraction
       ▼
vector_store.py  ──→  ChromaDB (sentence-transformers embeddings)
       │               • Persistent semantic index
       │               • Cosine similarity search
       ▼
app.py (Streamlit)
       ├── Tab 1: Summarized articles + filters
       ├── Tab 2: Natural language semantic search
       └── Tab 3: Analytics charts (Plotly)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.32 |
| Summarization | HuggingFace Transformers (BART, DistilBART, Pegasus) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent, local) |
| Data Source | NewsAPI.org |
| Visualization | Plotly Express |
| Keyword Extraction | Scikit-learn TF-IDF |

---

## 📊 Performance

| Metric | Result |
|---|---|
| Content Reduction | **~80%** average |
| Semantic Search Latency | **<500ms** |
| Supported Article Sources | **80,000+** via NewsAPI |
| Max Batch Size | **50 articles** per session |

---

## 🔑 Environment Variables (optional)

Instead of typing your key each time, create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add:  NEWSAPI_KEY=your_key_here
```

---

## 📬 Contact

**Navya Manjunatha** · Data Analyst  
📧 manjunatha.navya10@gmail.com  
🔗 [linkedin.com/in/navya-manjunatha](https://www.linkedin.com/in/navya-manjunatha/)

---
*⭐ Star this repo if it was useful!*
