---
title: RAGatouille
emoji: 🐀
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: "1.55.0"
app_file: app.py
pinned: false
---

# RAGatouille 🐀
*Your personal cooking assistant!*

## Overview

An Information Retrieval & RAG system built on Kochwiki.org, one of the largest German-language open recipe collections (~14,000 pages, 8,125 recipes after filtering). The system combines multiple retrieval methods with an agentic RAG pipeline for conversational recipe search.

Part of my Master's coursework in Information Extraction and Retrieval (MA Multilingual Technologies) at Hochschule Campus Wien.

🔗 **Try it:** [RAGatouille](https://huggingface.co/spaces/kaaath-i/ragatouille)
(*Note:* The interface is in German, as the underlying data is a German-language recipe corpus.)

## Data Source

Kochwiki.org: https://www.kochwiki.org/ (Creative Commons Attribution-ShareAlike)

Scraped via the MediaWiki API. Raw data is in MediaWiki markup (wikitext), parsed into structured JSON containing metadata, ingredients, preparation steps, and cuisine classifications.

*Note:* The scraped data is not included in this repository due to file size (~150 MB). Run the [scraper](data_retrieval/kochwiki_scraper.py) to generate the data locally (**Attention**: this takes about 3 hours). Alternatively, the dataset and index data are available on [Hugging Face](https://huggingface.co/datasets/kaaath-i/kochwiki-ir-data/tree/main).

## System Architecture

**Retrieval**
- BM25 (rank-bm25) — keyword-based retrieval with German stopword removal
- Semantic Search — FAISS with `intfloat/multilingual-e5-base`, cosine similarity (IndexFlatIP + L2 normalization)
- Hybrid Search — Reciprocal Rank Fusion (RRF) combining BM25 + FAISS
- Knowledge Graph (NetworkX) — ingredient relations, used for query expansion and filtering

**RAG Pipeline**
- Agent (smolagents) — iterative search using Hybrid Search as a tool
- Generation — Groq API (`llama-3.3-70b-versatile`)
- Chat History — multi-turn conversation support

---

*This is an academic project for learning purposes.*




