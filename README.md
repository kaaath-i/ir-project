<!--
title: Kochwiki Suche
emoji: 🍳
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: "1.55.0"
app_file: app.py
pinned: false
-->

# KochWiki Information Retrieval System
🚧 *Work in progress*

## Overview

An Information Retrieval & Extraction system built on Kochwiki.org, one of the largest German-language open recipe collections (~14,000 recipes + ~4,000 ingredient pages). The system compares different retrieval methods on German-language food data.

Part of my Master's coursework in Information Extraction and Retrieval (Multilingual Technologies) at Hochschule Campus Wien.

🔗 **Try the prototype:** [Kochwiki Search](https://huggingface.co/spaces/kaaath-i/kochwiki-suche)
(*Note:* The user interface is in German, as the underlying data is a German-language recipe corpus.)

## Data Source

Kochwiki.org: https://www.kochwiki.org/ (Creative Commons Attribution-ShareAlike)

Scraped via the MediaWiki API. Raw data is in MediaWiki markup (wikitext), parsed into structured JSON containing metadata, ingredients, preparation steps, nutritional data, and cuisine classifications.

*Note:* The scraped data is not included in this repository due to file size (~150 MB). Run the [scraper](data_retrieval/kochwiki_scraper.py) to generate the data locally (**Attention**: this takes about 3 hours).  Alternatively, the dataset (as well as the index data) is available on [Hugging Face](https://huggingface.co/datasets/kaaath-i/kochwiki-ir-data/tree/main) or on [Google Drive](https://drive.google.com/drive/folders/1lJlfBTZ34HFxGQgMoxhd1-4LoFgxjNBu?usp=drive_link).

## Project Status
- [x] Data scraping & parsing
- [x] BM25 indexing & retrieval
- [x] Semantic search (FAISS)
- [x] Knowledge graph (NetworkX)
- [x] Hybrid search
- [x] Query expansion (graph-based synonyms)
- [x] Stopword removal
- [x] Initial evaluation (10 queries)
- [x] Streamlit UI prototype
- [ ] Continuously improving throughout the semester

## Retrieval Methods (in progress)

- Baseline: Simple substring matching on full-text
- BM25: Okapi BM25 via rank-bm25
- Semantic Search: Sentence embeddings via SentenceTransformers (T-Systems-onsite/cross-en-de-roberta-sentence-transformer)
- Knowledge Graph (NetworkX)

---

*This is an academic project for learning purposes.*




