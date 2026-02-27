# KochWiki Information Retrieval System
🚧 *Work in progress*

## Overview

An Information Retrieval & Extraction system built on Kochwiki.org, one of the largest German-language open recipe collections (~14,000 recipes + ~4,000 ingredient pages). The system compares different retrieval methods on German-language food data.

Part of my Master's coursework in Information Extraction and Retrieval (Multilingual Technologies) at Hochschule Campus Wien.

## Data Source

Kochwiki.org: https://www.kochwiki.org/ (Creative Commons Attribution-ShareAlike)

Scraped via the MediaWiki API. Raw data is in MediaWiki markup (wikitext), parsed into structured JSON containing metadata, ingredients, preparation steps, nutritional data, and cuisine classifications.

*Note:* The scraped data is not included in this repository due to file size (~150 MB). Run the [scraper](data_retrieval/kochwiki_scraper.py) to generate the data locally (**Attention**: this takes about 3 hours).  Alternatively, the dataset is available [here](https://drive.google.com/drive/folders/1djCIp51luMTYEICLFRYm4aH-Ucm6IfXh?usp=drive_link)

## Project Status
- [x] Data scraping & parsing
- [x] BM25 indexing & retrieval
- [x] Semantic search
- [x] Initial evaluation (10 queries)
- [ ] Continuously improving throughout the semester

## Retrieval Methods (in progress)

- Baseline: Simple substring matching on full-text
- BM25: Okapi BM25 via rank-bm25
- Semantic Search: Sentence embeddings via SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)

---

*This is an academic project for learning purposes.*




