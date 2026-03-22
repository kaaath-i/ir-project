from __future__ import annotations
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from collections import defaultdict, Counter
import re
import networkx as nx
import os
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

INDEX_DIR = os.environ.get("INDEX_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "indexing", "index_data"))

GERMAN_STOPWORDS = set(stopwords.words('german'))

def tokenize(text, remove_stopwords=True):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in GERMAN_STOPWORDS]
    return tokens

# ===== SYONONYMS ======

def build_synonyms_from_graph(graph):
    synonyms = {}
    for node in graph.nodes():
        if node.startswith("zutat:"):
            name = node.replace("zutat:", "")
            related = []
            for neighbor in graph.neighbors(node):
                if neighbor.startswith("zutat:"):
                    related.append(neighbor.replace("zutat:", ""))
            if related:
                synonyms[name] = related
    return synonyms

def expand_query(query, synonyms=None):
    tokens = tokenize(query)
    expanded = list(tokens)
    for token in tokens:
        if token in synonyms:
            expanded.extend(synonyms[token][:5])
    return " ".join(expanded)

# ====== LOAD ======

def load_indices():
    with open(f"{INDEX_DIR}/corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    
    with open(f"{INDEX_DIR}/inverted_index.pkl", "rb") as f:
        inverted_index = pickle.load(f)

    with open(f"{INDEX_DIR}/bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)

    print(f"Loaded: {len(corpus)} documents, {len(inverted_index)} tokens")
    return corpus, inverted_index, bm25_data

def load_faiss():
    index = faiss.read_index(f"{INDEX_DIR}/faiss_index.bin")
    with open(f"{INDEX_DIR}/faiss_doc_ids.pkl", "rb") as f:
        doc_ids = pickle.load(f)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print(f"FAISS loaded: {index.ntotal} vectors")
    return index, doc_ids, model

def load_graph():
    graph = nx.read_gml(f"{INDEX_DIR}/knowledge_graph.gml")
    synonyms = build_synonyms_from_graph(graph)
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph, synonyms

# ===== SEARCH ======

def bm25_search(query, corpus, bm25_data, n=5, synonyms=None, doc_type=None):
    if synonyms:
        query = expand_query(query, synonyms)
    doc_ids, bm25 = bm25_data
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in top_indices:
        if doc_type and corpus[doc_ids[i]]["type"] != doc_type:
            continue
        results.append((doc_ids[i], corpus[doc_ids[i]]["title"], scores[i]))
        if len(results) >= n:
            break
    return results

def faiss_search(query, corpus, faiss_index, faiss_doc_ids, model, n=5, doc_type=None):
    query_embedding = model.encode([query]).astype('float32')
    search_n = n * 3 if doc_type else n
    distances, indices = faiss_index.search(query_embedding, search_n)

    results = []
    for j, i in enumerate(indices[0]):
        if doc_type and corpus[faiss_doc_ids[i]]["type"] != doc_type:
            continue
        results.append((faiss_doc_ids[i], corpus[faiss_doc_ids[i]]["title"], distances[0][j]))
        if len(results) >= n:
            break
    return results

def graph_search(query_zutaten, graph, corpus, n=5):
    recipe_sets = []

    for zutat in query_zutaten:
        zutat_lower = zutat.lower()
        matching_nodes = [node for node in graph.nodes() 
                         if node.startswith("zutat:") and zutat_lower in node]
        recipes = set()

        for node in matching_nodes:
            for neighbor in graph.neighbors(node):
                if neighbor.startswith("recipe_"):
                    recipes.add(neighbor)
        recipe_sets.append(recipes)

    if not recipe_sets:
        return []
    
    common = recipe_sets[0]
    for s in recipe_sets[1:]:
        common = common & s
        
    return [(doc_id, corpus[doc_id]["title"]) for doc_id in list(common)[:n]]

# ===== COMBINED SEARCH ======

def hybrid_search(query, corpus, bm25_data, faiss_index, faiss_doc_ids, model, graph=None, filter_zutaten=None, n=5, bm25_weight=0.5, synonyms=None, doc_type=None):
    if synonyms:
        query = expand_query(query, synonyms)
    doc_ids, bm25 = bm25_data
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_norm = bm25_scores / max_bm25

    query_embedding = model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, len(corpus))
    
    faiss_scores = np.zeros(len(corpus))
    for j, i in enumerate(indices[0]):
        faiss_scores[i] = 1 / (1 + distances[0][j])
    
    max_faiss = max(faiss_scores) if max(faiss_scores) > 0 else 1
    faiss_norm = faiss_scores / max_faiss

    combined = bm25_weight * bm25_norm + (1 - bm25_weight) * faiss_norm
    top_indices = combined.argsort()[::-1]

    if graph and filter_zutaten:
        graph_results = graph_search(filter_zutaten, graph, corpus, n=len(corpus))
        valid_docs = set(doc_id for doc_id, _ in graph_results)
        for i, doc_id in enumerate(doc_ids):
            if doc_id not in valid_docs:
                combined[i] = -1
                
    top_indices = combined.argsort()[::-1]

    results = []
    for i in top_indices:
        if doc_type and corpus[doc_ids[i]]["type"] != doc_type:
            continue
        results.append((doc_ids[i], corpus[doc_ids[i]]["title"], combined[i]))
        if len(results) >= n:
            break
    return results