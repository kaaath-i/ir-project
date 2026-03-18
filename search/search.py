import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from collections import defaultdict, Counter
import re
import networkx as nx

INDEX_DIR = "indexing/index_data"

def simple_tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

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
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph

# ===== SEARCH ======

def bm25_search(query, corpus, bm25_data, n=5):
    tokenized_query = simple_tokenize(query)
    doc_ids, bm25 = bm25_data
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [(doc_ids[i], corpus[doc_ids[i]]["title"], scores[i]) for i in top_indices]

def faiss_search(query, corpus, faiss_index, faiss_doc_ids, model, n=5):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, n)
    return [(faiss_doc_ids[i], corpus[faiss_doc_ids[i]]["title"], distances[0][j]) for j, i in enumerate(indices[0])]

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