import pickle
import os
import json
import re
from collections import defaultdict, Counter
from rank_bm25 import BM25Okapi
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np  
import faiss
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "indexing", "index_data")
DATA_DIR = "data_retrieval/kochwiki_data"

os.makedirs(INDEX_DIR, exist_ok=True)

GERMAN_STOPWORDS = set(stopwords.words('german'))

def tokenize(text, remove_stopwords=True):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in GERMAN_STOPWORDS]
    return tokens

# ====== CORPUS ======

def build_corpus():
    with open(f"{DATA_DIR}/rezepte_parsed.json", "r") as f:
        recipes = json.load(f)

    with open(f"{DATA_DIR}/zutaten_parsed.json", "r") as f:
        ingredients = json.load(f)

    corpus = {}
    for i, r in enumerate(recipes):
        doc_id = f"recipe_{i}"
        corpus[doc_id] = {
            "title": r["title"],
            "text": r["plaintext"],
            "type": "recipe"
        }

    for i, z in enumerate(ingredients):
        doc_id = f"ingredient_{i}"
        corpus[doc_id] = {
            "title": z["title"],
            "text": z["plaintext"],
            "type": "ingredient"
        }

    with open(f"{INDEX_DIR}/corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)
    
    print(f"Corpus built with {len(corpus)} documents.")
    return corpus

# ====== INVERTED INDEX ======

def build_inverted_index(corpus):
    inverted_index = defaultdict(list)
    for doc_id, doc in corpus.items():
        tokens = tokenize(doc["text"])
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            inverted_index[token].append((doc_id, count))

    with open(f"{INDEX_DIR}/inverted_index.pkl", "wb") as f:
        pickle.dump(dict(inverted_index), f)
    
    print(f"Inverted index built with {len(inverted_index)} unique tokens.")
    return inverted_index

# ====== BM25 ======

def build_bm25(corpus):
    doc_ids = list(corpus.keys())
    tokenized_corpus = [tokenize(doc["text"]) for doc in corpus.values()]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(f"{INDEX_DIR}/bm25.pkl", "wb") as f:
        pickle.dump((doc_ids, bm25), f)

    print(f"BM25 model built with {len(doc_ids)} documents.")
    return doc_ids, bm25

# ====== FAISS ======

def build_faiss(corpus):
    model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer')
    doc_ids = list(corpus.keys())
    texts = [doc["text"] for doc in corpus.values()]

    print("Encoding documents...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, f"{INDEX_DIR}/faiss_index.bin")
    with open(f"{INDEX_DIR}/faiss_doc_ids.pkl", "wb") as f:
        pickle.dump(doc_ids, f)

    print(f"FAISS index built with {len(doc_ids)} documents.")
    return doc_ids, index

# ====== KNOWLEDGE GRAPH ======

def build_knowledge_graph(corpus):
    G = nx.Graph()

    with open(f"{DATA_DIR}/rezepte_parsed.json", "r") as f:
        recipes = json.load(f)

    with open(f"{DATA_DIR}/zutaten_parsed.json", "r") as f:
        ingredients = json.load(f)

    for i, r in enumerate(recipes):
        G.add_node(f"recipe_{i}", 
                   title=r["title"],
                   type="recipe",
                   schwierigkeit=r["metadata"]["schwierigkeit"],
                   zeit=r["metadata"]["zeit"])

        for zutat in r["zutaten_namen"]:
            G.add_edge(f"recipe_{i}", f"zutat:{zutat.lower()}", relation="uses")
    
    for i, z in enumerate(ingredients):
        node_id = f"zutat:{z['name'].lower()}"
        G.add_node(node_id,
                   title=z["name"],
                   type="ingredient")

        for verwandt in z.get("verwandte_zutaten", []):
            G.add_edge(node_id, f"zutat:{verwandt.lower()}", relation="related_to")

    nx.write_gml(G, f"{INDEX_DIR}/knowledge_graph.gml")
    print(f"Knowledge graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

if __name__ == "__main__":
    corpus = build_corpus()
    inverted_index = build_inverted_index(corpus)
    doc_ids, bm25 = build_bm25(corpus)
    doc_ids, faiss_index = build_faiss(corpus)
    graph = build_knowledge_graph(corpus)
    print("\nAll indices built successfully!")