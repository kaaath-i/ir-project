import streamlit as st
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Kochwiki Search", page_icon="🍳", layout="wide")

st.title("🍳 Kochwiki Search")
st.caption("Semantic recipe search across 14,000+ German recipes from Kochwiki.org")

HF_REPO = "kaaath-i/kochwiki-ir-data"

@st.cache_resource
def download_and_load():
    local_index = os.path.join("indexing", "index_data", "corpus.pkl")
    
    if os.path.exists(local_index):
        os.environ["INDEX_DIR"] = os.path.join("indexing", "index_data")
    else:
        os.makedirs("index_data", exist_ok=True)
        files = [
            "index_data/corpus.pkl",
            "index_data/inverted_index.pkl",
            "index_data/bm25.pkl",
            "index_data/faiss_index.bin",
            "index_data/faiss_doc_ids.pkl",
            "index_data/knowledge_graph.gml"
        ]
        for f in files:
            hf_hub_download(repo_id=HF_REPO, filename=f, repo_type="dataset", local_dir=".")
        os.environ["INDEX_DIR"] = "index_data"

    from search.retrieval import load_indices, load_faiss, load_graph
    
    corpus, inverted_index, bm25_data = load_indices()
    faiss_index, faiss_doc_ids, model = load_faiss()
    graph, synonyms = load_graph()
    return corpus, inverted_index, bm25_data, faiss_index, faiss_doc_ids, model, graph, synonyms

corpus, inverted_index, bm25_data, faiss_index, faiss_doc_ids, model, graph, synonyms = download_and_load()

from search.retrieval import bm25_search, faiss_search, graph_search, hybrid_search


with st.sidebar:
    st.header("🔧 Filter")
    search_method = st.radio("Search Method", ["Hybrid", "BM25", "Semantic (FAISS)", "Graph"])
    doc_type = st.radio("Document Type", ["Recipes only", "Ingredients only", "Both"])
    n_results = st.slider("Number of results", 3, 20, 5)

    doc_type_map = {"Recipes only": "recipe", "Ingredients only": "ingredient", "Both": None}
    selected_type = doc_type_map[doc_type]

query = st.text_input("🔍 Was möchtest du kochen?", placeholder="z.B. Kartoffelsuppe, Curry, japanische Suppe...")

if search_method == "Graph":
    zutaten_input = st.text_input("🥕 Zutaten (kommagetrennt)", placeholder="z.B. Kartoffel, Zwiebel, Speck")

if query or (search_method == "Graph" and 'zutaten_input' in dir() and zutaten_input):
    
    if search_method == "BM25":
        results = bm25_search(query, corpus, bm25_data, n=n_results, synonyms=synonyms, doc_type=selected_type)
    elif search_method == "Semantic (FAISS)":
        results = faiss_search(query, corpus, faiss_index, faiss_doc_ids, model, n=n_results, doc_type=selected_type)
    elif search_method == "Graph":
        zutaten = [z.strip() for z in zutaten_input.split(",")]
        results = graph_search(zutaten, graph, corpus, n=n_results)
    else:
        results = hybrid_search(query, corpus, bm25_data, faiss_index, faiss_doc_ids, model, 
                                synonyms=synonyms, doc_type=selected_type, n=n_results)

    st.markdown(f"### Results ({len(results)})")
    
    for i, result in enumerate(results):
        if search_method == "Graph":
            doc_id, title = result
            score = None
        else:
            doc_id, title, score = result
        
        wiki_url = f"https://www.kochwiki.org/wiki/{title.replace(' ', '_')}"
        
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{i+1}. [{title}]({wiki_url})**")
                text_preview = corpus[doc_id]["text"][:200] + "..."
                st.caption(text_preview)
            with col2:
                if score is not None:
                    st.metric("Score", f"{score:.3f}")
            st.divider()