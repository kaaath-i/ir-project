import streamlit as st
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="KochWiki Suche", page_icon="🍳", layout="wide")

st.title("🍳 KochWiki Suche")
st.caption("Durchsuche 8000+ Rezepte von kochwiki.org")

st.markdown("""
<style>
a { color: #2B1700 !important; }
a:hover { color: #A66038 !important; }
</style>
""", unsafe_allow_html=True)

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
    from search.rag import load_rag
    
    corpus, inverted_index, bm25_data = load_indices()
    faiss_index, faiss_doc_ids, model = load_faiss()
    graph, synonyms = load_graph()
    generator = load_rag()
    return corpus, inverted_index, bm25_data, faiss_index, faiss_doc_ids, model, graph, synonyms, generator

corpus, inverted_index, bm25_data, faiss_index, faiss_doc_ids, model, graph, synonyms, generator = download_and_load()

from search.retrieval import bm25_search, faiss_search, graph_search, hybrid_search
from search.rag import rag_search

# ====== TABS ======
tab1, tab2 = st.tabs(["🧑🏼‍🍳🐀 RAGatouille"], ["🔍 Einfache Suche"])

# ====== TAB 2: KOCH-ASSISTENT ======
with tab1:
    st.caption("Stelle Fragen, bitte um Rezeptempfehlungen oder sag 'gib mir ein anderes Rezept'.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages_display" not in st.session_state:
        st.session_state.messages_display = []

    for message in st.session_state.messages_display:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Frag deinen Koch-Assistenten..."):
 
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages_display.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.status("🍲 In Zubereitung...") as status:
                answer, sources, st.session_state.chat_history = rag_search(
                    prompt, corpus, bm25_data, faiss_index, faiss_doc_ids, model,
                    generator, graph=graph, synonyms=synonyms,
                    chat_history=st.session_state.chat_history
                )
                status.update(label="Fertig!", state="complete")
            
            st.markdown(answer)
            
            with st.expander("📚 Verwendete Rezepte"):
                for doc_id, title, score in sources:
                    wiki_url = f"https://www.kochwiki.org/wiki/{title.replace(' ', '_')}"
                    st.markdown(f"- [{title}]({wiki_url})")

        st.session_state.messages_display.append({"role": "assistant", "content": answer})

    if st.session_state.messages_display:
        if st.button("🗑️ Chat zurücksetzen"):
            st.session_state.chat_history = []
            st.session_state.messages_display = []
            st.rerun()

# ====== TAB 1: SUCHE ======
with tab2:
    with st.sidebar:
        st.header("🔧 Filter")
        search_method = st.radio("Suchmethode", ["Hybrid", "BM25", "Semantic (FAISS)", "Graph"])
        n_results = st.slider("Anzahl der Suchergebnisse", 3, 20, 5)

    query = st.text_input("🔍 Was möchtest du kochen?", placeholder="z.B. Kartoffelsuppe, Curry, japanische Suppe...")

    if search_method == "Graph":
        zutaten_input = st.text_input("🥕 Zutaten (kommagetrennt)", placeholder="z.B. Kartoffel, Zwiebel, Speck")

    if query or (search_method == "Graph" and 'zutaten_input' in dir() and zutaten_input):
        if search_method == "BM25":
            results = bm25_search(query, corpus, bm25_data, n=n_results, synonyms=synonyms)
        elif search_method == "Semantic (FAISS)":
            results = faiss_search(query, corpus, faiss_index, faiss_doc_ids, model, n=n_results)
        elif search_method == "Graph":
            zutaten = [z.strip() for z in zutaten_input.split(",")]
            results = graph_search(zutaten, graph, corpus, n=n_results)
        else:
            results = hybrid_search(query, corpus, bm25_data, faiss_index, faiss_doc_ids, model,
                                    synonyms=synonyms, n=n_results)

        st.markdown(f"### Ergebnisse ({len(results)})")
        
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

