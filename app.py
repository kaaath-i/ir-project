import streamlit as st
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="RAGatouille 🧑🏼‍🍳🐀", page_icon="🧑🏼‍🍳🐀", layout="wide")

st.title("🧑🏼‍🍳🐀 RAGatouille")
st.caption("Dein persönlicher Kochassistent. Frag nach Rezepten, lass dir Alternativen vorschlagen oder hol dir einfach Inspiration für dein nächstes Gericht!")

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
    from search.agent import load_agent, init_agent_resources
    
    corpus, inverted_index, bm25_data = load_indices()
    faiss_index, faiss_doc_ids, model = load_faiss()
    graph, synonyms = load_graph()
    generator = load_rag()
    init_agent_resources(corpus, bm25_data, faiss_index, faiss_doc_ids, model, graph, synonyms)
    agent = load_agent()
    return corpus, inverted_index, bm25_data, faiss_index, faiss_doc_ids, model, graph, synonyms, generator, agent

corpus, inverted_index, bm25_data, faiss_index, faiss_doc_ids, model, graph, synonyms, generator, agent = download_and_load()

from search.retrieval import hybrid_search
from search.agent import agent_search
from search.rag import rag_from_agent

# ====== SIDEBAR ======
with st.sidebar:
    st.header("🔧 Filter")
    n_results = st.slider("Anzahl Suchergebnisse", 3, 20, 5)
    
    st.divider()
    st.markdown("### 🔍 Schnellsuche")
    search_query = st.text_input("", placeholder="z.B. Kartoffelsuppe, Curry...")
    
    if search_query:
        results = hybrid_search(
            search_query, corpus, bm25_data, faiss_index, faiss_doc_ids, model,
            synonyms=synonyms, n=n_results
        )
        for i, (doc_id, title, score) in enumerate(results):
            wiki_url = f"https://www.kochwiki.org/wiki/{title.replace(' ', '_')}"
            st.markdown(f"{i+1}. [{title}]({wiki_url})")

# ====== MAIN: CHAT ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages_display" not in st.session_state:
    st.session_state.messages_display = []

# Chat History 
for message in st.session_state.messages_display:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Frag RAGatouille..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages_display.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.status("🍲 In Zubereitung...") as status:
            titles = agent_search(prompt, agent)
            answer, st.session_state.chat_history = rag_from_agent(
                prompt, titles, corpus, generator,
                chat_history=st.session_state.chat_history
            )
            status.update(label="✅ Fertig!", state="complete")
        
        st.markdown(answer)
        
        with st.expander("📜 Verwendete Rezepte"):
            for title in titles:
                wiki_url = f"https://www.kochwiki.org/wiki/{title.replace(' ', '_')}"
                st.markdown(f"- [{title}]({wiki_url})")

    st.session_state.messages_display.append({"role": "assistant", "content": answer})

# Chat 
if st.session_state.messages_display:
    if st.button("🗑️ Chat zurücksetzen"):
        st.session_state.chat_history = []
        st.session_state.messages_display = []
        st.rerun()
