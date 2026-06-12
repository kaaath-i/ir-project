import os
from smolagents import tool, ToolCallingAgent, InferenceClientModel
from search.retrieval import hybrid_search, load_indices, load_faiss, load_graph

_corpus = None
_bm25_data = None
_faiss_index = None
_faiss_doc_ids = None
_model = None
_graph = None
_synonyms = None

def init_agent_resources(corpus, bm25_data, faiss_index, faiss_doc_ids, model, graph, synonyms):
    global _corpus, _bm25_data, _faiss_index, _faiss_doc_ids, _model, _graph, _synonyms
    _corpus = corpus
    _bm25_data = bm25_data
    _faiss_index = faiss_index
    _faiss_doc_ids = faiss_doc_ids
    _model = model
    _graph = graph
    _synonyms = synonyms

def build_context(results, corpus):
    context = ""
    for doc_id, title, score in results:
        context += f"---\nTitel: {title}\n{corpus[doc_id]['text'][:300]}\n---\n"
    return context

@tool
def recipe_search(query: str) -> str:
    """Search for recipes in the KochWiki database using hybrid search.
    Use this tool to find recipes based on ingredients, dish names, or cooking styles.
    Args:
        query: The search query for recipes, e.g. 'vegetarisches Gulasch' or 'schnelles Pastagericht'
    """
    results = hybrid_search(
        query, _corpus, _bm25_data, _faiss_index, _faiss_doc_ids, _model,
        graph=_graph, synonyms=_synonyms, n=3
    )
    return build_context(results, _corpus)

def load_agent():
    agent = ToolCallingAgent(
        tools=[recipe_search],
        model=InferenceClientModel(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            token=os.environ.get("GROQ_API_KEY"),
            provider="groq"
        )
    )
    return agent


def agent_search(query, agent):
    response = agent.run(
        f"Suche nach Rezepten für: '{query}'. "
        f"Antworte NUR mit den exakten Rezepttiteln, kommagetrennt. "
        f"Beispiel: 'Kartoffelsuppe, Lauch-Kartoffelsuppe'. "
        f"Keine Erklärung, kein anderer Text, nur Titel."
    )
    raw = str(response).strip()
    print(f"Agent raw response: {repr(raw)}")
    
    if "keine" in raw.lower() or "gefunden" in raw.lower() or len(raw) > 200:
        print("Agent Fallback → hybrid search")
        results = hybrid_search(query, _corpus, _bm25_data, _faiss_index, _faiss_doc_ids, _model,
                               graph=_graph, synonyms=_synonyms, n=3)
        titles = [title for _, title, _ in results]
    else:
        titles = [t.strip() for t in raw.split(",")]
    
    print(f"Parsed titles: {titles}")
    return titles