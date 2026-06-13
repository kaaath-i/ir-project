import os
from smolagents import tool, CodeAgent, InferenceClientModel
from dotenv import load_dotenv
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
    try:
        results = hybrid_search(
            query,
            _corpus,
            _bm25_data,
            _faiss_index,
            _faiss_doc_ids,
            _model,
            graph=_graph,
            synonyms=_synonyms,
            n=3
        )

        if not results:
            return "NO_RESULTS"

        return build_context(results, _corpus)

    except Exception as e:
        return f"ERROR: {str(e)}"

def load_agent():
    agent = CodeAgent(
        tools=[recipe_search],
        model=InferenceClientModel(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            token=os.environ.get("GROQ_API_KEY"),
            provider="groq",
            timeout=20
            ),
        max_steps=3
    )
    return agent


def agent_search(query, agent):
    try:
        prompt = (
            f"Finde passende Rezepte für: {query}. "
            f"Gib die besten Rezeptnamen zurück. "
            f"Wenn keine gefunden werden, sag 'keine gefunden'."
        )

        response = agent.run(prompt)

        if not response:
            return []

        raw = str(response).strip()
        print(f"Agent raw response: {raw}")

        if "keine" in raw.lower() or "no_results" in raw.lower() or "error" in raw.lower():
            results = hybrid_search(
                query,
                _corpus,
                _bm25_data,
                _faiss_index,
                _faiss_doc_ids,
                _model,
                graph=_graph,
                synonyms=_synonyms,
                n=3
            )
            return [title for _, title, _ in results]

        titles = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]

        if len(titles) == 0:
            results = hybrid_search(
                query,
                _corpus,
                _bm25_data,
                _faiss_index,
                _faiss_doc_ids,
                _model,
                graph=_graph,
                synonyms=_synonyms,
                n=3
            )
            return [title for _, title, _ in results]

        return titles

    except Exception as e:
        print(f"Agent error: {e}")

        results = hybrid_search(
            query,
            _corpus,
            _bm25_data,
            _faiss_index,
            _faiss_doc_ids,
            _model,
            graph=_graph,
            synonyms=_synonyms,
            n=3
        )
        return [title for _, title, _ in results]