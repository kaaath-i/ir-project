from rag import *
from retrieval import load_indices, load_faiss, load_graph

corpus, inverted_index, bm25_data = load_indices()
faiss_index, faiss_doc_ids, model = load_faiss()
graph, synonyms = load_graph()
generator = load_rag()

rag_evaluation = [
    {
        "query": "Ich suche ein schnelles Pastagericht",
        "expected_keywords": ["Nudeln", "Pasta", "Minuten"]
    },
    {
        "query": "Was kann ich mit Kartoffeln und Speck machen?",
        "expected_keywords": ["Kartoffel", "Speck", "Bratkartoffeln"]
    }
]

for item in rag_evaluation:
    answer, sources, _ = rag_search(
        item["query"], corpus, bm25_data, faiss_index, faiss_doc_ids, model, generator, chat_history=None
    )
    hits = sum(1 for kw in item["expected_keywords"] if kw.lower() in answer.lower())
    print(f"Query: {item['query']}")
    print(f"Keyword hits: {hits}/{len(item['expected_keywords'])}")
    print(f"Answer: {answer[:200]}...\n")