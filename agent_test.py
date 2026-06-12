import os
from dotenv import load_dotenv
load_dotenv(override=True)

from search.retrieval import load_indices, load_faiss, load_graph
from search.agent import load_agent, agent_search
from search.rag import load_rag, rag_from_agent

corpus, inverted_index, bm25_data = load_indices()
faiss_index, faiss_doc_ids, model = load_faiss()
graph, synonyms = load_graph()

from search.agent import init_agent_resources
init_agent_resources(corpus, bm25_data, faiss_index, faiss_doc_ids, model, graph, synonyms)

agent = load_agent()
generator = load_rag()

query = "Ich suche ein vegetarisches Gulasch"
titles = agent_search(query, agent)
print(f"Agent gefundene Titel: {titles}")

answer, _ = rag_from_agent(query, titles, corpus, generator)
print(f"\nAntwort:\n{answer}")