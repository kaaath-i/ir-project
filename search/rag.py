from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from search.retrieval import hybrid_search
#from dotenv import load_dotenv
#load_dotenv(override=True)
import os
from haystack.utils import Secret

SYSTEM_PROMPT = """Du bist ein hilfreicher Kochassistent basierend auf KochWiki. 
Du hilfst Nutzern beim Finden und Verstehen von Rezepten.
Beantworte Fragen basierend auf den gefundenen Rezepten.
Wenn du gebeten wirst ein anderes Rezept zu zeigen, schlage eine Alternative vor.
Antworte immer auf Deutsch."""


def load_rag():
    generator = HuggingFaceAPIChatGenerator(
        api_type="serverless_inference_api",
        api_params={"model": "meta-llama/Meta-Llama-3-8B-Instruct"},
        generation_kwargs={"max_tokens": 512},
        token=Secret.from_env_var("HF_TOKEN")
    )
    generator.warm_up()
    return generator

def load_rag():
    generator = HuggingFaceAPIChatGenerator(
        api_type="serverless_inference_api",
        api_params={"model": "meta-llama/Meta-Llama-3-8B-Instruct"},
        generation_kwargs={"max_tokens": 512},
        token=Secret.from_env_var("HF_TOKEN")
    )
    generator.warm_up()
    return generator

def build_context(results, corpus):
    context = ""
    for doc_id, title, score in results:
        context += f"---\n{corpus[doc_id]['text']}\n---\n"
    return context

def rag_search(query, corpus, bm25_data, faiss_index, faiss_doc_ids, model, generator, graph=None, synonyms=None, n=3, chat_history=None):
    results = hybrid_search(
        query, corpus, bm25_data, faiss_index, faiss_doc_ids, model,
        graph=graph, synonyms=synonyms, n=n
    )
    
    context = build_context(results, corpus)
    
    messages = [ChatMessage.from_system(SYSTEM_PROMPT)]
    
    if chat_history:
        messages.extend(chat_history)
    
    user_message = f"""Gefundene Rezepte:
{context}

Frage: {query}"""
    
    messages.append(ChatMessage.from_user(user_message))
    
    response = generator.run(messages)
    answer = response["replies"][0].text
    
    updated_history = list(chat_history) if chat_history else []
    updated_history.append(ChatMessage.from_user(user_message))
    updated_history.append(ChatMessage.from_assistant(answer))
    
    return answer, results, updated_history