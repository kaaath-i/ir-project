from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from search.retrieval import hybrid_search
#from dotenv import load_dotenv
#load_dotenv(override=True)
import os
from haystack.utils import Secret

SYSTEM_PROMPT = """Du bist RAGatouille, ein cooler Kochassistent basierend auf KochWiki.
Du duzt den User immer.
Schreib locker und freundlich, nicht zu förmlich.
Wenn die gefundenen Rezepte nicht zur Anfrage passen, sag das ehrlich und schlage eine Alternative vor.
Erfinde keine Eigenschaften die nicht in den Rezepten stehen.
Antworte immer auf Deutsch."""


def load_rag():
    generator = HuggingFaceAPIChatGenerator(
        api_type="serverless_inference_api",
        api_params={"model": "meta-llama/Llama-3.3-70B-Instruct"},
        generation_kwargs={"max_tokens": 512},
        token=Secret.from_env_var("HF_TOKEN")
    )
    generator.warm_up()
    return generator

def load_rag():
    generator = HuggingFaceAPIChatGenerator(
        api_type="serverless_inference_api",
        api_params={"model": "meta-llama/Llama-3.3-70B-Instruct"},
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

def rag_from_agent(query, titles, corpus, generator, chat_history=None):
    documents = []
    for doc_id, doc in corpus.items():
        if doc["title"] in titles:
            documents.append(f"Titel: {doc['title']}\n{doc['text'][:500]}")
    
    if not documents:
        return "Leider konnte ich keine passenden Rezepte finden.", []

    context = "\n---\n".join(documents)
    
    messages = [ChatMessage.from_system(SYSTEM_PROMPT)]
    
    if chat_history:
        messages.extend(chat_history)
    
    user_message = f"""Gefundene Rezepte:
{context}

Frage: {query}

Gib eine hilfreiche Antwort mit Rezeptname, kurzer Beschreibung und Link im Format:
🔗 [Rezeptname](https://www.kochwiki.org/wiki/{{Rezeptname.replace(' ', '_')}})"""
    
    messages.append(ChatMessage.from_user(user_message))
    response = generator.run(messages)
    answer = response["replies"][0].text
    
    updated_history = list(chat_history) if chat_history else []
    updated_history.append(ChatMessage.from_user(user_message))
    updated_history.append(ChatMessage.from_assistant(answer))
    
    return answer, updated_history