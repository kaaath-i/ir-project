from groq import Groq
from search.retrieval import hybrid_search
#from dotenv import load_dotenv
#load_dotenv(override=True)
import os

SYSTEM_PROMPT = """Du bist RAGatouille, ein cooler Kochassistent basierend auf KochWiki.
WICHTIG: Du duzt den User IMMER. Niemals "Sie", immer "du/dich/dir".
Schreib locker und freundlich, wie ein Kumpel der gut kochen kann — nicht zu förmlich, nicht zu steif.
Wenn du nach Rezepten suchst, gib nur die exakten Titel zurück, kommagetrennt.
Wenn du kein passendes Rezept findest, sag es ehrlich und schlage Alternativen vor.
Erfinde keine Zutaten oder Eigenschaften die nicht im Rezept stehen.
Antworte immer auf Deutsch."""

def load_rag():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return client

def rag_from_agent(query, titles, corpus, generator, chat_history=None):
    documents = []
    for doc_id, doc in corpus.items():
        for title in titles:
            if title.lower() in doc["title"].lower() or doc["title"].lower() in title.lower():
                documents.append(f"Titel: {doc['title']}\n{doc['text'][:500]}")
                break

    if not documents:
        return "Hmm, da hab ich leider gar nichts gefunden. Versuch mal eine andere Suchanfrage!", []

    context = "\n---\n".join(documents)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if chat_history:
        messages.extend(chat_history[-4:])
    
    messages.append({"role": "user", "content": f"""Gefundene Rezepte:
{context}

Frage: {query}

Wenn die Rezepte nicht exakt passen, sag das kurz und schlage sie trotzdem als Alternativen vor.
Verlink die Rezepte im Format: [Rezepttitel](https://www.kochwiki.org/wiki/Rezepttitel_mit_Unterstrichen)"""})

    response = generator.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=512
    )
    
    answer = response.choices[0].message.content

    import re
    answer = re.sub(r'\[([^\]]+)\]', lambda m: '[' + m.group(1).replace('_', ' ') + ']', answer)

    updated_history = list(chat_history) if chat_history else []
    updated_history.append({"role": "user", "content": messages[-1]["content"]})
    updated_history.append({"role": "assistant", "content": answer})

    return answer, updated_history