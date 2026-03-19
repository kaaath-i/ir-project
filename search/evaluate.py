from retrieval import *

evaluation = {
    "Kartoffelsuppe": ["Vegetarische Kartoffelsuppe", "Vegane Kartoffelsuppe", "Leipziger Kartoffelsuppe", "Leas Kartoffelsuppe", "Lauch-Kartoffelsuppe"],
    "Gulasch": ["Würstchengulasch mit Paprika", "Wurstgulasch", "Winterlicher Krautgulasch", "Wildschweingulasch mit Pilzen und Senfsauce", "Wildschweingulasch mit Pilzen"],
    "Schokoladentorte": ["Himmlische Schokoladentorte", "Schokoladentorte, himmlisch", "Schokoladentorte mit Mascarpone-Minze-Creme", "Maroni-Schokoladentorte", "Brigittes Schokoladentorte"],
    "Weihnachtsplätzchen": ["Walisische Weihnachtsplätzchen", "Waliser Weihnachtsplätzchen", "Detmolder Weihnachtsplätzchen", "Zimtsterne mit Mandeln", "Plätzchen zum Ausstechen"],
    "Brot mit Sauerteig": ["Sauerteigbrot ohne Hefe aus dem Backautomaten", "Sauerteigbrot (hefefrei) aus dem Backautomaten", "Römisches Sauerteigbrot", "Roggenbrot auf Sauerteigbasis", "Kartoffelbrot mit Sauerteig"],
    "japanische Suppe": ["Misosuppe mit Tofu", "Miso-Suppe mit Tofu", "Ramen (Japanische Nudelsuppe)", "Japanische Nudelsuppe", "Miso Shiru"],
    "Curry": ["Tomaten-Linsen-Curry", "Thailändisches Gemüsecurry", "Veganes Blumenkohlcurry", "Thai-Lychee-Curry", "Zanderfilet mit Apfel-Gemüse-Curry"],
    "schnelles Mittagessen mit Nudeln": ["Zitronennudeln", "Studenten-Nudelauflauf", "Tagliatelle in Schinken-Sahnesauce", "Tagliatelle mit Lachs-Sherrysauce", "Zha-Jiang Nudeln nach Peking-Art"],
    "französische Soße": ["Sauce ravigote", "Vinaigrette mit roter Paprikaschote", "Vinaigrette mit Feta", "Erdbeervinaigrette", "Walnusssoße"],
    "Rezepte mit Kartoffeln, Zwiebeln und Speck": ["Bratkartoffeln", "Bauernfrühstück", "Himmel und Erde", "Kartoffelpfanne mit Käse", "Döppekoche"],
}

def evaluate(search_fn, name):
    total_hits = 0
    queries_with_hit = 0
    
    for query, expected in evaluation.items():
        results = search_fn(query)
        retrieved_titles = [title for _, title, *_ in results]
        hits = sum(1 for doc in expected if doc in retrieved_titles)
        total_hits += hits
        if hits > 0:
            queries_with_hit += 1
    
    print(f"{name}: {total_hits}/50 hits, {queries_with_hit}/10 queries with ≥1 hit, Recall: {total_hits/50:.0%}")

def detailed_evaluate(search_fn, name):
    print(f"\n===== DETAILED EVALUATION: {name} =====")
    
    total_hits = 0
    queries_with_hit = 0
    
    for query, expected in evaluation.items():
        results = search_fn(query)
        retrieved_titles = [title for _, title, *_ in results]
        
        hits = sum(1 for doc in expected if doc in retrieved_titles)
        total_hits += hits
        if hits > 0:
            queries_with_hit += 1
        
        print(f"\nQuery: '{query}'")
        print(f"  Hits: {hits}/5")
        for title in retrieved_titles:
            marker = "✓" if title in expected else "✗"
            print(f"  {marker} {title}")
    
    print(f"\nStrict Recall: {total_hits}/50 ({total_hits/50:.0%})")
    print(f"Queries with ≥1 hit: {queries_with_hit}/10")

if __name__ == "__main__":
    corpus, inverted_index, bm25_data = load_indices()
    faiss_index, faiss_doc_ids, model = load_faiss()
    graph, synonyms = load_graph()

    evaluate(lambda q: bm25_search(q, corpus, bm25_data, synonyms=synonyms, doc_type="recipe"), "BM25")
    evaluate(lambda q: faiss_search(q, corpus, faiss_index, faiss_doc_ids, model, doc_type="recipe"), "FAISS")
    evaluate(lambda q: graph_search(q.split(), graph, corpus, n=5), "Graph")
    evaluate(lambda q: hybrid_search(q, corpus, bm25_data, faiss_index, faiss_doc_ids, model, graph=graph, synonyms=synonyms, doc_type="recipe"), "Hybrid")
    detailed_evaluate(lambda q: hybrid_search(q, corpus, bm25_data, faiss_index, faiss_doc_ids, model, graph=graph, synonyms=synonyms, doc_type="recipe"), "Hybrid")