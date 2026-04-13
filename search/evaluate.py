from retrieval import *

evaluation = {
    "Kartoffelsuppe": ["Leipziger Kartoffelsuppe", "Lauch-Kartoffelsuppe", "Kartoffelsuppe I", "Feine Kartoffelsuppe", "Einfache Kartoffelsuppe"],
    "Gulasch": ["Würstchengulasch mit Paprika", "Wurstgulasch", "Exotisches Gulasch nach Großmutters Art (Rimpf)", "Wildschweingulasch mit Pilzen", "Alt-Wiener Kalbsrahmgulasch"],
    "Schokokuchen": ["Erdnuss-Schokokuchen", "Schokokuchen", "Sprudel-Schokokuchen", "Schneller Schokoladenkuchen", "Türkischer Schokoladenkuchen"],
    "Weihnachtsplätzchen": ["Walisische Weihnachtsplätzchen", "Detmolder Weihnachtsplätzchen", "Ausstechplätzchen", "Belgische Plätzchen", "Buchstabenplätzchen"],
    "Brot mit Hefe": ["Sonnenblumenbrot", "Kraftbrot", "Vollkornbauernbrot", "Olivenbrot", "No-Knead-Bread"],
    "japanische Gerichte": ["Miso-Suppe mit Tofu", "Japanische Nudelsuppe", "Miso Shiru", "Japanische Soba-Nudeln", "Japanisches Sojahuhn"],
    "Curry": ["Tomaten-Linsen-Curry", "Thailändisches Gemüsecurry", "Veganes Blumenkohlcurry", "Thai-Lychee-Curry", "Curry-Dip"],
    "schnelles Mittagessen mit Nudeln": ["Zitronennudeln", "Tagliatelle in Schinken-Sahnesauce", "Tagliatelle mit Lachs-Sherrysauce", "Zha-Jiang Nudeln nach Peking-Art", "Nudeln mit Spinat und Lachs"],
    "französische Soße": ["Vinaigrette mit roter Paprikaschote", "Vinaigrette mit Feta", "Erdbeervinaigrette", "Béchamelsauce", "Sauce hollandaise"],
    "Rezepte mit Kartoffeln, Zwiebeln und Speck": ["Bratkartoffeln", "Bauernfrühstück", "Himmel und Erde", "Kartoffelpfanne mit Käse", "Döppekoche"],
}

def evaluate(search_fn, name, k=10):
    total_hits = 0
    queries_with_hit = 0
    
    for query, expected in evaluation.items():
        results = search_fn(query, n=k)
        retrieved_titles = [title for _, title, *_ in results]
        hits = sum(1 for doc in expected if doc in retrieved_titles)
        total_hits += hits
        if hits > 0:
            queries_with_hit += 1
    
    print(f"{name}: {total_hits}/50 hits, {queries_with_hit}/10 queries with ≥1 hit, Recall: {total_hits/50:.0%}")

def detailed_evaluate(search_fn, name, k=10):
    print(f"\n===== DETAILED EVALUATION: {name} =====")
    total_hits = 0
    queries_with_hit = 0
    
    for query, expected in evaluation.items():
        results = search_fn(query, n=k)
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
    
    print(f"\nStrict Recall@{k}: {total_hits}/50 ({total_hits/50:.0%})")
    print(f"Queries with ≥1 hit: {queries_with_hit}/10")

def r_precision(search_fn, name, r=5):
    print(f"\n===== R-PRECISION EVALUATION: {name} =====")
    total_rp = 0
    
    for query, expected in evaluation.items():
        results = search_fn(query, n=r)
        retrieved_titles = [title for _, title, *_ in results]
        hits = sum(1 for doc in expected if doc in retrieved_titles)
        total_rp += hits / r
    
    avg_rp = total_rp / len(evaluation)
    print(f"\n{name}: R-Precision@{r} = {avg_rp:.3f}")

if __name__ == "__main__":
    corpus, inverted_index, bm25_data = load_indices()
    faiss_index, faiss_doc_ids, model = load_faiss()
    graph, synonyms = load_graph()

    evaluate(lambda q, n=10: bm25_search(q, corpus, bm25_data, n=n, synonyms=synonyms, doc_type="recipe"), "BM25")
    evaluate(lambda q, n=10: faiss_search(q, corpus, faiss_index, faiss_doc_ids, model, n=n), "FAISS")
    evaluate(lambda q, n=10: hybrid_search(q, corpus, bm25_data, faiss_index, faiss_doc_ids, model, graph=graph, synonyms=synonyms, doc_type="recipe", n=n), "Hybrid")
    detailed_evaluate(lambda q, n=10: hybrid_search(q, corpus, bm25_data, faiss_index, faiss_doc_ids, model, graph=graph, synonyms=synonyms, doc_type="recipe", n=n), "Hybrid")