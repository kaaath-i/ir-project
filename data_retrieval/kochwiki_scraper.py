import requests
import json
import time
import os
from datetime import datetime

API_URL = "https://www.kochwiki.org/api.php"
OUTPUT_DIR = "kochwiki_data"
DELAY = 0.5

def get_all_pages(namespace=0, limit=50):
    all_pages = []
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": namespace,
        "aplimit": limit,
        "format": "json"
    }
    
    while True:
        print(f"  Fetching pages... (so far: {len(all_pages)})")
        response = requests.get(API_URL, params=params)
        data = response.json()
        
        pages = data["query"]["allpages"]
        all_pages.extend(pages)
        
        if "continue" in data:
            params["apcontinue"] = data["continue"]["apcontinue"]
        else:
            break
        
        time.sleep(DELAY)
    
    print(f"  → {len(all_pages)} pages found in namespace {namespace}")
    return all_pages

def get_page_content(title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions|categories",
        "rvprop": "content|timestamp",
        "rvslots": "main",
        "cllimit": "50",
        "format": "json"
    }
    
    response = requests.get(API_URL, params=params)
    data = response.json()
    
    pages = data["query"]["pages"]
    page_id = list(pages.keys())[0]
    
    if page_id == "-1":
        return None
    
    page = pages[page_id]
    
    result = {
        "page_id": int(page_id),
        "title": page["title"],
        "wikitext": page["revisions"][0]["slots"]["main"]["*"],
        "timestamp": page["revisions"][0]["timestamp"],
        "categories": [cat["title"].replace("category:", "") 
                       for cat in page.get("categories", [])]
    }
    
    return result


def get_page_plaintext(title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,  
        "exsectionformat": "plain",
        "format": "json"
    }
    
    response = requests.get(API_URL, params=params)
    data = response.json()
    
    pages = data["query"]["pages"]
    page_id = list(pages.keys())[0]
    
    if page_id == "-1":
        return None
    
    return data["query"]["pages"][page_id].get("extract", "")


def get_categories():
    all_categories = []
    params = {
        "action": "query",
        "list": "allcategories",
        "aclimit": 500,
        "format": "json"
    }
    
    while True:
        response = requests.get(API_URL, params=params)
        data = response.json()
        
        categories = data["query"]["allcategories"]
        all_categories.extend([cat["*"] for cat in categories])
        
        if "continue" in data:
            params["accontinue"] = data["continue"]["accontinue"]
        else:
            break
        
        time.sleep(DELAY)
    
    print(f"→ {len(all_categories)} categories found")
    return all_categories


def get_pages_in_category(category, namespace=0):
    all_pages = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Kategorie:{category}",
        "cmnamespace": namespace,
        "cmlimit": 500,
        "format": "json"
    }
    
    while True:
        response = requests.get(API_URL, params=params)
        data = response.json()
        
        members = data["query"]["categorymembers"]
        all_pages.extend(members)
        
        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
        else:
            break
        
        time.sleep(DELAY)
    
    print(f"→ {len(all_pages)} pages in category '{category}'")
    return all_pages


def scrape_all(namespaces=None):
    if namespaces is None:
        namespaces = {
            0: "rezepte",         
            100: "zutaten",       
            102: "zubereitungen", 
        }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for ns_id, ns_name in namespaces.items():
        print(f"\n{'='*50}")
        print(f"Scraping Namespace: {ns_name} (ID: {ns_id})")
        print(f"{'='*50}")
        
        pages = get_all_pages(namespace=ns_id)
        
        all_content = []
        for i, page in enumerate(pages):
            title = page["title"]
            print(f"  [{i+1}/{len(pages)}] {title}")
            
            content = get_page_content(title)
            if content:
                all_content.append(content)
            
            time.sleep(DELAY)
        
        output_file = os.path.join(OUTPUT_DIR, f"{ns_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_content, f, ensure_ascii=False, indent=2)
        
        print(f"\n→ {len(all_content)} pages saved to {output_file}")
    
    print(f"\n{'='*50}")
    print("Scraping Categories...")
    print(f"{'='*50}")
    categories = get_categories()
    
    cat_file = os.path.join(OUTPUT_DIR, "kategorien.json")
    with open(cat_file, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    
    print(f"\nDone! All data saved in '{OUTPUT_DIR}/'")
