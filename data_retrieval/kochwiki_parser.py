import re
import json
import os
import mwparserfromhell

def parse_rezept_template(wikitext):
    parsed = mwparserfromhell.parse(wikitext)
    templates = parsed.filter_templates()
    
    metadata = {}
    
    for template in templates:
        name = str(template.name).strip().lower()
        
        if name == "rezept":
            for param in template.params:
                key = str(param.name).strip()
                value = str(param.value).strip()
                
                if key and value:
                    metadata[key.lower()] = value
    
    return metadata


def extract_sections(wikitext):
    sections = {}
    current_section = "intro"
    current_content = []
    
    for line in wikitext.split("\n"):
        header_match = re.match(r'^(={2,3})\s*(.+?)\s*\1\s*$', line)
        
        if header_match:
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            
            current_section = header_match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)
    
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()
    
    return sections


def parse_zutaten(zutaten_text):
    zutaten = []
    
    for line in zutaten_text.split("\n"):
        line = line.strip()
        
        if not line.startswith("*"):
            continue
        
        line = line.lstrip("* ").strip()
        
        if not line:
            continue
        
        line_clean = re.sub(r'\[\[(?:Zutat:)?([^|\]]*\|)?([^\]]+)\]\]', r'\2', line)
        line_clean = re.sub(r"'{2,3}", "", line_clean)  
        line_clean = line_clean.strip()
        
        zutat_parsed = parse_single_zutat(line_clean)
        zutat_parsed["raw"] = line_clean
        zutaten.append(zutat_parsed)
    
    return zutaten


def parse_single_zutat(text):
    einheiten = [
        "kg", "g", "mg", "l", "ml", "cl", "dl",
        "EL", "TL", "Tasse", "Tassen", "Becher",
        "Prise", "Prisen", "Bund", "Stück", "Scheibe", "Scheiben",
        "Packung", "Päckchen", "Dose", "Dosen", "Glas", "Gläser",
        "Blatt", "Blätter", "Zweig", "Zweige", "Zehe", "Zehen",
        "Messerspitze", "Msp", "Handvoll",
    ]
    
    pattern = r'^(\d+[\d.,/–\-\s]*(?:\d+)?)\s*(' + '|'.join(einheiten) + r')?\s*(.+)$'
    match = re.match(pattern, text, re.IGNORECASE)
    
    if match:
        return {
            "menge": match.group(1).strip(),
            "einheit": (match.group(2) or "").strip(),
            "name": match.group(3).strip()
        }
    
    vague_pattern = r'^(etwas|wenig|viel|nach Bedarf|nach Geschmack|evtl\.?)\s+(.+)$'
    vague_match = re.match(vague_pattern, text, re.IGNORECASE)
    
    if vague_match:
        return {
            "menge": vague_match.group(1).strip(),
            "einheit": "",
            "name": vague_match.group(2).strip()
        }
    
    return {
        "menge": "",
        "einheit": "",
        "name": text
    }


def wikitext_to_plaintext(wikitext):
    parsed = mwparserfromhell.parse(wikitext)
    
    for template in parsed.filter_templates():
        try:
            parsed.remove(template)
        except ValueError:
            pass
    
    text = str(parsed)
    
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    
    text = re.sub(r'\[https?://[^\s\]]+ ([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    
    text = re.sub(r"'{2,5}", "", text)  # '''fett''', ''kursiv''
    
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+/?>', '', text)
    
    text = re.sub(r'^(={2,4})\s*(.+?)\s*\1\s*$', r'\2', text, flags=re.MULTILINE)

    text = re.sub(r'^\*+\s*', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'\[\[Kategorie:[^\]]+\]\]', '', text)

    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    text = re.sub(r'__[A-Z]+__', '', text)
    
    return text.strip()


def parse_recipe(wikitext, title="", categories=None):
    if categories is None:
        categories = []
    
    metadata = parse_rezept_template(wikitext)
    
    sections = extract_sections(wikitext)
    
    zutaten_raw = ""
    for key in sections:
        if "zutat" in key.lower() or "ingredient" in key.lower():
            zutaten_raw = sections[key]
            break
    
    zutaten = parse_zutaten(zutaten_raw) if zutaten_raw else []
    
    zubereitung = ""
    for key in sections:
        if "zubereitung" in key.lower() or "anleitung" in key.lower():
            zubereitung = sections[key]
            break
    
    plaintext = wikitext_to_plaintext(wikitext)
    
    kueche = [cat for cat in categories if "Küche" in cat or "küche" in cat]
    
    schwierigkeit = metadata.get("schwierigkeit", "").lower()
    
    result = {
        "title": title,
        "metadata": {
            "menge": metadata.get("menge", ""),
            "zeit": metadata.get("zeit", ""),
            "schwierigkeit": schwierigkeit,
            "alkohol": metadata.get("alkohol", ""),
            "bild": metadata.get("bild", ""),
        },
        "categories": categories,
        "kueche": kueche,
        "zutaten": zutaten,
        "zutaten_namen": [z["name"] for z in zutaten],  
        "zubereitung_raw": wikitext_to_plaintext(zubereitung) if zubereitung else "",
        "sections": {k: wikitext_to_plaintext(v) for k, v in sections.items()},
        "plaintext": plaintext,  
        "wikitext": wikitext,   
    }
    
    return result


def parse_zutat_template(wikitext):
    parsed = mwparserfromhell.parse(wikitext)
    templates = parsed.filter_templates()
    
    metadata = {}
    
    for template in templates:
        name = str(template.name).strip().lower()
        
        if name in ("zutat", "zutatübersicht"):
            metadata["template_type"] = name
            for param in template.params:
                key = str(param.name).strip()
                value = str(param.value).strip()
                if key and value:
                    metadata[key.lower()] = value
    
    return metadata


def parse_zutat(wikitext, title="", categories=None):
    if categories is None:
        categories = []
    
    clean_name = title.replace("Zutat:", "").strip()
    
    template_data = parse_zutat_template(wikitext)
    
    naehrwerte = {}
    naehrwert_keys = {
        "kcal": "kcal", "kj": "kj",
        "fett": "fett", "kohlenhydrate": "kohlenhydrate",
        "eiweiß": "eiweiss", "cholesterin": "cholesterin",
        "ballaststoffe": "ballaststoffe",
    }
    
    for raw_key, clean_key in naehrwert_keys.items():
        value = template_data.get(raw_key, "")
        num_match = re.search(r'([\d.,]+)', value)
        if num_match:
            try:
                naehrwerte[clean_key] = float(num_match.group(1).replace(",", "."))
            except ValueError:
                naehrwerte[clean_key] = None
        else:
            naehrwerte[clean_key] = None
    
    basismenge = template_data.get("basismenge", "")
    
    sections = extract_sections(wikitext)
    
    plaintext = wikitext_to_plaintext(wikitext)
    
    verwandte_zutaten = re.findall(r'\[\[Zutat:([^|\]]+)', wikitext)
    verwandte_zutaten = list(set(verwandte_zutaten)) 
    
    result = {
        "type": "zutat",
        "title": title,
        "name": clean_name,
        "bild": template_data.get("bild", ""),
        "basismenge": basismenge,
        "naehrwerte": naehrwerte,
        "has_naehrwerte": any(v is not None for v in naehrwerte.values()),
        "categories": categories,
        "verwandte_zutaten": verwandte_zutaten,
        "sections": {k: wikitext_to_plaintext(v) for k, v in sections.items()},
        "plaintext": plaintext,
        "wikitext": wikitext,
    }
    
    return result


def parse_all_recipes(input_file, output_file):
 
    print(f"Lade {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        raw_recipes = json.load(f)
    
    print(f"{len(raw_recipes)} recipes loaded. Start Parsing...")
    
    parsed_recipes = []
    errors = []
    
    for i, recipe in enumerate(raw_recipes):
        try:
            parsed = parse_recipe(
                wikitext=recipe["wikitext"],
                title=recipe["title"],
                categories=recipe.get("categories", [])
            )
            parsed_recipes.append(parsed)
            
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(raw_recipes)}] processed...")
                
        except Exception as e:
            errors.append({"title": recipe.get("title", "?"), "error": str(e)})
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_recipes, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ {len(parsed_recipes)} recipes successfully parsed")
    print(f"✗ {len(errors)} Errors")
    print(f"→ Saved to {output_file}")
    
    if errors:
        error_file = output_file.replace(".json", "_errors.json")
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"→ Fehler-Log in {error_file}")
    
    return parsed_recipes


def parse_all_zutaten(input_file, output_file):
    print(f"Load {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        raw_zutaten = json.load(f)
    
    print(f"{len(raw_zutaten)} ingredients loaded. Start Parsing...")
    
    parsed_zutaten = []
    errors = []
    
    for i, zutat in enumerate(raw_zutaten):
        try:
            parsed = parse_zutat(
                wikitext=zutat["wikitext"],
                title=zutat["title"],
                categories=zutat.get("categories", [])
            )
            parsed_zutaten.append(parsed)
            
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(raw_zutaten)}] processed...")
                
        except Exception as e:
            errors.append({"title": zutat.get("title", "?"), "error": str(e)})
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_zutaten, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ {len(parsed_zutaten)} ingredients successfully parsed")
    print(f"✗ {len(errors)} Errors")
    print(f"→ Saved to {output_file}")
    
    if errors:
        error_file = output_file.replace(".json", "_errors.json")
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"→ Error-Log in {error_file}")
    
    return parsed_zutaten