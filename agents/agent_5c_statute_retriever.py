"""
Agent 5C — Statute Retriever (3-Stage Weighted Process)

Input:  legal_facts.json, legal_signals.json, ingredients_ipc.json
Output: statute_candidates.json

UPGRADE: 3-Stage Process
1. Query Construction: fact types + signal descriptions + actor-object relationships
2. Category Filtering: ONLY keep sections where signal categories overlap with ingredient categories
3. Weighted Scoring: BM25 + ingredient term bonuses (+2 death, +2 force, +1 object, +1 actor)
4. Final Filter: Remove sections with 0 signal alignment/ingredient overlap
"""

import re
from pathlib import Path

from config import (
    get_case_dir,
    load_json,
    save_json,
    build_provenance,
    get_human_feedback_patterns,
    IPC_INGREDIENTS_PATH,
    GOOGLE_API_KEY,
    DEFAULT_MODEL,
    LLM_SAFETY_PROMPT,
)

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

def _build_query_components(legal_facts: list[dict], signals: list[dict]) -> tuple[list[str], list[str], list[str], set]:
    query_terms = []
    actors = []
    objects = []
    signal_categories = set()

    # Facts
    for fact in legal_facts:
        ft = fact.get("type", "")
        if ft and ft not in ("unknown_action", "no_action"):
            query_terms.append(ft.replace("_", " "))
        
        orig = fact.get("original_action", "")
        if orig and orig != "unknown_action":
            query_terms.append(orig)

        actor = fact.get("actor")
        target = fact.get("target")
        obj = fact.get("object")
        
        if actor: actors.append(actor)
        if target: actors.append(target)
        if obj: objects.append(obj)
            
        if actor and target:
            query_terms.append(f"{actor} against {target}")
        if actor and orig:
            query_terms.append(f"{actor} {orig}")

    # Signals & Categories
    for signal in signals:
        sig_name = signal.get("signal", "").lower()
        if sig_name:
            query_terms.append(sig_name.replace("_", " "))
            
        cats = signal.get("categories", [])
        if isinstance(cats, list):
            for c in cats:
                signal_categories.update(_tokenize(c))

    # Deduplicate terms
    seen = set()
    unique_terms = []
    for t in query_terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_terms.append(t)

    return unique_terms, list(set(actors)), list(set(objects)), signal_categories

def _load_ipc_sections() -> list[dict]:
    raw = load_json(IPC_INGREDIENTS_PATH)
    ipc_dict = raw.get("IPC", {})

    sections = []
    for skey, sdata in ipc_dict.items():
        if sdata.get("status", {}).get("repealed", False):
            continue
        
        # UPGRADE: Index sections even if ingredients list is empty (for Punishment sections)
        ingredients = sdata.get("ingredients", [])

        raw_cats = sdata.get("offence_category", [])
        tokenized_cats = set()
        for rc in raw_cats:
            tokenized_cats.update(_tokenize(rc))

        sections.append({
            "section_id": sdata.get("section_id", skey),
            "heading": sdata.get("heading", ""),
            "canonical_text": sdata.get("canonical_text", ""),
            "punishment": sdata.get("punishment", ""),
            "ingredients": ingredients,
            "offence_categories": tokenized_cats,
            "raw_categories": raw_cats,
        })
    return sections

def _build_section_document(section: dict) -> str:
    parts = [
        str(section.get("heading") or ""),
        str(section.get("canonical_text") or ""),
        str(section.get("punishment") or "")
    ]
    for ing in section.get("ingredients", []):
        if ing.get("text"): 
            parts.append(str(ing["text"]))
        norm = ing.get("normalized", {})
        if isinstance(norm, dict):
            for v in norm.values():
                if v: parts.append(str(v))
    for c in section.get("raw_categories", []):
        if c: parts.append(str(c))
    return " ".join(parts)

def _calculate_weighted_score(
    section: dict, 
    bm25_score: float, 
    actors: list[str], 
    objects: list[str], 
    signal_categories: set, 
    query_tokens: set
) -> float:
    score = bm25_score
    doc_text = _build_section_document(section).lower()
    heading_tokens = set(_tokenize(section["heading"]))

    # 1. Heading Boost (+10 or +20 for critical)
    # If any query token matches a heading token, give a big boost
    overlap = query_tokens & heading_tokens
    if overlap:
        score += 10.0
        # Extra boost for critical heading matches
        if any(t in overlap for t in ["murder", "homicide", "death", "killing"]):
            score += 15.0
    # 2. Semantic Bridge Boost (Massive Contextual RAG bridge)
    # Since query is neutral, we must explicitly bridge 'death' to 'murder/homicide'
    if "death" in signal_categories:
        if "murder" in doc_text or "homicide" in doc_text:
            score += 50.0
    if "property" in signal_categories:
        if any(t in doc_text for t in ["cheating", "fraud", "theft", "extortion", "robbery"]):
            score += 20.0

    # 3. Term Penalties / Bonuses
    # +10 -> ingredient mentions death-related terms (Primary objective)
    death_terms = {"death", "murder", "kill", "homicide", "die", "deceased", "killing"}
    if any(t in doc_text for t in death_terms):
        score += 10.0
        
    # +3 -> ingredient mentions force/violence
    force_terms = {"force", "violence", "hurt", "assault", "beat", "strike", "injury", "weapon", "struck"}
    if any(t in doc_text for t in force_terms):
        score += 3.0
        
    # +5 -> common legal action match (cheating specific)
    fraud_terms = {"cheating", "deception", "fraud", "dishonestly", "induce"}
    if any(t in doc_text for t in fraud_terms):
        score += 5.0
        
    # +1 -> weapon/object match
    for obj in objects:
        if obj.lower() in doc_text:
            score += 1.0
            
    # +1 -> actor-role alignment
    for actor in actors:
        if actor.lower() in doc_text:
            score += 1.0

    # 4. Domain Specificity Penalties (Anti-Hallucination for Statute Relevance)
    # Penalize highly specific circumstances if the input facts lack them entirely.
    query_string = " ".join(query_tokens).lower()
    
    if any(t in doc_text for t in ["dowry", "marriage", "husband", "wife"]):
        if not any(t in query_string for t in ["dowry", "marriage", "husband", "wife", "woman", "married"]):
            score -= 50.0
            
    if "miscarriage" in doc_text or "unborn" in doc_text:
        if not any(t in query_string for t in ["miscarriage", "pregnant", "unborn"]):
            score -= 50.0
            
    if any(t in doc_text for t in ["child ", "minor", "infant"]):
        if not any(t in query_string for t in ["child", "minor", "baby", "infant", "kid"]):
            score -= 50.0
            
    if "person other than" in doc_text or "whose death was intended" in doc_text:
        if "other than" not in query_string and "intended" not in query_string:
            score -= 50.0
            
    if any(t in doc_text for t in ["insane", "intoxicated", "idiot"]):
        if not any(t in query_string for t in ["insane", "drunk", "intoxicated", "mad"]):
            score -= 50.0
            
    if any(t in doc_text for t in ["theft", "robbery", "extortion", "dacoity", "mischief", "trespass", "misappropriation"]):
        if "property" not in signal_categories and not any(t in query_string for t in ["property", "extortion", "rob", "steal", "trespass", "stolen", "money"]):
            score -= 50.0
            
    if any(t in doc_text for t in ["public servant", "government"]):
        if not any(t in query_string for t in ["public servant", "government", "officer", "police"]):
            score -= 30.0
            
    if "counterfeit" in doc_text or "forgery" in doc_text:
        if "evidence" not in signal_categories and not any(t in query_string for t in ["fake", "forged", "counterfeit", "document"]):
            score -= 50.0

    return score

def run(tenant_id: str, case_id: str, top_k: int = 15) -> dict:
    case_dir = get_case_dir(tenant_id, case_id)
    legal_facts_data = load_json(case_dir / "legal_facts.json")
    signals_data = load_json(case_dir / "legal_signals.json")

    legal_facts = legal_facts_data.get("legal_facts", [])
    signals = signals_data.get("signals", [])

    sections = _load_ipc_sections()
    
    query_terms, actors, objects, signal_categories = _build_query_components(legal_facts, signals)

    # Step 2: Category Filtering (NEW)
    filtered_sections = []
    if signal_categories:
        for section in sections:
            doc_text = _build_section_document(section).lower()
            # Overlap exists?
            overlap = False
            for cat in signal_categories:
                if re.search(rf"\b{re.escape(cat)}\b", doc_text):
                    overlap = True
                    break
                    
            if overlap:
                filtered_sections.append(section)
    else:
        filtered_sections = sections

    if not filtered_sections or not query_terms:
        provenance = build_provenance(case_id, tenant_id, "agent_5c_statute_retriever", ["legal_facts.json", "legal_signals.json"])
        result = {**provenance, "candidates": [], "total_sections_indexed": len(sections)}
        save_json(case_dir / "statute_candidates.json", result)
        return result

    # BM25 Engine
    corpus_texts = [_build_section_document(s) for s in filtered_sections]
    tokenized_corpus = [_tokenize(doc) for doc in corpus_texts]
    query_text = " ".join(query_terms)
    tokenized_query = _tokenize(query_text)

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Step 3: Weighted Scoring & Matching
    scored = []
    query_token_set = set(tokenized_query)

    for idx, base_score in enumerate(scores):
        if base_score > 0:
            section = filtered_sections[idx]
            
            # Weighted bonuses
            final_score = _calculate_weighted_score(
                section, 
                float(base_score), 
                actors, 
                objects, 
                signal_categories, 
                query_token_set
            )

            matched_ingredients = []
            query_token_set = set(tokenized_query)
            for ing in section["ingredients"]:
                ing_text = ing.get("text", "")
                norm = ing.get("normalized", {})
                norm_text = " ".join(str(v) for v in norm.values() if isinstance(v, str)) if isinstance(norm, dict) else ""
                combined = f"{ing_text} {norm_text}".lower()
                ing_tokens = set(_tokenize(combined))

                if ing_tokens & query_token_set:
                    matched_ingredients.append({
                        "ingredient_id": ing.get("id", ""),
                        "text": ing_text,
                        "element_type": ing.get("element_type", "")
                    })

            # Handle punishment-only sections (e.g., 302, 304) which have no ingredients
            if not section["ingredients"]:
                doc_text = _build_section_document(section).lower()
                doc_tokens = set(_tokenize(doc_text))
                if doc_tokens & query_token_set:
                    matched_ingredients.append({
                        "ingredient_id": section["section_id"],
                        "text": section.get("canonical_text") or section.get("heading", ""),
                        "element_type": "punishment_text"
                    })

            # Step 4: Final Filter (Remove if no overlap)
            if matched_ingredients:
                scored.append({
                    "section": section["section_id"],
                    "heading": section.get("heading", ""),
                    "score": round(final_score, 4),
                    "matched_ingredients": matched_ingredients,
                    "offence_category": section.get("raw_categories", [])
                })

    scored.sort(key=lambda x: x["score"], reverse=True)
    candidates = scored[:top_k]
    
    # NEW STEP: LLM Filtering
    if GOOGLE_API_KEY and candidates:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=GOOGLE_API_KEY)
            
            facts_str = json.dumps(legal_facts, indent=1)
            cands_str = json.dumps(
                [{
                    "section": c["section"], 
                    "heading": c["heading"], 
                    "matched_ingredients": [i["text"] for i in c.get("matched_ingredients", [])]
                } for c in candidates], 
                indent=1
            )
            
            prompt = f"""You are a legal statute retrieval system.

Your task is to retrieve ONLY those IPC sections whose core ingredients have a DIRECT factual connection to the events.

STRICT FILTERING RULES:
1. Include a section ONLY if at least one core ingredient is explicitly supported by facts.
2. Exclude sections requiring conditions not present in facts (e.g., dowry, abetment, private defence, life-convict status).
3. Prioritize sections involving:
   - Causing death
   - Causing bodily injury
   - Intention or knowledge
4. Avoid over-retrieval. Maximum 5–7 highly relevant sections.

OUTPUT:
Return ranked sections with justification mapping facts → ingredients. This removes noise without hardcoding any section numbers.

FACTS:
{facts_str}

CANDIDATE SECTIONS FROM SEARCH:
{cands_str}

HUMAN FEEDBACK (apply these learned preferences to your retrieval strategy if relevant):
{get_human_feedback_patterns(tenant_id) or "None"}

Output ONLY a JSON array of objects with keys: "section", "justification". Do NOT output markdown formatting like ```json."""
            
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    system_instruction=LLM_SAFETY_PROMPT,
                    response_mime_type="application/json",
                ),
            )
            
            try:
                llm_output = json.loads(response.text)
                allowed_sections = {item.get("section") for item in llm_output if isinstance(item, dict)}
                if allowed_sections:
                    # Filter candidates to only those approved by the LLM, preserving original data structure
                    filtered_candidates = [c for c in candidates if c["section"] in allowed_sections]
                    # Update validations with justifications if provided
                    for c in filtered_candidates:
                        for llm_c in llm_output:
                            if llm_c.get("section") == c["section"]:
                                c["retrieval_justification"] = llm_c.get("justification", "")
                    
                    if filtered_candidates:
                        candidates = filtered_candidates
            except json.JSONDecodeError:
                pass
        except Exception as e:
            print(f"LLM filtering failed in Agent 5C: {e}")

    provenance = build_provenance(case_id, tenant_id, "agent_5c_statute_retriever", ["legal_facts.json", "legal_signals.json"])

    result = {
        **provenance,
        "query_terms": query_terms,
        "total_sections_indexed": len(sections),
        "candidates": candidates,
    }

    save_json(case_dir / "statute_candidates.json", result)
    return result
