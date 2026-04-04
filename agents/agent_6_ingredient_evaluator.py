"""
Agent 6 — Ingredient Evaluator

Input:  event_graph.json, legal_facts.json, legal_signals.json,
        statute_candidates.json, ingredients_ipc.json
Output: ingredient_report.json

6-Step deterministic pipeline:
  1. Consistency check  — detect contradictions between signals/facts/events
  2. Deterministic matching — events > facts > signals priority
  3. LLM validation    — second pass, merge with MORE CONSERVATIVE rule
  4. Core ingredient rule — if actus_reus or mens_rea NOT_SATISFIED → score=0
  5. Scoring            — (S + 0.5*P) / total
  6. Output             — per-statute ingredient evaluations

SAFETY CONTRACT:
  - NO invented facts
  - NO inference of intent unless explicitly stated
  - NO assumption of causation
  - Signals alone NEVER satisfy core ingredients
  - Prefer underconfidence over hallucination
"""

import json
import re
import logging
from pathlib import Path

from config import (
    get_case_dir,
    load_json,
    save_json,
    build_provenance,
    IPC_INGREDIENTS_PATH,
    IPC_INGREDIENTS_SIMPLE_PATH,
    CORE_INGREDIENT_TYPES,
    GOOGLE_API_KEY,
    LLM_SAFETY_PROMPT,
    DEFAULT_MODEL,
)

logger = logging.getLogger("lexplain.agent6")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1 — Consistency Check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _consistency_check(
    events: list[dict],
    legal_facts: list[dict],
    signals: list[dict],
) -> list[dict]:
    """Detect contradictions between evidence layers."""
    flags = []
    
    # Build lookup sets
    event_actions = set()
    event_objects = set()
    for evt in events:
        action = (evt.get("action") or "").lower()
        if action:
            event_actions.add(action)
        obj = (evt.get("object") or "").lower()
        if obj:
            event_objects.add(obj)
    
    fact_types = {f.get("type", "") for f in legal_facts}
    signal_names = {s.get("signal", "") for s in signals}
    
    # Check: signal says death but no event/fact shows death
    death_signals = {"death_occurred", "death_by_violence", "death_following_force"}
    death_facts = {"causing_death"}
    death_actions = {"died", "killed", "murdered", "death"}
    
    if death_signals & signal_names:
        has_death_event = bool(death_actions & event_actions)
        has_death_fact = bool(death_facts & fact_types)
        if not has_death_event and not has_death_fact:
            flags.append({
                "type": "signal_event_mismatch",
                "detail": "Signal indicates death but no event or fact confirms death occurred",
                "signal": list(death_signals & signal_names),
            })
    
    # Check: signal says weapon but no event/fact supports it
    weapon_signals = {"weapon_recovered", "weapon_present"}
    if weapon_signals & signal_names:
        weapon_terms = {"knife", "gun", "weapon", "sword", "pistol", "rod", "axe", "blade", "stick"}
        has_weapon_event = bool(weapon_terms & event_objects) or any(
            t in " ".join(event_objects) for t in weapon_terms
        )
        if not has_weapon_event:
            flags.append({
                "type": "signal_event_mismatch",
                "detail": "Signal indicates weapon but no event or object confirms weapon presence",
                "signal": list(weapon_signals & signal_names),
            })
    
    # Check: fact says causing_death but no death event
    if "causing_death" in fact_types:
        has_death_event = bool(death_actions & event_actions)
        if not has_death_event:
            flags.append({
                "type": "fact_event_mismatch",
                "detail": "Legal fact 'causing_death' present but no event shows death",
            })
    
    return flags


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2 — Deterministic Matching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Mapping from ingredient text keywords to evidence types
INGREDIENT_EVENT_KEYWORDS = {
    # Death / causation
    "causing death": {"actions": {"died", "killed", "murdered", "death", "succumbed", "perished"},
                      "category": "actus_reus"},
    "death": {"actions": {"died", "killed", "murdered", "death", "succumbed"},
              "category": "actus_reus"},
    # Force / violence
    "bodily injury": {"actions": {"attacked", "stabbed", "hit", "beat", "struck", "assaulted", "punched", "kicked"},
                      "category": "actus_reus"},
    "bodily pain": {"actions": {"attacked", "stabbed", "hit", "beat", "struck", "assaulted", "punched", "kicked"},
                    "category": "actus_reus"},
    "force": {"actions": {"attacked", "stabbed", "hit", "beat", "struck", "assaulted", "pushed", "punched"},
              "category": "actus_reus"},
    "hurt": {"actions": {"attacked", "stabbed", "hit", "beat", "struck", "assaulted", "wounded"},
             "category": "actus_reus"},
    # Intent / knowledge (mens rea)
    "intention of causing death": {"actions": set(), "category": "mens_rea"},
    "intention": {"actions": set(), "category": "mens_rea"},
    "knowledge": {"actions": set(), "category": "mens_rea"},
    "dishonest": {"actions": set(), "category": "mens_rea"},
    "fraudulent": {"actions": set(), "category": "mens_rea"},
    # Weapon
    "instrument": {"actions": {"knife", "gun", "weapon", "sword", "rod", "axe"},
                   "category": "means"},
    "weapon": {"actions": {"knife", "gun", "weapon", "sword", "rod", "axe"},
               "category": "means"},
    "stabbing": {"actions": {"stabbed", "knife"},
                 "category": "means"},
    # Deception
    "deceiving": {"actions": {"deceived", "lied", "cheated", "misrepresented", "promised"},
                  "category": "actus_reus"},
    "cheating": {"actions": {"deceived", "cheated", "misrepresented", "promised"},
                 "category": "actus_reus"},
    # Property
    "property": {"actions": {"took", "stole", "snatched", "robbed", "transferred"},
                 "category": "actus_reus"},
    "delivery": {"actions": {"transferred", "delivered", "gave"},
                 "category": "actus_reus"},
}

FACT_TYPE_TO_INGREDIENT = {
    "causing_death": ["death", "causing death"],
    "use_of_force": ["force", "bodily injury", "hurt", "bodily pain"],
    "deception": ["deceiving", "cheating"],
    "property_transfer": ["property", "delivery"],
    "evidence_recovery": ["instrument", "weapon"],
    "criminal_intimidation": ["threat"],
    "unlawful_entry": ["trespass", "entry"],
    "forgery": ["forgery", "document"],
    "conspiracy": ["conspiracy", "abetment"],
    "unlawful_restraint": ["restraint", "confinement"],
}

SIGNAL_TO_INGREDIENT = {
    "death_occurred": ["death", "causing death"],
    "death_by_violence": ["death", "causing death", "force"],
    "death_following_force": ["death", "causing death", "force"],
    "physical_force_used": ["force", "bodily injury", "hurt"],
    "weapon_recovered": ["instrument", "weapon"],
    "weapon_present": ["instrument", "weapon"],
    "physical_evidence_recovered": ["instrument", "weapon"],
    "property_taken": ["property"],
    "property_transferred": ["property", "delivery"],
    "deceptive_conduct_observed": ["deceiving", "cheating"],
    "property_obtained_by_deception": ["deceiving", "property"],
    "threat_made": ["threat"],
}


def _classify_ingredient(ingredient_text: str) -> str:
    """Classify an ingredient as actus_reus, mens_rea, or circumstance."""
    lower = ingredient_text.lower()
    
    # Mens rea indicators
    mens_rea_terms = [
        "intention", "intent", "knowledge", "knew", "knows",
        "dishonest", "fraudulent", "mens rea", "premeditat",
        "motive", "purpose of", "with a view to",
    ]
    if any(t in lower for t in mens_rea_terms):
        return "mens_rea"
    
    # Actus reus indicators
    actus_reus_terms = [
        "causing", "death", "bodily", "hurt", "force", "act",
        "does any act", "commission", "assault", "taking",
        "deceiving", "property", "enters", "commits",
    ]
    if any(t in lower for t in actus_reus_terms):
        return "actus_reus"
    
    return "circumstance"


def _match_ingredient_to_evidence(
    ingredient_text: str,
    ingredient_element_type: str,
    events: list[dict],
    legal_facts: list[dict],
    signals: list[dict],
) -> dict:
    """
    Match a single ingredient against evidence layers.
    Priority: events > facts > signals.
    Returns dict with status, confidence, supporting_nodes, evidence_type, reasoning.
    """
    lower_text = ingredient_text.lower()
    ingredient_category = _classify_ingredient(ingredient_text)
    
    # If element_type from the dataset indicates it's mens_rea or actus_reus, use that
    if ingredient_element_type in CORE_INGREDIENT_TYPES:
        ingredient_category = ingredient_element_type
    
    # ── Check Events First ──
    event_matches = []
    for evt in events:
        action = (evt.get("action") or "").lower()
        obj = (evt.get("object") or "").lower()
        desc = (evt.get("description") or "").lower()
        
        # Find keyword matches between ingredient text and event
        for kw, kw_info in INGREDIENT_EVENT_KEYWORDS.items():
            if kw in lower_text:
                matched_actions = kw_info["actions"]
                if action in matched_actions or any(ma in obj for ma in matched_actions):
                    event_matches.append({
                        "node": evt.get("event_id", ""),
                        "action": action,
                        "keyword": kw,
                    })
                    break
                # Also check description for broader matches
                if any(ma in desc for ma in matched_actions):
                    event_matches.append({
                        "node": evt.get("event_id", ""),
                        "action": action,
                        "keyword": kw,
                    })
                    break
    
    if event_matches:
        # Mens rea ingredients can NEVER be satisfied by events alone
        # (events show what happened, not intent)
        if ingredient_category == "mens_rea":
            return {
                "status": "partial",
                "confidence": 0.4,
                "supporting_nodes": [m["node"] for m in event_matches],
                "evidence_type": "event",
                "reasoning": f"Event shows action but intent/knowledge cannot be directly inferred from events alone.",
            }
        return {
            "status": "satisfied",
            "confidence": 0.8,
            "supporting_nodes": [m["node"] for m in event_matches],
            "evidence_type": "event",
            "reasoning": f"Event directly supports ingredient: matched on '{event_matches[0]['keyword']}' via action '{event_matches[0]['action']}'.",
        }
    
    # ── Check Facts Second ──
    fact_matches = []
    for fact in legal_facts:
        ft = fact.get("type", "")
        if ft in FACT_TYPE_TO_INGREDIENT:
            mapping_keywords = FACT_TYPE_TO_INGREDIENT[ft]
            if any(kw in lower_text for kw in mapping_keywords):
                fact_matches.append({
                    "node": fact.get("fact_id", ""),
                    "type": ft,
                })
    
    if fact_matches:
        if ingredient_category == "mens_rea":
            return {
                "status": "partial",
                "confidence": 0.3,
                "supporting_nodes": [m["node"] for m in fact_matches],
                "evidence_type": "fact",
                "reasoning": f"Legal fact '{fact_matches[0]['type']}' partially supports but intent cannot be established from facts alone.",
            }
        return {
            "status": "satisfied",
            "confidence": 0.7,
            "supporting_nodes": [m["node"] for m in fact_matches],
            "evidence_type": "fact",
            "reasoning": f"Legal fact '{fact_matches[0]['type']}' supports this ingredient.",
        }
    
    # ── Check Signals Third (weakest) ──
    signal_matches = []
    for sig in signals:
        sname = sig.get("signal", "")
        if sname in SIGNAL_TO_INGREDIENT:
            mapping_keywords = SIGNAL_TO_INGREDIENT[sname]
            if any(kw in lower_text for kw in mapping_keywords):
                signal_matches.append({
                    "node": sig.get("signal_id", ""),
                    "signal": sname,
                })
    
    if signal_matches:
        # CRITICAL: Signals ALONE cannot fully satisfy core ingredients
        if ingredient_category in CORE_INGREDIENT_TYPES:
            return {
                "status": "partial",
                "confidence": 0.3,
                "supporting_nodes": [m["node"] for m in signal_matches],
                "evidence_type": "signal",
                "reasoning": f"Signal '{signal_matches[0]['signal']}' provides weak support but signals alone cannot satisfy core ingredient '{ingredient_category}'.",
            }
        return {
            "status": "partial",
            "confidence": 0.5,
            "supporting_nodes": [m["node"] for m in signal_matches],
            "evidence_type": "signal",
            "reasoning": f"Signal '{signal_matches[0]['signal']}' partially supports this ingredient.",
        }
    
    # ── No evidence found ──
    return {
        "status": "not_satisfied",
        "confidence": 0.1,
        "supporting_nodes": [],
        "evidence_type": "none",
        "reasoning": f"No evidence found in events, facts, or signals to support this ingredient.",
    }


def _evaluate_statute_deterministic(
    statute_section: str,
    ingredients: list[dict],
    events: list[dict],
    legal_facts: list[dict],
    signals: list[dict],
) -> list[dict]:
    """Deterministic evaluation of all ingredients for one statute."""
    results = []
    for ing in ingredients:
        ing_text = ing.get("text", "") or ing.get("description", "")
        ing_id = ing.get("id", "") or ing.get("ingredient_id", "")
        element_type = ing.get("element_type", "")
        
        if not ing_text:
            continue
        
        match = _match_ingredient_to_evidence(
            ing_text, element_type, events, legal_facts, signals
        )
        results.append({
            "ingredient_id": ing_id,
            "ingredient_text": ing_text,
            "element_type": element_type,
            "ingredient_category": _classify_ingredient(ing_text),
            **match,
        })
    
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3 — LLM Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STATUS_ORDER = {"not_satisfied": 0, "partial": 1, "satisfied": 2}


def _llm_validate(
    statute_section: str,
    ingredients_with_status: list[dict],
    events: list[dict],
    legal_facts: list[dict],
    signals: list[dict],
) -> list[dict]:
    """
    LLM second pass — can reason but MUST NOT hallucinate.
    Merges results conservatively: takes the MORE CONSERVATIVE status.
    """
    if not GOOGLE_API_KEY:
        return ingredients_with_status
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Build context
        events_str = json.dumps(events[:30], indent=1)
        facts_str = json.dumps(legal_facts[:20], indent=1)
        signals_str = json.dumps(signals[:20], indent=1)
        ingredients_str = json.dumps(
            [{"id": i["ingredient_id"], "text": i["ingredient_text"],
              "current_status": i["status"]} for i in ingredients_with_status],
            indent=1,
        )
        
        prompt = f"""You are a legal ingredient evaluator.

For each statute:
1. Evaluate each ingredient strictly based on facts.
2. Classify the statute overall as: STRONGLY SATISFIED, PARTIALLY SATISFIED, or NOT SATISFIED.

IMPORTANT:
- Do NOT finalize the offence.
- Do NOT compare statutes yet.
- Do NOT assume intent unless logically inferred from act + context.
- Clearly separate:
   → intention
   → knowledge
   → causation

EVIDENCE (read carefully):
Events: {events_str}
Legal Facts: {facts_str}
Signals: {signals_str}

INGREDIENTS TO EVALUATE FOR SECTION {statute_section}:
{ingredients_str}

OUTPUT:
Return ONLY a JSON object with this exact structure:
{{
  "statute_classification": "STRONGLY SATISFIED",
  "statute_reasoning": "...",
  "ingredients": [
    {{
      "id": "...",
      "llm_status": "satisfied|partial|not_satisfied",
      "reasoning": "..."
    }}
  ]
}}
Do NOT output markdown blocks like ```json."""
        
        AGENT_6_SYSTEM_PROMPT = """You are a meticulous legal ingredient evaluator.
Your ONLY function is to rigorously evaluate whether specific statutory ingredients are satisfied by provided evidence, keeping intent/knowledge/causation strictly separated."""

        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=AGENT_6_SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )
        
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        with open("llm_debug_output.txt", "a") as f:
            f.write(f"\n--- SECTION {statute_section} ---\n{text}\n")
        
        llm_response_data = json.loads(text)
        if not isinstance(llm_response_data, dict):
            return ingredients_with_status
            
        llm_results = llm_response_data.get("ingredients", [])
        statute_class = llm_response_data.get("statute_classification", "")
        statute_reason = llm_response_data.get("statute_reasoning", "")
        
        if not isinstance(llm_results, list):
            return ingredients_with_status
        
        # Build lookup
        llm_map = {}
        for lr in llm_results:
            llm_map[lr.get("id", "")] = lr
        
        # Merge: take MORE CONSERVATIVE (lower status wins)
        for ing in ingredients_with_status:
            ing_id = ing["ingredient_id"]
            if ing_id in llm_map:
                llm_status = llm_map[ing_id].get("llm_status", "")
                llm_reasoning = llm_map[ing_id].get("reasoning", "")
                
                if llm_status in STATUS_ORDER:
                    det_rank = STATUS_ORDER.get(ing["status"], 1)
                    llm_rank = STATUS_ORDER.get(llm_status, 1)
                    
                    # Take the more conservative (lower rank)
                    if llm_rank < det_rank:
                        ing["status"] = llm_status
                        ing["confidence"] = max(0.1, ing["confidence"] - 0.2)
                        ing["reasoning"] = f"[LLM downgrade] {llm_reasoning}"
                    elif llm_rank > det_rank:
                        # LLM found evidence -> we can upgrade to satisfied or partial
                        ing["status"] = llm_status
                        ing["confidence"] = 0.8 if llm_status == "satisfied" else 0.5
                        ing["reasoning"] = f"[LLM upgrade] {llm_reasoning}"
                    
                    # Add LLM note
                    ing["llm_validated"] = True
                    ing["llm_status"] = llm_status
            
            # Inject the overall statute classification into the first ingredient or as a side-effect
            if ingredients_with_status:
                ingredients_with_status[0]["_llm_statute_classification"] = statute_class
                ingredients_with_status[0]["_llm_statute_reasoning"] = statute_reason
        
    except Exception as e:
        print(f"!!! CRITICAL LLM FALLBACK ERROR for Section {statute_section}: {e} !!!")
        import traceback
        traceback.print_exc()
        logger.warning(f"LLM validation failed for section {statute_section}: {e}")
    
    return ingredients_with_status


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Steps 4-5 — Core Rule + Scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_score(ingredients: list[dict]) -> tuple[float, str]:
    """
    Score = (S + 0.5*P) / total
    If any core ingredient (actus_reus / mens_rea) is NOT_SATISFIED → score = 0
    """
    if not ingredients:
        return 0.0, "invalid"
    
    # Core ingredient check
    for ing in ingredients:
        cat = ing.get("ingredient_category", "")
        if cat in CORE_INGREDIENT_TYPES and ing["status"] == "not_satisfied":
            return 0.0, "invalid"
    
    total = len(ingredients)
    satisfied = sum(1 for i in ingredients if i["status"] == "satisfied")
    partial = sum(1 for i in ingredients if i["status"] == "partial")
    
    score = (satisfied + 0.5 * partial) / total
    
    if score >= 0.7:
        status = "valid"
    elif score >= 0.3:
        status = "weak"
    else:
        status = "invalid"
    
    return round(score, 4), status


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_ingredients_for_section(section_id: str) -> list[dict]:
    """Get ingredients from both IPC datasets."""
    # Try the detailed dataset first
    try:
        detailed = load_json(IPC_INGREDIENTS_PATH)
        ipc_dict = detailed.get("IPC", {})
        if section_id in ipc_dict:
            return ipc_dict[section_id].get("ingredients", [])
    except Exception:
        pass
    
    # Fallback to simplified dataset
    try:
        simple = load_json(IPC_INGREDIENTS_SIMPLE_PATH)
        for sec in simple.get("sections", []):
            if sec.get("section") == section_id:
                # Convert simple format to detailed format
                return [
                    {
                        "id": f"{section_id}.{i+1}",
                        "text": ing_text,
                        "element_type": _classify_ingredient(ing_text),
                    }
                    for i, ing_text in enumerate(sec.get("ingredients", []))
                ]
    except Exception:
        pass
    
    return []


def run(tenant_id: str, case_id: str) -> dict:
    """
    Run Agent 6 — Ingredient Evaluator.
    
    Reads: event_graph.json, legal_facts.json, legal_signals.json, statute_candidates.json
    Writes: ingredient_report.json
    """
    case_dir = get_case_dir(tenant_id, case_id)
    
    # Load inputs
    event_graph = load_json(case_dir / "event_graph.json")
    legal_facts_data = load_json(case_dir / "legal_facts.json")
    signals_data = load_json(case_dir / "legal_signals.json")
    statute_candidates = load_json(case_dir / "statute_candidates.json")
    
    events = event_graph.get("events", [])
    legal_facts = legal_facts_data.get("legal_facts", [])
    signals = signals_data.get("signals", [])
    candidates = statute_candidates.get("candidates", [])
    
    # Step 1: Consistency Check
    consistency_flags = _consistency_check(events, legal_facts, signals)
    
    # Steps 2-5: Evaluate each statute
    statute_evaluations = []
    
    import concurrent.futures

    def _process_candidate(candidate):
        section_id = candidate.get("section", "")
        
        # Get full ingredient list for this section
        ingredients = _get_ingredients_for_section(section_id)
        
        if not ingredients:
            # Use matched_ingredients from statute_candidates as fallback
            matched = candidate.get("matched_ingredients", [])
            ingredients = [
                {
                    "id": m.get("ingredient_id", ""),
                    "text": m.get("text", ""),
                    "element_type": m.get("element_type", ""),
                }
                for m in matched
            ]
        
        if not ingredients:
            return None
        
        # Step 2: Deterministic matching
        ingredient_results = _evaluate_statute_deterministic(
            section_id, ingredients, events, legal_facts, signals
        )
        
        # Step 3: LLM validation (merge conservatively)
        ingredient_results = _llm_validate(
            section_id, ingredient_results, events, legal_facts, signals
        )
        
        # Steps 4-5: Scoring with core rule
        overall_score, status = _compute_score(ingredient_results)
        
        # Override status if LLM gave a firm classification
        llm_stat_class = ingredient_results[0].pop("_llm_statute_classification", "") if ingredient_results else ""
        llm_stat_reason = ingredient_results[0].pop("_llm_statute_reasoning", "") if ingredient_results else ""
        
        if llm_stat_class == "STRONGLY SATISFIED":
            status = "valid"
        elif llm_stat_class == "NOT SATISFIED":
            status = "invalid"
            
        return {
            "statute_id": section_id,
            "heading": candidate.get("heading", ""),
            "overall_score": overall_score,
            "status": status,
            "llm_classification": llm_stat_class,
            "llm_classification_reasoning": llm_stat_reason,
            "ingredients": ingredient_results,
            "consistency_flags": [
                f for f in consistency_flags
            ],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_process_candidate, c) for c in candidates]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                statute_evaluations.append(res)
    # Sort by score descending
    statute_evaluations.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
    
    # Build output
    provenance = build_provenance(
        case_id, tenant_id, "agent_6_ingredient_evaluator",
        ["event_graph.json", "legal_facts.json", "legal_signals.json", "statute_candidates.json"],
    )
    
    result = {
        **provenance,
        "statute_evaluations": statute_evaluations,
        "consistency_flags": consistency_flags,
    }
    
    save_json(case_dir / "ingredient_report.json", result)
    return result
