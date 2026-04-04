"""
Agent 5B — Legal Signal Extractor (LLM Allowed)

Input:  event_graph.json, legal_facts.json
Output: legal_signals.json

Extract observable, factual, non-interpretive legal signals.
UPGRADE: ENFORCE UNIVERSAL SIGNAL RULES
1. Signals must be observable, factual, non-interpretive.
2. FORBIDDEN: any crime label (murder, fraud, rape, theft).
3. CO-OCCURRENCE: Only generate derived signals if BOTH base facts exist.
4. VALIDATION: Ensure every signal maps to source facts and no crime words.
"""

import json
import re
from pathlib import Path

from config import (
    get_case_dir,
    load_json,
    save_json,
    build_provenance,
    FORBIDDEN_SIGNAL_WORDS,
    GOOGLE_API_KEY,
    LLM_SAFETY_PROMPT,
    DEFAULT_MODEL
)

# Deterministic signal map from the new abstractions
FACT_TO_SIGNAL = {
    "causing_death": {
        "signal": "death_occurred",
        "category": ["death"],
    },
    "death_occurred": {
        "signal": "death_occurred",
        "category": ["death"],
    },
    "use_of_force": {
        "signal": "physical_force_used",
        "category": ["force"],
    },
    "injury_occurred": {
        "signal": "injury_detected",
        "category": ["force"],
    },
    "property_transfer": {
        "signal": "property_transferred",
        "category": ["property"],
    },
    "property_taken": {
        "signal": "property_taken",
        "category": ["property"],
    },
    "deception": {
        "signal": "deceptive_conduct_observed",
        "category": ["property", "evidence"],
    },
    "threat": {
        "signal": "threat_made",
        "category": ["force"],
    },
    "unlawful_entry": {
        "signal": "unauthorized_entry",
        "category": ["property"],
    },
    "forgery": {
        "signal": "document_falsification_detected",
        "category": ["evidence"],
    },
    "arson": {
        "signal": "fire_destruction_observed",
        "category": ["property"],
    },
    "property_damage": {
        "signal": "property_damage_observed",
        "category": ["property"],
    },
    "conspiracy": {
        "signal": "coordinated_planning_detected",
        "category": ["evidence"],
    },
    "abduction": {
        "signal": "person_removed",
        "category": ["force"],
    },
    "evidence_present": {
        "signal": "evidence_found",
        "category": ["evidence"],
    },
    "failure_to_act": {
        "signal": "failure_to_deliver",
        "category": ["property"],
    },
    "non_response": {
        "signal": "absence_of_response",
        "category": ["evidence"],
    },
    "absence": {
        "signal": "person_missing",
        "category": ["evidence"],
    },
}

# Strict Co-occurrence Rules
CO_OCCURRENCE_SIGNALS = [
    {
        "required_facts": ["use_of_force", "causing_death"],
        "signal": "death_following_force",
        "category": ["death", "force"],
    },
    {
        "required_facts": ["property_transfer", "deception"],
        "signal": "property_obtained_by_deception",
        "category": ["property"],
    },
    {
        "required_facts": ["use_of_force", "property_taken"],
        "signal": "forceful_property_taking",
        "category": ["property", "force"],
    },
    {
        "required_facts": ["use_of_force", "evidence_present"],
        "signal": "weapon_present",
        "category": ["weapon", "evidence", "force"],
    },
]

def _is_signal_forbidden(signal_name: str) -> bool:
    lower = signal_name.lower()
    for word in FORBIDDEN_SIGNAL_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return True
    return False

def _extract_deterministic_signals(legal_facts: list[dict]) -> list[dict]:
    signals = []
    
    # Track fact types and their IDs
    fact_map = {}
    for fact in legal_facts:
        ft = fact.get("type", "")
        if ft not in fact_map:
            fact_map[ft] = []
        fact_map[ft].append(fact["fact_id"])

    signal_counter = 0

    # 1. Base Facts -> Signals
    for ft, fact_ids in fact_map.items():
        if ft in FACT_TO_SIGNAL:
            sig_info = FACT_TO_SIGNAL[ft]
            signal_counter += 1
            signals.append({
                "signal_id": f"sig_{signal_counter:03d}",
                "signal": sig_info["signal"],
                "categories": sig_info["category"],
                "confidence": 1.0,
                "source": "deterministic",
                "source_fact_ids": fact_ids,
            })

    # 2. Co-occurrence
    for co in CO_OCCURRENCE_SIGNALS:
        if all(req in fact_map for req in co["required_facts"]):
            signal_counter += 1
            
            source_ids = []
            for req in co["required_facts"]:
                source_ids.extend(fact_map[req])
                
            signals.append({
                "signal_id": f"sig_{signal_counter:03d}",
                "signal": co["signal"],
                "categories": co["category"],
                "confidence": 0.9,
                "source": "co_occurrence",
                "source_fact_ids": list(set(source_ids)),
            })

    return signals

def _extract_llm_signals(events: list[dict], legal_facts: list[dict], existing_names: list[str]) -> list[dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    events_str = json.dumps(events[:30], indent=1)
    facts_str = json.dumps(legal_facts[:30], indent=1)
    existing_str = json.dumps(existing_names)

    prompt = f"""Analyze these legal events and facts to extract strictly observable legal signals.

RULES:
- Signals MUST be observable and factual (e.g., "death_occurred", "property_taken")
- Signals MUST NOT be legal conclusions (e.g., "murder_committed", "fraud_detected")
- FORBIDDEN words: murder, fraud, theft, robbery, rape, assault, arson, crime, guilty, offence
- Output ONLY a JSON array: [{{"signal": "...", "categories": ["..."], "source_fact_ids": ["..."]}}]
- "categories" must be a list of broad IPC categories (e.g. ["violence"], ["property"]).
- Do not duplicate these signals: {existing_str}

Events:
{events_str}

Legal Facts:
{facts_str}"""

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=LLM_SAFETY_PROMPT,
        ),
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        raw = json.loads(text)
        return raw if isinstance(raw, list) else []
    except Exception:
        return []

def _validate_signals(signals: list[dict], all_fact_ids: set) -> list[dict]:
    """Ensure signals contain no crime words and map validly to source facts."""
    valid = []
    seen = set()

    for sig in signals:
        name = sig.get("signal", "")
        # Remove crimes
        if _is_signal_forbidden(name):
            continue
            
        # Deduplicate
        if name in seen:
            continue
            
        # Ensure source facts exist if it is a co-occurrence
        source_ids = sig.get("source_fact_ids", [])
        if sig.get("source") == "co_occurrence" or sig.get("source") == "llm":
            if not source_ids or not all(fid in all_fact_ids for fid in source_ids):
                # If LLM failed to map proper source facts, drop it or map if deterministic
                if sig.get("source") == "co_occurrence":
                    continue
        
        # Ensure categories is a list
        if "category" in sig and "categories" not in sig:
            sig["categories"] = [sig["category"]] if isinstance(sig["category"], str) else sig["category"]
            
        seen.add(name)
        valid.append(sig)

    return valid

def run(tenant_id: str, case_id: str) -> dict:
    case_dir = get_case_dir(tenant_id, case_id)
    event_graph = load_json(case_dir / "event_graph.json")
    legal_facts_data = load_json(case_dir / "legal_facts.json")

    events = event_graph.get("events", [])
    legal_facts = legal_facts_data.get("legal_facts", [])

    all_fact_ids = {f["fact_id"] for f in legal_facts}

    signals = _extract_deterministic_signals(legal_facts)

    if GOOGLE_API_KEY:
        try:
            existing_names = [s["signal"] for s in signals]
            llm_sigs = _extract_llm_signals(events, legal_facts, existing_names)
            
            counter = len(signals)
            for ls in llm_sigs:
                if not ls.get("signal"): continue
                counter += 1
                signals.append({
                    "signal_id": f"sig_{counter:03d}",
                    "signal": ls["signal"],
                    "categories": ls.get("categories", ["uncategorized"]),
                    "confidence": 0.7,
                    "source": "llm",
                    "source_fact_ids": [fid for fid in ls.get("source_fact_ids", []) if fid in all_fact_ids],
                })
        except Exception:
            pass

    valid_signals = _validate_signals(signals, all_fact_ids)

    for i, sig in enumerate(valid_signals):
        sig["signal_id"] = f"sig_{i + 1:03d}"

    provenance = build_provenance(
        case_id, tenant_id, "agent_5b_legal_signal_extractor", ["event_graph.json", "legal_facts.json"]
    )

    result = {
        **provenance,
        "signals": valid_signals,
    }

    save_json(case_dir / "legal_signals.json", result)
    return result
