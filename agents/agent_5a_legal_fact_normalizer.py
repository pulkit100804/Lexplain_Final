"""
Agent 5A — Legal Fact Normalizer (Hybrid: Base Map + Synonym Expansion)

Input:  event_graph.json
Output: legal_facts.json

Convert events → legal abstractions using controlled vocabulary.
UPGRADE:
  1. Add full synonym expansion directory mapped to vocabulary.
  2. Map semantically similar verbs.
  3. Controlled abstraction vocabulary list (e.g. use_of_force, causing_death).
  4. Fallback -> "unknown_action"
  STRICT: DO NOT force incorrect mapping. Preserve original action.
"""

import re
from pathlib import Path

from config import get_case_dir, load_json, save_json, build_provenance

# Controlled Abstraction Vocabulary
CONTROLLED_VOCAB = {
    "use_of_force": ["attacked", "hit", "beat", "assaulted", "punched", "kicked", "slapped", "struck", "pushed", "stabbed", "shot"],
    "causing_death": ["killed", "murdered", "died", "dead"],
    "injury_occurred": ["injured", "wounded", "unconscious"],
    "property_transfer": ["transferred", "delivered", "paid", "received"],
    "property_taken": ["took", "stole", "snatched", "robbed", "grabbed", "looted"],
    "deception": ["lied", "deceived", "cheated", "misrepresented", "promised"],
    "threat": ["threatened", "intimidated", "coerced", "blackmailed"],
    "unlawful_entry": ["entered", "trespassed", "broke in", "intruded"],
    "forgery": ["forged", "fabricated", "falsified", "counterfeited"],
    "arson": ["set fire", "burned", "torched"],
    "conspiracy": ["conspired", "planned", "plotted"],
    "abduction": ["abducted", "kidnapped", "confined", "detained"],
    "flight": ["fled", "escaped", "absconded"],
    "evidence_present": ["recovered", "seized", "found", "discovered", "examined"],
    "failure_to_act": ["failed", "refused", "denied", "not delivered", "failure to deliver"],
    "non_response": ["stopped responding"],
    "absence": ["missing", "absent"],
    "property_damage": ["broken", "damaged", "destroyed"],
}

# Reverse map for O(1) synonym lookup
SYNONYM_MAP = {}
for abstraction, syn_list in CONTROLLED_VOCAB.items():
    for syn in syn_list:
        SYNONYM_MAP[syn.lower()] = abstraction

def _resolve_action(action: str | None) -> tuple[str, str]:
    if not action:
        return ("no_action", "none")
        
    action_lower = action.lower().strip()
    
    # 1. Direct synonym matching
    if action_lower in SYNONYM_MAP:
        return (SYNONYM_MAP[action_lower], "synonym_expansion")
        
    # 2. Substring matching
    for syn, abstraction in SYNONYM_MAP.items():
        if syn in action_lower:
            return (abstraction, "synonym_substring")
            
    return ("unknown_action", "unknown")

def run(tenant_id: str, case_id: str) -> dict:
    case_dir = get_case_dir(tenant_id, case_id)
    event_graph = load_json(case_dir / "event_graph.json")
    events = event_graph.get("events", [])

    legal_facts = []
    for i, event in enumerate(events):
        action = event.get("action")
        # Ensure we always preserve original action per STRICT rules
        original_action = action
        
        fact_type, mapped_from = _resolve_action(action)

        # Contextual description scan fallback for unknown/no actions
        if fact_type in ("unknown_action", "no_action") and event.get("description"):
            desc = event["description"].lower()
            found = False
            for syn, abstraction in SYNONYM_MAP.items():
                # FIX: ONLY map causing_death if explicit explicit action exists, not from description fallback
                if abstraction == "causing_death":
                    continue
                    
                if re.search(rf"\b{re.escape(syn)}\b", desc):
                    fact_type = abstraction
                    mapped_from = "description_scan"
                    # Preserve standard if no discrete action verb was originally found
                    if fact_type == "no_action":
                        action = syn
                    found = True
                    break

        legal_fact = {
            "fact_id": f"lf_{i + 1:03d}",
            "type": fact_type,
            "original_action": original_action,
            "mapped_from": mapped_from,
            "source_event_id": event.get("event_id"),
            "actor": event.get("actor"),
            "target": event.get("target"),
            "object": event.get("object"),
        }

        legal_facts.append(legal_fact)

    provenance = build_provenance(
        case_id, tenant_id, "agent_5a_legal_fact_normalizer", ["event_graph.json"]
    )

    result = {
        **provenance,
        "legal_facts": legal_facts,
    }

    save_json(case_dir / "legal_facts.json", result)
    return result
