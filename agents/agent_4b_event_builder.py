"""
Agent 4B — Event Builder (Typed Event System)

Input:  role_tagged_graph.json, entities.json
Output: event_graph.json

Convert nodes → structured events with typed event system.
UPGRADE: Structured Extraction Rules
1. Actor/Target Resolution based on sentence structure (subject/object of verb).
2. Strict Event Typing (MANDATORY): action, state, evidence, context.
3. Verb Classification: predefined sets map to state/evidence/action.
4. STRICT: NEVER swap actor/target incorrectly. NEVER invent fields. NEVER discard nodes.
"""

import re
from pathlib import Path

from config import get_case_dir, load_json, save_json, build_provenance

# Verb classifications
STATE_VERBS = [
    "stopped responding", "not delivered", "unconscious",
    "destroyed", "injured", "wounded",
    "damaged", "refused", "stopped", "absent", "broken",
    "failed", "denied", "died", "dead"
]

EVIDENCE_VERBS = {
    "recovered", "seized", "found", "discovered", "examined"
}

ACTION_VERBS = [
    "attacked", "stabbed", "shot", "hit", "beat", "killed", "murdered",
    "assaulted", "punched", "kicked", "slapped", "struck", "pushed",
    "took", "stole", "snatched", "robbed", "grabbed", "looted",
    "lied", "deceived", "cheated", "misrepresented", "promised",
    "threatened", "intimidated", "coerced", "blackmailed",
    "entered", "trespassed", "broke in", "intruded",
    "forged", "fabricated", "falsified", "counterfeited",
    "set fire", "burned", "torched",
    "conspired", "planned", "plotted",
    "abducted", "kidnapped", "confined", "detained",
    "fled", "escaped", "absconded",
    "transferred", "delivered", "paid", "received", "promised", "struck"
]

def _extract_verbs(text: str) -> list[str]:
    """Extract all verbs from text across all categories."""
    lower = text.lower()
    
    # Enforce verb normalization
    lower = re.sub(r'\binjuries\b', 'injured', lower)
    lower = re.sub(r'\bdeath\b', 'died', lower)
    
    found = []
    # Check states
    for v in STATE_VERBS:
        if re.search(rf"\b{re.escape(v)}\b", lower) and v not in found:
            found.append(v)
            
    # Check evidence
    for v in EVIDENCE_VERBS:
        if re.search(rf"\b{re.escape(v)}\b", lower) and v not in found:
            found.append(v)
            
    # Check actions
    for v in ACTION_VERBS:
        if re.search(rf"\b{re.escape(v)}\b", lower) and v not in found:
            found.append(v)
            
    return found if found else [None]

def _resolve_actor_target(text: str, entities: list[dict], nid: str, verb: str | None) -> tuple[str | None, str | None]:
    """
    Resolve Actor and Target using sentence structure relative to verb.
    Approximation for Subject (Actor) and Object (Target).
    """
    node_actors = [e["value"] for e in entities if e["type"] == "actor" and nid in e.get("source_node_ids", [])]
    
    if not node_actors:
        return None, None
    if not verb:
        # Default to order if no verb
        return node_actors[0], node_actors[1] if len(node_actors) > 1 else None

    # Check for passive voice (e.g., "was attacked by")
    is_passive = bool(re.search(rf"\b(?:was|were|is|are|been)\s+{re.escape(verb)}\b", text.lower()))
    
    verb_pos = text.lower().find(verb)
    if verb_pos == -1:
        return node_actors[0], node_actors[1] if len(node_actors) > 1 else None

    before_verb = []
    after_verb = []

    for a in node_actors:
        # roughly check position
        pos = text.lower().find(a.lower())
        if pos != -1 and pos < verb_pos:
            before_verb.append(a)
        elif pos != -1 and pos > verb_pos:
            after_verb.append(a)
            
    if is_passive:
        # Subject is target, Object (after "by") is actor
        target = before_verb[0] if before_verb else (after_verb[0] if after_verb else None)
        actor = None
        # Try to find actor after 'by'
        by_match = re.search(rf"{re.escape(verb)}.*?\bby\b\s+([^,\.]*)", text.lower())
        if by_match:
            by_text = by_match.group(1)
            for a in after_verb:
                if a.lower() in by_text:
                    actor = a
                    break
        if not actor and len(after_verb) > 0:
            # Fallback if multiple
            actor = after_verb[0] if target != after_verb[0] else None
            
        return actor, target
    else:
        # Active voice: Subject is actor, Object is target
        actor = before_verb[0] if before_verb else (node_actors[0] if node_actors else None)
        target = after_verb[0] if after_verb else None
        # Handle case where both are before or both after due to complex sentences
        if len(before_verb) > 1 and not after_verb:
            target = before_verb[1]
        elif len(after_verb) > 1 and not before_verb:
            actor = after_verb[0]
            target = after_verb[1]
            
        if actor == target:
            target = None
            
        return actor, target

def _classify_event_type(verb: str | None, has_actor: bool, has_object: bool) -> str:
    if not verb:
        if has_actor or has_object:
            return "context"  # just mentions entities
        return "context"
        
    if verb in STATE_VERBS:
        return "state"
    if verb in EVIDENCE_VERBS:
        return "evidence"
        
    # Otherwise it's an action
    return "action"

def run(tenant_id: str, case_id: str) -> dict:
    case_dir = get_case_dir(tenant_id, case_id)
    graph = load_json(case_dir / "role_tagged_graph.json")
    entities_data = load_json(case_dir / "entities.json")

    nodes = graph.get("nodes", [])
    entities = entities_data.get("entities", [])

    events = []
    event_counter = 1
    for node in nodes:
        nid = node["node_id"]
        text = node["text"]
        
        # Fix possessives
        text = text.replace("'s", "").replace("’s", "")

        verbs = _extract_verbs(text)
        
        for verb in verbs:
            actor, target = _resolve_actor_target(text, entities, nid, verb)
            
            # Object resolution
            node_objects = [e["value"] for e in entities if e["type"] == "object" and nid in e.get("source_node_ids", [])]
            obj = node_objects[0] if node_objects else None
            if obj:
                obj = obj.replace("'s", "").replace("’s", "")
            
            # Time / Location
            node_times = [e["value"] for e in entities if e["type"] == "time" and nid in e.get("source_node_ids", [])]
            time = node_times[0] if node_times else None
            
            node_locs = [e["value"] for e in entities if e["type"] == "location" and nid in e.get("source_node_ids", [])]
            location = node_locs[0] if node_locs else None
            if location:
                location = location.replace("'s", "").replace("’s", "")

            event_type = _classify_event_type(verb, bool(actor), bool(obj))

            event = {
                "event_id": f"evt_{event_counter:03d}",
                "event_type": event_type,
                "actor": actor,
                "action": verb,
                "target": target,
                "object": obj,
                "time": time,
                "location": location,
                "description": text,
                "source_node_id": nid,
            }
            events.append(event)
            event_counter += 1

    provenance = build_provenance(
        case_id, tenant_id, "agent_4b_event_builder", ["role_tagged_graph.json", "entities.json"]
    )

    result = {
        **provenance,
        "events": events,
    }

    save_json(case_dir / "event_graph.json", result)
    return result
