"""
Agent 4A — Entity Extraction

Input:  role_tagged_graph.json
Output: entities.json

Extract actors, objects, locations, and time expressions.
Deterministic (regex + heuristics).
STRICT: Only extract entities explicitly present. NO inferred roles.

UPGRADE: Added Entity Normalization Layer
  1. Deduplicate by longest meaningful span
  2. Time normalization (combine fragments, remove partials)
  3. Object cleanup (prefer detailed over generic)
  4. Only valid, meaningful entities output
"""

import re
from pathlib import Path

from config import get_case_dir, load_json, save_json, build_provenance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Actor patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACTOR_PATTERNS = [
    (r"\bthe accused\b", "accused"),
    (r"\baccused\b", "accused"),
    (r"\bthe victim\b", "victim"),
    (r"\bvictim\b", "victim"),
    (r"\bthe complainant\b", "complainant"),
    (r"\bcomplainant\b", "complainant"),
    (r"\bthe deceased\b", "deceased"),
    (r"\bdeceased\b", "deceased"),
    (r"\bthe witness\b", "witness"),
    (r"\bwitness\b", "witness"),
    (r"\bthe informant\b", "informant"),
    (r"\binformant\b", "informant"),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Object patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OBJECT_KEYWORDS = [
    "knife", "gun", "pistol", "revolver", "rifle", "sword",
    "axe", "hammer", "rod", "stick", "bat", "stone", "brick",
    "money", "cash", "property", "jewellery", "jewelry", "gold",
    "vehicle", "car", "motorcycle", "bike", "truck", "scooter",
    "phone", "mobile", "laptop", "computer",
    "document", "cheque", "check", "letter",
    "poison", "acid", "bottle", "rope",
    "blood", "fingerprint", "footprint",
    "goods", "payment",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Location patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOCATION_KEYWORDS = [
    "house", "home", "residence", "shop", "market", "road",
    "street", "lane", "village", "city", "district", "station",
    "hospital", "office", "park", "field", "ground",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Time patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIME_PATTERNS = [
    r"\b(?:on\s+the\s+)?(?:night|morning|afternoon|evening)\s+of\s+\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
    r"\b(?:on|at)\s+(?:the\s+)?(?:night|morning|afternoon|evening|dawn|dusk)\b",
    r"\bat\s+(?:about\s+)?\d{1,2}(?::\d{2})?\s*(?:hours?|hrs?|am|pm|AM|PM|a\.m\.|p\.m\.)\b",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extraction Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_actors(text: str) -> list[str]:
    found = set()
    lower = text.lower()
    for pattern, label in ACTOR_PATTERNS:
        if re.search(pattern, lower):
            found.add(label)
    return list(found)

def _extract_objects(text: str) -> list[str]:
    found = set()
    lower = text.lower()
    for keyword in OBJECT_KEYWORDS:
        # Match with preceding modifiers (e.g., blood-stained knife)
        pattern = rf"\b(?:[\w-]+\s+)?{re.escape(keyword)}s?\b"
        matches = re.finditer(pattern, lower)
        for m in matches:
            val = m.group().strip()
            # Object cleanup: discard "a blood" fragment (blood is a keyword, but 'a blood' is bad)
            if val == "a blood" or val == "the blood":
                continue
            found.add(val)
    return list(found)

def _extract_locations(text: str) -> list[str]:
    found = set()
    lower = text.lower()
    for keyword in LOCATION_KEYWORDS:
        pattern = rf"\b(?:[\w-]+\s+)?{re.escape(keyword)}\b"
        matches = re.finditer(pattern, lower)
        for m in matches:
            found.add(m.group().strip())
    return list(found)

def _extract_times(text: str) -> list[str]:
    found = set()
    for pattern in TIME_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for m in matches:
            found.add(m.group().strip())
    return list(found)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Normalization Layer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _normalize_entities(raw_entities: list[dict]) -> list[dict]:
    """
    1. Deduplicate by longest meaningful span
    2. Remove partial time fragments
    3. Remove invalid/incomplete objects
    """
    # Group by type
    by_type: dict[str, list[dict]] = {"actor": [], "object": [], "location": [], "time": []}
    for e in raw_entities:
        by_type[e["type"]].append(e)

    normalized = []

    # 1. Actors (keep unique values)
    seen_actors = set()
    for e in by_type["actor"]:
        if e["value"] not in seen_actors:
            seen_actors.add(e["value"])
            normalized.append(e)

    # 2. Objects (prefer longest span)
    objects = by_type["object"]
    objects.sort(key=lambda x: len(x["value"]), reverse=True)
    kept_objects = []
    for obj in objects:
        val = obj["value"]
        # If this is a substring of an already kept object, skip it
        # (e.g., "knife" is skipped if "blood-stained knife" is kept)
        if any(val in k["value"] for k in kept_objects):
            # Just merge source node IDs
            parent = next(k for k in kept_objects if val in k["value"])
            parent["source_node_ids"] = list(set(parent["source_node_ids"] + obj["source_node_ids"]))
            continue
        kept_objects.append(obj)
    normalized.extend(kept_objects)

    # 3. Locations (prefer longest span)
    locations = by_type["location"]
    locations.sort(key=lambda x: len(x["value"]), reverse=True)
    kept_locs = []
    for loc in locations:
        val = loc["value"]
        if any(val in k["value"] for k in kept_locs):
            parent = next(k for k in kept_locs if val in k["value"])
            parent["source_node_ids"] = list(set(parent["source_node_ids"] + loc["source_node_ids"]))
            continue
        kept_locs.append(loc)
    normalized.extend(kept_locs)

    # 4. Times (combine fragments)
    times = by_type["time"]
    times.sort(key=lambda x: len(x["value"]), reverse=True)
    kept_times = []
    for t in times:
        val = t["value"].lower()
        # Remove partials like "night of" or "on the night" if they stand alone without date
        if val in ["night of", "on the night", "morning of"]:
            continue
        
        # Subsume smaller into larger
        if any(val in k["value"].lower() for k in kept_times):
            parent = next(k for k in kept_times if val in k["value"].lower())
            parent["source_node_ids"] = list(set(parent["source_node_ids"] + t["source_node_ids"]))
            continue
            
        kept_times.append(t)
    normalized.extend(kept_times)

    return normalized

def run(tenant_id: str, case_id: str) -> dict:
    case_dir = get_case_dir(tenant_id, case_id)
    graph = load_json(case_dir / "role_tagged_graph.json")
    nodes = graph["nodes"]

    raw_entities = []
    entity_counter = 0

    for node in nodes:
        text = node["text"]
        nid = node["node_id"]

        actors = _extract_actors(text)
        objects = _extract_objects(text)
        locations = _extract_locations(text)
        times = _extract_times(text)

        for a in actors:
            raw_entities.append({"type": "actor", "value": a, "source_node_ids": [nid]})
        for o in objects:
            raw_entities.append({"type": "object", "value": o, "source_node_ids": [nid]})
        for l in locations:
            raw_entities.append({"type": "location", "value": l, "source_node_ids": [nid]})
        for t in times:
            raw_entities.append({"type": "time", "value": t, "source_node_ids": [nid]})

    # Normalize
    normalized_entities = _normalize_entities(raw_entities)
    
    # Assign fresh IDs
    for i, e in enumerate(normalized_entities):
        e["entity_id"] = f"ent_{i + 1:03d}"

    provenance = build_provenance(
        case_id, tenant_id, "agent_4a_entity_extractor", ["role_tagged_graph.json"]
    )

    result = {
        **provenance,
        "entities": normalized_entities,
    }

    save_json(case_dir / "entities.json", result)

    return result
