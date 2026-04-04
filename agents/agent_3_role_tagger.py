"""
Agent 3 — Role Tagging (LLM Allowed)

Input:  document_graph.json
Output: role_tagged_graph.json

Classify each sentence node into one of 7 legal roles.
Uses Google Gemini with temperature=0 for classification.
LLM MUST ONLY classify — no explanations, no new text generation.
If unsure → "background".
"""

import json
from pathlib import Path

from config import (
    get_case_dir,
    load_json,
    save_json,
    build_provenance,
    ALLOWED_ROLES,
    GOOGLE_API_KEY,
    LLM_SAFETY_PROMPT,
    DEFAULT_MODEL
)


def _classify_batch_llm(nodes: list[dict]) -> list[str]:
    """
    Classify a batch of nodes using Google Gemini.

    Returns a list of role labels aligned with the input nodes.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)

    allowed = json.dumps(ALLOWED_ROLES)
    node_texts = []
    for n in nodes:
        node_texts.append(f'{n["node_id"]}: {n["text"]}')

    prompt = f"""Classify each sentence into exactly one role from: {allowed}

Rules:
- Output ONLY a JSON array of objects: [{{"node_id": "...", "role": "..."}}]
- No explanations, no extra text
- If unsure, use "background"
- Classify based on content, not position

Sentences:
{chr(10).join(node_texts)}"""

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=LLM_SAFETY_PROMPT,
        ),
    )

    # Parse response
    text = response.text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        results = json.loads(text)
        role_map = {}
        for r in results:
            role = r.get("role", "background")
            if role not in ALLOWED_ROLES:
                role = "background"
            role_map[r["node_id"]] = role
        return role_map
    except (json.JSONDecodeError, KeyError):
        # Fallback: all background
        return {n["node_id"]: "background" for n in nodes}


def _classify_deterministic(nodes: list[dict]) -> dict[str, str]:
    """Fallback deterministic classifier when LLM is unavailable."""
    role_map = {}
    for node in nodes:
        text = node["text"].lower()
        nid = node["node_id"]

        if any(w in text for w in ["filed", "registered", "fir", "complaint", "charge sheet", "court"]):
            role_map[nid] = "procedural"
        elif any(w in text for w in ["witness", "testified", "deposed", "stated that"]):
            role_map[nid] = "witness_statement"
        elif any(w in text for w in ["section", "ipc", "under", "punishable", "offence"]):
            role_map[nid] = "legal_claim"
        elif any(w in text for w in ["recovered", "seized", "found", "forensic", "report", "exhibit"]):
            role_map[nid] = "evidence"
        elif any(w in text for w in ["alleged", "claiming", "accused of", "charge"]):
            role_map[nid] = "allegation"
        elif any(w in text for w in ["on the", "the accused", "the victim", "attacked",
                                       "stabbed", "killed", "took", "stole", "hit",
                                       "died", "injured", "threatened"]):
            role_map[nid] = "fact"
        else:
            role_map[nid] = "background"

    return role_map


def run(tenant_id: str, case_id: str) -> dict:
    """
    Tag each node in the document graph with a legal role.

    Returns
    -------
    dict
        The role-tagged document graph with provenance.
    """
    case_dir = get_case_dir(tenant_id, case_id)
    doc_graph = load_json(case_dir / "document_graph.json")
    nodes = doc_graph["nodes"]
    edges = doc_graph.get("edges", [])

    # Attempt LLM classification, fall back to deterministic
    if GOOGLE_API_KEY:
        try:
            # Batch nodes (up to 20 per call)
            role_map = {}
            batch_size = 20
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]
                batch_roles = _classify_batch_llm(batch)
                role_map.update(batch_roles)
        except Exception:
            role_map = _classify_deterministic(nodes)
    else:
        role_map = _classify_deterministic(nodes)

    # Apply roles to nodes
    tagged_nodes = []
    for node in nodes:
        tagged_node = {**node, "role": role_map.get(node["node_id"], "background")}
        tagged_nodes.append(tagged_node)

    provenance = build_provenance(
        case_id, tenant_id, "agent_3_role_tagger", ["document_graph.json"]
    )

    role_tagged_graph = {
        **provenance,
        "nodes": tagged_nodes,
        "edges": edges,
    }

    save_json(case_dir / "role_tagged_graph.json", role_tagged_graph)

    return role_tagged_graph
