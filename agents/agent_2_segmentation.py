"""
Agent 2 — Segmentation

Input:  normalized_text.txt
Output: document_graph.json

Deterministic only.
- Split into sentence-level nodes
- Maintain span indices
- Build adjacency edges
"""

import re
from pathlib import Path

from config import get_case_dir, save_json, build_provenance


def _split_sentences(text: str) -> list[dict]:
    """
    Split text into sentences with span tracking.

    Uses regex that handles:
    - Standard sentence-ending punctuation (. ! ?)
    - Abbreviations (single capital letter followed by period)
    - Decimal numbers
    - Quoted text
    """
    # Pattern: split on period/exclamation/question followed by space and uppercase letter or end
    # But not after single uppercase letters (abbreviations) or numbers
    sentence_pattern = re.compile(
        r'(?<!\b[A-Z])(?<!\b\d)(?<=[.!?])\s+(?=[A-Z"\'])|(?<=[.!?])\s*$',
    )

    nodes = []
    start = 0

    # Use finditer to get split positions
    parts = sentence_pattern.split(text)

    # Fallback: if regex doesn't split well, use simpler approach
    if len(parts) <= 1 and len(text) > 50:
        # Simple but robust sentence splitter
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        parts = [s.strip() for s in raw_sentences if s.strip()]

    offset = 0
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # Find actual position in original text
        idx = text.find(part, offset)
        if idx == -1:
            idx = offset

        node = {
            "node_id": f"n_{i + 1:03d}",
            "text": part,
            "span": {
                "start": idx,
                "end": idx + len(part),
            },
        }
        nodes.append(node)
        offset = idx + len(part)

    return nodes


def _build_edges(nodes: list[dict]) -> list[dict]:
    """Build sequential adjacency edges between nodes."""
    edges = []
    for i in range(len(nodes) - 1):
        edges.append({
            "source": nodes[i]["node_id"],
            "target": nodes[i + 1]["node_id"],
            "relation": "sequential",
        })
    return edges


def run(tenant_id: str, case_id: str) -> dict:
    """
    Segment normalized text into a document graph.

    Returns
    -------
    dict
        The document graph with provenance.
    """
    case_dir = get_case_dir(tenant_id, case_id)
    text = (case_dir / "normalized_text.txt").read_text(encoding="utf-8")

    nodes = _split_sentences(text)
    edges = _build_edges(nodes)

    provenance = build_provenance(
        case_id, tenant_id, "agent_2_segmentation", ["normalized_text.txt"]
    )

    document_graph = {
        **provenance,
        "nodes": nodes,
        "edges": edges,
    }

    save_json(case_dir / "document_graph.json", document_graph)

    return document_graph
