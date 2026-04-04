"""
Lexplain — Local Judgment Search (BM25)

Provides local BM25-based search over paragraph_chunks.jsonl
for Agent 7's precedent retrieval. No Elasticsearch required.

Also provides optional ES search if available.
"""

import json
import re
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CHUNKS_FILE, JUDGMENTS_DIR

logger = logging.getLogger("lexplain.search")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _load_chunks() -> list[dict]:
    """Load all paragraph chunks from JSONL file."""
    if not CHUNKS_FILE.exists():
        logger.warning(f"Chunks file not found: {CHUNKS_FILE}. Run chunk_judgments.py first.")
        return []
    
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except Exception:
                    continue
    return chunks


# Cache loaded chunks
_CHUNKS_CACHE: list[dict] | None = None


def _get_chunks() -> list[dict]:
    global _CHUNKS_CACHE
    if _CHUNKS_CACHE is None:
        _CHUNKS_CACHE = _load_chunks()
    return _CHUNKS_CACHE


def search_judgments_bm25(query: str, top_k: int = 10) -> list[dict]:
    """
    Search judgment chunks using BM25.
    
    Returns list of dicts with: chunk_id, source_file, case_name, year, text, score, metadata.
    """
    chunks = _get_chunks()
    
    if not chunks:
        # Fallback: scan judgment files directly
        return _search_judgments_direct(query, top_k)
    
    from rank_bm25 import BM25Okapi
    
    tokenized_query = _tokenize(query)
    corpus = [_tokenize(c.get("text", "")) for c in chunks]
    
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized_query)
    
    # Rank and return top_k
    scored = []
    for idx, score in enumerate(scores):
        if score > 0:
            chunk = chunks[idx].copy()
            chunk["score"] = float(score)
            scored.append(chunk)
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _search_judgments_direct(query: str, top_k: int = 10) -> list[dict]:
    """
    Fallback: scan raw judgment files directly if no chunks file exists.
    Uses simple keyword overlap scoring.
    """
    query_tokens = set(_tokenize(query))
    results = []
    
    if not JUDGMENTS_DIR.exists():
        return []
    
    for root, _, files in os.walk(JUDGMENTS_DIR):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Build searchable text from key fields
                parts = []
                jr = data.get("judicial_reasoning", {})
                if isinstance(jr, dict):
                    parts.append(jr.get("core_finding", ""))
                    parts.extend(jr.get("legal_tests_applied", []))
                    parts.extend(jr.get("why_appellant_lost", []))
                    parts.extend(jr.get("why_respondent_won", []))
                
                rd = data.get("ratio_decidendi", {})
                if isinstance(rd, dict):
                    parts.append(rd.get("ratio", ""))
                
                full_text = " ".join(str(p) for p in parts if p)
                tokens = set(_tokenize(full_text))
                
                overlap = len(query_tokens & tokens)
                if overlap > 0:
                    results.append({
                        "source_file": fname,
                        "case_name": data.get("title", fname),
                        "year": data.get("year", "Unknown"),
                        "text": full_text[:2000],
                        "score": overlap / max(len(query_tokens), 1),
                        "metadata": {
                            "appeal_result": data.get("decision", {}).get("appeal_result", ""),
                        },
                    })
            except Exception:
                continue
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def get_full_judgment(source_file: str) -> dict | None:
    """Load the full judgment JSON given its source filename."""
    if not JUDGMENTS_DIR.exists():
        return None
    
    for root, _, files in os.walk(JUDGMENTS_DIR):
        if source_file in files:
            fpath = os.path.join(root, source_file)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
    return None


def get_full_case_text_from_chunks(source_file: str) -> str:
    """Reconstruct full case text from chunks."""
    chunks = _get_chunks()
    case_chunks = [c for c in chunks if c.get("source_file") == source_file]
    case_chunks.sort(key=lambda x: x.get("paragraph_id", 0))
    return " ".join(c.get("text", "") for c in case_chunks)