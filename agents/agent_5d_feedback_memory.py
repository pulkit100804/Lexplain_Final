"""
Agent 5D — Feedback Memory

Runs AFTER Agent 9 (OUTSIDE core reasoning pipeline).
Stores feedback for future reference.
Agent 7 may use stored patterns as ranking boost ONLY.

SAFETY:
  - Does NOT modify core pipeline outputs
  - Stores structured feedback only
  - No LLM usage
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone

from config import (
    get_case_dir,
    load_json,
    save_json,
    build_provenance,
    DATA_DIR,
)

logger = logging.getLogger("lexplain.agent5d")

FEEDBACK_STORE_PATH = DATA_DIR / "feedback_memory.jsonl"


def _load_feedback_store() -> list[dict]:
    """Load all stored feedback entries."""
    if not FEEDBACK_STORE_PATH.exists():
        return []
    
    entries = []
    with open(FEEDBACK_STORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    return entries


def _append_feedback(entry: dict) -> None:
    """Append a feedback entry to the store."""
    with open(FEEDBACK_STORE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _extract_case_patterns(
    ingredient_report: dict,
    final_argument: dict,
) -> list[dict]:
    """Extract reusable patterns from the case for future retrieval boosting."""
    patterns = []
    
    for eval_item in ingredient_report.get("statute_evaluations", []):
        statute_id = eval_item.get("statute_id", "")
        status = eval_item.get("status", "")
        score = eval_item.get("overall_score", 0)
        
        # Extract failure patterns
        failures = []
        for ing in eval_item.get("ingredients", []):
            if ing.get("status") in ("not_satisfied", "partial"):
                failures.append({
                    "category": ing.get("ingredient_category", ""),
                    "status": ing.get("status", ""),
                    "evidence_type": ing.get("evidence_type", ""),
                })
        
        if failures:
            patterns.append({
                "statute_id": statute_id,
                "overall_status": status,
                "overall_score": score,
                "failure_patterns": failures,
            })
    
    return patterns


def get_boost_terms(query_terms: list[str]) -> list[str]:
    """
    Get ranking boost terms from feedback memory.
    Used by Agent 7 as OPTIONAL boost only.
    """
    feedback = _load_feedback_store()
    
    boost_terms = []
    query_lower = " ".join(t.lower() for t in query_terms)
    
    for entry in feedback[-50:]:  # Last 50 entries
        patterns = entry.get("patterns", [])
        for pattern in patterns:
            for fp in pattern.get("failure_patterns", []):
                cat = fp.get("category", "")
                if cat and cat in query_lower:
                    boost_terms.append(cat)
    
    return list(set(boost_terms))


def run(tenant_id: str, case_id: str) -> dict:
    """
    Run Agent 5D — Feedback Memory Writer.
    
    Reads: ingredient_report.json, final_argument.json
    Writes: feedback_memory.json (case-level), appends to data/feedback_memory.jsonl (global)
    """
    case_dir = get_case_dir(tenant_id, case_id)
    
    ingredient_report = load_json(case_dir / "ingredient_report.json")
    final_argument = load_json(case_dir / "final_argument.json")
    
    # Extract patterns
    patterns = _extract_case_patterns(ingredient_report, final_argument)
    
    # Build feedback entry
    feedback_entry = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": final_argument.get("role", "neutral"),
        "patterns": patterns,
        "loopholes_count": final_argument.get("loopholes_summary", {}).get("total", 0),
    }
    
    # Store globally
    _append_feedback(feedback_entry)
    
    # Store per-case
    provenance = build_provenance(
        case_id, tenant_id, "agent_5d_feedback_memory",
        ["ingredient_report.json", "final_argument.json"],
    )
    
    result = {
        **provenance,
        **feedback_entry,
    }
    
    save_json(case_dir / "feedback_memory.json", result)
    return result
