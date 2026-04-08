"""
Agent 7 — Precedent Comparator (RAG + Structured)

Input:  ingredient_report.json, legal_facts.json, legal_signals.json,
        judgment dataset (folder-based JSON)
Output: precedent_comparison.json

6-Step Process:
  1. Dynamic Query Builder  — from structured data ONLY (no section numbers, no labels)
  2. Retrieval              — BM25 search over pre-chunked judgment paragraphs
  3. Controlled Extraction  — LLM extracts ONLY tests/failure_reasons/rule/outcome
  4. Domain Filter          — discard irrelevant cases
  5. Structured Comparison  — compare ingredient failures with precedent patterns
  6. Output                 — structured comparison report

SAFETY:
  - Retrieval ≠ truth
  - NO summarization or new reasoning
  - ONLY extract existing judicial logic
  - LLM can reason but MUST NOT hallucinate
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
    GOOGLE_API_KEY,
    DEFAULT_MODEL,
    LLM_SAFETY_PROMPT,
)

from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

load_dotenv()

es = Elasticsearch(
    os.getenv("ELASTIC_HOST"),
    api_key=os.getenv("ELASTIC_API_KEY")
)

INDEX_NAME = "judgments_chunks"

logger = logging.getLogger("lexplain.agent7")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1 — Dynamic Query Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Forbidden query terms — no section numbers, no crime labels
FORBIDDEN_QUERY_TERMS = {
    "murder", "homicide", "fraud", "theft", "rape", "robbery",
    "arson", "kidnapping", "forgery", "assault",
    "ipc", "section", "302", "304", "300", "307", "420", "415",
    "guilty", "convicted", "criminal", "offence", "offense",
}


def _build_retrieval_query(
    ingredient_report: dict,
    legal_facts: list[dict],
    signals: list[dict],
) -> str:
    """
    Build a neutral factual query from structured data.
    INCLUDE: fact types, signal names, missing ingredients
    DO NOT INCLUDE: IPC section numbers, labels like 'murder', conclusions
    """
    terms = []
    
    # 1. Legal fact types
    for fact in legal_facts:
        ft = fact.get("type") or ""
        if ft and ft not in ("unknown_action", "no_action"):
            terms.append(ft.replace("_", " "))
    
    # 2. Signal names (cleaned)
    for sig in signals:
        sname = (sig.get("signal") or "").replace("_", " ")
        if sname:
            terms.append(sname)
    
    # 3. Missing / weak ingredients from Agent 6
    for eval_item in ingredient_report.get("statute_evaluations", []):
        for ing in eval_item.get("ingredients", []):
            if ing.get("status") in ("not_satisfied", "partial"):
                cat = ing.get("ingredient_category", "")
                if cat:
                    terms.append(f"{cat} missing")
                # Add ingredient keywords
                ing_text = ing.get("ingredient_text", "").lower()
                # Extract key phrases
                for phrase in ["intention", "knowledge", "causation", "death", "force",
                               "premeditation", "sudden fight", "bodily injury"]:
                    if phrase in ing_text:
                        terms.append(phrase)
    
    # Deduplicate and filter
    seen = set()
    filtered = []
    for t in terms:
        t_lower = t.lower().strip()
        if t_lower in seen:
            continue
        if any(fw in t_lower for fw in FORBIDDEN_QUERY_TERMS):
            continue
        if t_lower:
            seen.add(t_lower)
            filtered.append(t_lower)
    
    return " ".join(filtered)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2 — Retrieval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _retrieve_precedents(query: str, top_k: int = 10):
    res = es.search(
        index=INDEX_NAME,
        size=top_k,
        query={
            "match": {
                "text": query
            }
        }
    )

    results = []
    for hit in res["hits"]["hits"]:
        results.append({
            "text": hit["_source"]["text"],
            "case_name": hit["_source"].get("case_name", ""),
            "source_file": hit["_source"].get("source_file", "")
        })

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3 — Controlled Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_judicial_reasoning(case_text: str, case_name: str) -> dict | None:
    """
    Extract structured judicial reasoning using LLM.
    ONLY extracts: tests, failure_reasons, rule, outcome.
    NO summarization. NO new reasoning. NO inference.
    """
    if not GOOGLE_API_KEY or not case_text.strip():
        return None
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Truncate to fit context
        text_input = case_text[:4000]
        
        prompt = f"""Extract ONLY the following structured elements from this judicial text.
DO NOT summarize. DO NOT add reasoning. DO NOT infer anything.
ONLY extract what is EXPLICITLY stated in the text.

Text from case "{case_name}":
{text_input}

Output ONLY this JSON structure:
{{
  "tests": ["list of legal tests the court applied"],
  "failure_reasons": ["reasons the court identified for why charges failed or succeeded"],
  "rule": "the legal rule or principle stated by the court",
  "outcome": "conviction/acquittal/upheld/overturned/partial"
}}

If a field has no information in the text, use an empty list [] or empty string "".
Do NOT make up any content."""

        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                system_instruction=LLM_SAFETY_PROMPT,
            ),
        )
        
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        result = json.loads(text)
        
        # Validate structure
        if not isinstance(result, dict):
            return None
        if "tests" not in result:
            result["tests"] = []
        if "failure_reasons" not in result:
            result["failure_reasons"] = []
        if "rule" not in result:
            result["rule"] = ""
        if "outcome" not in result:
            result["outcome"] = ""
        
        return result
        
    except Exception as e:
        logger.warning(f"LLM extraction failed for {case_name}: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 4 — Domain Filter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Broad domain categories
DOMAIN_KEYWORDS = {
    "homicide": {"death", "killed", "murder", "homicide", "culpable", "fatal", "died"},
    "property": {"theft", "robbery", "cheating", "property", "misappropriation", "dacoity"},
    "violence": {"hurt", "injury", "grievous", "assault", "force", "wound", "weapon"},
    "sexual": {"rape", "modesty", "sexual", "consent"},
    "civil": {"contract", "landlord", "tenant", "civil", "revenue", "specific performance"},
    "service": {"employment", "service", "disciplinary", "termination", "dismissal"},
}


def _detect_case_domain(legal_facts: list[dict], signals: list[dict]) -> set[str]:
    """Detect the domain of the current case from facts/signals."""
    domains = set()
    
    all_text = " ".join(
        (f.get("type") or "") + " " + (f.get("original_action") or "")
        for f in legal_facts
    ) + " " + " ".join(
        (s.get("signal") or "") for s in signals
    )
    all_text = all_text.lower()
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in all_text for kw in keywords):
            domains.add(domain)
    
    return domains if domains else {"general"}


def _is_domain_relevant(case_text: str, case_domains: set[str]) -> bool:
    """Check if a retrieved case is in a relevant domain."""
    lower = case_text.lower()
    
    # If our case involves homicide, reject civil/service cases
    if "homicide" in case_domains or "violence" in case_domains:
        civil_indicators = {"contract", "landlord", "tenant", "specific performance",
                          "employment", "service law", "disciplinary"}
        if any(ci in lower for ci in civil_indicators) and not any(
            kw in lower for kw in {"death", "killed", "injury", "hurt", "weapon"}
        ):
            return False
    
    # If our case involves property/cheating, reject homicide-only cases
    if "property" in case_domains and "homicide" not in case_domains:
        if any(kw in lower for kw in {"murder", "homicide", "death sentence"}) and not any(
            kw in lower for kw in {"property", "cheating", "fraud", "theft"}
        ):
            return False
    
    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 5 — Structured Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compare_with_precedent(
    ingredient_report: dict,
    precedent_reasoning: dict,
    case_name: str,
    source_file: str,
) -> dict:
    """
    Compare current case ingredient failures with precedent reasoning.
    No text comparison. No narrative comparison.
    """
    matched_patterns = []
    differences = []
    
    # Get current case's failures and weaknesses
    current_failures = []
    current_weak = []
    for eval_item in ingredient_report.get("statute_evaluations", []):
        for ing in eval_item.get("ingredients", []):
            if ing.get("status") == "not_satisfied":
                current_failures.append({
                    "category": ing.get("ingredient_category", ""),
                    "text": ing.get("ingredient_text", ""),
                    "statute": eval_item.get("statute_id", ""),
                })
            elif ing.get("status") == "partial":
                current_weak.append({
                    "category": ing.get("ingredient_category", ""),
                    "text": ing.get("ingredient_text", ""),
                    "statute": eval_item.get("statute_id", ""),
                })
    
    # Compare with precedent failure reasons
    failure_reasons = precedent_reasoning.get("failure_reasons", [])
    tests = precedent_reasoning.get("tests", [])
    
    # Pattern matching: ingredient failures ↔ precedent failure_reasons
    for failure in current_failures:
        for reason in failure_reasons:
            reason_lower = reason.lower() if isinstance(reason, str) else ""
            cat = failure["category"]
            
            # Check if precedent has similar failure pattern
            if cat == "mens_rea" and any(kw in reason_lower for kw in
                ["intent", "intention", "knowledge", "mens rea", "premeditation"]):
                matched_patterns.append({
                    "pattern": f"Missing {cat} — precedent also found intention/knowledge issues",
                    "precedent_reference": case_name,
                    "why_matched": reason,
                })
            elif cat == "actus_reus" and any(kw in reason_lower for kw in
                ["causation", "act", "causing", "evidence", "proof"]):
                matched_patterns.append({
                    "pattern": f"Weak {cat} — precedent also found evidential issues",
                    "precedent_reference": case_name,
                    "why_matched": reason,
                })
    
    # Check if precedent tests match ingredient patterns
    for test in tests:
        test_lower = test.lower() if isinstance(test, str) else ""
        for failure in current_failures + current_weak:
            cat = failure["category"]
            if cat == "mens_rea" and any(kw in test_lower for kw in
                ["intent", "intention", "knowledge", "mental", "guilty mind"]):
                matched_patterns.append({
                    "pattern": f"Legal test on {cat} applied by court",
                    "precedent_reference": case_name,
                    "why_matched": test,
                })
    
    # Identify differences
    rule = precedent_reasoning.get("rule", "")
    outcome = precedent_reasoning.get("outcome", "")
    if rule:
        differences.append({
            "difference": f"Precedent rule: {rule[:200]}",
            "why_it_matters": "This rule may apply differently to current facts.",
        })
    if outcome:
        differences.append({
            "difference": f"Precedent outcome: {outcome}",
            "why_it_matters": "Compare with expected outcome of current case.",
        })
    
    # Compute simple similarity score
    total_possible = len(current_failures) + len(current_weak)
    if total_possible > 0:
        similarity = min(1.0, len(matched_patterns) / max(total_possible, 1))
    else:
        similarity = 0.0
    
    return {
        "case_name": case_name,
        "source_file": source_file,
        "similarity_score": round(similarity, 2),
        "matched_patterns": matched_patterns,
        "differences": differences,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(tenant_id: str, case_id: str) -> dict:
    """
    Run Agent 7 — Precedent Comparator.
    
    Reads: ingredient_report.json, legal_facts.json, legal_signals.json
    Writes: precedent_comparison.json
    """
    case_dir = get_case_dir(tenant_id, case_id)
    
    ingredient_report = load_json(case_dir / "ingredient_report.json")
    legal_facts_data = load_json(case_dir / "legal_facts.json")
    signals_data = load_json(case_dir / "legal_signals.json")
    
    legal_facts = legal_facts_data.get("legal_facts", [])
    signals = signals_data.get("signals", [])
    
    # Step 1: Build query
    query = _build_retrieval_query(ingredient_report, legal_facts, signals)
    logger.info(f"Agent 7 query: {query}")
    
    # Step 2: Retrieve precedents
    retrieved = _retrieve_precedents(query, top_k=10)
    
    # Detect case domain for filtering
    case_domains = _detect_case_domain(legal_facts, signals)
    
    # Group by source file (deduplicate chunks from same case)
    case_groups: dict[str, list[dict]] = {}
    for chunk in retrieved:
        sf = chunk.get("source_file", "unknown")
        if sf not in case_groups:
            case_groups[sf] = []
        case_groups[sf].append(chunk)
    
    # Steps 3-5: For each unique retrieved case
    all_patterns = []
    all_differences = []
    all_references = []
    overall_similarity = 0.0
    comparison_count = 0
    
    for source_file, chunks in list(case_groups.items())[:5]:  # Top 5 cases
        # Reconstruct case text
        case_text = " ".join(c.get("text", "") for c in chunks)
        case_name = chunks[0].get("case_name", source_file)
        
        # Step 4: Domain filter
        if not _is_domain_relevant(case_text, case_domains):
            continue
        
        # Step 3: Controlled extraction
        reasoning = _extract_judicial_reasoning(case_text, case_name)
        
        if reasoning is None:
            # Fallback: try to load full judgment and extract from structured fields
            from search import get_full_judgment
            full_judgment = get_full_judgment(source_file)
            if full_judgment:
                jr = full_judgment.get("judicial_reasoning", {})
                if isinstance(jr, dict):
                    reasoning = {
                        "tests": jr.get("legal_tests_applied", []),
                        "failure_reasons": jr.get("why_appellant_lost", []) + jr.get("why_respondent_won", []),
                        "rule": jr.get("core_finding", ""),
                        "outcome": full_judgment.get("decision", {}).get("appeal_result", ""),
                    }
        
        if reasoning is None:
            reasoning = {"tests": [], "failure_reasons": [], "rule": "", "outcome": ""}
        
        # Step 5: Structured comparison
        comparison = _compare_with_precedent(
            ingredient_report, reasoning, case_name, source_file
        )
        
        all_patterns.extend(comparison["matched_patterns"])
        all_differences.extend(comparison["differences"])
        overall_similarity += comparison["similarity_score"]
        comparison_count += 1
        
        # Add reference
        year = chunks[0].get("year", "Unknown")
        all_references.append({
            "case_name": case_name,
            "citation": f"{year}",
            "source": source_file,
        })
    
    # Compute average similarity
    avg_similarity = round(overall_similarity / max(comparison_count, 1), 2)
    
    # Build output
    provenance = build_provenance(
        case_id, tenant_id, "agent_7_precedent_comparator",
        ["ingredient_report.json", "legal_facts.json", "legal_signals.json"],
    )
    
    result = {
        **provenance,
        "query_used": query,
        "cases_retrieved": len(case_groups),
        "cases_after_filter": comparison_count,
        "similarity_score": avg_similarity,
        "matched_patterns": all_patterns,
        "differences": all_differences,
        "precedent_references": all_references,
    }
    
    save_json(case_dir / "precedent_comparison.json", result)
    return result
