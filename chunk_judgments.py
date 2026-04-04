"""
Lexplain — Judgment Chunker (Adapted)

Reads judgment JSON files from data/Judgements/ and produces
paragraph_chunks.jsonl for local BM25 retrieval.

Usage:
    python chunk_judgments.py         # chunk all judgments
    python chunk_judgments.py --check # just count files
"""

import json
import os
import re
import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import JUDGMENTS_DIR, CHUNKS_FILE


# -------- HELPER: SAFE JSON PARSER --------
def try_parse_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


# -------- CLEAN TEXT --------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\\n", " ").replace("\n", " ").replace('\\"', '"')
    text = " ".join(text.split())
    if text.startswith("{") or text.startswith("["):
        return ""
    if text.lower().startswith("issues:"):
        return ""
    if len(text) < 40:
        return ""
    return text


# -------- CORE EXTRACTION --------
def extract_text(data):
    """Extract all meaningful text from a judgment JSON."""
    parts = []
    data = try_parse_json(data)

    # 1. PROCEDURAL HISTORY
    ph = try_parse_json(data.get("procedural_history", {}))
    if isinstance(ph, dict):
        for v in ph.values():
            v = try_parse_json(v)
            if isinstance(v, str):
                parts.append(v)

    # 2. ISSUES + REASONING
    issues = try_parse_json(data.get("issues", []))
    for item in issues:
        item = try_parse_json(item)
        if isinstance(item, dict):
            issue_block = try_parse_json(item.get("issue", {}))
            if isinstance(issue_block, dict):
                issue_text = issue_block.get("issue_text")
                if isinstance(issue_text, str):
                    parts.append(issue_text)

                reasoning = try_parse_json(issue_block.get("reasoning", {}))
                if isinstance(reasoning, dict):
                    for section in ["facts", "law", "application"]:
                        for val in reasoning.get(section, []):
                            val = try_parse_json(val)
                            if isinstance(val, str):
                                parts.append(val)

                conclusion = issue_block.get("conclusion")
                if isinstance(conclusion, str):
                    parts.append(conclusion)

    # 3. GLOBAL REASONING
    reasoning = try_parse_json(data.get("reasoning", {}))
    if isinstance(reasoning, dict):
        for value in reasoning.values():
            value = try_parse_json(value)
            if isinstance(value, list):
                for item in value:
                    item = try_parse_json(item)
                    if isinstance(item, str):
                        parts.append(item)
            elif isinstance(value, str):
                parts.append(value)

    # 4. LAW
    law = try_parse_json(data.get("law", []))
    for item in law:
        item = try_parse_json(item)
        if isinstance(item, dict):
            fact = item.get("fact")
            if isinstance(fact, str):
                parts.append(fact)

    # 5. CONCLUSION
    conclusion = data.get("conclusion")
    if isinstance(conclusion, str):
        parts.append(conclusion)

    # 6. JUDICIAL REASONING (CRITICAL for Agent 7)
    jr = try_parse_json(data.get("judicial_reasoning", {}))
    if isinstance(jr, dict):
        core = jr.get("core_finding")
        if isinstance(core, str):
            parts.append(core)
        tests = jr.get("legal_tests_applied", [])
        for t in tests:
            if isinstance(t, str):
                parts.append(t)
        for field in ["why_appellant_lost", "why_respondent_won", "policy_considerations"]:
            vals = jr.get(field, [])
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str):
                        parts.append(v)

    # 7. RATIO DECIDENDI
    rd = try_parse_json(data.get("ratio_decidendi", {}))
    if isinstance(rd, dict):
        ratio = rd.get("ratio")
        if isinstance(ratio, str):
            parts.append(ratio)
        obiter = rd.get("obiter", [])
        for o in obiter:
            if isinstance(o, str):
                parts.append(o)

    # CLEAN ALL
    cleaned_parts = [p for p in (clean_text(p) for p in parts) if p]
    return "\n".join(cleaned_parts)


# -------- PARAGRAPH SPLIT --------
def split_into_paragraphs(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s.lower().startswith("whether") and len(s) < 80:
            continue
        if len(s) > 40:
            cleaned.append(s)
    return cleaned


# -------- EXTRACT STRUCTURED METADATA --------
def extract_metadata(data):
    """Extract structured metadata from judgment for domain filtering."""
    data = try_parse_json(data)
    metadata = {}
    
    # Citations / statutes mentioned
    citations = data.get("citations", {})
    if isinstance(citations, dict):
        statutes = citations.get("statutes", [])
        metadata["statute_sections"] = [
            s.get("section", "") for s in statutes if isinstance(s, dict)
        ]
        cases = citations.get("cases", [])
        metadata["cited_cases"] = [
            {"name": c.get("case_name", ""), "citation": c.get("citation", "")}
            for c in cases if isinstance(c, dict)
        ]
    
    # Decision outcome
    decision = data.get("decision", {})
    if isinstance(decision, dict):
        metadata["appeal_result"] = decision.get("appeal_result", "")
        metadata["conviction_status"] = decision.get("conviction_status", "")
    
    return metadata

def _extract_case_name_fallback(data: dict, file_path: str) -> str:
    """
    Attempt to extract a meaningful case name from the judgment JSON
    when the top-level 'title' key is absent.
    Checks (in order):
      1. citations.cases[0].case_name
      2. procedural_history.impugned_order (first sentence)
      3. filename without extension
    """
    # 1. First cited case
    try:
        cases = data.get("citations", {}).get("cases", [])
        if cases and isinstance(cases[0], dict):
            name = cases[0].get("case_name", "")
            if name:
                return name
    except Exception:
        pass

    # 2. Procedural history impugned_order — first sentence may contain case ref
    try:
        ph = data.get("procedural_history", {})
        impugned = ph.get("impugned_order", "") if isinstance(ph, dict) else ""
        if impugned:
            # grab first clause before "of the"
            fragment = impugned.split(" of the ")[0]
            if len(fragment) < 100:
                return fragment.strip()
    except Exception:
        pass

    # 3. Filename
    return str(Path(file_path).stem).replace("_", " ").replace("EN", "").strip()


# -------- PROCESS FILE --------
def process_judgment(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = extract_text(data)
    if not text:
        return []

    case_name = (
        data.get("title")
        or data.get("case_name")
        or _extract_case_name_fallback(data, file_path)
    )
    year = data.get("year") or "Unknown"
    metadata = extract_metadata(data)

    paragraphs = split_into_paragraphs(text)

    chunks = []
    for i, p in enumerate(paragraphs):
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "source_file": os.path.basename(file_path),
            "case_name": case_name,
            "year": year,
            "paragraph_id": i,
            "text": p,
            "metadata": metadata,
        })
    return chunks


# -------- MAIN --------
def chunk_all_judgments():
    """Chunk all judgment files from JUDGMENTS_DIR → CHUNKS_FILE."""
    total_files = 0
    total_chunks = 0

    with open(CHUNKS_FILE, "w", encoding="utf-8") as out:
        for root, _, files in os.walk(JUDGMENTS_DIR):
            for file in files:
                if not file.endswith(".json"):
                    continue

                file_path = os.path.join(root, file)
                total_files += 1

                chunks = process_judgment(file_path)
                for c in chunks:
                    out.write(json.dumps(c, ensure_ascii=False) + "\n")

                total_chunks += len(chunks)

                if total_files % 50 == 0:
                    print(f"  Processed {total_files} files...")

    print(f"\n✅ Chunking complete: {total_files} files → {total_chunks} chunks")
    print(f"   Output: {CHUNKS_FILE}")
    return total_files, total_chunks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chunk judgment files for RAG")
    parser.add_argument("--check", action="store_true", help="Just count files")
    args = parser.parse_args()

    if args.check:
        count = sum(
            1 for root, _, files in os.walk(JUDGMENTS_DIR)
            for f in files if f.endswith(".json")
        )
        print(f"Found {count} judgment files in {JUDGMENTS_DIR}")
    else:
        chunk_all_judgments()


if __name__ == "__main__":
    main()