"""
Agent 1 — Normalization

Input:  raw.txt
Output: normalized_text.txt

Deterministic, rule-based only.
- Normalize whitespace and punctuation
- Expand abbreviations
- Remove legal boilerplate
- Standardize references
- DO NOT remove factual content
- DO NOT summarize
"""

import re
from pathlib import Path

from config import get_case_dir, build_provenance, save_json


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Abbreviation dictionary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ABBREVIATIONS: dict[str, str] = {
    r"\bcomplt\.": "complaint",
    r"\bgovt\.": "government",
    r"\bsr\.": "senior",
    r"\bjr\.": "junior",
    r"\bpvt\.": "private",
    r"\bno\.": "number",
    r"\bsec\.": "section",
    r"\bdept\.": "department",
    r"\bapprox\.": "approximately",
    r"\bvs\.": "versus",
    r"\bv\.": "versus",
    r"\bw\.r\.t\.": "with respect to",
    r"\bi\.e\.": "that is",
    r"\be\.g\.": "for example",
    r"\betc\.": "and so on",
    r"\bdr\.": "doctor",
    r"\bmr\.": "mister",
    r"\bmrs\.": "mistress",
    r"\bms\.": "miss",
    r"\bpara\.": "paragraph",
    r"\bfir\b": "first information report",
    r"\bFIR\b": "First Information Report",
    r"\bio\b": "investigating officer",
    r"\bIO\b": "Investigating Officer",
    r"\bacp\b": "assistant commissioner of police",
    r"\bACP\b": "Assistant Commissioner of Police",
    r"\bdcp\b": "deputy commissioner of police",
    r"\bDCP\b": "Deputy Commissioner of Police",
    r"\bsho\b": "station house officer",
    r"\bSHO\b": "Station House Officer",
    r"\bps\b": "police station",
    r"\bPS\b": "Police Station",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Legal boilerplate phrases to remove
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BOILERPLATE_PATTERNS: list[str] = [
    r"\bherein\b",
    r"\bthereof\b",
    r"\bhereof\b",
    r"\btherein\b",
    r"\bhereby\b",
    r"\bthereby\b",
    r"\baforementioned\b",
    r"\baforesaid\b",
    r"\bhereinafter\b",
    r"\bhereinbefore\b",
    r"\bhereinabove\b",
    r"\bherewith\b",
    r"\bwhereof\b",
    r"\bwherein\b",
    r"\bwhereas\b",
    r"\bnotwithstanding\b",
    r"\binasmuch\b",
    r"\binsofar\b",
    r"\bthereunder\b",
    r"\bthereunto\b",
    r"\bhereunder\b",
    r"\bhereunto\b",
]


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace/newlines into single spaces."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_punctuation(text: str) -> str:
    """Normalize smart quotes and other punctuation oddities."""
    replacements = {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2026": "...",  # ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _expand_abbreviations(text: str) -> str:
    """Expand known abbreviations."""
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _remove_boilerplate(text: str) -> str:
    """Remove legal boilerplate words."""
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Clean up double spaces after removal
    text = re.sub(r"  +", " ", text)
    return text


def run(tenant_id: str, case_id: str) -> dict:
    """
    Normalize the raw case text.

    Returns
    -------
    dict
        Provenance metadata.
    """
    case_dir = get_case_dir(tenant_id, case_id)
    raw_path = case_dir / "raw.txt"
    text = raw_path.read_text(encoding="utf-8")

    # Pipeline of transformations
    text = _normalize_whitespace(text)
    text = _normalize_punctuation(text)
    text = _expand_abbreviations(text)
    text = _remove_boilerplate(text)
    # Final whitespace cleanup
    text = _normalize_whitespace(text)

    # Write output
    out_path = case_dir / "normalized_text.txt"
    out_path.write_text(text, encoding="utf-8")

    provenance = build_provenance(
        case_id, tenant_id, "agent_1_normalization", ["raw.txt"]
    )
    save_json(case_dir / "normalization_meta.json", provenance)

    return provenance
