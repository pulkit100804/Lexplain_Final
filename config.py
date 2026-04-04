"""
Lexplain — Shared configuration and utilities.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paths
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASE_DIR = Path(__file__).parent
CASES_DIR = BASE_DIR / "cases"
DATA_DIR = BASE_DIR / "data"
IPC_INGREDIENTS_PATH = DATA_DIR / "ingredients_ipc.json"
IPC_INGREDIENTS_SIMPLE_PATH = DATA_DIR / "ipc_ingredients.json"
JUDGMENTS_DIR = DATA_DIR / "Judgements"
CHUNKS_FILE = DATA_DIR / "paragraph_chunks.jsonl"

# Core ingredients — if ANY is NOT_SATISFIED → overall_score = 0
CORE_INGREDIENT_TYPES = {"actus_reus", "mens_rea"}


def get_case_dir(tenant_id: str, case_id: str) -> Path:
    """Return the case directory for a given tenant and case."""
    return CASES_DIR / tenant_id / case_id


def ensure_case_dir(tenant_id: str, case_id: str) -> Path:
    """Ensure the case directory exists and return it."""
    case_dir = get_case_dir(tenant_id, case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def get_human_feedback_patterns(tenant_id: str) -> str:
    """Retrieve human feedback patterns to inject into LLM prompts."""
    feedback_file = DATA_DIR / "feedback_store.json"
    if not feedback_file.exists():
        return ""
    try:
        with open(feedback_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        patterns = []
        for entry in data:
            if entry.get("tenant_id") == tenant_id:
                for pat in entry.get("patterns_learned", []):
                    comp = pat.get("component", "general")
                    text = pat.get("pattern", "")
                    if text and text != "confirmed_correct":
                        patterns.append(f"[{comp.upper()}] {text}")
        if patterns:
            unique_patterns = list(dict.fromkeys(patterns))
            return "\n".join(f"- {p}" for p in unique_patterns)
    except Exception:
        pass
    return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON I/O
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_json(path: Path) -> dict:
    """Load JSON from a file path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """Save data as JSON to a file path."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Provenance header builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_provenance(
    case_id: str,
    tenant_id: str,
    agent: str,
    input_refs: list[str],
) -> dict:
    """Build the provenance header that MUST be included in every agent output JSON."""
    return {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "agent": agent,
        "input_refs": input_refs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALLOWED_ROLES = [
    "fact",
    "allegation",
    "evidence",
    "witness_statement",
    "procedural",
    "legal_claim",
    "background",
]

ENTITY_TYPES = ["actor", "object", "location", "time"]

EVENT_TYPES = ["action", "evidence", "state", "context"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Legal Fact Abstraction Mappings (base)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEGAL_FACT_BASE_MAP: dict[str, str] = {
    "stabbed": "use_of_force",
    "shot": "use_of_force",
    "hit": "use_of_force",
    "beat": "use_of_force",
    "assaulted": "use_of_force",
    "attacked": "use_of_force",
    "punched": "use_of_force",
    "kicked": "use_of_force",
    "slapped": "use_of_force",
    "struck": "use_of_force",
    "killed": "causing_death",
    "murdered": "causing_death",
    "caused death": "causing_death",
    "died": "causing_death",
    "took": "property_transfer",
    "stole": "property_transfer",
    "snatched": "property_transfer",
    "robbed": "property_transfer",
    "transferred": "property_transfer",
    "lied": "deception",
    "deceived": "deception",
    "misrepresented": "deception",
    "cheated": "deception",
    "promised": "deception",
    "threatened": "criminal_intimidation",
    "intimidated": "criminal_intimidation",
    "entered": "unlawful_entry",
    "trespassed": "unlawful_entry",
    "broke in": "unlawful_entry",
    "forged": "forgery",
    "fabricated": "forgery",
    "falsified": "forgery",
    "set fire": "destruction_by_fire",
    "burned": "destruction_by_fire",
    "conspired": "conspiracy",
    "planned together": "conspiracy",
    "abducted": "unlawful_restraint",
    "kidnapped": "unlawful_restraint",
    "confined": "unlawful_restraint",
    "detained": "unlawful_restraint",
    "stopped responding": "non_cooperation",
    "fled": "absconding",
    "escaped": "absconding",
    "recovered": "evidence_recovery",
    "found": "evidence_recovery",
    "seized": "evidence_recovery",
}

# Synonym expansion table — maps synonyms to canonical base map keys
LEGAL_FACT_SYNONYMS: dict[str, str] = {
    # use_of_force synonyms
    "battered": "assaulted",
    "thrashed": "beat",
    "wounded": "stabbed",
    "injured": "hit",
    "harmed": "hit",
    "mauled": "attacked",
    "chopped": "stabbed",
    "hacked": "stabbed",
    "slashed": "stabbed",
    "bludgeoned": "hit",
    "strangled": "attacked",
    "throttled": "attacked",
    "pushed": "hit",
    "shoved": "hit",
    # causing_death synonyms
    "slain": "killed",
    "slaughtered": "killed",
    "executed": "killed",
    "eliminated": "killed",
    "succumbed": "died",
    "perished": "died",
    "expired": "died",
    # property_transfer synonyms
    "looted": "robbed",
    "pilfered": "stole",
    "embezzled": "stole",
    "misappropriated": "stole",
    "grabbed": "snatched",
    "seized property": "snatched",
    "dispossessed": "took",
    # deception synonyms
    "duped": "deceived",
    "conned": "cheated",
    "defrauded": "cheated",
    "tricked": "deceived",
    "hoodwinked": "deceived",
    "swindled": "cheated",
    "falsely promised": "promised",
    # criminal_intimidation synonyms
    "menaced": "threatened",
    "coerced": "intimidated",
    "bullied": "intimidated",
    "blackmailed": "threatened",
    "extorted": "threatened",
    # unlawful_entry synonyms
    "intruded": "trespassed",
    "broke into": "broke in",
    "invaded": "entered",
    "forced entry": "broke in",
    # forgery synonyms
    "counterfeited": "forged",
    "tampered": "fabricated",
    "doctored": "fabricated",
    "altered": "fabricated",
    # destruction_by_fire synonyms
    "torched": "set fire",
    "ignited": "set fire",
    "incinerated": "burned",
    "blazed": "burned",
    # conspiracy synonyms
    "colluded": "conspired",
    "plotted": "conspired",
    "schemed": "conspired",
    # unlawful_restraint synonyms
    "captured": "abducted",
    "imprisoned": "confined",
    "held captive": "confined",
    "restrained": "detained",
    "held hostage": "abducted",
}

# Forbidden crime words that must NOT appear in legal signals
FORBIDDEN_SIGNAL_WORDS = {
    "murder", "murdered", "homicide",
    "fraud", "defraud", "defrauded",
    "theft", "thief",
    "robbery", "robbing",
    "rape", "raped",
    "assault",  # as crime label, not observation
    "arson",
    "kidnapping",
    "forgery",
    "committed",
    "guilty",
    "convicted",
    "crime", "criminal",
    "offence", "offense",
    "culprit",
    "perpetrator",
}

# LLM safety system prompt
LLM_SAFETY_PROMPT = """You are a legal document classifier. You MUST follow these rules:
- DO NOT invent facts
- DO NOT infer intent
- DO NOT assume missing information
- ONLY use explicitly provided input
- If unsure → return conservative output
- NEVER output legal conclusions (e.g., murder, fraud, theft, guilty)
Violation = invalid output."""

from dotenv import load_dotenv
# Look for .env in current dir and parent dir
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# Model configuration
DEFAULT_MODEL = "gemini-2.5-flash"


# Google API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
