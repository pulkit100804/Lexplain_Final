"""
Agent 0 — Ingestion

Input:  raw case text (string or file path), tenant_id
Output: raw.txt, metadata.json
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import ensure_case_dir, save_json, build_provenance


def run(
    text: str,
    tenant_id: str,
    case_id: str | None = None,
) -> dict:
    """
    Ingest a raw case text.

    Parameters
    ----------
    text : str
        The raw legal case narrative.
    tenant_id : str
        Tenant identifier for multi-tenant isolation.
    case_id : str or None
        If provided, reloads into existing case directory.

    Returns
    -------
    dict
        metadata including case_id and tenant_id.
    """
    if case_id is None:
        case_id = str(uuid.uuid4())

    case_dir = ensure_case_dir(tenant_id, case_id)

    # Write raw text
    raw_path = case_dir / "raw.txt"
    raw_path.write_text(text, encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()

    # Build metadata
    metadata = {
        **build_provenance(case_id, tenant_id, "agent_0_ingestion", []),
        "char_count": len(text),
        "word_count": len(text.split()),
        "provenance": "user_input",
    }

    save_json(case_dir / "metadata.json", metadata)

    return metadata
