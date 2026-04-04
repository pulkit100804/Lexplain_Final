"""
Lexplain — Pipeline Orchestrator

Runs Agents 0 → 1 → 2 → 3 → 4A → 4B → 5A → 5B → 5C → 6 → 7 → 8 → 9 sequentially.
Agent 5D runs AFTER the main pipeline (post-processing).
Each step is independently callable.
"""

import sys
import time
import logging
from pathlib import Path

# Ensure the lexplain package root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from agents import (
    agent_0_ingestion,
    agent_1_normalization,
    agent_2_segmentation,
    agent_3_role_tagger,
    agent_4a_entity_extractor,
    agent_4b_event_builder,
    agent_5a_legal_fact_normalizer,
    agent_5b_legal_signal_extractor,
    agent_5c_statute_retriever,
    agent_6_ingredient_evaluator,
    agent_7_precedent_comparator,
    agent_8_loophole_miner,
    agent_9_final_argument,
    agent_5d_feedback_memory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("lexplain")

# Ordered agent list (main pipeline)
AGENTS = [
    ("Agent 0 — Ingestion", None),  # Special: needs text input
    ("Agent 1 — Normalization", agent_1_normalization),
    ("Agent 2 — Segmentation", agent_2_segmentation),
    ("Agent 3 — Role Tagging", agent_3_role_tagger),
    ("Agent 4A — Entity Extraction", agent_4a_entity_extractor),
    ("Agent 4B — Event Builder", agent_4b_event_builder),
    ("Agent 5A — Legal Fact Normalizer", agent_5a_legal_fact_normalizer),
    ("Agent 5B — Legal Signal Extractor", agent_5b_legal_signal_extractor),
    ("Agent 5C — Statute Retriever", agent_5c_statute_retriever),
    ("Agent 6 — Ingredient Evaluator", agent_6_ingredient_evaluator),
    ("Agent 7 — Precedent Comparator", agent_7_precedent_comparator),
    ("Agent 8 — Loophole Miner", agent_8_loophole_miner),
    ("Agent 9 — Final Argument Engine", None),  # Special: needs role param
]

AGENT_MAP = {
    "0": agent_0_ingestion,
    "1": agent_1_normalization,
    "2": agent_2_segmentation,
    "3": agent_3_role_tagger,
    "4a": agent_4a_entity_extractor,
    "4b": agent_4b_event_builder,
    "5a": agent_5a_legal_fact_normalizer,
    "5b": agent_5b_legal_signal_extractor,
    "5c": agent_5c_statute_retriever,
    "6": agent_6_ingredient_evaluator,
    "7": agent_7_precedent_comparator,
    "8": agent_8_loophole_miner,
    "9": agent_9_final_argument,
    "5d": agent_5d_feedback_memory,
}


def run_full_pipeline(
    text: str,
    tenant_id: str,
    role: str = "neutral",
) -> dict:
    """
    Run the full pipeline from raw text to final argument.

    Parameters
    ----------
    text : str
        Raw case narrative text.
    tenant_id : str
        Tenant identifier.
    role : str
        Argument role for Agent 9: prosecution / defence / neutral.

    Returns
    -------
    dict
        Final result including case_id and all outputs.
    """
    results = {}

    # Agent 0: Ingestion
    logger.info("▶ Agent 0 — Ingestion")
    t0 = time.time()
    metadata = agent_0_ingestion.run(text=text, tenant_id=tenant_id)
    case_id = metadata["case_id"]
    logger.info(f"  ✔ case_id={case_id} ({time.time() - t0:.2f}s)")
    results["agent_0"] = metadata

    # Agent 1–5C (standard agents)
    for name, agent_module in AGENTS[1:9]:
        logger.info(f"▶ {name}")
        t0 = time.time()
        result = agent_module.run(tenant_id=tenant_id, case_id=case_id)
        logger.info(f"  ✔ Done ({time.time() - t0:.2f}s)")
        agent_key = name.split("—")[0].strip().lower().replace(" ", "_")
        results[agent_key] = result

    # Agent 6: Ingredient Evaluator
    logger.info("▶ Agent 6 — Ingredient Evaluator")
    t0 = time.time()
    result = agent_6_ingredient_evaluator.run(tenant_id=tenant_id, case_id=case_id)
    logger.info(f"  ✔ Done ({time.time() - t0:.2f}s)")
    results["agent_6"] = result

    # Agent 7: Precedent Comparator
    logger.info("▶ Agent 7 — Precedent Comparator")
    t0 = time.time()
    result = agent_7_precedent_comparator.run(tenant_id=tenant_id, case_id=case_id)
    logger.info(f"  ✔ Done ({time.time() - t0:.2f}s)")
    results["agent_7"] = result

    # Agent 8: Loophole Miner
    logger.info("▶ Agent 8 — Loophole Miner")
    t0 = time.time()
    result = agent_8_loophole_miner.run(tenant_id=tenant_id, case_id=case_id)
    logger.info(f"  ✔ Done ({time.time() - t0:.2f}s)")
    results["agent_8"] = result

    # Agent 9: Final Argument Engine
    logger.info(f"▶ Agent 9 — Final Argument Engine (role={role})")
    t0 = time.time()
    result = agent_9_final_argument.run(
        tenant_id=tenant_id, case_id=case_id, role=role
    )
    logger.info(f"  ✔ Done ({time.time() - t0:.2f}s)")
    results["agent_9"] = result

    # Agent 5D: Feedback Memory (post-pipeline, separate)
    logger.info("▶ Agent 5D — Feedback Memory (post-pipeline)")
    t0 = time.time()
    result = agent_5d_feedback_memory.run(tenant_id=tenant_id, case_id=case_id)
    logger.info(f"  ✔ Done ({time.time() - t0:.2f}s)")
    results["agent_5d"] = result

    logger.info(f"✅ Pipeline complete for case {case_id}")
    return {"case_id": case_id, "tenant_id": tenant_id, "results": results}


def run_single_agent(
    agent_name: str,
    tenant_id: str,
    case_id: str,
    text: str | None = None,
    role: str = "neutral",
) -> dict:
    """
    Run a single agent independently.

    Parameters
    ----------
    agent_name : str
        Agent identifier (e.g., "0", "1", "3", "4a", "5c", "6", "7", "8", "9", "5d").
    tenant_id : str
        Tenant identifier.
    case_id : str
        Case identifier.
    text : str or None
        Required only for Agent 0.
    role : str
        Required only for Agent 9.

    Returns
    -------
    dict
        Agent output.
    """
    agent_module = AGENT_MAP.get(agent_name.lower())
    if agent_module is None:
        raise ValueError(f"Unknown agent: {agent_name}. Valid: {list(AGENT_MAP.keys())}")

    if agent_name == "0":
        if text is None:
            raise ValueError("Agent 0 requires text input")
        return agent_module.run(text=text, tenant_id=tenant_id, case_id=case_id)
    elif agent_name == "9":
        return agent_module.run(tenant_id=tenant_id, case_id=case_id, role=role)
    else:
        return agent_module.run(tenant_id=tenant_id, case_id=case_id)


from typing import Callable, Any

def run_full_pipeline_streaming(
    text: str,
    tenant_id: str,
    role: str = "neutral",
    on_step: Callable[[str, str, str], Any] | None = None,
) -> dict:
    """
    Run the full pipeline with a per-agent callback for real-time UI updates.

    Parameters
    ----------
    text : str
        Raw case narrative text.
    tenant_id : str
        Tenant identifier.
    role : str
        Argument role for Agent 9.
    on_step : callable(name, status, detail)
        Called before and after each agent:
          - name:   human-readable agent name
          - status: "running" | "done" | "error"
          - detail: short description of what happened

    Returns
    -------
    dict  — same format as run_full_pipeline
    """
    def _notify(name: str, status: str, detail: str = ""):
        if on_step:
            try:
                on_step(name, status, detail)
            except Exception:
                pass

    results = {}

    # ── Agent 0: Ingestion ──────────────────────────────────────────────────
    _notify("Agent 0 — Ingestion", "running", "Parsing and storing raw case text")
    t0 = time.time()
    metadata = agent_0_ingestion.run(text=text, tenant_id=tenant_id)
    case_id = metadata["case_id"]
    _notify("Agent 0 — Ingestion", "done", f"case_id = {case_id} ({time.time()-t0:.1f}s)")
    results["agent_0"] = metadata

    # ── Agents 1–5C ─────────────────────────────────────────────────────────
    for name, agent_module in AGENTS[1:9]:
        _notify(name, "running", f"Processing…")
        t0 = time.time()
        result = agent_module.run(tenant_id=tenant_id, case_id=case_id)
        elapsed = time.time() - t0
        detail = ""
        # derive a useful one-liner per agent
        if "1" in name:
            detail = "Text normalised and cleaned"
        elif "2" in name:
            detail = f"{len(result.get('segments', []))} segments created"
        elif "3" in name:
            detail = "Roles tagged (fact / allegation / evidence)"
        elif "4A" in name:
            detail = f"{len(result.get('entities', []))} entities extracted"
        elif "4B" in name:
            detail = f"{len(result.get('events', []))} events built"
        elif "5A" in name:
            detail = f"{len(result.get('legal_facts', []))} legal facts normalised"
        elif "5B" in name:
            detail = f"{len(result.get('signals', []))} legal signals found"
        elif "5C" in name:
            n = len(result.get("statute_candidates", {}))
            detail = f"{n} statute candidates retrieved"
        _notify(name, "done", f"{detail} ({elapsed:.1f}s)")
        agent_key = name.split("—")[0].strip().lower().replace(" ", "_")
        results[agent_key] = result

    # ── Agent 6 ─────────────────────────────────────────────────────────────
    _notify("Agent 6 — Ingredient Evaluator", "running",
            "Evaluating legal ingredients for each statute…")
    t0 = time.time()
    result = agent_6_ingredient_evaluator.run(tenant_id=tenant_id, case_id=case_id)
    valid_count = sum(1 for e in result.get("statute_evaluations", [])
                      if e.get("status") == "valid")
    _notify("Agent 6 — Ingredient Evaluator", "done",
            f"{valid_count} valid statute(s) ({time.time()-t0:.1f}s)")
    results["agent_6"] = result

    # ── Agent 7 ─────────────────────────────────────────────────────────────
    _notify("Agent 7 — Precedent Comparator", "running",
            "Searching judgment database for relevant precedents…")
    t0 = time.time()
    result = agent_7_precedent_comparator.run(tenant_id=tenant_id, case_id=case_id)
    n_cases = result.get("cases_after_filter", 0)
    _notify("Agent 7 — Precedent Comparator", "done",
            f"{n_cases} precedent(s) matched ({time.time()-t0:.1f}s)")
    results["agent_7"] = result

    # ── Agent 8 ─────────────────────────────────────────────────────────────
    _notify("Agent 8 — Loophole Miner", "running",
            "Mining missing ingredients and weak evidence…")
    t0 = time.time()
    result = agent_8_loophole_miner.run(tenant_id=tenant_id, case_id=case_id)
    n_loops = len(result.get("loopholes", []))
    _notify("Agent 8 — Loophole Miner", "done",
            f"{n_loops} loophole(s) identified ({time.time()-t0:.1f}s)")
    results["agent_8"] = result

    # ── Agent 9 ─────────────────────────────────────────────────────────────
    _notify("Agent 9 — Final Argument Engine", "running",
            f"Compiling final legal argument (role: {role})…")
    t0 = time.time()
    result = agent_9_final_argument.run(
        tenant_id=tenant_id, case_id=case_id, role=role
    )
    _notify("Agent 9 — Final Argument Engine", "done",
            f"Argument ready ({time.time()-t0:.1f}s)")
    results["agent_9"] = result

    # ── Agent 5D ────────────────────────────────────────────────────────────
    _notify("Agent 5D — Feedback Memory", "running",
            "Applying learned patterns from past feedback…")
    t0 = time.time()
    result = agent_5d_feedback_memory.run(tenant_id=tenant_id, case_id=case_id)
    _notify("Agent 5D — Feedback Memory", "done",
            f"Memory updated ({time.time()-t0:.1f}s)")
    results["agent_5d"] = result

    logger.info(f"✅ Streaming pipeline complete for case {case_id}")
    return {"case_id": case_id, "tenant_id": tenant_id, "results": results}

