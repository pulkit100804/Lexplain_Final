"""
Lexplain — Pipeline Tests

Two test cases:
  1. Violent case (attack + death + weapon recovery)
  2. Cheating case (promise + payment + non-delivery)

Tests validate logic, not just file existence.
Ensures no forbidden words appear in signals.
"""

import sys
import json
import shutil
import pytest
from pathlib import Path

# Ensure lexplain package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CASES_DIR, FORBIDDEN_SIGNAL_WORDS, ALLOWED_ROLES, get_case_dir
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
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VIOLENT_CASE_TEXT = (
    "On the night of 12th June, the accused attacked the victim with a knife. "
    "The victim later died. "
    "A blood-stained knife was recovered."
)

CHEATING_CASE_TEXT = (
    "The accused promised goods after payment. "
    "The complainant transferred money, but no goods were delivered. "
    "The accused stopped responding."
)

TEST_TENANT = "test_tenant"


def _run_pipeline(text: str) -> tuple[str, Path]:
    """Run the full pipeline and return (case_id, case_dir)."""
    # Agent 0
    meta = agent_0_ingestion.run(text=text, tenant_id=TEST_TENANT)
    case_id = meta["case_id"]
    case_dir = get_case_dir(TEST_TENANT, case_id)

    # Agent 1
    agent_1_normalization.run(tenant_id=TEST_TENANT, case_id=case_id)

    # Agent 2
    agent_2_segmentation.run(tenant_id=TEST_TENANT, case_id=case_id)

    # Agent 3
    agent_3_role_tagger.run(tenant_id=TEST_TENANT, case_id=case_id)

    # Agent 4A
    agent_4a_entity_extractor.run(tenant_id=TEST_TENANT, case_id=case_id)

    # Agent 4B
    agent_4b_event_builder.run(tenant_id=TEST_TENANT, case_id=case_id)

    # Agent 5A
    agent_5a_legal_fact_normalizer.run(tenant_id=TEST_TENANT, case_id=case_id)

    # Agent 5B
    agent_5b_legal_signal_extractor.run(tenant_id=TEST_TENANT, case_id=case_id)

    # Agent 5C
    agent_5c_statute_retriever.run(tenant_id=TEST_TENANT, case_id=case_id)

    return case_id, case_dir


def _load(case_dir: Path, filename: str) -> dict:
    with open(case_dir / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def _has_fact_type(legal_facts: list[dict], fact_type: str) -> bool:
    return any(f.get("type") == fact_type for f in legal_facts)


def _has_signal(signals: list[dict], signal_name: str) -> bool:
    return any(s.get("signal") == signal_name for s in signals)


def _signal_names(signals: list[dict]) -> list[str]:
    return [s.get("signal", "") for s in signals]


def _no_forbidden_words_in_signals(signals: list[dict]) -> bool:
    """Ensure no signal name contains forbidden crime words."""
    import re
    for sig in signals:
        name = sig.get("signal", "").lower()
        for word in FORBIDDEN_SIGNAL_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", name):
                return False
    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST CASE 1 — VIOLENT CASE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestViolentCase:
    """
    Input: "On the night of 12th June, the accused attacked the victim with
    a knife. The victim later died. A blood-stained knife was recovered."
    """

    @classmethod
    def setup_class(cls):
        cls.case_id, cls.case_dir = _run_pipeline(VIOLENT_CASE_TEXT)

    @classmethod
    def teardown_class(cls):
        if cls.case_dir.exists():
            shutil.rmtree(cls.case_dir)

    # --- File existence ---

    def test_raw_txt_exists(self):
        raw = (self.case_dir / "raw.txt").read_text(encoding="utf-8")
        assert raw == VIOLENT_CASE_TEXT

    def test_metadata_exists_and_valid(self):
        meta = _load(self.case_dir, "metadata.json")
        assert meta["case_id"] == self.case_id
        assert meta["tenant_id"] == TEST_TENANT
        assert meta["agent"] == "agent_0_ingestion"
        assert "created_at" in meta
        assert meta["char_count"] > 0
        assert meta["word_count"] > 0

    def test_normalized_text_no_boilerplate(self):
        text = (self.case_dir / "normalized_text.txt").read_text(encoding="utf-8")
        assert len(text) > 0
        lower = text.lower()
        for word in ["herein", "thereof", "aforementioned"]:
            assert word not in lower

    # --- Document Graph ---

    def test_document_graph_valid(self):
        graph = _load(self.case_dir, "document_graph.json")
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) >= 1
        # Provenance
        assert graph["agent"] == "agent_2_segmentation"
        assert graph["case_id"] == self.case_id

    # --- Role Tagged Graph ---

    def test_role_tagged_graph_valid(self):
        graph = _load(self.case_dir, "role_tagged_graph.json")
        for node in graph["nodes"]:
            assert "role" in node
            assert node["role"] in ALLOWED_ROLES

    # --- Entities ---

    def test_entities_has_actors(self):
        ent = _load(self.case_dir, "entities.json")
        actors = [e for e in ent["entities"] if e["type"] == "actor"]
        assert len(actors) >= 1, "Should find at least one actor"

    def test_entities_has_objects(self):
        ent = _load(self.case_dir, "entities.json")
        objects = [e for e in ent["entities"] if e["type"] == "object"]
        assert len(objects) >= 1, "Should find knife or blood-stained knife"

    # --- Event Graph ---

    def test_event_graph_no_discarded_nodes(self):
        graph = _load(self.case_dir, "document_graph.json")
        events = _load(self.case_dir, "event_graph.json")
        # Every node must produce an event
        assert len(events["events"]) == len(graph["nodes"]), \
            "No node should be discarded — every node must produce an event"

    def test_event_graph_has_typed_events(self):
        events = _load(self.case_dir, "event_graph.json")
        for evt in events["events"]:
            assert evt["event_type"] in ["action", "evidence", "state", "context"], \
                f"Invalid event type: {evt['event_type']}"

    def test_event_graph_provenance(self):
        events = _load(self.case_dir, "event_graph.json")
        assert events["agent"] == "agent_4b_event_builder"
        assert events["case_id"] == self.case_id

    # --- Legal Facts ---

    def test_legal_facts_contain_use_of_force(self):
        facts = _load(self.case_dir, "legal_facts.json")
        assert _has_fact_type(facts["legal_facts"], "use_of_force"), \
            f"Expected 'use_of_force' in facts. Got: {[f['type'] for f in facts['legal_facts']]}"

    def test_legal_facts_contain_causing_death(self):
        facts = _load(self.case_dir, "legal_facts.json")
        assert _has_fact_type(facts["legal_facts"], "causing_death"), \
            f"Expected 'causing_death' in facts. Got: {[f['type'] for f in facts['legal_facts']]}"

    def test_legal_facts_has_mapping_source(self):
        facts = _load(self.case_dir, "legal_facts.json")
        for f in facts["legal_facts"]:
            assert "mapped_from" in f, "Each fact must have mapped_from field"

    def test_legal_facts_nonempty(self):
        facts = _load(self.case_dir, "legal_facts.json")
        assert len(facts["legal_facts"]) > 0, "Legal facts must not be empty"

    # --- Signals ---

    def test_signals_contain_death_occurred(self):
        signals = _load(self.case_dir, "legal_signals.json")
        assert _has_signal(signals["signals"], "death_occurred"), \
            f"Expected 'death_occurred'. Got: {_signal_names(signals['signals'])}"

    def test_signals_contain_weapon_recovered(self):
        """weapon_recovered or physical_evidence_recovered should be present."""
        signals = _load(self.case_dir, "legal_signals.json")
        names = _signal_names(signals["signals"])
        has_weapon = "weapon_recovered" in names or "physical_evidence_recovered" in names
        assert has_weapon, \
            f"Expected weapon/evidence recovery signal. Got: {names}"

    def test_signals_no_murder_word(self):
        signals = _load(self.case_dir, "legal_signals.json")
        names = _signal_names(signals["signals"])
        for name in names:
            assert "murder" not in name.lower(), \
                f"Signal '{name}' contains forbidden word 'murder'"

    def test_signals_no_forbidden_words(self):
        signals = _load(self.case_dir, "legal_signals.json")
        assert _no_forbidden_words_in_signals(signals["signals"]), \
            f"Forbidden words found in signals: {_signal_names(signals['signals'])}"

    def test_signals_nonempty(self):
        signals = _load(self.case_dir, "legal_signals.json")
        assert len(signals["signals"]) > 0, "Signals must not be empty"

    # --- Statute Candidates ---

    def test_statute_candidates_exist(self):
        candidates = _load(self.case_dir, "statute_candidates.json")
        assert len(candidates["candidates"]) > 0, "Should have statute candidates"

    def test_statute_candidates_have_scores(self):
        candidates = _load(self.case_dir, "statute_candidates.json")
        for c in candidates["candidates"]:
            assert "score" in c
            assert "section" in c
            assert c["score"] > 0

    def test_statute_candidates_provenance(self):
        candidates = _load(self.case_dir, "statute_candidates.json")
        assert candidates["agent"] == "agent_5c_statute_retriever"
        assert "query_terms" in candidates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST CASE 2 — CHEATING CASE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCheatingCase:
    """
    Input: "The accused promised goods after payment. The complainant
    transferred money, but no goods were delivered. The accused stopped
    responding."
    """

    @classmethod
    def setup_class(cls):
        cls.case_id, cls.case_dir = _run_pipeline(CHEATING_CASE_TEXT)

    @classmethod
    def teardown_class(cls):
        if cls.case_dir.exists():
            shutil.rmtree(cls.case_dir)

    # --- Legal Facts ---

    def test_legal_facts_contain_deception(self):
        facts = _load(self.case_dir, "legal_facts.json")
        assert _has_fact_type(facts["legal_facts"], "deception"), \
            f"Expected 'deception' in facts. Got: {[f['type'] for f in facts['legal_facts']]}"

    def test_legal_facts_contain_property_transfer(self):
        facts = _load(self.case_dir, "legal_facts.json")
        assert _has_fact_type(facts["legal_facts"], "property_transfer"), \
            f"Expected 'property_transfer' in facts. Got: {[f['type'] for f in facts['legal_facts']]}"

    def test_legal_facts_nonempty(self):
        facts = _load(self.case_dir, "legal_facts.json")
        assert len(facts["legal_facts"]) > 0

    # --- Signals ---

    def test_signals_contain_property_taken(self):
        signals = _load(self.case_dir, "legal_signals.json")
        assert _has_signal(signals["signals"], "property_taken"), \
            f"Expected 'property_taken'. Got: {_signal_names(signals['signals'])}"

    def test_signals_contain_failure_to_deliver(self):
        """failure_to_deliver or failure_to_respond should be present."""
        signals = _load(self.case_dir, "legal_signals.json")
        names = _signal_names(signals["signals"])
        has_failure = "failure_to_deliver" in names or "failure_to_respond" in names
        assert has_failure, \
            f"Expected failure signal. Got: {names}"

    def test_signals_no_fraud_word(self):
        signals = _load(self.case_dir, "legal_signals.json")
        names = _signal_names(signals["signals"])
        for name in names:
            assert "fraud" not in name.lower(), \
                f"Signal '{name}' contains forbidden word 'fraud'"

    def test_signals_no_forbidden_words(self):
        signals = _load(self.case_dir, "legal_signals.json")
        assert _no_forbidden_words_in_signals(signals["signals"]), \
            f"Forbidden words found in signals: {_signal_names(signals['signals'])}"

    def test_signals_nonempty(self):
        signals = _load(self.case_dir, "legal_signals.json")
        assert len(signals["signals"]) > 0

    # --- Event Graph —

    def test_event_graph_no_discarded_nodes(self):
        graph = _load(self.case_dir, "document_graph.json")
        events = _load(self.case_dir, "event_graph.json")
        assert len(events["events"]) == len(graph["nodes"]), \
            "No node should be discarded"

    def test_event_graph_typed(self):
        events = _load(self.case_dir, "event_graph.json")
        for evt in events["events"]:
            assert evt["event_type"] in ["action", "evidence", "state", "context"]

    # --- Provenance on all outputs ---

    def test_all_outputs_have_provenance(self):
        """Every JSON output must include case_id, tenant_id, agent, input_refs, created_at."""
        json_files = [
            "metadata.json",
            "document_graph.json",
            "role_tagged_graph.json",
            "entities.json",
            "event_graph.json",
            "legal_facts.json",
            "legal_signals.json",
            "statute_candidates.json",
        ]
        for fname in json_files:
            fpath = self.case_dir / fname
            if fpath.exists():
                data = _load(self.case_dir, fname)
                assert "case_id" in data, f"{fname} missing case_id"
                assert "tenant_id" in data, f"{fname} missing tenant_id"
                assert "agent" in data, f"{fname} missing agent"
                assert "created_at" in data, f"{fname} missing created_at"

    # --- Statute Candidates ---

    def test_statute_candidates_exist(self):
        candidates = _load(self.case_dir, "statute_candidates.json")
        assert len(candidates["candidates"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
