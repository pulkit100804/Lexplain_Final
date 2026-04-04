"""
Lexplain MCP — Tool Registry

Maps tool names to agent handler functions.
Each tool has: name, description, input_schema, handler.
JSON-only communication. Trace IDs via provenance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOOL_REGISTRY = {
    "ingestor": {
        "name": "ingestor",
        "description": "Agent 0 — Ingests raw case text and creates case directory",
        "agent": "agent_0_ingestion",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Raw case narrative text"},
                "tenant_id": {"type": "string", "description": "Tenant identifier"},
                "case_id": {"type": "string", "description": "Optional case ID"},
            },
            "required": ["text", "tenant_id"],
        },
        "handler": lambda params: agent_0_ingestion.run(
            text=params["text"],
            tenant_id=params["tenant_id"],
            case_id=params.get("case_id"),
        ),
    },
    "normalizer": {
        "name": "normalizer",
        "description": "Agent 1 — Normalizes raw text (remove boilerplate, clean formatting)",
        "agent": "agent_1_normalization",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_1_normalization.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "segmenter": {
        "name": "segmenter",
        "description": "Agent 2 — Segments text into document graph nodes",
        "agent": "agent_2_segmentation",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_2_segmentation.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "role_tagger": {
        "name": "role_tagger",
        "description": "Agent 3 — Tags each node with a legal role",
        "agent": "agent_3_role_tagger",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_3_role_tagger.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "entity_extractor": {
        "name": "entity_extractor",
        "description": "Agent 4A — Extracts actors, objects, locations, times from tagged nodes",
        "agent": "agent_4a_entity_extractor",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_4a_entity_extractor.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "event_builder": {
        "name": "event_builder",
        "description": "Agent 4B — Builds structured events from entities and roles",
        "agent": "agent_4b_event_builder",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_4b_event_builder.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "legal_fact_normalizer": {
        "name": "legal_fact_normalizer",
        "description": "Agent 5A — Normalizes events into legal fact abstractions",
        "agent": "agent_5a_legal_fact_normalizer",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_5a_legal_fact_normalizer.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "legal_signal_extractor": {
        "name": "legal_signal_extractor",
        "description": "Agent 5B — Extracts legal signals from facts and events",
        "agent": "agent_5b_legal_signal_extractor",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_5b_legal_signal_extractor.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "statute_retriever": {
        "name": "statute_retriever",
        "description": "Agent 5C — Retrieves relevant IPC statute candidates using BM25",
        "agent": "agent_5c_statute_retriever",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_5c_statute_retriever.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "ingredient_evaluator": {
        "name": "ingredient_evaluator",
        "description": "Agent 6 — Evaluates IPC ingredients against case evidence (deterministic + LLM)",
        "agent": "agent_6_ingredient_evaluator",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_6_ingredient_evaluator.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "precedent_retriever": {
        "name": "precedent_retriever",
        "description": "Agent 7 (retrieval) — Retrieves precedent cases using BM25 over judgment dataset",
        "agent": "agent_7_precedent_comparator",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_7_precedent_comparator.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "precedent_extractor": {
        "name": "precedent_extractor",
        "description": "Agent 7 (extraction) — Extracts judicial reasoning from retrieved precedents",
        "agent": "agent_7_precedent_comparator",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_7_precedent_comparator.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "pattern_comparator": {
        "name": "pattern_comparator",
        "description": "Agent 7 (comparison) — Compares ingredient failures with precedent patterns",
        "agent": "agent_7_precedent_comparator",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_7_precedent_comparator.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "loophole_miner": {
        "name": "loophole_miner",
        "description": "Agent 8 — Identifies legal loopholes from ingredient failures and precedent patterns",
        "agent": "agent_8_loophole_miner",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_8_loophole_miner.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
    "argument_generator": {
        "name": "argument_generator",
        "description": "Agent 9 — Generates final legal argument (prosecution/defence/neutral)",
        "agent": "agent_9_final_argument",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
                "role": {
                    "type": "string",
                    "enum": ["prosecution", "defence", "neutral"],
                    "default": "neutral",
                },
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_9_final_argument.run(
            tenant_id=params["tenant_id"],
            case_id=params["case_id"],
            role=params.get("role", "neutral"),
        ),
    },
    "memory_writer": {
        "name": "memory_writer",
        "description": "Agent 5D — Stores feedback patterns for future retrieval boosting",
        "agent": "agent_5d_feedback_memory",
        "input_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "case_id": {"type": "string"},
            },
            "required": ["tenant_id", "case_id"],
        },
        "handler": lambda params: agent_5d_feedback_memory.run(
            tenant_id=params["tenant_id"], case_id=params["case_id"]
        ),
    },
}


# Pipeline order for full execution
PIPELINE_ORDER = [
    "ingestor",
    "normalizer",
    "segmenter",
    "role_tagger",
    "entity_extractor",
    "event_builder",
    "legal_fact_normalizer",
    "legal_signal_extractor",
    "statute_retriever",
    "ingredient_evaluator",
    "precedent_retriever",
    "loophole_miner",
    "argument_generator",
    "memory_writer",
]


def get_tool(name: str) -> dict | None:
    """Get a tool definition by name."""
    return TOOL_REGISTRY.get(name)


def list_tools() -> list[dict]:
    """List all registered tools."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "agent": t["agent"],
            "input_schema": t["input_schema"],
        }
        for t in TOOL_REGISTRY.values()
    ]


def invoke_tool(name: str, params: dict) -> dict:
    """Invoke a tool by name with given parameters."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        raise ValueError(f"Unknown tool: {name}. Available: {list(TOOL_REGISTRY.keys())}")
    
    return tool["handler"](params)
