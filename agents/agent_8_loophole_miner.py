"""
Agent 8 — Legal Reasoning and Conflict Resolution Engine (formerly Loophole Miner)

Input:  ingredient_report.json, legal_facts.json
Output: loopholes.json

Identifies mitigating factors, handles statutory hierarchy, and performs 
complex mental state reductions (e.g., intention -> knowledge).
"""

import json
import logging
from pathlib import Path

from config import (
    get_case_dir,
    load_json,
    save_json,
    build_provenance,
    get_human_feedback_patterns,
    GOOGLE_API_KEY,
    DEFAULT_MODEL,
    LLM_SAFETY_PROMPT
)

logger = logging.getLogger("lexplain.agent8")

def run(tenant_id: str, case_id: str) -> dict:
    """Run Conflict Resolution Engine."""
    case_dir = get_case_dir(tenant_id, case_id)

    ingredient_report = load_json(case_dir / "ingredient_report.json")
    legal_facts_data = load_json(case_dir / "legal_facts.json")

    evaluations = ingredient_report.get("statute_evaluations", [])
    facts = legal_facts_data.get("legal_facts", [])

    provenance = build_provenance(
        case_id, tenant_id, "agent_8_conflict_resolution",
        ["ingredient_report.json", "legal_facts.json"],
    )

    if not GOOGLE_API_KEY or not evaluations:
        # Fallback empty result
        result = {
            **provenance,
            "selected_offence": "Unknown",
            "rejected_offences": [],
            "explanation": "Agent 8 LLM missing or skipped.",
            "loopholes": [],
            "total_loopholes": 0,
            "by_type": {}
        }
        save_json(case_dir / "loopholes.json", result)
        return result

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Summarize evaluations to save tokens
        evaluations_summary = []
        for e in evaluations:
            evaluations_summary.append({
                "statute_id": e.get("statute_id", ""),
                "heading": e.get("heading", ""),
                "status": e.get("status", ""),
                "llm_classification": e.get("llm_classification", ""),
                "satisfied_ingredients": [i["ingredient_text"] for i in e.get("ingredients", []) if i["status"] == "satisfied"],
                "missing_ingredients": [i["ingredient_text"] for i in e.get("ingredients", []) if i["status"] != "satisfied"]
            })

        prompt = f"""You are a legal reasoning and conflict resolution engine.

You are given multiple statutes that are satisfied to varying degrees.

Your task is to determine the MOST LEGALLY APPROPRIATE classification.

Follow this reasoning process:

STEP 1: Identify the most severe offence whose ingredients are satisfied.

STEP 2: Independently evaluate whether any mitigating conditions exist in facts, such as:
- Sudden fight
- Lack of premeditation
- Heat of passion
- Single act vs repeated acts
- Nature of weapon (improvised vs deadly)
- Prior enmity or absence of it

STEP 3: Check if these mitigating factors weaken the mental element (mens rea) required for the higher offence.

STEP 4: If the higher offence requires a stronger mental state (e.g., intention to cause death), but facts only support a weaker mental state (e.g., knowledge of likelihood):
    → Reject or downgrade the higher offence

STEP 5: Select the offence whose mental element and act BOTH align most precisely with facts.

IMPORTANT RULES:
- Do NOT use section numbers to decide.
- Do NOT assume hierarchy blindly.
- Base decision ONLY on:
   → ingredient satisfaction
   → mental state alignment
   → factual context

EVIDENCE DATA:
Legal Facts: {json.dumps(facts[:30], indent=1)}
Evaluated Statutes: {json.dumps(evaluations_summary, indent=1)}

HUMAN FEEDBACK (apply these learned preferences to your decisions if relevant):
{get_human_feedback_patterns(tenant_id) or "None"}

OUTPUT:
Return ONLY a JSON object with this exact structure:
{{
  "selected_offence": "Section XXX",
  "rejected_offences": [
    {{
      "section": "Section YYY",
      "reason": "Why it was rejected (e.g., downgraded due to heat of passion)"
    }}
  ],
  "explanation": "Detailed explanation of how facts align better with selected offence."
}}
Do NOT output markdown code blocks."""

        AGENT_8_SYSTEM_PROMPT = """You are a meticulous, objective conflict resolution engine. Follow instructions exactly."""

        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=AGENT_8_SYSTEM_PROMPT,
                response_mime_type="application/json",
            ),
        )

        llm_data = json.loads(response.text)
        
        # Translate the rejected offences into "loopholes" format so that the UI
        # and Agent 9 can still process them natively without breaking downstream logic.
        loopholes_format = []
        for rej in llm_data.get("rejected_offences", []):
            loopholes_format.append({
                "type": "mitigation_downgrade",
                "related_section": rej.get("section", ""),
                "reason": rej.get("reason", ""),
                "supporting_reference": []
            })
            
        result = {
            **provenance,
            "selected_offence": llm_data.get("selected_offence", ""),
            "rejected_offences": llm_data.get("rejected_offences", []),
            "explanation": llm_data.get("explanation", ""),
            "loopholes": loopholes_format,
            "total_loopholes": len(loopholes_format),
            "by_type": {"mitigation_downgrade": len(loopholes_format)}
        }
        
    except Exception as e:
        logger.warning(f"Agent 8 LLM Failure: {e}")
        result = {
            **provenance,
            "selected_offence": "Unknown",
            "rejected_offences": [],
            "explanation": f"Failed to execute LLM mitigation reasoning: {e}",
            "loopholes": [],
            "total_loopholes": 0,
            "by_type": {}
        }

    save_json(case_dir / "loopholes.json", result)
    return result
