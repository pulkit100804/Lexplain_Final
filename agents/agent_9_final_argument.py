"""
Agent 9 — Final Argument Engine

Input:  ingredient_report.json, loopholes.json, precedent_comparison.json
Output: final_argument.json

Generates structured legal argument in one of three modes:
  - prosecution  — emphasize satisfied ingredients, strong evidence, supporting precedents
  - defence      — emphasize loopholes, missing ingredients, weak evidence, precedent failures
  - neutral      — balanced view highlighting uncertainty

SAFETY:
  - NO new facts
  - NO hallucination
  - ONLY structured inputs
  - NO invented precedent reasoning
  - LLM can reason about existing evidence but MUST NOT fabricate
"""

import json
import logging
from pathlib import Path

from config import (
    get_case_dir,
    load_json,
    save_json,
    build_provenance,
    GOOGLE_API_KEY,
    LLM_SAFETY_PROMPT,
    DEFAULT_MODEL
)

logger = logging.getLogger("lexplain.agent9")

VALID_ROLES = {"prosecution", "defence", "neutral"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section Builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_case_summary(ingredient_report: dict) -> str:
    """Build a factual case summary from ingredient data."""
    evaluations = ingredient_report.get("statute_evaluations", [])
    
    # Count statutes by status
    valid = [e for e in evaluations if e.get("status") == "valid"]
    weak = [e for e in evaluations if e.get("status") == "weak"]
    invalid = [e for e in evaluations if e.get("status") == "invalid"]
    
    parts = []
    parts.append(f"Case involves {len(evaluations)} statute evaluation(s).")
    if valid:
        parts.append(f"{len(valid)} statute(s) have VALID ingredient satisfaction: "
                     f"{', '.join(e['statute_id'] for e in valid)}.")
    if weak:
        parts.append(f"{len(weak)} statute(s) have WEAK ingredient satisfaction: "
                     f"{', '.join(e['statute_id'] for e in weak)}.")
    if invalid:
        parts.append(f"{len(invalid)} statute(s) are INVALID (core ingredient failure): "
                     f"{', '.join(e['statute_id'] for e in invalid)}.")
    
    flags = ingredient_report.get("consistency_flags", [])
    if flags:
        parts.append(f"{len(flags)} consistency issue(s) detected.")
    
    return " ".join(parts)


def _build_applicable_sections(ingredient_report: dict) -> list[dict]:
    """List applicable IPC sections with scores."""
    sections = []
    for e in ingredient_report.get("statute_evaluations", []):
        sections.append({
            "section": e.get("statute_id", ""),
            "heading": e.get("heading", ""),
            "score": e.get("overall_score", 0),
            "status": e.get("status", ""),
        })
    return sorted(sections, key=lambda x: x["score"], reverse=True)


def _build_ingredient_analysis(ingredient_report: dict, role: str) -> list[dict]:
    """Build ingredient-level analysis based on role."""
    analysis = []
    
    for e in ingredient_report.get("statute_evaluations", []):
        statute_id = e.get("statute_id", "")
        ingredients = e.get("ingredients", [])
        
        for ing in ingredients:
            status = ing.get("status", "")
            
            # Filter based on role
            if role == "prosecution" and status == "not_satisfied":
                continue  # Skip failures in prosecution mode (mention only briefly)
            if role == "defence" and status == "satisfied":
                continue  # Skip successes in defence mode
            
            analysis.append({
                "section": statute_id,
                "ingredient": ing.get("ingredient_text", ""),
                "category": ing.get("ingredient_category", ""),
                "status": status,
                "confidence": ing.get("confidence", 0),
                "evidence_type": ing.get("evidence_type", ""),
                "reasoning": ing.get("reasoning", ""),
            })
    
    return analysis


def _build_missing_weak_elements(ingredient_report: dict, loopholes: dict) -> list[dict]:
    """Compile missing and weak elements."""
    elements = []
    
    for e in ingredient_report.get("statute_evaluations", []):
        for ing in e.get("ingredients", []):
            if ing.get("status") in ("not_satisfied", "partial"):
                elements.append({
                    "section": e.get("statute_id", ""),
                    "ingredient": ing.get("ingredient_text", ""),
                    "status": ing.get("status", ""),
                    "category": ing.get("ingredient_category", ""),
                    "reason": ing.get("reasoning", ""),
                })
    
    # Add loophole context
    for lh in loopholes.get("loopholes", []):
        elements.append({
            "section": lh.get("related_section", ""),
            "type": lh.get("type", ""),
            "reason": lh.get("reason", ""),
        })
    
    return elements


def _build_precedent_section(precedent_comparison: dict) -> dict:
    """Build precedent comparison section."""
    return {
        "similarity_score": precedent_comparison.get("similarity_score", 0),
        "matched_patterns": precedent_comparison.get("matched_patterns", []),
        "differences": precedent_comparison.get("differences", []),
        "references": precedent_comparison.get("precedent_references", []),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final Argument Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _generate_argument_text(
    tenant_id: str,
    role: str,
    case_summary: str,
    sections: list[dict],
    ingredient_analysis: list[dict],
    missing_elements: list[dict],
    precedent_section: dict,
    loopholes: dict,
) -> tuple[str, list[dict], dict]:
    """
    Generate the final argument text using LLM.
    LLM can REASON about existing evidence but MUST NOT fabricate facts.
    """
    if not GOOGLE_API_KEY:
        return _generate_argument_deterministic(
            role, case_summary, sections, ingredient_analysis,
            missing_elements, precedent_section, loopholes
        )
    
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Build structured context
        context = {
            "case_summary": case_summary,
            "sections": sections,
            "ingredient_analysis": ingredient_analysis[:20],
            "missing_elements": missing_elements[:15],
            "precedent": precedent_section,
            "loopholes": loopholes.get("loopholes", [])[:10],
        }
        
        role_instruction = {
            "prosecution": (
                "You are a PROSECUTION lawyer. Your goal is to construct a compelling, ingredient-level argument for conviction.\n"
                "- Emphasize satisfied ingredients (e.g., actus reus, causation) with strong evidence.\n"
                "- If direct intent is missing, argue how 'knowledge' or implicit intent can be legally inferred from the events (e.g., nature of weapon, force used).\n"
                "- Highlight precedents that support serious charges despite evidentiary gaps.\n"
                "- Do NOT just say 'the accused is guilty'. Use strict statutory language."
            ),
            "defence": (
                "You are a DEFENCE lawyer. Your goal is to dismantle the prosecution's case using strict legal logic.\n"
                "- Emphasize missing core ingredients (e.g., lack of explicit mens rea, weak causal linkage).\n"
                "- Highlight loopholes identified in the analysis and explain exactly why the charge fails statutory requirements.\n"
                "- Use precedents to support downgrading the charge (e.g., murder to culpable homicide) or complete acquittal.\n"
                "- Do NOT just say 'the accused is not guilty'. Point exclusively to ingredient-level deficiencies."
            ),
            "neutral": (
                "You are a NEUTRAL legal analyst. Your goal is to provide a balanced, objective assessment of the case's merits.\n"
                "- Clearly delineate which statutory ingredients are satisfied (e.g., actus reus) and which are missing (e.g., mens rea).\n"
                "- Explain the exact legal distinctions created by these gaps (e.g., the difference between Section 302 Murder and Section 304 Culpable Homicide).\n"
                "- Present how precedents historically treat these specific ingredient alignments.\n"
                "- Conclude on the strongest realistic legal classification based purely on the evidence."
            ),
        }
        
        prompt = f"""Generate a highly detailed, comprehensive legal argument based ONLY on the provided evidence.

ROLE: {role.upper()}
{role_instruction.get(role, role_instruction['neutral'])}

STRICT RULES:
- Use ONLY the data provided below
- Do NOT invent any facts or evidence
- Do NOT create fictional precedents
- Every claim must be traceable to the provided data
- If evidence is missing, say so explicitly
- You MUST write expansively. Do not write brief summaries. Provide in-depth analysis for each point.
- ALWAYS use ingredient-level reasoning (explicitly reference actus reus, mens rea, causation, etc.).
- ALWAYS justify arguments using the provided precedents.
- IMPORTANT: Your analysis MUST apply generalized legal reasoning suitable for ANY type of case (e.g., theft, fraud, assault, corporate disputes), not just homicide or murder. Adapt your language and focus to the specific sections and facts provided.

EVIDENCE DATA:
{json.dumps(context, indent=1)}

HUMAN FEEDBACK (apply these learned preferences to your perspective if relevant):
{get_human_feedback_patterns(tenant_id) or "None"}

You must return your response as a valid JSON object with the exact following structure:
{{
  "final_argument_text": "The full multi-paragraph argument text string, encompassing sections 1-6 (CASE SUMMARY, APPLICABLE SECTIONS, INGREDIENT ANALYSIS, MISSING EVIDENCE, PRECEDENT COMPARISON, FINAL ARGUMENT).",
  "structured_loopholes": [
    {{
      "title": "Short title of the loophole or missing element",
      "argument": "A detailed, structured paragraph explaining how a critical analyst or defence would leverage this specific loophole/gap in evidence."
    }}
  ],
  "final_decision": {{
    "final_offence": "Select ONE final offence that BEST matches: Act, Mental state, and Context. Do NOT default to highest punishment. Focus on PRECISION, not severity. (e.g. 'Section 304 Part II'). If no offence applies, return 'Acquittal'.",
    "why_it_fits_best": "Explain why this offence is the most precise fit, ensuring no higher offence is selected if mitigating factors reduce culpability, and no lower offence is selected if facts satisfy higher intent.",
    "why_others_rejected": "Explain why other candidates were rejected."
  }}
}}"""

        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction=LLM_SAFETY_PROMPT,
                response_mime_type="application/json",
            ),
        )
        
        try:
            data = json.loads(response.text)
            return data.get("final_argument_text", ""), data.get("structured_loopholes", []), data.get("final_decision", {})
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON in Agent 9.")
            return response.text.strip(), [], {}
        
    except Exception as e:
        logger.warning(f"LLM argument generation failed: {e}")
        return _generate_argument_deterministic(
            role, case_summary, sections, ingredient_analysis,
            missing_elements, precedent_section, loopholes
        )


def _generate_argument_deterministic(
    role: str,
    case_summary: str,
    sections: list[dict],
    ingredient_analysis: list[dict],
    missing_elements: list[dict],
    precedent_section: dict,
    loopholes: dict,
) -> tuple[str, list[dict], dict]:
    """Deterministic fallback argument generation."""
    lines = []
    
    lines.append("=" * 60)
    lines.append(f"LEGAL ARGUMENT — {role.upper()} PERSPECTIVE")
    lines.append("=" * 60)
    
    lines.append("\n1. CASE SUMMARY")
    lines.append("-" * 40)
    lines.append(case_summary)
    
    lines.append("\n2. APPLICABLE IPC SECTIONS")
    lines.append("-" * 40)
    for s in sections:
        lines.append(f"  Section {s['section']} ({s['heading']}): "
                     f"Score={s['score']}, Status={s['status']}")
    
    lines.append("\n3. INGREDIENT ANALYSIS")
    lines.append("-" * 40)
    for ia in ingredient_analysis[:15]:
        lines.append(f"  [{ia['section']}] {ia['ingredient']}")
        lines.append(f"    Status: {ia['status']} | Confidence: {ia['confidence']:.2f} | "
                     f"Evidence: {ia['evidence_type']}")
        lines.append(f"    Reasoning: {ia['reasoning']}")
    
    lines.append("\n4. MISSING / WEAK ELEMENTS")
    lines.append("-" * 40)
    for me in missing_elements[:10]:
        if "ingredient" in me:
            lines.append(f"  [{me.get('section', '')}] {me['ingredient']} — {me.get('status', '')}")
        else:
            lines.append(f"  [{me.get('section', '')}] {me.get('type', '')}: {me.get('reason', '')}")
    
    lines.append("\n5. PRECEDENT COMPARISON")
    lines.append("-" * 40)
    lines.append(f"  Similarity Score: {precedent_section.get('similarity_score', 0)}")
    for mp in precedent_section.get("matched_patterns", [])[:5]:
        lines.append(f"  Pattern: {mp.get('pattern', '')}")
        lines.append(f"    Reference: {mp.get('precedent_reference', '')}")
    
    lines.append("\n6. FINAL ARGUMENT")
    lines.append("-" * 40)
    
    if role == "prosecution":
        satisfied_count = sum(1 for ia in ingredient_analysis if ia["status"] == "satisfied")
        lines.append(f"  The prosecution submits a detailed argument based on {satisfied_count} ingredient(s) that are ")
        lines.append("  fully satisfied by direct evidence.")
        valid_sections = [s for s in sections if s["status"] == "valid"]
        if valid_sections:
            lines.append(f"  Sections {', '.join(s['section'] for s in valid_sections)} "
                         "have valid ingredient satisfaction.")
        lines.append("  The evidence clearly establishes the actus reus and connects the accused to the events.")
    
    elif role == "defence":
        loophole_list = loopholes.get("loopholes", [])
        lines.append(f"  The defence submits a comprehensive argument highlighting {len(loophole_list)} legal loophole(s).")
        core_missing = [lh for lh in loophole_list if lh["type"] == "missing_core_ingredient"]
        if core_missing:
            lines.append(f"  Specifically, {len(core_missing)} core ingredient(s) are NOT SATISFIED.")
            for cm in core_missing:
                lines.append(f"    - {cm.get('reason', 'Missing core element')}")
            lines.append("  These fatal flaws render the charge entirely unsustainable.")
        weak = [lh for lh in loophole_list if lh["type"] == "weak_evidence"]
        if weak:
            lines.append(f"  Furthermore, {len(weak)} ingredient(s) have critically weak evidence support.")
            for w in weak:
                lines.append(f"    - {w.get('reason', 'Weak evidence')}")
    
    else:  # neutral
        lines.append("  This comprehensive analysis presents a detailed, balanced view of the case.")
        valid = [s for s in sections if s["status"] == "valid"]
        invalid = [s for s in sections if s["status"] == "invalid"]
        if valid:
            lines.append(f"  A strong, legally sound case exists for: {', '.join(s['section'] for s in valid)}")
        if invalid:
            lines.append(f"  However, there is a weak or invalid baseline for: {', '.join(s['section'] for s in invalid)}")
        lines.append("  Key uncertainties heavily exist around intent and causation elements, requiring further evidentiary review.")
    
    structured_loopholes = []
    if role in ("defence", "neutral"):
        for lh in loopholes.get("loopholes", []):
            structured_loopholes.append({
                "title": f"Loophole: {lh.get('type', 'Unknown')}",
                "argument": f"The ingredient/evidence is deficient: {lh.get('reason', '')}"
            })
            
    final_decision = {
        "final_offence": "Unable to determine without LLM",
        "why_it_fits_best": "Deterministic fallback used.",
        "why_others_rejected": "Deterministic fallback used."
    }
            
    return "\n".join(lines), structured_loopholes, final_decision


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(tenant_id: str, case_id: str, role: str = "neutral") -> dict:
    """
    Run Agent 9 — Final Argument Engine.
    
    Parameters
    ----------
    role : str
        One of 'prosecution', 'defence', 'neutral' (default='neutral').
    
    Reads: ingredient_report.json, loopholes.json, precedent_comparison.json
    Writes: final_argument.json
    """
    if role not in VALID_ROLES:
        role = "neutral"
    
    case_dir = get_case_dir(tenant_id, case_id)
    
    ingredient_report = load_json(case_dir / "ingredient_report.json")
    loopholes = load_json(case_dir / "loopholes.json")
    precedent_comparison = load_json(case_dir / "precedent_comparison.json")
    
    # Build argument sections
    case_summary = _build_case_summary(ingredient_report)
    sections = _build_applicable_sections(ingredient_report)
    ingredient_analysis = _build_ingredient_analysis(ingredient_report, role)
    missing_elements = _build_missing_weak_elements(ingredient_report, loopholes)
    precedent_section = _build_precedent_section(precedent_comparison)
    
    # Generate final argument text
    argument_text, structured_loopholes, final_decision = _generate_argument_text(
        tenant_id, role, case_summary, sections, ingredient_analysis,
        missing_elements, precedent_section, loopholes,
    )
    
    # Build output
    provenance = build_provenance(
        case_id, tenant_id, "agent_9_final_argument",
        ["ingredient_report.json", "loopholes.json", "precedent_comparison.json"],
    )
    
    result = {
        **provenance,
        "role": role,
        "case_summary": case_summary,
        "applicable_sections": sections,
        "ingredient_analysis": ingredient_analysis,
        "missing_weak_elements": missing_elements,
        "precedent_comparison": precedent_section,
        "loopholes_summary": {
            "total": loopholes.get("total_loopholes", 0),
            "by_type": loopholes.get("by_type", {}),
        },
        "final_decision": final_decision,
        "final_argument": argument_text,
        "structured_loophole_arguments": structured_loopholes,
    }
    
    save_json(case_dir / "final_argument.json", result)
    return result
