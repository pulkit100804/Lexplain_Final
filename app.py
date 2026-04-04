"""
Lexplain — Rich UI Web Application
====================================
Run:  python app.py
Opens: http://localhost:5000

Provides:
  GET  /                         → serve UI
  POST /api/analyze              → start pipeline, return job_id
  GET  /api/stream/<job_id>      → SSE stream of agent progress
  GET  /api/history              → list of all past analyses
  GET  /api/case/<tenant>/<id>   → full results for a case
  POST /api/feedback             → submit per-component feedback
"""

import json
import os
import sys
import time
import uuid
import threading
import logging
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty

# Ensure our modules are importable
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, Response, send_from_directory
from pipeline import run_full_pipeline_streaming
from config import CASES_DIR, DATA_DIR

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lexplain.app")

# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────
UI_DIR = Path(__file__).parent / "ui"
app = Flask(__name__, static_folder=str(UI_DIR), static_url_path="/ui")

# In-memory job store: job_id → Queue of SSE events
_JOBS: dict[str, Queue] = {}
_JOB_RESULTS: dict[str, dict] = {}

# ──────────────────────────────────────────────────────────────────────────────
# Storage helpers
# ──────────────────────────────────────────────────────────────────────────────
HISTORY_FILE = DATA_DIR / "history.json"
FEEDBACK_FILE = DATA_DIR / "feedback_store.json"

def _load_json_list(path: Path) -> list:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _append_json_list(path: Path, item: dict) -> None:
    data = _load_json_list(path)
    data.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def _save_history_entry(case_id: str, tenant_id: str, snippet: str,
                        valid_sections: list, status: str) -> None:
    entry = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "snippet": snippet[:180],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "valid_sections": valid_sections,
        "status": status,
    }
    _append_json_list(HISTORY_FILE, entry)


# ──────────────────────────────────────────────────────────────────────────────
# Routes — Static UI
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(UI_DIR), "index.html")


@app.route("/ui/<path:filename>")
def ui_static(filename):
    return send_from_directory(str(UI_DIR), filename)


# ──────────────────────────────────────────────────────────────────────────────
# Routes — Analysis
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Start analysis. Returns {job_id} immediately."""
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    role = data.get("role", "neutral")
    tenant_id = data.get("tenant_id", "ui_tenant")

    if not text:
        return jsonify({"error": "No case text provided"}), 400

    job_id = str(uuid.uuid4())
    q: Queue = Queue()
    _JOBS[job_id] = q

    def _run():
        try:
            result = run_full_pipeline_streaming(
                text=text,
                tenant_id=tenant_id,
                role=role,
                on_step=lambda name, status, detail: q.put({
                    "type": "progress",
                    "stage": name,
                    "status": status,
                    "detail": detail,
                }),
            )
            _JOB_RESULTS[job_id] = result

            # Save to history
            final = result.get("results", {}).get("agent_9", {})
            valid_sections = [
                s["section"]
                for s in final.get("applicable_sections", [])
                if s.get("status") in ("valid", "weak")
            ]
            _save_history_entry(
                case_id=result["case_id"],
                tenant_id=tenant_id,
                snippet=text,
                valid_sections=valid_sections,
                status="complete",
            )

            q.put({"type": "done", "case_id": result["case_id"],
                   "tenant_id": tenant_id})
        except Exception as exc:
            logger.exception("Pipeline error")
            q.put({"type": "error", "message": str(exc)})

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream(job_id):
    """SSE endpoint — streams agent progress events."""
    q = _JOBS.get(job_id)
    if q is None:
        return jsonify({"error": "Unknown job"}), 404

    def _generate():
        yield "retry: 1000\n\n"
        while True:
            try:
                event = q.get(timeout=60)
            except Empty:
                yield ": keepalive\n\n"
                continue

            yield f"data: {json.dumps(event)}\n\n"

            if event.get("type") in ("done", "error"):
                _JOBS.pop(job_id, None)
                break

    return Response(_generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ──────────────────────────────────────────────────────────────────────────────
# Routes — History & Case Detail
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/history")
def history():
    """Return list of past analyses (most recent first)."""
    entries = _load_json_list(HISTORY_FILE)
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return jsonify(entries)


@app.route("/api/case/<tenant_id>/<case_id>")
def case_detail(tenant_id, case_id):
    """Return all result files for a completed case."""
    case_dir = CASES_DIR / tenant_id / case_id
    if not case_dir.exists():
        return jsonify({"error": "Case not found"}), 404

    def _read(fname):
        p = case_dir / fname
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    return jsonify({
        "case_id": case_id,
        "tenant_id": tenant_id,
        "final_argument": _read("final_argument.json"),
        "precedent_comparison": _read("precedent_comparison.json"),
        "loopholes": _read("loopholes.json"),
        "ingredient_report": _read("ingredient_report.json"),
        "statute_candidates": _read("statute_candidates.json"),
        "legal_facts": _read("legal_facts.json"),
        "legal_signals": _read("legal_signals.json"),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Routes — Feedback
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/feedback", methods=["POST"])
def feedback():
    """Accept per-component feedback and persist to feedback_store.json."""
    data = request.get_json(force=True)
    case_id = data.get("case_id", "")
    tenant_id = data.get("tenant_id", "")
    components = data.get("components", {})

    if not case_id:
        return jsonify({"error": "case_id required"}), 400

    # Extract learned patterns from positive feedback
    patterns_learned = []
    for component, fb in components.items():
        if fb.get("correct"):
            patterns_learned.append({
                "component": component,
                "case_id": case_id,
                "pattern": fb.get("comment", "confirmed_correct"),
            })

    entry = {
        "case_id": case_id,
        "tenant_id": tenant_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": components,
        "patterns_learned": patterns_learned,
    }
    _append_json_list(FEEDBACK_FILE, entry)

    # Trigger Agent 5D pattern learning if available
    try:
        from agents import agent_5d_feedback_memory
        agent_5d_feedback_memory.run(tenant_id=tenant_id, case_id=case_id)
    except Exception as e:
        logger.warning(f"Agent 5D update skipped: {e}")

    return jsonify({"status": "saved", "patterns_learned": len(patterns_learned)})


@app.route("/api/feedback/<tenant_id>/<case_id>")
def get_feedback(tenant_id, case_id):
    """Get existing feedback for a case."""
    all_feedback = _load_json_list(FEEDBACK_FILE)
    matching = [f for f in all_feedback
                if f.get("case_id") == case_id and f.get("tenant_id") == tenant_id]
    return jsonify(matching[-1] if matching else {})


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    url = f"http://localhost:{port}"
    print(f"\n{'='*60}")
    print(f"  🏛  Lexplain Legal Analysis UI")
    print(f"  📡  Server: {url}")
    print(f"  ⚡  Streaming: SSE enabled")
    print(f"{'='*60}\n")
    # Open browser after a short delay
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
