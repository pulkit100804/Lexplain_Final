"""
Lexplain MCP — Server

JSON-RPC style server that receives tool invocation requests,
routes to registered tool handlers, and returns structured JSON responses.
Includes logging with trace IDs.
"""

import json
import uuid
import time
import logging
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.tools import TOOL_REGISTRY, list_tools, invoke_tool, PIPELINE_ORDER

logger = logging.getLogger("lexplain.mcp.server")


class MCPRequestHandler(BaseHTTPRequestHandler):
    """Handle JSON-RPC style MCP requests."""

    def _send_json(self, status: int, data: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode("utf-8"))

    def do_POST(self):
        """Handle tool invocation requests."""
        trace_id = str(uuid.uuid4())[:8]
        t0 = time.time()

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            request = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {
                "error": f"Invalid JSON: {e}",
                "trace_id": trace_id,
            })
            return

        method = request.get("method", "")
        params = request.get("params", {})

        logger.info(f"[{trace_id}] {method} — params={list(params.keys())}")

        if method == "tools/list":
            self._send_json(200, {
                "trace_id": trace_id,
                "tools": list_tools(),
            })
            return

        if method == "tools/invoke":
            tool_name = params.get("tool_name", "")
            tool_params = params.get("arguments", {})

            if tool_name not in TOOL_REGISTRY:
                self._send_json(404, {
                    "error": f"Unknown tool: {tool_name}",
                    "available": list(TOOL_REGISTRY.keys()),
                    "trace_id": trace_id,
                })
                return

            try:
                result = invoke_tool(tool_name, tool_params)
                elapsed = time.time() - t0
                logger.info(f"[{trace_id}] {tool_name} completed in {elapsed:.2f}s")

                self._send_json(200, {
                    "trace_id": trace_id,
                    "tool": tool_name,
                    "result": result,
                    "elapsed_seconds": round(elapsed, 2),
                })
            except Exception as e:
                logger.error(f"[{trace_id}] {tool_name} failed: {e}")
                self._send_json(500, {
                    "error": str(e),
                    "trace_id": trace_id,
                    "tool": tool_name,
                })
            return

        if method == "pipeline/run":
            text = params.get("text", "")
            tenant_id = params.get("tenant_id", "")
            role = params.get("role", "neutral")

            if not text or not tenant_id:
                self._send_json(400, {
                    "error": "Missing 'text' or 'tenant_id'",
                    "trace_id": trace_id,
                })
                return

            try:
                from pipeline import run_full_pipeline
                result = run_full_pipeline(text=text, tenant_id=tenant_id, role=role)
                elapsed = time.time() - t0
                logger.info(f"[{trace_id}] Pipeline completed in {elapsed:.2f}s")

                self._send_json(200, {
                    "trace_id": trace_id,
                    "method": "pipeline/run",
                    "case_id": result.get("case_id"),
                    "tenant_id": result.get("tenant_id"),
                    "elapsed_seconds": round(elapsed, 2),
                })
            except Exception as e:
                logger.error(f"[{trace_id}] Pipeline failed: {e}")
                self._send_json(500, {
                    "error": str(e),
                    "trace_id": trace_id,
                })
            return

        self._send_json(400, {
            "error": f"Unknown method: {method}. Use tools/list, tools/invoke, or pipeline/run",
            "trace_id": trace_id,
        })

    def do_GET(self):
        """Health check and info endpoint."""
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "tools": len(TOOL_REGISTRY)})
        elif self.path == "/tools":
            self._send_json(200, {"tools": list_tools()})
        elif self.path == "/pipeline":
            self._send_json(200, {"pipeline_order": PIPELINE_ORDER})
        else:
            self._send_json(200, {
                "name": "Lexplain MCP Server",
                "version": "1.0.0",
                "endpoints": {
                    "GET /health": "Health check",
                    "GET /tools": "List all tools",
                    "GET /pipeline": "Pipeline execution order",
                    "POST": "Tool invocation (JSON-RPC)",
                },
            })

    def log_message(self, format, *args):
        """Suppress default HTTP logging (we use our own)."""
        pass


def start_server(port: int = 8765):
    """Start the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    server = HTTPServer(("0.0.0.0", port), MCPRequestHandler)
    logger.info(f"🚀 Lexplain MCP Server running on http://0.0.0.0:{port}")
    logger.info(f"   Tools registered: {len(TOOL_REGISTRY)}")
    logger.info(f"   Endpoints: /health, /tools, /pipeline")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")
        server.server_close()


if __name__ == "__main__":
    start_server()
