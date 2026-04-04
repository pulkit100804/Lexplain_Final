"""
Lexplain MCP — Client

Client that sends tool invocation requests to the MCP server.
Supports individual tool invocation and full pipeline orchestration.
"""

import json
import urllib.request
import logging
from typing import Any

logger = logging.getLogger("lexplain.mcp.client")


class MCPClient:
    """Client for the Lexplain MCP Server."""

    def __init__(self, base_url: str = "http://localhost:8765"):
        self.base_url = base_url.rstrip("/")

    def _post(self, data: dict) -> dict:
        """Send a POST request to the MCP server."""
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"MCP request failed: {e}")
            raise

    def _get(self, path: str) -> dict:
        """Send a GET request to the MCP server."""
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"MCP GET request failed: {e}")
            raise

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def health(self) -> dict:
        """Check server health."""
        return self._get("/health")

    def list_tools(self) -> list[dict]:
        """List all available tools."""
        response = self._post({"method": "tools/list"})
        return response.get("tools", [])

    def invoke_tool(self, tool_name: str, **kwargs: Any) -> dict:
        """
        Invoke a single tool.

        Parameters
        ----------
        tool_name : str
            Name of the tool to invoke.
        **kwargs
            Tool parameters (tenant_id, case_id, text, role, etc.)

        Returns
        -------
        dict
            Tool result.
        """
        response = self._post({
            "method": "tools/invoke",
            "params": {
                "tool_name": tool_name,
                "arguments": kwargs,
            },
        })
        return response

    def run_pipeline(
        self,
        text: str,
        tenant_id: str,
        role: str = "neutral",
    ) -> dict:
        """
        Run the full pipeline via MCP server.

        Parameters
        ----------
        text : str
            Raw case narrative text.
        tenant_id : str
            Tenant identifier.
        role : str
            Argument role for Agent 9.

        Returns
        -------
        dict
            Pipeline result with case_id.
        """
        response = self._post({
            "method": "pipeline/run",
            "params": {
                "text": text,
                "tenant_id": tenant_id,
                "role": role,
            },
        })
        return response

    def get_pipeline_order(self) -> list[str]:
        """Get the pipeline execution order."""
        response = self._get("/pipeline")
        return response.get("pipeline_order", [])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_client(base_url: str = "http://localhost:8765") -> MCPClient:
    """Create an MCP client."""
    return MCPClient(base_url=base_url)


if __name__ == "__main__":
    # Quick test
    client = create_client()
    print("Health:", client.health())
    print("Tools:", json.dumps(client.list_tools(), indent=2))
