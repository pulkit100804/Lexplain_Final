"""
Lexplain — CLI Entry Point

Usage:
    python cli.py run --tenant <id> --input <file>
    python cli.py run --tenant <id> --text "case narrative..."
    python cli.py run --tenant <id> --text "..." --role defence
    python cli.py run-agent --agent <N> --tenant <id> --case <id>
    python cli.py chunk-judgments
    python cli.py mcp-server
"""

import argparse
import sys
from pathlib import Path

# Ensure package root is on path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import run_full_pipeline, run_single_agent


def main():
    parser = argparse.ArgumentParser(
        description="Lexplain — Multi-Agent Legal Reasoning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("--tenant", required=True, help="Tenant ID")
    run_parser.add_argument("--input", dest="input_file", help="Path to input text file")
    run_parser.add_argument("--text", help="Raw case text (alternative to --input)")
    run_parser.add_argument(
        "--role",
        choices=["prosecution", "defence", "neutral"],
        default="neutral",
        help="Argument role for Agent 9 (default: neutral)",
    )

    # --- run-agent command ---
    agent_parser = subparsers.add_parser("run-agent", help="Run a single agent")
    agent_parser.add_argument(
        "--agent",
        required=True,
        help="Agent name (0, 1, 2, 3, 4a, 4b, 5a, 5b, 5c, 6, 7, 8, 9, 5d)",
    )
    agent_parser.add_argument("--tenant", required=True, help="Tenant ID")
    agent_parser.add_argument("--case", required=True, help="Case ID")
    agent_parser.add_argument("--input", dest="input_file", help="Input file (for Agent 0)")
    agent_parser.add_argument("--text", help="Raw text (for Agent 0)")
    agent_parser.add_argument(
        "--role",
        choices=["prosecution", "defence", "neutral"],
        default="neutral",
        help="Argument role (for Agent 9)",
    )

    # --- chunk-judgments command ---
    subparsers.add_parser("chunk-judgments", help="Chunk judgment files for RAG retrieval")

    # --- mcp-server command ---
    mcp_parser = subparsers.add_parser("mcp-server", help="Start MCP server")
    mcp_parser.add_argument("--port", type=int, default=8765, help="Server port")

    args = parser.parse_args()

    if args.command == "run":
        # Get text from file or --text
        if args.input_file:
            text = Path(args.input_file).read_text(encoding="utf-8")
        elif args.text:
            text = args.text
        else:
            print("Error: Provide --input <file> or --text <string>")
            sys.exit(1)

        result = run_full_pipeline(text=text, tenant_id=args.tenant, role=args.role)
        print(f"\n{'='*60}")
        print(f"Pipeline complete!")
        print(f"  Case ID:  {result['case_id']}")
        print(f"  Tenant:   {result['tenant_id']}")
        print(f"  Role:     {args.role}")
        print(f"  Output:   cases/{result['tenant_id']}/{result['case_id']}/")
        print(f"{'='*60}")

    elif args.command == "run-agent":
        text = None
        if args.agent == "0":
            if args.input_file:
                text = Path(args.input_file).read_text(encoding="utf-8")
            elif args.text:
                text = args.text
            else:
                print("Error: Agent 0 requires --input <file> or --text <string>")
                sys.exit(1)

        result = run_single_agent(
            agent_name=args.agent,
            tenant_id=args.tenant,
            case_id=args.case,
            text=text,
            role=args.role,
        )
        print(f"Agent {args.agent} complete.")

    elif args.command == "chunk-judgments":
        from chunk_judgments import chunk_all_judgments
        chunk_all_judgments()

    elif args.command == "mcp-server":
        from mcp.server import start_server
        start_server(port=args.port)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
