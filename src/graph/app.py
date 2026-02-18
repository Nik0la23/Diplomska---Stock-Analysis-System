"""
LangGraph Studio entrypoint.

This module intentionally stays thin:
- exports a compiled `graph` for `langgraph dev` / Studio
- provides an optional `__main__` visualization helper to write graph_schema.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from src.graph.workflow import create_stock_analysis_workflow
except ModuleNotFoundError:
    # Allows running `python src/graph/app.py` directly from repo root.
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.graph.workflow import create_stock_analysis_workflow


def make_graph(config: Any | None = None):
    """Factory for LangGraph CLI/Studio (config currently unused)."""
    _ = config
    return create_stock_analysis_workflow()


# Studio expects an importable compiled graph variable.
graph = make_graph()


def _write_graph_schema_png(output_path: Path) -> None:
    png = graph.get_graph().draw_mermaid_png()
    output_path.write_bytes(png)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    _write_graph_schema_png(repo_root / "graph_schema.png")
