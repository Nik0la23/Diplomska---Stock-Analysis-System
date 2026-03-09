"""
Quick runner — executes the full 14-node pipeline for any ticker
and prints Node 13 (beginner) + Node 14 (technical) explanations.

Usage:
    python scripts/run_analysis.py NVDA
    python scripts/run_analysis.py AAPL
    python scripts/run_analysis.py MSFT
"""

import os
import sys
import logging
from pathlib import Path

# Disable LangSmith tracing — the full state (38 MB+) exceeds the 26 MB
# LangSmith payload limit, causing noisy 422 upload errors on every run.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"]    = "false"

# Ensure project root is on sys.path when running as a script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Suppress noisy library logs — pipeline INFO already visible by default
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "urllib3", "yfinance", "peewee",
              "transformers", "langsmith"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from src.graph.workflow import run_stock_analysis


def main() -> None:
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    print(f"\nRunning full pipeline for {ticker}...\n")

    result = run_stock_analysis(ticker)

    # -------------------------------------------------------------------------
    # Detect early-exit (Node 1 failed, no price data)
    # -------------------------------------------------------------------------
    errors = result.get("errors") or []
    if result.get("final_signal") is None:
        print("=" * 70)
        print(f"  {ticker}  |  Pipeline did not complete (Node 1 failed)")
        print("=" * 70)
        for e in errors:
            print(f"  ERROR: {e}")
        print("\n  Tip: yfinance rate-limits after repeated runs.")
        print("  Wait 10-15 min then try again, or run a cached ticker (e.g. PLTR).")
        return

    # -------------------------------------------------------------------------
    # Key stats header
    # -------------------------------------------------------------------------
    mc  = result.get("market_context")    or {}
    sc  = result.get("signal_components") or {}
    rs  = sc.get("risk_summary")          or {}
    conf = (result.get("final_confidence") or 0.0) * 100

    print("=" * 70)
    print(f"  {ticker}  |  Signal: {result.get('final_signal')}  |  Confidence: {conf:.1f}%")
    print("=" * 70)
    print(f"  VIX: {mc.get('vix_level', 'N/A')} ({mc.get('vix_category', 'N/A')})")
    headwind = mc.get("market_headwind_score")
    headwind_str = f"{headwind:+.3f}" if headwind is not None else "N/A"
    print(f"  Market headwind: {headwind_str}  |  Risk: {rs.get('overall_risk_level', 'N/A')}")
    print(f"  Errors: {len(errors)}")
    for e in errors:
        print(f"    - {e}")

    # -------------------------------------------------------------------------
    # Node 13 — Beginner Explanation
    # -------------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  NODE 13 — BEGINNER EXPLANATION")
    print("─" * 70)
    print(result.get("beginner_explanation") or "(not generated)")

    # -------------------------------------------------------------------------
    # Node 14 — Technical Report
    # -------------------------------------------------------------------------
    print("\n" + "─" * 70)
    print("  NODE 14 — TECHNICAL EXPLANATION")
    print("─" * 70)
    print(result.get("technical_explanation") or "(not generated)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
