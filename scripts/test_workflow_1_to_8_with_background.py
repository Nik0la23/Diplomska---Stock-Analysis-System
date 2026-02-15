"""
End-to-End Test: Workflow Nodes 1–8 + Background Script

Runs the complete pipeline and background script to validate integration:

  1. Run full workflow (Node 1 → 3 → 2 → 9A → 4,5,6,7 → Node 8) for a ticker
  2. Run background script (update_news_outcomes) to populate news_outcomes
  3. Optionally run workflow again to see Node 8 with real historical outcomes
  4. Print DB stats (news_articles, news_outcomes, source_reliability)

Usage:
  python -m scripts.test_workflow_1_to_8_with_background [TICKER]
  python -m scripts.test_workflow_1_to_8_with_background NVDA
  python -m scripts.test_workflow_1_to_8_with_background AAPL --skip-workflow  # only run background script
  python -m scripts.test_workflow_1_to_8_with_background NVDA --workflow-only  # no background script
"""

import argparse
import sys
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] if "/" in __file__ else ".")

from src.graph.workflow import create_stock_analysis_workflow, run_stock_analysis, print_analysis_summary
from src.graph.state import create_initial_state
from src.database.db_manager import get_connection, DEFAULT_DB_PATH


def get_db_stats(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Return counts for news_articles, news_outcomes, source_reliability."""
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM news_articles")
            n_articles = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM news_outcomes")
            n_outcomes = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM source_reliability")
            n_sources = cur.fetchone()[0]
            return {"news_articles": n_articles, "news_outcomes": n_outcomes, "source_reliability": n_sources}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Test workflow Nodes 1–8 and background script (news outcomes)"
    )
    parser.add_argument("ticker", nargs="?", default="NVDA", help="Ticker to run (default: NVDA)")
    parser.add_argument(
        "--skip-workflow",
        action="store_true",
        help="Skip workflow; only run background script and print DB stats",
    )
    parser.add_argument(
        "--workflow-only",
        action="store_true",
        help="Only run workflow (Nodes 1–8); do not run background script",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max articles for background script (default: 500)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less output from workflow and background script",
    )
    args = parser.parse_args()
    ticker = args.ticker.upper()

    print("\n" + "=" * 60)
    print("End-to-End Test: Workflow 1–8 + Background Script")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Time:   {datetime.now().isoformat()}")
    print("=" * 60 + "\n")

    # DB stats before
    stats_before = get_db_stats()
    if "error" in stats_before:
        print(f"Warning: Could not read DB stats: {stats_before['error']}")
    else:
        print("DB before:")
        print(f"  news_articles:      {stats_before['news_articles']}")
        print(f"  news_outcomes:      {stats_before['news_outcomes']}")
        print(f"  source_reliability: {stats_before['source_reliability']}")
        print()

    # 1) Run workflow (Nodes 1–8) unless skipped
    if not args.skip_workflow:
        print("Step 1: Running full workflow (Node 1 → 3 → 2 → 9A → 4,5,6,7 → Node 8)...")
        try:
            result = run_stock_analysis(ticker)
            print("Workflow completed successfully.\n")
            if not args.quiet:
                print_analysis_summary(result)
            else:
                # Minimal summary
                if result.get("news_impact_verification"):
                    niv = result["news_impact_verification"]
                    print(f"  Node 8: learning_adjustment={niv.get('learning_adjustment', 1.0):.3f}, "
                          f"insufficient_data={niv.get('insufficient_data', True)}")
        except Exception as e:
            print(f"Workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("Step 1: Skipped (--skip-workflow).\n")

    # 2) Run background script unless workflow-only
    if not args.workflow_only:
        print("Step 2: Running background script (update_news_outcomes)...")
        try:
            from scripts.update_news_outcomes import run_evaluation
            bg_result = run_evaluation(ticker=ticker, limit=args.limit, verbose=not args.quiet)
            print(f"Background script completed: evaluated={bg_result['evaluated']}, "
                  f"skipped={bg_result['skipped']}, accuracy={bg_result['accuracy_pct']:.1f}%\n")
        except Exception as e:
            print(f"Background script failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("Step 2: Skipped (--workflow-only).\n")

    # 3) DB stats after
    stats_after = get_db_stats()
    if "error" not in stats_after:
        print("DB after:")
        print(f"  news_articles:      {stats_after['news_articles']}")
        print(f"  news_outcomes:      {stats_after['news_outcomes']}")
        print(f"  source_reliability: {stats_after['source_reliability']}")
        if not args.skip_workflow and not args.workflow_only:
            print(f"  (Outcomes added:      {stats_after['news_outcomes'] - stats_before.get('news_outcomes', 0)})")
        print()

    print("=" * 60)
    print("End-to-end test finished.")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
