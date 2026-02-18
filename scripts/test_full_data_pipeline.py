"""
Comprehensive data pipeline test.

Shows exactly how much data is gathered across all four streams:
  1. Stock price data          (Node 1 / yfinance)
  2. Stock news                (Alpha Vantage, ticker-specific)
  3. Global market news        (Alpha Vantage, financial_markets topic)
  4. Related-company (peers)   (Finnhub peers → Alpha Vantage news)

For each news stream also reports sentiment coverage so you can judge
whether FinBERT is needed for the articles that lack AV sentiment.

Usage:
    PYTHONPATH=. python scripts/test_full_data_pipeline.py [TICKER]
    e.g.  PYTHONPATH=. python scripts/test_full_data_pipeline.py NVDA
"""

import sys
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

# ── project root on path ────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.db_manager import init_database
from src.graph.state import create_initial_state
from src.langgraph_nodes.node_01_data_fetching import fetch_price_data_node
from src.langgraph_nodes.node_02_news_fetching import fetch_all_news_node
from src.langgraph_nodes.node_03_related_companies import detect_related_companies_node

SEP = "=" * 72


def fmt_days(ts_list: list) -> str:
    """Return 'X.X days ago' string for min/max of a timestamp list."""
    if not ts_list:
        return "n/a"
    now = datetime.now().timestamp()
    oldest = (now - min(ts_list)) / 86400.0
    newest = (now - max(ts_list)) / 86400.0
    return f"oldest={oldest:.0f}d ago, newest={newest:.1f}d ago"


def sentiment_report(articles: list) -> tuple:
    """Return (with_sentiment_count, pct, label_breakdown) for an article list."""
    total = len(articles)
    if total == 0:
        return 0, 0.0, {}

    label_counts: dict = {}
    with_s = 0
    for a in articles:
        lbl = a.get("overall_sentiment_label")
        score = a.get("overall_sentiment_score")
        if lbl is not None or score is not None:
            with_s += 1
            key = lbl if lbl else "score_only"
            label_counts[key] = label_counts.get(key, 0) + 1

    pct = with_s / total * 100.0
    return with_s, pct, label_counts


def print_stream(name: str, articles: list) -> None:
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    total = len(articles)
    print(f"  Total articles:    {total}")
    if total == 0:
        print("  (no articles fetched)")
        return

    ts_list = [
        a.get("datetime", 0)
        for a in articles
        if isinstance(a.get("datetime", 0), (int, float)) and a["datetime"] > 0
    ]
    print(f"  Date range:        {fmt_days(ts_list)}")

    with_s, pct, labels = sentiment_report(articles)
    without_s = total - with_s
    print(f"  With AV sentiment: {with_s:>5}  ({pct:.1f}%)")
    print(f"  Missing sentiment: {without_s:>5}  ({100-pct:.1f}%)  ← FinBERT candidates")

    if labels:
        print("  Sentiment labels (AV):")
        for lbl, cnt in sorted(labels.items(), key=lambda x: -x[1]):
            bar = "█" * (cnt * 30 // max(labels.values()))
            print(f"    {lbl:<22} {cnt:>5}  {bar}")


def main() -> None:
    # Default to AAPL unless an explicit ticker is provided
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()

    print(SEP)
    print(f"  Full Pipeline Data Test — {ticker}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)

    # ── DB init ─────────────────────────────────────────────────────────────
    print("\n[0] Initialising database …")
    try:
        init_database()
        print("    ✓ Database ready")
    except Exception as e:
        print(f"    ⚠  DB init warning: {e}")

    state = create_initial_state(ticker)

    # ── Node 1: prices ───────────────────────────────────────────────────────
    print(f"\n[1] Fetching stock price data for {ticker} …")
    t0 = datetime.now()
    state = fetch_price_data_node(state)
    elapsed = (datetime.now() - t0).total_seconds()

    price_df = state.get("raw_price_data")
    print(f"\n{'─'*60}")
    print(f"  Stock price data — {ticker}")
    print(f"{'─'*60}")
    if price_df is not None and not price_df.empty:
        print(f"  Trading days:   {len(price_df)}")
        print(f"  Date range:     {price_df['date'].min()}  →  {price_df['date'].max()}")
        print(f"  Latest close:   ${price_df.iloc[-1]['close']:.2f}")
        print(f"  Fetch time:     {elapsed:.1f}s")
    else:
        print("  ✗ No price data returned")

    # ── Node 3: peers ────────────────────────────────────────────────────────
    print(f"\n[3] Fetching related companies (Finnhub peers) for {ticker} …")
    t0 = datetime.now()
    state = detect_related_companies_node(state)
    elapsed = (datetime.now() - t0).total_seconds()

    peers = state.get("related_companies", [])
    print(f"\n{'─'*60}")
    print(f"  Related companies (peers) — {ticker}")
    print(f"{'─'*60}")
    print(f"  Peers found:    {len(peers)}")
    print(f"  Tickers:        {', '.join(peers) if peers else 'none'}")
    print(f"  Fetch time:     {elapsed:.1f}s")

    # ── Node 2: news ─────────────────────────────────────────────────────────
    print(f"\n[2] Fetching 6-month news from Alpha Vantage …")
    print(f"    Stream 1 — stock news for {ticker}")
    print(f"    Stream 2 — global market news (financial_markets topic)")
    if peers:
        print(f"    Stream 3 — peer news for: {', '.join(peers)}")
    print(f"    Each stream = 6 windows × 1 000 articles (sort=RELEVANCE)")
    print(f"    (expect ~1–3 minutes)\n")

    t0 = datetime.now()
    state = fetch_all_news_node(state)
    elapsed = (datetime.now() - t0).total_seconds()

    stock_news   = state.get("stock_news", []) or []
    market_news  = state.get("market_news", []) or []
    related_news = state.get("related_company_news", []) or []

    # ── Results ──────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  NEWS RESULTS")
    print(SEP)

    print_stream(f"Stock news — {ticker}", stock_news)
    print_stream("Global market news (financial_markets)", market_news)
    print_stream(
        f"Related-company news — peers: {', '.join(peers) if peers else 'none'}",
        related_news,
    )

    all_news = stock_news + market_news + related_news
    total_all = len(all_news)
    total_with_s = sum(
        1 for a in all_news
        if a.get("overall_sentiment_label") is not None
        or a.get("overall_sentiment_score") is not None
    )
    total_missing = total_all - total_with_s

    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print(f"  Total articles (all streams):  {total_all}")
    print(f"  With AV sentiment:             {total_with_s}  ({total_with_s/total_all*100:.1f}% of total)" if total_all else "  n/a")
    print(f"  Missing sentiment:             {total_missing}  ({total_missing/total_all*100:.1f}% of total)" if total_all else "  n/a")
    print(f"  News fetch time:               {elapsed:.1f}s")

    if total_all > 0:
        missing_pct = total_missing / total_all * 100.0
        print()
        if missing_pct == 0:
            print("  ✓ Alpha Vantage provides sentiment for ALL articles.")
            print("    FinBERT is not needed for sentiment scoring.")
        elif missing_pct < 10:
            print(f"  ⚠  {missing_pct:.1f}% of articles lack AV sentiment.")
            print("    FinBERT optional — small gap, low priority.")
        elif missing_pct < 40:
            print(f"  ⚠  {missing_pct:.1f}% of articles lack AV sentiment.")
            print("    FinBERT recommended to fill the sentiment gap.")
        else:
            print(f"  ✗  {missing_pct:.1f}% of articles lack AV sentiment.")
            print("    FinBERT strongly recommended.")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
