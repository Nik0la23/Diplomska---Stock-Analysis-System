"""
Test Alpha Vantage 6-month historical stock news coverage and sentiment.

Usage:
    python scripts/test_alpha_vantage_6months.py
"""

import os
from datetime import datetime

from dotenv import load_dotenv

from src.graph.state import create_initial_state
from src.langgraph_nodes.node_02_news_fetching import fetch_all_news_node


def main() -> None:
    load_dotenv()

    ticker = os.getenv("TEST_TICKER", "AAPL").upper()
    print("=" * 80)
    print(f" Alpha Vantage 6-Month News Test for {ticker}")
    print("=" * 80)

    state = create_initial_state(ticker)

    # Run Node 2 directly – with an empty DB this should trigger a full 6‑month fetch.
    state = fetch_all_news_node(state)

    stock_news = state.get("stock_news", []) or []

    now_ts = datetime.now().timestamp()
    timestamps = [
        a.get("datetime", 0)
        for a in stock_news
        if isinstance(a.get("datetime", 0), (int, float))
    ]

    if timestamps:
        oldest_ts = min(timestamps)
        newest_ts = max(timestamps)
        oldest_days_back = (now_ts - oldest_ts) / 86400.0
        newest_days_back = (now_ts - newest_ts) / 86400.0
    else:
        oldest_days_back = None
        newest_days_back = None

    with_sentiment = [
        a
        for a in stock_news
        if a.get("overall_sentiment_label") is not None
        or a.get("overall_sentiment_score") is not None
    ]

    total = len(stock_news)
    with_sentiment_count = len(with_sentiment)
    pct_with_sentiment = (with_sentiment_count / total * 100.0) if total > 0 else 0.0

    print(f"\nTOTAL_STOCK_ARTICLES:        {total}")
    print(f"OLDEST_ARTICLE_DAYS_BACK:   {oldest_days_back:.1f}" if oldest_days_back is not None else "OLDEST_ARTICLE_DAYS_BACK:   None")
    print(f"NEWEST_ARTICLE_DAYS_BACK:   {newest_days_back:.1f}" if newest_days_back is not None else "NEWEST_ARTICLE_DAYS_BACK:   None")
    print(f"WITH_SENTIMENT_COUNT:       {with_sentiment_count}")
    print(f"SENTIMENT_COVERAGE_PERCENT: {pct_with_sentiment:.1f}%")


if __name__ == "__main__":
    main()

