"""
Market-news sieve diagnostic.

Runs each filter stage in isolation to show exactly how many dates survive
each step for a given ticker.  Mirrors the logic in backtest_sentiment_stream()
so numbers here match what Node 10 actually sees.

Usage:
    python -m scripts.db_market_eating_test          # default: AAPL, 180 days
    python -m scripts.db_market_eating_test TSLA 365
"""

import sys
import numpy as np
import pandas as pd

from src.database.db_manager import get_news_with_outcomes
from src.langgraph_nodes.node_10_backtesting import (
    aggregate_daily_sentiment,
    aggregate_daily_sentiment_weighted,
    build_sentiment_baseline,
    MIN_SENTIMENT_MAGNITUDE,
)


def run(ticker: str = "AAPL", days: int = 180) -> None:
    print(f"\n=== Market-news sieve diagnostic: {ticker} ({days} days) ===\n")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = get_news_with_outcomes(ticker, days=days)
    if df.empty:
        print("No data returned from DB — run the pipeline first.")
        return

    news_events = df.to_dict("records")
    market_events = [e for e in news_events if e.get("news_type") == "market"]

    print(f"Total articles in DB ({days}d):   {len(news_events)}")
    print(f"Market-news articles:            {len(market_events)}")

    if not market_events:
        print("No market-news articles found.")
        return

    # ------------------------------------------------------------------
    # Sieve 0: group by date
    # ------------------------------------------------------------------
    by_date: dict = {}
    for e in market_events:
        pub = e.get("published_at", "")
        d = str(pub)[:10]
        if d:
            by_date.setdefault(d, []).append(e)

    total_dates = len(by_date)
    print(f"\nSieve 0 — unique dates with market news: {total_dates}")

    # ------------------------------------------------------------------
    # Sieve 1: date has at least one article (already implied by grouping)
    # ------------------------------------------------------------------
    has_articles = sum(1 for v in by_date.values() if len(v) > 0)
    print(f"Sieve 1 — dates with ≥1 article:         {has_articles}")

    # ------------------------------------------------------------------
    # Sieve 2: published_at is non-empty / parseable
    # ------------------------------------------------------------------
    parseable = 0
    for d in by_date:
        try:
            pd.to_datetime(d)
            parseable += 1
        except Exception:
            pass
    print(f"Sieve 2 — dates with parseable timestamp: {parseable}")

    # ------------------------------------------------------------------
    # Sieve 3: at least one article on that date has price_change_7day
    # ------------------------------------------------------------------
    has_outcome = sum(
        1 for articles in by_date.values()
        if any(a.get("price_change_7day") is not None for a in articles)
    )
    print(f"Sieve 3 — dates with price_change_7day:  {has_outcome}  "
          f"(lost {total_dates - has_outcome})")

    # ------------------------------------------------------------------
    # Sieve 4: |raw_score| >= MIN_SENTIMENT_MAGNITUDE  (unweighted, no baseline)
    # ------------------------------------------------------------------
    above_magnitude_raw = sum(
        1 for articles in by_date.values()
        if abs(aggregate_daily_sentiment(articles)) >= MIN_SENTIMENT_MAGNITUDE
    )
    print(f"Sieve 4a — |raw score| >= {MIN_SENTIMENT_MAGNITUDE} (unweighted, no baseline): "
          f"{above_magnitude_raw}  (lost {has_outcome - above_magnitude_raw})")

    # ------------------------------------------------------------------
    # Sieve 4b: |corrected_score| >= threshold  (weighted + baseline, as Node 10 does)
    # ------------------------------------------------------------------
    src_rel: dict = {}  # no source_reliability outside a full pipeline run
    baseline_by_date = build_sentiment_baseline(by_date, src_rel)

    above_magnitude_corrected = 0
    skipped_no_outcome = 0
    skipped_magnitude = 0

    for date_str, day_articles in sorted(by_date.items()):
        price_changes = [
            float(a["price_change_7day"])
            for a in day_articles
            if a.get("price_change_7day") is not None
        ]
        if not price_changes:
            skipped_no_outcome += 1
            continue

        raw_score = aggregate_daily_sentiment_weighted(day_articles, src_rel)
        baseline  = baseline_by_date.get(date_str, 0.0)
        corrected = raw_score - baseline

        if abs(corrected) < MIN_SENTIMENT_MAGNITUDE:
            skipped_magnitude += 1
        else:
            above_magnitude_corrected += 1

    print(f"Sieve 4b — |corrected score| >= {MIN_SENTIMENT_MAGNITUDE} (weighted + baseline): "
          f"{above_magnitude_corrected}  (lost {skipped_magnitude} to magnitude, "
          f"{skipped_no_outcome} to missing outcome)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n--- Summary ---")
    print(f"  Started with:              {total_dates} dates")
    print(f"  Lost to missing outcome:   {total_dates - has_outcome}")
    print(f"  Lost to low magnitude:     {skipped_magnitude}")
    print(f"  Surviving (Node 10 sees):  {above_magnitude_corrected}")

    if total_dates > 0:
        survival_pct = above_magnitude_corrected / total_dates * 100
        print(f"  Survival rate:             {survival_pct:.1f}%")

    # ------------------------------------------------------------------
    # Bonus: distribution of raw scores to show where the mass sits
    # ------------------------------------------------------------------
    raw_scores = [aggregate_daily_sentiment(v) for v in by_date.values()]
    arr = np.array(raw_scores)
    print(f"\n--- Raw score distribution across {total_dates} dates ---")
    print(f"  mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"std={arr.std():.4f}")
    print(f"  min={arr.min():.4f}  max={arr.max():.4f}")
    pct_below = (np.abs(arr) < MIN_SENTIMENT_MAGNITUDE).mean() * 100
    print(f"  |score| < {MIN_SENTIMENT_MAGNITUDE}: {pct_below:.1f}% of dates  "
          f"← this is Sieve 4's kill rate")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    days   = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    run(ticker, days)
