#!/usr/bin/env python3
"""
Isolated test for Node 6: Market Context Analysis.

Calls each helper function individually so every layer's contribution
is visible, then runs the full node and prints the final composite.

Usage:
    python scripts/test_node_06_isolated.py [TICKER]
    python scripts/test_node_06_isolated.py NVDA
    python scripts/test_node_06_isolated.py TSLA
"""

import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Enable INFO-level logging so all internal logs from the helpers are printed.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)

import yfinance as yf
import pandas as pd

from src.langgraph_nodes.node_06_market_context import (
    W_SPY, W_VIX, W_SECTOR, W_PEERS, W_NEWS,
    BUY_THRESHOLD, SELL_THRESHOLD,
    get_stock_sector,
    get_vix_level,
    get_market_trend_multitimeframe,
    get_sector_performance,
    analyze_related_companies,
    calculate_correlation,
    analyze_market_news_sentiment,
    compute_headwind_tailwind_score,
    market_context_node,
)

DIVIDER = "=" * 72


def hdr(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def fetch_minimal_price_data(ticker: str, period: str = "90d") -> pd.DataFrame:
    """Fetch OHLCV from yfinance and convert to the format Node 1 produces."""
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        return pd.DataFrame()
    hist = hist.reset_index()
    hist.columns = [c.lower() for c in hist.columns]
    hist = hist.rename(columns={"date": "date"})
    if "date" not in hist.columns and "Datetime" in hist.columns:
        hist = hist.rename(columns={"datetime": "date"})
    # Ensure 'close' column exists
    if "close" not in hist.columns:
        return pd.DataFrame()
    return hist[["date", "open", "high", "low", "close", "volume"]].copy()


def fetch_related_tickers(ticker: str) -> list:
    """Quick heuristic: grab a few well-known peers for common tickers."""
    peers_map = {
        "NVDA": ["AMD", "INTC", "QCOM", "TSM", "AVGO"],
        "TSLA": ["GM", "F", "RIVN", "LCID", "NIO"],
        "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
        "META": ["GOOGL", "SNAP", "PINS", "TWTR"],
        "MSFT": ["AAPL", "GOOGL", "AMZN", "CRM"],
    }
    return peers_map.get(ticker.upper(), [])


# ============================================================================
# MAIN
# ============================================================================

def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"

    print(f"\n{'#'*72}")
    print(f"#  NODE 6 ISOLATED TEST  —  {ticker}  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'#'*72}")

    # ------------------------------------------------------------------ #
    # LAYER 0: prep inputs
    # ------------------------------------------------------------------ #
    hdr("PREP: Fetching price data & related tickers (bypassing Nodes 1 & 3)")
    price_data = fetch_minimal_price_data(ticker)
    related_tickers = fetch_related_tickers(ticker)
    print(f"  Price rows fetched : {len(price_data)}")
    print(f"  Related tickers    : {related_tickers}")

    # ------------------------------------------------------------------ #
    # LAYER 1: Sector
    # ------------------------------------------------------------------ #
    hdr("LAYER 1/5 — Sector lookup (yfinance)")
    sector, industry = get_stock_sector(ticker)
    print(f"  Sector   : {sector}")
    print(f"  Industry : {industry}")

    # ------------------------------------------------------------------ #
    # LAYER 2: VIX  (weight 25%)
    # ------------------------------------------------------------------ #
    hdr(f"LAYER 2/5 — VIX fear gauge  [weight={W_VIX*100:.0f}%]")
    vix = get_vix_level()
    print(f"  VIX level      : {vix['vix_level']:.2f}")
    print(f"  VIX category   : {vix['vix_category']}")
    print(f"  Contribution   : {vix['vix_contribution']:+.3f}  →  weighted = {W_VIX * vix['vix_contribution']:+.4f}")

    # ------------------------------------------------------------------ #
    # LAYER 3: SPY multi-timeframe  (weight 35%)
    # ------------------------------------------------------------------ #
    hdr(f"LAYER 3/5 — SPY multi-timeframe  [weight={W_SPY*100:.0f}%]")
    spy = get_market_trend_multitimeframe()
    print(f"  SPY 1d return  : {spy['spy_return_1d']:+.3f}%")
    print(f"  SPY 5d return  : {spy['spy_return_5d']:+.3f}%")
    print(f"  SPY 21d return : {spy['spy_return_21d']:+.3f}%")
    print(f"  SPY composite  : {spy['spy_composite_score']:+.4f}  (0.2×1d + 0.4×5d + 0.4×21d)")
    print(f"  Market trend   : {spy['market_trend']}")
    print(f"  21d volatility : {spy['volatility']:.3f}%")
    print(f"  Contribution   : {spy['spy_composite_score']:+.3f}  →  weighted = {W_SPY * spy['spy_composite_score']:+.4f}")

    # ------------------------------------------------------------------ #
    # LAYER 4: Sector ETF  (weight 20%)
    # ------------------------------------------------------------------ #
    hdr(f"LAYER 4/5 — Sector ETF performance (5-day)  [weight={W_SECTOR*100:.0f}%]")
    sec = get_sector_performance(sector, days=5)
    print(f"  Sector ETF     : {sec['etf_ticker']}")
    print(f"  5d performance : {sec['performance']:+.3f}%")
    print(f"  Trend          : {sec['trend']}")
    print(f"  Sector score   : {sec['sector_score']:+.3f}  →  weighted = {W_SECTOR * sec['sector_score']:+.4f}")

    # ------------------------------------------------------------------ #
    # LAYER 5: Peers  (weight 10%)
    # ------------------------------------------------------------------ #
    hdr(f"LAYER 5/5 — Related-company peer performance  [weight={W_PEERS*100:.0f}%]")
    peers = analyze_related_companies(related_tickers)
    if peers['related_companies']:
        for p in peers['related_companies']:
            print(f"    {p['ticker']:6s}  {p['performance']:+.3f}%  ({p['trend']})")
    else:
        print("  No peer data available.")
    print(f"  Avg peer perf  : {peers['avg_performance']:+.3f}%")
    print(f"  Up / Down      : {peers['up_count']} up / {peers['down_count']} down")
    print(f"  Overall signal : {peers['overall_signal']}")
    print(f"  Peers score    : {peers['peers_score']:+.3f}  →  weighted = {W_PEERS * peers['peers_score']:+.4f}")

    # ------------------------------------------------------------------ #
    # BONUS: Correlation (not in headwind composite directly)
    # ------------------------------------------------------------------ #
    hdr("BONUS — Stock–SPY correlation (informational only, not in composite)")
    corr = calculate_correlation(ticker, price_data)
    print(f"  Pearson corr   : {corr['market_correlation']:+.4f}  ({corr['correlation_strength']})")
    print(f"  Beta           : {corr['beta']:+.4f}")

    # ------------------------------------------------------------------ #
    # News layer  (weight 10%)
    # ------------------------------------------------------------------ #
    hdr(f"NEWS layer — market_news sentiment  [weight={W_NEWS*100:.0f}%]  (no cleaned_market_news in isolated test)")
    news = analyze_market_news_sentiment([])
    print(f"  Articles used  : {news['market_news_count']}  (0 = no state from Node 9A)")
    print(f"  News sentiment : {news['market_news_sentiment']:+.4f}  →  weighted = {W_NEWS * news['market_news_sentiment']:+.4f}")

    # ------------------------------------------------------------------ #
    # COMPOSITE
    # ------------------------------------------------------------------ #
    hdr("COMPOSITE HEADWIND / TAILWIND SCORE")
    sig = compute_headwind_tailwind_score(
        spy_composite    = spy['spy_composite_score'],
        vix_contribution = vix['vix_contribution'],
        sector_score     = sec['sector_score'],
        peers_score      = peers['peers_score'],
        news_sentiment   = news['market_news_sentiment'],
    )

    w_spy_c    = W_SPY    * spy['spy_composite_score']
    w_vix_c    = W_VIX    * vix['vix_contribution']
    w_sec_c    = W_SECTOR * sec['sector_score']
    w_peers_c  = W_PEERS  * peers['peers_score']
    w_news_c   = W_NEWS   * news['market_news_sentiment']
    raw_total  = w_spy_c + w_vix_c + w_sec_c + w_peers_c + w_news_c

    print(f"\n  Contribution breakdown:")
    print(f"    SPY    ({W_SPY*100:.0f}%):  {spy['spy_composite_score']:+.4f}  ×  {W_SPY:.2f}  =  {w_spy_c:+.6f}")
    print(f"    VIX    ({W_VIX*100:.0f}%):  {vix['vix_contribution']:+.4f}  ×  {W_VIX:.2f}  =  {w_vix_c:+.6f}")
    print(f"    Sector ({W_SECTOR*100:.0f}%):  {sec['sector_score']:+.4f}  ×  {W_SECTOR:.2f}  =  {w_sec_c:+.6f}")
    print(f"    Peers  ({W_PEERS*100:.0f}%):  {peers['peers_score']:+.4f}  ×  {W_PEERS:.2f}  =  {w_peers_c:+.6f}")
    print(f"    News   ({W_NEWS*100:.0f}%):  {news['market_news_sentiment']:+.4f}  ×  {W_NEWS:.2f}  =  {w_news_c:+.6f}")
    print(f"    {'─'*50}")
    print(f"    RAW TOTAL               =  {raw_total:+.6f}")
    print(f"    Clamped to [-1,+1]      =  {sig['market_headwind_score']:+.6f}")
    print(f"\n  context_signal  : {sig['context_signal']}  (BUY≥{BUY_THRESHOLD}, SELL≤{SELL_THRESHOLD})")
    print(f"  confidence      : {sig['confidence']:.1f}%")

    # ------------------------------------------------------------------ #
    # FULL NODE CALL (with minimal fake state)
    # ------------------------------------------------------------------ #
    hdr("FULL market_context_node() CALL  (minimal state, no news)")
    minimal_state = {
        "ticker":              ticker,
        "raw_price_data":      price_data if not price_data.empty else None,
        "related_companies":   related_tickers,
        "cleaned_market_news": [],
    }

    node_result = market_context_node(minimal_state)
    mc = node_result.get("market_context", {})

    print(f"\n  ── market_context dict ──")
    for k, v in mc.items():
        if isinstance(v, list):
            print(f"    {k:<35} : [{len(v)} items]")
        elif isinstance(v, float):
            print(f"    {k:<35} : {v:+.4f}")
        else:
            print(f"    {k:<35} : {v}")

    elapsed = node_result.get("node_execution_times", {}).get("node_6", 0)
    print(f"\n  Node 6 wall-clock time : {elapsed:.2f}s")

    if node_result.get("errors"):
        print(f"\n  ERRORS:")
        for e in node_result["errors"]:
            print(f"    {e}")

    print(f"\n{'#'*72}")
    print(f"#  TEST COMPLETE  —  {datetime.now():%H:%M:%S}")
    print(f"{'#'*72}\n")


if __name__ == "__main__":
    main()
