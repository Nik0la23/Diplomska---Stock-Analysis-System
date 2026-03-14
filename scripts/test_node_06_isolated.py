#!/usr/bin/env python3
"""
Isolated test for Node 6: Market Context Analysis (FMP MCP, async).

Tests three things in sequence:
  1. MCP connection  — can we connect to FMP and list tools?
  2. Data checks     — do the key tools return parseable, non-empty data?
  3. Full node run   — does market_context_node() return a valid schema?

Usage:
    python scripts/test_node_06_isolated.py [TICKER]
    python scripts/test_node_06_isolated.py NVDA
    python scripts/test_node_06_isolated.py TSLA

Requirements:
    FMP_API_KEY and ANTHROPIC_API_KEY must be set in your .env file.
"""

import sys
import os
import json
import asyncio
import logging
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_node_06")

import pandas as pd
import yfinance as yf

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic

from src.langgraph_nodes.node_06_market_context import (
    market_context_node,
    _parse_tool_response,
    get_market_trend_multitimeframe_from_history,
    get_vix_level_from_price,
    classify_market_regime,
    compute_relative_strength,
    _derive_market_cap_tier,
    calculate_correlation,
    analyze_market_news_sentiment,
)

FMP_API_KEY   = os.getenv("FMP_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
FMP_URL       = f"https://financialmodelingprep.com/mcp?apikey={FMP_API_KEY}"

DIVIDER = "=" * 72


def hdr(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def ok(msg: str)   -> None: print(f"  ✅  {msg}")
def warn(msg: str) -> None: print(f"  ⚠️   {msg}")
def fail(msg: str) -> None: print(f"  ❌  {msg}")


def fetch_price_data(ticker: str, period: str = "90d") -> pd.DataFrame:
    """Fetch OHLCV from yfinance in the same format Node 1 produces."""
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        return pd.DataFrame()
    hist = hist.reset_index()
    hist.columns = [c.lower() for c in hist.columns]
    if "close" not in hist.columns:
        return pd.DataFrame()
    return hist[["date", "open", "high", "low", "close", "volume"]].copy()


# ============================================================================
# TEST 1: MCP CONNECTION
# ============================================================================

async def test_mcp_connection(client: MultiServerMCPClient) -> dict:
    """Verify that the FMP MCP session opens and tools load."""
    hdr("TEST 1 — MCP Connection & Tool Discovery")

    if not FMP_API_KEY:
        fail("FMP_API_KEY is not set — cannot connect")
        return {}

    try:
        tools = await client.get_tools()
        tools_by_name = {t.name: t for t in tools}

        ok(f"Connected to FMP MCP — {len(tools)} tools loaded")
        print(f"\n  All available tool names:")
        for name in sorted(tools_by_name.keys()):
            print(f"    • {name}")

        # Check for the tools Node 6 specifically needs
        required = ["profile-symbol", "historical-sector-performance", "historical-industry-performance", "quote"]
        optional = []  # commodities-quote removed from node 6 (402)

        print(f"\n  Required tool check:")
        all_present = True
        for t in required:
            if t in tools_by_name:
                ok(f"{t} — present")
            else:
                fail(f"{t} — MISSING")
                all_present = False

        if optional:
            print(f"\n  Optional tool check:")
            for t in optional:
                if t in tools_by_name:
                    ok(f"{t} — present")
                else:
                    warn(f"{t} — not found")

        print(f"\n  SPY historical price tool:")
        spy_tool_name = "historical-price-eod-light"
        if spy_tool_name in tools_by_name:
            ok(f"'{spy_tool_name}' — present (will be used for SPY history)")
        else:
            warn(f"'{spy_tool_name}' not found — SPY returns will fall back to zeros")

        return tools_by_name

    except Exception as e:
        fail(f"MCP connection failed: {e}")
        return {}


# ============================================================================
# TEST 2: DATA CHECKS (individual tool calls)
# ============================================================================

async def test_data_checks(tools_by_name: dict, ticker: str) -> None:
    """Call each key FMP tool and verify the response is parseable and non-empty."""
    hdr(f"TEST 2 — Individual FMP Tool Data Checks  ({ticker})")

    # --- profile-symbol (company profile) ---
    print(f"\n  [profile-symbol]")
    tool = tools_by_name.get("profile-symbol")
    if tool:
        try:
            raw = await tool.ainvoke({"symbol": ticker})
            print(f"  RAW profile-symbol response: {raw}")
            data = _parse_tool_response(raw)
            if isinstance(data, list):
                data = data[0] if data else {}
            company_name = data.get("companyName", "?")
            sector       = data.get("sector", "?")
            industry     = data.get("industry", "?")
            exchange     = data.get("exchange") or data.get("exchangeShortName") or "?"
            mkt_cap      = data.get("marketCap") or data.get("mktCap")
            beta         = data.get("beta")
            indices      = data.get("indexMemberships", [])
            ok(f"company_name    : {company_name}")
            ok(f"sector          : {sector}")
            ok(f"industry        : {industry}")
            ok(f"exchange        : {exchange}")
            ok(f"market_cap      : {mkt_cap}")
            ok(f"beta_fmp        : {beta}")
            ok(f"index_memberships: {indices}")
        except Exception as e:
            fail(f"profile-symbol failed: {e}")
    else:
        fail("profile-symbol tool not available")

    # --- historical-sector-performance (sector param; node 6 computes 5d from list) ---
    print(f"\n  [historical-sector-performance]")
    tool = tools_by_name.get("historical-sector-performance")
    if tool:
        try:
            raw = await tool.ainvoke({"sector": "Technology"})
            print(f"  RAW sector response: {str(raw)[:500]}")
            data = _parse_tool_response(raw)
            ok(f"Raw response (first 300 chars): {str(data)[:300]}")
            if isinstance(data, dict):
                ok(f"Keys: {list(data.keys())}")
            elif isinstance(data, list) and data:
                ok(f"List item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
                if len(data) >= 6:
                    latest = float(data[0].get("averageChange") or data[0].get("performance") or 0.0)
                    older = float(data[5].get("averageChange") or data[5].get("performance") or 0.0)
                    sector_return_5d = latest - older
                    ok(f"5d return (latest - older): {sector_return_5d:+.3f}%")
        except Exception as e:
            fail(f"historical-sector-performance failed: {e}")
    else:
        warn("historical-sector-performance not available")

    # --- historical-industry-performance (same pattern as sector) ---
    print(f"\n  [historical-industry-performance]")
    tool = tools_by_name.get("historical-industry-performance")
    if tool:
        try:
            raw = await tool.ainvoke({"industry": "Semiconductors"})
            data = _parse_tool_response(raw)
            ok(f"Raw response (first 300 chars): {str(data)[:300]}")
            if isinstance(data, list) and data:
                ok(f"List item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
                if len(data) >= 6:
                    latest = float(data[0].get("averageChange") or data[0].get("performance") or 0.0)
                    older = float(data[5].get("averageChange") or data[5].get("performance") or 0.0)
                    industry_return_5d = latest - older
                    ok(f"5d return (latest - older): {industry_return_5d:+.3f}%")
        except Exception as e:
            fail(f"historical-industry-performance failed: {e}")
    else:
        warn("historical-industry-performance not available")

    # --- quote for VIX ---
    print(f"\n  [quote — ^VIX]")
    tool = tools_by_name.get("quote")
    if tool:
        try:
            raw = await tool.ainvoke({"symbol": "^VIX"})
            print(f"  RAW quote response: {raw}")
            data = _parse_tool_response(raw)
            if isinstance(data, list) and data:
                price = data[0].get("price")
            elif isinstance(data, dict):
                price = data.get("price")
            else:
                price = None
            if price is not None:
                ok(f"VIX price: {price}")
                vix_info = get_vix_level_from_price(float(price))
                ok(f"VIX category: {vix_info['vix_category']}")
                ok(f"VIX interpretation: {vix_info['vix_interpretation']}")
            else:
                warn(f"price field not found in response: {str(data)[:200]}")
        except Exception as e:
            fail(f"quote(^VIX) failed: {e}")
    else:
        warn("quote tool not available")

    # --- SPY historical price tool (historical-price-eod-light) ---
    print(f"\n  [historical-price-eod-light — SPY]")
    spy_tool_name = "historical-price-eod-light"
    spy_tool = tools_by_name.get(spy_tool_name)
    if spy_tool:
        try:
            raw = await spy_tool.ainvoke({"symbol": "SPY", "days": 30})
            print(f"  RAW SPY history response (first 500 chars): {str(raw)[:500]}")
            data = _parse_tool_response(raw)
            if isinstance(data, list):
                closes = [
                    float(r.get("price") or r.get("close"))
                    for r in data
                    if isinstance(r, dict) and (r.get("price") is not None or r.get("close") is not None)
                ]
                closes.reverse()  # FMP newest-first; helper expects oldest-first
                ok(f"Tool '{spy_tool_name}' returned {len(closes)} closing prices")
                if closes:
                    ok(f"First close (oldest): {closes[0]:.2f}  Last close (newest): {closes[-1]:.2f}")
                    spy_data = get_market_trend_multitimeframe_from_history(closes)
                    ok(f"SPY 1d return  : {spy_data['spy_return_1d']:+.3f}%")
                    ok(f"SPY 5d return  : {spy_data['spy_return_5d']:+.3f}%")
                    ok(f"SPY 21d return : {spy_data['spy_return_21d']:+.3f}%")
                    ok(f"SPY trend label: {spy_data['spy_trend_label']}")
                    ok(f"Volatility     : {spy_data['market_volatility_pct']:.3f}%")
            else:
                warn(f"Unexpected response shape: {str(data)[:200]}")
        except Exception as e:
            fail(f"SPY historical tool failed: {e}")
    else:
        warn("historical-price-eod-light not found — this is expected if FMP plan doesn't include it")

    # --- commodities-quote (skipped in node 6 — 402) ---
    print(f"\n  [commodities-quote — skipped; node 6 does not fetch commodities]")
    # Node 6 keeps commodity_prices/commodity_trends empty; macro LLM still identifies factors.


# ============================================================================
# TEST 3: PURE PYTHON HELPERS
# ============================================================================

def test_pure_helpers(ticker: str, price_data: pd.DataFrame) -> None:
    """Test pure Python helpers without any network calls."""
    hdr("TEST 3 — Pure Python Helpers")

    # VIX categorization
    print(f"\n  [get_vix_level_from_price]")
    for level in [12.0, 17.0, 22.0, 27.0, 35.0]:
        info = get_vix_level_from_price(level)
        ok(f"VIX {level:.0f} → {info['vix_category']:10s} | {info['vix_interpretation']}")

    # Regime classifier
    print(f"\n  [classify_market_regime]")
    test_cases = [
        (1.5, 2.0, 15.0, "should be RISK_ON_BULL"),
        (0.0, 0.0, 12.0, "should be LOW_VOL_GRIND"),
        (0.5, 1.5, 25.0, "should be HIGH_VOL_BULL"),
        (-0.3, -0.2, 23.0, "should be DISTRIBUTION"),
        (-1.5, -2.0, 25.0, "should be RISK_OFF_BEAR"),
        (-1.0, -3.0, 31.0, "should be PANIC_SELLOFF"),
        (1.5, -1.0, 18.0, "should be RECOVERY"),
        (0.0, 0.0, 20.0, "should be NEUTRAL"),
    ]
    for spy_1d, spy_5d, vix, note in test_cases:
        regime = classify_market_regime(spy_1d, spy_5d, vix)
        ok(f"{regime['regime_label']:15s} ← spy_1d={spy_1d:+.1f}% spy_5d={spy_5d:+.1f}% vix={vix:.0f}  [{note}]")

    # Relative strength
    print(f"\n  [compute_relative_strength]")
    rs_cases = [
        (3.0, 5.0, 1.0, 2.0, "LEADING"),
        (1.0, 2.0, 1.2, 2.5, "IN_LINE"),
        (-0.5, -1.0, 1.0, 2.0, "LAGGING"),
    ]
    for s5, s21, sec5, sec21, expected in rs_cases:
        rs = compute_relative_strength(s5, s21, sec5, sec21)
        match = "✅" if rs["relative_strength_label"] == expected else "❌"
        ok(f"{match} stock_vs_sector_5d={rs['stock_vs_sector_5d']:+.2f}% → {rs['relative_strength_label']}  (expected {expected})")

    # Market cap tier
    print(f"\n  [_derive_market_cap_tier]")
    cap_cases = [
        (3_000_000_000_000, "mega"),
        (50_000_000_000, "large"),
        (5_000_000_000, "mid"),
        (500_000_000, "small"),
        (None, "Unknown"),
    ]
    for cap, expected in cap_cases:
        result = _derive_market_cap_tier(cap)
        match = "✅" if result == expected else "❌"
        ok(f"{match} cap={cap} → {result}  (expected {expected})")

    # Correlation helper
    print(f"\n  [calculate_correlation]")
    if not price_data.empty:
        corr = calculate_correlation(ticker, price_data)
        ok(f"market_correlation  : {corr['market_correlation']:+.4f}  ({corr['correlation_strength']})")
        ok(f"beta                : {corr['beta']:+.4f}")
    else:
        warn("No price data — skipping correlation test")

    # News sentiment
    print(f"\n  [analyze_market_news_sentiment]")
    fake_news = [
        {"datetime": datetime.now().timestamp(), "overall_sentiment_score": 0.25},
        {"datetime": datetime.now().timestamp(), "overall_sentiment_score": -0.10},
        {"datetime": datetime.now().timestamp(), "overall_sentiment_score": 0.30},
    ]
    result = analyze_market_news_sentiment(fake_news)
    ok(f"market_news_sentiment: {result['market_news_sentiment']:+.4f}")
    ok(f"market_news_count    : {result['market_news_count']}")

    result_empty = analyze_market_news_sentiment([])
    ok(f"Empty news → sentiment: {result_empty['market_news_sentiment']}  count: {result_empty['market_news_count']}")


# ============================================================================
# TEST 4: FULL NODE RUN
# ============================================================================

async def test_full_node(tools_by_name: dict, ticker: str, price_data: pd.DataFrame) -> None:
    """Run market_context_node() end-to-end and validate the output schema."""
    hdr(f"TEST 4 — Full market_context_node() Run  ({ticker})")

    if not ANTHROPIC_KEY:
        fail("ANTHROPIC_API_KEY not set — cannot run LLM steps")
        return

    llm = ChatAnthropic(model="claude-sonnet-4-20250514")

    config = {
        "configurable": {
            "tools_by_name":  tools_by_name,
            "llm":            llm,
            "llm_with_tools": None,  # Node 6 never uses Pattern B
        }
    }

    state = {
        "ticker":              ticker,
        "raw_price_data":      price_data if not price_data.empty else None,
        "cleaned_market_news": [],  # no Node 9A data in isolated test
    }

    print(f"\n  Running node... (this will make real FMP + LLM calls)")
    try:
        result = await market_context_node(state, config)
    except Exception as e:
        fail(f"Node raised an exception: {e}")
        return

    # --- Errors ---
    errors = result.get("errors", [])
    if errors:
        warn(f"Node returned {len(errors)} error(s):")
        for e in errors:
            print(f"    ⚠️  {e}")
    else:
        ok("No errors returned")

    # --- Timing ---
    elapsed = result.get("node_execution_times", {}).get("node_6", 0)
    ok(f"Wall-clock time: {elapsed:.2f}s")

    # --- Schema validation ---
    print(f"\n  Schema validation:")
    mc = result.get("market_context", {})
    if not mc:
        fail("market_context is empty or missing")
        return

    required_sub_dicts = [
        "stock_classification",
        "market_regime",
        "sector_industry_context",
        "macro_factor_exposure",
        "market_correlation_profile",
        "news_sentiment_context",
    ]

    required_fields = {
        "stock_classification": [
            "ticker", "company_name", "sector", "industry", "exchange",
            "index_memberships", "market_cap_tier", "beta_fmp",
        ],
        "market_regime": [
            "regime_label", "regime_description", "spy_return_1d", "spy_return_5d",
            "spy_return_21d", "spy_trend_label", "vix_level", "vix_category",
            "vix_interpretation", "market_volatility_pct",
        ],
        "sector_industry_context": [
            "sector_return_5d", "sector_trend", "industry_return_5d", "industry_trend",
            "stock_vs_sector_5d", "stock_vs_sector_21d", "relative_strength_label",
            "sector_context_note",
        ],
        "macro_factor_exposure": [
            "identified_factors", "commodity_prices", "commodity_trends", "macro_summary",
        ],
        "market_correlation_profile": [
            "market_correlation", "correlation_strength", "beta_calculated",
            "beta_interpretation", "correlation_note",
        ],
        "news_sentiment_context": [
            "market_news_sentiment", "market_news_count", "sentiment_label",
            "sentiment_interpretation",
        ],
    }

    all_valid = True
    for sub_dict in required_sub_dicts:
        section = mc.get(sub_dict, {})
        if not section:
            fail(f"{sub_dict} — MISSING")
            all_valid = False
            continue
        missing = [f for f in required_fields[sub_dict] if f not in section]
        if missing:
            fail(f"{sub_dict} — missing fields: {missing}")
            all_valid = False
        else:
            ok(f"{sub_dict} — all {len(required_fields[sub_dict])} fields present")

    # --- Print full output ---
    print(f"\n  Full market_context output:")

    sc = mc.get("stock_classification", {})
    print(f"\n  stock_classification:")
    print(f"    ticker          : {sc.get('ticker')}")
    print(f"    company_name    : {sc.get('company_name')}")
    print(f"    sector          : {sc.get('sector')}")
    print(f"    industry        : {sc.get('industry')}")
    print(f"    exchange        : {sc.get('exchange')}")
    print(f"    index_memberships: {sc.get('index_memberships')}")
    print(f"    market_cap_tier : {sc.get('market_cap_tier')}")
    print(f"    beta_fmp        : {sc.get('beta_fmp')}")

    mr = mc.get("market_regime", {})
    print(f"\n  market_regime:")
    print(f"    regime_label    : {mr.get('regime_label')}")
    print(f"    regime_desc     : {mr.get('regime_description')}")
    print(f"    spy 1d/5d/21d   : {mr.get('spy_return_1d'):+.3f}% / {mr.get('spy_return_5d'):+.3f}% / {mr.get('spy_return_21d'):+.3f}%")
    print(f"    spy_trend_label : {mr.get('spy_trend_label')}")
    print(f"    vix_level       : {mr.get('vix_level'):.2f}  ({mr.get('vix_category')})")
    print(f"    vix_interpret   : {mr.get('vix_interpretation')}")
    print(f"    volatility      : {mr.get('market_volatility_pct'):.3f}%")

    sic = mc.get("sector_industry_context", {})
    print(f"\n  sector_industry_context:")
    print(f"    sector_return_5d   : {sic.get('sector_return_5d'):+.3f}%  ({sic.get('sector_trend')})")
    print(f"    industry_return_5d : {sic.get('industry_return_5d'):+.3f}%  ({sic.get('industry_trend')})")
    print(f"    stock_vs_sector_5d : {sic.get('stock_vs_sector_5d'):+.3f}%")
    print(f"    stock_vs_sector_21d: {sic.get('stock_vs_sector_21d'):+.3f}%")
    print(f"    rel_strength       : {sic.get('relative_strength_label')}")
    print(f"    sector_note        : {sic.get('sector_context_note')}")

    mfe = mc.get("macro_factor_exposure", {})
    print(f"\n  macro_factor_exposure:")
    print(f"    identified_factors ({len(mfe.get('identified_factors', []))}):")
    for f in mfe.get("identified_factors", []):
        print(f"      • {f.get('factor_name')} [{f.get('exposure_type')}] — {f.get('exposure_explanation')}")
    print(f"    commodity_prices   : {mfe.get('commodity_prices')}")
    print(f"    commodity_trends   : {mfe.get('commodity_trends')}")
    print(f"    macro_summary      :\n      {mfe.get('macro_summary')}")

    mcp_out = mc.get("market_correlation_profile", {})
    print(f"\n  market_correlation_profile:")
    print(f"    correlation        : {mcp_out.get('market_correlation'):+.4f}  ({mcp_out.get('correlation_strength')})")
    print(f"    beta_calculated    : {mcp_out.get('beta_calculated'):+.4f}")
    print(f"    beta_interpretation: {mcp_out.get('beta_interpretation')}")
    print(f"    correlation_note   : {mcp_out.get('correlation_note')}")

    nsc = mc.get("news_sentiment_context", {})
    print(f"\n  news_sentiment_context:")
    print(f"    market_news_sentiment : {nsc.get('market_news_sentiment'):+.4f}  ({nsc.get('sentiment_label')})")
    print(f"    market_news_count     : {nsc.get('market_news_count')}")
    print(f"    sentiment_interp      : {nsc.get('sentiment_interpretation')}")

    print(f"\n  {'Schema OK ✅' if all_valid else 'Schema has issues ❌'}")

    # --- Fallback guard test ---
    print(f"\n  Fallback guard test (no tools_by_name):")
    bad_config = {"configurable": {"tools_by_name": {}, "llm": None, "llm_with_tools": None}}
    fallback = await market_context_node(state, bad_config)
    if fallback.get("errors"):
        ok(f"Guard triggered correctly — error: {fallback['errors'][0]}")
    else:
        fail("Guard did NOT trigger — something is wrong")

    mc_fb = fallback.get("market_context", {})
    if mc_fb.get("market_regime", {}).get("regime_label") == "NEUTRAL":
        ok("Fallback market_context has correct NEUTRAL defaults")
    else:
        fail("Fallback market_context is malformed")


# ============================================================================
# MAIN
# ============================================================================

async def main() -> None:
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"

    print(f"\n{'#' * 72}")
    print(f"#  NODE 6 ISOLATED TEST  —  {ticker}  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'#' * 72}")

    # Fetch price data once — used in helpers test and full node test
    hdr("PREP — Fetching price data (yfinance, bypassing Node 1)")
    price_data = fetch_price_data(ticker)
    print(f"  Price rows fetched: {len(price_data)}")

    if not FMP_API_KEY:
        fail("FMP_API_KEY not set — tests 1, 2, 4 will be skipped")
        test_pure_helpers(ticker, price_data)
        return

    # langchain-mcp-adapters >= 0.1.0: MultiServerMCPClient is NOT a context
    # manager. Just instantiate and call get_tools() directly.
    client = MultiServerMCPClient({
        "fmp": {
            "url": FMP_URL,
            "transport": "streamable_http",
        }
    })

    # Test 1 — MCP connection
    tools_by_name = await test_mcp_connection(client)
    if not tools_by_name:
        fail("Cannot proceed without MCP tools")
        return

    # Test 2 — Individual data checks
    await test_data_checks(tools_by_name, ticker)

    # Test 3 — Pure Python helpers (no network)
    test_pure_helpers(ticker, price_data)

    # Test 4 — Full node run
    await test_full_node(tools_by_name, ticker, price_data)

    print(f"\n{'#' * 72}")
    print(f"#  TEST COMPLETE  —  {datetime.now():%H:%M:%S}")
    print(f"{'#' * 72}\n")


if __name__ == "__main__":
    asyncio.run(main())