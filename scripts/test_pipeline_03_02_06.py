#!/usr/bin/env python3
"""
Isolated pipeline test: Node 3 → Node 2 → Node 6.

Shows state before/after each node, LLM prompts & responses,
and a summary of what each node produced.

Usage:
    python scripts/test_pipeline_03_02_06.py [TICKER]
    python scripts/test_pipeline_03_02_06.py NVDA
    python scripts/test_pipeline_03_02_06.py TSLA

Requirements:
    FMP_API_KEY, ANTHROPIC_API_KEY, and ALPHA_VANTAGE_API_KEY must be
    set in your .env file.

Notes:
    - Node 2 is a synchronous function that calls asyncio.run() internally;
      it is executed in a ThreadPoolExecutor so the test event loop is
      not blocked or nested.
    - Node 6 receives cleaned_market_news = market_news (Node 9A, which
      normally cleans the news, is not part of this pipeline slice).
    - raw_price_data is not available (Node 1 not in scope); Node 6's
      correlation and stock-return paths use their safe fallback values.
"""

import sys
import os
import json
import asyncio
import logging
import concurrent.futures
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_pipeline_03_02_06")

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic

from src.langgraph_nodes.node_03_related_companies import (
    detect_related_companies_node,
    PEER_CLASSIFICATION_PROMPT,
)
from src.langgraph_nodes.node_02_news_fetching import fetch_all_news_node
from src.langgraph_nodes.node_06_market_context import (
    market_context_node,
    MACRO_FACTORS_PROMPT,
)

FMP_API_KEY   = os.getenv("FMP_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
FMP_URL       = f"https://financialmodelingprep.com/mcp?apikey={FMP_API_KEY}"

W = 72
DIVIDER = "=" * W


# ============================================================================
# PRINT HELPERS
# ============================================================================

def hdr(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def section(title: str) -> None:
    print(f"\n  {'─' * 64}")
    print(f"  {title}")
    print(f"  {'─' * 64}")


def ok(msg: str)   -> None: print(f"  ✅  {msg}")
def warn(msg: str) -> None: print(f"  ⚠️   {msg}")
def fail(msg: str) -> None: print(f"  ❌  {msg}")


def _fmt_val(val) -> str:
    if isinstance(val, list):
        return f"list[{len(val)}]"
    if isinstance(val, dict):
        keys = list(val.keys())[:5]
        more = "…" if len(val) > 5 else ""
        return f"dict  keys={keys}{more}"
    if isinstance(val, str):
        return f"str   {repr(val[:80])}"
    if val is None:
        return "None"
    import pandas as pd
    if isinstance(val, pd.DataFrame):
        return f"DataFrame  shape={val.shape}"
    return f"{type(val).__name__}  {val}"


def print_state_snapshot(label: str, state: dict) -> None:
    section(f"STATE SNAPSHOT — {label}")
    for key in sorted(state.keys()):
        print(f"    {key:<36} {_fmt_val(state[key])}")


# ============================================================================
# NODE-SPECIFIC PRINTERS
# ============================================================================

def show_node3_input(state: dict, ticker: str) -> None:
    section("NODE 3 — INPUT")
    print(f"    ticker               : {ticker}")
    print(f"    errors               : {state['errors']}")
    print(f"    node_execution_times : {state['node_execution_times']}")
    section("NODE 3 — LLM PROMPT TEMPLATE (filled at runtime)")
    print(f"    Prompt variables injected by Node 3:")
    print(f"      {{ticker}}   → {ticker}")
    print(f"      {{sector}}   → fetched via FMP profile-symbol")
    print(f"      {{industry}} → fetched via FMP profile-symbol")
    print(f"      {{peers}}    → fetched via FMP peers (up to 5)")
    print(f"\n    Full template:")
    for line in PEER_CLASSIFICATION_PROMPT.strip().split("\n"):
        print(f"      {line}")


def show_node3_output(result: dict) -> None:
    related = result.get("related_companies", [])
    elapsed = result.get("node_execution_times", {}).get("node_3", 0)

    section("NODE 3 — OUTPUT (LLM RESPONSE — parsed classifications)")
    if not related:
        warn("No related companies returned")
    else:
        ok(f"{len(related)} companies classified by LLM")
        print(f"\n    {'Ticker':<8}  {'Relationship':<14}  Reason")
        print(f"    {'─'*8}  {'─'*14}  {'─'*46}")
        for c in related:
            print(
                f"    {c.get('ticker','?'):<8}  "
                f"{c.get('relationship','?'):<14}  "
                f"{c.get('reason','')}"
            )

    section("NODE 3 — TIMING")
    ok(f"Wall-clock time: {elapsed:.2f}s")

    if result.get("errors"):
        for e in result["errors"]:
            warn(f"Error: {e}")


def show_node2_input(state: dict, ticker: str) -> None:
    related = state.get("related_companies", [])
    peer_tickers = [
        c["ticker"] if isinstance(c, dict) else c
        for c in related
    ]
    section("NODE 2 — INPUT")
    print(f"    ticker                   : {ticker}")
    print(f"    related_companies count  : {len(related)}")
    print(f"    peer tickers for news    : {peer_tickers}")
    print(f"\n    Node 2 fetches from Alpha Vantage:")
    print(f"      • stock news  — ticker={ticker}, 6-month window, chunked by 30d")
    print(f"      • market news — topic=financial_markets, 6-month window, chunked by 30d")
    print(f"      • related co. — 1 call per peer (sort=RELEVANCE, limit=100)")
    print(f"\n    (No LLM calls in Node 2 — pure API data fetching)")


def show_node2_output(state: dict) -> None:
    stock_news   = state.get("stock_news", [])
    market_news  = state.get("market_news", [])
    related_news = state.get("related_company_news", [])
    metadata     = state.get("news_fetch_metadata", {})
    elapsed      = state.get("node_execution_times", {}).get("node_2", 0)

    section("NODE 2 — OUTPUT — article counts")
    ok(f"stock_news           : {len(stock_news)} articles")
    ok(f"market_news          : {len(market_news)} articles")
    ok(f"related_company_news : {len(related_news)} articles")

    section("NODE 2 — OUTPUT — news_fetch_metadata")
    for k, v in (metadata or {}).items():
        print(f"    {k:<32} {v}")

    section("NODE 2 — OUTPUT — sample stock headlines (first 3)")
    for i, art in enumerate(stock_news[:3], 1):
        ts  = datetime.fromtimestamp(art.get("datetime", 0)).strftime("%Y-%m-%d")
        src = str(art.get("source", "?"))
        print(f"    [{i}] [{ts}] [{src:<20}] {art.get('headline','')[:60]}")
        print(
            f"         overall_sentiment={art.get('overall_sentiment_score','?')}"
            f"  ticker_sentiment={art.get('ticker_sentiment_score','?')}"
            f"  relevance={art.get('ticker_relevance_score','?')}"
        )

    section("NODE 2 — OUTPUT — sample market headlines (first 3)")
    for i, art in enumerate(market_news[:3], 1):
        ts  = datetime.fromtimestamp(art.get("datetime", 0)).strftime("%Y-%m-%d")
        src = str(art.get("source", "?"))
        print(f"    [{i}] [{ts}] [{src:<20}] {art.get('headline','')[:60]}")
        print(f"         overall_sentiment={art.get('overall_sentiment_score','?')}")

    if related_news:
        section("NODE 2 — OUTPUT — sample related-company headlines (first 3)")
        for i, art in enumerate(related_news[:3], 1):
            ts  = datetime.fromtimestamp(art.get("datetime", 0)).strftime("%Y-%m-%d")
            src = str(art.get("source", "?"))
            rel = str(art.get("related_ticker", "?"))
            print(f"    [{i}] [{ts}] [{rel}] [{src:<18}] {art.get('headline','')[:55]}")

    section("NODE 2 — TIMING")
    ok(f"Wall-clock time: {elapsed:.2f}s")


def show_node6_input(state: dict, ticker: str) -> None:
    related = state.get("related_companies", [])
    cleaned = state.get("cleaned_market_news", [])
    price   = state.get("raw_price_data")
    section("NODE 6 — INPUT")
    print(f"    ticker                   : {ticker}")
    print(f"    raw_price_data           : {'DataFrame ' + str(price.shape) if price is not None else 'None  (Node 1 not in scope)'}")  # noqa: E501
    print(f"    related_companies count  : {len(related)}")
    print(f"    cleaned_market_news count: {len(cleaned)}  (= market_news, Node 9A not in scope)")
    print(f"\n    FMP tools Node 6 will call:")
    print(f"      • profile-symbol                 — company name, sector, industry, beta, market cap")
    print(f"      • historical-sector-performance  — sector 5d return")
    print(f"      • historical-industry-performance— industry 5d return")
    print(f"      • historical-price-eod-light      — SPY closes (30d) for regime calc")
    print(f"      • quote                           — ^VIX current price")
    print(f"\n    LLM calls Node 6 will make:")
    print(f"      1. MACRO_FACTORS_PROMPT  → identifies top macro factors (Pattern C)")
    print(f"      2. macro_summary prompt  → plain-text implication paragraph (Pattern C)")
    section("NODE 6 — LLM PROMPT TEMPLATES")
    print(f"\n    [1] MACRO_FACTORS_PROMPT (variables: ticker, company_name, sector, industry):")
    for line in MACRO_FACTORS_PROMPT.strip().split("\n"):
        print(f"      {line}")
    print(f"\n    [2] macro_summary prompt (built at runtime from macro factors + commodity prices + peers)")


def show_node6_output(result: dict) -> None:
    mc      = result.get("market_context", {})
    elapsed = result.get("node_execution_times", {}).get("node_6", 0)

    section("NODE 6 — OUTPUT — stock_classification")
    for k, v in mc.get("stock_classification", {}).items():
        print(f"    {k:<24} {v}")

    section("NODE 6 — OUTPUT — market_regime")
    for k, v in mc.get("market_regime", {}).items():
        print(f"    {k:<32} {v}")

    section("NODE 6 — OUTPUT — sector_industry_context")
    for k, v in mc.get("sector_industry_context", {}).items():
        print(f"    {k:<36} {v}")

    section("NODE 6 — OUTPUT — market_correlation_profile")
    for k, v in mc.get("market_correlation_profile", {}).items():
        print(f"    {k:<32} {v}")

    section("NODE 6 — OUTPUT — news_sentiment_context")
    for k, v in mc.get("news_sentiment_context", {}).items():
        print(f"    {k:<32} {v}")

    mfe = mc.get("macro_factor_exposure", {})
    factors = mfe.get("identified_factors", [])

    section("NODE 6 — LLM RESPONSE [1] — macro factors (identified_factors)")
    ok(f"{len(factors)} macro factors identified by LLM:")
    for i, f in enumerate(factors, 1):
        print(f"\n    [{i}] factor_name         : {f.get('factor_name','?')}")
        print(f"         fmp_commodity_symbol: {f.get('fmp_commodity_symbol', 'null')}")
        print(f"         exposure_type       : {f.get('exposure_type','?')}")
        print(f"         exposure_explanation: {f.get('exposure_explanation','?')}")

    section("NODE 6 — LLM RESPONSE [2] — macro_summary (full paragraph)")
    print(f"\n{mfe.get('macro_summary', 'Unavailable.')}\n")

    section("NODE 6 — TIMING")
    ok(f"Wall-clock time: {elapsed:.2f}s")

    if result.get("errors"):
        for e in result["errors"]:
            warn(f"Error: {e}")


# ============================================================================
# STATE HELPERS
# ============================================================================

def _merge_state(state: dict, partial: dict) -> None:
    """
    Merge a node's partial return into state, mirroring LangGraph's reducer.
    node_execution_times is merged (not overwritten) so cumulative timing is
    preserved across nodes.
    """
    for k, v in partial.items():
        if k == "node_execution_times" and isinstance(v, dict):
            state.setdefault("node_execution_times", {}).update(v)
        elif k == "errors" and isinstance(v, list):
            state.setdefault("errors", []).extend(v)
        else:
            state[k] = v


# ============================================================================
# MAIN
# ============================================================================

async def main() -> None:
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"

    print(f"\n{'#' * W}")
    print(f"#  PIPELINE TEST: Node 3 → 2 → 6  —  {ticker}  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'#' * W}")

    # -- Guard env vars -------------------------------------------------------
    missing = []
    if not FMP_API_KEY:
        missing.append("FMP_API_KEY")
    if not ANTHROPIC_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        for m in missing:
            fail(f"{m} is not set")
        return

    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------
    hdr("SETUP — FMP MCP + LLM")

    client = MultiServerMCPClient({
        "fmp": {
            "url":       FMP_URL,
            "transport": "streamable_http",
        }
    })

    tools = await client.get_tools()
    tools_by_name = {t.name: t for t in tools}
    ok(f"FMP MCP connected — {len(tools)} tools loaded")
    print(f"\n  Available tools: {sorted(tools_by_name.keys())}")

    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    llm_with_tools = llm.bind_tools(list(tools_by_name.values()))
    ok("ChatAnthropic LLM ready (claude-sonnet-4-20250514)")

    config = {
        "configurable": {
            "tools_by_name":  tools_by_name,
            "llm":            llm,
            "llm_with_tools": llm_with_tools,
        }
    }

    # -------------------------------------------------------------------------
    # INITIAL STATE
    # -------------------------------------------------------------------------
    state: dict = {
        "ticker":               ticker,
        "errors":               [],
        "node_execution_times": {},
    }
    print_state_snapshot("INITIAL", state)

    # =========================================================================
    # NODE 3
    # =========================================================================
    hdr(f"NODE 3 — Related Companies Detection  ({ticker})")

    show_node3_input(state, ticker)

    print(f"\n  Running detect_related_companies_node() ...")
    node3_result = await detect_related_companies_node(state, config)
    _merge_state(state, node3_result)

    show_node3_output(node3_result)
    print_state_snapshot("AFTER NODE 3", state)

    # =========================================================================
    # NODE 2
    # =========================================================================
    hdr(f"NODE 2 — News Fetching  ({ticker})")

    show_node2_input(state, ticker)

    print(f"\n  Running fetch_all_news_node() in thread executor ...")
    print(f"  (Node 2 is synchronous and calls asyncio.run() internally)")
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        node2_result = await loop.run_in_executor(pool, fetch_all_news_node, state)
    _merge_state(state, node2_result)

    show_node2_output(state)

    if state.get("errors"):
        for e in state["errors"]:
            warn(f"Error after Node 2: {e}")

    print_state_snapshot("AFTER NODE 2", state)

    # =========================================================================
    # NODE 6  (needs cleaned_market_news; use market_news as stand-in)
    # =========================================================================
    hdr(f"NODE 6 — Market Context Analysis  ({ticker})")

    # Node 9A (content analysis) normally populates cleaned_market_news.
    # Since 9A is out of scope here we pass market_news directly.
    state["cleaned_market_news"] = state.get("market_news", [])

    show_node6_input(state, ticker)

    print(f"\n  Running market_context_node() ...")
    node6_result = await market_context_node(state, config)
    _merge_state(state, node6_result)

    show_node6_output(node6_result)

    if state.get("errors"):
        for e in state["errors"]:
            warn(f"Error after Node 6: {e}")

    print_state_snapshot("AFTER NODE 6 (FINAL)", state)

    # =========================================================================
    # PIPELINE SUMMARY
    # =========================================================================
    hdr("PIPELINE SUMMARY")
    times = state.get("node_execution_times", {})
    related   = state.get("related_companies", [])
    stock_n   = state.get("stock_news", [])
    market_n  = state.get("market_news", [])
    related_n = state.get("related_company_news", [])
    mc        = state.get("market_context", {})

    ok(
        f"Node 3 : {times.get('node_3', 0):>6.2f}s  "
        f"→ {len(related)} related companies: "
        f"{[c['ticker'] for c in related]}"
    )
    ok(
        f"Node 2 : {times.get('node_2', 0):>6.2f}s  "
        f"→ {len(stock_n)} stock + {len(market_n)} market + {len(related_n)} related articles"
    )
    ok(
        f"Node 6 : {times.get('node_6', 0):>6.2f}s  "
        f"→ regime={mc.get('market_regime', {}).get('regime_label', '?')}  "
        f"vix_cat={mc.get('market_regime', {}).get('vix_category', '?')}  "
        f"rel_strength={mc.get('sector_industry_context', {}).get('relative_strength_label', '?')}"
    )
    total = sum(times.get(k, 0) for k in ["node_3", "node_2", "node_6"])
    ok(f"Total  : {total:>6.2f}s")

    errs = state.get("errors", [])
    if errs:
        print(f"\n  Errors accumulated ({len(errs)}):")
        for e in errs:
            warn(f"  {e}")
    else:
        ok("No errors accumulated across the pipeline")

    print(f"\n{'#' * W}")
    print(f"#  TEST COMPLETE  —  {datetime.now():%H:%M:%S}")
    print(f"{'#' * W}\n")


if __name__ == "__main__":
    asyncio.run(main())
