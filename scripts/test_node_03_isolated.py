#!/usr/bin/env python3
"""
Isolated test for Node 3: Related Companies Detection (FMP MCP + LLM, async).

Tests three things in sequence:
  1. MCP connection  — can we connect to FMP and list tools?
  2. Data checks     — do peers and profile-symbol return parseable data?
  3. Full node run   — does detect_related_companies_node() return a valid schema?

Usage:
    python scripts/test_node_03_isolated.py [TICKER]
    python scripts/test_node_03_isolated.py NVDA
    python scripts/test_node_03_isolated.py TSLA

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
logger = logging.getLogger("test_node_03")

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic

from src.langgraph_nodes.node_03_related_companies import (
    detect_related_companies_node,
    _parse_tool_response,
    _extract_tickers,
    _parse_classifications,
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


# ============================================================================
# TEST 1: MCP CONNECTION
# ============================================================================

async def test_mcp_connection(client: MultiServerMCPClient) -> dict:
    """Verify FMP MCP session opens and the tools Node 3 needs are present."""
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

        print(f"\n  Required tool check (Node 3):")
        required = ["peers", "profile-symbol"]
        all_present = True
        for t in required:
            if t in tools_by_name:
                ok(f"{t} — present")
            else:
                fail(f"{t} — MISSING")
                all_present = False

        if not all_present:
            warn("One or more required tools are missing — node will fall back to PEERS_FALLBACK")

        return tools_by_name

    except Exception as e:
        fail(f"MCP connection failed: {e}")
        return {}


# ============================================================================
# TEST 2: INDIVIDUAL TOOL DATA CHECKS
# ============================================================================

async def test_data_checks(tools_by_name: dict, ticker: str) -> None:
    """Call peers and profile-symbol directly and verify responses."""
    hdr(f"TEST 2 — Individual FMP Tool Data Checks  ({ticker})")

    # --- peers ---
    print(f"\n  [peers]")
    peers_tool = tools_by_name.get("peers")
    if peers_tool:
        try:
            raw = await peers_tool.ainvoke({"symbol": ticker})
            print(f"  RAW peers response: {raw}")
            parsed = _parse_tool_response(raw)
            tickers = _extract_tickers(parsed)
            tickers = [t for t in tickers if t != ticker.upper()]
            ok(f"Parsed {len(tickers)} peer tickers: {tickers}")
        except Exception as e:
            fail(f"peers failed: {e}")
    else:
        warn("peers tool not found — node will use PEERS_FALLBACK")

    # --- profile-symbol ---
    print(f"\n  [profile-symbol]")
    profile_tool = tools_by_name.get("profile-symbol")
    if profile_tool:
        try:
            raw = await profile_tool.ainvoke({"symbol": ticker})
            print(f"  RAW profile-symbol response (first 500 chars): {str(raw)[:500]}")
            parsed = _parse_tool_response(raw)
            if isinstance(parsed, list) and parsed:
                profile = parsed[0]
            elif isinstance(parsed, dict):
                profile = parsed
            else:
                profile = {}
            ok(f"companyName : {profile.get('companyName', '?')}")
            ok(f"sector      : {profile.get('sector', '?')}")
            ok(f"industry    : {profile.get('industry', '?')}")
            ok(f"exchange    : {profile.get('exchange', '?')}")
        except Exception as e:
            fail(f"profile-symbol failed: {e}")
    else:
        warn("profile-symbol tool not found — LLM prompt will use 'Unknown' sector/industry")


# ============================================================================
# TEST 3: PURE HELPER UNIT CHECKS
# ============================================================================

def test_pure_helpers() -> None:
    """Quick sanity-check of _extract_tickers and _parse_classifications."""
    hdr("TEST 3 — Pure Helper Unit Checks")

    # _extract_tickers
    print(f"\n  [_extract_tickers]")
    cases = [
        (["AMD", "INTC", "QCOM"],                       ["AMD", "INTC", "QCOM"],  "plain string list"),
        ([{"symbol": "AMD"}, {"symbol": "INTC"}],       ["AMD", "INTC"],           "list of {symbol:}"),
        ([{"peerSymbol": "AMD"}, {"ticker": "INTC"}],   ["AMD", "INTC"],           "list of {peerSymbol:/ticker:}"),
        ("not a list",                                  [],                        "bad input (str)"),
        ([],                                            [],                        "empty list"),
    ]
    all_pass = True
    for input_val, expected, label in cases:
        result = _extract_tickers(input_val)
        match = result == expected
        icon = "✅" if match else "❌"
        print(f"  {icon}  {label}: {result}  (expected {expected})")
        if not match:
            all_pass = False
    ok("All _extract_tickers cases passed") if all_pass else fail("Some _extract_tickers cases failed")

    # _parse_classifications — valid JSON
    print(f"\n  [_parse_classifications — valid JSON]")
    peers = ["AMD", "TSM", "INTC"]
    valid_json = json.dumps({
        "classifications": [
            {"ticker": "AMD",  "relationship": "COMPETITOR", "reason": "Direct GPU rival."},
            {"ticker": "TSM",  "relationship": "SUPPLIER",   "reason": "Fabricates NVDA chips."},
            {"ticker": "INTC", "relationship": "COMPETITOR", "reason": "Competes in data centre."},
        ]
    })
    result = _parse_classifications(valid_json, peers)
    ok(f"Parsed {len(result)} classifications")
    for c in result:
        ok(f"  {c['ticker']:6s} → {c['relationship']:12s} | {c['reason']}")

    # _parse_classifications — broken JSON falls back gracefully
    print(f"\n  [_parse_classifications — broken JSON fallback]")
    result_bad = _parse_classifications("not valid json {{{", peers)
    ok(f"Fallback returned {len(result_bad)} entries (one per peer)")
    for c in result_bad:
        assert c["relationship"] == "SAME_SECTOR", "Expected SAME_SECTOR fallback"
        ok(f"  {c['ticker']:6s} → {c['relationship']}  ✅")


# ============================================================================
# TEST 4: FULL NODE RUN
# ============================================================================

async def test_full_node(tools_by_name: dict, ticker: str) -> None:
    """Run detect_related_companies_node() end-to-end and validate the output."""
    hdr(f"TEST 4 — Full detect_related_companies_node() Run  ({ticker})")

    if not ANTHROPIC_KEY:
        fail("ANTHROPIC_API_KEY not set — cannot run LLM steps")
        return

    llm = ChatAnthropic(model="claude-sonnet-4-20250514")
    llm_with_tools = llm.bind_tools(list(tools_by_name.values()))

    config = {
        "configurable": {
            "tools_by_name":  tools_by_name,
            "llm":            llm,
            "llm_with_tools": llm_with_tools,
        }
    }

    state = {
        "ticker":               ticker,
        "errors":               [],
        "node_execution_times": {},
    }

    print(f"\n  Running node... (real FMP + LLM calls)")
    try:
        result = await detect_related_companies_node(state, config)
    except Exception as e:
        fail(f"Node raised an unexpected exception: {e}")
        import traceback; traceback.print_exc()
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
    elapsed = result.get("node_execution_times", {}).get("node_3", 0)
    ok(f"Wall-clock time: {elapsed:.2f}s")

    # --- Schema validation ---
    print(f"\n  Schema validation:")
    related = result.get("related_companies", [])

    if not isinstance(related, list):
        fail(f"related_companies is not a list: {type(related)}")
        return

    ok(f"related_companies is a list with {len(related)} entries")

    all_valid = True
    for i, entry in enumerate(related):
        if not isinstance(entry, dict):
            fail(f"Entry {i} is not a dict: {entry!r}")
            all_valid = False
            continue
        for field in ("ticker", "relationship", "reason"):
            if field not in entry:
                fail(f"Entry {i} missing field '{field}': {entry}")
                all_valid = False

        valid_relationships = {"COMPETITOR", "SUPPLIER", "CUSTOMER", "SAME_SECTOR"}
        rel = entry.get("relationship", "")
        if rel not in valid_relationships:
            warn(f"Entry {i} has unexpected relationship value: '{rel}'")

    if all_valid:
        ok("All entries have correct schema (ticker, relationship, reason)")

    # --- Print full LLM output ---
    print(f"\n  Full related_companies output (LLM classifications):")
    print(f"  {'Ticker':<8}  {'Relationship':<14}  Reason")
    print(f"  {'-'*8}  {'-'*14}  {'-'*40}")
    for c in related:
        print(f"  {c.get('ticker','?'):<8}  {c.get('relationship','?'):<14}  {c.get('reason','')}")

    # --- Fallback guard test ---
    print(f"\n  Fallback guard test (no FMP config):")
    bad_config = {"configurable": {"tools_by_name": {}, "llm": None, "llm_with_tools": None}}
    fallback = await detect_related_companies_node(state, bad_config)
    if fallback.get("errors"):
        ok(f"Guard triggered correctly — error: {fallback['errors'][0]}")
    else:
        fail("Guard did NOT trigger — something is wrong")
    if fallback.get("related_companies") == []:
        ok("Fallback returns empty related_companies list ✅")

    print(f"\n  {'Schema OK ✅' if all_valid else 'Schema has issues ❌'}")


# ============================================================================
# MAIN
# ============================================================================

async def main() -> None:
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"

    print(f"\n{'#' * 72}")
    print(f"#  NODE 3 ISOLATED TEST  —  {ticker}  —  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'#' * 72}")

    if not FMP_API_KEY:
        fail("FMP_API_KEY not set — tests 1, 2, 4 will be skipped")
        test_pure_helpers()
        return

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

    # Test 2 — Individual tool data checks
    await test_data_checks(tools_by_name, ticker)

    # Test 3 — Pure helpers (no network)
    test_pure_helpers()

    # Test 4 — Full node run
    await test_full_node(tools_by_name, ticker)

    print(f"\n{'#' * 72}")
    print(f"#  TEST COMPLETE  —  {datetime.now():%H:%M:%S}")
    print(f"{'#' * 72}\n")


if __name__ == "__main__":
    asyncio.run(main())
