"""
Node 3: Related Companies Detection

Identifies related peer companies using the FMP MCP peers tool,
then classifies each peer's relationship to the target via a bare LLM call.

Data Sources:
- FMP MCP: peers  (GET /stable/peers?symbol=AAPL)
- FMP MCP: profile-symbol (sector/industry for LLM prompt)
- Claude LLM: relationship classification (Pattern C — bare call, no tools)

Runs AFTER:  Node 1 (price data available in state)
Runs BEFORE: Node 2 (news fetching needs related company list)
Can run in PARALLEL with: Nothing
"""

import ast
import asyncio
import json
import logging
import time
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langgraph.types import RunnableConfig

from src.graph.state import StockAnalysisState

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_PEERS = 5

# Fallback tickers used when FMP returns nothing for a well-known ticker
PEERS_FALLBACK: Dict[str, List[str]] = {
    "TSLA":  ["RIVN", "NIO", "GM", "F", "LCID"],
    "NVDA":  ["AMD", "INTC", "QCOM", "TSM", "AVGO"],
    "AAPL":  ["MSFT", "GOOGL", "META", "AMZN", "DELL"],
    "MSFT":  ["AAPL", "GOOGL", "AMZN", "ORCL", "SAP"],
    "GOOGL": ["META", "MSFT", "SNAP", "AMZN", "NFLX"],
    "META":  ["SNAP", "GOOGL", "PINS", "NFLX", "AMZN"],
    "AMZN":  ["WMT", "EBAY", "SHOP", "TGT", "COST"],
    "AMD":   ["NVDA", "INTC", "QCOM", "TSM", "AVGO"],
}

PEER_CLASSIFICATION_PROMPT = """
You are a sell-side equity analyst at Goldman Sachs.

Target stock: {ticker}
  Sector  : {sector}
  Industry: {industry}
  Business: {target_description}

Classify each peer below in relation to {ticker}.
Use ONLY these four relationship types:
- COMPETITOR  : directly competes for the same customers or market share
- SUPPLIER    : sells components, services, or inputs TO {ticker}
- CUSTOMER    : buys products or services FROM {ticker}
- SAME_SECTOR : same sector/industry but no direct competitive or supply relationship

IMPORTANT — base your classifications SOLELY on the business descriptions provided
below. Do NOT rely on your training knowledge; company strategies and product lines
change frequently and your knowledge may be outdated.

Peers to classify:
{peer_profiles}

Return ONLY valid JSON. No preamble. No explanation outside the JSON.

{{
  "classifications": [
    {{
      "ticker": "string",
      "relationship": "COMPETITOR | SUPPLIER | CUSTOMER | SAME_SECTOR",
      "reason": "one sentence max, grounded in the descriptions above"
    }}
  ]
}}
"""


# ============================================================================
# HELPERS
# ============================================================================

def _parse_tool_response(raw: str) -> Any:
    """
    Parse string returned by tool.ainvoke() into a Python object.

    FMP MCP tools sometimes return plain JSON strings and sometimes return
    a Python repr of the MCP content envelope:
        "[{'type': 'text', 'text': '[{...real json...}]'}]"
    Both forms are handled here.
    """
    # Fast path: raw is already valid JSON
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Slow path: MCP content envelope — use ast.literal_eval then extract .text
    try:
        content_list = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if isinstance(content_list, list):
            for item in content_list:
                if isinstance(item, dict) and item.get("type") == "text":
                    return json.loads(item["text"])
    except Exception:
        pass

    logger.warning(f"[Node 3] Could not parse tool result: {raw!r:.200}")
    return []


def _extract_tickers(parsed: Any) -> List[str]:
    """
    Extract ticker strings from FMP peers response.

    FMP may return:
      - ["AMD", "INTC", ...]                          (plain list of strings)
      - [{"symbol": "AMD"}, ...]                      (list of dicts)
      - [{"peerSymbol": "AMD"}, ...]                  (alternate key)
    """
    if not isinstance(parsed, list):
        return []
    tickers: List[str] = []
    for item in parsed:
        if isinstance(item, str) and item.strip():
            tickers.append(item.strip().upper())
        elif isinstance(item, dict):
            sym = (
                item.get("symbol")
                or item.get("ticker")
                or item.get("peerSymbol")
            )
            if sym:
                tickers.append(str(sym).strip().upper())
    return tickers


def _parse_classifications(content: str, peers: List[str]) -> List[Dict[str, Any]]:
    """
    Parse the LLM JSON response into a list of classification dicts.
    Falls back to SAME_SECTOR for any peer the LLM omits or on parse error.
    """
    try:
        text = content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)
        raw_list = data.get("classifications", [])
        if not isinstance(raw_list, list):
            raise ValueError("'classifications' is not a list")

        return [
            {
                "ticker": str(c.get("ticker", "")).strip().upper(),
                "relationship": str(c.get("relationship", "SAME_SECTOR")).strip(),
                "reason": str(c.get("reason", "")).strip(),
            }
            for c in raw_list
            if isinstance(c, dict) and c.get("ticker")
        ]
    except Exception as exc:
        logger.warning(f"[Node 3] LLM classification parse failed: {exc}")
        return [
            {"ticker": p, "relationship": "SAME_SECTOR", "reason": "Classification unavailable."}
            for p in peers
        ]


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

async def detect_related_companies_node(
    state: StockAnalysisState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Node 3: Related Companies Detection.

    Execution flow:
    1. Validate FMP config from RunnableConfig
    2. Call FMP peers tool → raw ticker list
    3. If empty → use PEERS_FALLBACK
    4. Call FMP profile-symbol tool → sector/industry for LLM prompt
    5. Call bare LLM with PEER_CLASSIFICATION_PROMPT → JSON classifications
    6. Parse and merge results → List[Dict] with ticker, relationship, reason
    7. Return updated state keys

    Args:
        state:  Current LangGraph state.
        config: RunnableConfig injected by the workflow. Must contain
                config["configurable"]["tools_by_name"],
                config["configurable"]["llm"], and
                config["configurable"]["llm_with_tools"].

    Returns:
        Partial state dict with keys:
          - related_companies: List[Dict] — each entry has ticker, relationship, reason
          - node_execution_times: {"node_3": elapsed}
    """
    start = time.time()
    ticker = state["ticker"]
    logger.info(f"[Node 3] Starting related companies detection for {ticker}")

    # ------------------------------------------------------------------
    # 1. Extract shared resources from config
    # ------------------------------------------------------------------
    configurable   = config["configurable"]
    tools_by_name  = configurable["tools_by_name"]
    llm            = configurable["llm"]
    llm_with_tools = configurable["llm_with_tools"]

    # ------------------------------------------------------------------
    # 2. Guard: graceful degradation when FMP session is unavailable
    # ------------------------------------------------------------------
    if not tools_by_name or llm is None or llm_with_tools is None:
        logger.warning(f"[Node 3] FMP config incomplete — returning empty peers")
        return {
            "related_companies": [],
            "errors": [f"Node 3: FMP config unavailable for {ticker}"],
            "node_execution_times": {"node_3": time.time() - start},
        }

    try:
        # ------------------------------------------------------------------
        # Step 1: Fetch peers via FMP peers (Pattern A)
        # ------------------------------------------------------------------
        peers: List[str] = []
        peers_tool = tools_by_name.get("peers")
        company_tool = tools_by_name.get("company")
        if peers_tool is not None:
            try:
                raw = await peers_tool.ainvoke({"symbol": ticker})
                parsed = _parse_tool_response(raw)
                peers = _extract_tickers(parsed)
                # Never include the target ticker in its own peer list
                peers = [p for p in peers if p != ticker.upper()]
                logger.info(f"[Node 3] FMP peers → {len(peers)} peers: {peers}")
            except Exception as exc:
                logger.warning(f"[Node 3] peers tool failed for {ticker}: {exc}")
        elif company_tool is not None:
            # New FMP MCP schema exposes peers as company(endpoint="peers", ...).
            try:
                raw = await company_tool.ainvoke({"endpoint": "peers", "symbol": ticker})
                parsed = _parse_tool_response(raw)
                peers = _extract_tickers(parsed)
                peers = [p for p in peers if p != ticker.upper()]
                logger.info(f"[Node 3] FMP company/peers → {len(peers)} peers: {peers}")
            except Exception as exc:
                logger.warning(f"[Node 3] company(peers) failed for {ticker}: {exc}")
        else:
            logger.warning(
                "[Node 3] neither 'peers' nor 'company' tool found in tools_by_name — check MCP tool names"
            )

        # Fallback when FMP returned nothing
        if not peers:
            peers = PEERS_FALLBACK.get(ticker.upper(), [])
            if peers:
                logger.info(f"[Node 3] Using PEERS_FALLBACK for {ticker}: {peers}")
            else:
                logger.warning(f"[Node 3] No peers found and no fallback available for {ticker}")
                elapsed = time.time() - start
                return {
                    "related_companies": [],
                    "node_execution_times": {"node_3": elapsed},
                }

        # Cap at MAX_PEERS
        peers = peers[:MAX_PEERS]

        # ------------------------------------------------------------------
        # Step 2: Fetch target + all peer profiles concurrently (Pattern A)
        # Each profile gives us sector, industry, and business description
        # so the LLM can reason from data rather than training memory.
        # ------------------------------------------------------------------
        sector             = "Unknown"
        industry           = "Unknown"
        target_description = "No description available."
        peer_descriptions: Dict[str, str] = {}

        profile_tool = tools_by_name.get("profile-symbol")
        if profile_tool is not None or company_tool is not None:
            symbols_to_fetch = [ticker] + peers

            async def _fetch_profile(sym: str) -> tuple[str, dict]:
                try:
                    if profile_tool is not None:
                        raw = await profile_tool.ainvoke({"symbol": sym})
                    else:
                        # New FMP MCP schema: company(endpoint="profile-symbol", ...).
                        raw = await company_tool.ainvoke({"endpoint": "profile-symbol", "symbol": sym})
                    parsed = _parse_tool_response(raw)
                    if isinstance(parsed, list) and parsed:
                        return sym, parsed[0]
                    if isinstance(parsed, dict):
                        return sym, parsed
                except Exception as exc:
                    logger.warning(f"[Node 3] profile lookup failed for {sym}: {exc}")
                return sym, {}

            results = await asyncio.gather(*[_fetch_profile(s) for s in symbols_to_fetch])
            profiles: Dict[str, dict] = dict(results)

            target_prof = profiles.get(ticker, {})
            sector             = str(target_prof.get("sector")      or "Unknown")
            industry           = str(target_prof.get("industry")    or "Unknown")
            target_description = str(target_prof.get("description") or "No description available.")
            if len(target_description) > 300:
                target_description = target_description[:297] + "..."

            for p in peers:
                prof = profiles.get(p, {})
                desc = str(prof.get("description") or "No description available.")
                if len(desc) > 300:
                    desc = desc[:297] + "..."
                sec  = str(prof.get("sector")   or "Unknown")
                ind  = str(prof.get("industry") or "Unknown")
                peer_descriptions[p] = f"Sector: {sec} | Industry: {ind} | {desc}"

            logger.info(f"[Node 3] {ticker} — sector={sector}, industry={industry}")
        else:
            logger.warning(
                "[Node 3] neither 'profile-symbol' nor 'company' tool found — LLM will use ticker symbols only"
            )
            for p in peers:
                peer_descriptions[p] = "No description available."

        # Format peer block for prompt
        peer_profiles_block = "\n".join(
            f"- {p}: {peer_descriptions.get(p, 'No description available.')}"
            for p in peers
        )

        # ------------------------------------------------------------------
        # Step 3: Classify peers via bare LLM call (Pattern C)
        # ------------------------------------------------------------------
        prompt = PEER_CLASSIFICATION_PROMPT.format(
            ticker=ticker,
            sector=sector,
            industry=industry,
            target_description=target_description,
            peer_profiles=peer_profiles_block,
        )
        classifications: List[Dict[str, Any]] = []
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content
            if isinstance(content, str):
                classifications = _parse_classifications(content, peers)
            else:
                raise ValueError(f"Unexpected LLM response content type: {type(content)}")
            logger.info(f"[Node 3] LLM classified {len(classifications)} peers")
        except Exception as exc:
            logger.warning(f"[Node 3] LLM classification failed for {ticker}: {exc}")
            classifications = [
                {"ticker": p, "relationship": "SAME_SECTOR", "reason": "Classification unavailable."}
                for p in peers
            ]

        # ------------------------------------------------------------------
        # Step 4: Ensure every peer from Step 1 has a classification entry
        # ------------------------------------------------------------------
        classified_tickers = {c["ticker"] for c in classifications}
        for p in peers:
            if p not in classified_tickers:
                classifications.append(
                    {"ticker": p, "relationship": "SAME_SECTOR", "reason": "Not returned by LLM."}
                )

        elapsed = time.time() - start
        logger.info(
            f"[Node 3] Completed for {ticker} in {elapsed:.2f}s — "
            f"{len(classifications)} related companies: "
            f"{[c['ticker'] for c in classifications]}"
        )

        return {
            "related_companies": classifications,
            "node_execution_times": {"node_3": elapsed},
        }

    except Exception as exc:
        elapsed = time.time() - start
        logger.error(f"[Node 3] Unexpected error for {ticker}: {exc}", exc_info=True)
        return {
            "related_companies": [],
            "errors": [f"Node 3: Unexpected error — {str(exc)}"],
            "node_execution_times": {"node_3": elapsed},
        }
