"""
loading_screen.py — Real-time LangGraph execution visualizer

Uses workflow.astream_events() to intercept on_chain_start / on_chain_end
events and updates an st.empty() placeholder on each transition so the
pipeline diagram stays in sync with what the graph is actually doing.

No fake timers. No polling. Every frame is driven by a real LangGraph event.
"""

import asyncio
import logging
import os
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ─── Pipeline layout ──────────────────────────────────────────────────────────
# Each phase: {"label", "layout", "nodes": [(node_name, display_label, badge)]}
# Nodes not shown (internal): parallel_join, run_background_outcomes

_PHASES = [
    {
        "label":  "DATA ACQUISITION",
        "layout": "sequential",
        "nodes": [
            ("fetch_price",       "Price Data",       "01"),
            ("related_companies", "Related Co.",       "03"),
            ("fetch_news",        "News Fetch",        "02"),
            ("content_analysis",  "Content Analysis",  "09A"),
        ],
    },
    {
        "label":  "PARALLEL ANALYSIS",
        "layout": "parallel",
        "nodes": [
            ("technical_analysis", "Technical", "04"),
            ("sentiment_analysis", "Sentiment", "05"),
            ("market_context",     "Market",    "06"),
        ],
    },
    {
        "label":  "SIGNAL PROCESSING",
        "layout": "sequential",
        "nodes": [
            ("monte_carlo",                  "Monte Carlo", "07"),
            ("news_verification",            "News Verify", "08"),
            ("behavioral_anomaly_detection", "Behavioral",  "09B"),
            ("backtesting",                  "Backtesting", "10"),
            ("adaptive_weights",             "Weights",     "11"),
            ("signal_generation",            "Signal Gen",  "12"),
        ],
    },
    {
        "label":  "INTELLIGENCE",
        "layout": "sequential",
        "nodes": [
            ("sec_fundamentals",      "SEC Filings",    "16"),
            ("beginner_explanation",  "Explain (Beg)",  "13"),
            ("technical_explanation", "Explain (Tech)", "14"),
            ("dashboard",             "Dashboard",      "15"),
        ],
    },
]

_ALL_DISPLAY_NODES: list = [
    n for phase in _PHASES for n, _, _ in phase["nodes"]
]

# Nodes we track events for (includes internal ones not shown in UI)
_KNOWN_NODES: Set[str] = set(_ALL_DISPLAY_NODES) | {
    "parallel_join", "run_background_outcomes"
}

_TOTAL = len(_ALL_DISPLAY_NODES)

# node_name → display label (for "currently running" status line)
_LABEL: Dict[str, str] = {
    n: lbl for phase in _PHASES for n, lbl, _ in phase["nodes"]
}

# ─── CSS (injected once per render; scoped with lg- prefix) ──────────────────

_CSS = """
<style>
.lg-wrap {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0a0a0a;
    border: 1px solid #1c1c1c;
    border-radius: 12px;
    padding: 22px 26px 18px;
    color: #e0e0e0;
}
.lg-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid #181818;
}
.lg-brand {
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: 10px;
    color: #444;
    letter-spacing: .10em;
    margin-bottom: 4px;
    text-transform: uppercase;
}
.lg-ticker {
    font-family: 'DM Mono', 'Courier New', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    letter-spacing: .06em;
    line-height: 1;
}
.lg-live {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 10px;
    color: #444;
    text-transform: uppercase;
    letter-spacing: .09em;
    padding-top: 6px;
}
.lg-pulse {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22c55e;
    flex-shrink: 0;
    animation: lg-blink 1.5s ease-in-out infinite;
}
@keyframes lg-blink {
    0%, 100% { opacity: 1;  transform: scale(1.0); }
    50%       { opacity: .3; transform: scale(.75); }
}
/* ── phase block ── */
.lg-phase { margin-bottom: 13px; }
.lg-plabel {
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .11em;
    color: #2e2e2e;
    margin-bottom: 7px;
}
.lg-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
}
.lg-arr { color: #232323; font-size: 11px; flex-shrink: 0; user-select: none; }
/* ── node chip ── */
.lg-node {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 11px 5px 9px;
    border-radius: 5px;
    font-size: 11px;
    font-family: 'DM Mono', 'Courier New', monospace;
    border: 1px solid;
    white-space: nowrap;
    line-height: 1.2;
}
.lg-badge {
    font-size: 9px;
    opacity: .35;
    font-weight: 300;
    letter-spacing: .02em;
}
/* pending */
.n-p { background:#0d0d0d; border-color:#1c1c1c; color:#2e2e2e; }
/* running */
.n-r {
    background:#061309; border-color:#15803d; color:#4ade80;
    animation: lg-glow 1.8s ease-in-out infinite;
}
@keyframes lg-glow {
    0%, 100% { box-shadow: 0 0 0 0   rgba(34,197,94,.00); border-color:#15803d; }
    50%       { box-shadow: 0 0 0 4px rgba(34,197,94,.14); border-color:#22c55e; }
}
/* done */
.n-d { background:#060e06; border-color:#14532d; color:#22c55e; }
/* error */
.n-e { background:#0e0606; border-color:#7f1d1d; color:#f87171; }
/* ── progress footer ── */
.lg-foot {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #181818;
}
.lg-foot-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
}
.lg-foot-txt { font-size: 10px; color: #383838; }
.lg-foot-pct {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #22c55e;
}
.lg-bar-bg { height: 2px; background: #181818; border-radius: 2px; overflow: hidden; }
.lg-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #15803d 0%, #22c55e 100%);
    transition: width .4s ease;
}
.lg-running-txt {
    margin-top: 8px;
    min-height: 15px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: #3a7c50;
    letter-spacing: .04em;
}
</style>
"""


# ─── HTML renderer ────────────────────────────────────────────────────────────

def _chip(name: str, label: str, badge: str,
          active: Set[str], completed: Set[str], failed: Set[str]) -> str:
    if name in active:
        cls, icon = "n-r", "●"
    elif name in completed:
        cls, icon = "n-d", "✓"
    elif name in failed:
        cls, icon = "n-e", "✗"
    else:
        cls, icon = "n-p", "○"
    return (
        f'<div class="lg-node {cls}">'
        f'<span class="lg-badge">{badge}</span>'
        f'{icon}&nbsp;{label}'
        f'</div>'
    )


def render_loading_html(
    ticker: str,
    active:    Set[str],
    completed: Set[str],
    failed:    Set[str],
) -> str:
    done = sum(1 for n in _ALL_DISPLAY_NODES if n in completed)
    pct  = int(done / _TOTAL * 100)

    running_labels = [_LABEL[n] for n in _ALL_DISPLAY_NODES if n in active]
    status_txt = (
        "↳ " + " · ".join(running_labels)
        if running_labels else ""
    )

    phases_html = ""
    for phase in _PHASES:
        row = ""
        for i, (name, label, badge) in enumerate(phase["nodes"]):
            if i > 0 and phase["layout"] == "sequential":
                row += '<span class="lg-arr">→</span>'
            row += _chip(name, label, badge, active, completed, failed)
        phases_html += (
            f'<div class="lg-phase">'
            f'<div class="lg-plabel">{phase["label"]}</div>'
            f'<div class="lg-row">{row}</div>'
            f'</div>'
        )

    return f"""{_CSS}
<div class="lg-wrap">
  <div class="lg-head">
    <div>
      <div class="lg-brand">◈ StockSentinel</div>
      <div class="lg-ticker">{ticker}</div>
    </div>
    <div class="lg-live"><div class="lg-pulse"></div>Analyzing</div>
  </div>
  {phases_html}
  <div class="lg-foot">
    <div class="lg-foot-row">
      <span class="lg-foot-txt">{done} / {_TOTAL} nodes complete</span>
      <span class="lg-foot-pct">{pct}%</span>
    </div>
    <div class="lg-bar-bg">
      <div class="lg-bar-fill" style="width:{pct}%"></div>
    </div>
    <div class="lg-running-txt">{status_txt}</div>
  </div>
</div>"""


# ─── Streaming runner ─────────────────────────────────────────────────────────

async def run_with_loading_screen(
    ticker:      str,
    placeholder,  # st.empty() instance
) -> Tuple[Optional[dict], Optional[str]]:
    """
    Execute the full LangGraph pipeline while keeping the loading screen in sync.

    Intercepts on_chain_start / on_chain_end / on_chain_error events from
    astream_events() and re-renders the placeholder on every node transition.

    Args:
        ticker:      Stock ticker symbol, e.g. "AAPL".
        placeholder: A Streamlit st.empty() container to render into.

    Returns:
        (final_state, error_message)
        error_message is None on success; final_state is None on hard failure.
    """
    from src.graph.workflow import create_stock_analysis_workflow
    from src.graph.state import create_initial_state
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_anthropic import ChatAnthropic
    from dotenv import load_dotenv

    load_dotenv()

    active:    Set[str] = set()
    completed: Set[str] = set()
    failed:    Set[str] = set()
    final_state: Optional[dict] = None
    error_msg:   Optional[str]  = None

    def _push() -> None:
        placeholder.markdown(
            render_loading_html(ticker, active, completed, failed),
            unsafe_allow_html=True,
        )

    _push()  # initial frame — all nodes pending

    try:
        fmp_api_key = os.environ.get("FMP_API_KEY", "")
        fmp_url = f"https://financialmodelingprep.com/mcp?apikey={fmp_api_key}"

        client = MultiServerMCPClient({
            "fmp": {"url": fmp_url, "transport": "streamable_http"}
        })
        tools         = await client.get_tools()
        tools_by_name = {t.name: t for t in tools}
        llm           = ChatAnthropic(model="claude-sonnet-4-20250514")
        llm_with_tools = llm.bind_tools(tools)

        config = {
            "configurable": {
                "tools_by_name":  tools_by_name,
                "llm":            llm,
                "llm_with_tools": llm_with_tools,
            }
        }

        workflow      = create_stock_analysis_workflow()
        initial_state = create_initial_state(ticker)

        async for event in workflow.astream_events(
            initial_state, config=config, version="v2"
        ):
            etype = event["event"]
            name  = event.get("name", "")

            # ── node started ─────────────────────────────────────────────────
            if etype == "on_chain_start" and name in _KNOWN_NODES:
                if name in _ALL_DISPLAY_NODES:
                    active.add(name)
                    _push()

            # ── node finished ─────────────────────────────────────────────────
            elif etype == "on_chain_end" and name in _KNOWN_NODES:
                if name in _ALL_DISPLAY_NODES:
                    active.discard(name)
                    completed.add(name)
                    _push()

            # ── node errored ──────────────────────────────────────────────────
            elif etype == "on_chain_error" and name in _KNOWN_NODES:
                if name in _ALL_DISPLAY_NODES:
                    active.discard(name)
                    failed.add(name)
                    _push()

            # ── capture final state from root graph completion ─────────────────
            # LangGraph emits on_chain_end for the root graph with the full state.
            # The root graph name is typically "LangGraph" or empty string.
            if etype == "on_chain_end":
                output = event.get("data", {}).get("output")
                if (
                    isinstance(output, dict)
                    and output.get("ticker")
                    and output.get("final_signal")
                ):
                    final_state = output

    except Exception as exc:
        if hasattr(exc, "exceptions") and exc.exceptions:
            causes = "\n".join(
                f"[{i+1}] {type(sub).__name__}: {sub}"
                for i, sub in enumerate(exc.exceptions)
            )
            error_msg = f"{exc}\n\nSub-exceptions:\n{causes}"
        else:
            error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Pipeline streaming error: %s", error_msg)

    return final_state, error_msg
