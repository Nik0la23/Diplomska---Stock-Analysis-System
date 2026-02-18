"""
Integration test for Node 9B: Behavioral Anomaly Detection

Runs the full upstream pipeline with a real stock ticker:
  Node 1 (Price) -> Node 3 (Related) -> Node 2 (News) -> Node 9A (Content) ->
  Node 4/5/6/7 (Analysis, sequential) -> Node 8 (Learning) -> Node 9B (Behavioral)

Run with:
    ./venv/bin/python scripts/test_node_09b_integration.py [TICKER]
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging

logging.basicConfig(
    level=logging.WARNING,          # suppress per-node DEBUG noise
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _sep(title: str = "", width: int = 70) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'='*2} {title} {'='*pad}")
    else:
        print("=" * width)


def _run_pipeline(ticker: str) -> dict:
    """
    Run nodes 1-9B sequentially and return final state.
    Each node is wrapped in a resilient call: if a node crashes or returns a
    partial dict (missing required keys), we merge its output onto the existing
    state so downstream nodes always see a complete state.
    """
    from src.graph.state import create_initial_state
    from src.langgraph_nodes.node_01_data_fetching import fetch_price_data_node
    from src.langgraph_nodes.node_03_related_companies import detect_related_companies_node
    from src.langgraph_nodes.node_02_news_fetching import fetch_all_news_node
    from src.langgraph_nodes.node_09a_content_analysis import content_analysis_node
    from src.langgraph_nodes.node_04_technical_analysis import technical_analysis_node
    from src.langgraph_nodes.node_05_sentiment_analysis import sentiment_analysis_node
    from src.langgraph_nodes.node_06_market_context import market_context_node
    from src.langgraph_nodes.node_07_monte_carlo import monte_carlo_forecasting_node
    from src.langgraph_nodes.node_08_news_verification import news_verification_node
    from src.langgraph_nodes.node_09b_behavioral_anomaly import behavioral_anomaly_detection_node

    state = create_initial_state(ticker)

    steps = [
        ("Node 1  — Price Data",           fetch_price_data_node),
        ("Node 3  — Related Companies",    detect_related_companies_node),
        ("Node 2  — News Fetching",        fetch_all_news_node),
        ("Node 9A — Content Analysis",     content_analysis_node),
        ("Node 4  — Technical Analysis",   technical_analysis_node),
        ("Node 5  — Sentiment Analysis",   sentiment_analysis_node),
        ("Node 6  — Market Context",       market_context_node),
        ("Node 7  — Monte Carlo",          monte_carlo_forecasting_node),
        ("Node 8  — News Verification",    news_verification_node),
        ("Node 9B — Behavioral Anomaly",   behavioral_anomaly_detection_node),
    ]

    for label, node_fn in steps:
        print(f"  Running {label}...", end=" ", flush=True)
        try:
            result = node_fn(state)
            # If the node returned a full state (has 'ticker') use it directly.
            # If it returned a partial dict, merge it onto the existing state
            # so no previously-set keys are lost.
            if isinstance(result, dict):
                if "ticker" not in result:
                    state.update(result)   # merge partial return
                else:
                    state = result         # full state replacement
            print("done")
        except Exception as exc:
            print(f"FAILED ({exc})")
            state.setdefault("errors", []).append(f"{label} failed: {exc}")

    return state


def _print_results(state: dict) -> None:
    ticker = state["ticker"]
    bad    = state.get("behavioral_anomaly_detection") or {}

    # ── Overall verdict ──────────────────────────────────────────────────────
    _sep(f"NODE 9B RESULTS  —  {ticker}")

    risk  = bad.get("risk_level", "N/A")
    score = bad.get("pump_and_dump_score", 0)
    rec   = bad.get("trading_recommendation", "N/A")
    summary = bad.get("behavioral_summary", "")

    risk_color = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨", "CRITICAL": "🛑"}.get(risk, "❓")
    print(f"\n{risk_color}  Risk Level          : {risk}")
    print(f"📊  Pump-and-Dump Score : {score}/100")
    print(f"💡  Recommendation      : {rec}")
    print(f"📝  Summary             : {summary}")
    exec_time = bad.get("execution_time", 0)
    print(f"⏱   Execution Time      : {exec_time:.3f}s")

    # ── Detection breakdown ──────────────────────────────────────────────────
    _sep("DETECTION BREAKDOWN")
    breakdown = bad.get("detection_breakdown", {})
    labels = {
        "volume_anomaly":               "Volume Anomaly          (max 20)",
        "source_reliability_divergence":"Source Reliability Div. (max 20)",
        "news_velocity_anomaly":        "News Velocity Anomaly   (max 18)",
        "news_price_divergence":        "News-Price Divergence   (max 25)",
        "cross_stream_incoherence":     "Cross-Stream Coherence  (max 10)",
        "historical_pattern_match":     "Historical Pattern Match(max 10)",
    }
    for key, label in labels.items():
        pts = breakdown.get(key, 0)
        bar = "█" * pts + "░" * (25 - pts)
        print(f"  {label}: {pts:3d}  |{bar}|")

    primary = bad.get("primary_risk_factors", [])
    if primary:
        print(f"\n  Top risk factors: {', '.join(primary)}")

    # ── Per-detector detail ───────────────────────────────────────────────────
    _sep("DETECTOR DETAILS")

    def _det(name: str, key: str) -> None:
        d = bad.get(key, {})
        detected = "🔴 DETECTED" if d.get("detected") else "🟢 clean"
        sev  = d.get("severity", "LOW")
        pts  = d.get("contribution_score", 0)
        print(f"\n  [{name}]  {detected}  severity={sev}  points={pts}")
        # Print key sub-values depending on detector
        extras = {k: v for k, v in d.items()
                  if k not in ("detected", "severity", "contribution_score", "explanation",
                               "insufficient_data", "outcomes")}
        for k, v in extras.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
        if d.get("explanation"):
            print(f"    explanation: {d['explanation']}")
        if d.get("insufficient_data"):
            print(f"    (insufficient historical data)")
        if d.get("outcomes"):
            o = d["outcomes"]
            print(f"    outcomes → pct_crash={o.get('pct_ended_in_crash',0):.1%}  "
                  f"avg_7d_chg={o.get('avg_price_change_7d',0):.2f}%  "
                  f"worst={o.get('worst_outcome',0):.2f}%")

    _det("1. Volume Anomaly",            "volume_anomaly")
    _det("2. Source Reliability Div.",   "source_reliability_divergence")
    _det("3. News Velocity",             "news_velocity_anomaly")
    _det("4. News-Price Divergence",     "news_price_divergence")
    _det("5. Cross-Stream Coherence",    "cross_stream_coherence")
    _det("6. Historical Pattern Match",  "historical_pattern_match")

    # ── Alerts ───────────────────────────────────────────────────────────────
    alerts = bad.get("alerts", [])
    if alerts:
        _sep("ALERTS")
        for i, a in enumerate(alerts, 1):
            print(f"  {i}. {a}")
    else:
        _sep("ALERTS")
        print("  No alerts generated (all signals clean)")

    # ── Upstream pipeline context ─────────────────────────────────────────────
    _sep("UPSTREAM CONTEXT")
    if state.get("technical_signal"):
        print(f"  Technical Signal  : {state['technical_signal']} "
              f"(confidence {(state.get('technical_confidence') or 0)*100:.0f}%)")
    if state.get("sentiment_signal"):
        print(f"  Sentiment Signal  : {state['sentiment_signal']} "
              f"(aggregated={state.get('aggregated_sentiment', 0):.3f})")
    mc = state.get("market_context") or {}
    if mc.get("context_signal"):
        print(f"  Market Signal     : {mc['context_signal']} "
              f"(trend={mc.get('market_trend','?')})")

    niv = state.get("news_impact_verification") or {}
    if niv:
        print(f"  Node 8 Accuracy   : {niv.get('news_accuracy_score', 0):.1f}%  "
              f"(sample={niv.get('sample_size', 0)}, "
              f"insufficient={niv.get('insufficient_data', True)})")

    cas = state.get("content_analysis_summary") or {}
    if cas:
        avg = (cas.get("average_scores") or {})
        print(f"  Node 9A avg anomaly: {avg.get('composite_anomaly', 0):.3f}  "
              f"(high-risk articles: {cas.get('high_risk_articles', 0)}/{cas.get('total_articles_processed', 0)})")

    # ── Timing ───────────────────────────────────────────────────────────────
    _sep("EXECUTION TIMES")
    times = state.get("node_execution_times", {})
    total = sum(times.values())
    for node, t in sorted(times.items()):
        bar = "█" * min(40, int(t * 10))
        print(f"  {node:<12}: {t:5.2f}s  {bar}")
    print(f"  {'TOTAL':<12}: {total:5.2f}s")

    # ── Errors ───────────────────────────────────────────────────────────────
    errors = state.get("errors", [])
    if errors:
        _sep("ERRORS / WARNINGS")
        for e in errors:
            print(f"  ⚠  {e}")

    _sep()
    print()


if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"

    print()
    _sep(f"Node 9B Integration Test  —  {ticker}")
    print(f"  Running full pipeline: 1 → 3 → 2 → 9A → 4/5/6/7 → 8 → 9B")
    print()

    state = _run_pipeline(ticker)
    _print_results(state)
