"""
Detailed AAPL test runner — runs the full 14-node pipeline and prints
the output of every node, data volumes, freshness info, and the full
Node 13 / Node 14 LLM responses.

All values read directly from state. No derivation or computation here.
If a value prints N/A it means the node that writes it either failed or
hasn't been built yet — useful diagnostic signal.

Usage:
    python scripts/test_aapl_detailed.py
"""

import os
import sys
import logging
from pathlib import Path
import json

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"]    = "false"

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Enable INFO for src modules; silence noisy libs
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
)
for noisy in ("httpx", "httpcore", "urllib3", "yfinance", "peewee",
              "transformers", "langsmith", "anthropic", "openai",
              "sentence_transformers", "torch", "filelock", "PIL"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from src.graph.workflow import run_stock_analysis

TICKER = "AAPL"
SEP  = "=" * 72
DASH = "─" * 72


def _safe(v, default="N/A"):
    return v if v is not None else default


def _fmt_float(v, decimals=4, default="N/A"):
    if v is None:
        return default
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _json_preview(d, max_keys=8):
    """Return a compact preview of a dict (first N keys)."""
    if not isinstance(d, dict):
        return str(d)
    keys = list(d.keys())[:max_keys]
    snippet = {k: d[k] for k in keys}
    try:
        return json.dumps(snippet, indent=2, default=str)
    except Exception:
        return str(snippet)


def print_section(title):
    print(f"\n{DASH}")
    print(f"  {title}")
    print(DASH)


# ---------------------------------------------------------------------------
# Pre-run: show what is currently in the DB for AAPL
# ---------------------------------------------------------------------------
print(SEP)
print(f"  AAPL FULL PIPELINE TEST  —  Pre-run DB snapshot")
print(SEP)

try:
    import sqlite3
    conn = sqlite3.connect(str(_root / "data" / "stock_prices.db"))
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*), MIN(date), MAX(date), MAX(created_at) FROM price_data WHERE ticker='AAPL'")
    pr = cur.fetchone()
    print(f"  [DB] Price rows   : {pr[0]}  |  range {pr[1]} → {pr[2]}  |  last updated {pr[3]}")

    cur.execute("SELECT COUNT(*), MIN(published_at), MAX(published_at) FROM news_articles WHERE ticker='AAPL'")
    nr = cur.fetchone()
    print(f"  [DB] News rows    : {nr[0]}  |  range {nr[1]} → {nr[2]}")

    cur.execute("SELECT COUNT(*) FROM news_outcomes WHERE ticker='AAPL'")
    no = cur.fetchone()
    print(f"  [DB] News outcomes: {no[0]}")
    conn.close()
except Exception as e:
    print(f"  [DB] Could not query DB: {e}")

print(f"\n  Running pipeline for {TICKER} …\n")

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
result = run_stock_analysis(TICKER)

# ---------------------------------------------------------------------------
# Post-run: show updated DB snapshot
# ---------------------------------------------------------------------------
print_section("POST-RUN DB SNAPSHOT")
try:
    conn = sqlite3.connect(str(_root / "data" / "stock_prices.db"))
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*), MIN(date), MAX(date), MAX(created_at) FROM price_data WHERE ticker='AAPL'")
    pr2 = cur.fetchone()
    print(f"  [DB] Price rows   : {pr2[0]}  |  range {pr2[1]} → {pr2[2]}  |  last updated {pr2[3]}")

    cur.execute("SELECT COUNT(*), MIN(published_at), MAX(published_at) FROM news_articles WHERE ticker='AAPL'")
    nr2 = cur.fetchone()
    print(f"  [DB] News rows    : {nr2[0]}  |  range {nr2[1]} → {nr2[2]}")

    new_price = (pr2[0] or 0) - (pr[0] or 0)
    new_news  = (nr2[0] or 0) - (nr[0] or 0)
    print(f"\n  → NEW price rows added : {new_price}")
    print(f"  → NEW news  rows added : {new_news}")
    conn.close()
except Exception as e:
    print(f"  Could not query DB: {e}")

# ---------------------------------------------------------------------------
# Data freshness verdict
# ---------------------------------------------------------------------------
print_section("DATA FRESHNESS VERDICT")
nfm = result.get("news_fetch_metadata") or {}
print(f"  Price data strategy : {nfm.get('price_fetch_strategy', 'see Node 1 logs above')}")
print(f"  News  data strategy : {nfm.get('news_fetch_strategy',  'see Node 2 logs above')}")
print()
print("  Interpretation:")
print("  ─ If rows were added above → INCREMENTAL (gap-fill: only missing days fetched)")
print("  ─ If DB was already current → CACHE HIT (zero API calls)")
print("  ─ On first-ever run        → FULL FRESH (all 180 days / 6 months pulled)")

# ---------------------------------------------------------------------------
# NODE 1 — Price Data
# ---------------------------------------------------------------------------
print_section("NODE 1 — Price Data Fetching")
df = result.get("raw_price_data")
if df is not None:
    print(f"  Rows loaded into state : {len(df)}")
    print(f"  Date range             : {df.index[0] if hasattr(df, 'index') else df.iloc[0,0]}"
          f"  →  {df.index[-1] if hasattr(df, 'index') else df.iloc[-1,0]}")
    print(f"  Columns                : {list(df.columns)}")
else:
    print("  raw_price_data = None  (Node 1 failed)")

# ---------------------------------------------------------------------------
# NODE 3 — Related Companies
# ---------------------------------------------------------------------------
print_section("NODE 3 — Related Companies")
rc = result.get("related_companies") or []
print(f"  Peers found: {rc}")

# ---------------------------------------------------------------------------
# NODE 2 — News Fetching
# ---------------------------------------------------------------------------
print_section("NODE 2 — News Fetching")
sn = result.get("stock_news")             or []
mn = result.get("market_news")            or []
rn = result.get("related_company_news")   or []
print(f"  Stock-specific articles  : {len(sn)}")
print(f"  Market / global articles : {len(mn)}")
print(f"  Related-company articles : {len(rn)}")
print(f"  Total articles in state  : {len(sn)+len(mn)+len(rn)}")
if nfm:
    print(f"\n  news_fetch_metadata preview:\n{_json_preview(nfm)}")

# ---------------------------------------------------------------------------
# NODE 9A — Content Analysis
# ---------------------------------------------------------------------------
print_section("NODE 9A — Content Analysis & Feature Extraction")
csn = result.get("cleaned_stock_news")              or []
cmn = result.get("cleaned_market_news")             or []
crn = result.get("cleaned_related_company_news")    or []
cas = result.get("content_analysis_summary")        or {}
print(f"  Cleaned stock   news : {len(csn)}")
print(f"  Cleaned market  news : {len(cmn)}")
print(f"  Cleaned related news : {len(crn)}")
if cas:
    print(f"\n  content_analysis_summary:\n{_json_preview(cas, max_keys=12)}")
else:
    print("  content_analysis_summary = None  (Node 9A failed or not built)")

# ---------------------------------------------------------------------------
# NODE 4 — Technical Analysis
# ---------------------------------------------------------------------------
print_section("NODE 4 — Technical Analysis")

# State fields that Node 4 writes (technical_signal / technical_confidence
# remain in state.py for backward compatibility but Node 4 intentionally
# leaves them None — Node 12 is the only signal producer)
print(f"  technical_signal     : {_safe(result.get('technical_signal'))}   "
      f"(always None — Node 4 no longer writes this)")
print(f"  technical_confidence : {_safe(result.get('technical_confidence'))}   "
      f"(always None — Node 4 no longer writes this)")

ti = result.get("technical_indicators") or {}
if ti:
    print(f"\n  ── Core Indicators ──")
    for key in ("rsi", "adx"):
        print(f"    {key:<30} {_safe(ti.get(key))}")

    macd = ti.get("macd") or {}
    if macd:
        print(f"\n  ── MACD ──")
        for k, v in macd.items():
            print(f"    {k:<30} {v}")

    bb = ti.get("bollinger_bands") or {}
    if bb:
        print(f"\n  ── Bollinger Bands ──")
        for k, v in bb.items():
            print(f"    {k:<30} {v}")

    ma = ti.get("moving_averages") or {}
    if ma:
        print(f"\n  ── Moving Averages ──")
        for k, v in ma.items():
            print(f"    {k:<30} {v}")

    vol = ti.get("volume") or {}
    if vol:
        print(f"\n  ── Volume ──")
        for k, v in vol.items():
            print(f"    {k:<30} {v}")

    print(f"\n  ── Scoring Parameters (consumed by Node 12) ──")
    print(f"    {'normalized_score':<30} {_safe(ti.get('normalized_score'))}")
    print(f"    {'hold_low':<30} {_safe(ti.get('hold_low'))}")
    print(f"    {'hold_high':<30} {_safe(ti.get('hold_high'))}")
    print(f"    {'market_regime':<30} {_safe(ti.get('market_regime'))}")
    print(f"    {'pressure_applied':<30} {_safe(ti.get('pressure_applied'))}")
    print(f"    {'pressure_adjustment':<30} {_safe(ti.get('pressure_adjustment'))}")

    pp = ti.get("persistent_pressure") or {}
    if pp:
        print(f"\n  ── Persistent Pressure ──")
        for k, v in pp.items():
            print(f"    {k:<30} {v}")

    print(f"\n  ── Technical Summary (for Node 14 LLM) ──")
    print(f"  {_safe(ti.get('technical_summary'))}")
else:
    print("  technical_indicators = None  (Node 4 failed)")

# ---------------------------------------------------------------------------
# NODE 5 — Sentiment Analysis
# ---------------------------------------------------------------------------
print_section("NODE 5 — Sentiment Analysis")
print(f"  aggregated_sentiment : {_safe(result.get('aggregated_sentiment'))}")
print(f"  sentiment_signal     : {_safe(result.get('sentiment_signal'))}")
print(f"  sentiment_confidence : {_safe(result.get('sentiment_confidence'))}")

sb = result.get("sentiment_breakdown") or {}
if sb:
    print(f"\n  sentiment_breakdown keys: {list(sb.keys())}")
    # Correct keys: 'stock', 'market', 'related', 'overall'
    for stream_key in ("stock", "market", "related"):
        stream = sb.get(stream_key) or {}
        if stream:
            print(f"\n  [{stream_key}]")
            for k, v in list(stream.items())[:6]:
                if k != "top_articles":
                    print(f"    {k:<30} {v}")
            articles = stream.get("top_articles") or []
            print(f"    top_articles count         : {len(articles)}")

    overall = sb.get("overall") or {}
    if overall:
        print(f"\n  [overall]")
        for k, v in overall.items():
            if k != "credibility":
                print(f"    {k:<30} {v}")
        cred = overall.get("credibility") or {}
        if cred:
            print(f"    credibility:")
            for k, v in cred.items():
                print(f"      {k:<28} {v}")
else:
    print("  sentiment_breakdown = None  (Node 5 failed or not built)")

# ---------------------------------------------------------------------------
# NODE 6 — Market Context
# ---------------------------------------------------------------------------
print_section("NODE 6 — Market Context")
mc = result.get("market_context") or {}
if mc:
    print(f"  context_signal        : {mc.get('context_signal', 'N/A')}")
    print(f"  market_headwind_score : {mc.get('market_headwind_score', 'N/A')}")
    print(f"  market_correlation    : {mc.get('market_correlation', 'N/A')}")
    print(f"  market_trend          : {mc.get('market_trend', 'N/A')}")
    print(f"  spy_5day_change       : {mc.get('spy_5day_change', 'N/A')}")
    print(f"  sector_performance    : {mc.get('sector_performance', 'N/A')}")
    print(f"  beta                  : {mc.get('beta', 'N/A')}")
    print(f"  sector                : {mc.get('sector', 'N/A')}")
    rel = mc.get("related_companies_performance") or {}
    if rel:
        print(f"\n  related_companies_performance (top 5):")
        for t, v in list(rel.items())[:5]:
            print(f"    {t}: {v:+.2f}%" if isinstance(v, float) else f"    {t}: {v}")
else:
    print("  market_context = None  (Node 6 failed or not built)")

# ---------------------------------------------------------------------------
# NODE 7 — Monte Carlo
# ---------------------------------------------------------------------------
print_section("NODE 7 — Monte Carlo Forecasting")
mcr = result.get("monte_carlo_results") or {}
if mcr:
    ci95 = mcr.get("confidence_95") or {}
    print(f"  current_price     : {mcr.get('current_price', 'N/A')}")
    print(f"  mean_forecast     : {mcr.get('mean_forecast', 'N/A')}")
    print(f"  probability_up    : {mcr.get('probability_up', 'N/A')}")
    print(f"  expected_return   : {mcr.get('expected_return', 'N/A')}")
    print(f"  volatility        : {mcr.get('volatility', 'N/A')}")
    print(f"  simulation_count  : {mcr.get('simulation_count', 'N/A')}")
    print(f"  time_horizon_days : {mcr.get('time_horizon_days', 'N/A')}")
    print(f"  95% CI lower      : {ci95.get('lower', 'N/A')}")
    print(f"  95% CI upper      : {ci95.get('upper', 'N/A')}")
else:
    print("  monte_carlo_results = None  (Node 7 failed or not built)")
print(f"  forecasted_price  : {_safe(result.get('forecasted_price'))}")
print(f"  price_range       : {_safe(result.get('price_range'))}")

# ---------------------------------------------------------------------------
# NODE 8 — News Verification / Learning
# ---------------------------------------------------------------------------
print_section("NODE 8 — News Verification & Learning")
niv = result.get("news_impact_verification") or {}
sa  = result.get("sentiment_analysis") or {}
if niv:
    print(f"  learning_adjustment    : {niv.get('learning_adjustment', 'N/A')}")
    print(f"  news_accuracy_score    : {niv.get('news_accuracy_score', 'N/A')}")
    print(f"  historical_correlation : {niv.get('historical_correlation', 'N/A')}")
    print(f"  sample_size            : {niv.get('sample_size', 'N/A')}")
    src_rel = niv.get("source_reliability") or {}
    if src_rel:
        print(f"\n  source_reliability (top 5):")
        for src, v in list(src_rel.items())[:5]:
            print(f"    {src}: accuracy={v.get('accuracy_rate')}  "
                  f"articles={v.get('total_articles')}  "
                  f"multiplier={v.get('confidence_multiplier')}")
else:
    print("  news_impact_verification = None  (Node 8 failed or not built)")
if sa:
    print(f"\n  Adjusted sentiment_analysis:")
    print(f"    aggregated_sentiment   : {sa.get('aggregated_sentiment', 'N/A')}")
    print(f"    stock_news_sentiment   : {sa.get('stock_news_sentiment', 'N/A')}")
    print(f"    market_news_sentiment  : {sa.get('market_news_sentiment', 'N/A')}")
    print(f"    related_news_sentiment : {sa.get('related_news_sentiment', 'N/A')}")
else:
    print("  sentiment_analysis = None  (Node 8 did not adjust sentiment)")

# ---------------------------------------------------------------------------
# NODE 9B — Behavioral Anomaly Detection
# ---------------------------------------------------------------------------
print_section("NODE 9B — Behavioral Anomaly Detection")
bad = result.get("behavioral_anomaly_detection") or {}
if bad:
    print(f"  risk_level            : {bad.get('risk_level', 'N/A')}")
    print(f"  pump_and_dump_score   : {bad.get('pump_and_dump_score', 'N/A')}")
    print(f"  trading_safe          : {bad.get('trading_safe', 'N/A')}")
    print(f"  trading_recommendation: {bad.get('trading_recommendation', 'N/A')}")
    print(f"  behavioral_summary    : {bad.get('behavioral_summary', 'N/A')}")
    print(f"  primary_risk_factors  : {bad.get('primary_risk_factors', 'N/A')}")
    print(f"  alerts                : {bad.get('alerts', 'N/A')}")
    dd = bad.get("detection_breakdown") or {}
    if dd:
        print(f"\n  detection_breakdown:")
        for k, v in dd.items():
            print(f"    {k:<30} {v}")
else:
    print("  behavioral_anomaly_detection = None  (Node 9B failed or not built)")

# ---------------------------------------------------------------------------
# NODE 10 — Backtesting
# ---------------------------------------------------------------------------
print_section("NODE 10 — Backtesting")
br = result.get("backtest_results") or {}
if br:
    for stream in ("technical", "stock_news", "market_news", "related_news"):
        sd = br.get(stream) or {}
        if isinstance(sd, dict):
            print(f"  {stream:<18}  "
                  f"full_acc={sd.get('full_accuracy','N/A')}  "
                  f"recent_acc={sd.get('recent_accuracy','N/A')}  "
                  f"significant={sd.get('is_significant','N/A')}  "
                  f"sufficient={sd.get('is_sufficient','N/A')}")
        else:
            print(f"  {stream}: {sd}")
else:
    print("  backtest_results = None  (Node 10 failed or not built)")

# ---------------------------------------------------------------------------
# NODE 11 — Adaptive Weights
# ---------------------------------------------------------------------------
print_section("NODE 11 — Adaptive Weights")
aw = result.get("adaptive_weights") or {}
if aw:
    for k, v in aw.items():
        print(f"  {k:<35} {_fmt_float(v) if isinstance(v, float) else v}")
else:
    print("  adaptive_weights = None  (Node 11 failed or not built)")

# ---------------------------------------------------------------------------
# NODE 12 — Final Signal Generation
# ---------------------------------------------------------------------------
print_section("NODE 12 — Final Signal Generation")

# Top-level state fields written by Node 12
print(f"  final_signal     : {_safe(result.get('final_signal'))}")
print(f"  final_confidence : {_fmt_float(result.get('final_confidence'))}")

sc = result.get("signal_components") or {}
if sc:
    print(f"\n  ── Job 1: Weighted Signal Combination ──")
    print(f"    final_score      : {_fmt_float(sc.get('final_score'), 6)}")
    print(f"    signal_agreement : {_safe(sc.get('signal_agreement'))}/4 streams")
    print(f"    streams_missing  : {sc.get('streams_missing') or 'none'}")

    ss = sc.get("stream_scores") or {}
    if ss:
        print(f"\n  ── Stream Scores (raw → weighted contribution) ──")
        for stream in ("technical", "sentiment", "market", "monte_carlo"):
            d = ss.get(stream) or {}
            print(f"    {stream:<14}  "
                  f"raw={_fmt_float(d.get('raw_score'), 4)}  "
                  f"weight={_fmt_float(d.get('weight'), 4)}  "
                  f"contribution={_fmt_float(d.get('contribution'), 4)}")

    print(f"\n  ── Job 2: Historical Pattern Matching ──")
    pp = sc.get("pattern_prediction") or {}
    print(f"    sufficient_data          : {_safe(pp.get('sufficient_data'))}")
    if pp.get("sufficient_data"):
        print(f"    similar_days_found       : {_safe(pp.get('similar_days_found'))}")
        print(f"    similarity_threshold     : {_safe(pp.get('similarity_threshold_used'))}")
        print(f"    prob_up                  : {_fmt_float(pp.get('prob_up'), 4)}")
        print(f"    prob_down                : {_fmt_float(pp.get('prob_down'), 4)}")
        print(f"    expected_return_7d       : {_fmt_float(pp.get('expected_return_7d'), 4)}%")
        print(f"    worst_case_7d (10th pct) : {_fmt_float(pp.get('worst_case_7d'), 4)}%")
        print(f"    median_return_7d         : {_fmt_float(pp.get('median_return_7d'), 4)}%")
        print(f"    best_case_7d  (90th pct) : {_fmt_float(pp.get('best_case_7d'), 4)}%")

        detail = pp.get("similar_days_detail") or []
        if detail:
            print(f"\n    Top similar historical days:")
            for d in detail:
                print(f"      {d.get('date')}  "
                      f"sim={_fmt_float(d.get('similarity_score'), 3)}  "
                      f"change={_fmt_float(d.get('actual_change_7d'), 2)}%  "
                      f"{d.get('direction')}")
    else:
        print(f"    reason                   : {_safe(pp.get('reason'))}")

    print(f"\n  ── Job 1 vs Job 2 Agreement (defines confidence adjustment) ──")
    print(f"    agreement_with_job1      : {_safe(pp.get('agreement_with_job1'))}")
    print(f"    confidence_multiplier    : {_fmt_float(pp.get('confidence_multiplier'), 4)}")
    print(f"    raw_confidence (pre-adj) : derived from abs(final_score)")
    print(f"    final_confidence (adj)   : {_fmt_float(result.get('final_confidence'), 4)}")

    print(f"\n  ── Prediction Blending (Job 2 empirical 60% + GBM 40%) ──")
    pg = sc.get("prediction_graph_data") or {}
    print(f"    data_source              : {_safe(pg.get('data_source'))}")
    print(f"    blended_expected_return  : {_fmt_float(pg.get('blended_expected_return'), 4)}%")
    print(f"    gbm_expected_return      : {_fmt_float(pg.get('gbm_expected_return'), 4)}%")
    print(f"    empirical_expected_return: {_safe(pg.get('empirical_expected_return'))}")
    print(f"    gbm_spread               : {_fmt_float(pg.get('gbm_spread_lower'), 4)}% "
          f"→ {_fmt_float(pg.get('gbm_spread_upper'), 4)}%")
    print(f"    empirical_spread         : {_safe(pg.get('empirical_lower'))} "
          f"→ {_safe(pg.get('empirical_upper'))}")

    print(f"\n  ── Price Targets ──")
    pt = sc.get("price_targets") or {}
    print(f"    current_price     : {_safe(pt.get('current_price'))}")
    print(f"    forecasted_price  : {_safe(pt.get('forecasted_price'))}")
    print(f"    price_range_lower : {_safe(pt.get('price_range_lower'))}")
    print(f"    price_range_upper : {_safe(pt.get('price_range_upper'))}")
    print(f"    expected_return   : {_safe(pt.get('expected_return_pct'))}")

    print(f"\n  ── Risk Summary ──")
    rs = sc.get("risk_summary") or {}
    print(f"    overall_risk_level    : {_safe(rs.get('overall_risk_level'))}")
    print(f"    trading_safe          : {_safe(rs.get('trading_safe'))}")
    print(f"    trading_recommendation: {_safe(rs.get('trading_recommendation'))}")
    print(f"    pump_and_dump_score   : {_safe(rs.get('pump_and_dump_score'))}")

    print(f"\n  ── Backtest Context ──")
    bc = sc.get("backtest_context") or {}
    print(f"    hold_threshold_pct   : {_safe(bc.get('hold_threshold_pct'))}")
    print(f"    streams_reliable     : {_safe(bc.get('streams_reliable'))}")
    print(f"    weights_are_fallback : {_safe(bc.get('weights_are_fallback'))}")
else:
    print("  signal_components = None  (Node 12 failed or not built)")

# ---------------------------------------------------------------------------
# System tracking
# ---------------------------------------------------------------------------
print_section("SYSTEM TRACKING — Node Execution Times")
net = result.get("node_execution_times") or {}
total = 0.0
for node, t in sorted(net.items()):
    print(f"  {node:<40} {t:.3f}s")
    total += t
print(f"\n  Sum of node times    : {total:.3f}s")
print(f"  total_execution_time : {_safe(result.get('total_execution_time'))}")
errors = result.get("errors") or []
print(f"\n  Errors ({len(errors)}):")
if errors:
    for e in errors:
        print(f"    ✗ {e}")
else:
    print("    ✓ No errors")

# ---------------------------------------------------------------------------
# NODE 13 — Beginner Explanation (full LLM response)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  NODE 13 — BEGINNER EXPLANATION (full LLM response)")
print(SEP)
print(result.get("beginner_explanation") or "(not generated — Node 13 failed or not built)")

# ---------------------------------------------------------------------------
# NODE 14 — Technical Explanation (full LLM response)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  NODE 14 — TECHNICAL EXPLANATION (full LLM response)")
print(SEP)
print(result.get("technical_explanation") or "(not generated — Node 14 failed or not built)")

print(f"\n{SEP}\n")