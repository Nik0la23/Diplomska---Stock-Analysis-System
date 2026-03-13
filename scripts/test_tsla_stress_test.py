"""
TSLA Stress Test — Full Pipeline with Per-Node Health Checks

Every node has:
  - A None/empty check on every output field it is supposed to write
  - A real-world range check where applicable
  - A PASS / FAIL tally that prints at the end

Node 6 (Market Context) is inspected especially deeply because it is the
most fragile node — it fetches SPY, sector ETF (XLY), and peer data live
from yfinance, and any one of those calls can silently return None and
corrupt the market stream all the way through to Node 12.

REAL-WORLD GROUND TRUTH (March 11, 2026)
------------------------------------------
  Current price          : ~$396
  52-week high           : $498.83  (Dec 22 2025)
  Drawdown from peak     : -19.9%
  YTD performance        : -11.1%
  Analyst consensus      : HOLD  (27 analysts, 10/34 say SELL)
  Consensus price target : $396.23
  Below 50-day SMA since : early January 2026
  Sector                 : Consumer Discretionary  (XLY, ~19% weight)
  XLY YTD               : -3.7%
  Beta                   : ~2.5
  Q4 2025 earnings       : missed — auto revenue -11% YoY

Usage:
    python scripts/test_tsla_stress_test.py
"""

import os
import sys
import logging
from pathlib import Path

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"]    = "false"

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
for noisy in ("httpx", "httpcore", "urllib3", "yfinance", "peewee",
              "transformers", "langsmith", "anthropic", "openai",
              "sentence_transformers", "torch", "filelock", "PIL"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from src.graph.workflow import run_stock_analysis

TICKER = "IONQ"
SEP  = "=" * 72
DASH = "─" * 60

# ---------------------------------------------------------------------------
# Check infrastructure — every check appends to FAILURES or PASSES
# ---------------------------------------------------------------------------
FAILURES = []   # (node_label, field, description)
PASSES   = []

def _safe(v, default="N/A"):
    return v if v is not None else default

def _fmt(v, d=4, default="N/A"):
    if v is None:
        return default
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)

def _sec(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

def _pass(node, field, note=""):
    PASSES.append((node, field))
    tag = f"  ✓  {node:<10} {field}"
    if note:
        tag += f"  → {note}"
    print(tag)

def _fail(node, field, actual, note=""):
    FAILURES.append((node, field, actual, note))
    tag = f"  🚩 {node:<10} {field} = {actual!r}"
    if note:
        tag += f"  → {note}"
    print(tag)

def chk_present(node, field, value):
    """Fail if value is None, empty dict, or empty list."""
    empty = (value is None
             or (isinstance(value, dict) and len(value) == 0)
             or (isinstance(value, list) and len(value) == 0)
             or (isinstance(value, str) and value.strip() == ""))
    if empty:
        _fail(node, field, value, "MISSING — node may have failed or not written this field")
        return False
    _pass(node, field)
    return True

def chk_range(node, field, value, lo, hi, note=""):
    """Fail if numeric value is outside [lo, hi]."""
    if value is None:
        _fail(node, field, None, f"None — expected {lo}–{hi}")
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        _fail(node, field, value, f"not numeric — expected {lo}–{hi}")
        return False
    if lo <= v <= hi:
        _pass(node, field, f"{v:.4g}  [OK: {lo}–{hi}]" + (f"  {note}" if note else ""))
        return True
    else:
        _fail(node, field, v, f"outside [{lo}–{hi}]" + (f"  {note}" if note else ""))
        return False

def chk_one_of(node, field, value, options, note=""):
    """Fail if value is not in the options list."""
    if value is None:
        _fail(node, field, None, f"None — expected one of {options}")
        return False
    if value in options:
        _pass(node, field, f"'{value}'  [OK: {options}]" + (f"  {note}" if note else ""))
        return True
    else:
        _fail(node, field, value, f"expected one of {options}" + (f"  {note}" if note else ""))
        return False

def chk_len(node, field, value, min_len=1, note=""):
    """Fail if list/dict/string has fewer than min_len items."""
    if value is None:
        _fail(node, field, None, f"None — expected ≥{min_len} items")
        return False
    n = len(value)
    if n >= min_len:
        _pass(node, field, f"len={n}  [OK: ≥{min_len}]" + (f"  {note}" if note else ""))
        return True
    else:
        _fail(node, field, f"len={n}", f"expected ≥{min_len}" + (f"  {note}" if note else ""))
        return False

def info(text):
    print(f"  ℹ️  {text}")

# ---------------------------------------------------------------------------
# DB pre-check
# ---------------------------------------------------------------------------
print(SEP)
print("  TSLA STRESS TEST — Per-Node Health Checks + Real-World Comparison")
print("  Ground truth: HOLD consensus ~$396, 30% analysts SELL, -19% from peak")
print(SEP)

try:
    import sqlite3
    conn = sqlite3.connect(str(_root / "data" / "stock_prices.db"))
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM price_data WHERE ticker='TSLA'")
    pr = cur.fetchone()
    print(f"\n  [DB] TSLA price rows : {pr[0]}  ({pr[1]} → {pr[2]})")
    cur.execute("SELECT COUNT(*) FROM news_articles WHERE ticker='TSLA'")
    print(f"  [DB] TSLA news rows  : {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM news_outcomes WHERE ticker='TSLA'")
    print(f"  [DB] TSLA outcomes   : {cur.fetchone()[0]}")
    conn.close()
except Exception as e:
    print(f"  [DB] Could not query: {e}")

print(f"\n  Running pipeline for {TICKER}…\n")
result = run_stock_analysis(TICKER)

# ===========================================================================
#  NODE 1 — Price Data
# ===========================================================================
_sec("NODE 1 — Price Data Fetching")
df = result.get("raw_price_data")

if not chk_present("Node 1", "raw_price_data", df):
    info("All downstream nodes will fail without price data — check yfinance / ticker symbol")
else:
    chk_len   ("Node 1", "row_count",    df, 50,
                "need ≥50 rows for technical analysis")
    chk_range ("Node 1", "row_count",    len(df), 50, 400,
                ">400 rows may indicate wrong date window")
    required = {"open", "high", "low", "close", "volume"}
    actual   = set(c.lower() for c in df.columns)
    missing  = required - actual
    if missing:
        _fail("Node 1", "ohlcv_columns", str(missing), "required columns absent")
    else:
        _pass("Node 1", "ohlcv_columns", "open/high/low/close/volume all present")
    print(f"\n  Date range : {df.index[0] if hasattr(df,'index') else '?'}  →  "
          f"{df.index[-1] if hasattr(df,'index') else '?'}")
    print(f"  Columns    : {list(df.columns)}")

# ===========================================================================
#  NODE 2 — News Fetching
# ===========================================================================
_sec("NODE 2 — News Fetching")
sn  = result.get("stock_news")           or []
mn  = result.get("market_news")          or []
rn  = result.get("related_company_news") or []
nfm = result.get("news_fetch_metadata")  or {}

chk_len("Node 2", "stock_news",           sn,  5,
        "TSLA should have abundant stock-specific news")
chk_len("Node 2", "market_news",          mn,  min_len=3)
chk_len("Node 2", "related_company_news", rn,  min_len=1)
chk_present("Node 2", "news_fetch_metadata", nfm)

total = len(sn) + len(mn) + len(rn)
chk_range("Node 2", "total_articles", total, 20, 20_000,
          "<20 means API fetch failed; TSLA regularly fetches 5k-15k articles from DB")
print(f"\n  stock={len(sn)}  market={len(mn)}  related={len(rn)}  total={total}")

# ===========================================================================
#  NODE 3 — Related Companies
# ===========================================================================
_sec("NODE 3 — Related Companies")
rc = result.get("related_companies") or []
chk_len("Node 3", "related_companies", rc, 1,
        "Expected RIVN, NIO, GM, F, LCID, or similar EV peers")
print(f"  Peers : {rc}")

# ===========================================================================
#  NODE 9A — Content Analysis & Feature Extraction
# ===========================================================================
_sec("NODE 9A — Content Analysis & Feature Extraction")
csn = result.get("cleaned_stock_news")           or []
cmn = result.get("cleaned_market_news")          or []
crn = result.get("cleaned_related_company_news") or []
cas = result.get("content_analysis_summary")     or {}

chk_len    ("Node 9A", "cleaned_stock_news",           csn, min_len=1)
chk_len    ("Node 9A", "cleaned_market_news",          cmn, min_len=1)
chk_len    ("Node 9A", "cleaned_related_company_news", crn, min_len=1)
chk_present("Node 9A", "content_analysis_summary",     cas)

if cas:
    # Node 9A writes: average_scores, high_risk_articles, source_credibility_distribution, top_keywords
    # Fields early_risk_level / anomaly_flags / suspicious_patterns are NOT written by this node.
    avg = cas.get("average_scores") or {}
    chk_present("Node 9A", "average_scores",                  avg)
    chk_present("Node 9A", "average_scores.composite_anomaly",avg.get("composite_anomaly"))
    chk_present("Node 9A", "average_scores.source_credibility",avg.get("source_credibility"))
    chk_present("Node 9A", "high_risk_articles",              cas.get("high_risk_articles"))
    chk_present("Node 9A", "high_risk_percentage",            cas.get("high_risk_percentage"))
    chk_present("Node 9A", "source_credibility_distribution", cas.get("source_credibility_distribution"))
    chk_present("Node 9A", "total_articles_processed",        cas.get("total_articles_processed"))
    if avg.get("composite_anomaly") is not None:
        chk_range("Node 9A", "avg_composite_anomaly", avg["composite_anomaly"], 0.0, 1.0)
    if avg.get("composite_anomaly", 0) > 0.6:
        info("avg composite_anomaly > 0.6 for TSLA — high news risk score; check individual articles")

# ===========================================================================
#  NODE 4 — Technical Analysis
# ===========================================================================
_sec("NODE 4 — Technical Analysis")
ti = result.get("technical_indicators") or {}
chk_present("Node 4", "technical_indicators", ti)

# These two fields intentionally remain None — Node 4 no longer writes them
ts = result.get("technical_signal")
tc = result.get("technical_confidence")
print(f"\n  technical_signal     : {_safe(ts)}  ← intentionally None (Node 4 no longer writes this)")
print(f"  technical_confidence : {_safe(tc)}  ← intentionally None (Node 4 no longer writes this)")
if ts is not None:
    _fail("Node 4", "technical_signal",
          ts, "should be None — Node 4 must NOT produce a live signal (architecture rule)")
if tc is not None:
    _fail("Node 4", "technical_confidence",
          tc, "should be None — Node 4 must NOT produce a live signal (architecture rule)")

if ti:
    print()
    # Sub-indicators
    chk_present("Node 4", "rsi",             ti.get("rsi"))
    chk_present("Node 4", "adx",             ti.get("adx"))
    chk_present("Node 4", "macd",            ti.get("macd"))
    chk_present("Node 4", "bollinger_bands", ti.get("bollinger_bands"))
    chk_present("Node 4", "moving_averages", ti.get("moving_averages"))
    chk_present("Node 4", "volume",          ti.get("volume"))

    # Real-world range checks
    chk_range("Node 4", "rsi", ti.get("rsi"), 20, 70,
              "valid RSI range; oversold (<30) or overbought (>70) are also real signals")
    chk_range("Node 4", "adx", ti.get("adx"), 10, 45,
              ">25 = trending, <18 = choppy; both valid for TSLA right now")

    bb = ti.get("bollinger_bands") or {}
    if bb:
        chk_range("Node 4", "bollinger_bands.bandwidth", bb.get("bandwidth"), 5, 80,
                  "TSLA's beta ~2.5 means wide bands; <5 is suspiciously compressed")
        chk_present("Node 4", "bollinger_bands.position", bb.get("position"))

    ma = ti.get("moving_averages") or {}
    if ma:
        chk_present("Node 4", "moving_averages.trend",       ma.get("trend"))
        chk_present("Node 4", "moving_averages.sma_20",      ma.get("sma_20"))
        chk_present("Node 4", "moving_averages.sma_50",      ma.get("sma_50"))
        chk_one_of ("Node 4", "moving_averages.trend",
                    ma.get("trend"),
                    ["downtrend", "strong_downtrend", "neutral"],
                    "price below 50-SMA since Jan 2026 — uptrend would be surprising")

    macd = ti.get("macd") or {}
    if macd:
        chk_present("Node 4", "macd.macd",      macd.get("macd"))
        chk_present("Node 4", "macd.signal",    macd.get("signal"))
        chk_present("Node 4", "macd.histogram", macd.get("histogram"))
        chk_present("Node 4", "macd.crossover", macd.get("crossover"))

    # Scoring fields consumed by Node 12 — these MUST exist
    print()
    chk_present("Node 4", "normalized_score",   ti.get("normalized_score"))
    chk_present("Node 4", "hold_low",           ti.get("hold_low"))
    chk_present("Node 4", "hold_high",          ti.get("hold_high"))
    chk_present("Node 4", "market_regime",      ti.get("market_regime"))
    chk_present("Node 4", "pressure_applied",   ti.get("pressure_applied"))
    chk_present("Node 4", "pressure_adjustment",ti.get("pressure_adjustment"))
    chk_present("Node 4", "persistent_pressure",ti.get("persistent_pressure"))
    chk_present("Node 4", "technical_summary",  ti.get("technical_summary"))

    chk_range  ("Node 4", "normalized_score", ti.get("normalized_score"), 15, 58,
                ">58 is bullish but TSLA is 19% below its peak — unlikely")
    chk_one_of ("Node 4", "market_regime",
                ti.get("market_regime"),
                ["trending_down", "choppy", "trending_up"],
                "trending_down or choppy expected; trending_up would be surprising")

    pp = ti.get("persistent_pressure") or {}
    if pp:
        chk_present("Node 4", "persistent_pressure.pressure_direction", pp.get("pressure_direction"))
        chk_present("Node 4", "persistent_pressure.pressure_strength",  pp.get("pressure_strength"))
        chk_present("Node 4", "persistent_pressure.consecutive_below_sma", pp.get("consecutive_below_sma"))
        chk_present("Node 4", "persistent_pressure.sma20_slope",        pp.get("sma20_slope"))
        chk_present("Node 4", "persistent_pressure.macd_histogram_slope",pp.get("macd_histogram_slope"))
        chk_one_of ("Node 4", "persistent_pressure.pressure_direction",
                    pp.get("pressure_direction"), ["bearish", "neutral", "bullish"],
                    "bearish expected given multi-week grind below SMA20")

    print(f"\n  normalized_score  : {_safe(ti.get('normalized_score'))}")
    print(f"  hold_low/high     : {_safe(ti.get('hold_low'))} / {_safe(ti.get('hold_high'))}")
    print(f"  market_regime     : {_safe(ti.get('market_regime'))}")
    print(f"  pressure_applied  : {_safe(ti.get('pressure_applied'))}")
    print(f"  pressure_dir      : {_safe((ti.get('persistent_pressure') or {}).get('pressure_direction'))}")
    print(f"  technical_summary : {_safe(ti.get('technical_summary'))}")

# ===========================================================================
#  NODE 5 — Sentiment Analysis
# ===========================================================================
_sec("NODE 5 — Sentiment Analysis")
agg  = result.get("aggregated_sentiment")
ssig = result.get("sentiment_signal")
scon = result.get("sentiment_confidence")
sb   = result.get("sentiment_breakdown") or {}

chk_present("Node 5", "aggregated_sentiment",  agg)
chk_present("Node 5", "sentiment_signal",       ssig)
chk_present("Node 5", "sentiment_confidence",   scon)
chk_present("Node 5", "sentiment_breakdown",    sb)

if agg is not None:
    chk_range("Node 5", "aggregated_sentiment", agg, -0.50, 0.50,
              "TSLA news is split — extreme scores (< -0.5 or > 0.5) mean credibility weighting is broken")

if sb:
    for key in ("stock", "market", "related", "overall"):
        chk_present("Node 5", f"sentiment_breakdown['{key}']", sb.get(key))
    for key in ("stock", "market", "related"):
        stream = sb.get(key) or {}
        if stream:
            chk_present("Node 5", f"sentiment_breakdown['{key}'].weighted_sentiment",
                        stream.get("weighted_sentiment"))
            chk_present("Node 5", f"sentiment_breakdown['{key}'].article_count",
                        stream.get("article_count"))
            chk_present("Node 5", f"sentiment_breakdown['{key}'].dominant_label",
                        stream.get("dominant_label"))

    print(f"\n  aggregated_sentiment : {_safe(agg)}")
    print(f"  sentiment_signal     : {_safe(ssig)}")
    print(f"  sentiment_confidence : {_safe(scon)}")
    for key in ("stock", "market", "related"):
        s = sb.get(key) or {}
        print(f"  [{key}]  weighted={_safe(s.get('weighted_sentiment'))}  "
              f"articles={_safe(s.get('article_count'))}  "
              f"dominant={_safe(s.get('dominant_label'))}")

# ===========================================================================
#  NODE 6 — Market Context  ← SUSPECTED PROBLEM NODE
# ===========================================================================
_sec("NODE 6 — Market Context  ⚠️  [SUSPECTED FRAGILE — DEEP INSPECTION]")
info("TSLA is in Consumer Discretionary (XLY). Sector -11% YTD. Beta ~2.5.")
info("Node 6 fetches SPY, XLY, peer data live from yfinance — any call can silently return None.")
info("If market_context is None here, Node 12's 'market' stream will be 0 contribution.")

mc6 = result.get("market_context") or {}

# Top-level presence check — if this fails the rest of the section is moot
if not chk_present("Node 6", "market_context", mc6):
    print(f"\n  🚩 market_context is completely None.")
    print(f"     Possible causes:")
    print(f"       1. yfinance rate-limit or timeout fetching SPY data")
    print(f"       2. Sector ETF lookup failed (TSLA → XLY mapping)")
    print(f"       3. Beta calculation raised an exception and the whole node crashed")
    print(f"       4. Node 6 has no error handling and returned early on first None")
    print(f"     Impact: Node 12 market stream weight = 0, signal based on 3/4 streams only")
    print(f"     Fix: Add try/except per sub-fetch in Node 6; default each to None not crash")
else:
    # Every field Node 6 is supposed to write
    required_fields = [
        ("context_signal",               ["SELL","HOLD","BUY"],    "node 6 final opinion on market context"),
        ("market_trend",                 None,                     "overall direction of SPY"),
        ("market_headwind_score",        None,                     "negative = headwind for TSLA"),
        ("market_correlation",           None,                     "TSLA vs SPY correlation"),
        ("beta",                         None,                     "TSLA beta — should be ~2.5"),
        ("sector",                       None,                     "Consumer Discretionary"),
        ("sector_performance",           None,                     "XLY YTD ~ -3.7%"),
        ("spy_return_5d",                None,                     "SPY 5-day price return %"),
        ("related_companies_signals",    None,                     "list of peer performance records"),
        ("vix_category",                 None,                     "LOW/MODERATE/HIGH/EXTREME — TSLA should be HIGH/EXTREME"),
    ]

    for fname, options, note in required_fields:
        val = mc6.get(fname)
        if options:
            chk_one_of("Node 6", fname, val, options, note)
        else:
            chk_present("Node 6", fname, val)

    # Range checks against real-world data
    chk_present("Node 6", "beta",               mc6.get("beta"))
    chk_present("Node 6", "market_correlation", mc6.get("market_correlation"))
    import math as _math
    if mc6.get("beta") is not None and not _math.isfinite(float(mc6["beta"])):
        _fail("Node 6", "beta", mc6["beta"], "non-finite value")
    if mc6.get("market_correlation") is not None and not _math.isfinite(float(mc6["market_correlation"])):
        _fail("Node 6", "market_correlation", mc6["market_correlation"], "non-finite value")
    # After the date-alignment fix, beta and correlation should be physically plausible.
    # Wide range to accommodate genuine market regime changes.
    chk_range("Node 6", "beta",
              mc6.get("beta"), 0.5, 5.0,
              "TSLA beta historically ~2.0-2.8; negative beta means date-alignment bug is back")
    chk_range("Node 6", "market_correlation",
              mc6.get("market_correlation"), 0.0, 1.0,
              "TSLA moves with the market; negative correlation means misaligned date windows")

    chk_range("Node 6", "sector_performance",
              mc6.get("sector_performance"), -20.0, 10.0,
              "XLY YTD ~ -3.7%; values outside -20 to +10 suggest wrong ETF or date window")

    chk_range("Node 6", "spy_return_5d",
              mc6.get("spy_return_5d"), -8.0, 8.0,
              "SPY rarely moves >8% in a 5-day window outside crashes")

    chk_range("Node 6", "market_headwind_score",
              mc6.get("market_headwind_score"), -1.0, 1.0,
              "should be negative-to-neutral given sector headwinds")

    # Sector identity check — TSLA must map to Consumer Discretionary
    sector = mc6.get("sector")
    if sector:
        expected_sectors = [
            "Consumer Discretionary", "consumer_discretionary",
            "ConsumerDiscretionary", "Consumer Cyclical",
            "Technology", "Automobiles"
        ]
        if sector not in expected_sectors:
            _fail("Node 6", "sector", sector,
                  f"TSLA is Consumer Discretionary per GICS — got '{sector}'")
        else:
            _pass("Node 6", "sector", f"'{sector}'")

    # related_companies_signals — must be a non-empty list of {ticker, performance, trend}
    rcs = mc6.get("related_companies_signals") or []
    chk_len("Node 6", "related_companies_signals", rcs, 1,
            "should contain at least one peer (RIVN, NIO, GM, F…)")

    # Individual peer checks — performance should be a reasonable price change %
    for rec in rcs:
        peer   = rec.get("ticker", "?")
        change = rec.get("performance")
        if change is not None:
            chk_range("Node 6", f"related_companies_signals['{peer}'].performance",
                      change, -50.0, 50.0,
                      "single-period price change >50% is suspicious unless major news")
        else:
            _fail("Node 6", f"related_companies_signals['{peer}'].performance",
                  None, "peer fetch returned None — yfinance timeout?")

    # vix_category check — Node 6 uses: "LOW", "MODERATE", "ELEVATED", "HIGH", "EXTREME"
    vr = mc6.get("vix_category")
    if vr:
        chk_one_of("Node 6", "vix_category", vr,
                   ["LOW", "MODERATE", "ELEVATED", "HIGH", "EXTREME"],
                   "TSLA beta ~2.5 → expected 'ELEVATED', 'HIGH', or 'EXTREME'")

    # Print everything raw so you can spot surprises
    print(f"\n  ── All Node 6 fields ──")
    for k, v in mc6.items():
        if k == "related_companies_signals":
            print(f"  {'related_companies_signals':<35}")
            for rec in (v or []):
                print(f"    {rec.get('ticker','?')}: perf={rec.get('performance')}  trend={rec.get('trend')}")
        else:
            print(f"  {k:<35} {v}")

# ===========================================================================
#  NODE 7 — Monte Carlo
# ===========================================================================
_sec("NODE 7 — Monte Carlo Forecasting")
mcr = result.get("monte_carlo_results") or {}

chk_present("Node 7", "monte_carlo_results",  mcr)
chk_present("Node 7", "forecasted_price",     result.get("forecasted_price"))
chk_present("Node 7", "price_range",          result.get("price_range"))

if mcr:
    chk_present("Node 7", "current_price",   mcr.get("current_price"))
    chk_present("Node 7", "mean_forecast",   mcr.get("mean_forecast"))
    chk_present("Node 7", "probability_up",  mcr.get("probability_up"))
    chk_present("Node 7", "expected_return", mcr.get("expected_return"))
    chk_present("Node 7", "volatility",      mcr.get("volatility"))
    # Node 7 stores simulation count as 'num_simulations', not 'simulation_count'
    chk_present("Node 7", "num_simulations", mcr.get("num_simulations"))
    chk_present("Node 7", "confidence_95",   mcr.get("confidence_95"))

    ci = mcr.get("confidence_95") or {}
    chk_present("Node 7", "confidence_95.lower", ci.get("lower"))
    chk_present("Node 7", "confidence_95.upper", ci.get("upper"))

    chk_range("Node 7", "probability_up",  mcr.get("probability_up"), 0.30, 0.70,
              "analyst consensus is HOLD with flat target — result should be near coin-flip")
    chk_range("Node 7", "expected_return", mcr.get("expected_return"), -10.0, 8.0,
              "flat consensus target implies near-zero expected return")
    # Node 7 stores DAILY volatility (std of daily returns), not annualised.
    # TSLA daily vol is typically 0.015–0.045 (i.e. 1.5–4.5% per day).
    chk_range("Node 7", "volatility",      mcr.get("volatility"), 0.005, 0.08,
              "daily volatility; TSLA typically 0.015–0.040 per day")
    chk_range("Node 7", "num_simulations", mcr.get("num_simulations"), 100, 100_000,
              "<100 simulations produces unreliable statistics")

    print(f"\n  current_price   : {mcr.get('current_price')}")
    print(f"  mean_forecast   : {mcr.get('mean_forecast')}")
    print(f"  probability_up  : {mcr.get('probability_up')}")
    print(f"  expected_return : {mcr.get('expected_return')}")
    print(f"  volatility      : {mcr.get('volatility')}  (daily)")
    print(f"  num_simulations : {mcr.get('num_simulations')}")
    print(f"  95% CI          : {ci.get('lower')} → {ci.get('upper')}")

# ===========================================================================
#  NODE 8 — News Learning & Verification
# ===========================================================================
_sec("NODE 8 — News Learning & Verification")
niv = result.get("news_impact_verification") or {}
sa8 = result.get("sentiment_analysis")       or {}

chk_present("Node 8", "news_impact_verification", niv)
chk_present("Node 8", "sentiment_analysis",        sa8)

if niv:
    chk_present("Node 8", "learning_adjustment",    niv.get("learning_adjustment"))
    chk_present("Node 8", "news_accuracy_score",    niv.get("news_accuracy_score"))
    chk_present("Node 8", "historical_correlation", niv.get("historical_correlation"))
    chk_present("Node 8", "sample_size",            niv.get("sample_size"))
    chk_present("Node 8", "source_reliability",     niv.get("source_reliability"))

    src = niv.get("source_reliability") or {}
    chk_len("Node 8", "source_reliability", src, 1,
            "should contain at least one source entry")

    print(f"\n  learning_adjustment    : {niv.get('learning_adjustment')}")
    print(f"  news_accuracy_score    : {niv.get('news_accuracy_score')}")
    print(f"  historical_correlation : {niv.get('historical_correlation')}")
    print(f"  sample_size            : {niv.get('sample_size')}")
    if src:
        print(f"  Top sources:")
        for s, v in list(src.items())[:4]:
            print(f"    {s}: acc={v.get('accuracy_rate')}  "
                  f"n={v.get('total_articles')}  "
                  f"mult={v.get('confidence_multiplier')}")

if sa8:
    # Node 8 builds sentiment_analysis from Node 5's flat state keys:
    #   {confidence, signal, aggregated_sentiment} + confidence_adjustment added by Node 8.
    # Per-stream sub-fields (stock/market/related_news_sentiment) are NOT written here;
    # they live in state['sentiment_breakdown'] from Node 5.
    chk_present("Node 8", "sentiment_analysis.aggregated_sentiment",
                sa8.get("aggregated_sentiment"))
    chk_present("Node 8", "sentiment_analysis.confidence",
                sa8.get("confidence"))
    chk_present("Node 8", "sentiment_analysis.signal",
                sa8.get("signal"))
    chk_present("Node 8", "sentiment_analysis.confidence_adjustment",
                sa8.get("confidence_adjustment"))

# ===========================================================================
#  NODE 9B — Behavioral Anomaly Detection
# ===========================================================================
_sec("NODE 9B — Behavioral Anomaly Detection")
bad = result.get("behavioral_anomaly_detection") or {}
chk_present("Node 9B", "behavioral_anomaly_detection", bad)

if bad:
    chk_present("Node 9B", "risk_level",             bad.get("risk_level"))
    chk_present("Node 9B", "pump_and_dump_score",    bad.get("pump_and_dump_score"))
    # trading_safe is NOT written by Node 9B; it lives in signal_components.risk_summary (Node 12)
    chk_present("Node 9B", "trading_recommendation", bad.get("trading_recommendation"))
    chk_present("Node 9B", "behavioral_summary",     bad.get("behavioral_summary"))
    chk_present("Node 9B", "detection_breakdown",    bad.get("detection_breakdown"))

    chk_one_of ("Node 9B", "risk_level",
                bad.get("risk_level"), ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                "LOW or MEDIUM expected — TSLA is volatile but legitimate")
    chk_range  ("Node 9B", "pump_and_dump_score",
                bad.get("pump_and_dump_score"), 0, 65,
                ">65 is likely a false positive — TSLA is a $1.5T company with real products")

    pds = bad.get("pump_and_dump_score")
    rl  = bad.get("risk_level")
    if pds and float(pds) > 65:
        info(f"pump_and_dump_score={pds} is very high for TSLA. "
             "Node 9B may be miscalibrated for large-cap high-beta stocks.")
    if rl in ("HIGH", "CRITICAL"):
        info(f"risk_level={rl} for TSLA — this is likely a false positive. "
             "Check Node 9B detection_breakdown to identify which sub-score is inflated.")

    print(f"\n  risk_level            : {_safe(bad.get('risk_level'))}")
    print(f"  pump_and_dump_score   : {_safe(bad.get('pump_and_dump_score'))}")
    print(f"  trading_safe          : {_safe(bad.get('trading_safe'))}")
    print(f"  trading_recommendation: {_safe(bad.get('trading_recommendation'))}")

    dd = bad.get("detection_breakdown") or {}
    if dd:
        print(f"\n  detection_breakdown:")
        for k, v in dd.items():
            print(f"    {k:<40} {v}")

# ===========================================================================
#  NODE 10 — Backtesting
# ===========================================================================
_sec("NODE 10 — Backtesting")
br = result.get("backtest_results") or {}
chk_present("Node 10", "backtest_results", br)

if br:
    chk_present("Node 10", "hold_threshold_pct", br.get("hold_threshold_pct"))
    chk_range  ("Node 10", "hold_threshold_pct", br.get("hold_threshold_pct"), 1.0, 10.0,
                "TSLA's hold threshold should be 1-10% (dynamic, based on avg 7-day move)")

    print()
    for stream in ("technical", "stock_news", "market_news", "related_news"):
        sd = br.get(stream)
        chk_present("Node 10", f"backtest_results['{stream}']", sd)
        if sd and isinstance(sd, dict):
            chk_present("Node 10", f"{stream}.full_accuracy",   sd.get("full_accuracy"))
            # recent_accuracy is intentionally None when fewer than MIN_RECENT_DAYS_FOR_ACCURACY
            # (= 5) directional signals fall in the last 60 days — this is by design.
            if sd.get("recent_accuracy") is None:
                _pass("Node 10", f"{stream}.recent_accuracy",
                      "None — insufficient recent signals (< 5 days); expected behaviour")
            else:
                chk_present("Node 10", f"{stream}.recent_accuracy", sd.get("recent_accuracy"))
            chk_present("Node 10", f"{stream}.signal_count",    sd.get("signal_count"))
            chk_present("Node 10", f"{stream}.is_significant",    sd.get("is_significant"))
            chk_present("Node 10", f"{stream}.is_anti_predictive", sd.get("is_anti_predictive"))
            chk_present("Node 10", f"{stream}.p_value",            sd.get("p_value"))
            chk_present("Node 10", f"{stream}.p_value_less",       sd.get("p_value_less"))
            full   = sd.get("full_accuracy")
            recent = sd.get("recent_accuracy")
            print(f"  {stream:<18}  full={_fmt(full,3)}  recent={_fmt(recent,3)}  "
                  f"signals={sd.get('signal_count','N/A')}  "
                  f"sig={sd.get('is_significant','N/A')}  "
                  f"anti={sd.get('is_anti_predictive','N/A')}  "
                  f"p={_fmt(sd.get('p_value'),3)}  p_less={_fmt(sd.get('p_value_less'),3)}")
            if full and float(full) > 0.65:
                _fail("Node 10", f"{stream}.full_accuracy",
                      full, ">65% is suspiciously high — check for data leakage")
            if sd.get("is_anti_predictive"):
                info(f"{stream} is anti-predictive — Node 11 will give it zero weight")

# ===========================================================================
#  NODE 11 — Adaptive Weights
# ===========================================================================
_sec("NODE 11 — Adaptive Weights")
aw = result.get("adaptive_weights") or {}
chk_present("Node 11", "adaptive_weights", aw)

if aw:
    required_weight_keys = [
        "technical_weight", "stock_news_weight",
        "market_news_weight", "related_news_weight"
    ]
    for k in required_weight_keys:
        chk_present("Node 11", k, aw.get(k))
        if aw.get(k) is not None:
            chk_range("Node 11", k, aw.get(k), 0.01, 0.95,
                      "individual weight should be between 1% and 95%")

    total_w = sum(float(aw.get(k, 0)) for k in required_weight_keys if aw.get(k) is not None)
    chk_range("Node 11", "sum_of_four_weights", total_w, 0.97, 1.03,
              "weights must sum to ~1.0")

    print(f"\n  technical_weight    : {_safe(aw.get('technical_weight'))}")
    print(f"  stock_news_weight   : {_safe(aw.get('stock_news_weight'))}")
    print(f"  market_news_weight  : {_safe(aw.get('market_news_weight'))}")
    print(f"  related_news_weight : {_safe(aw.get('related_news_weight'))}")

# ===========================================================================
#  NODE 12 — Final Signal Generation
# ===========================================================================
_sec("NODE 12 — Final Signal Generation  [CORE OUTPUT]")
fs = result.get("final_signal")
fc = result.get("final_confidence")
sc = result.get("signal_components") or {}

chk_present   ("Node 12", "final_signal",      fs)
chk_present   ("Node 12", "final_confidence",  fc)
chk_present   ("Node 12", "signal_components", sc)
chk_one_of    ("Node 12", "final_signal",      fs,
               ["BUY", "SELL", "HOLD"],
               "HOLD or SELL expected — analyst consensus is HOLD, 30% say SELL")
chk_range     ("Node 12", "final_confidence",  fc, 0.001, 0.95,
               "split consensus → moderate-to-low confidence expected")

# Overconfidence traps
if fs == "BUY" and fc and float(fc) > 0.70:
    _fail("Node 12", "overconfident_buy", fc,
          "ignoring 30% SELL analysts and 19% drawdown from peak")
elif fs == "SELL" and fc and float(fc) > 0.80:
    _fail("Node 12", "overconfident_sell", fc,
          "ignoring Cybercab production start and robotaxi expansion news")

if sc:
    chk_present("Node 12", "signal_components.final_score",       sc.get("final_score"))
    chk_present("Node 12", "signal_components.stream_scores",     sc.get("stream_scores"))
    chk_present("Node 12", "signal_components.signal_agreement",  sc.get("signal_agreement"))
    chk_present("Node 12", "signal_components.pattern_prediction",sc.get("pattern_prediction"))
    chk_present("Node 12", "signal_components.price_targets",     sc.get("price_targets"))
    chk_present("Node 12", "signal_components.risk_summary",      sc.get("risk_summary"))
    chk_present("Node 12", "signal_components.backtest_context",  sc.get("backtest_context"))
    chk_present("Node 12", "signal_components.prediction_graph_data", sc.get("prediction_graph_data"))

    ss = sc.get("stream_scores") or {}
    for stream in ("technical", "sentiment", "market", "monte_carlo"):
        d = ss.get(stream) or {}
        chk_present("Node 12", f"stream_scores['{stream}']", d if d else None)
        if d:
            chk_present("Node 12", f"stream_scores.{stream}.raw_score",   d.get("raw_score"))
            chk_present("Node 12", f"stream_scores.{stream}.weight",      d.get("weight"))
            chk_present("Node 12", f"stream_scores.{stream}.contribution",d.get("contribution"))

    missing_streams = sc.get("streams_missing") or []
    if missing_streams:
        _fail("Node 12", "streams_missing", missing_streams,
              "upstream nodes failed to deliver data — signal is less reliable")

    # Pattern prediction sub-fields
    pp2 = sc.get("pattern_prediction") or {}
    if pp2:
        chk_present("Node 12", "pattern_prediction.sufficient_data", pp2.get("sufficient_data"))
        if pp2.get("sufficient_data"):
            chk_present("Node 12", "pattern_prediction.prob_up",           pp2.get("prob_up"))
            chk_present("Node 12", "pattern_prediction.expected_return_7d",pp2.get("expected_return_7d"))
            chk_present("Node 12", "pattern_prediction.agreement_with_job1",pp2.get("agreement_with_job1"))
            chk_present("Node 12", "pattern_prediction.confidence_multiplier",pp2.get("confidence_multiplier"))
            chk_range  ("Node 12", "pattern_prediction.prob_up",
                        pp2.get("prob_up"), 0.25, 0.75,
                        "split consensus → Job 2 should also be near coin-flip")

    # Price targets
    pt = sc.get("price_targets") or {}
    if pt:
        chk_present("Node 12", "price_targets.current_price",    pt.get("current_price"))
        chk_present("Node 12", "price_targets.forecasted_price", pt.get("forecasted_price"))
        fp = pt.get("forecasted_price")
        if fp:
            chk_range("Node 12", "price_targets.forecasted_price", fp, 280, 520,
                      "analyst consensus target $396; <$280 or >$520 is extreme vs consensus")

    # Risk summary
    rs = sc.get("risk_summary") or {}
    chk_present("Node 12", "risk_summary.overall_risk_level",    rs.get("overall_risk_level"))
    chk_present("Node 12", "risk_summary.trading_safe",          rs.get("trading_safe"))
    chk_present("Node 12", "risk_summary.trading_recommendation",rs.get("trading_recommendation"))

    # Print summary
    print(f"\n  final_signal        : {_safe(fs)}")
    print(f"  final_confidence    : {_fmt(fc, 4)}")
    print(f"  final_score         : {_fmt(sc.get('final_score'), 6)}")
    print(f"  signal_agreement    : {_safe(sc.get('signal_agreement'))}/4")
    print(f"  streams_missing     : {sc.get('streams_missing') or 'none'}")
    print(f"\n  Stream contributions:")
    for stream in ("technical", "sentiment", "market", "monte_carlo"):
        d = (ss.get(stream) or {})
        print(f"  {stream:<14}  raw={_fmt(d.get('raw_score'),4)}  "
              f"weight={_fmt(d.get('weight'),4)}  "
              f"contribution={_fmt(d.get('contribution'),4)}")
    print(f"\n  forecasted_price    : {_safe((sc.get('price_targets') or {}).get('forecasted_price'))}")
    print(f"  Benchmark target    : $396.23  (analyst consensus)")
    print(f"  Job1/Job2 agreement : {pp2.get('agreement_with_job1','N/A') if pp2 else 'N/A'}")
    print(f"  conf multiplier     : {_fmt((pp2 or {}).get('confidence_multiplier'), 4)}")

# ===========================================================================
#  NODE 13 — Beginner Explanation
# ===========================================================================
_sec("NODE 13 — Beginner Explanation")
be = result.get("beginner_explanation")
chk_present("Node 13", "beginner_explanation", be)
if be:
    wc = len(be.split())
    chk_range("Node 13", "word_count", wc, 60, 700,
              "<60 words is too thin; >700 is too long for a beginner explanation")
    print(f"  Word count : {wc}")

# ===========================================================================
#  NODE 14 — Technical Explanation
# ===========================================================================
_sec("NODE 14 — Technical Explanation")
te = result.get("technical_explanation")
chk_present("Node 14", "technical_explanation", te)
if te:
    wc = len(te.split())
    chk_range("Node 14", "word_count", wc, 350, 1400,
              "technical report should be ~600-900 words per system prompt")
    print(f"  Word count : {wc}")
    for section in ("Technical Analysis", "Sentiment", "Monte Carlo", "Risk"):
        if section.lower() in te.lower():
            _pass("Node 14", f"section_present: '{section}'")
        else:
            _fail("Node 14", f"section_present: '{section}'",
                  "missing", "expected section not found in report")

# ===========================================================================
#  NODE 15 — Dashboard Data  (NOT YET IMPLEMENTED)
# ===========================================================================
_sec("NODE 15 — Dashboard Data  ⚠️  [NOT YET IMPLEMENTED — SKIPPED]")
info("node_15_dashboard.py does not exist yet.")
info("The workflow currently ends at technical_explanation → END.")
info("dashboard_data will be None until Node 15 is implemented — skipping checks.")
dd15 = result.get("dashboard_data")
if dd15 is None:
    _pass("Node 15", "dashboard_data_absent_as_expected",
          "None — Node 15 not yet in pipeline")
else:
    info("dashboard_data is populated — Node 15 may have been added. Enabling checks.")
    for key in ("ticker", "signal", "confidence", "price_targets",
                "technical_summary", "sentiment_summary", "risk_level"):
        chk_present("Node 15", f"dashboard_data['{key}']", dd15.get(key))

# ===========================================================================
#  SYSTEM — Execution & Pipeline Errors
# ===========================================================================
_sec("SYSTEM — Execution Times & Pipeline Errors")
net = result.get("node_execution_times") or {}
chk_present("System", "node_execution_times", net)

if net:
    print()
    for node, t in sorted(net.items()):
        flag = "  ⚠️ " if t > 60 else "     "
        print(f"{flag}  {node:<40} {t:.3f}s")
    total_t = result.get("total_execution_time")
    print(f"\n  total_execution_time : {_safe(total_t)}")

state_errors = result.get("errors") or []
print(f"\n  Pipeline errors: {len(state_errors)}")
if state_errors:
    for e in state_errors:
        _fail("System", "pipeline_error", e, "check node logs")
else:
    _pass("System", "pipeline_errors", "none reported")

# ===========================================================================
#  FINAL TALLY
# ===========================================================================
print(f"\n{SEP}")
print(f"  FINAL CHECK TALLY")
print(SEP)
print(f"\n  ✓  Passed : {len(PASSES)}")
print(f"  🚩 Failed : {len(FAILURES)}\n")

if FAILURES:
    print(f"  FAILURES TO INVESTIGATE:")
    print(f"  {'#':<4} {'Node':<10} {'Field':<55} {'Actual':<20} Note")
    print(f"  {'─'*4} {'─'*10} {'─'*55} {'─'*20} {'─'*30}")
    for i, (node, field, actual, note) in enumerate(FAILURES, 1):
        a_str = str(actual)[:18]
        print(f"  {i:<4} {node:<10} {field:<55} {a_str:<20} {note}")
else:
    print(f"  ✓  All checks passed — output is consistent with real-world data.")

print(f"""
  REAL-WORLD BENCHMARKS (March 11 2026) — compare against your output:
  ────────────────────────────────────────────────────────────────────
  Current price       ~$396        normalized_score    : 15–58
  Analyst consensus   HOLD         market_regime       : trending_down / choppy
  30% analysts SELL               final_signal        : HOLD or SELL
  Price target        $396.23      final_confidence    : 0.15–0.70
  Down 19% from peak              prob_up (MC)        : 0.30–0.70
  Sector XLY YTD      -3.7%        sector_performance  : negative
  Beta                ~2.5         beta                : 1.2–4.0
  Legitimate company              pump_and_dump_score : < 65
""")

# ===========================================================================
#  FULL LLM OUTPUTS
# ===========================================================================
print(f"\n{SEP}\n  NODE 13 — BEGINNER EXPLANATION\n{SEP}")
print(result.get("beginner_explanation") or "(not generated)")

print(f"\n{SEP}\n  NODE 14 — TECHNICAL EXPLANATION\n{SEP}")
print(result.get("technical_explanation") or "(not generated)")

print(f"\n{SEP}\n")