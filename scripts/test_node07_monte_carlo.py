"""
Node 7 — Monte Carlo Forecasting: Isolated Test

Calls monte_carlo_forecasting_node() directly with real yfinance price data
and a synthetic market_context, then prints all outputs with real-world
comparison benchmarks.

Scenarios tested
----------------
  IONQ  — volatile quantum-computing stock, current price ~$27–35
  AAPL  — blue-chip baseline, low volatility
  TSLA  — high-beta, high-vol stock (~2.5 beta)

Usage:
    python scripts/test_node07_monte_carlo.py
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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")

import yfinance as yf
import pandas as pd
import numpy as np

from src.langgraph_nodes.node_07_monte_carlo import (
    monte_carlo_forecasting_node,
    calculate_historical_statistics,
    calculate_regime_adjusted_volatility,
)

SEP  = "=" * 72
DASH = "─" * 60

PASSES   = []
FAILURES = []

# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def _pass(label, field, note=""):
    PASSES.append((label, field))
    tag = f"  ✓  {label:<12} {field}"
    if note:
        tag += f"  →  {note}"
    print(tag)

def _fail(label, field, actual, note=""):
    FAILURES.append((label, field, actual, note))
    tag = f"  🚩 {label:<12} {field} = {actual!r}"
    if note:
        tag += f"  →  {note}"
    print(tag)

def _fmt(v, d=4, default="N/A"):
    if v is None:
        return default
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)

def chk_present(label, field, value):
    empty = (value is None
             or (isinstance(value, dict)  and len(value) == 0)
             or (isinstance(value, list)  and len(value) == 0)
             or (isinstance(value, str)   and value.strip() == ""))
    if empty:
        _fail(label, field, value, "MISSING")
        return False
    _pass(label, field)
    return True

def chk_range(label, field, value, lo, hi, note=""):
    if value is None:
        _fail(label, field, None, f"None — expected [{lo}, {hi}]")
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        _fail(label, field, value, f"not numeric")
        return False
    if lo <= v <= hi:
        _pass(label, field, f"{v:.4g}  [OK: {lo}–{hi}]" + (f"  {note}" if note else ""))
        return True
    _fail(label, field, v, f"outside [{lo}, {hi}]" + (f"  {note}" if note else ""))
    return False

def _sec(title):
    print(f"\n{DASH}\n  {title}\n{DASH}")

# ---------------------------------------------------------------------------
# Market context presets  (mimics Node 6 output structure)
# ---------------------------------------------------------------------------

MARKET_CONTEXTS = {
    "neutral": {
        "market_regime": {
            "vix_level": 20.0,
            "regime_label": "NEUTRAL",
        },
        "market_correlation_profile": {
            "beta_calculated": 1.0,
        },
    },
    "elevated_vix_high_beta": {
        "market_regime": {
            "vix_level": 30.0,
            "regime_label": "RISK_OFF_BEAR",
        },
        "market_correlation_profile": {
            "beta_calculated": 2.5,   # TSLA-like
        },
    },
    "calm_low_beta": {
        "market_regime": {
            "vix_level": 13.0,
            "regime_label": "LOW_VOL_BULL",
        },
        "market_correlation_profile": {
            "beta_calculated": 0.8,   # AAPL-like
        },
    },
    "panic": {
        "market_regime": {
            "vix_level": 50.0,
            "regime_label": "PANIC_SELLOFF",
        },
        "market_correlation_profile": {
            "beta_calculated": 3.0,
        },
    },
}

# ---------------------------------------------------------------------------
# Fetch price data
# ---------------------------------------------------------------------------

def fetch_price_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    print(f"  Fetching {period} of price data for {ticker} via yfinance...")
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    df = raw.copy()
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df = df[["open","high","low","close","volume"]].dropna()
    print(f"  Got {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")
    return df

# ---------------------------------------------------------------------------
# Run a single ticker scenario
# ---------------------------------------------------------------------------

def run_scenario(ticker: str, context_name: str, real_world: dict):
    label = f"{ticker}/{context_name}"
    _sec(f"SCENARIO  {label}")

    # --- fetch ---
    try:
        price_data = fetch_price_data(ticker)
    except Exception as e:
        _fail(label, "price_data", None, str(e))
        return

    # --- build state ---
    state = {
        "ticker": ticker,
        "raw_price_data": price_data,
        "market_context": MARKET_CONTEXTS[context_name],
        "errors": [],
        "node_execution_times": {},
    }

    # --- call node ---
    result = monte_carlo_forecasting_node(state)

    # --- check errors ---
    if result.get("errors"):
        for e in result["errors"]:
            _fail(label, "node_error", e, "node raised an error")
        return

    mcr = result.get("monte_carlo_results") or {}
    fp  = result.get("forecasted_price")
    pr  = result.get("price_range")
    t   = (result.get("node_execution_times") or {}).get("node_7")

    # ── presence checks ──────────────────────────────────────────────────────
    print()
    chk_present(label, "monte_carlo_results", mcr)
    chk_present(label, "forecasted_price",    fp)
    chk_present(label, "price_range",         pr)

    if not mcr:
        return

    for key in ("current_price","mean_forecast","median_forecast","std_dev",
                "confidence_68","confidence_95","probability_up","probability_down",
                "expected_return","min_price","max_price","drift","volatility",
                "base_volatility","vix_level","regime_label","volatility_scalar",
                "forecast_days","num_simulations"):
        chk_present(label, key, mcr.get(key))

    ci68 = mcr.get("confidence_68") or {}
    ci95 = mcr.get("confidence_95") or {}
    chk_present(label, "confidence_68.lower", ci68.get("lower"))
    chk_present(label, "confidence_68.upper", ci68.get("upper"))
    chk_present(label, "confidence_95.lower", ci95.get("lower"))
    chk_present(label, "confidence_95.upper", ci95.get("upper"))

    # ── range / sanity checks ─────────────────────────────────────────────────
    print()
    cur = float(mcr["current_price"])
    lo_price = cur * 0.30
    hi_price = cur * 4.00

    chk_range(label, "current_price",    cur,                        lo_price, hi_price, "sanity: same order of magnitude as market price")
    chk_range(label, "mean_forecast",    mcr.get("mean_forecast"),   lo_price, hi_price, "mean forecast in plausible range")
    chk_range(label, "probability_up",   mcr.get("probability_up"),  0.10, 0.90,         "GBM should never be near 0 or 1 over 7 days")
    chk_range(label, "probability_down", mcr.get("probability_down"),0.10, 0.90)
    chk_range(label, "expected_return",  mcr.get("expected_return"), -25.0, 25.0,        "7-day expected return; >±25% is extreme")
    chk_range(label, "volatility",       mcr.get("volatility"),      0.005, 0.20,        "daily vol; base=0.01-0.09, regime-adjusted (scalar up to 3×)")
    chk_range(label, "base_volatility",  mcr.get("base_volatility"), 0.005, 0.10,        "raw 30-day historical daily vol")
    chk_range(label, "volatility_scalar",mcr.get("volatility_scalar"),0.50, 3.0,         "VIX/beta scalar; clamped to [0.5, 3.0]")
    chk_range(label, "num_simulations",  mcr.get("num_simulations"), 100, 100_000)
    chk_range(label, "forecast_days",    mcr.get("forecast_days"),   5, 30,              "should be 7")

    # CI ordering
    if ci95.get("lower") and ci95.get("upper"):
        if ci95["lower"] < ci95["upper"]:
            _pass(label, "ci95_ordering", f"{ci95['lower']:.2f} < {ci95['upper']:.2f}")
        else:
            _fail(label, "ci95_ordering", ci95, "lower ≥ upper")
    if ci68.get("lower") and ci68.get("upper"):
        if ci68["lower"] < ci68["upper"]:
            _pass(label, "ci68_ordering", f"{ci68['lower']:.2f} < {ci68['upper']:.2f}")
        else:
            _fail(label, "ci68_ordering", ci68, "lower ≥ upper")
    # 95% CI should be wider than 68% CI
    if all(v is not None for v in [ci68.get("lower"), ci68.get("upper"),
                                    ci95.get("lower"), ci95.get("upper")]):
        width68 = ci68["upper"] - ci68["lower"]
        width95 = ci95["upper"] - ci95["lower"]
        if width95 > width68:
            _pass(label, "ci95_wider_than_ci68", f"95% width={width95:.2f} > 68% width={width68:.2f}")
        else:
            _fail(label, "ci95_wider_than_ci68", f"95%={width95:.2f} 68%={width68:.2f}", "95% CI must be wider than 68% CI")

    # prob_up + prob_down ≈ 1.0
    pu = mcr.get("probability_up")
    pd_ = mcr.get("probability_down")
    if pu is not None and pd_ is not None:
        total = float(pu) + float(pd_)
        if abs(total - 1.0) < 0.01:
            _pass(label, "prob_sums_to_1", f"{total:.4f}")
        else:
            _fail(label, "prob_sums_to_1", total, "prob_up + prob_down should be 1.0")

    # VIX / regime carried through correctly
    ctx = MARKET_CONTEXTS[context_name]
    expected_vix   = ctx["market_regime"]["vix_level"]
    expected_regime = ctx["market_regime"]["regime_label"]
    if abs(float(mcr.get("vix_level", 0)) - expected_vix) < 0.01:
        _pass(label, "vix_level_passthrough", f"{mcr['vix_level']} == {expected_vix}")
    else:
        _fail(label, "vix_level_passthrough", mcr.get("vix_level"), f"expected {expected_vix}")
    if mcr.get("regime_label") == expected_regime:
        _pass(label, "regime_label_passthrough", expected_regime)
    else:
        _fail(label, "regime_label_passthrough", mcr.get("regime_label"), f"expected {expected_regime}")

    # ── real-world comparison ─────────────────────────────────────────────────
    print(f"\n  ── Real-world comparison ({ticker}) ──")
    rw_price   = real_world.get("price")
    rw_vol_lo  = real_world.get("daily_vol_lo")
    rw_vol_hi  = real_world.get("daily_vol_hi")
    rw_beta    = MARKET_CONTEXTS[context_name]["market_correlation_profile"]["beta_calculated"]

    if rw_price:
        pct_diff = abs(cur - rw_price) / rw_price * 100
        note = f"  current_price={cur:.2f}  real_world≈{rw_price}  diff={pct_diff:.1f}%"
        if pct_diff < 30:
            _pass(label, "current_price_vs_real_world", note)
        else:
            _fail(label, "current_price_vs_real_world", cur, note)

    if rw_vol_lo and rw_vol_hi:
        bv = float(mcr.get("base_volatility", 0))
        if rw_vol_lo <= bv <= rw_vol_hi:
            _pass(label, "base_vol_in_expected_range", f"{bv:.4f} in [{rw_vol_lo}, {rw_vol_hi}]")
        else:
            _fail(label, "base_vol_in_expected_range", bv, f"expected [{rw_vol_lo}, {rw_vol_hi}]")

    # Higher VIX / higher beta → wider vol scalar
    scalar = float(mcr.get("volatility_scalar", 1.0))
    expected_scalar = calculate_regime_adjusted_volatility(
        base_volatility=1.0,      # normalised
        vix_level=expected_vix,
        regime_label=expected_regime,
        beta=rw_beta,
    )
    if abs(scalar - expected_scalar) < 0.02:
        _pass(label, "volatility_scalar_formula", f"{scalar:.4f} ≈ {expected_scalar:.4f}")
    else:
        _fail(label, "volatility_scalar_formula", scalar, f"formula says {expected_scalar:.4f}")

    # ── print full summary ────────────────────────────────────────────────────
    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  {ticker:<6}  {context_name:<30}                    │
  ├─────────────────────────────────────────────────────┤
  │  Current price       : ${cur:>10.2f}                │
  │  Mean forecast (7d)  : ${float(mcr.get('mean_forecast',0)):>10.2f}               │
  │  Median forecast     : ${float(mcr.get('median_forecast',0)):>10.2f}               │
  │  Expected return     : {_fmt(mcr.get('expected_return'),3):>10}%              │
  │  Probability up      : {float(mcr.get('probability_up',0))*100:>9.1f}%               │
  │  Probability down    : {float(mcr.get('probability_down',0))*100:>9.1f}%               │
  ├─────────────────────────────────────────────────────┤
  │  68% CI  : ${float(ci68.get('lower',0)):>8.2f}  →  ${float(ci68.get('upper',0)):>8.2f}        │
  │  95% CI  : ${float(ci95.get('lower',0)):>8.2f}  →  ${float(ci95.get('upper',0)):>8.2f}        │
  │  Min sim : ${float(mcr.get('min_price',0)):>8.2f}    Max sim : ${float(mcr.get('max_price',0)):>8.2f}  │
  ├─────────────────────────────────────────────────────┤
  │  Base volatility     : {_fmt(mcr.get('base_volatility'),5):>10}  (daily, raw)     │
  │  Regime-adj vol      : {_fmt(mcr.get('volatility'),5):>10}  (used for GBM)   │
  │  Volatility scalar   : {_fmt(mcr.get('volatility_scalar'),4):>10}               │
  │  VIX                 : {_fmt(mcr.get('vix_level'),1):>10}               │
  │  Regime              : {str(mcr.get('regime_label','?')):<16}               │
  │  Drift (daily)       : {_fmt(mcr.get('drift'),6):>10}               │
  │  Simulations         : {mcr.get('num_simulations',0):>10}               │
  │  Elapsed             : {f'{t:.2f}s' if t else 'N/A':>10}               │
  └─────────────────────────────────────────────────────┘""")


# ---------------------------------------------------------------------------
# Volatility-scaling unit tests
# ---------------------------------------------------------------------------

def run_volatility_scaling_tests():
    _sec("UNIT TESTS — VIX/Beta Volatility Scaling")
    label = "VIX-scale"
    base = 0.02

    # expected_scalar: what (output / base) should be
    cases = [
        # (vix, regime,           beta, expected_scalar, desc)
        (20.0, "NEUTRAL",         1.0,  1.00,       "neutral VIX, beta=1 → no change"),
        (30.0, "NEUTRAL",         1.0,  1.2247,     "VIX=30 → +22% (sqrt(30/20))"),
        (13.0, "LOW_VOL_BULL",    1.0,  0.8062,     "VIX=13 → sqrt(13/20)=0.806"),
        (30.0, "NEUTRAL",         2.5,  1.2247*1.75,"VIX=30, beta=2.5 → regime×beta scalar"),
        (50.0, "PANIC_SELLOFF",   3.0,  3.0,        "panic + high beta → clamped at 3×"),
        (20.0, "NEUTRAL",         0.5,  0.75,       "neutral VIX, low-beta → narrower"),
    ]

    for vix, regime, beta, expected_scalar, desc in cases:
        adjusted = calculate_regime_adjusted_volatility(base, vix, regime, beta)
        actual_scalar = adjusted / base
        expected_abs  = base * expected_scalar   # absolute expected output

        if regime == "PANIC_SELLOFF":
            # clamped at 3× base
            expected_clamped = base * 3.0
            if abs(adjusted - expected_clamped) < 0.001:
                _pass(label, f"[{regime} vix={vix} β={beta}]",
                      f"output={adjusted:.4f} == {expected_clamped:.4f}  {desc}")
            else:
                _fail(label, f"[{regime} vix={vix} β={beta}]", adjusted,
                      f"expected clamped at {expected_clamped:.4f}  {desc}")
        else:
            tol = 0.005
            if abs(actual_scalar - expected_scalar) <= tol + abs(expected_scalar) * 0.02:
                _pass(label, f"[{regime} vix={vix} β={beta}]",
                      f"scalar={actual_scalar:.4f} ≈ {expected_scalar:.4f}  {desc}")
            else:
                _fail(label, f"[{regime} vix={vix} β={beta}]", actual_scalar,
                      f"expected scalar≈{expected_scalar:.4f}  {desc}")

    # Monotonicity: higher VIX → higher scalar (same regime, same beta)
    s20 = calculate_regime_adjusted_volatility(base, 20.0, "NEUTRAL", 1.0)
    s30 = calculate_regime_adjusted_volatility(base, 30.0, "NEUTRAL", 1.0)
    s40 = calculate_regime_adjusted_volatility(base, 40.0, "NEUTRAL", 1.0)
    if s20 < s30 < s40:
        _pass(label, "monotonic_vix_scaling", f"s20={s20:.4f} < s30={s30:.4f} < s40={s40:.4f}")
    else:
        _fail(label, "monotonic_vix_scaling", (s20, s30, s40), "expected strictly increasing with VIX")

    # Monotonicity: higher beta → higher scalar (same VIX, same regime)
    s1 = calculate_regime_adjusted_volatility(base, 25.0, "NEUTRAL", 1.0)
    s2 = calculate_regime_adjusted_volatility(base, 25.0, "NEUTRAL", 2.0)
    s3 = calculate_regime_adjusted_volatility(base, 25.0, "NEUTRAL", 3.0)
    if s1 < s2 < s3:
        _pass(label, "monotonic_beta_scaling", f"β1={s1:.4f} < β2={s2:.4f} < β3={s3:.4f}")
    else:
        _fail(label, "monotonic_beta_scaling", (s1, s2, s3), "expected strictly increasing with beta")

    # Clamp: excessively high VIX + beta shouldn't exceed 3× base
    s_max = calculate_regime_adjusted_volatility(base, 200.0, "NEUTRAL", 10.0)
    if s_max <= base * 3.0 + 1e-9:
        _pass(label, "upper_clamp_at_3x", f"{s_max:.4f} ≤ {base*3.0:.4f}")
    else:
        _fail(label, "upper_clamp_at_3x", s_max, f"exceeded 3× base={base*3.0:.4f}")

    # Clamp: VIX=1, low beta shouldn't go below 0.5× base
    s_min = calculate_regime_adjusted_volatility(base, 1.0, "NEUTRAL", 0.1)
    if s_min >= base * 0.5 - 1e-9:
        _pass(label, "lower_clamp_at_0.5x", f"{s_min:.4f} ≥ {base*0.5:.4f}")
    else:
        _fail(label, "lower_clamp_at_0.5x", s_min, f"below 0.5× base={base*0.5:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TICKERS = [
    {
        "ticker": "IONQ",
        "context": "elevated_vix_high_beta",
        "real_world": {
            "price": 30.0,          # rough current price Mar 2026
            "daily_vol_lo": 0.025,  # IONQ is extremely volatile
            "daily_vol_hi": 0.09,
        },
    },
    {
        "ticker": "TSLA",
        "context": "elevated_vix_high_beta",
        "real_world": {
            "price": 395.0,         # current price Mar 2026 (~$398)
            "daily_vol_lo": 0.015,
            "daily_vol_hi": 0.055,
        },
    },
    {
        "ticker": "AAPL",
        "context": "calm_low_beta",
        "real_world": {
            "price": 220.0,
            "daily_vol_lo": 0.008,
            "daily_vol_hi": 0.025,
        },
    },
]

if __name__ == "__main__":
    print(SEP)
    print("  NODE 7 — Monte Carlo Forecasting: Isolated Test")
    print("  Tests the node directly with real price data and synthetic market context")
    print(SEP)

    # Unit tests — no network required
    run_volatility_scaling_tests()

    # Integration scenarios — requires yfinance
    for cfg in TICKERS:
        run_scenario(cfg["ticker"], cfg["context"], cfg["real_world"])

    # Additional: same ticker, different context — checks volatility scalar changes
    _sec("CROSS-CONTEXT COMPARISON  TSLA  (neutral vs elevated vs panic)")
    try:
        price_data = fetch_price_data("TSLA")
        results = {}
        for ctx_name in ("neutral", "elevated_vix_high_beta", "panic"):
            state = {
                "ticker": "TSLA",
                "raw_price_data": price_data,
                "market_context": MARKET_CONTEXTS[ctx_name],
                "errors": [],
                "node_execution_times": {},
            }
            r = monte_carlo_forecasting_node(state)
            mcr = r.get("monte_carlo_results") or {}
            results[ctx_name] = mcr

        print(f"\n  {'Context':<30} {'base_vol':>10}  {'adj_vol':>10}  {'scalar':>8}  "
              f"{'95%_width':>10}  {'prob_up':>8}")
        print(f"  {'─'*30} {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*8}")
        for ctx_name, mcr in results.items():
            ci95 = mcr.get("confidence_95") or {}
            width = (ci95.get("upper", 0) or 0) - (ci95.get("lower", 0) or 0)
            print(f"  {ctx_name:<30} "
                  f"{_fmt(mcr.get('base_volatility'),5):>10}  "
                  f"{_fmt(mcr.get('volatility'),5):>10}  "
                  f"{_fmt(mcr.get('volatility_scalar'),4):>8}  "
                  f"{width:>10.2f}  "
                  f"{float(mcr.get('probability_up',0))*100:>7.1f}%")

        # monotonicity check: panic > elevated > neutral (CI width)
        widths = {}
        for ctx_name, mcr in results.items():
            ci95 = mcr.get("confidence_95") or {}
            widths[ctx_name] = (ci95.get("upper", 0) or 0) - (ci95.get("lower", 0) or 0)
        if widths.get("panic", 0) > widths.get("elevated_vix_high_beta", 0) > widths.get("neutral", 0):
            _pass("cross-ctx", "ci95_width_monotonic",
                  f"panic({widths['panic']:.2f}) > elevated({widths['elevated_vix_high_beta']:.2f}) > neutral({widths['neutral']:.2f})")
        else:
            _fail("cross-ctx", "ci95_width_monotonic", widths,
                  "expected CI to widen as VIX/beta increase")

    except Exception as e:
        _fail("cross-ctx", "comparison_run", None, str(e))

    # ── Final tally ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  FINAL CHECK TALLY")
    print(SEP)
    print(f"\n  ✓  Passed : {len(PASSES)}")
    print(f"  🚩 Failed : {len(FAILURES)}\n")

    if FAILURES:
        print(f"  FAILURES:")
        print(f"  {'#':<4} {'Label':<18} {'Field':<45} {'Actual':<20} Note")
        print(f"  {'─'*4} {'─'*18} {'─'*45} {'─'*20} {'─'*30}")
        for i, (label, field, actual, note) in enumerate(FAILURES, 1):
            print(f"  {i:<4} {label:<18} {field:<45} {str(actual)[:18]:<20} {note}")
    else:
        print("  ✓  All checks passed.")

    print(f"""
  REAL-WORLD BENCHMARKS (Mar 2026):
  ─────────────────────────────────────────────────────────────
  IONQ  price ≈ $30        daily vol ≈ 0.03–0.07
  TSLA  price ≈ $398       daily vol ≈ 0.02–0.05    beta ≈ 2.5
  AAPL  price ≈ $220       daily vol ≈ 0.01–0.02    beta ≈ 0.8
  ─────────────────────────────────────────────────────────────
  VIX ≈ 20 neutral → scalar ≈ 1.0
  VIX ≈ 30 + beta 2.5     → scalar ≈ 2.1
  VIX ≈ 50 + PANIC_SELLOFF → scalar clamped at 3.0
""")
