"""
streamlit_app.py — Trading Signal Dashboard

Connects to the LangGraph pipeline, triggers analysis for a given ticker,
and displays signal, charts, and LLM explanations.

Charts:
  1. Stock price history (1D / 7D / 1M / 3M / 6M) — live via yfinance
  2. Monte Carlo simulation paths — from Node 7 visualization_data
  3. 7-day price forecast — GBM cone + empirical range + blended line
  4. Historical similar days outcome distribution — from Job 2

Dependencies:
  pip install streamlit plotly yfinance pandas

Run:
  streamlit run streamlit_app.py

Fixes applied vs previous version:
  1. TypeError "multiple values for keyword argument 'xaxis'" — CHART_LAYOUT
     has been split into _CHART_BASE (no axis keys) + _XAXIS_DEFAULT /
     _YAXIS_DEFAULT. The _layout() helper merges them cleanly so no chart
     function ever passes xaxis/yaxis both via **spread and as explicit kwarg.
  2. Deprecated use_container_width=True replaced with width='stretch'
     on every st.plotly_chart() call.
  3. Duplicate import of run_stock_analysis_async removed (was imported
     twice — once at module top level and once inside the try/except block).
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ─── Pipeline import ──────────────────────────────────────────────────────────

try:
    from src.graph.workflow import run_stock_analysis_async
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Signal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

section[data-testid="stSidebar"] { background: #0a0a0a; border-right: 1px solid #1e1e1e; }
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: #141414 !important; border: 1px solid #2a2a2a !important;
    border-radius: 6px !important; color: #e0e0e0 !important;
    font-family: 'DM Mono', monospace !important; font-size: 15px !important;
    letter-spacing: .05em !important; text-transform: uppercase !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: #e0e0e0 !important; color: #0a0a0a !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 13px !important; letter-spacing: .04em !important;
    width: 100% !important; padding: 10px !important;
}
section[data-testid="stSidebar"] .stButton button:hover { background: #ffffff !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 12px !important; }

[data-testid="metric-container"] {
    background: #ffffff; border: 1px solid #ebebeb;
    border-radius: 10px; padding: 14px 18px;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 11px !important; font-weight: 500 !important;
    text-transform: uppercase !important; letter-spacing: .06em !important;
    color: #888 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 22px !important; color: #0a0a0a !important;
}

.stTabs [data-baseweb="tab-list"] { gap:2px; background:#f5f5f5; border-radius:8px; padding:3px; }
.stTabs [data-baseweb="tab"] { border-radius:6px !important; font-size:12px !important; font-weight:500 !important; padding:5px 16px !important; color:#888 !important; }
.stTabs [aria-selected="true"] { background:#ffffff !important; color:#0a0a0a !important; box-shadow:0 1px 3px rgba(0,0,0,.08) !important; }

.section-header { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:#999; margin:1.5rem 0 .75rem; padding-bottom:.4rem; border-bottom:1px solid #ebebeb; }

.signal-buy  { display:inline-block; background:#ecfdf5; color:#065f46; font-size:13px; font-weight:600; padding:4px 14px; border-radius:999px; letter-spacing:.02em; }
.signal-sell { display:inline-block; background:#fef2f2; color:#991b1b; font-size:13px; font-weight:600; padding:4px 14px; border-radius:999px; letter-spacing:.02em; }
.signal-hold { display:inline-block; background:#fffbeb; color:#92400e; font-size:13px; font-weight:600; padding:4px 14px; border-radius:999px; letter-spacing:.02em; }

.explanation-body { font-size:14px; line-height:1.8; color:#e8e8e8; font-family:'DM Sans',sans-serif; }
.explanation-body h1,.explanation-body h2,.explanation-body h3 { font-size:14px; font-weight:600; color:#ffffff; margin:1rem 0 .3rem; text-transform:uppercase; letter-spacing:.04em; }
.explanation-body strong { color:#ffffff; }
.explanation-body em { color:#c4c4c4; }

.sim-table { width:100%; border-collapse:collapse; font-size:12px; }
.sim-table th { text-align:left; padding:5px 8px; color:#999; font-weight:500; border-bottom:1px solid #ebebeb; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
.sim-table td { padding:5px 8px; color:#333; border-bottom:1px solid #f5f5f5; font-family:'DM Mono',monospace; }
.sim-table tr:last-child td { border-bottom:none; }
.up { color:#065f46; } .dn { color:#991b1b; }

.risk-low    { color:#065f46; background:#ecfdf5; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }
.risk-medium { color:#92400e; background:#fffbeb; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }
.risk-high   { color:#991b1b; background:#fef2f2; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }

.prior-badge { color:#92400e; background:#fffbeb; padding:1px 6px; border-radius:4px; font-size:10px; font-weight:500; margin-left:4px; }
.live-badge  { color:#065f46; background:#ecfdf5; padding:1px 6px; border-radius:4px; font-size:10px; font-weight:500; margin-left:4px; }

.chart-title    { font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:#aaa; margin-bottom:2px; }
.chart-subtitle { font-size:13px; color:#c8c8c8; margin-bottom:12px; }

.disclaimer { font-size:11px; color:#bbb; margin-top:1rem; padding:.6rem .9rem; background:#f9f9f9; border-radius:6px; border-left:3px solid #e0e0e0; line-height:1.6; }

.building-badge { display:inline-block; font-size:11px; color:#92400e; background:#fffbeb; padding:3px 10px; border-radius:4px; border:1px solid #fde68a; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def signal_pill(signal: str) -> str:
    cls = {"BUY": "signal-buy", "SELL": "signal-sell"}.get(signal, "signal-hold")
    return f'<span class="{cls}">{signal}</span>'


def risk_badge(level: str) -> str:
    cls = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high",
           "CRITICAL": "risk-high"}.get(level.upper(), "risk-medium")
    return f'<span class="{cls}">{level}</span>'


def fmt_pct(v: Optional[float], plus: bool = True) -> str:
    if v is None:
        return "N/A"
    sign = "+" if plus and v > 0 else ""
    return f"{sign}{v:.1f}%"


def fmt_price(v: Optional[float]) -> str:
    return f"${v:.2f}" if v is not None else "N/A"


@st.cache_data(ttl=300)
def fetch_price_history(ticker: str, period: str) -> pd.DataFrame:
    # Use 2d for "1D" so the chart always has at least one session even on
    # weekends or after-hours; we then keep only the most recent trading day.
    period_map = {
        "1D": ("2d", "5m"), "7D": ("7d", "30m"),
        "1M": ("1mo", "1d"), "3M": ("3mo", "1d"), "6M": ("6mo", "1d"),
    }
    yf_period, interval = period_map.get(period, ("1mo", "1d"))
    df = yf.download(ticker, period=yf_period, interval=interval,
                     progress=False, auto_adjust=True, multi_level_index=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # Flatten MultiIndex columns if present
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Normalise the datetime column name — yfinance uses "Datetime" for
    # intraday and "Date" for daily; fall back to the first datetime-like column.
    if "Datetime" not in df.columns and "Date" not in df.columns:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df = df.rename(columns={col: "Datetime"})
                break
    # For "1D" trim to the most recent trading day only
    if period == "1D" and "Datetime" in df.columns:
        last_date = df["Datetime"].dt.date.max()
        df = df[df["Datetime"].dt.date == last_date].reset_index(drop=True)
    return df


# ─── Chart layout helper ──────────────────────────────────────────────────────
# FIX 1: Split the old monolithic CHART_LAYOUT dict into a base dict (no axis
# keys) and separate axis defaults. The _layout() helper merges them so that
# charts which need custom axis config never produce a duplicate-keyword error.

_CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#555", size=11),
    margin=dict(l=8, r=8, t=8, b=8),
    legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)",
                orientation="h", y=1.08),
    hovermode="x unified",
)

_XAXIS_DEFAULT = dict(showgrid=False, zeroline=False, showline=False,
                      tickfont=dict(size=10, color="#aaa"))
_YAXIS_DEFAULT = dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False,
                      showline=False, tickfont=dict(size=10, color="#aaa"))


def _layout(height: int,
            xaxis: Optional[dict] = None,
            yaxis: Optional[dict] = None,
            **kwargs) -> dict:
    """Return a complete Plotly layout dict.  Axis overrides are merged on top
    of defaults so callers never touch both **_CHART_BASE and xaxis= directly."""
    return {
        **_CHART_BASE,
        "height": height,
        "xaxis": {**_XAXIS_DEFAULT, **(xaxis or {})},
        "yaxis": {**_YAXIS_DEFAULT, **(yaxis or {})},
        **kwargs,
    }


# ─── Chart builders ──────────────────────────────────────────────────────────

def build_stock_chart(df: pd.DataFrame, ticker: str, period: str) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(color="#aaa", size=12))
        fig.update_layout(**_layout(280))
        return fig

    date_col = "Datetime" if "Datetime" in df.columns else "Date"

    if period == "1D":
        fig = go.Figure(go.Scatter(
            x=df[date_col], y=df["Close"], mode="lines",
            line=dict(color="#0a0a0a", width=1.5),
            fill="tozeroy", fillcolor="rgba(10,10,10,0.04)",
            hovertemplate="%{y:.2f}<extra></extra>",
        ))
    else:
        fig = go.Figure(go.Candlestick(
            x=df[date_col],
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            increasing=dict(line=dict(color="#16a34a", width=1), fillcolor="#22c55e"),
            decreasing=dict(line=dict(color="#dc2626", width=1), fillcolor="#ef4444"),
            hovertext=ticker,
        ))
        fig.update_layout(xaxis_rangeslider_visible=False)

    fig.update_layout(**_layout(280))
    return fig


def build_monte_carlo_chart(mc_results: Dict) -> go.Figure:
    viz          = mc_results.get("visualization_data") or {}
    sample_paths = viz.get("sample_paths") or []
    mean_path    = viz.get("mean_path") or []
    upper_95     = viz.get("upper_95") or []
    lower_95     = viz.get("lower_95") or []
    upper_68     = viz.get("upper_68") or []
    lower_68     = viz.get("lower_68") or []
    days         = viz.get("days") or list(range(8))

    fig = go.Figure()

    for path in (sample_paths[:80] if len(sample_paths) >= 80 else sample_paths):
        fig.add_trace(go.Scatter(
            x=days, y=path, mode="lines",
            line=dict(color="rgba(10,10,10,0.04)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    if upper_95 and lower_95:
        fig.add_trace(go.Scatter(
            x=days + days[::-1],
            y=list(upper_95) + list(reversed(lower_95)),
            fill="toself", fillcolor="rgba(6,95,70,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI", hoverinfo="skip",
        ))

    if upper_68 and lower_68:
        fig.add_trace(go.Scatter(
            x=days + days[::-1],
            y=list(upper_68) + list(reversed(lower_68)),
            fill="toself", fillcolor="rgba(6,95,70,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="68% CI", hoverinfo="skip",
        ))

    if mean_path:
        fig.add_trace(go.Scatter(
            x=days, y=mean_path, mode="lines",
            line=dict(color="#065f46", width=2),
            name="Mean path",
            hovertemplate="Day %{x}: $%{y:.2f}<extra></extra>",
        ))

    fig.update_layout(**_layout(
        260,
        xaxis=dict(title="Trading day"),
        yaxis=dict(title="Price ($)", tickprefix="$"),
    ))
    return fig


def build_forecast_chart(sc: Dict, mc: Dict) -> go.Figure:
    pg            = sc.get("prediction_graph_data") or {}
    current_price = float(mc.get("current_price") or 0)
    days          = list(range(8))

    def pct_to_price(pct):
        return current_price * (1 + pct / 100) if current_price else None

    gbm_lower   = pct_to_price(pg.get("gbm_spread_lower") or -5)
    gbm_upper   = pct_to_price(pg.get("gbm_spread_upper") or  5)
    emp_lower   = (pct_to_price(pg.get("empirical_lower") or -4)
                   if pg.get("empirical_lower") is not None else None)
    emp_upper   = (pct_to_price(pg.get("empirical_upper") or  4)
                   if pg.get("empirical_upper") is not None else None)
    blended_ret = pg.get("blended_expected_return") or 0
    gbm_ret     = pg.get("gbm_expected_return") or 0

    fig = go.Figure()

    if gbm_upper and gbm_lower:
        gbm_u = [current_price + (gbm_upper - current_price) * d / 7 for d in days]
        gbm_l = [current_price + (gbm_lower - current_price) * d / 7 for d in days]
        fig.add_trace(go.Scatter(
            x=days + days[::-1], y=gbm_u + list(reversed(gbm_l)),
            fill="toself", fillcolor="rgba(99,153,34,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="GBM 95% cone", hoverinfo="skip",
        ))

    if emp_upper is not None and emp_lower is not None:
        emp_u = [current_price + (emp_upper - current_price) * d / 7 for d in days]
        emp_l = [current_price + (emp_lower - current_price) * d / 7 for d in days]
        fig.add_trace(go.Scatter(
            x=days + days[::-1], y=emp_u + list(reversed(emp_l)),
            fill="toself", fillcolor="rgba(6,95,70,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Empirical range", hoverinfo="skip",
        ))

    gbm_end     = current_price * (1 + gbm_ret     / 100)
    blended_end = current_price * (1 + blended_ret / 100)

    fig.add_trace(go.Scatter(
        x=[0, 7], y=[current_price, gbm_end], mode="lines",
        line=dict(color="#aaa", width=1.5, dash="dot"),
        name="GBM central", hovertemplate="$%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 7], y=[current_price, blended_end],
        mode="lines+markers",
        line=dict(color="#065f46", width=2.5),
        marker=dict(size=[8, 8], color=["#0a0a0a", "#065f46"]),
        name="Blended forecast", hovertemplate="$%{y:.2f}<extra></extra>",
    ))

    fig.add_annotation(x=7, y=blended_end, text=f"${blended_end:.2f}",
                       showarrow=False, xanchor="left", xshift=6,
                       font=dict(size=11, color="#065f46", family="DM Mono"))
    fig.add_annotation(x=0, y=current_price, text=f"${current_price:.2f}",
                       showarrow=False, xanchor="right", xshift=-6,
                       font=dict(size=11, color="#0a0a0a", family="DM Mono"))

    fig.update_layout(**_layout(
        260,
        xaxis=dict(
            tickvals=list(range(8)),
            ticktext=["Today", "1", "2", "3", "4", "5", "6", "7d"],
        ),
        yaxis=dict(tickprefix="$"),
    ))
    return fig


def build_similar_days_chart(pp: Dict) -> go.Figure:
    detail = pp.get("similar_days_detail") or []
    if not detail:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient historical data",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#aaa", size=12))
        fig.update_layout(**_layout(220))
        return fig

    changes = [d.get("actual_change_7d", 0) for d in detail
               if d.get("actual_change_7d") is not None]
    colors  = ["#22c55e" if c >= 0 else "#ef4444" for c in changes]
    borders = ["#16a34a" if c >= 0 else "#dc2626" for c in changes]
    labels  = [d.get("date", "")[:10] for d in detail]

    fig = go.Figure(go.Bar(
        x=labels, y=changes,
        marker=dict(color=colors, line=dict(color=borders, width=1)),
        text=[f"{c:+.1f}%" for c in changes],
        textposition="outside",
        textfont=dict(size=9, family="DM Mono"),
        hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#ddd", width=1))

    exp_ret = pp.get("expected_return_7d")
    if exp_ret is not None:
        fig.add_hline(y=exp_ret, line=dict(color="#065f46", width=1, dash="dot"),
                      annotation_text=f"Avg {exp_ret:+.1f}%",
                      annotation_font=dict(size=9, color="#065f46"))

    fig.update_layout(**_layout(220, yaxis=dict(ticksuffix="%"), bargap=0.25))
    return fig


def build_stream_bars(sc: Dict) -> go.Figure:
    ss      = sc.get("stream_scores") or {}
    streams = ["technical", "sentiment", "market", "related_news"]
    labels  = ["Technical", "Sentiment", "Market", "Related news"]
    scores  = [float((ss.get(s) or {}).get("raw_score") or 0) for s in streams]
    weights = [float((ss.get(s) or {}).get("weight")    or 0.25) for s in streams]
    colors  = ["#ecfdf5" if s >= 0 else "#fef2f2" for s in scores]
    borders = ["#065f46" if s >= 0 else "#991b1b" for s in scores]

    fig = go.Figure(go.Bar(
        y=labels, x=scores, orientation="h",
        marker=dict(color=colors, line=dict(color=borders, width=1)),
        text=[f"{s:+.2f}  (wt {w:.0%})" for s, w in zip(scores, weights)],
        textposition="inside",
        textfont=dict(size=10, family="DM Mono, monospace", color="#333"),
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#ddd", width=1))
    fig.update_layout(**_layout(
        180,
        xaxis=dict(range=[-1.1, 1.1],
                   tickvals=[-1, -0.5, 0, 0.5, 1],
                   ticktext=["-1", "-0.5", "0", "+0.5", "+1"]),
        yaxis=dict(showgrid=False),
    ))
    return fig


# ─── Mock data ────────────────────────────────────────────────────────────────

def _mock_state(ticker: str) -> Dict:
    np.random.seed(42)
    price = 185.40
    drift = 0.002
    vol   = 0.018
    days  = 8

    paths = []
    for _ in range(100):
        p = [price]
        for _ in range(days - 1):
            p.append(p[-1] * np.exp(drift - 0.5*vol**2 + vol * np.random.randn()))
        paths.append(p)

    arr      = np.array(paths)
    mean_p   = arr.mean(axis=0).tolist()
    upper_95 = np.percentile(arr, 97.5, axis=0).tolist()
    lower_95 = np.percentile(arr,  2.5, axis=0).tolist()
    upper_68 = np.percentile(arr,   84, axis=0).tolist()
    lower_68 = np.percentile(arr,   16, axis=0).tolist()

    similar = [
        {"date": "2024-09-12", "similarity_score": 0.91, "actual_change_7d":  4.2, "direction": "UP"},
        {"date": "2024-07-03", "similarity_score": 0.88, "actual_change_7d":  3.1, "direction": "UP"},
        {"date": "2024-05-21", "similarity_score": 0.85, "actual_change_7d":  2.8, "direction": "UP"},
        {"date": "2024-03-08", "similarity_score": 0.83, "actual_change_7d": -1.4, "direction": "DOWN"},
        {"date": "2023-11-14", "similarity_score": 0.81, "actual_change_7d":  5.6, "direction": "UP"},
    ]

    return {
        "ticker": ticker,
        "final_signal":     "BUY",
        "final_confidence": 0.6412,
        "signal_strength":  78,
        "trustworthiness":  0.6412,
        "monte_carlo_results": {
            "current_price":    price,
            "probability_up":   0.623,
            "expected_return":  2.08,
            "mean_forecast":    price * 1.0208,
            "confidence_95":    {"lower": 175.20, "upper": 196.80},
            "simulation_count": 1000,
            "time_horizon_days": 7,
            "volatility":       vol,
            "visualization_data": {
                "sample_paths": paths,
                "mean_path":    mean_p,
                "upper_95":     upper_95,
                "lower_95":     lower_95,
                "upper_68":     upper_68,
                "lower_68":     lower_68,
                "days":         list(range(days)),
            },
        },
        "signal_components": {
            "final_signal":     "BUY",
            "final_confidence": 0.6412,
            "final_score":      0.6091,
            "signal_agreement": 4,
            "signal_strength":  78,
            "trustworthiness":  0.6412,
            "streams_missing":  [],
            "stream_scores": {
                "technical":    {"raw_score": 0.720, "weight": 0.380, "contribution": 0.2736},
                "sentiment":    {"raw_score": 0.610, "weight": 0.240, "contribution": 0.1464},
                "market":       {"raw_score": 0.540, "weight": 0.220, "contribution": 0.1188},
                "related_news": {"raw_score": 0.110, "weight": 0.160, "contribution": 0.0176},
            },
            "risk_summary": {
                "overall_risk_level":   "LOW",
                "pump_and_dump_score":  8,
                "trading_safe":         True,
                "behavioral_summary":   "No anomalous patterns detected.",
                "primary_risk_factors": [],
                "alerts":               [],
            },
            "price_targets": {
                "current_price":       price,
                "forecasted_price":    price * 1.0208,
                "price_range_lower":   175.20,
                "price_range_upper":   196.80,
                "expected_return_pct": 2.08,
            },
            "pattern_prediction": {
                "sufficient_data":     True,
                "similar_days_found":  18,
                "prob_up":             0.72,
                "prob_down":           0.28,
                "expected_return_7d":  2.14,
                "worst_case_7d":      -3.80,
                "best_case_7d":        6.20,
                "median_return_7d":    2.05,
                "agreement_with_job1": "strong",
                "similar_days_detail": similar,
            },
            "prediction_graph_data": {
                "blended_expected_return":   1.86,
                "gbm_expected_return":       2.08,
                "empirical_expected_return": 1.64,
                "job2_blend_weight":  0.504,
                "gbm_blend_weight":   0.496,
                "similar_days_used":  18,
                "gbm_spread_lower":  -5.2,
                "gbm_spread_upper":   6.8,
                "empirical_lower":   -3.8,
                "empirical_upper":    6.2,
                "data_source":        "blended",
            },
            "trustworthiness_breakdown": {
                "reliability_score":    0.648,
                "agreement_score":      0.75,
                "pattern_score":        1.0,
                "insufficient_history": False,
                "using_prior":          False,
                "stream_hit_rates": {
                    "technical":    {"hit_rate": 0.71, "using_prior": False},
                    "sentiment":    {"hit_rate": 0.58, "using_prior": False},
                    "market":       {"hit_rate": 0.63, "using_prior": False},
                    "related_news": {"hit_rate": None, "using_prior": True},
                },
            },
            "backtest_context": {
                "streams_reliable":     3,
                "weights_are_fallback": False,
                "hold_threshold_pct":   1.5,
            },
        },
        "beginner_explanation": (
            f"**{ticker} — BUY · Signal Strength 78/100**\n\n"
            f"Our system is giving a **strong BUY** for {ticker}. "
            "Signal strength 78/100 — all four indicators agree. "
            "Trustworthiness **64%** — historically right ~64% of the time.\n\n"
            "**What's driving this:** Technical momentum strongly bullish (38% weight) · "
            "News sentiment slightly positive (24%) · Market mildly supportive (22%) · "
            "Related news neutral (16%)\n\n"
            "**Historical pattern:** 18 similar past days — stock rose 72% of those weeks, "
            "average +2.1%.\n\n"
            f"**Price:** ${price:.2f} now · forecast ${price*1.0208:.2f} · "
            "range $175.20 – $196.80.\n\n"
            "**Risk:** LOW. No unusual patterns.\n\n"
            "*This is an automated analysis, not financial advice. "
            "Always conduct your own research before making investment decisions.*"
        ),
        "technical_explanation": (
            f"## Executive Summary\n\n"
            f"{ticker} **BUY** (score +0.6091, trust 64%, strength 78/100). "
            "4/4 stream agreement. Technical dominant (+0.2736). "
            "Job 1/2 **strong agreement** — 18 similar days, prob_up 72%.\n\n"
            "## Signal Decomposition & Quality\n\n"
            "Technical 71% hit rate (live) · Sentiment 58% (live) · "
            "Market 63% (live) · Related news 55% (prior, n<30). "
            "Reliability 0.648 · Agreement 0.75 · Pattern 1.0.\n\n"
            "## Technical & Quantitative Analysis\n\n"
            "RSI 61.2 · MACD +1.84 > signal +1.21 · Price above 20d/50d SMA. "
            "MC: prob_up 62.3%, expected +2.08%, 95% CI [$175.20, $196.80]. "
            "Blended: +1.86% (50/50 empirical/GBM).\n\n"
            "## Sentiment & News Analysis\n\n"
            "Stock +0.61 · Market +0.54 · Related +0.11. "
            "Node 8 adjustment 1.24×. Credibility 0.74.\n\n"
            "## Market Context & Risk\n\n"
            "SPY 5d +1.8% · VIX 16.4 · Tech +2.3% · Beta 1.24 · Corr 0.81.\n\n"
            "## Anomaly Detection\n\nRisk LOW · P&D 8/100 · NORMAL. No alerts.\n\n"
            "## Methodology Notes\n\n"
            "3/4 streams have >30 signals. Related news transitions from prior once "
            "sample count reaches 30. Job 2 threshold 0.65."
        ),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="margin-bottom:1.5rem">
            <div style="font-size:22px;font-weight:700;color:#e0e0e0;
                        font-family:'DM Mono',monospace;letter-spacing:.06em">◈ SIGNAL</div>
            <div style="font-size:11px;color:#555;margin-top:2px;letter-spacing:.04em">
                AI TRADING ANALYSIS</div>
        </div>
        """, unsafe_allow_html=True)

        ticker_input = st.text_input(
            "Ticker", value="", placeholder="Enter ticker (e.g. AAPL)",
            label_visibility="collapsed",
        ).upper().strip()

        run_btn = st.button("Analyse", type="primary", use_container_width=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        view_mode = st.radio(
            "View", ["Beginner", "Technical"],
            horizontal=False, label_visibility="collapsed",
        )
        st.markdown("---")

        if "last_ticker" in st.session_state:
            lt      = st.session_state.last_ticker
            elapsed = st.session_state.get("elapsed", 0)
            st.markdown(f"""
            <div style="font-size:11px;color:#555;line-height:1.8">
                <b style="color:#888">Last run</b><br/>{lt} · {elapsed:.1f}s<br/><br/>
                <b style="color:#888">Pipeline</b><br/>14 nodes · LangGraph<br/><br/>
                <b style="color:#888">Simulations</b><br/>1,000 GBM paths
            </div>""", unsafe_allow_html=True)

        if not PIPELINE_AVAILABLE:
            st.markdown("""
            <div style="margin-top:1rem;font-size:10px;color:#c0392b;
                        padding:.5rem;background:#1a0a0a;border-radius:4px;
                        border:1px solid #4a1010">
                ⚠ Pipeline import failed — mock data only
            </div>""", unsafe_allow_html=True)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if run_btn:
        if not ticker_input:
            st.sidebar.warning("Please enter a ticker symbol.")
        else:
            with st.spinner(f"Running analysis for {ticker_input}…"):
                t0 = datetime.now()
                pipeline_error: Optional[str] = None
                if PIPELINE_AVAILABLE:
                    try:
                        state = asyncio.run(run_stock_analysis_async(ticker_input))
                    except Exception as e:
                        pipeline_error = str(e)
                        state = _mock_state(ticker_input)
                else:
                    pipeline_error = "Pipeline not available — import failed."
                    state = _mock_state(ticker_input)
                st.session_state.state          = state
                st.session_state.last_ticker    = ticker_input
                st.session_state.elapsed        = (datetime.now() - t0).total_seconds()
                st.session_state.pipeline_error = pipeline_error

    if "state" not in st.session_state:
        st.info("Enter a ticker in the sidebar and click **Analyse** to start.")
        st.stop()

    state = st.session_state.state

    pipeline_error = st.session_state.get("pipeline_error")
    if pipeline_error:
        st.error(
            f"**Workflow failed — displaying mock data.**\n\n"
            f"```\n{pipeline_error}\n```",
            icon="🔴",
        )

    sc    = state.get("signal_components") or {}
    mc    = state.get("monte_carlo_results") or {}
    pp    = sc.get("pattern_prediction") or {}
    pt    = sc.get("price_targets") or {}
    rs    = sc.get("risk_summary") or {}
    pg    = sc.get("prediction_graph_data") or {}
    tw    = sc.get("trustworthiness_breakdown") or {}

    ticker        = state.get("ticker", ticker_input)
    final_signal  = state.get("final_signal", "HOLD")
    sig_strength  = int(state.get("signal_strength") or 0)
    trust         = float(state.get("trustworthiness") or 0)
    current_price = float(mc.get("current_price") or pt.get("current_price") or 0)
    risk_level    = rs.get("overall_risk_level", "UNKNOWN")
    insuff        = tw.get("insufficient_history", True)

    # ── Header ────────────────────────────────────────────────────────────────
    col_t, col_s, _ = st.columns([2, 3, 5])
    with col_t:
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:28px;font-weight:700;
                    color:#0a0a0a;letter-spacing:.04em;line-height:1">{ticker}</div>
        <div style="font-size:11px;color:#aaa;margin-top:3px;letter-spacing:.04em">
            {datetime.now().strftime('%d %b %Y · %H:%M')}</div>
        """, unsafe_allow_html=True)
    with col_s:
        agree = int(sc.get("signal_agreement") or 0)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding-top:4px">
            {signal_pill(final_signal)}
            <span style="font-size:12px;color:#aaa">{agree}/4 streams agree</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Metric tiles ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Current price", fmt_price(current_price))
    with m2:
        forecast  = pt.get("forecasted_price")
        delta_pct = ((forecast - current_price) / current_price * 100
                     if forecast and current_price else None)
        st.metric("7-day forecast", fmt_price(forecast),
                  delta=f"{delta_pct:+.1f}%" if delta_pct else None)
    with m3:
        st.metric("Signal strength", f"{sig_strength}/100")
    with m4:
        st.metric("Trustworthiness",
                  "Building…" if insuff else f"{trust:.0%}")
    with m5:
        prob_up = pp.get("prob_up")
        n_sim   = pp.get("similar_days_found", 0)
        st.metric("Hist. prob up",
                  f"{prob_up:.0%}" if prob_up else "N/A",
                  delta=(f"{n_sim} similar days" if pp.get("sufficient_data")
                         else "Insufficient data"))
    with m6:
        pnds = int(rs.get("pump_and_dump_score") or 0)
        st.metric("Risk", risk_level, delta=f"P&D {pnds}/100",
                  delta_color="inverse")

    if insuff:
        st.markdown("""
        <div class="building-badge">
            ⏳ Trustworthiness uses a statistical prior (55%) —
            not enough backtested history yet.
        </div>""", unsafe_allow_html=True)

    # ── Stock chart ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Price history</div>',
                unsafe_allow_html=True)

    if "period" not in st.session_state:
        st.session_state.period = "1M"

    pcols = st.columns([1, 1, 1, 1, 1, 10])
    for i, p in enumerate(["1D", "7D", "1M", "3M", "6M"]):
        with pcols[i]:
            if st.button(p, key=f"period_{p}",
                         type="primary" if st.session_state.period == p else "secondary"):
                st.session_state.period = p

    # FIX 2: width='stretch' replaces deprecated use_container_width=True
    price_df = fetch_price_history(ticker, st.session_state.period)
    st.plotly_chart(build_stock_chart(price_df, ticker, st.session_state.period),
                    width="stretch", config={"displayModeBar": False})

    # ── Forecast + stream bars ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Forecast & signals</div>',
                unsafe_allow_html=True)
    ch1, ch2 = st.columns([3, 2])

    with ch1:
        j2w  = pg.get("job2_blend_weight", 0)
        gbmw = pg.get("gbm_blend_weight", 1)
        src  = pg.get("data_source", "gbm_only")
        st.markdown('<div class="chart-title">7-day price forecast</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="chart-subtitle">Blended: {j2w:.0%} empirical '
            f'+ {gbmw:.0%} GBM · {src}</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_forecast_chart(sc, mc), width="stretch",
                        config={"displayModeBar": False})

    with ch2:
        st.markdown('<div class="chart-title">Stream scores</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Raw score × adaptive weight</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(build_stream_bars(sc), width="stretch",
                        config={"displayModeBar": False})

    # ── Monte Carlo + similar days ────────────────────────────────────────────
    st.markdown('<div class="section-header">Simulation & historical patterns</div>',
                unsafe_allow_html=True)
    ch3, ch4 = st.columns([3, 2])

    with ch3:
        n_mc = mc.get("simulation_count", 1000)
        vol  = mc.get("volatility", 0)
        st.markdown('<div class="chart-title">Monte Carlo paths</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="chart-subtitle">{n_mc:,} simulations · '
            f'vol {vol:.2%} · 68% & 95% CI bands</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_monte_carlo_chart(mc), width="stretch",
                        config={"displayModeBar": False})

    with ch4:
        n_days = pp.get("similar_days_found", 0)
        prob_u = pp.get("prob_up", 0)
        st.markdown('<div class="chart-title">Similar historical days</div>',
                    unsafe_allow_html=True)
        st.markdown(
            (f'<div class="chart-subtitle">{n_days} days · '
             f'prob up {prob_u:.0%} · '
             f'avg {fmt_pct(pp.get("expected_return_7d"))}</div>')
            if pp.get("sufficient_data")
            else '<div class="chart-subtitle">Insufficient data</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_similar_days_chart(pp), width="stretch",
                        config={"displayModeBar": False})

    # ── Macro & Commodity Factors ─────────────────────────────────────────────
    macro_exp = (state.get("market_context") or {}).get("macro_factor_exposure") or {}
    factors   = macro_exp.get("identified_factors") or []
    if factors:
        st.markdown(
            '<div class="section-header">Macro &amp; commodity factors</div>',
            unsafe_allow_html=True,
        )
        _EXPOSURE_COLORS: dict = {
            "COST_INPUT":           ("#7f1d1d", "#fca5a5"),   # dark red bg, light red text
            "REVENUE_DRIVER":       ("#14532d", "#86efac"),   # dark green bg, light green text
            "COMPETITOR_PRESSURE":  ("#78350f", "#fcd34d"),   # dark amber bg, amber text
            "MACRO_SENSITIVITY":    ("#1e3a5f", "#93c5fd"),   # dark blue bg, blue text
        }
        _TREND_ICON: dict = {
            "UP":   ('<span style="color:#4ade80;font-weight:700">↑ UP</span>',),
            "DOWN": ('<span style="color:#f87171;font-weight:700">↓ DOWN</span>',),
            "FLAT": ('<span style="color:#9ca3af;font-weight:700">→ FLAT</span>',),
        }

        rows_html = ""
        for fac in factors:
            if not isinstance(fac, dict):
                continue
            name     = fac.get("factor_name", "—")
            exp_type = fac.get("exposure_type", "")
            expl     = fac.get("exposure_explanation", "")
            price    = fac.get("current_price")
            trend    = fac.get("price_trend")

            bg_c, txt_c = _EXPOSURE_COLORS.get(exp_type, ("#374151", "#d1d5db"))
            badge_html = (
                f'<span style="background:{bg_c};color:{txt_c};'
                f'padding:2px 7px;border-radius:4px;font-size:11px;'
                f'font-family:\'DM Mono\',monospace;white-space:nowrap">'
                f'{exp_type}</span>'
            )
            price_str = (
                f'<span style="font-family:\'DM Mono\',monospace">'
                f'${price:,.2f}</span>'
                if price is not None
                else '<span style="color:#6b7280">N/A</span>'
            )
            trend_html = (
                _TREND_ICON.get(trend, ('<span style="color:#9ca3af">—</span>',))[0]
                if trend else '<span style="color:#9ca3af">—</span>'
            )
            rows_html += (
                f"<tr>"
                f'<td style="font-size:13px;padding:6px 10px;color:#e5e7eb">{name}</td>'
                f'<td style="padding:6px 10px">{badge_html}</td>'
                f'<td style="padding:6px 10px;text-align:right">{price_str}</td>'
                f'<td style="padding:6px 10px;text-align:center">{trend_html}</td>'
                f'<td style="font-size:12px;color:#9ca3af;padding:6px 10px">{expl}</td>'
                f"</tr>"
            )

        st.markdown(
            f"""<table style="width:100%;border-collapse:collapse;
                background:#111827;border-radius:8px;overflow:hidden;margin-bottom:16px">
              <thead>
                <tr style="border-bottom:1px solid #374151">
                  <th style="text-align:left;padding:8px 10px;
                      font-size:11px;color:#6b7280;font-weight:500">Factor</th>
                  <th style="text-align:left;padding:8px 10px;
                      font-size:11px;color:#6b7280;font-weight:500">Exposure</th>
                  <th style="text-align:right;padding:8px 10px;
                      font-size:11px;color:#6b7280;font-weight:500">Price</th>
                  <th style="text-align:center;padding:8px 10px;
                      font-size:11px;color:#6b7280;font-weight:500">5d trend</th>
                  <th style="text-align:left;padding:8px 10px;
                      font-size:11px;color:#6b7280;font-weight:500">Why it matters</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>""",
            unsafe_allow_html=True,
        )

        macro_summary = macro_exp.get("macro_summary") or ""
        if macro_summary and macro_summary != "Unavailable.":
            st.markdown(
                f'<div style="font-size:12px;color:#9ca3af;line-height:1.6;'
                f'margin-bottom:16px;padding:0 2px">{macro_summary}</div>',
                unsafe_allow_html=True,
            )

    # ── Analysis text ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Analysis</div>',
                unsafe_allow_html=True)
    exp_col, detail_col = st.columns([3, 2])

    with exp_col:
        if view_mode == "Beginner":
            st.markdown(
                f'<div class="explanation-body">'
                f'{state.get("beginner_explanation","")}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(state.get("technical_explanation", ""))

    with detail_col:
        st.markdown("**Signal quality**")
        hr_data = tw.get("stream_hit_rates") or {}
        rows = []
        for stream, label in [("technical","Technical"), ("sentiment","Sentiment"),
                               ("market","Market"), ("related_news","Related")]:
            hr    = hr_data.get(stream) or {}
            hit   = hr.get("hit_rate")
            prior = hr.get("using_prior", True)
            badge = ('<span class="prior-badge">prior</span>' if prior
                     else '<span class="live-badge">live</span>')
            val   = f"{hit:.0%}" if hit is not None else "55%"
            rows.append(f'<tr><td style="color:#666;font-size:12px">{label}</td>'
                        f'<td style="font-family:\'DM Mono\',monospace;font-size:12px">'
                        f'{val}{badge}</td></tr>')

        rel = tw.get("reliability_score", 0)
        agr = tw.get("agreement_score",   0)
        pat = tw.get("pattern_score",     0)

        st.markdown(f"""
        <table class="sim-table" style="margin-bottom:12px">
            <tr><th>Stream</th><th>Hit rate</th></tr>
            {''.join(rows)}
        </table>
        <div style="font-size:11px;color:#aaa;line-height:1.8">
            Reliability <span style="font-family:'DM Mono';color:#333">{rel:.0%}</span>
            &nbsp;·&nbsp;
            Agreement   <span style="font-family:'DM Mono';color:#333">{agr:.0%}</span>
            &nbsp;·&nbsp;
            Pattern     <span style="font-family:'DM Mono';color:#333">{pat:.0%}</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if pp.get("sufficient_data"):
            st.markdown("**Top similar days**")
            detail   = pp.get("similar_days_detail") or []
            rows_sim = []
            for d in detail[:5]:
                chg = d.get("actual_change_7d", 0)
                cls = "up" if chg >= 0 else "dn"
                rows_sim.append(
                    f'<tr><td>{d.get("date","")}</td>'
                    f'<td style="color:#aaa">{d.get("similarity_score",0):.2f}</td>'
                    f'<td class="{cls}">{chg:+.1f}%</td></tr>'
                )
            st.markdown(f"""
            <table class="sim-table">
                <tr><th>Date</th><th>Sim</th><th>7d chg</th></tr>
                {''.join(rows_sim)}
            </table>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("**Risk**")
        alerts       = rs.get("alerts") or []
        risk_factors = rs.get("primary_risk_factors") or []
        pnd          = int(rs.get("pump_and_dump_score") or 0)
        st.markdown(f"""
        <div style="font-size:12px;color:#666;line-height:1.9">
            Level &nbsp;{risk_badge(risk_level)}<br/>
            P&D score &nbsp;<span style="font-family:'DM Mono'">{pnd}/100</span><br/>
            Alerts &nbsp;<span style="font-family:'DM Mono'">{len(alerts)}</span><br/>
            Risk factors &nbsp;<span style="font-family:'DM Mono'">{len(risk_factors)}</span>
        </div>""", unsafe_allow_html=True)
        for a in alerts[:3]:
            st.markdown(
                f'<div style="font-size:11px;color:#991b1b;margin-top:3px">⚠ {a}</div>',
                unsafe_allow_html=True,
            )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        This is an automated analysis, not financial advice.
        Always conduct your own research before making investment decisions.
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()