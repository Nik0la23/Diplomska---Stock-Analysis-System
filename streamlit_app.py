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
import pickle
import pathlib
from datetime import datetime
from typing import Dict, Optional

_HISTORY_FILE = pathlib.Path(__file__).parent / ".signal_history.pkl"

def _load_history() -> list:
    try:
        if _HISTORY_FILE.exists():
            return pickle.loads(_HISTORY_FILE.read_bytes())
    except Exception:
        pass
    return []

def _save_history(history: list) -> None:
    try:
        _HISTORY_FILE.write_bytes(pickle.dumps(history))
    except Exception:
        pass

from streamlit_app.components.signal_diagnostics import render_signal_diagnostics

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

try:
    from src.langgraph_nodes.node_15_dashboard import dashboard_node as _dashboard_node
    DASHBOARD_NODE_AVAILABLE = True
except ImportError:
    _dashboard_node = None
    DASHBOARD_NODE_AVAILABLE = False

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

section[data-testid="stSidebar"] {
    background: #0a0a0a !important; border-right: 1px solid #1e1e1e !important;
    transform: translateX(0) !important;
    min-width: 244px !important; width: 244px !important;
    visibility: visible !important; display: block !important;
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: #141414 !important; border: 1px solid #2a2a2a !important;
    border-radius: 6px !important; color: #e0e0e0 !important;
    font-family: 'DM Mono', monospace !important; font-size: 15px !important;
    letter-spacing: .05em !important; text-transform: uppercase !important;
}
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: #e0e0e0 !important; color: #0a0a0a !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 13px !important; letter-spacing: .04em !important;
    width: 100% !important; padding: 10px !important;
}
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] * { color: #0a0a0a !important; background: transparent !important; }
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover { background: #ffffff !important; }
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover * { color: #0a0a0a !important; }
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

/* collapse button is a div wrapper, not a button element */
[data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] { display:none !important; }


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


@st.cache_data(ttl=300)
def fetch_commodity_live_prices(yf_tickers: tuple) -> Dict[str, Dict]:
    """Fetch live price and 1-day change for a set of yfinance commodity tickers.

    Args:
        yf_tickers: tuple of yfinance ticker strings (e.g. ("CL=F", "GC=F")).
                    Must be a tuple (not list) so Streamlit can hash it for caching.

    Returns:
        dict keyed by yf_ticker: {"price": float, "change_pct": float}
        Missing or failed tickers are omitted.
    """
    result: Dict[str, Dict] = {}
    for sym in yf_tickers:
        if not sym:
            continue
        try:
            df = yf.download(sym, period="5d", interval="1d",
                             progress=False, auto_adjust=True,
                             multi_level_index=False)
            if df.empty:
                continue
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            closes = df["Close"].dropna()
            if len(closes) < 2:
                continue
            latest   = float(closes.iloc[-1])
            prev     = float(closes.iloc[-2])
            chg_pct  = (latest - prev) / prev * 100 if prev else 0.0
            result[sym] = {"price": latest, "change_pct": chg_pct}
        except Exception:
            pass
    return result


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


def build_monte_carlo_chart(viz: Dict) -> go.Figure:
    """viz is dashboard_data["visualization"]["mc_visualization"]."""
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


def build_forecast_chart(pg: Dict, current_price: float) -> go.Figure:
    """pg is dashboard_data["visualization"]["prediction_graph_data"]."""
    days = list(range(8))

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


def build_similar_days_chart(detail: list, exp_ret: Optional[float]) -> go.Figure:
    """detail is dashboard_data["visualization"]["similar_days_detail"]."""
    if not detail:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient historical data",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#aaa", size=12))
        fig.update_layout(**_layout(220))
        return fig

    valid   = [d for d in detail if d.get("actual_change_7d") is not None]
    changes = [d.get("actual_change_7d", 0) for d in valid]
    colors  = ["#22c55e" if c >= 0 else "#ef4444" for c in changes]
    borders = ["#16a34a" if c >= 0 else "#dc2626" for c in changes]
    labels  = [d.get("date", "")[:10] for d in valid]

    fig = go.Figure(go.Bar(
        x=labels, y=changes,
        marker=dict(color=colors, line=dict(color=borders, width=1)),
        text=[f"{c:+.1f}%" for c in changes],
        textposition="outside",
        textfont=dict(size=9, family="DM Mono"),
        hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#ddd", width=1))

    if exp_ret is not None:
        fig.add_hline(y=exp_ret, line=dict(color="#065f46", width=1, dash="dot"),
                      annotation_text=f"Avg {exp_ret:+.1f}%",
                      annotation_font=dict(size=9, color="#065f46"))

    fig.update_layout(**_layout(220, yaxis=dict(ticksuffix="%"), bargap=0.25))
    return fig


def build_stream_bars(stream_scores: Dict) -> go.Figure:
    """stream_scores is dashboard_data["visualization"]["stream_scores"]."""
    ss      = stream_scores
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

    state = {
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
        "market_context": {
            "market_correlation_profile": {
                "beta_calculated":    1.24,
                "market_correlation": 0.81,
            },
            "market_regime": {
                "vix_level":      16.4,
                "vix_category":   "LOW",
                "spy_trend_label": "BULLISH",
                "spy_return_5d":  0.018,
                "regime_label":   "RISK_ON",
            },
        },
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
    # Populate dashboard_data so the app always has one consistent data source.
    if DASHBOARD_NODE_AVAILABLE:
        state = _dashboard_node(state)
    return state


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── Load history from disk once per session ───────────────────────────────
    if "run_history" not in st.session_state:
        st.session_state.run_history = _load_history()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="margin-bottom:1.5rem">
            <div style="font-size:22px;font-weight:700;color:#e0e0e0;
                        font-family:'DM Mono',monospace;letter-spacing:.06em">◈ StockSentinel</div>
            <div style="font-size:11px;color:#555;margin-top:2px;letter-spacing:.04em">
                AI STOCK ANALYSIS SYSTEM</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("ticker_form", border=False):
            ticker_input = st.text_input(
                "Ticker", value="", placeholder="Enter ticker (e.g. AAPL)",
                label_visibility="collapsed",
            ).upper().strip()
            run_btn = st.form_submit_button("Analyse", type="primary", use_container_width=True)
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

        # ── History ───────────────────────────────────────────────────────────
        history = st.session_state.get("run_history", [])
        if history:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:10px;font-weight:600;text-transform:uppercase;
                        letter-spacing:.08em;color:#555;margin-bottom:6px">History</div>
            """, unsafe_allow_html=True)
            today = datetime.now().date()
            for i, entry in enumerate(history):
                ts: datetime = entry["timestamp"]
                label = f"{entry['ticker']}  ·  {ts.strftime('%H:%M')}" if ts.date() == today \
                    else f"{entry['ticker']}  ·  {ts.strftime('%d %b')}"
                col_h, col_r = st.columns([3, 1]) if ts.date() < today else (st.columns([1])[0], None)
                with col_h:
                    if st.button(label, key=f"hist_{i}", use_container_width=True):
                        st.session_state.state       = entry["state"]
                        st.session_state.last_ticker = entry["ticker"]
                        st.session_state.elapsed     = entry["elapsed"]
                if col_r is not None:
                    with col_r:
                        if st.button("↺", key=f"rerun_{i}", help="Rerun analysis"):
                            st.session_state.rerun_ticker = entry["ticker"]

        if not PIPELINE_AVAILABLE:
            st.markdown("""
            <div style="margin-top:1rem;font-size:10px;color:#c0392b;
                        padding:.5rem;background:#1a0a0a;border-radius:4px;
                        border:1px solid #4a1010">
                ⚠ Pipeline import failed — mock data only
            </div>""", unsafe_allow_html=True)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    _rerun_ticker = None
    if "rerun_ticker" in st.session_state:
        _rerun_ticker = st.session_state["rerun_ticker"]
        del st.session_state["rerun_ticker"]
    _effective_ticker = ticker_input or _rerun_ticker or ""
    if run_btn or _rerun_ticker:
        if not _effective_ticker:
            st.sidebar.warning("Please enter a ticker symbol.")
        else:
            with st.spinner(f"Running analysis for {_effective_ticker}…"):
                t0 = datetime.now()
                pipeline_error: Optional[str] = None
                if PIPELINE_AVAILABLE:
                    try:
                        state = asyncio.run(run_stock_analysis_async(_effective_ticker))
                    except Exception as e:
                        # ExceptionGroup (Python 3.11+ TaskGroup) hides the real cause.
                        # Unwrap one level so the user sees the actual sub-exception.
                        if hasattr(e, "exceptions") and e.exceptions:
                            causes = "\n".join(
                                f"[{i+1}] {type(sub).__name__}: {sub}"
                                for i, sub in enumerate(e.exceptions)
                            )
                            pipeline_error = f"{e}\n\nSub-exceptions:\n{causes}"
                        else:
                            pipeline_error = f"{type(e).__name__}: {e}"
                        state = _mock_state(_effective_ticker)
                else:
                    pipeline_error = "Pipeline not available — import failed."
                    state = _mock_state(_effective_ticker)
                elapsed = (datetime.now() - t0).total_seconds()
                st.session_state.state          = state
                st.session_state.last_ticker    = _effective_ticker
                st.session_state.elapsed        = elapsed
                st.session_state.pipeline_error = pipeline_error
                # Append to run history (max 10 entries, most recent first)
                if "run_history" not in st.session_state:
                    st.session_state.run_history = []
                st.session_state.run_history.insert(0, {
                    "ticker":    _effective_ticker,
                    "timestamp": datetime.now(),
                    "state":     state,
                    "elapsed":   elapsed,
                })
                st.session_state.run_history = st.session_state.run_history[:10]
                _save_history(st.session_state.run_history)

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

    # ── All display data comes from Node 15's dashboard_data ─────────────────
    dd   = state.get("dashboard_data") or {}
    es   = dd.get("executive_summary")    or {}
    pi   = dd.get("price_intelligence")   or {}
    mc_d = dd.get("monte_carlo")          or {}
    risk = dd.get("risk_assessment")      or {}
    hp   = dd.get("historical_pattern")   or {}
    llm  = dd.get("llm_analysis")         or {}
    viz  = dd.get("visualization")        or {}
    macro= dd.get("macro_factors")        or {}
    peers= dd.get("peer_companies")       or {}
    sec_fund = dd.get("sec_fundamentals") or {}
    diag = dd.get("diagnostics")          or {}
    tw   = dd.get("trustworthiness")      or {}
    mp   = dd.get("model_performance")    or {}
    sa   = dd.get("system_accuracy")      or {}

    ticker        = es.get("ticker") or ticker_input
    final_signal  = es.get("recommendation", "HOLD")
    sig_strength  = int(es.get("signal_strength") or 0)
    trust         = float(es.get("confidence_raw") or 0)
    current_price = float(es.get("current_price_usd") or 0)
    risk_level    = risk.get("overall_risk_level", "UNKNOWN")
    vol_risk      = risk.get("volatility_risk", "UNKNOWN")
    insuff        = tw.get("insufficient_history", True)

    # ── Header ────────────────────────────────────────────────────────────────
    col_t, col_s, _ = st.columns([2, 3, 5])
    with col_t:
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:28px;font-weight:700;
                    color:#ffffff;letter-spacing:.04em;line-height:1">{ticker}</div>
        <div style="font-size:11px;color:#aaa;margin-top:3px;letter-spacing:.04em">
            {datetime.now().strftime('%d %b %Y · %H:%M')}</div>
        """, unsafe_allow_html=True)
    with col_s:
        agree = int(es.get("streams_agreeing") or 0)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding-top:4px">
            {signal_pill(final_signal)}
            <span style="font-size:12px;color:#aaa">{agree}/4 streams agree</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Metric tiles ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    with m1:
        st.metric("Current price", fmt_price(current_price))
    with m2:
        forecast  = pi.get("forecasted_price_7d_usd")
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
        prob_up = hp.get("prob_up_empirical")
        n_sim   = hp.get("similar_days_found", 0)
        st.metric("Hist. prob up",
                  f"{prob_up:.0%}" if prob_up else "N/A",
                  delta=(f"{n_sim} similar days" if hp.get("sufficient_data")
                         else "Insufficient data"))
    with m6:
        pnds = int(risk.get("pump_and_dump_score") or 0)
        st.metric("Manip. Risk", risk_level, delta=f"Vol: {vol_risk}",
                  delta_color="inverse")
    with m7:
        score  = sa.get("composite_score")
        sa_ins = sa.get("insufficient_history", True)
        lives  = int(sa.get("live_streams") or 0)
        st.metric(
            "System accuracy",
            "Building…" if sa_ins else (f"{score:.0%}" if score is not None else "N/A"),
            delta=f"{lives}/4 streams live" if not sa_ins else None,
        )

    # ── System accuracy breakdown ─────────────────────────────────────────────
    if not sa.get("insufficient_history", True) and sa.get("composite_score") is not None:
        with st.expander("System accuracy breakdown", expanded=False):
            sig_acc  = sa.get("signal_accuracy")
            tech_q   = sa.get("technical_model_quality")
            sent_acc = sa.get("sentiment_accuracy")
            comp     = sa.get("data_completeness")
            ic_raw   = sa.get("ic_raw")
            ic_sig   = sa.get("ic_significant")

            def _bar(value: float, label: str, weight: str, note: str = "") -> str:
                pct = int(value * 100)
                color = "#065f46" if value >= 0.60 else "#92400e" if value >= 0.45 else "#991b1b"
                note_html = f'<span style="font-size:10px;color:#aaa;margin-left:6px">{note}</span>' if note else ""
                return f"""
                <div style="margin-bottom:10px">
                  <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                    <span style="font-size:12px;color:#555">{label}
                      <span style="font-size:10px;color:#aaa">({weight} weight)</span>
                    </span>
                    <span style="font-family:'DM Mono',monospace;font-size:12px;color:{color}">{pct}%{note_html}</span>
                  </div>
                  <div style="background:#f0f0f0;border-radius:3px;height:5px">
                    <div style="background:{color};width:{pct}%;height:5px;border-radius:3px"></div>
                  </div>
                </div>"""

            ic_note = f"IC={ic_raw:+.3f}{'✦' if ic_sig else ''}" if ic_raw is not None else "IC N/A"
            st.markdown(
                _bar(sig_acc  or 0.55, "Signal accuracy (BUY/SELL/HOLD hit rate)", "40%") +
                _bar(tech_q  or 0.50, "Technical model quality (IC score)", "25%", ic_note) +
                _bar(sent_acc or 0.55, "Sentiment prediction accuracy", "20%") +
                _bar(comp    or 0.0,  "Data completeness (live streams)", "15%",
                     f"{int((comp or 0)*4)}/4 streams"),
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="font-size:10px;color:#aaa;margin-top:6px;line-height:1.6">'
                'Composite of backtested directional accuracy (Node 11 weighted hit rates), '
                'Ridge regression IC score, news sentiment stream accuracy, and fraction of '
                'streams with ≥30 signals of live history. Clipped to [35%, 95%].'
                '</div>',
                unsafe_allow_html=True,
            )

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
        pg   = viz.get("prediction_graph_data") or {}
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
        st.plotly_chart(
            build_forecast_chart(pg, viz.get("current_price", 0)),
            width="stretch", config={"displayModeBar": False},
        )

    with ch2:
        st.markdown('<div class="chart-title">Stream scores</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Raw score × adaptive weight</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            build_stream_bars(viz.get("stream_scores") or {}),
            width="stretch", config={"displayModeBar": False},
        )

    # ── Monte Carlo + similar days ────────────────────────────────────────────
    st.markdown('<div class="section-header">Simulation & historical patterns</div>',
                unsafe_allow_html=True)
    ch3, ch4 = st.columns([3, 2])

    with ch3:
        n_mc = mc_d.get("num_simulations", 1000)
        vol  = mc_d.get("volatility_daily", 0)
        st.markdown('<div class="chart-title">Monte Carlo paths</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="chart-subtitle">{n_mc:,} simulations · '
            f'vol {vol:.2%} · 68% & 95% CI bands</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_monte_carlo_chart(viz.get("mc_visualization") or {}),
            width="stretch", config={"displayModeBar": False},
        )

    with ch4:
        n_days = hp.get("similar_days_found", 0)
        prob_u = hp.get("prob_up_empirical") or 0
        st.markdown('<div class="chart-title">Similar historical days</div>',
                    unsafe_allow_html=True)
        st.markdown(
            (f'<div class="chart-subtitle">{n_days} days · '
             f'prob up {prob_u:.0%} · '
             f'avg {fmt_pct(hp.get("expected_return_7d_pct"))}</div>')
            if hp.get("sufficient_data")
            else '<div class="chart-subtitle">Insufficient data</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_similar_days_chart(
                viz.get("similar_days_detail") or [],
                hp.get("expected_return_7d_pct"),
            ),
            width="stretch", config={"displayModeBar": False},
        )

    # ── Macro & Commodity Factors ─────────────────────────────────────────────
    factors = macro.get("identified_factors") or []
    if factors:
        st.markdown(
            '<div class="section-header">Macro &amp; commodity factors</div>',
            unsafe_allow_html=True,
        )
        _EXPOSURE_COLORS: dict = {
            "COST_INPUT":           ("#7f1d1d", "#fca5a5"),
            "REVENUE_DRIVER":       ("#14532d", "#86efac"),
            "COMPETITOR_PRESSURE":  ("#78350f", "#fcd34d"),
            "MACRO_SENSITIVITY":    ("#1e3a5f", "#93c5fd"),
        }

        # Fetch live prices for all factors that have a yf_ticker
        commodity_tickers = tuple(
            fac["yf_ticker"]
            for fac in factors
            if isinstance(fac, dict) and fac.get("yf_ticker")
        )
        live_prices: Dict[str, Dict] = (
            fetch_commodity_live_prices(commodity_tickers)
            if commodity_tickers else {}
        )

        rows_html = ""
        for fac in factors:
            if not isinstance(fac, dict):
                continue
            name     = fac.get("factor_name", "—")
            exp_type = fac.get("exposure_type", "")
            expl     = fac.get("exposure_explanation", "")
            yf_sym   = fac.get("yf_ticker")

            # Prefer live price; fall back to pipeline price if live fetch failed
            live     = live_prices.get(yf_sym) if yf_sym else None
            price    = live["price"]    if live else fac.get("current_price")
            chg_pct  = live["change_pct"] if live else None

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
            if chg_pct is not None:
                chg_color = "#4ade80" if chg_pct >= 0 else "#f87171"
                sign      = "+" if chg_pct >= 0 else ""
                chg_html  = (
                    f'<span style="font-family:\'DM Mono\',monospace;'
                    f'color:{chg_color};font-weight:600">'
                    f'{sign}{chg_pct:.2f}%</span>'
                )
            else:
                chg_html = '<span style="color:#6b7280">—</span>'

            rows_html += (
                f"<tr>"
                f'<td style="font-size:13px;padding:6px 10px;color:#e5e7eb">{name}</td>'
                f'<td style="padding:6px 10px">{badge_html}</td>'
                f'<td style="padding:6px 10px;text-align:right">{price_str}</td>'
                f'<td style="padding:6px 10px;text-align:right">{chg_html}</td>'
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
                  <th style="text-align:right;padding:8px 10px;
                      font-size:11px;color:#6b7280;font-weight:500">1d chg</th>
                  <th style="text-align:left;padding:8px 10px;
                      font-size:11px;color:#6b7280;font-weight:500">Why it matters</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>""",
            unsafe_allow_html=True,
        )

        macro_summary = macro.get("macro_summary") or ""
        if macro_summary and macro_summary != "Unavailable.":
            st.markdown(
                f'<div style="font-size:12px;color:#9ca3af;line-height:1.6;'
                f'margin-bottom:16px;padding:0 2px">{macro_summary}</div>',
                unsafe_allow_html=True,
            )

    # ── Peer Companies ────────────────────────────────────────────────────────
    peer_list = peers.get("companies") or []
    if peer_list:
        st.markdown(
            '<div class="section-header">Peer companies</div>',
            unsafe_allow_html=True,
        )
        _REL_COLORS: dict = {
            "COMPETITOR":      ("#7f1d1d", "#fca5a5"),
            "SUPPLIER":        ("#14532d", "#86efac"),
            "CUSTOMER":        ("#1e3a5f", "#93c5fd"),
            "SAME_SECTOR":     ("#374151", "#d1d5db"),
            "PARTNER":         ("#3b0764", "#d8b4fe"),
        }
        peer_rows_html = ""
        for c in peer_list:
            if not isinstance(c, dict):
                continue
            t   = c.get("ticker", "—")
            rel = c.get("relationship", "SAME_SECTOR")
            rsn = c.get("reason", "")
            bg_c, txt_c = _REL_COLORS.get(rel, ("#374151", "#d1d5db"))
            rel_label = rel.replace("_", " ").title()
            badge_html = (
                f'<span style="background:{bg_c};color:{txt_c};'
                f'padding:2px 7px;border-radius:4px;font-size:11px;'
                f'font-family:\'DM Mono\',monospace;white-space:nowrap">'
                f'{rel_label}</span>'
            )
            peer_rows_html += (
                f"<tr>"
                f'<td style="font-size:13px;padding:6px 10px;color:#e5e7eb;'
                f'font-family:\'DM Mono\',monospace;font-weight:600">{t}</td>'
                f'<td style="padding:6px 10px">{badge_html}</td>'
                f'<td style="font-size:12px;color:#9ca3af;padding:6px 10px">{rsn}</td>'
                f"</tr>"
            )
        st.markdown(
            f"""<table style="width:100%;border-collapse:collapse;
                background:#111827;border-radius:8px;overflow:hidden;margin-bottom:16px">
              <thead>
                <tr style="border-bottom:1px solid #374151">
                  <th style="text-align:left;padding:8px 10px;
                      font-size:11px;color:#6b7280;font-weight:500">Ticker</th>
                  <th style="text-align:left;padding:8px 10px;
                      font-size:11px;color:#6b7280;font-weight:500">Relationship</th>
                  <th style="text-align:left;padding:8px 10px;
                      font-size:11px;color:#6b7280;font-weight:500">Why it matters</th>
                </tr>
              </thead>
              <tbody>{peer_rows_html}</tbody>
            </table>""",
            unsafe_allow_html=True,
        )

    # ── SEC Fundamentals (Node 16) ────────────────────────────────────────────
    if sec_fund.get("available"):
        st.markdown(
            '<div class="section-header">SEC Fundamentals</div>',
            unsafe_allow_html=True,
        )

        # ── Summary row: quarters coverage / revenue / margin ───────────────
        _TREND_COLORS = {
            "growing":   ("#14532d", "#86efac"),
            "improving": ("#14532d", "#86efac"),
            "stable":    ("#374151", "#d1d5db"),
            "declining": ("#7f1d1d", "#fca5a5"),
            "mixed":     ("#78350f", "#fcd34d"),
        }

        def _fund_badge(label: str, color_map: dict, key: str) -> str:
            bg, fg = color_map.get(key, ("#374151", "#d1d5db"))
            return (
                f'<span style="background:{bg};color:{fg};padding:3px 10px;'
                f'border-radius:4px;font-size:12px;font-family:\'DM Mono\','
                f'monospace;font-weight:600;white-space:nowrap">{label}</span>'
            )

        rev_tr  = sec_fund.get("revenue_trend",  "stable")
        mrg_tr  = sec_fund.get("margin_trend",   "stable")
        mgmt_s  = sec_fund.get("management_sentiment")
        events  = sec_fund.get("recent_events") or []
        f_dates = sec_fund.get("filing_dates") or []
        qs      = sec_fund.get("quarters_covered", 0)
        dq      = sec_fund.get("data_quality", "partial")

        rev_badge = _fund_badge(rev_tr.capitalize(), _TREND_COLORS, rev_tr)
        mrg_badge = _fund_badge(mrg_tr.capitalize(), _TREND_COLORS, mrg_tr)

        dq_color = (
            "#4ade80" if dq == "complete" else "#fbbf24" if dq == "partial" else "#f87171"
        )
        quarters_line = (
            f'<span style="font-family:\'DM Mono\',monospace;font-size:15px;'
            f'font-weight:600;color:#e5e7eb">{qs}Q</span>'
            f'<span style="color:{dq_color};font-size:13px;margin-left:6px">· {dq}</span>'
        )

        # Sentiment bar: -1 (red) → 0 (grey) → +1 (green), 200px wide
        if mgmt_s is not None:
            pct_pos = int((mgmt_s + 1) / 2 * 100)
            bar_color = "#22c55e" if mgmt_s >= 0 else "#ef4444"
            sent_html = (
                f'<div style="font-size:11px;color:#6b7280;margin-bottom:4px">'
                f'Management tone '
                f'<span style="font-family:\'DM Mono\',monospace;'
                f'color:#e5e7eb">{mgmt_s:+.3f}</span></div>'
                f'<div style="background:#374151;border-radius:3px;height:6px;width:200px">'
                f'<div style="background:{bar_color};width:{pct_pos}%;height:6px;border-radius:3px"></div>'
                f'</div>'
            )
        else:
            sent_html = '<div style="font-size:11px;color:#6b7280">Management tone N/A</div>'

        events_html = "".join(
            f'<span style="background:#1f2937;color:#d1d5db;padding:2px 8px;'
            f'border-radius:3px;font-size:11px;margin-right:4px;margin-bottom:4px;'
            f'display:inline-block;font-family:\'DM Mono\',monospace">{ev}</span>'
            for ev in events[:6]
        ) or '<span style="color:#6b7280;font-size:11px">No recent events</span>'

        dates_str = " · ".join(f_dates) if f_dates else "—"

        st.markdown(
            f"""<div style="background:#111827;border-radius:8px;
                padding:16px 20px;margin-bottom:16px">
              <div style="display:flex;align-items:flex-start;gap:32px;flex-wrap:wrap">
                <div>
                  <div style="font-size:11px;color:#6b7280;margin-bottom:6px">Filing coverage</div>
                  {quarters_line}
                </div>
                <div>
                  <div style="font-size:11px;color:#6b7280;margin-bottom:6px">Revenue trend</div>
                  {rev_badge}
                </div>
                <div>
                  <div style="font-size:11px;color:#6b7280;margin-bottom:6px">Margin trend</div>
                  {mrg_badge}
                </div>
                <div>
                  {sent_html}
                </div>
              </div>
              <div style="margin-top:14px">
                <div style="font-size:11px;color:#6b7280;margin-bottom:6px">Recent events</div>
                <div style="display:flex;flex-wrap:wrap;gap:4px">{events_html}</div>
              </div>
              <div style="margin-top:10px;font-size:11px;color:#4b5563">
                Filings: {dates_str}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Peer filing events ───────────────────────────────────────────────
        peer_fc_list = sec_fund.get("peers") or []
        if peer_fc_list:
            peer_fc_rows = ""
            for p in peer_fc_list:
                pt      = p.get("ticker", "—")
                rel     = p.get("relationship", "")
                p_evs   = p.get("recent_events") or []
                p_sent  = p.get("event_sentiment")
                bg_c, txt_c = {
                    "COMPETITOR":  ("#7f1d1d", "#fca5a5"),
                    "SUPPLIER":    ("#14532d", "#86efac"),
                    "CUSTOMER":    ("#1e3a5f", "#93c5fd"),
                    "SAME_SECTOR": ("#374151", "#d1d5db"),
                    "PARTNER":     ("#3b0764", "#d8b4fe"),
                }.get(rel, ("#374151", "#d1d5db"))
                rel_label   = rel.replace("_", " ").title()
                rel_badge   = (
                    f'<span style="background:{bg_c};color:{txt_c};'
                    f'padding:2px 7px;border-radius:4px;font-size:11px;'
                    f'font-family:\'DM Mono\',monospace">{rel_label}</span>'
                )
                evs_text = ", ".join(p_evs[:3]) if p_evs else "—"
                if p_sent is not None:
                    sc_color = "#4ade80" if p_sent > 0.05 else "#f87171" if p_sent < -0.05 else "#9ca3af"
                    sent_cell = f'<span style="font-family:\'DM Mono\',monospace;color:{sc_color}">{p_sent:+.3f}</span>'
                else:
                    sent_cell = '<span style="color:#6b7280">N/A</span>'
                peer_fc_rows += (
                    f"<tr>"
                    f'<td style="font-size:13px;padding:6px 10px;color:#e5e7eb;'
                    f'font-family:\'DM Mono\',monospace;font-weight:600">{pt}</td>'
                    f'<td style="padding:6px 10px">{rel_badge}</td>'
                    f'<td style="font-size:12px;color:#9ca3af;padding:6px 10px">{evs_text}</td>'
                    f'<td style="padding:6px 10px;text-align:right">{sent_cell}</td>'
                    f"</tr>"
                )
            st.markdown(
                f"""<table style="width:100%;border-collapse:collapse;
                    background:#111827;border-radius:8px;overflow:hidden;margin-bottom:12px">
                  <thead>
                    <tr style="border-bottom:1px solid #374151">
                      <th style="text-align:left;padding:8px 10px;font-size:11px;color:#6b7280;font-weight:500">Peer</th>
                      <th style="text-align:left;padding:8px 10px;font-size:11px;color:#6b7280;font-weight:500">Relationship</th>
                      <th style="text-align:left;padding:8px 10px;font-size:11px;color:#6b7280;font-weight:500">Recent 8-K events</th>
                      <th style="text-align:right;padding:8px 10px;font-size:11px;color:#6b7280;font-weight:500">Event sentiment</th>
                    </tr>
                  </thead>
                  <tbody>{peer_fc_rows}</tbody>
                </table>""",
                unsafe_allow_html=True,
            )

    # ── Analysis text ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Analysis</div>',
                unsafe_allow_html=True)
    exp_col, detail_col = st.columns([3, 2])

    with exp_col:
        import re as _re
        if view_mode == "Beginner":
            _beg = _re.sub(r'(?i)#+\s*\d*\.?\s*DISCLAIMER.*', '', llm.get("beginner_explanation", ""), flags=_re.DOTALL).strip()
            st.markdown(
                f'<div class="explanation-body">{_beg}</div>',
                unsafe_allow_html=True,
            )
        else:
            _tech = _re.sub(r'(?i)#+\s*\d*\.?\s*DISCLAIMER.*', '', llm.get("technical_explanation", ""), flags=_re.DOTALL).strip()
            st.markdown(
                f'<div class="explanation-body">{_tech}</div>',
                unsafe_allow_html=True,
            )

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
        agr = tw.get("agreement_score", 0)
        pat = tw.get("pattern_score", 0)

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

        # ── IC Scores table ───────────────────────────────────────────────
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("**Information Coefficient (IC)**")

        def _ic_row(label: str, ic_key: str, p_key: str, sig_key: str, note: str = "") -> str:
            ic_val = mp.get(ic_key)
            ic_p   = mp.get(p_key)
            ic_sig = mp.get(sig_key)
            if ic_val is None:
                val_html = '<span style="color:#aaa">N/A</span>'
            else:
                color = ("#065f46" if (ic_sig and ic_val > 0)
                         else "#991b1b" if (ic_sig and ic_val < 0)
                         else "#92400e" if abs(ic_val) > 0.03 else "#aaa")
                star  = " ✦" if ic_sig else ""
                p_str = f'<span style="color:#aaa;font-size:10px"> p={ic_p:.3f}</span>' if ic_p is not None else ""
                val_html = (
                    f'<span style="font-family:\'DM Mono\',monospace;color:{color}">'
                    f'{ic_val:+.3f}{star}</span>{p_str}'
                )
            note_html = f'<span style="font-size:10px;color:#aaa"> {note}</span>' if note else ""
            return (
                f'<tr><td style="color:#666;font-size:12px">{label}</td>'
                f'<td style="font-size:12px">{val_html}{note_html}</td></tr>'
            )

        mkt_note = "idiosyn." if mp.get("market_news_decomposed") else "raw ret"
        ic_rows = "".join([
            _ic_row("Technical",   "technical_ic",   "technical_ic_p",   "technical_ic_significant"),
            _ic_row("Stock news",  "stock_news_ic",  "stock_news_ic_p",  "stock_news_ic_significant"),
            _ic_row("Market news", "market_news_ic", "market_news_ic_p", "market_news_ic_significant", mkt_note),
            _ic_row("Related",     "related_news_ic","related_news_ic_p","related_news_ic_significant"),
        ])
        st.markdown(f"""
        <table class="sim-table" style="margin-bottom:6px">
            <tr><th>Stream</th><th>IC score</th></tr>
            {ic_rows}
        </table>
        <div style="font-size:10px;color:#aaa;line-height:1.6">
            Pearson correlation between stream score and 7-day return.
            ✦ = significant (p&lt;0.05). Market news measured against
            {'idiosyncratic return (stock − beta×SPY)' if mp.get('market_news_decomposed') else 'raw stock return'}.
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        if hp.get("sufficient_data"):
            st.markdown("**Top similar days**")
            sim_detail = viz.get("similar_days_detail") or []
            rows_sim   = []
            for d in sim_detail[:5]:
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
        alerts       = risk.get("alerts") or []
        risk_factors = risk.get("primary_risk_factors") or []
        pnd          = int(risk.get("pump_and_dump_score") or 0)
        _beta        = risk.get("beta", 1.0)
        _vix         = risk.get("vix_level", 20.0)
        alert_rows = "".join(
            f'<div style="font-size:13px;color:#ff6b6b;margin-top:5px;line-height:1.5">⚠ {a}</div>'
            for a in alerts[:3]
        )
        st.markdown(f"""
        <div style="background:transparent;border:1px solid #ffffff33;border-radius:8px;padding:14px 16px;margin-top:4px">
            <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#ffffff99;margin-bottom:10px">Risk</div>
            <div style="font-size:14px;color:#ffffff;line-height:2.1">
                <span style="color:#ffffff99;font-size:12px">Manipulation risk</span>&nbsp;&nbsp;{risk_badge(risk_level)}<br/>
                <span style="color:#ffffff99;font-size:12px">P&D score</span>&nbsp;&nbsp;<span style="font-family:'DM Mono';font-weight:600">{pnd}/100</span><br/>
                <span style="color:#ffffff99;font-size:12px">Volatility risk</span>&nbsp;&nbsp;{risk_badge(vol_risk)}<br/>
                <span style="color:#ffffff99;font-size:12px">Beta</span>&nbsp;&nbsp;<span style="font-family:'DM Mono';font-weight:600">{_beta:.2f}</span>&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#ffffff99;font-size:12px">VIX</span>&nbsp;&nbsp;<span style="font-family:'DM Mono';font-weight:600">{_vix:.1f}</span><br/>
                <span style="color:#ffffff99;font-size:12px">Alerts</span>&nbsp;&nbsp;<span style="font-family:'DM Mono';font-weight:600">{len(alerts)}</span>&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:#ffffff99;font-size:12px">Risk factors</span>&nbsp;&nbsp;<span style="font-family:'DM Mono';font-weight:600">{len(risk_factors)}</span>
            </div>
            {alert_rows}
        </div>""", unsafe_allow_html=True)

    # ── Signal diagnostics ────────────────────────────────────────────────────
    render_signal_diagnostics(diag)



if __name__ == "__main__":
    main()