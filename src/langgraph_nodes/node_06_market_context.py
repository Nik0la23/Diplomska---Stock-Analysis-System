"""
Node 6: Market Context Analysis (FMP MCP + structured market_context schema).

Provides the macro "zoom out" view — what is the world around this stock doing
right now — as a rich, structured briefing in state["market_context"].

Execution context:
- Runs AFTER: Node 1 (raw_price_data), Node 9A (cleaned_market_news)
- Runs IN PARALLEL WITH: Nodes 4, 5, 7 (no data dependency conflicts)
- Runs BEFORE: Node 8, Node 11, Node 12, Nodes 13 & 14
"""

import json
import math
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.messages import HumanMessage
from langgraph.types import RunnableConfig

from src.graph.state import StockAnalysisState

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

SECTOR_ETFS: Dict[str, str] = {
    # GICS canonical names
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Financials":             "XLF",
    "Energy":                 "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Industrials":            "XLI",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Communication Services": "XLC",
    # yfinance alternate spellings (map to the same ETF)
    "Consumer Cyclical":      "XLY",   # yfinance name for Consumer Discretionary
    "Financial Services":     "XLF",   # yfinance name for Financials
    "Basic Materials":        "XLB",   # yfinance name for Materials
    "Communication":          "XLC",
    "Health Care":            "XLV",
}

# Hardcoded sector/industry fallback for common tickers.
# Used when yfinance .info returns an empty dict or omits sector data
# (happens under rate-limiting or network issues during stress runs).
SECTOR_FALLBACK: Dict[str, Tuple[str, str]] = {
    "TSLA":  ("Consumer Cyclical",       "Auto Manufacturers"),
    "AAPL":  ("Technology",             "Consumer Electronics"),
    "MSFT":  ("Technology",             "Software—Infrastructure"),
    "NVDA":  ("Technology",             "Semiconductors"),
    "AMD":   ("Technology",             "Semiconductors"),
    "INTC":  ("Technology",             "Semiconductors"),
    "GOOGL": ("Communication Services", "Internet Content & Information"),
    "GOOG":  ("Communication Services", "Internet Content & Information"),
    "META":  ("Communication Services", "Internet Content & Information"),
    "AMZN":  ("Consumer Cyclical",       "Internet Retail"),
    "NFLX":  ("Communication Services", "Entertainment"),
    "JPM":   ("Financials",             "Banks—Diversified"),
    "BAC":   ("Financials",             "Banks—Diversified"),
    "XOM":   ("Energy",                 "Oil & Gas Integrated"),
    "CVX":   ("Energy",                 "Oil & Gas Integrated"),
    "JNJ":   ("Healthcare",             "Drug Manufacturers—General"),
    "PFE":   ("Healthcare",             "Drug Manufacturers—General"),
    "RIVN":  ("Consumer Cyclical",       "Auto Manufacturers"),
    "NIO":   ("Consumer Cyclical",       "Auto Manufacturers"),
    "GM":    ("Consumer Cyclical",       "Auto Manufacturers"),
    "F":     ("Consumer Cyclical",       "Auto Manufacturers"),
    "LCID":  ("Consumer Cyclical",       "Auto Manufacturers"),
    "QCOM":  ("Technology",             "Semiconductors"),
    "TSM":   ("Technology",             "Semiconductors"),
    "AVGO":  ("Technology",             "Semiconductors"),
    "IONQ":  ("Technology",             "Computer Hardware"),
}

MARKET_INDICES: Dict[str, str] = {
    "S&P 500": "SPY",
    "NASDAQ":  "QQQ",
    "Dow Jones": "DIA",
}

# Maps FMP commodity symbol names (from LLM output) to yfinance futures tickers
_FMP_TO_YF: Dict[str, str] = {
    "CRUDE_OIL":   "CL=F",
    "NATURAL_GAS": "NG=F",
    "GOLD":        "GC=F",
    "SILVER":      "SI=F",
    "COPPER":      "HG=F",
    "WHEAT":       "ZW=F",
    "CORN":        "ZC=F",
    "SOYBEANS":    "ZS=F",
    "PLATINUM":    "PL=F",
    "ALUMINUM":    "ALI=F",
    "IRON_ORE":    "TIO=F",
    "LITHIUM":     "LIT",
    "NICKEL":      "NI=F",
    "ZINC":        "ZNC=F",
    "COAL":        "MTF=F",
    "URANIUM":     "UX=F",
    "LUMBER":      "LBS=F",
    "COTTON":      "CT=F",
    "SUGAR":       "SB=F",
    "COFFEE":      "KC=F",
    "COCOA":       "CC=F",
    "OIL":         "CL=F",
    "BRENT_OIL":   "BZ=F",
    "GASOLINE":    "RB=F",
    "HEATING_OIL": "HO=F",
}

# How many days of market news to include in sentiment average
MARKET_NEWS_LOOKBACK_DAYS: int = 14

# Alpha Vantage sentiment normalization divisor
# AV overall_sentiment_score ~ [-0.35, +0.35]; divide by this to map to [-1, +1]
AV_SENTIMENT_NORM: float = 0.35

MACRO_FACTORS_PROMPT = """
You are a macro analyst at a major investment bank.

Given:
- Stock: {ticker} ({company_name})
- Sector: {sector}
- Industry: {industry}

Identify the top 3-5 external macro factors that most directly affect this stock's price.
Focus on: commodities, interest rates, currency exposure, geopolitical risks, supply chain dependencies.

Return ONLY valid JSON. No preamble. No explanation outside the JSON.

{{
  "factors": [
    {{
      "factor_name": "string — human readable name",
      "fmp_commodity_symbol": "string or null — FMP symbol if commodity (e.g. COPPER, SILVER, CRUDE_OIL, NATURAL_GAS), null otherwise",
      "exposure_type": "COST_INPUT | REVENUE_DRIVER | COMPETITOR_PRESSURE | MACRO_SENSITIVITY",
      "exposure_explanation": "one sentence explaining why this factor matters for this stock"
    }}
  ]
}}
"""


def _fetch_commodity_prices_sync(
    fmp_symbols: List[str],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Fetch latest closing prices and 5-day trends for a list of FMP commodity symbols
    using yfinance. Returns (commodity_prices, commodity_trends) keyed by fmp_symbol.

    Trend thresholds: >+1% → UP, <-1% → DOWN, otherwise FLAT.
    """
    prices: Dict[str, float] = {}
    trends: Dict[str, str] = {}

    # Build yf_symbol → fmp_symbol reverse map (some fmp symbols may share a yf symbol)
    yf_to_fmp: Dict[str, str] = {}
    yf_symbols: List[str] = []
    for fmp_sym in fmp_symbols:
        yf_sym = _FMP_TO_YF.get(fmp_sym.upper())
        if yf_sym:
            yf_to_fmp[yf_sym] = fmp_sym
            yf_symbols.append(yf_sym)

    if not yf_symbols:
        return prices, trends

    try:
        # Download 10 calendar days to ensure ≥5 trading-day closes
        data = yf.download(
            yf_symbols,
            period="10d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        if data.empty:
            return prices, trends

        # yfinance returns MultiIndex columns when >1 symbol, flat when exactly 1
        close_df: pd.DataFrame
        if isinstance(data.columns, pd.MultiIndex):
            close_df = data["Close"]
        else:
            close_df = data[["Close"]].rename(columns={"Close": yf_symbols[0]})

        for yf_sym in yf_symbols:
            fmp_sym = yf_to_fmp[yf_sym]
            if yf_sym not in close_df.columns:
                continue
            series = close_df[yf_sym].dropna()
            if series.empty:
                continue
            latest_price = float(series.iloc[-1])
            prices[fmp_sym] = round(latest_price, 4)
            if len(series) >= 5:
                five_day_ago = float(series.iloc[-5])
                if five_day_ago != 0:
                    pct_change = (latest_price - five_day_ago) / five_day_ago * 100
                    if pct_change > 1.0:
                        trends[fmp_sym] = "UP"
                    elif pct_change < -1.0:
                        trends[fmp_sym] = "DOWN"
                    else:
                        trends[fmp_sym] = "FLAT"
                else:
                    trends[fmp_sym] = "FLAT"
            else:
                trends[fmp_sym] = "FLAT"
    except Exception as exc:
        logger.warning(f"[Node 6] yfinance commodity fetch failed: {exc}")

    return prices, trends


def _parse_tool_response(raw: Any) -> Any:
    """
    MCP adapter wraps all responses in:
    [{'type': 'text', 'text': '<json string>', 'id': '...'}]
    Extract the actual data before parsing.
    """
    try:
        # Already a parsed Python object (list/dict) — return directly
        if isinstance(raw, (dict, list)) and not (
            isinstance(raw, list) and raw and isinstance(raw[0], dict) and "type" in raw[0]
        ):
            return raw
        # MCP content block wrapper
        if isinstance(raw, list) and raw and isinstance(raw[0], dict) and raw[0].get("type") == "text":
            return json.loads(raw[0]["text"])
        # Plain string
        if isinstance(raw, str):
            return json.loads(raw)
        return raw
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return {}


# ============================================================================
# PURE HELPERS (SPY/VIX/regime, classification)
# ============================================================================


def _safe_spy_returns_from_history(closes: List[float]) -> Dict[str, float]:
    """
    Compute SPY multi-timeframe % returns and volatility from a list of closes.

    Expects prices ordered oldest→newest and at least ~25 trading days of data
    to reliably compute 1d/5d/21d returns. Returns zeros when data is
    insufficient.

    Args:
        closes: List of close prices ordered oldest→newest.

    Returns:
        Dict with keys: spy_return_1d, spy_return_5d, spy_return_21d,
        market_volatility_pct.
    """
    if len(closes) < 5:
        return {
            "spy_return_1d": 0.0,
            "spy_return_5d": 0.0,
            "spy_return_21d": 0.0,
            "market_volatility_pct": 0.0,
        }

    def _pct(start_idx: int) -> float:
        if len(closes) <= abs(start_idx):
            return 0.0
        start = float(closes[start_idx])
        end = float(closes[-1])
        if start == 0:
            return 0.0
        return float((end - start) / start * 100.0)

    ret_1d = _pct(-2)
    ret_5d = _pct(-6) if len(closes) >= 6 else _pct(0)
    ret_21d = _pct(-22) if len(closes) >= 22 else _pct(0)

    arr = np.asarray(closes, dtype=float)
    returns = np.diff(arr) / arr[:-1]
    volatility = float(np.std(returns) * 100.0) if returns.size > 1 else 0.0

    return {
        "spy_return_1d": ret_1d,
        "spy_return_5d": ret_5d,
        "spy_return_21d": ret_21d,
        "market_volatility_pct": volatility,
    }


def get_market_trend_multitimeframe_from_history(closes: List[float]) -> Dict[str, Any]:
    """
    Derive SPY trend and volatility from a historical closes list.

    Preserves the logic of the previous implementation but takes data from
    an FMP historical price tool instead of calling yfinance directly.

    Args:
        closes: List of SPY close prices ordered oldest→newest.

    Returns:
        {
            'spy_return_1d': float,
            'spy_return_5d': float,
            'spy_return_21d': float,
            'spy_trend_label': str,        # BULLISH | BEARISH | NEUTRAL
            'market_volatility_pct': float
        }
    """
    base = _safe_spy_returns_from_history(closes)

    def _score(pct: float) -> float:
        if pct > 1.0:
            return 1.0
        if pct > 0.5:
            return 0.5
        if pct > -0.5:
            return 0.0
        if pct > -1.0:
            return -0.5
        return -1.0

    composite = (
        0.2 * _score(base["spy_return_1d"])
        + 0.4 * _score(base["spy_return_5d"])
        + 0.4 * _score(base["spy_return_21d"])
    )

    if composite >= 0.3:
        trend = "BULLISH"
    elif composite <= -0.3:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return {
        "spy_return_1d": base["spy_return_1d"],
        "spy_return_5d": base["spy_return_5d"],
        "spy_return_21d": base["spy_return_21d"],
        "spy_trend_label": trend,
        "market_volatility_pct": base["market_volatility_pct"],
    }


def get_vix_level_from_price(vix_level: float) -> Dict[str, Any]:
    """
    Classify market fear from a VIX price and provide interpretation.

    Categories:
    - <15   CALM
    - 15-20 MODERATE
    - 20-25 ELEVATED
    - 25-30 HIGH
    - >30   PANIC
    """
    level = float(vix_level)
    if level < 15:
        category = "CALM"
        interpretation = "Low implied volatility; macro fear is muted."
    elif level < 20:
        category = "MODERATE"
        interpretation = "Normal market volatility; no major stress signals."
    elif level < 25:
        category = "ELEVATED"
        interpretation = "Volatility is elevated; macro headlines matter more."
    elif level < 30:
        category = "HIGH"
        interpretation = "High volatility; risk-off flows can dominate moves."
    else:
        category = "PANIC"
        interpretation = "Extreme volatility; short‑term moves are driven by fear."

    return {
        "vix_level": level,
        "vix_category": category,
        "vix_interpretation": interpretation,
    }


def classify_market_regime(
    spy_return_1d: float,
    spy_return_5d: float,
    vix_level: float,
) -> Dict[str, str]:
    """
    Classify overall market regime from SPY returns and VIX level.

    Implements the 8-label taxonomy from the Node 6 refactor guide.
    """
    vix_info = get_vix_level_from_price(vix_level)
    vix_category = vix_info["vix_category"]

    regime_label = "NEUTRAL"
    if spy_return_5d > 1.0 and vix_level < 18:
        regime_label = "RISK_ON_BULL"
    elif abs(spy_return_5d) <= 0.5 and vix_level < 15:
        regime_label = "LOW_VOL_GRIND"
    elif spy_return_5d > 1.0 and vix_level > 22:
        regime_label = "HIGH_VOL_BULL"
    elif abs(spy_return_5d) <= 0.5 and vix_level > 20:
        regime_label = "DISTRIBUTION"
    elif spy_return_5d < -1.0 and vix_level > 22:
        regime_label = "RISK_OFF_BEAR"
    elif spy_return_5d < -2.0 and vix_level > 28:
        regime_label = "PANIC_SELLOFF"
    elif spy_return_1d > 1.0 and spy_return_5d < 0:
        regime_label = "RECOVERY"

    descriptions = {
        "RISK_ON_BULL": "Broad market is in a risk-on, bullish phase; positive stock news is more likely to be rewarded.",
        "LOW_VOL_GRIND": "Market is grinding higher or sideways with low volatility; individual events can drive outsized moves.",
        "HIGH_VOL_BULL": "Market is rising but nervous; gains can be sharp but fragile.",
        "DISTRIBUTION": "Market shows topping behaviour; rallies are being sold into.",
        "RISK_OFF_BEAR": "Market is in risk-off mode; broad selling pressure can mute positive stock-specific catalysts.",
        "PANIC_SELLOFF": "Market is in panic; flows dominate fundamentals in the short term.",
        "RECOVERY": "Market is bouncing from oversold levels; follow‑through will confirm or reject the recovery.",
        "NEUTRAL": "No dominant macro regime; price action is balanced between macro and stock-specific drivers.",
    }

    return {
        "regime_label": regime_label,
        "regime_description": descriptions[regime_label],
        "vix_category": vix_category,
    }


def compute_relative_strength(
    stock_return_5d: float,
    stock_return_21d: float,
    sector_return_5d: float,
    sector_return_21d: float,
) -> Dict[str, Any]:
    """
    Compute relative strength of the stock vs its sector over 5d and 21d.
    """
    stock_vs_sector_5d = stock_return_5d - sector_return_5d
    stock_vs_sector_21d = stock_return_21d - sector_return_21d

    if stock_vs_sector_5d > 0.5:
        label = "LEADING"
    elif stock_vs_sector_5d < -0.5:
        label = "LAGGING"
    else:
        label = "IN_LINE"

    return {
        "stock_vs_sector_5d": stock_vs_sector_5d,
        "stock_vs_sector_21d": stock_vs_sector_21d,
        "relative_strength_label": label,
    }


def _derive_market_cap_tier(market_cap: Optional[float]) -> str:
    """
    Map numeric market cap to a tier string: mega/large/mid/small/Unknown.
    """
    if market_cap is None:
        return "Unknown"
    try:
        cap = float(market_cap)
    except (TypeError, ValueError):
        return "Unknown"

    if cap > 200_000_000_000:
        return "mega"
    if cap > 10_000_000_000:
        return "large"
    if cap > 2_000_000_000:
        return "mid"
    return "small"


def _summarise_index_memberships(raw_indices: Optional[List[str]]) -> List[str]:
    """
    Normalise index membership list from company_profile into human-friendly labels.
    """
    if not raw_indices:
        return []
    return list({str(idx) for idx in raw_indices})

def get_stock_sector(ticker: str) -> Tuple[str, str]:
    """
    Get the sector and industry for a stock using yfinance.

    Falls back to SECTOR_FALLBACK when yfinance returns an empty info dict
    (common under rate-limiting or during rapid stress-test runs).

    Args:
        ticker: Stock symbol (e.g., 'NVDA').

    Returns:
        (sector, industry) — 'Unknown' when unavailable.

    Example:
        >>> sector, industry = get_stock_sector('NVDA')
        >>> print(sector)
        'Technology'
    """
    upper = ticker.upper()
    try:
        info = yf.Ticker(ticker).info
        sector   = info.get("sector")   or ""
        industry = info.get("industry") or ""

        if sector:
            logger.info(f"  Sector lookup: {ticker} → {sector} / {industry}")
            return sector, industry or "Unknown"

        # yfinance returned an empty or incomplete info dict
        logger.warning(
            f"  Sector lookup: yfinance returned no sector for {ticker} — trying fallback"
        )
    except Exception as exc:
        logger.warning(f"  Sector lookup failed for {ticker}: {exc}")

    # Hardcoded fallback
    if upper in SECTOR_FALLBACK:
        sector, industry = SECTOR_FALLBACK[upper]
        logger.info(f"  Sector fallback: {ticker} → {sector} / {industry}")
        return sector, industry

    logger.warning(f"  Sector: no fallback entry for {ticker}, returning Unknown")
    return "Unknown", "Unknown"


# ============================================================================
# HELPER 2: VIX Fear Gauge
# ============================================================================

def get_vix_level() -> Dict[str, Any]:
    """
    Fetch current VIX level and classify market fear.

    VIX categories and their headwind contributions:
    - <15   CALM      → +0.30 (tailwind — investors complacent)
    - 15-20 MODERATE  →  0.00 (neutral)
    - 20-25 ELEVATED  → -0.30 (mild headwind)
    - 25-30 HIGH      → -0.60 (headwind)
    - >30   PANIC     → -1.00 (strong headwind)

    Returns:
        {
            'vix_level':      float,   # latest closing VIX value
            'vix_category':   str,     # CALM | MODERATE | ELEVATED | HIGH | PANIC
            'vix_contribution': float, # headwind score component on [-1, +1]
        }
    """
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="3d")

        if hist.empty:
            logger.warning("  VIX: no data, defaulting to MODERATE")
            return {"vix_level": 20.0, "vix_category": "MODERATE", "vix_contribution": 0.0}

        level = float(hist["Close"].iloc[-1])

        if level < 15:
            category, contribution = "CALM",     0.30
        elif level < 20:
            category, contribution = "MODERATE", 0.00
        elif level < 25:
            category, contribution = "ELEVATED", -0.30
        elif level < 30:
            category, contribution = "HIGH",     -0.60
        else:
            category, contribution = "PANIC",    -1.00

        logger.info(f"  VIX: {level:.1f} → {category} (contribution={contribution:+.2f})")
        return {"vix_level": level, "vix_category": category, "vix_contribution": contribution}

    except Exception as exc:
        logger.warning(f"  VIX fetch failed: {exc}")
        return {"vix_level": 20.0, "vix_category": "MODERATE", "vix_contribution": 0.0}


# ============================================================================
# HELPER 3: SPY Multi-Timeframe Trend
# ============================================================================

def get_market_trend_multitimeframe() -> Dict[str, Any]:
    """
    Compute SPY performance across three timeframes: 1d, 5d (1w), 21d (1m).

    Each timeframe is scored on [-1, +1]:
        >  1.0 % → +1.0
        >  0.5 % → +0.5
        > -0.5 % →  0.0
        > -1.0 % → -0.5
        ≤ -1.0 % → -1.0

    Composite SPY score = 0.2 × score_1d + 0.4 × score_5d + 0.4 × score_21d.

    Returns:
        {
            'market_trend':         str,    # BULLISH | BEARISH | NEUTRAL
            'market_performance':   float,  # 5d % (kept for backward compat)
            'spy_return_1d':        float,
            'spy_return_5d':        float,
            'spy_return_21d':       float,
            'volatility':           float,  # 21d daily-return std × 100
            'spy_composite_score':  float,  # weighted [-1, +1]
        }
    """
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="35d")  # 35 calendar ≈ 25 trading days

        if hist.empty or len(hist) < 5:
            logger.warning("  SPY: insufficient data")
            return {
                "market_trend": "NEUTRAL",
                "market_performance": 0.0,
                "spy_return_1d": 0.0,
                "spy_return_5d": 0.0,
                "spy_return_21d": 0.0,
                "volatility": 0.0,
                "spy_composite_score": 0.0,
            }

        closes = hist["Close"]

        def _pct(start_idx: int) -> float:
            """Return % change from closes.iloc[start_idx] to last close."""
            if len(closes) <= abs(start_idx):
                return 0.0
            return float((closes.iloc[-1] - closes.iloc[start_idx]) / closes.iloc[start_idx] * 100)

        ret_1d  = _pct(-2)
        ret_5d  = _pct(-6)  if len(closes) >= 6  else _pct(0)
        ret_21d = _pct(-22) if len(closes) >= 22 else _pct(0)

        def _score(pct: float) -> float:
            if pct >  1.0: return  1.0
            if pct >  0.5: return  0.5
            if pct > -0.5: return  0.0
            if pct > -1.0: return -0.5
            return -1.0

        composite = 0.2 * _score(ret_1d) + 0.4 * _score(ret_5d) + 0.4 * _score(ret_21d)

        returns = closes.pct_change().dropna()
        volatility = float(returns.std() * 100) if len(returns) > 1 else 0.0

        # Classify broad trend using 5d return for backward compat label
        if composite >= 0.3:
            trend = "BULLISH"
        elif composite <= -0.3:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        logger.info(
            f"  SPY: 1d={ret_1d:+.2f}% 5d={ret_5d:+.2f}% 21d={ret_21d:+.2f}% "
            f"→ composite={composite:+.3f} ({trend})"
        )
        return {
            "market_trend":        trend,
            "market_performance":  ret_5d,
            "spy_return_1d":       ret_1d,
            "spy_return_5d":       ret_5d,
            "spy_return_21d":      ret_21d,
            "volatility":          volatility,
            "spy_composite_score": composite,
        }

    except Exception as exc:
        logger.warning(f"  SPY multi-timeframe failed: {exc}")
        return {
            "market_trend": "NEUTRAL",
            "market_performance": 0.0,
            "spy_return_1d": 0.0,
            "spy_return_5d": 0.0,
            "spy_return_21d": 0.0,
            "volatility": 0.0,
            "spy_composite_score": 0.0,
        }


# ============================================================================
# HELPER 4: Sector Performance (5-day)
# ============================================================================

def get_sector_performance(sector: str, days: int = 5) -> Dict[str, Any]:
    """
    Get the performance of a sector ETF over the last N trading days.

    Uses the SECTOR_ETFS mapping (e.g., Technology → XLK).  5-day default
    avoids 1-day noise.

    Args:
        sector: Sector name (e.g., 'Technology').
        days:   Trading days to look back (default 5).

    Returns:
        {
            'sector':        str,
            'etf_ticker':    str | None,
            'performance':   float,  # % change
            'trend':         str,    # UP | DOWN | FLAT
            'sector_score':  float,  # contribution on [-1, +1]
        }
    """
    etf_ticker = SECTOR_ETFS.get(sector)

    _default = {
        "sector":       sector,
        "etf_ticker":   etf_ticker,
        "performance":  0.0,
        "trend":        "FLAT",
        "sector_score": 0.0,
    }

    if not etf_ticker:
        logger.warning(f"  Sector: no ETF mapping for '{sector}'")
        return _default

    try:
        hist = yf.Ticker(etf_ticker).history(period=f"{days + 2}d")

        if hist.empty or len(hist) < 2:
            logger.warning(f"  Sector: insufficient data for {etf_ticker}")
            return _default

        p_start   = float(hist["Close"].iloc[0])
        p_current = float(hist["Close"].iloc[-1])
        performance = (p_current - p_start) / p_start * 100

        if performance >  1.0: trend, score = "UP",   1.0
        elif performance > 0.0: trend, score = "UP",   0.5
        elif performance > -1.0: trend, score = "FLAT", 0.0
        elif performance > -3.0: trend, score = "DOWN", -0.5
        else:                    trend, score = "DOWN", -1.0

        if performance < -0.5:
            trend = "DOWN"
        elif performance > 0.5:
            trend = "UP"

        logger.info(f"  Sector {sector} ({etf_ticker}) {days}d: {performance:+.2f}% → {trend}")
        return {
            "sector":       sector,
            "etf_ticker":   etf_ticker,
            "performance":  performance,
            "trend":        trend,
            "sector_score": score,
        }

    except Exception as exc:
        logger.warning(f"  Sector performance failed for {etf_ticker}: {exc}")
        return _default


# ============================================================================
# HELPER 5: Stock–Market Correlation (NaN-safe)
# ============================================================================

def calculate_correlation(
    ticker: str,
    price_data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate Pearson correlation and beta between the stock and SPY.

    Fix: uses numpy.corrcoef on .values arrays instead of pandas .corr()
    to avoid NaN from timezone-aware vs timezone-naive DatetimeIndex mismatch.

    Args:
        ticker:     Stock symbol.
        price_data: OHLCV DataFrame from Node 1 (must contain 'close' column).

    Returns:
        {
            'market_correlation':   float,  # [-1, +1]; fallback 0.5 on error
            'correlation_strength': str,    # HIGH | MEDIUM | LOW
            'beta':                 float,
        }
    """
    _default = {"market_correlation": 0.5, "correlation_strength": "MEDIUM", "beta": 1.0}

    try:
        if price_data is None or len(price_data) < 15:
            logger.warning(f"  Correlation: insufficient price data ({len(price_data) if price_data is not None else 0} rows)")
            return _default

        # --- Stock returns — build a date-indexed Series from Node 1's DataFrame.
        #     Node 1 stores dates in a 'date' column (integer-indexed after reset_index).
        #     We normalise to timezone-naive date for joining with SPY data.
        stock_df = price_data[["date", "close"]].copy().dropna()
        stock_df["date"] = pd.to_datetime(stock_df["date"]).dt.normalize().dt.tz_localize(None)
        stock_df = stock_df.set_index("date").sort_index()
        stock_returns = stock_df["close"].pct_change().dropna().tail(60)

        if len(stock_returns) < 10:
            logger.warning("  Correlation: too few stock return points")
            return _default

        # --- SPY returns — fetch 90 calendar days to guarantee overlap with Node 1 window.
        spy_hist = yf.Ticker("SPY").history(period="90d")
        if spy_hist.empty or len(spy_hist) < 10:
            logger.warning("  Correlation: insufficient SPY data")
            return _default

        spy_returns = spy_hist["Close"].pct_change().dropna()
        spy_returns.index = pd.to_datetime(spy_returns.index).normalize().tz_localize(None)

        # --- Align on matching trading dates (inner join by date index) ---
        aligned = pd.concat(
            [stock_returns.rename("stock"), spy_returns.rename("spy")],
            axis=1,
            join="inner",
        ).dropna()

        if len(aligned) < 10:
            logger.warning(f"  Correlation: only {len(aligned)} aligned trading days — using default")
            return _default

        s_arr = aligned["stock"].values
        m_arr = aligned["spy"].values

        corr_matrix = np.corrcoef(s_arr, m_arr)
        correlation = float(corr_matrix[0, 1])

        # Guard against residual NaN/inf from zero-variance series
        if not math.isfinite(correlation):
            logger.warning("  Correlation: result not finite, using 0.5")
            correlation = 0.5

        s_std = float(np.std(s_arr))
        m_std = float(np.std(m_arr))
        beta  = (s_std / m_std * correlation) if m_std > 0 else 1.0

        strength = "HIGH" if abs(correlation) > 0.7 else ("MEDIUM" if abs(correlation) > 0.4 else "LOW")

        logger.info(
            f"  Correlation: {ticker}↔SPY = {correlation:.3f} ({strength}), beta={beta:.2f}"
            f"  [{len(aligned)} aligned days]"
        )
        return {
            "market_correlation":   correlation,
            "correlation_strength": strength,
            "beta":                 beta,
        }

    except Exception as exc:
        logger.warning(f"  Correlation calc failed for {ticker}: {exc}")
        return _default


# ============================================================================
# HELPER 7: Market News Sentiment (from cleaned_market_news)
# ============================================================================

def analyze_market_news_sentiment(
    cleaned_market_news: List[Dict[str, Any]],
    lookback_days: int = MARKET_NEWS_LOOKBACK_DAYS,
) -> Dict[str, Any]:
    """
    Aggregate sentiment from global market news fetched by Node 2 and
    cleaned by Node 9A.  Each article carries Alpha Vantage's
    overall_sentiment_score (roughly -0.35 … +0.35).

    Only articles from the last `lookback_days` days are included.

    Args:
        cleaned_market_news: List of market news article dicts from state.
        lookback_days:        How far back to look (default 14 days).

    Returns:
        {
            'market_news_sentiment': float,  # normalised [-1, +1]
            'market_news_count':     int,    # articles used
            'news_sentiment_score':  float,  # same as market_news_sentiment (alias)
        }
    """
    _default = {"market_news_sentiment": 0.0, "market_news_count": 0, "news_sentiment_score": 0.0}

    if not cleaned_market_news:
        logger.debug("  Market news sentiment: no articles in state")
        return _default

    cutoff_ts = (datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)).timestamp()

    scores: List[float] = []
    for article in cleaned_market_news:
        ts = article.get("datetime", 0)
        if ts < cutoff_ts:
            continue
        raw = article.get("overall_sentiment_score")
        if raw is None:
            continue
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            continue

    if not scores:
        logger.debug(f"  Market news sentiment: 0 recent articles (lookback={lookback_days}d)")
        return _default

    avg_raw = sum(scores) / len(scores)
    # Normalize from AV scale (~±0.35) to [-1, +1]
    normalised = max(-1.0, min(1.0, avg_raw / AV_SENTIMENT_NORM))

    logger.info(
        f"  Market news: {len(scores)} articles, avg AV score={avg_raw:.4f} → norm={normalised:+.3f}"
    )
    return {
        "market_news_sentiment": normalised,
        "market_news_count":     len(scores),
        "news_sentiment_score":  normalised,
    }


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def _build_fallback_market_context(ticker: str, elapsed: float, error_message: str) -> Dict[str, Any]:
    """
    Build the full fallback market_context dict as specified in the Node 6 refactor guide.
    """
    return {
        "errors": [error_message],
        "market_context": {
            "stock_classification": {
                "ticker": ticker,
                "company_name": "Unknown",
                "sector": "Unknown",
                "industry": "Unknown",
                "exchange": "Unknown",
                "index_memberships": [],
                "market_cap_tier": "Unknown",
                "beta_fmp": 1.0,
            },
            "market_regime": {
                "regime_label": "NEUTRAL",
                "regime_description": "Market context unavailable.",
                "spy_return_1d": 0.0,
                "spy_return_5d": 0.0,
                "spy_return_21d": 0.0,
                "spy_trend_label": "NEUTRAL",
                "vix_level": 20.0,
                "vix_category": "MODERATE",
                "vix_interpretation": "Unavailable.",
                "market_volatility_pct": 0.0,
            },
            "sector_industry_context": {
                "sector_return_5d": 0.0,
                "sector_trend": "FLAT",
                "industry_return_5d": 0.0,
                "industry_trend": "FLAT",
                "stock_vs_sector_5d": 0.0,
                "stock_vs_sector_21d": 0.0,
                "relative_strength_label": "IN_LINE",
                "sector_context_note": "Unavailable.",
            },
            "macro_factor_exposure": {
                "identified_factors": [],
                "commodity_prices": {},
                "commodity_trends": {},
                "macro_summary": "Unavailable.",
            },
            "market_correlation_profile": {
                "market_correlation": 0.5,
                "correlation_strength": "MEDIUM",
                "beta_calculated": 1.0,
                "beta_interpretation": "Unavailable.",
                "correlation_note": "Unavailable.",
            },
            "news_sentiment_context": {
                "market_news_sentiment": 0.0,
                "market_news_count": 0,
                "sentiment_label": "NEUTRAL",
                "sentiment_interpretation": "Unavailable.",
            },
        },
        "node_execution_times": {"node_6": elapsed},
    }


async def market_context_node(state: StockAnalysisState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Node 6: Market Context Analysis (FMP MCP, async).

    Uses FMP tools (Pattern A) and bare LLM calls (Pattern C) to construct
    state["market_context"] with six sub-dicts:
    - stock_classification
    - market_regime
    - sector_industry_context
    - macro_factor_exposure
    - market_correlation_profile
    - news_sentiment_context
    """
    start = time.time()
    ticker = state["ticker"]
    logger.info(f"[Node 6] Starting for {ticker}")

    configurable = config["configurable"]
    tools_by_name = configurable["tools_by_name"]
    llm = configurable["llm"]
    llm_with_tools = configurable.get("llm_with_tools")  # not used in this node

    # Guard: Node 6 uses Pattern A (direct tools) and Pattern C (bare LLM),
    # but never Pattern B (agent loop). We intentionally do NOT require
    # llm_with_tools here; only tools_by_name and llm must be present.
    if not tools_by_name or llm is None:
        logger.warning(f"[Node 6] FMP config incomplete — skipping FMP calls")
        elapsed = time.time() - start
        return _build_fallback_market_context(
            ticker, elapsed, f"Node 6: FMP config unavailable for {ticker}"
        )

    try:
        # ------------------------------------------------------------------
        # MCP compatibility layer:
        # - Legacy tool surface: profile-symbol / historical-*/quote direct tools
        # - Current tool surface: company / marketPerformance / chart / quote(endpoint=...)
        # Keeps Node 6 output schema unchanged for downstream nodes.
        # ------------------------------------------------------------------
        company_tool = tools_by_name.get("company")
        market_performance_tool = tools_by_name.get("marketPerformance")
        chart_tool = tools_by_name.get("chart")

        async def _invoke_with_payload_fallbacks(tool: Any, payloads: List[Dict[str, Any]]) -> Any:
            last_exc: Optional[Exception] = None
            for payload in payloads:
                try:
                    return await tool.ainvoke(payload)
                except Exception as exc:  # try next payload style
                    last_exc = exc
            if last_exc is not None:
                raise last_exc
            return None

        # ------------------------------------------------------------------
        # Step 1: company_profile (Pattern A)
        # ------------------------------------------------------------------
        company_profile_tool = tools_by_name.get("profile-symbol")
        company_profile_data: Dict[str, Any] = {}
        if company_profile_tool is not None or company_tool is not None:
            try:
                if company_profile_tool is not None:
                    raw_profile = await company_profile_tool.ainvoke({"symbol": ticker})
                else:
                    raw_profile = await company_tool.ainvoke(
                        {"endpoint": "profile-symbol", "symbol": ticker}
                    )
                parsed_profile = _parse_tool_response(raw_profile)
                if isinstance(parsed_profile, list) and parsed_profile:
                    company_profile_data = parsed_profile[0]
                elif isinstance(parsed_profile, dict):
                    company_profile_data = parsed_profile
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    f"[Node 6] Could not parse company_profile result for {ticker}: {exc}"
                )
                company_profile_data = {}
            except Exception as exc:
                logger.warning(
                    f"[Node 6] company_profile tool failed for {ticker}: {exc}"
                )

        company_name = str(company_profile_data.get("companyName") or "Unknown")
        sector = str(company_profile_data.get("sector") or "Unknown")
        industry = str(company_profile_data.get("industry") or "Unknown")
        exchange = str(company_profile_data.get("exchange") or "Unknown")
        raw_indices = company_profile_data.get("indexMemberships") or []
        index_memberships = _summarise_index_memberships(raw_indices)
        beta_fmp = float(company_profile_data.get("beta") or 1.0)
        market_cap_tier = _derive_market_cap_tier(company_profile_data.get("marketCap"))

        # ------------------------------------------------------------------
        # Step 2: Macro factor identification (Pattern C – bare LLM)
        # ------------------------------------------------------------------
        macro_prompt = MACRO_FACTORS_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            sector=sector,
            industry=industry,
        )
        identified_factors: List[Dict[str, Any]] = []
        commodity_symbols: List[str] = []
        try:
            response = await llm.ainvoke([HumanMessage(content=macro_prompt)])
            content = response.content  # type: ignore[union-attr]
            # Strip markdown code fences if present
            if isinstance(content, str):
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()
            factors_data = json.loads(content) if isinstance(content, str) else {}
            if isinstance(factors_data, dict):
                identified_factors = factors_data.get("factors", []) or []
            commodity_symbols = [
                f["fmp_commodity_symbol"]
                for f in identified_factors
                if isinstance(f, dict) and f.get("fmp_commodity_symbol")
            ]
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            logger.warning(
                f"[Node 6] Macro factor LLM parse failed for {ticker}: {exc}"
            )
            identified_factors = []
            commodity_symbols = []
        except Exception as exc:
            logger.warning(
                f"[Node 6] Macro factor LLM call failed for {ticker}: {exc}"
            )
            identified_factors = []
            commodity_symbols = []

        # ------------------------------------------------------------------
        # Step 3: Sector/industry, SPY history, VIX, commodities (Pattern A)
        # ------------------------------------------------------------------
        sector_return_5d = 0.0
        sector_trend = "FLAT"
        industry_return_5d = 0.0
        industry_trend = "FLAT"
        spy_closes: List[float] = []
        vix_price = 20.0
        commodity_prices: Dict[str, float] = {}
        commodity_trends: Dict[str, str] = {}

        sector_tool = tools_by_name.get("historical-sector-performance")
        if (sector_tool is not None or market_performance_tool is not None) and sector != "Unknown":
            try:
                if sector_tool is not None:
                    raw = await sector_tool.ainvoke({"sector": sector})
                else:
                    raw = await market_performance_tool.ainvoke(
                        {"endpoint": "historical-sector-performance", "sector": sector}
                    )
                logger.info(f"[Node 6] RAW sector response (first 500 chars): {str(raw)[:500]}")
                data = _parse_tool_response(raw)
                if isinstance(data, list) and len(data) >= 6:
                    # data is ordered newest first
                    latest = float(data[0].get("averageChange") or data[0].get("performance") or 0.0)
                    older = float(data[5].get("averageChange") or data[5].get("performance") or 0.0)
                    sector_return_5d = latest - older
                    if sector_return_5d > 0.5:
                        sector_trend = "UP"
                    elif sector_return_5d < -0.5:
                        sector_trend = "DOWN"
                    else:
                        sector_trend = "FLAT"
                elif isinstance(data, dict):
                    sector_return_5d = float(data.get("return_5d") or 0.0)
                    sector_trend = str(data.get("trend") or "FLAT")
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    f"[Node 6] sector_performance parse failed for {ticker}: {exc}"
                )
            except Exception as exc:
                logger.warning(
                    f"[Node 6] sector_performance tool failed for {ticker}: {exc}"
                )

        industry_tool = tools_by_name.get("historical-industry-performance")
        if (industry_tool is not None or market_performance_tool is not None) and industry != "Unknown":
            try:
                if industry_tool is not None:
                    raw = await industry_tool.ainvoke({"industry": industry})
                else:
                    raw = await market_performance_tool.ainvoke(
                        {"endpoint": "historical-industry-performance", "industry": industry}
                    )
                data = _parse_tool_response(raw)
                if isinstance(data, list) and len(data) >= 6:
                    # data is ordered newest first
                    latest = float(data[0].get("averageChange") or data[0].get("performance") or 0.0)
                    older = float(data[5].get("averageChange") or data[5].get("performance") or 0.0)
                    industry_return_5d = latest - older
                    if industry_return_5d > 0.5:
                        industry_trend = "UP"
                    elif industry_return_5d < -0.5:
                        industry_trend = "DOWN"
                    else:
                        industry_trend = "FLAT"
                elif isinstance(data, dict):
                    industry_return_5d = float(data.get("return_5d") or 0.0)
                    industry_trend = str(data.get("trend") or "FLAT")
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    f"[Node 6] industry_performance parse failed for {ticker}: {exc}"
                )
            except Exception as exc:
                logger.warning(
                    f"[Node 6] industry_performance tool failed for {ticker}: {exc}"
                )

        # SPY historical prices – use FMP historical-price-eod-light.
        spy_history_tool = tools_by_name.get("historical-price-eod-light")
        if spy_history_tool is not None or chart_tool is not None:
            try:
                if spy_history_tool is not None:
                    # Legacy shape used by older MCP adapters.
                    raw = await spy_history_tool.ainvoke({"symbol": "SPY", "days": 30})
                else:
                    # Current MCP chart endpoint expects explicit date range.
                    to_date = datetime.now(tz=timezone.utc).date()
                    from_date = to_date - timedelta(days=45)
                    raw = await chart_tool.ainvoke({
                        "endpoint": "historical-price-eod-light",
                        "symbol": "SPY",
                        "from": str(from_date),
                        "to": str(to_date),
                    })
                data = _parse_tool_response(raw)
                if isinstance(data, list):
                    spy_closes = [
                        float(row.get("price") or row.get("close"))
                        for row in data
                        if isinstance(row, dict) and (row.get("price") is not None or row.get("close") is not None)
                    ]
                    spy_closes.reverse()  # FMP returns newest-first; helper expects oldest-first
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    f"[Node 6] SPY historical parse failed for {ticker}: {exc}"
                )
                spy_closes = []
            except Exception as exc:
                logger.warning(
                    f"[Node 6] SPY historical tool failed for {ticker}: {exc}"
                )
                spy_closes = []

        vix_quote_tool = tools_by_name.get("quote")
        if vix_quote_tool is not None:
            try:
                # Try legacy payload first, then current MCP payload requiring endpoint.
                raw = await _invoke_with_payload_fallbacks(
                    vix_quote_tool,
                    [
                        {"symbol": "^VIX"},
                        {"endpoint": "quote", "symbol": "^VIX"},
                    ],
                )
                data = _parse_tool_response(raw)
                if isinstance(data, list) and data:
                    vix_price = float(data[0].get("price") or vix_price)
                elif isinstance(data, dict):
                    vix_price = float(data.get("price") or vix_price)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    f"[Node 6] VIX quote parse failed for {ticker}: {exc}"
                )
            except Exception as exc:
                logger.warning(
                    f"[Node 6] VIX quote tool failed for {ticker}: {exc}"
                )

        # Fetch live commodity prices via yfinance for identified commodity symbols.
        if commodity_symbols:
            try:
                import asyncio
                commodity_prices, commodity_trends = await asyncio.to_thread(
                    _fetch_commodity_prices_sync, commodity_symbols
                )
                logger.info(
                    f"[Node 6] Fetched commodity prices for {ticker}: "
                    f"{list(commodity_prices.keys())}"
                )
            except Exception as exc:
                logger.warning(
                    f"[Node 6] Commodity price fetch via asyncio.to_thread failed: {exc}"
                )
                commodity_prices = {}
                commodity_trends = {}

        # Enrich identified_factors in-place with price + trend for downstream nodes
        for factor in identified_factors:
            if not isinstance(factor, dict):
                continue
            fmp_sym = factor.get("fmp_commodity_symbol")
            yf_ticker = _FMP_TO_YF.get(fmp_sym) if fmp_sym else None
            factor["yf_ticker"] = yf_ticker
            if fmp_sym and fmp_sym in commodity_prices:
                factor["current_price"] = commodity_prices[fmp_sym]
                factor["price_trend"] = commodity_trends.get(fmp_sym, "FLAT")
            else:
                factor["current_price"] = None
                factor["price_trend"] = None

        # ------------------------------------------------------------------
        # Step 4: Pure Python layers (SPY/VIX/regime, relative strength)
        # ------------------------------------------------------------------
        if spy_closes:
            spy_trend_data = get_market_trend_multitimeframe_from_history(spy_closes)
        else:
            spy_trend_data = {
                "spy_return_1d": 0.0,
                "spy_return_5d": 0.0,
                "spy_return_21d": 0.0,
                "spy_trend_label": "NEUTRAL",
                "market_volatility_pct": 0.0,
            }

        vix_info = get_vix_level_from_price(vix_price)
        regime_info = classify_market_regime(
            spy_return_1d=spy_trend_data["spy_return_1d"],
            spy_return_5d=spy_trend_data["spy_return_5d"],
            vix_level=vix_info["vix_level"],
        )

        # Stock returns from Node 1 price data.
        stock_return_5d = 0.0
        stock_return_21d = 0.0
        price_data = state.get("raw_price_data")
        if isinstance(price_data, pd.DataFrame) and not price_data.empty:
            try:
                closes_series = price_data["close"]
                if len(closes_series) >= 6:
                    stock_return_5d = float(
                        (closes_series.iloc[-1] - closes_series.iloc[-6])
                        / closes_series.iloc[-6]
                        * 100.0
                    )
                if len(closes_series) >= 22:
                    stock_return_21d = float(
                        (closes_series.iloc[-1] - closes_series.iloc[-22])
                        / closes_series.iloc[-22]
                        * 100.0
                    )
            except Exception as exc:
                logger.warning(
                    f"[Node 6] stock return calculation failed for {ticker}: {exc}"
                )

        rs_info = compute_relative_strength(
            stock_return_5d=stock_return_5d,
            stock_return_21d=stock_return_21d,
            sector_return_5d=sector_return_5d,
            sector_return_21d=industry_return_5d,
        )

        # Correlation profile (existing helper with yfinance fallback).
        if isinstance(price_data, pd.DataFrame) and not price_data.empty:
            correlation_data = calculate_correlation(ticker, price_data)
        else:
            correlation_data = {
                "market_correlation": 0.5,
                "correlation_strength": "MEDIUM",
                "beta": 1.0,
            }

        beta_calculated = float(correlation_data["beta"])
        if abs(correlation_data["market_correlation"]) > 0.7:
            corr_note = "Stock moves closely with the market; macro regime has strong influence."
        elif abs(correlation_data["market_correlation"]) < 0.4:
            corr_note = "Stock is relatively de‑coupled from the market; company‑specific news dominates."
        else:
            corr_note = "Stock has a moderate link to the market; both macro and company news matter."

        beta_interpretation = (
            f"Beta of {beta_calculated:.2f} means {ticker} tends to move about "
            f"{abs(beta_calculated):.1f}× the market; in {regime_info['regime_label']} this amplifies "
            f"{'upside in bull markets and downside in bear markets' if beta_calculated > 1 else 'moves'}."
        )

        # News sentiment context.
        cleaned_market_news = state.get("cleaned_market_news") or []
        news_data = analyze_market_news_sentiment(cleaned_market_news)
        sentiment_score = float(news_data["market_news_sentiment"])
        if sentiment_score > 0.15:
            sentiment_label = "POSITIVE"
        elif sentiment_score < -0.15:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"
        sentiment_interpretation = (
            "Market narrative is broadly positive."
            if sentiment_label == "POSITIVE"
            else "Market narrative is broadly negative."
            if sentiment_label == "NEGATIVE"
            else "Market narrative is mixed or neutral."
        )

        # ------------------------------------------------------------------
        # Step 5: Macro summary (Pattern C – bare LLM, fire-and-forget)
        # Peers from Node 3 are included as qualitative context so the LLM
        # can reason about supplier/customer/competitor amplification effects.
        # ------------------------------------------------------------------
        macro_summary = "Unavailable."
        try:
            # Build peers context string from state["related_companies"].
            # Handles both Dict format {"ticker":.., "relationship":.., "reason":..}
            # and legacy str format gracefully.
            peers_context = ""
            related = state.get("related_companies") or []
            if related:
                peer_lines = []
                for c in related:
                    if isinstance(c, dict):
                        peer_lines.append(
                            f"  - {c['ticker']} ({c.get('relationship', 'SAME_SECTOR')}): {c.get('reason', '')}"
                        )
                    else:
                        peer_lines.append(f"  - {c}")
                peers_context = "Related companies:\n" + "\n".join(peer_lines)

            summary_prompt = (
                "You are a macro analyst explaining how current macro and commodity "
                "conditions affect a single stock.\n\n"
                f"Stock: {ticker} ({company_name}), sector={sector}, industry={industry}\n"
                f"Identified macro factors: {identified_factors}\n"
                f"Commodity prices: {commodity_prices}\n"
                f"Commodity trends (5d): {commodity_trends}\n"
                f"{peers_context}\n\n"
                "Write ONE short paragraph describing what this combination of macro "
                "factors and commodity moves likely means for the stock PRICE going "
                "forward. Where relevant, mention how the related companies above "
                "(suppliers, customers, competitors) amplify or reduce these risks. "
                "Do not repeat raw numbers — focus on implications for the stock price."
            )
            summary_response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
            content = getattr(summary_response, "content", None)
            if isinstance(content, str) and content.strip():
                macro_summary = content.strip()
        except Exception as exc:
            logger.warning(
                f"[Node 6] Macro summary LLM call failed for {ticker}: {exc}"
            )

        # ------------------------------------------------------------------
        # Step 6: Assemble market_context output
        # ------------------------------------------------------------------
        stock_classification = {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "exchange": exchange,
            "index_memberships": index_memberships,
            "market_cap_tier": market_cap_tier,
            "beta_fmp": beta_fmp,
        }

        market_regime = {
            "regime_label": regime_info["regime_label"],
            "regime_description": regime_info["regime_description"],
            "spy_return_1d": spy_trend_data["spy_return_1d"],
            "spy_return_5d": spy_trend_data["spy_return_5d"],
            "spy_return_21d": spy_trend_data["spy_return_21d"],
            "spy_trend_label": spy_trend_data["spy_trend_label"],
            "vix_level": vix_info["vix_level"],
            "vix_category": vix_info["vix_category"],
            "vix_interpretation": vix_info["vix_interpretation"],
            "market_volatility_pct": spy_trend_data["market_volatility_pct"],
        }

        sector_context_note = (
            "Stock is showing relative strength versus its sector."
            if rs_info["relative_strength_label"] == "LEADING"
            else "Stock is lagging its sector; sector strength may not be flowing into this name."
            if rs_info["relative_strength_label"] == "LAGGING"
            else "Stock is broadly moving in line with its sector."
        )
        sector_industry_context = {
            "sector_return_5d": sector_return_5d,
            "sector_trend": sector_trend,
            "industry_return_5d": industry_return_5d,
            "industry_trend": industry_trend,
            "stock_vs_sector_5d": rs_info["stock_vs_sector_5d"],
            "stock_vs_sector_21d": rs_info["stock_vs_sector_21d"],
            "relative_strength_label": rs_info["relative_strength_label"],
            "sector_context_note": sector_context_note,
        }

        macro_factor_exposure = {
            "identified_factors": identified_factors,
            "commodity_prices": commodity_prices,
            "commodity_trends": commodity_trends,
            "macro_summary": macro_summary,
        }

        market_correlation_profile = {
            "market_correlation": correlation_data["market_correlation"],
            "correlation_strength": correlation_data["correlation_strength"],
            "beta_calculated": beta_calculated,
            "beta_interpretation": beta_interpretation,
            "correlation_note": corr_note,
        }

        news_sentiment_context = {
            "market_news_sentiment": sentiment_score,
            "market_news_count": news_data["market_news_count"],
            "sentiment_label": sentiment_label,
            "sentiment_interpretation": sentiment_interpretation,
        }

        market_context = {
            "stock_classification":       stock_classification,
            "market_regime":              market_regime,
            "sector_industry_context":    sector_industry_context,
            "macro_factor_exposure":      macro_factor_exposure,
            "market_correlation_profile": market_correlation_profile,
            "news_sentiment_context":     news_sentiment_context,
        }

        elapsed = time.time() - start
        logger.info(
            f"[Node 6] Completed for {ticker} in {elapsed:.2f}s — "
            f"regime={market_regime['regime_label']} "
            f"rel_strength={sector_industry_context['relative_strength_label']} "
            f"news_count={news_sentiment_context['market_news_count']}"
        )
        return {
            "market_context": market_context,
            "node_execution_times": {"node_6": elapsed},
        }

    except Exception as exc:
        elapsed = time.time() - start
        logger.error(f"[Node 6] Failed for {ticker}: {exc}", exc_info=True)
        return _build_fallback_market_context(
            ticker, elapsed, f"Node 6: market context failed — {str(exc)}"
        )
