"""
Node 6: Market Context Analysis

Provides the macro "zoom out" view — is the whole market on fire while this
stock looks locally fine? Combines five independent layers into a single
continuous headwind/tailwind score on [-1, +1].

Layers (weighted composite):
1. SPY multi-timeframe (1d / 5d / 21d) — 35 %   price trend across short, medium, long windows
2. VIX fear gauge                        — 25 %   market-wide fear level
3. Sector ETF performance (5-day)        — 20 %   sector health
4. Related-company peer performance      — 10 %   peers moving together?
5. Market news sentiment (last 14 days)  — 10 %   narrative from cleaned_market_news

Key fixes vs previous version:
- Correlation NaN bug fixed: numpy corrcoef on .values avoids timezone index mismatch
- Sector now uses 5-day window (1-day is noise)
- VIX added as a direct fear proxy
- Market news from state['cleaned_market_news'] (was ignored before)
- Output: market_headwind_score on [-1,+1]; context_signal BUY/SELL/HOLD kept for
  Node 12 backward compatibility

Runs in PARALLEL with: Nodes 4, 5, 7
Runs AFTER:  Node 9A (cleaned_market_news available)
Runs BEFORE: Node 8 (news verification)
"""

import math
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

SECTOR_ETFS: Dict[str, str] = {
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
}

MARKET_INDICES: Dict[str, str] = {
    "S&P 500": "SPY",
    "NASDAQ":  "QQQ",
    "Dow Jones": "DIA",
}

# Weights for headwind/tailwind composite (must sum to 1.0)
W_SPY   = 0.35
W_VIX   = 0.25
W_SECTOR = 0.20
W_PEERS  = 0.10
W_NEWS   = 0.10

# Thresholds for context_signal (backward compat with Node 12)
BUY_THRESHOLD:  float =  0.30
SELL_THRESHOLD: float = -0.30

# How many days of market news to include in sentiment average
MARKET_NEWS_LOOKBACK_DAYS: int = 14

# Alpha Vantage sentiment normalization divisor
# AV overall_sentiment_score ~ [-0.35, +0.35]; divide by this to map to [-1, +1]
AV_SENTIMENT_NORM: float = 0.35


# ============================================================================
# HELPER 1: Get Stock Sector
# ============================================================================

def get_stock_sector(ticker: str) -> Tuple[str, str]:
    """
    Get the sector and industry for a stock using yfinance.

    Args:
        ticker: Stock symbol (e.g., 'NVDA').

    Returns:
        (sector, industry) — 'Unknown' when unavailable.

    Example:
        >>> sector, industry = get_stock_sector('NVDA')
        >>> print(sector)
        'Technology'
    """
    try:
        info = yf.Ticker(ticker).info
        sector   = info.get("sector",   "Unknown")
        industry = info.get("industry", "Unknown")
        logger.info(f"  Sector lookup: {ticker} → {sector} / {industry}")
        return sector, industry
    except Exception as exc:
        logger.warning(f"  Sector lookup failed for {ticker}: {exc}")
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
# HELPER 5: Related Companies (peers)
# ============================================================================

def analyze_related_companies(related_tickers: List[str]) -> Dict[str, Any]:
    """
    Measure 1-day performance of related peers.

    Args:
        related_tickers: List of ticker symbols from Node 3.

    Returns:
        {
            'related_companies':        list,
            'avg_performance':          float,
            'up_count':                 int,
            'down_count':               int,
            'overall_signal':           str,    # BULLISH | BEARISH | NEUTRAL
            'peers_score':              float,  # contribution on [-1, +1]
        }
    """
    _default = {
        "related_companies": [],
        "avg_performance":   0.0,
        "up_count":          0,
        "down_count":        0,
        "overall_signal":    "NEUTRAL",
        "peers_score":       0.0,
    }

    if not related_tickers:
        return _default

    try:
        records: List[Dict[str, Any]] = []

        for tkr in related_tickers:
            try:
                hist = yf.Ticker(tkr).history(period="2d")
                if len(hist) >= 2:
                    prev    = float(hist["Close"].iloc[-2])
                    current = float(hist["Close"].iloc[-1])
                    perf    = (current - prev) / prev * 100
                    direction = "UP" if perf > 0.5 else ("DOWN" if perf < -0.5 else "FLAT")
                    records.append({"ticker": tkr, "performance": perf, "trend": direction})
            except Exception:
                continue

        if not records:
            return _default

        perfs       = [r["performance"] for r in records]
        avg_perf    = sum(perfs) / len(perfs)
        up_count    = sum(1 for r in records if r["trend"] == "UP")
        down_count  = sum(1 for r in records if r["trend"] == "DOWN")

        if avg_perf >  1.0:
            overall, score = "BULLISH",  1.0
        elif avg_perf > 0.0:
            overall, score = "BULLISH",  0.5
        elif avg_perf > -1.0:
            overall, score = "NEUTRAL",  0.0 if avg_perf > -0.5 else -0.5
        else:
            overall, score = "BEARISH", -1.0

        logger.info(
            f"  Peers: {up_count} up / {down_count} down, avg={avg_perf:+.2f}% → {overall}"
        )
        return {
            "related_companies": records,
            "avg_performance":   avg_perf,
            "up_count":          up_count,
            "down_count":        down_count,
            "overall_signal":    overall,
            "peers_score":       score,
        }

    except Exception as exc:
        logger.warning(f"  Related companies analysis failed: {exc}")
        return _default


# ============================================================================
# HELPER 6: Stock–Market Correlation (NaN-safe)
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

        # --- stock returns (last 30 rows) ---
        stock_close   = price_data["close"].dropna()
        stock_returns = stock_close.pct_change().dropna().tail(30)

        if len(stock_returns) < 10:
            logger.warning("  Correlation: too few stock return points")
            return _default

        # --- SPY returns ---
        spy_hist    = yf.Ticker("SPY").history(period="35d")
        if spy_hist.empty or len(spy_hist) < 10:
            logger.warning("  Correlation: insufficient SPY data")
            return _default

        spy_returns = spy_hist["Close"].pct_change().dropna().tail(30)

        # --- align by length using numpy (avoids DatetimeIndex timezone issues) ---
        min_len = min(len(stock_returns), len(spy_returns))
        s_arr   = stock_returns.values[-min_len:]
        m_arr   = spy_returns.values[-min_len:]

        if min_len < 10:
            logger.warning("  Correlation: too few aligned points")
            return _default

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
# HELPER 8: Composite Headwind / Tailwind Score
# ============================================================================

def compute_headwind_tailwind_score(
    spy_composite:    float,
    vix_contribution: float,
    sector_score:     float,
    peers_score:      float,
    news_sentiment:   float,
) -> Dict[str, Any]:
    """
    Combine all five layers into a single market headwind/tailwind score.

    Score on [-1, +1]:
        +1.0  strong tailwind   (everything is green)
        +0.3  mild tailwind
         0.0  neutral
        -0.3  mild headwind
        -1.0  strong headwind   (market is burning)

    context_signal is BUY / SELL / HOLD (kept for Node 12 backward compat).
    confidence is abs(score) × 100.

    Args:
        spy_composite:    Weighted SPY multi-timeframe score [-1, +1].
        vix_contribution: VIX fear contribution [-1, +0.3].
        sector_score:     Sector ETF 5d score [-1, +1].
        peers_score:      Related companies score [-1, +1].
        news_sentiment:   Market news normalised sentiment [-1, +1].

    Returns:
        {
            'market_headwind_score': float,
            'context_signal':        str,
            'confidence':            float,  # 0-100
        }
    """
    raw = (
        W_SPY    * spy_composite
        + W_VIX  * vix_contribution
        + W_SECTOR * sector_score
        + W_PEERS  * peers_score
        + W_NEWS   * news_sentiment
    )
    score = max(-1.0, min(1.0, raw))

    if score >= BUY_THRESHOLD:
        context_signal = "BUY"
    elif score <= SELL_THRESHOLD:
        context_signal = "SELL"
    else:
        context_signal = "HOLD"

    confidence = round(abs(score) * 100, 2)

    logger.info(
        f"  Headwind score: SPY={spy_composite:+.3f} VIX={vix_contribution:+.3f} "
        f"sector={sector_score:+.3f} peers={peers_score:+.3f} news={news_sentiment:+.3f} "
        f"→ raw={raw:+.4f} → {score:+.3f} ({context_signal})"
    )
    return {
        "market_headwind_score": score,
        "context_signal":        context_signal,
        "confidence":            confidence,
    }


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def market_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 6: Market Context Analysis — five-layer headwind/tailwind composite.

    Execution flow:
    1. Get stock sector (yfinance)
    2. Fetch VIX fear gauge
    3. Fetch SPY multi-timeframe performance (1d/5d/21d)
    4. Fetch sector ETF performance (5-day)
    5. Fetch related-company peer performance
    6. Calculate stock–SPY correlation (numpy, NaN-safe)
    7. Aggregate market news sentiment from cleaned_market_news
    8. Compute composite headwind/tailwind score
    9. Return partial state update

    Runs in PARALLEL with: Nodes 4, 5, 7
    Runs AFTER:  Node 9A
    Runs BEFORE: Node 8

    Args:
        state: LangGraph state dict.

    Returns:
        Partial state update with market_context populated.
    """
    start_time = datetime.now()
    ticker: str = state["ticker"]

    logger.info(f"Node 6: Market context analysis for {ticker}")

    try:
        # ====================================================================
        # STEP 1: Sector
        # ====================================================================
        sector, industry = get_stock_sector(ticker)

        # ====================================================================
        # STEP 2: VIX
        # ====================================================================
        vix_data = get_vix_level()

        # ====================================================================
        # STEP 3: SPY multi-timeframe
        # ====================================================================
        spy_data = get_market_trend_multitimeframe()

        # ====================================================================
        # STEP 4: Sector ETF (5-day)
        # ====================================================================
        sector_data = get_sector_performance(sector, days=5)

        # ====================================================================
        # STEP 5: Related companies
        # ====================================================================
        related_tickers: List[str] = state.get("related_companies") or []
        peers_data = analyze_related_companies(related_tickers)

        # ====================================================================
        # STEP 6: Stock–market correlation (NaN-safe)
        # ====================================================================
        price_data: Optional[pd.DataFrame] = state.get("raw_price_data")
        if price_data is not None and not price_data.empty:
            corr_data = calculate_correlation(ticker, price_data)
        else:
            logger.warning("  Correlation: no price_data in state, using defaults")
            corr_data = {"market_correlation": 0.5, "correlation_strength": "MEDIUM", "beta": 1.0}

        # ====================================================================
        # STEP 7: Market news sentiment
        # ====================================================================
        cleaned_market_news: List[Dict[str, Any]] = state.get("cleaned_market_news") or []
        news_data = analyze_market_news_sentiment(cleaned_market_news)

        # ====================================================================
        # STEP 8: Composite headwind / tailwind score
        # ====================================================================
        signal_data = compute_headwind_tailwind_score(
            spy_composite    = spy_data["spy_composite_score"],
            vix_contribution = vix_data["vix_contribution"],
            sector_score     = sector_data["sector_score"],
            peers_score      = peers_data["peers_score"],
            news_sentiment   = news_data["market_news_sentiment"],
        )

        # ====================================================================
        # STEP 9: Build results dict
        # ====================================================================
        results: Dict[str, Any] = {
            # Sector
            "sector":             sector,
            "industry":           industry,
            "sector_performance": sector_data["performance"],
            "sector_trend":       sector_data["trend"],
            # SPY
            "market_trend":       spy_data["market_trend"],
            "market_performance": spy_data["market_performance"],
            "spy_return_1d":      spy_data["spy_return_1d"],
            "spy_return_5d":      spy_data["spy_return_5d"],
            "spy_return_21d":     spy_data["spy_return_21d"],
            "volatility":         spy_data["volatility"],
            # VIX
            "vix_level":          vix_data["vix_level"],
            "vix_category":       vix_data["vix_category"],
            # Peers
            "related_companies_signals": peers_data["related_companies"],
            "related_companies_avg":     peers_data["avg_performance"],
            "related_companies_signal":  peers_data["overall_signal"],
            # Correlation
            "market_correlation":   corr_data["market_correlation"],
            "correlation_strength": corr_data["correlation_strength"],
            "beta":                 corr_data["beta"],
            # Market news
            "market_news_sentiment": news_data["market_news_sentiment"],
            "market_news_count":     news_data["market_news_count"],
            # Signal (backward compat + new headwind score)
            "market_headwind_score": signal_data["market_headwind_score"],
            "context_signal":        signal_data["context_signal"],
            "confidence":            signal_data["confidence"],
        }

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Node 6: completed in {elapsed:.2f}s — "
            f"headwind={signal_data['market_headwind_score']:+.3f} "
            f"({signal_data['context_signal']}) "
            f"VIX={vix_data['vix_level']:.1f}/{vix_data['vix_category']} "
            f"SPY={spy_data['spy_return_5d']:+.2f}%5d "
            f"sector={sector_data['performance']:+.2f}%5d "
            f"news_articles={news_data['market_news_count']}"
        )

        return {
            "market_context":       results,
            "node_execution_times": {"node_6": elapsed},
        }

    except Exception as exc:
        logger.error(f"Node 6: failed for {ticker}: {exc}")
        elapsed = (datetime.now() - start_time).total_seconds()
        return {
            "errors": [f"Node 6: market context analysis failed — {exc}"],
            "market_context": {
                "sector":                "Unknown",
                "industry":              "Unknown",
                "sector_performance":    0.0,
                "sector_trend":          "FLAT",
                "market_trend":          "NEUTRAL",
                "market_performance":    0.0,
                "spy_return_1d":         0.0,
                "spy_return_5d":         0.0,
                "spy_return_21d":        0.0,
                "volatility":            0.0,
                "vix_level":             20.0,
                "vix_category":          "MODERATE",
                "related_companies_signals": [],
                "related_companies_avg": 0.0,
                "related_companies_signal": "NEUTRAL",
                "market_correlation":    0.5,
                "correlation_strength":  "MEDIUM",
                "beta":                  1.0,
                "market_news_sentiment": 0.0,
                "market_news_count":     0,
                "market_headwind_score": 0.0,
                "context_signal":        "HOLD",
                "confidence":            0.0,
            },
            "node_execution_times": {"node_6": elapsed},
        }
