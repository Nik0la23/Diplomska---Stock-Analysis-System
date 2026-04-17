"""
Node 10: Backtesting

Backtests 4 signal streams over the last 180 days using cached DB data.
Prepares RAW accuracy metrics for Node 11 (adaptive weighting).

IMPORTANT DESIGN CONTRACT:
- This node does NOT apply any weighting — all weighting logic belongs to Node 11.
- Returns separate full-period accuracy AND recent-period accuracy for each stream.
- Node 11 will apply recency weighting (e.g. 70% recent + 30% full).

The four signal streams backtested:
1. Technical Analysis  — reconstructed from historical price slices
2. Stock News Sentiment — aggregated from cached AV scores in DB
3. Market News Sentiment — aggregated from cached AV scores in DB
4. Related News Sentiment — aggregated from cached AV scores in DB

HOLD Threshold:
- Volatility-adaptive per stock: ±(avg_abs_7day_return * 0.5)
- Fallback: ±2.0% if insufficient price data
- Defined once, used by all streams — consistent correctness evaluation

Improvements over original:
1. Credibility-weighted sentiment aggregation (aggregate_daily_sentiment_weighted):
   Articles are weighted by source accuracy_rate from Node 8's source_reliability.
   Sources with <10 articles tracked get neutral weight (0.5). This prevents
   high-volume low-quality sources (e.g. Stock Traders Daily, 22.7% accuracy)
   from dominating the daily sentiment score.

2. Baseline correction (sentiment momentum):
   A 20-day rolling mean is computed per stream before the evaluation loop.
   Each day's raw sentiment is adjusted by subtracting the rolling baseline,
   converting the absolute (positively-biased) signal into a relative one.
   This directly fixes the 112 BUY / 1 SELL imbalance caused by financial
   news being structurally positive.

3. Magnitude threshold (replaces exact == 0.0 filter):
   Days are skipped when abs(corrected_score) < MIN_SENTIMENT_MAGNITUDE (0.05)
   rather than only when score == 0.0 exactly. Combined with the baseline
   correction, this allows more market news days to survive (from 6-9 to ~30+)
   while still filtering genuinely uninformative days.

4. Recency-decayed IC (calculate_news_ic_decay):
   A half-life of 60 days exponentially down-weights older observations.
   Sentiment-price relationships drift; recent data is more representative.
   Original Pearson IC (calculate_news_ic) is preserved alongside for
   comparison. Both are stored in stream metrics.

5. Tiered sufficiency / binomial-test thresholds:
   MIN_SIGNALS_FOR_SUFFICIENCY (10) controls whether a stream gets any weight
   at all. MIN_SIGNALS_FOR_BINOMIAL (20) controls whether the binomial
   significance test is run. Streams with 10-19 signals receive proportional
   weight based on raw directional accuracy, but are never falsely marked
   is_anti_predictive because the test lacks statistical power at that sample
   size. The Bayesian smoothing in Node 11 handles uncertainty from small n.

Runs AFTER: Node 9B (behavioral anomaly detection)
Runs BEFORE: Node 11 (adaptive weights calculation)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import binomtest, pearsonr

from src.database.db_manager import get_news_with_outcomes
from src.langgraph_nodes.node_04_technical_analysis import (
    analyze_volume,
    calculate_adx,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
    calculate_technical_score,
)
from src.utils.logger import get_node_logger

logger = get_node_logger("node_10")


# ============================================================================
# CONSTANTS
# ============================================================================

HOLD_THRESHOLD_MULTIPLIER: float = 0.5    # hold_threshold = avg_7day_abs_return * this
HOLD_THRESHOLD_FALLBACK_PCT: float = 2.0  # fallback when insufficient price data
HOLD_THRESHOLD_MIN_PCT: float = 0.5       # sanity floor
HOLD_THRESHOLD_MAX_PCT: float = 8.0       # sanity ceiling

MIN_PRICE_ROWS_FOR_TECHNICAL: int = 50    # Node 4 requires ≥50 rows for moving averages
ALPHA_SIGNAL_THRESHOLD: float = 0.10      # |alpha| < this → HOLD in technical reconstruction

# --- Tiered signal thresholds (Change 5) ---
# MIN_SIGNALS_FOR_SUFFICIENCY: stream must have at least this many directional
# signals (BUY+SELL) to receive ANY weight in Node 11. Lowered from 20 → 10
# so sparse but real streams (market news n=8, related news n=10) are not
# silently discarded. Node 11's Bayesian smoothing handles the uncertainty.
MIN_SIGNALS_FOR_SUFFICIENCY: int = 10

# MIN_SIGNALS_FOR_BINOMIAL: minimum signals before running the binomial
# significance test. With fewer than 20 signals, the test cannot detect
# accuracy above 70% at p<0.05 even if the stream is genuinely predictive.
# Streams below this threshold get is_significant=False, is_anti_predictive=False
# (near-random label) — they receive proportional weight based on raw accuracy
# rather than being falsely penalized as anti-predictive.
MIN_SIGNALS_FOR_BINOMIAL: int = 20

RECENT_PERIOD_DAYS: int = 60             # "recent" window in days (provided raw to Node 11)
MIN_RECENT_DAYS_FOR_ACCURACY: int = 3    # minimum recent days before reporting recent_accuracy
                                          # lowered from 5 → 3 for sparse streams

# Improvement 3: magnitude threshold — replaces exact == 0.0 filter.
MIN_SENTIMENT_MAGNITUDE: float = 0.05

# Improvement 2: rolling baseline window for sentiment momentum correction.
BASELINE_WINDOW: int = 20
MIN_BASELINE_DAYS: int = 5              # below this, no correction applied (cold start)

# Improvement 4: half-life in days for exponential recency decay in IC.
IC_DECAY_HALF_LIFE: int = 60

# Minimum number of daily samples (dates with outcomes) required to compute IC.
# Lowered from 10 to 5 to handle sparse streams (e.g. market news clusters on few dates).
# Pearson r is still meaningful at n=5–9; Node 11 Bayesian smoothing handles uncertainty.
MIN_IC_SAMPLES: int = 5


# ============================================================================
# HELPER 1: HOLD THRESHOLD
# ============================================================================

def calculate_hold_threshold(price_data: pd.DataFrame) -> float:
    """
    Calculate the volatility-adaptive HOLD threshold for this stock.

    HOLD is defined as: |7-day price change| <= hold_threshold.

    Different stocks have different volatility, so a single fixed threshold
    would be unfair: TSLA with ±6% average moves would almost never generate
    a directional signal, while JNJ with ±1.5% moves would always be directional.

    Formula:
        avg_abs_7day_return = mean(|close.pct_change(7)|) * 100
        hold_threshold = avg_abs_7day_return * HOLD_THRESHOLD_MULTIPLIER

    Args:
        price_data: DataFrame with 'close' and 'date' columns (from Node 1)

    Returns:
        HOLD threshold as percentage (e.g. 2.3 means ±2.3% is HOLD zone)
    """
    try:
        if price_data is None or len(price_data) < 30:
            logger.warning(
                f"Insufficient price data for HOLD threshold "
                f"(need 30, got {len(price_data) if price_data is not None else 0}), "
                f"using fallback {HOLD_THRESHOLD_FALLBACK_PCT}%"
            )
            return HOLD_THRESHOLD_FALLBACK_PCT

        df = price_data.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("date").reset_index(drop=True)
        returns_7d = df["close"].pct_change(7).abs() * 100
        returns_7d = returns_7d.dropna()

        if len(returns_7d) < 10:
            logger.warning(
                f"Too few valid 7-day returns ({len(returns_7d)}), "
                f"using fallback {HOLD_THRESHOLD_FALLBACK_PCT}%"
            )
            return HOLD_THRESHOLD_FALLBACK_PCT

        avg_abs_return = float(returns_7d.mean())
        hold_threshold = avg_abs_return * HOLD_THRESHOLD_MULTIPLIER
        hold_threshold = max(HOLD_THRESHOLD_MIN_PCT, min(HOLD_THRESHOLD_MAX_PCT, hold_threshold))

        logger.info(
            f"HOLD threshold: {hold_threshold:.2f}% "
            f"(avg 7-day abs move: {avg_abs_return:.2f}% × {HOLD_THRESHOLD_MULTIPLIER})"
        )
        return hold_threshold

    except Exception as e:
        logger.error(f"HOLD threshold calculation failed: {e}")
        return HOLD_THRESHOLD_FALLBACK_PCT


# ============================================================================
# HELPER 2: SIGNAL CORRECTNESS EVALUATION
# ============================================================================

def evaluate_outcome(signal: str, actual_change: float, hold_threshold: float) -> bool:
    """
    Determine whether a historical signal matched the actual price outcome.

    BUY/SELL use DIRECTIONAL agreement (any positive/negative move is correct).
    HOLD uses the stock-specific threshold.

    Args:
        signal: 'BUY', 'SELL', or 'HOLD'
        actual_change: Actual 7-calendar-day price change in percent
        hold_threshold: Stock-specific HOLD zone width in percent (used for HOLD only)

    Returns:
        True if the signal matched the actual outcome
    """
    if signal == "BUY":
        return actual_change > 0.0
    elif signal == "SELL":
        return actual_change < 0.0
    elif signal == "HOLD":
        return abs(actual_change) <= hold_threshold
    return False


# ============================================================================
# HELPER 3: TECHNICAL SIGNAL RECONSTRUCTION
# ============================================================================

def reconstruct_technical_signal(
    price_slice: pd.DataFrame,
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Reconstruct the technical signal for the last day of a historical price slice.

    Reuses Node 4's helper functions directly — Node 4 itself is not called.

    Args:
        price_slice: DataFrame with 'close', 'open', 'high', 'low', 'volume' columns.

    Returns:
        Tuple (signal, confidence, alpha) — ('BUY'/'SELL'/'HOLD', 0.0-1.0, signed float)
        Returns (None, None, None) if the slice has fewer than MIN_PRICE_ROWS_FOR_TECHNICAL rows.
        alpha is the raw signed Ridge regression output — positive = bullish, negative = bearish.
    """
    if price_slice is None or len(price_slice) < MIN_PRICE_ROWS_FOR_TECHNICAL:
        return None, None, None

    try:
        regression_slice = (
            price_slice.iloc[:-7]
            if len(price_slice) > MIN_PRICE_ROWS_FOR_TECHNICAL + 7
            else price_slice
        )

        rsi        = calculate_rsi(price_slice)
        macd       = calculate_macd(price_slice)
        bollinger  = calculate_bollinger_bands(price_slice)
        moving_avg = calculate_moving_averages(price_slice)
        volume     = analyze_volume(price_slice)
        adx        = calculate_adx(price_slice)

        score_result = calculate_technical_score(
            rsi, macd, bollinger, moving_avg, volume,
            price_data=regression_slice,
            adx=adx,
        )

        alpha = score_result['technical_alpha']

        if alpha >= ALPHA_SIGNAL_THRESHOLD:
            signal     = 'BUY'
            confidence = float(min(alpha, 1.0))
        elif alpha <= -ALPHA_SIGNAL_THRESHOLD:
            signal     = 'SELL'
            confidence = float(min(-alpha, 1.0))
        else:
            signal     = 'HOLD'
            confidence = float(1.0 - abs(alpha) / ALPHA_SIGNAL_THRESHOLD)

        return signal, float(confidence), float(alpha)

    except Exception as e:
        logger.debug(f"Technical signal reconstruction failed for {len(price_slice)}-row slice: {e}")
        return None, None, None


# ============================================================================
# HELPER 4: TECHNICAL STREAM BACKTEST
# ============================================================================

def backtest_technical_stream(
    price_data: pd.DataFrame,
    hold_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Backtest technical analysis signals over 180 days of price history.

    For each day D from row 50 onward:
        1. Slice price_data up to and including day D
        2. Reconstruct what Node 4 would have signalled on that day
        3. Find the actual close price 7 calendar days later
        4. Evaluate whether the signal was correct

    Args:
        price_data: Full OHLCV DataFrame from state (Node 1 output)
        hold_threshold: Stock-specific HOLD zone width in percent

    Returns:
        List of dicts: {date, signal, confidence, sentiment_score, actual_change_7d, correct, days_ago}
        sentiment_score is set to the signed technical_alpha so calculate_news_ic() can compute
        the out-of-sample rolling IC using the same code path as news streams.
    """
    results: List[Dict[str, Any]] = []

    if price_data is None or len(price_data) < MIN_PRICE_ROWS_FOR_TECHNICAL + 7:
        logger.warning(
            f"Insufficient price data for technical backtest "
            f"(need {MIN_PRICE_ROWS_FOR_TECHNICAL + 7}, "
            f"got {len(price_data) if price_data is not None else 0})"
        )
        return results

    try:
        df = price_data.copy().sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        date_to_close: Dict[str, float] = dict(zip(df["date"], df["close"]))
        sorted_dates = sorted(df["date"].tolist())
        today = pd.Timestamp.now().normalize()

        for i in range(MIN_PRICE_ROWS_FOR_TECHNICAL, len(df)):
            date_str = str(df.iloc[i]["date"])

            target_future = (
                pd.to_datetime(date_str) + pd.Timedelta(days=7)
            ).strftime("%Y-%m-%d")
            future_dates = [d for d in sorted_dates if d >= target_future]
            if not future_dates:
                continue

            price_today   = float(df.iloc[i]["close"])
            price_future  = float(date_to_close[future_dates[0]])
            actual_change = ((price_future - price_today) / price_today) * 100

            price_slice = df.iloc[: i + 1].copy()
            signal, confidence, alpha = reconstruct_technical_signal(price_slice)

            if signal is None:
                continue

            correct  = evaluate_outcome(signal, actual_change, hold_threshold)
            days_ago = int((today - pd.to_datetime(date_str)).days)

            results.append({
                "date":             date_str,
                "signal":           signal,
                "confidence":       round(confidence, 4) if confidence is not None else None,
                "sentiment_score":  round(alpha, 4) if alpha is not None else None,
                "actual_change_7d": round(actual_change, 4),
                "correct":          correct,
                "days_ago":         days_ago,
            })

        logger.info(f"Technical backtest: {len(results)} days evaluated")
        return results

    except Exception as e:
        logger.error(f"Technical backtest stream failed: {e}")
        return []


# ============================================================================
# HELPER 5A: ORIGINAL DAILY SENTIMENT AGGREGATION (preserved, used as fallback)
# ============================================================================

def aggregate_daily_sentiment(articles: List[Dict[str, Any]]) -> float:
    """
    Unweighted equal-average sentiment aggregation.
    Preserved for the combined stream backtest and as a fallback.

    Args:
        articles: List of article dicts with 'sentiment_label' and 'sentiment_score'

    Returns:
        Combined directional score in [-1.0, +1.0]. 0.0 if no articles.
    """
    if not articles:
        return 0.0

    signed_scores: List[float] = []
    for article in articles:
        label     = article.get("sentiment_label", "neutral") or "neutral"
        raw_score = article.get("sentiment_score")
        score     = float(raw_score) if raw_score is not None else 0.0

        if label == "positive":
            signed_scores.append(abs(score))
        elif label == "negative":
            signed_scores.append(-abs(score))
        else:
            signed_scores.append(0.0)

    return float(np.mean(signed_scores)) if signed_scores else 0.0


# ============================================================================
# HELPER 5B: CREDIBILITY-WEIGHTED SENTIMENT AGGREGATION (Improvement 1)
# ============================================================================

def aggregate_daily_sentiment_weighted(
    articles: List[Dict[str, Any]],
    source_reliability: Dict[str, Any],
) -> float:
    """
    Aggregate sentiment scores weighted by source credibility.

    Each article's signed sentiment is multiplied by its source's historical
    accuracy_rate from Node 8's source_reliability dict before averaging.

    Weighting rules:
    - Source with ≥10 tracked articles: use accuracy_rate as weight
    - Source with <10 articles or unknown: use neutral weight 0.5

    Args:
        articles:           List of article dicts with 'sentiment_label',
                            'sentiment_score', and 'source'.
        source_reliability: Node 8 source_reliability dict. Pass {} for equal weights.

    Returns:
        Credibility-weighted directional score in [-1.0, +1.0]. 0.0 if no articles.
    """
    if not articles:
        return 0.0

    if not source_reliability:
        return aggregate_daily_sentiment(articles)

    weighted_sum = 0.0
    total_weight = 0.0

    for article in articles:
        label     = article.get("sentiment_label", "neutral") or "neutral"
        raw_score = article.get("sentiment_score")
        score     = float(raw_score) if raw_score is not None else 0.0
        source    = str(article.get("source", "")).strip()

        src_data   = source_reliability.get(source, {})
        n_articles = int(src_data.get("total_articles", 0))
        cred = float(src_data.get("accuracy_rate", 0.5)) if src_data and n_articles >= 10 else 0.5

        if label == "positive":
            signed = abs(score)
        elif label == "negative":
            signed = -abs(score)
        else:
            signed = 0.0

        weighted_sum += signed * cred
        total_weight += cred

    return weighted_sum / total_weight if total_weight > 0 else 0.0


# ============================================================================
# HELPER 5C: SENTIMENT BASELINE BUILDER (Improvement 2)
# ============================================================================

def build_sentiment_baseline(
    by_date: Dict[str, List[Dict[str, Any]]],
    source_reliability: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute a rolling 20-day sentiment baseline per date for momentum correction.

    Subtracting the rolling baseline converts the absolute (positively-biased)
    sentiment level into a relative signal: is today better or worse than
    the recent average? This fixes the structural 112 BUY / 1 SELL imbalance.

    Args:
        by_date:            Dict mapping date strings to lists of article dicts.
        source_reliability: Source reliability dict from Node 8.

    Returns:
        Dict mapping each date string to its rolling baseline (0.0 = no correction).
    """
    all_dates_sorted = sorted(by_date.keys())

    raw_scores: Dict[str, float] = {
        d: aggregate_daily_sentiment_weighted(by_date[d], source_reliability)
        for d in all_dates_sorted
    }

    baseline_by_date: Dict[str, float] = {}
    for i, date_str in enumerate(all_dates_sorted):
        prior_dates = all_dates_sorted[max(0, i - BASELINE_WINDOW):i]
        if len(prior_dates) >= MIN_BASELINE_DAYS:
            baseline_by_date[date_str] = float(
                np.mean([raw_scores[d] for d in prior_dates])
            )
        else:
            baseline_by_date[date_str] = 0.0

    return baseline_by_date


# ============================================================================
# HELPER 6: SENTIMENT STREAM BACKTEST
# ============================================================================

def _get_spy_price_nearest(
    spy_prices: Dict[str, float],
    date_str: str,
    offset_days: int = 7,
) -> Optional[float]:
    """
    Return the SPY close price closest to (date_str + offset_days).
    Tries exact date first, then looks forward up to 5 calendar days.
    """
    from datetime import date as _date, timedelta as _timedelta
    try:
        base   = datetime.strptime(date_str, "%Y-%m-%d").date()
        target = base + _timedelta(days=offset_days)
        for delta in range(6):
            candidate = str(target + _timedelta(days=delta))
            if candidate in spy_prices:
                return spy_prices[candidate]
        return None
    except Exception:
        return None


def backtest_sentiment_stream(
    news_events: List[Dict[str, Any]],
    news_type: str,
    hold_threshold: float,
    spy_daily_prices: Optional[Dict[str, float]] = None,
    beta: Optional[float] = None,
    source_reliability: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Backtest a single sentiment stream (stock, market, or related) over 180 days.

    Applied improvements:
    1. Credibility-weighted aggregation via aggregate_daily_sentiment_weighted()
    2. Baseline correction: corrected_score = raw_score - rolling_20d_mean
    3. Magnitude threshold: skip days where |corrected_score| < MIN_SENTIMENT_MAGNITUDE
    4. Market news return decomposition: actual_change = stock - (beta × SPY)

    Args:
        news_events:        get_news_with_outcomes() output as list of dicts.
        news_type:          'stock', 'market', or 'related'.
        hold_threshold:     Stock-specific HOLD zone width in percent.
        spy_daily_prices:   Optional SPY history for market-news decomposition.
        beta:               Stock beta for market-news decomposition.
        source_reliability: Node 8 source_reliability dict (None = equal weights).

    Returns:
        List of dicts per evaluated day:
        {date, signal, sentiment_score, raw_sentiment, baseline_used,
         article_count, actual_change_7d, correct, days_ago}
        Market news adds: decomposed, raw_stock_return_7d, spy_return_7d.
    """
    results: List[Dict[str, Any]] = []

    typed_events = [e for e in news_events if e.get("news_type") == news_type]
    if not typed_events:
        logger.warning(f"No historical events found for news_type='{news_type}'")
        return results

    src_rel: Dict[str, Any] = source_reliability or {}

    try:
        today = pd.Timestamp.now().normalize()

        by_date: Dict[str, List[Dict[str, Any]]] = {}
        for event in typed_events:
            pub = event.get("published_at", "")
            if not pub:
                continue
            date_str = str(pub)[:10]
            by_date.setdefault(date_str, []).append(event)

        baseline_by_date = build_sentiment_baseline(by_date, src_rel)
        skipped_low_magnitude = 0

        for date_str, day_articles in sorted(by_date.items()):

            price_changes = [
                float(a["price_change_7day"])
                for a in day_articles
                if a.get("price_change_7day") is not None
            ]
            if not price_changes:
                continue

            raw_stock_return = float(np.median(price_changes))
            raw_score        = aggregate_daily_sentiment_weighted(day_articles, src_rel)
            baseline         = baseline_by_date.get(date_str, 0.0)
            daily_score      = raw_score - baseline

            if abs(daily_score) < MIN_SENTIMENT_MAGNITUDE:
                skipped_low_magnitude += 1
                continue

            actual_change = raw_stock_return
            spy_return_7d: Optional[float] = None
            decomposed = False

            if news_type == "market" and spy_daily_prices and beta and beta > 0:
                spy_t0 = spy_daily_prices.get(date_str)
                spy_t7 = _get_spy_price_nearest(spy_daily_prices, date_str, offset_days=7)
                if spy_t0 and spy_t7 and spy_t0 > 0:
                    spy_return_7d = (spy_t7 - spy_t0) / spy_t0
                    actual_change = raw_stock_return - (beta * spy_return_7d)
                    decomposed    = True

            sentiment_bullish = daily_score > 0.0
            price_up          = actual_change > 0.0
            correct           = (sentiment_bullish and price_up) or (
                not sentiment_bullish and not price_up
            )

            signal   = "BUY" if sentiment_bullish else "SELL"
            days_ago = int((today - pd.to_datetime(date_str)).days)

            entry: Dict[str, Any] = {
                "date":             date_str,
                "signal":           signal,
                "sentiment_score":  round(daily_score, 4),
                "raw_sentiment":    round(raw_score, 4),
                "baseline_used":    round(baseline, 4),
                "article_count":    len(day_articles),
                "actual_change_7d": round(actual_change, 4),
                "correct":          correct,
                "days_ago":         days_ago,
            }
            if news_type == "market":
                entry["decomposed"]          = decomposed
                entry["raw_stock_return_7d"] = round(raw_stock_return, 4)
                if spy_return_7d is not None:
                    entry["spy_return_7d"] = round(spy_return_7d, 4)

            results.append(entry)

        logger.info(
            f"Sentiment backtest '{news_type}': {len(results)} days evaluated "
            f"(skipped {skipped_low_magnitude} low-magnitude days)"
        )
        return results

    except Exception as e:
        logger.error(f"Sentiment backtest for '{news_type}' failed: {e}")
        return []


# ============================================================================
# HELPER 7: COMBINED STREAM BACKTEST
# ============================================================================

def backtest_combined_stream(
    price_data: pd.DataFrame,
    news_events: List[Dict[str, Any]],
    hold_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Backtest all 4 streams combined with equal weights over the full price history.

    Uses the original unweighted aggregate_daily_sentiment() intentionally —
    the combined stream is a baseline reference and should not incorporate
    improvements that benefit individual stream evaluation.

    Args:
        price_data:     Full OHLCV DataFrame from state (sorted ascending).
        news_events:    get_news_with_outcomes() output as list of dicts.
        hold_threshold: Stock-specific HOLD zone width in percent.

    Returns:
        List of dicts: {date, signal, combined_score, streams_used,
                        actual_change_7d, correct, days_ago}
    """
    COMBINED_HOLD_THRESHOLD = 0.05
    results: List[Dict[str, Any]] = []

    if price_data is None or len(price_data) < MIN_PRICE_ROWS_FOR_TECHNICAL + 7:
        logger.warning("Combined backtest: insufficient price data")
        return results

    try:
        df = price_data.copy().sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        date_to_close: Dict[str, float] = dict(zip(df["date"], df["close"]))
        sorted_dates  = sorted(df["date"].tolist())
        today         = pd.Timestamp.now().normalize()

        by_date_stock:   Dict[str, List[Dict]] = {}
        by_date_market:  Dict[str, List[Dict]] = {}
        by_date_related: Dict[str, List[Dict]] = {}
        for event in news_events:
            pub = event.get("published_at", "")
            if not pub:
                continue
            date_str = str(pub)[:10]
            ntype    = event.get("news_type", "")
            if ntype == "stock":
                by_date_stock.setdefault(date_str, []).append(event)
            elif ntype == "market":
                by_date_market.setdefault(date_str, []).append(event)
            elif ntype == "related":
                by_date_related.setdefault(date_str, []).append(event)

        for i in range(MIN_PRICE_ROWS_FOR_TECHNICAL, len(df)):
            date_str = str(df.iloc[i]["date"])

            target_future = (
                pd.to_datetime(date_str) + pd.Timedelta(days=7)
            ).strftime("%Y-%m-%d")
            future_dates = [d for d in sorted_dates if d >= target_future]
            if not future_dates:
                continue

            price_today   = float(df.iloc[i]["close"])
            price_future  = float(date_to_close[future_dates[0]])
            actual_change = ((price_future - price_today) / price_today) * 100

            price_slice = df.iloc[: i + 1].copy()
            tech_signal, tech_conf, _ = reconstruct_technical_signal(price_slice)
            if tech_signal is None:
                continue

            if tech_signal == 'BUY':
                tech_score = tech_conf if tech_conf is not None else 0.0
            elif tech_signal == 'SELL':
                tech_score = -(tech_conf if tech_conf is not None else 0.0)
            else:
                tech_score = 0.0

            stream_scores: List[float] = [tech_score]
            streams_used = ['technical']

            for stype, smap in [
                ('stock',   by_date_stock),
                ('market',  by_date_market),
                ('related', by_date_related),
            ]:
                articles = smap.get(date_str)
                if articles:
                    s = aggregate_daily_sentiment(articles)
                    stream_scores.append(s)
                    streams_used.append(stype)

            combined_score = float(np.mean(stream_scores))

            if combined_score > COMBINED_HOLD_THRESHOLD:
                signal = 'BUY'
            elif combined_score < -COMBINED_HOLD_THRESHOLD:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            correct  = evaluate_outcome(signal, actual_change, hold_threshold)
            days_ago = int((today - pd.to_datetime(date_str)).days)

            results.append({
                "date":             date_str,
                "signal":           signal,
                "combined_score":   round(combined_score, 4),
                "streams_used":     streams_used,
                "actual_change_7d": round(actual_change, 4),
                "correct":          correct,
                "days_ago":         days_ago,
            })

        logger.info(f"Combined stream backtest: {len(results)} days evaluated")
        return results

    except Exception as e:
        logger.error(f"Combined stream backtest failed: {e}")
        return []


# ============================================================================
# HELPER 8A: STANDARD NEWS IC (Pearson correlation)
# ============================================================================

def calculate_news_ic(daily_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute the Information Coefficient for a single news sentiment stream.

    IC = Pearson correlation between the daily corrected sentiment score
    and actual_change_7d. Market news targets idiosyncratic return when
    decomposed; stock/related news targets raw stock return.

    Returns:
        Dict with ic_score, ic_p_value, ic_significant, ic_n_samples,
        ic_used_idiosyncratic. All numeric fields are None when n < 10.
    """
    _NO_IC: Dict[str, Any] = {
        "ic_score": None, "ic_p_value": None,
        "ic_significant": None, "ic_n_samples": 0,
        "ic_used_idiosyncratic": False,
    }

    valid = [
        r for r in daily_results
        if r.get("sentiment_score") is not None
        and r.get("actual_change_7d") is not None
    ]
    if len(valid) < MIN_IC_SAMPLES:
        return {**_NO_IC, "ic_n_samples": len(valid)}

    scores  = [float(r["sentiment_score"])  for r in valid]
    returns = [float(r["actual_change_7d"]) for r in valid]
    used_idiosyncratic = any(r.get("decomposed") for r in valid)

    try:
        ic, ic_p = pearsonr(scores, returns)
        if ic != ic:
            return {**_NO_IC, "ic_n_samples": len(valid)}
        return {
            "ic_score":              round(float(ic), 4),
            "ic_p_value":            round(float(ic_p), 4),
            "ic_significant":        float(ic_p) < 0.05,
            "ic_n_samples":          len(valid),
            "ic_used_idiosyncratic": used_idiosyncratic,
        }
    except Exception as e:
        logger.warning(f"News IC calculation failed: {e}")
        return _NO_IC


# ============================================================================
# HELPER 8B: RECENCY-DECAYED NEWS IC (Improvement 4)
# ============================================================================

def calculate_news_ic_decay(
    daily_results: List[Dict[str, Any]],
    half_life_days: int = IC_DECAY_HALF_LIFE,
) -> Dict[str, Any]:
    """
    Compute Information Coefficient with exponential recency decay.

    Decay formula: weight_i = exp(-ln(2) * days_ago_i / half_life_days)

    With half_life_days=60:
        Today:     weight = 1.00
        60d ago:   weight = 0.50
        120d ago:  weight = 0.25
        180d ago:  weight = 0.125

    Args:
        daily_results:  List of daily dicts from backtest_sentiment_stream().
        half_life_days: Decay half-life in days. Default 60.

    Returns:
        Dict with ic_decay_score, ic_decay_n_effective, ic_decay_half_life.
    """
    _NO_IC_DECAY: Dict[str, Any] = {
        "ic_decay_score":        None,
        "ic_decay_n_effective":  0,
        "ic_decay_half_life":    half_life_days,
    }

    valid = [
        r for r in daily_results
        if r.get("sentiment_score") is not None
        and r.get("actual_change_7d") is not None
        and r.get("days_ago") is not None
    ]
    if len(valid) < MIN_IC_SAMPLES:
        return {**_NO_IC_DECAY, "ic_decay_n_effective": len(valid)}

    days_arr    = np.array([float(r["days_ago"])         for r in valid])
    scores_arr  = np.array([float(r["sentiment_score"])  for r in valid])
    returns_arr = np.array([float(r["actual_change_7d"]) for r in valid])

    weights = np.exp(-np.log(2) * days_arr / half_life_days)
    weights = weights / weights.sum()

    w_mean_s = float(np.average(scores_arr,   weights=weights))
    w_mean_r = float(np.average(returns_arr,  weights=weights))

    cov   = float(np.average((scores_arr - w_mean_s) * (returns_arr - w_mean_r), weights=weights))
    std_s = float(np.sqrt(np.average((scores_arr  - w_mean_s) ** 2, weights=weights)))
    std_r = float(np.sqrt(np.average((returns_arr - w_mean_r) ** 2, weights=weights)))

    if std_s < 1e-10 or std_r < 1e-10:
        return {**_NO_IC_DECAY, "ic_decay_n_effective": len(valid)}

    ic_decay = cov / (std_s * std_r)

    if np.isnan(ic_decay):
        return {**_NO_IC_DECAY, "ic_decay_n_effective": len(valid)}

    raw_weights = np.exp(-np.log(2) * days_arr / half_life_days)
    n_effective = float((raw_weights.sum() ** 2) / (raw_weights ** 2).sum())

    return {
        "ic_decay_score":       round(float(ic_decay), 4),
        "ic_decay_n_effective": round(n_effective, 1),
        "ic_decay_half_life":   half_life_days,
    }


# ============================================================================
# HELPER 8C: STREAM METRICS
# ============================================================================

def calculate_stream_metrics(
    daily_results: List[Dict[str, Any]],
    min_signals: int = MIN_SIGNALS_FOR_SUFFICIENCY,
) -> Optional[Dict[str, Any]]:
    """
    Compute raw accuracy metrics from a stream's daily backtest results.

    TIERED SUFFICIENCY LOGIC (Change 5):
    - is_sufficient: signal_count >= min_signals (default 10)
      Controls whether the stream gets any weight in Node 11.
    - Binomial significance test: only runs when signal_count >= MIN_SIGNALS_FOR_BINOMIAL (20)
      With fewer than 20 signals the test cannot reliably detect accuracy
      above 70% at p<0.05. Streams with 10-19 signals are marked
      is_significant=False, is_anti_predictive=False (near-random) and
      receive proportional weight based on raw directional accuracy.
      This prevents sparse-but-real streams from being falsely penalized.

    Args:
        daily_results: List of daily dicts from any backtest_*_stream function
        min_signals: Minimum BUY+SELL signals required for is_sufficient=True

    Returns:
        Dict with all metrics, or None if daily_results is empty.
    """
    if not daily_results:
        logger.warning("No daily results — cannot compute stream metrics")
        return None

    all_days    = daily_results
    recent_days = [r for r in all_days if r["days_ago"] <= RECENT_PERIOD_DAYS]
    directional = [r for r in all_days if r["signal"] in ("BUY", "SELL")]

    buy_count    = sum(1 for r in all_days if r["signal"] == "BUY")
    sell_count   = sum(1 for r in all_days if r["signal"] == "SELL")
    hold_count   = sum(1 for r in all_days if r["signal"] == "HOLD")
    signal_count = buy_count + sell_count

    is_sufficient = signal_count >= min_signals

    # Full-period accuracy
    full_correct  = sum(1 for r in all_days if r["correct"])
    full_accuracy = full_correct / len(all_days) if all_days else 0.5

    # Recent-period accuracy (last 60 days)
    recent_accuracy: Optional[float] = None
    if len(recent_days) >= MIN_RECENT_DAYS_FOR_ACCURACY:
        recent_correct  = sum(1 for r in recent_days if r["correct"])
        recent_accuracy = recent_correct / len(recent_days)

    # Directional-only accuracy
    directional_days        = [r for r in all_days   if r["signal"] in ("BUY", "SELL")]
    directional_correct_all = sum(1 for r in directional_days if r["correct"])
    directional_accuracy: Optional[float] = (
        directional_correct_all / len(directional_days)
        if directional_days else None
    )

    recent_directional_days    = [r for r in recent_days if r["signal"] in ("BUY", "SELL")]
    recent_directional_correct = sum(1 for r in recent_directional_days if r["correct"])
    recent_directional_accuracy: Optional[float] = (
        recent_directional_correct / len(recent_directional_days)
        if len(recent_directional_days) >= MIN_RECENT_DAYS_FOR_ACCURACY else None
    )

    # Price movement statistics
    all_changes     = [r["actual_change_7d"] for r in all_days]
    correct_changes = [r["actual_change_7d"] for r in all_days if r["correct"]]
    wrong_changes   = [r["actual_change_7d"] for r in all_days if not r["correct"]]

    avg_actual_change     = float(np.mean(all_changes))     if all_changes     else 0.0
    avg_change_on_correct = float(np.mean(correct_changes)) if correct_changes else 0.0
    avg_change_on_wrong   = float(np.mean(wrong_changes))   if wrong_changes   else 0.0

    # -----------------------------------------------------------------------
    # Statistical significance tests — tiered by sample size (Change 5)
    #
    # Only run binomial test when signal_count >= MIN_SIGNALS_FOR_BINOMIAL (20).
    # Below that threshold the test lacks power to detect real signal at p<0.05,
    # so we skip it entirely and mark both flags False (near-random treatment).
    # This prevents streams with 10-19 valid signals from being falsely flagged
    # as anti_predictive and having their weight zeroed by Node 11.
    # -----------------------------------------------------------------------
    p_value            = 1.0
    p_value_less       = 1.0
    is_significant     = False
    is_anti_predictive = False
    binomial_tested    = False

    if signal_count >= MIN_SIGNALS_FOR_BINOMIAL:
        binomial_tested     = True
        directional_correct = sum(1 for r in directional if r["correct"])
        try:
            test_greater       = binomtest(directional_correct, signal_count, p=0.5, alternative="greater")
            p_value            = float(test_greater.pvalue)
            is_significant     = p_value < 0.05

            test_less          = binomtest(directional_correct, signal_count, p=0.5, alternative="less")
            p_value_less       = float(test_less.pvalue)
            is_anti_predictive = p_value_less < 0.05
        except Exception as e:
            logger.warning(f"Binomial test failed: {e}")
    elif signal_count >= min_signals:
        # Sufficient data for weighting but not enough for a reliable binomial test.
        # Log this so it's visible in the dashboard — the stream is real but sparse.
        logger.info(
            f"  Stream has {signal_count} signals (>= {min_signals} sufficient, "
            f"< {MIN_SIGNALS_FOR_BINOMIAL} binomial threshold) — "
            f"skipping significance test, marking near-random"
        )

    # Per-signal hit rates: "when the system said BUY/SELL, how often was it right?"
    # Minimum 5 samples required to report a meaningful rate.
    _MIN_SIGNAL_HIT_N = 5
    buy_days  = [r for r in all_days if r["signal"] == "BUY"]
    sell_days = [r for r in all_days if r["signal"] == "SELL"]
    buy_hit_rate:  Optional[float] = (
        round(sum(1 for r in buy_days  if r["correct"]) / len(buy_days),  4)
        if len(buy_days)  >= _MIN_SIGNAL_HIT_N else None
    )
    sell_hit_rate: Optional[float] = (
        round(sum(1 for r in sell_days if r["correct"]) / len(sell_days), 4)
        if len(sell_days) >= _MIN_SIGNAL_HIT_N else None
    )

    return {
        "full_accuracy":                round(full_accuracy, 4),
        "recent_accuracy":              round(recent_accuracy, 4) if recent_accuracy is not None else None,
        "directional_accuracy":         round(directional_accuracy, 4) if directional_accuracy is not None else None,
        "recent_directional_accuracy":  round(recent_directional_accuracy, 4) if recent_directional_accuracy is not None else None,
        "signal_count":                 signal_count,
        "buy_count":                    buy_count,
        "sell_count":                   sell_count,
        "hold_count":                   hold_count,
        "buy_hit_rate":                 buy_hit_rate,
        "sell_hit_rate":                sell_hit_rate,
        "total_days_evaluated":         len(all_days),
        "recent_days_evaluated":        len(recent_days),
        "avg_actual_change":            round(avg_actual_change, 4),
        "avg_change_on_correct":        round(avg_change_on_correct, 4),
        "avg_change_on_wrong":          round(avg_change_on_wrong, 4),
        "is_sufficient":                is_sufficient,
        "is_significant":               is_significant,
        "is_anti_predictive":           is_anti_predictive,
        "binomial_tested":              binomial_tested,   # new — visible in dashboard
        "p_value":                      round(p_value, 6),
        "p_value_less":                 round(p_value_less, 6),
        # IC fields populated later by calculate_news_ic() / calculate_news_ic_decay()
        "ic_score":             None,
        "ic_p_value":           None,
        "ic_significant":       None,
        "ic_decay_score":       None,
        "ic_decay_n_effective": None,
        "daily_results":        daily_results,
    }


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def backtesting_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 10: Backtesting

    Backtests 4 signal streams over 180 days. Prepares RAW accuracy data
    for Node 11. Does NOT apply any weighting.

    Key changes from original:
    - MIN_SIGNALS_FOR_SUFFICIENCY lowered 20 → 10 so sparse streams (market
      news n=8, related news n=10) receive proportional weight instead of
      being silently dropped.
    - MIN_SIGNALS_FOR_BINOMIAL (20) separates sufficiency from significance:
      streams with 10-19 signals get weighted but are not tested binomially.
    - MIN_RECENT_DAYS_FOR_ACCURACY lowered 5 → 3 to capture recent accuracy
      for sparse streams.
    - source_reliability read from Node 8 and passed to sentiment backtests.
    - Both standard IC and decay IC computed for all news streams.

    Args:
        state: LangGraph state dict with ticker, raw_price_data,
               and optionally news_impact_verification (Node 8 output).

    Returns:
        Updated state with 'backtest_results' populated.
    """
    start_time = datetime.now()
    ticker     = state.get("ticker", "UNKNOWN")

    logger.info(f"Node 10: Starting backtesting for {ticker}")

    try:
        # ====================================================================
        # STEP 1: Validate inputs
        # ====================================================================

        price_data: Optional[pd.DataFrame] = state.get("raw_price_data")

        if price_data is None or len(price_data) < 30:
            logger.error(
                f"Node 10: No usable price data for {ticker} "
                f"(got {len(price_data) if price_data is not None else 0} rows)"
            )
            state.setdefault("errors", []).append(
                "Node 10: Insufficient price data for backtesting"
            )
            state["backtest_results"] = None
            state.setdefault("node_execution_times", {})["node_10"] = (
                datetime.now() - start_time
            ).total_seconds()
            return state

        # ====================================================================
        # STEP 2: Calculate stock-specific HOLD threshold
        # ====================================================================

        hold_threshold = calculate_hold_threshold(price_data)

        # ====================================================================
        # STEP 3: Extract source_reliability from Node 8 output
        # ====================================================================

        niv: Dict[str, Any] = state.get("news_impact_verification") or {}
        source_reliability: Dict[str, Any] = niv.get("source_reliability") or {}

        if source_reliability:
            logger.info(
                f"Node 10: Using source reliability from Node 8 "
                f"({len(source_reliability)} sources)"
            )
        else:
            logger.info(
                "Node 10: No source reliability data from Node 8 — "
                "using equal-weight sentiment aggregation"
            )

        # ====================================================================
        # STEP 4: Fetch all historical news + outcomes from DB
        # ====================================================================

        logger.info(f"Fetching historical news outcomes for {ticker}...")
        news_events: List[Dict[str, Any]] = []
        try:
            news_df = get_news_with_outcomes(ticker, days=180)
            if news_df is not None and not news_df.empty:
                raw_events = news_df.to_dict("records")
                for event in raw_events:
                    for k, v in event.items():
                        if isinstance(v, float) and pd.isna(v):
                            event[k] = None
                news_events = raw_events
            logger.info(f"Retrieved {len(news_events)} historical news events")
        except Exception as e:
            logger.error(f"Failed to fetch news outcomes from DB: {e}")

        # ====================================================================
        # STEP 5: Backtest each stream independently
        # ====================================================================

        technical_daily: List[Dict] = []
        try:
            logger.info("Backtesting technical analysis stream...")
            technical_daily = backtest_technical_stream(price_data, hold_threshold)
        except Exception as e:
            logger.error(f"Technical stream backtest failed: {e}")

        stock_news_daily: List[Dict] = []
        try:
            logger.info("Backtesting stock news sentiment stream...")
            stock_news_daily = backtest_sentiment_stream(
                news_events, "stock", hold_threshold,
                source_reliability=source_reliability,
            )
        except Exception as e:
            logger.error(f"Stock news stream backtest failed: {e}")

        market_news_daily: List[Dict] = []
        try:
            logger.info("Backtesting market news sentiment stream...")
            spy_prices: Optional[Dict[str, float]] = state.get("spy_daily_prices")
            mc           = state.get("market_context") or {}
            corr_profile = mc.get("market_correlation_profile") or {}
            stock_beta: Optional[float] = corr_profile.get("beta_calculated")

            if spy_prices and stock_beta:
                logger.info(
                    f"Market backtest: SPY decomposition active "
                    f"(beta={stock_beta:.2f}, {len(spy_prices)} SPY dates)"
                )
            else:
                logger.info(
                    "Market backtest: SPY decomposition unavailable "
                    f"(spy={'present' if spy_prices else 'missing'}, "
                    f"beta={'present' if stock_beta else 'missing'}) — using raw return"
                )

            market_news_daily = backtest_sentiment_stream(
                news_events, "market", hold_threshold,
                spy_daily_prices=spy_prices,
                beta=stock_beta,
                source_reliability=source_reliability,
            )
        except Exception as e:
            logger.error(f"Market news stream backtest failed: {e}")

        related_news_daily: List[Dict] = []
        try:
            logger.info("Backtesting related news sentiment stream...")
            related_news_daily = backtest_sentiment_stream(
                news_events, "related", hold_threshold,
                source_reliability=source_reliability,
            )
        except Exception as e:
            logger.error(f"Related news stream backtest failed: {e}")

        combined_daily: List[Dict] = []
        try:
            logger.info("Backtesting combined stream (equal weights, unweighted aggregation)...")
            combined_daily = backtest_combined_stream(price_data, news_events, hold_threshold)
        except Exception as e:
            logger.error(f"Combined stream backtest failed: {e}")

        # ====================================================================
        # STEP 6: Compute base metrics for each stream
        # ====================================================================

        technical_metrics = calculate_stream_metrics(technical_daily)   if technical_daily   else None
        stock_metrics     = calculate_stream_metrics(stock_news_daily)   if stock_news_daily   else None
        market_metrics    = calculate_stream_metrics(market_news_daily)  if market_news_daily  else None
        related_metrics   = calculate_stream_metrics(related_news_daily) if related_news_daily else None
        combined_metrics  = calculate_stream_metrics(combined_daily)     if combined_daily     else None

        # ====================================================================
        # STEP 7: Compute IC (standard + decay) for all four streams
        # ====================================================================

        for metrics, daily in [
            (technical_metrics, technical_daily),
            (stock_metrics,     stock_news_daily),
            (market_metrics,    market_news_daily),
            (related_metrics,   related_news_daily),
        ]:
            if metrics and daily:
                metrics.update(calculate_news_ic(daily))
                metrics.update(calculate_news_ic_decay(daily))
        # technical_daily dicts contain 'sentiment_score' = signed technical_alpha,
        # so calculate_news_ic computes the same out-of-sample Pearson IC as news streams.

        # Also store Node 4's in-sample Ridge regression IC separately — it is a
        # different quantity (in-sample pearsonr of Ridge predictions vs actual returns)
        # used by Node 11's accuracy estimation, and should coexist with the
        # out-of-sample IC above rather than replace it.
        if technical_metrics:
            ti = state.get("technical_indicators") or {}
            node4_ic = ti.get("ic_score")
            if node4_ic is not None:
                technical_metrics["node4_ic_score"] = float(node4_ic)

        # ====================================================================
        # STEP 8: Build and store results
        # ====================================================================

        backtest_results: Dict[str, Any] = {
            "hold_threshold_pct":           round(hold_threshold, 4),
            "sample_period_days":           180,
            "technical":                    technical_metrics,
            "stock_news":                   stock_metrics,
            "market_news":                  market_metrics,
            "related_news":                 related_metrics,
            "combined_stream":              combined_metrics,
            "credibility_weighting_active": bool(source_reliability),
            "baseline_correction_active":   True,
            "magnitude_threshold":          MIN_SENTIMENT_MAGNITUDE,
            "min_signals_for_sufficiency":  MIN_SIGNALS_FOR_SUFFICIENCY,
            "min_signals_for_binomial":     MIN_SIGNALS_FOR_BINOMIAL,
        }

        state["backtest_results"] = backtest_results

        # ====================================================================
        # STEP 9: Log summary
        # ====================================================================

        execution_time = (datetime.now() - start_time).total_seconds()
        state.setdefault("node_execution_times", {})["node_10"] = execution_time

        logger.info(f"Node 10 completed in {execution_time:.3f}s")
        logger.info(f"  HOLD threshold: {hold_threshold:.2f}%")
        logger.info(f"  Sufficiency threshold: {MIN_SIGNALS_FOR_SUFFICIENCY} signals")
        logger.info(f"  Binomial test threshold: {MIN_SIGNALS_FOR_BINOMIAL} signals")
        logger.info(
            f"  Credibility weighting: "
            f"{'active' if source_reliability else 'inactive (no Node 8 data)'}"
        )

        for stream_name, metrics in [
            ("Technical",    technical_metrics),
            ("Stock News",   stock_metrics),
            ("Market News",  market_metrics),
            ("Related News", related_metrics),
            ("Combined",     combined_metrics),
        ]:
            if metrics:
                recent     = metrics.get("recent_accuracy")
                recent_str = f"{recent:.1%}" if recent is not None else "N/A"
                ic         = metrics.get("ic_score")
                ic_p       = metrics.get("ic_p_value")
                ic_n       = metrics.get("ic_n_samples", 0)
                ic_idio    = metrics.get("ic_used_idiosyncratic", False)
                ic_decay   = metrics.get("ic_decay_score")
                ic_dn      = metrics.get("ic_decay_n_effective", 0)
                tested     = metrics.get("binomial_tested", False)

                ic_str = (
                    f"{ic:+.3f} (p={ic_p:.3f}, n={ic_n})"
                    + (" [idiosyn.]" if ic_idio else "")
                    if ic is not None else "N/A"
                )
                ic_decay_str = (
                    f"{ic_decay:+.3f} (n_eff={ic_dn:.0f})"
                    if ic_decay is not None else "N/A"
                )

                logger.info(
                    f"  {stream_name}: "
                    f"full={metrics['full_accuracy']:.1%}, "
                    f"recent={recent_str}, "
                    f"signals={metrics['signal_count']} "
                    f"(BUY={metrics['buy_count']}, SELL={metrics['sell_count']}), "
                    f"sufficient={metrics['is_sufficient']}, "
                    f"binomial_tested={tested}, "
                    f"significant={metrics['is_significant']} (p={metrics['p_value']:.3f}), "
                    f"anti_predictive={metrics['is_anti_predictive']}, "
                    f"IC={ic_str}, IC_decay={ic_decay_str}"
                )
            else:
                logger.warning(f"  {stream_name}: No data available")

        return state

    except Exception as e:
        logger.error(f"Node 10 failed: {e}")
        logger.exception("Full traceback:")

        state.setdefault("errors", []).append(f"Node 10 (backtesting) failed: {e}")
        state["backtest_results"] = None
        state.setdefault("node_execution_times", {})["node_10"] = (
            datetime.now() - start_time
        ).total_seconds()

        return state