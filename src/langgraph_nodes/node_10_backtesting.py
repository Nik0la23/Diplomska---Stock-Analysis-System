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

Runs AFTER: Node 9B (behavioral anomaly detection)
Runs BEFORE: Node 11 (adaptive weights calculation)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import binomtest

from src.database.db_manager import get_news_with_outcomes
from src.langgraph_nodes.node_04_technical_analysis import (
    analyze_volume,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
    generate_technical_signal,
)
from src.utils.logger import get_node_logger
from src.utils.sentiment_config import BUY_THRESHOLD, SELL_THRESHOLD

logger = get_node_logger("node_10")


# ============================================================================
# CONSTANTS
# ============================================================================

HOLD_THRESHOLD_MULTIPLIER: float = 0.5    # hold_threshold = avg_7day_abs_return * this
HOLD_THRESHOLD_FALLBACK_PCT: float = 2.0  # fallback when insufficient price data
HOLD_THRESHOLD_MIN_PCT: float = 0.5       # sanity floor
HOLD_THRESHOLD_MAX_PCT: float = 8.0       # sanity ceiling

MIN_PRICE_ROWS_FOR_TECHNICAL: int = 50    # Node 4 requires ≥50 rows for moving averages
MIN_SIGNALS_FOR_SUFFICIENCY: int = 20     # minimum BUY+SELL signals for meaningful accuracy
RECENT_PERIOD_DAYS: int = 60             # "recent" window in days (provided raw to Node 11)
MIN_RECENT_DAYS_FOR_ACCURACY: int = 5    # minimum recent days before reporting recent_accuracy


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

        df = price_data.copy().sort_values("date").reset_index(drop=True)
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

        # Sanity bounds
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

    Single definition used by ALL streams — ensures correctness is evaluated
    consistently across technical and all sentiment streams.

    Args:
        signal: 'BUY', 'SELL', or 'HOLD'
        actual_change: Actual 7-calendar-day price change in percent
        hold_threshold: Stock-specific HOLD zone width in percent

    Returns:
        True if the signal matched the actual outcome
    """
    if signal == "BUY":
        return actual_change > hold_threshold
    elif signal == "SELL":
        return actual_change < -hold_threshold
    elif signal == "HOLD":
        return abs(actual_change) <= hold_threshold
    return False


# ============================================================================
# HELPER 3: TECHNICAL SIGNAL RECONSTRUCTION
# ============================================================================

def reconstruct_technical_signal(
    price_slice: pd.DataFrame,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Reconstruct the technical signal for the last day of a historical price slice.

    Reuses Node 4's helper functions directly — Node 4 itself is not called.
    This gives us exactly what Node 4 would have produced on that historical day.

    Args:
        price_slice: DataFrame with all price data up to and including the target date
                     (must have 'close', 'open', 'high', 'low', 'volume' columns)

    Returns:
        Tuple (signal, confidence) — ('BUY'/'SELL'/'HOLD', 0.0–1.0)
        Returns (None, None) if the slice has fewer than MIN_PRICE_ROWS_FOR_TECHNICAL rows
    """
    if price_slice is None or len(price_slice) < MIN_PRICE_ROWS_FOR_TECHNICAL:
        return None, None

    try:
        rsi = calculate_rsi(price_slice)
        macd = calculate_macd(price_slice)
        bollinger = calculate_bollinger_bands(price_slice)
        moving_avg = calculate_moving_averages(price_slice)
        volume = analyze_volume(price_slice)

        signal, confidence = generate_technical_signal(rsi, macd, bollinger, moving_avg, volume)
        return signal, float(confidence)

    except Exception as e:
        logger.debug(f"Technical signal reconstruction failed for {len(price_slice)}-row slice: {e}")
        return None, None


# ============================================================================
# HELPER 4: TECHNICAL STREAM BACKTEST
# ============================================================================

def backtest_technical_stream(
    price_data: pd.DataFrame,
    hold_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Backtest technical analysis signals over 180 days of price history.

    For each day D from row 50 onward (where at least 50 rows precede it
    and at least 7 calendar days of price data follow it):
        1. Slice price_data up to and including day D
        2. Reconstruct what Node 4 would have signalled on that day
        3. Find the actual close price 7 calendar days later
        4. Evaluate whether the signal was correct

    Uses 7 CALENDAR days (not trading days) to be consistent with
    news_outcomes.price_change_7day used in sentiment backtests.

    Args:
        price_data: Full OHLCV DataFrame from state (Node 1 output)
        hold_threshold: Stock-specific HOLD zone width in percent

    Returns:
        List of dicts, one per evaluated day:
        {date, signal, confidence, actual_change_7d, correct, days_ago}
    """
    results: List[Dict[str, Any]] = []

    if price_data is None or len(price_data) < MIN_PRICE_ROWS_FOR_TECHNICAL + 7:
        logger.warning(
            f"Insufficient price data for technical backtest "
            f"(need {MIN_PRICE_ROWS_FOR_TECHNICAL + 7}, got {len(price_data) if price_data is not None else 0})"
        )
        return results

    try:
        df = price_data.copy().sort_values("date").reset_index(drop=True)
        date_to_close: Dict[str, float] = dict(zip(df["date"], df["close"]))
        sorted_dates = sorted(df["date"].tolist())
        today = pd.Timestamp.now().normalize()

        for i in range(MIN_PRICE_ROWS_FOR_TECHNICAL, len(df)):
            date_str = str(df.iloc[i]["date"])

            # Find closing price 7 calendar days later (next available trading day)
            target_future = (
                pd.to_datetime(date_str) + pd.Timedelta(days=7)
            ).strftime("%Y-%m-%d")
            future_dates = [d for d in sorted_dates if d >= target_future]
            if not future_dates:
                continue  # No future data available for this day

            price_today = float(df.iloc[i]["close"])
            price_future = float(date_to_close[future_dates[0]])
            actual_change = ((price_future - price_today) / price_today) * 100

            # Reconstruct the signal that would have been generated on day i
            price_slice = df.iloc[: i + 1].copy()
            signal, confidence = reconstruct_technical_signal(price_slice)

            if signal is None:
                continue

            correct = evaluate_outcome(signal, actual_change, hold_threshold)
            days_ago = int((today - pd.to_datetime(date_str)).days)

            results.append(
                {
                    "date": date_str,
                    "signal": signal,
                    "confidence": round(confidence, 4) if confidence is not None else None,
                    "actual_change_7d": round(actual_change, 4),
                    "correct": correct,
                    "days_ago": days_ago,
                }
            )

        logger.info(f"Technical backtest: {len(results)} days evaluated")
        return results

    except Exception as e:
        logger.error(f"Technical backtest stream failed: {e}")
        return []


# ============================================================================
# HELPER 5: DAILY SENTIMENT AGGREGATION
# ============================================================================

def aggregate_daily_sentiment(articles: List[Dict[str, Any]]) -> Tuple[float, str]:
    """
    Aggregate stored sentiment scores from multiple articles into a single
    daily score and signal.

    Mirrors Node 5's per-stream aggregation logic:
    - 'positive' label → +sentiment_score (0 to +1)
    - 'negative' label → -sentiment_score (-1 to 0)
    - 'neutral' label → 0
    Then average across all articles.

    NOTE: This operates on ONE stream at a time (stock / market / related).
    The 50/25/25 cross-stream weighting is NOT applied here — that is
    Node 11's responsibility after it receives per-stream accuracy.

    Args:
        articles: List of article dicts with 'sentiment_label' and 'sentiment_score'

    Returns:
        Tuple (combined_score [-1,+1], signal ['BUY'/'SELL'/'HOLD'])
    """
    if not articles:
        return 0.0, "HOLD"

    signed_scores: List[float] = []
    for article in articles:
        label = article.get("sentiment_label", "neutral") or "neutral"
        raw_score = article.get("sentiment_score")
        score = float(raw_score) if raw_score is not None else 0.5

        if label == "positive":
            signed_scores.append(score)
        elif label == "negative":
            signed_scores.append(-score)
        else:
            signed_scores.append(0.0)

    if not signed_scores:
        return 0.0, "HOLD"

    combined = float(np.mean(signed_scores))

    if combined > BUY_THRESHOLD:
        signal = "BUY"
    elif combined < SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    return combined, signal


# ============================================================================
# HELPER 6: SENTIMENT STREAM BACKTEST
# ============================================================================

def backtest_sentiment_stream(
    news_events: List[Dict[str, Any]],
    news_type: str,
    hold_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Backtest a single sentiment stream (stock, market, or related) over 180 days.

    For each day D that has cached articles of the given news_type:
        1. Aggregate stored AV sentiment scores → daily signal
        2. Read actual 7-calendar-day price change from news_outcomes.price_change_7day
           (pre-computed by the background script — no additional price lookups needed)
        3. Evaluate correctness using our volatility-adaptive hold_threshold
           (NOT the pre-stored prediction_was_accurate_7day boolean, which uses
           a fixed threshold and may differ from our stock-specific one)

    Uses median of price_change_7day across same-day articles to be robust
    to outliers (e.g. one article with a bad price lookup).

    Args:
        news_events: Output of get_news_with_outcomes() converted to list of dicts
        news_type: 'stock', 'market', or 'related'
        hold_threshold: Stock-specific HOLD zone width in percent

    Returns:
        List of dicts, one per evaluated day:
        {date, signal, sentiment_score, article_count, actual_change_7d, correct, days_ago}
    """
    results: List[Dict[str, Any]] = []

    typed_events = [e for e in news_events if e.get("news_type") == news_type]
    if not typed_events:
        logger.warning(f"No historical events found for news_type='{news_type}'")
        return results

    try:
        today = pd.Timestamp.now().normalize()

        # Group articles by date (YYYY-MM-DD from published_at)
        by_date: Dict[str, List[Dict[str, Any]]] = {}
        for event in typed_events:
            pub = event.get("published_at", "")
            if not pub:
                continue
            date_str = str(pub)[:10]
            by_date.setdefault(date_str, []).append(event)

        for date_str, day_articles in sorted(by_date.items()):

            # Get actual price change from stored outcomes
            # Use median to be robust against single-article price lookup errors
            price_changes = [
                float(a["price_change_7day"])
                for a in day_articles
                if a.get("price_change_7day") is not None
            ]
            if not price_changes:
                continue

            actual_change = float(np.median(price_changes))

            # Aggregate sentiment for this day's articles
            daily_score, signal = aggregate_daily_sentiment(day_articles)

            correct = evaluate_outcome(signal, actual_change, hold_threshold)
            days_ago = int((today - pd.to_datetime(date_str)).days)

            results.append(
                {
                    "date": date_str,
                    "signal": signal,
                    "sentiment_score": round(daily_score, 4),
                    "article_count": len(day_articles),
                    "actual_change_7d": round(actual_change, 4),
                    "correct": correct,
                    "days_ago": days_ago,
                }
            )

        logger.info(f"Sentiment backtest '{news_type}': {len(results)} days evaluated")
        return results

    except Exception as e:
        logger.error(f"Sentiment backtest for '{news_type}' failed: {e}")
        return []


# ============================================================================
# HELPER 7: STREAM METRICS
# ============================================================================

def calculate_stream_metrics(
    daily_results: List[Dict[str, Any]],
    min_signals: int = MIN_SIGNALS_FOR_SUFFICIENCY,
) -> Optional[Dict[str, Any]]:
    """
    Compute raw accuracy metrics from a stream's daily backtest results.

    DOES NOT apply any weighting between full-period and recent-period.
    Returns both values separately so Node 11 can decide how to combine them.

    Metrics returned:
    - full_accuracy: raw accuracy over all evaluated days
    - recent_accuracy: raw accuracy over the last RECENT_PERIOD_DAYS days
      (None if fewer than MIN_RECENT_DAYS_FOR_ACCURACY days available)
    - signal_count: BUY + SELL signals only (HOLD not counted)
    - buy_count / sell_count / hold_count
    - avg_actual_change: mean price movement regardless of signal
    - avg_change_on_correct: mean price movement when signal was right
    - avg_change_on_wrong: mean price movement when signal was wrong
    - is_sufficient: True if signal_count >= min_signals
    - is_significant: binomial test p_value < 0.05 against 0.5 baseline
    - p_value: exact p-value from binomial test
    - daily_results: full daily list for dashboard

    Args:
        daily_results: List of daily dicts from any backtest_*_stream function
        min_signals: Minimum BUY+SELL signals required for meaningful accuracy

    Returns:
        Dict with all metrics, or None if daily_results is empty
    """
    if not daily_results:
        logger.warning("No daily results — cannot compute stream metrics")
        return None

    today_ref = pd.Timestamp.now().normalize()

    all_days = daily_results
    recent_days = [r for r in all_days if r["days_ago"] <= RECENT_PERIOD_DAYS]
    directional = [r for r in all_days if r["signal"] in ("BUY", "SELL")]

    buy_count = sum(1 for r in all_days if r["signal"] == "BUY")
    sell_count = sum(1 for r in all_days if r["signal"] == "SELL")
    hold_count = sum(1 for r in all_days if r["signal"] == "HOLD")
    signal_count = buy_count + sell_count

    is_sufficient = signal_count >= min_signals

    # --- Full-period accuracy (all evaluated days) ---
    full_correct = sum(1 for r in all_days if r["correct"])
    full_accuracy = full_correct / len(all_days) if all_days else 0.5

    # --- Recent-period accuracy (last 60 days) — raw, no 70/30 weighting ---
    recent_accuracy: Optional[float] = None
    if len(recent_days) >= MIN_RECENT_DAYS_FOR_ACCURACY:
        recent_correct = sum(1 for r in recent_days if r["correct"])
        recent_accuracy = recent_correct / len(recent_days)

    # --- Actual price movements ---
    all_changes = [r["actual_change_7d"] for r in all_days]
    correct_changes = [r["actual_change_7d"] for r in all_days if r["correct"]]
    wrong_changes = [r["actual_change_7d"] for r in all_days if not r["correct"]]

    avg_actual_change = float(np.mean(all_changes)) if all_changes else 0.0
    avg_change_on_correct = float(np.mean(correct_changes)) if correct_changes else 0.0
    avg_change_on_wrong = float(np.mean(wrong_changes)) if wrong_changes else 0.0

    # --- Statistical significance: binomial test against p=0.5 baseline ---
    # Only directional signals (BUY/SELL) are tested — HOLD is excluded
    p_value = 1.0
    is_significant = False
    if signal_count >= min_signals:
        directional_correct = sum(1 for r in directional if r["correct"])
        try:
            test_result = binomtest(directional_correct, signal_count, p=0.5, alternative="greater")
            p_value = float(test_result.pvalue)
            is_significant = p_value < 0.05
        except Exception as e:
            logger.warning(f"Binomial test failed: {e}")

    return {
        "full_accuracy": round(full_accuracy, 4),
        "recent_accuracy": round(recent_accuracy, 4) if recent_accuracy is not None else None,
        "signal_count": signal_count,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "total_days_evaluated": len(all_days),
        "recent_days_evaluated": len(recent_days),
        "avg_actual_change": round(avg_actual_change, 4),
        "avg_change_on_correct": round(avg_change_on_correct, 4),
        "avg_change_on_wrong": round(avg_change_on_wrong, 4),
        "is_sufficient": is_sufficient,
        "is_significant": is_significant,
        "p_value": round(p_value, 6),
        "daily_results": daily_results,  # Full list preserved for dashboard
    }


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def backtesting_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 10: Backtesting

    Backtests 4 signal streams over 180 days. Prepares RAW accuracy data
    for Node 11. Does NOT apply any weighting.

    What it returns per stream:
    - full_accuracy: accuracy over all 180 days
    - recent_accuracy: accuracy over the last 60 days (raw, no weighting)
    - signal_count, buy/sell/hold counts
    - avg_actual_change, avg_change_on_correct, avg_change_on_wrong
    - is_sufficient, is_significant, p_value
    - daily_results: full list for dashboard visualization

    Node 11 will apply: weighted_accuracy = 0.7 * recent_accuracy + 0.3 * full_accuracy
    and then normalize into adaptive weights.

    Runs AFTER: Node 9B (behavioral anomaly detection)
    Runs BEFORE: Node 11 (adaptive weights calculation)

    Args:
        state: LangGraph state dict containing:
            - ticker: Stock ticker symbol
            - raw_price_data: 180-day OHLCV DataFrame from Node 1

    Returns:
        Updated state with 'backtest_results' populated
    """
    start_time = datetime.now()
    ticker = state.get("ticker", "UNKNOWN")

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
        # STEP 3: Fetch all historical news + outcomes from DB
        #         Single DB call — shared by all 3 sentiment streams
        # ====================================================================

        logger.info(f"Fetching historical news outcomes for {ticker}...")
        news_events: List[Dict[str, Any]] = []
        try:
            news_df = get_news_with_outcomes(ticker, days=180)
            if news_df is not None and not news_df.empty:
                raw_events = news_df.to_dict("records")
                # Clean NaN values so downstream code can use None checks
                for event in raw_events:
                    for k, v in event.items():
                        if isinstance(v, float) and pd.isna(v):
                            event[k] = None
                news_events = raw_events
            logger.info(f"Retrieved {len(news_events)} historical news events")
        except Exception as e:
            logger.error(f"Failed to fetch news outcomes from DB: {e}")
            # news_events stays [] — sentiment streams will return empty results

        # ====================================================================
        # STEP 4: Backtest each stream independently
        #         Failures are isolated — one bad stream does not stop others
        # ====================================================================

        # Technical stream
        technical_daily: List[Dict] = []
        try:
            logger.info("Backtesting technical analysis stream...")
            technical_daily = backtest_technical_stream(price_data, hold_threshold)
        except Exception as e:
            logger.error(f"Technical stream backtest failed: {e}")

        # Stock news sentiment stream
        stock_news_daily: List[Dict] = []
        try:
            logger.info("Backtesting stock news sentiment stream...")
            stock_news_daily = backtest_sentiment_stream(news_events, "stock", hold_threshold)
        except Exception as e:
            logger.error(f"Stock news stream backtest failed: {e}")

        # Market news sentiment stream
        market_news_daily: List[Dict] = []
        try:
            logger.info("Backtesting market news sentiment stream...")
            market_news_daily = backtest_sentiment_stream(news_events, "market", hold_threshold)
        except Exception as e:
            logger.error(f"Market news stream backtest failed: {e}")

        # Related news sentiment stream
        related_news_daily: List[Dict] = []
        try:
            logger.info("Backtesting related news sentiment stream...")
            related_news_daily = backtest_sentiment_stream(news_events, "related", hold_threshold)
        except Exception as e:
            logger.error(f"Related news stream backtest failed: {e}")

        # ====================================================================
        # STEP 5: Compute metrics for each stream
        # ====================================================================

        technical_metrics = calculate_stream_metrics(technical_daily) if technical_daily else None
        stock_metrics = calculate_stream_metrics(stock_news_daily) if stock_news_daily else None
        market_metrics = calculate_stream_metrics(market_news_daily) if market_news_daily else None
        related_metrics = calculate_stream_metrics(related_news_daily) if related_news_daily else None

        # ====================================================================
        # STEP 6: Build and store results
        # ====================================================================

        backtest_results: Dict[str, Any] = {
            "hold_threshold_pct": round(hold_threshold, 4),
            "sample_period_days": 180,
            "technical": technical_metrics,
            "stock_news": stock_metrics,
            "market_news": market_metrics,
            "related_news": related_metrics,
        }

        state["backtest_results"] = backtest_results

        # ====================================================================
        # STEP 7: Log summary
        # ====================================================================

        execution_time = (datetime.now() - start_time).total_seconds()
        state.setdefault("node_execution_times", {})["node_10"] = execution_time

        logger.info(f"Node 10 completed in {execution_time:.3f}s")
        logger.info(f"  HOLD threshold: {hold_threshold:.2f}%")

        for stream_name, metrics in [
            ("Technical", technical_metrics),
            ("Stock News", stock_metrics),
            ("Market News", market_metrics),
            ("Related News", related_metrics),
        ]:
            if metrics:
                recent = metrics.get("recent_accuracy")
                recent_str = f"{recent:.1%}" if recent is not None else "N/A"
                logger.info(
                    f"  {stream_name}: "
                    f"full={metrics['full_accuracy']:.1%}, "
                    f"recent={recent_str}, "
                    f"signals={metrics['signal_count']}, "
                    f"significant={metrics['is_significant']} (p={metrics['p_value']:.3f})"
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
