"""
Node 9B: Behavioral Anomaly Detection & Fraud Pattern Recognition
Analyzes price/volume/news behavior to detect manipulation patterns (e.g. pump-and-dump)
using seven coordinated detectors and produces a rich behavioral_anomaly_detection
structure in the shared LangGraph state.

Runs AFTER: Node 8 (news_verification)
Runs BEFORE: Node 10 (backtesting)
Can run in PARALLEL with: Nothing (sequential)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

import numpy as np
import pandas as pd

from src.graph.state import StockAnalysisState
from src.database.db_manager import (
    get_historical_daily_aggregates,
    get_volume_baseline,
    get_article_count_baseline,
)

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS - BASIC UTILITIES
# ============================================================================

def _safe_mean(values: List[float]) -> float:
    """Return mean of list or 0.0 if empty."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _price_change_pct(start: float, end: float) -> float:
    """Percentage change between two prices, 0.0 if invalid."""
    try:
        if start is None or end is None or start == 0:
            return 0.0
        return (end - start) / start * 100.0
    except Exception:
        return 0.0


def _neutral_detector_result() -> Dict[str, Any]:
    """Return a neutral detector result used when data is missing."""
    return {
        "detected": False,
        "severity": "LOW",
        "contribution_score": 0,
    }


# ============================================================================
# DETECTION SYSTEM 1: Volume Anomaly Detector
# ============================================================================

def detect_volume_anomaly(
    current_volume: float,
    historical_volumes: List[float],
    baseline_days: int = 30,
) -> Dict[str, Any]:
    """
    Detects volume anomalies by comparing current volume to historical baseline.

    Returns:
        Dict with keys:
            - detected: bool
            - volume_ratio: float
            - severity: 'LOW' | 'MEDIUM' | 'HIGH'
            - contribution_score: int (0-20)
            - baseline_volume: float
            - current_volume: float
    """
    if current_volume is None or current_volume <= 0 or not historical_volumes:
        result = _neutral_detector_result()
        result.update(
            {
                "volume_ratio": 1.0,
                "baseline_volume": 0.0,
                "current_volume": current_volume or 0.0,
            }
        )
        return result

    # Use the most recent N days for baseline (or all if fewer)
    relevant = historical_volumes[-baseline_days:] if len(historical_volumes) >= baseline_days else historical_volumes
    baseline = _safe_mean([v for v in relevant if v is not None and v > 0])
    if baseline <= 0:
        baseline = current_volume

    volume_ratio = float(current_volume) / float(baseline) if baseline > 0 else 1.0

    # Severity and scoring
    if volume_ratio < 2.0:
        severity = "LOW"
        score = 0
    elif 2.0 <= volume_ratio < 4.0:
        severity = "MEDIUM"
        score = 10
    elif 4.0 <= volume_ratio < 6.0:
        severity = "HIGH"
        score = 15
    else:  # >= 6.0
        severity = "HIGH"
        score = 20

    return {
        "detected": volume_ratio >= 2.0,
        "volume_ratio": volume_ratio,
        "severity": severity,
        "contribution_score": score,
        "baseline_volume": baseline,
        "current_volume": current_volume,
    }


# ============================================================================
# DETECTION SYSTEM 2: Source Reliability Divergence Detector
# ============================================================================

def detect_source_reliability_divergence(
    todays_articles: List[Dict[str, Any]],
    source_reliability_dict: Dict[str, Dict[str, Any]],
    historical_avg_reliability: float,
) -> Dict[str, Any]:
    """
    Detect when today's news sources are less reliable than usual.

    Returns dict with:
        detected, today_avg_accuracy, historical_avg_accuracy, divergence,
        severity, contribution_score, unreliable_source_count, total_article_count
    """
    if not todays_articles or historical_avg_reliability is None:
        result = _neutral_detector_result()
        result.update(
            {
                "today_avg_accuracy": 0.0,
                "historical_avg_accuracy": historical_avg_reliability or 0.0,
                "divergence": 0.0,
                "unreliable_source_count": 0,
                "total_article_count": 0,
            }
        )
        return result

    total_articles = 0
    weighted_sum = 0.0
    unreliable_count = 0

    for article in todays_articles:
        source_name = article.get("source") or article.get("source_name")
        if isinstance(source_name, dict):
            source_name = source_name.get("name")

        accuracy_rate = 0.5  # neutral default
        if source_name and source_name in source_reliability_dict:
            accuracy_rate = float(source_reliability_dict[source_name].get("accuracy_rate", 0.5))

        total_articles += 1
        weighted_sum += accuracy_rate

        if accuracy_rate < 0.6:
            unreliable_count += 1

    if total_articles == 0:
        today_avg_accuracy = 0.0
    else:
        today_avg_accuracy = weighted_sum / total_articles

    historical_avg_accuracy = float(historical_avg_reliability)
    divergence = max(0.0, historical_avg_accuracy - today_avg_accuracy)

    if divergence < 0.10:
        severity = "LOW"
        score = 0
    elif 0.10 <= divergence < 0.20:
        severity = "MEDIUM"
        score = 12
    elif 0.20 <= divergence < 0.30:
        severity = "HIGH"
        score = 18
    else:
        severity = "HIGH"
        score = 20

    detected = divergence >= 0.15

    return {
        "detected": detected,
        "today_avg_accuracy": today_avg_accuracy,
        "historical_avg_accuracy": historical_avg_accuracy,
        "divergence": divergence,
        "severity": severity,
        "contribution_score": score if detected else 0,
        "unreliable_source_count": unreliable_count,
        "total_article_count": total_articles,
    }


# ============================================================================
# DETECTION SYSTEM 3: News Velocity Anomaly Detector
# ============================================================================

def detect_news_velocity_anomaly(
    today_article_count: int,
    historical_daily_avg: Optional[float],
    content_analysis_summary: Optional[Dict[str, Any]],
    article_time_window_days: int = 1,
) -> Dict[str, Any]:
    """
    Detect abnormal spikes in news article volume.

    Compares the DAILY RATE of newly fetched articles against the historical
    daily average to catch pump-and-dump article floods in real time.

    Args:
        today_article_count: Number of articles in the current fetch batch.
            When Node 2 metadata is available this is `newly_fetched_count`
            (articles fetched fresh from APIs this run).  Falls back to the
            full 6-month batch count when metadata is absent.
        historical_daily_avg: Historical average articles per calendar day
            (from DB, using fetched_at).
        content_analysis_summary: Summary from Node 9A.
        article_time_window_days: How many days the articles span.  Dividing
            today_article_count by this value normalises both sides to the
            same unit (articles/day) before computing the velocity ratio.
            Defaults to 1 (single-day batch, no normalisation needed).
    """
    if historical_daily_avg is None or historical_daily_avg <= 0 or today_article_count <= 0:
        result = _neutral_detector_result()
        result.update(
            {
                "today_article_count": today_article_count,
                "articles_per_day": 0.0,
                "time_window_days": article_time_window_days,
                "historical_daily_avg": historical_daily_avg or 0.0,
                "velocity_ratio": 1.0,
                "high_anomaly_article_count": 0,
                "coordinated_keywords": False,
            }
        )
        return result

    # Normalise to daily rate before comparing against the historical baseline
    articles_per_day = float(today_article_count) / float(max(1, article_time_window_days))
    velocity_ratio = articles_per_day / float(historical_daily_avg)

    high_anomaly_article_count = 0
    coordinated_keywords = False

    if content_analysis_summary:
        high_anomaly_article_count = int(
            content_analysis_summary.get("high_risk_articles", 0)
        )
        top_keywords = content_analysis_summary.get("top_keywords", [])
        total_processed = content_analysis_summary.get("total_articles_processed", 0) or 0
        if top_keywords and total_processed > 0:
            # Check if the most frequent keyword appears in >60% of articles
            most_common_keyword, count = top_keywords[0]
            coordinated_keywords = (count / total_processed) >= 0.6

    if velocity_ratio < 3.0:
        severity = "LOW"
        score = 0
    elif 3.0 <= velocity_ratio < 5.0:
        severity = "MEDIUM"
        score = 8
    elif 5.0 <= velocity_ratio < 8.0:
        severity = "HIGH"
        score = 12
    else:
        severity = "HIGH"
        score = 15

    if coordinated_keywords:
        score += 3

    detected = velocity_ratio >= 3.0

    return {
        "detected": detected,
        "today_article_count": today_article_count,
        "articles_per_day": articles_per_day,
        "time_window_days": article_time_window_days,
        "historical_daily_avg": historical_daily_avg,
        "velocity_ratio": velocity_ratio,
        "severity": severity,
        "contribution_score": score if detected else 0,
        "high_anomaly_article_count": high_anomaly_article_count,
        "coordinated_keywords": coordinated_keywords,
    }


# ============================================================================
# DETECTION SYSTEM 4: News-Price Divergence Detector
# ============================================================================

def detect_news_price_divergence(
    sentiment_signal: Optional[str],
    sentiment_score: Optional[float],
    price_change: float,
    todays_articles: List[Dict[str, Any]],
    source_reliability_dict: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Detect divergence between news sentiment and actual price movement.
    """
    sentiment_score = sentiment_score or 0.0

    if sentiment_score > 0.2:
        sentiment_direction = "positive"
    elif sentiment_score < -0.2:
        sentiment_direction = "negative"
    else:
        sentiment_direction = "neutral"

    if price_change > 0.5:
        price_direction = "up"
    elif price_change < -0.5:
        price_direction = "down"
    else:
        price_direction = "flat"

    # Count high-credibility articles
    credible_count = 0
    for article in todays_articles:
        source_name = article.get("source") or article.get("source_name")
        if isinstance(source_name, dict):
            source_name = source_name.get("name")
        if source_name and source_name in source_reliability_dict:
            if float(source_reliability_dict[source_name].get("accuracy_rate", 0.0)) > 0.7:
                credible_count += 1

    credible_source_corroboration = credible_count >= 3

    divergence_type: Optional[str] = None
    severity = "LOW"
    score = 0
    explanation = ""

    # Type A: positive news, negative price
    if sentiment_direction == "positive" and price_direction == "down":
        divergence_type = "A"
        severity = "LOW"
        score = 5
        explanation = "Positive sentiment but price falling (possibly already priced in or market expected more)."

    # Type B: negative news, positive price
    elif sentiment_direction == "negative" and price_direction == "up":
        divergence_type = "B"
        severity = "MEDIUM"
        score = 10
        explanation = "Negative sentiment but price rising (short covering or market expected worse)."

    # Type C: low-cred positive news, price rising, no credible corroboration, high anomaly
    else:
        # Compute average composite anomaly score from articles
        composite_scores: List[float] = []
        for article in todays_articles:
            score_val = article.get("composite_anomaly_score")
            if isinstance(score_val, (int, float)):
                composite_scores.append(float(score_val))
        avg_anomaly = _safe_mean(composite_scores)

        if (
            sentiment_direction == "positive"
            and price_direction == "up"
            and not credible_source_corroboration
            and avg_anomaly > 0.6
        ):
            divergence_type = "C"
            severity = "CRITICAL"
            score = 25
            explanation = "Price rising on highly anomalous, low-credibility news only (pump-and-dump signal)."

    detected = divergence_type is not None

    return {
        "detected": detected,
        "divergence_type": divergence_type,
        "sentiment_direction": sentiment_direction,
        "price_direction": price_direction,
        "credible_source_corroboration": credible_source_corroboration,
        "severity": severity,
        "contribution_score": score if detected else 0,
        "explanation": explanation,
    }


# ============================================================================
# DETECTION SYSTEM 5: Cross-Stream Coherence Detector
# ============================================================================

def detect_cross_stream_incoherence(
    stock_sentiment: float,
    market_sentiment: float,
    related_sentiment: float,
    stock_article_count: int,
    market_article_count: int,
    related_article_count: int,
    stock_avg_anomaly: float,
) -> Dict[str, Any]:
    """
    Detect when stock news diverges from market/related news in suspicious ways.
    """
    # Sentiment coherence
    stock_market_diff = abs(stock_sentiment - market_sentiment)
    stock_related_diff = abs(stock_sentiment - related_sentiment)

    # Divide by 2.0: the max possible total diff is 2.0 (each diff ≤ 1.0),
    # so coherence stays in [0, 1] and is appropriately sensitive to divergence.
    coherence = 1.0 - (stock_market_diff + stock_related_diff) / 2.0
    coherence = max(0.0, min(1.0, coherence))

    isolated_signal = False
    if (
        stock_sentiment > 0.4
        and market_sentiment < 0.1
        and related_sentiment < 0.1
    ):
        isolated_signal = True
    if stock_article_count >= 50 and (market_article_count + related_article_count) < 10:
        isolated_signal = True

    if coherence > 0.7:
        severity = "LOW"
        score = 0
    elif 0.5 <= coherence <= 0.7:
        severity = "MEDIUM"
        score = 5
    else:
        severity = "HIGH"
        score = 8

    if isolated_signal and stock_avg_anomaly > 0.6:
        score += 2

    detected = coherence < 0.6 or isolated_signal
    explanation = ""
    if detected:
        explanation = "Stock news sentiment diverges from market/related streams."
        if isolated_signal:
            explanation += " Isolated stock-specific narrative detected."

    return {
        "detected": detected,
        "coherence_score": coherence,
        "stock_stream_sentiment": stock_sentiment,
        "market_stream_sentiment": market_sentiment,
        "related_stream_sentiment": related_sentiment,
        "isolated_signal": isolated_signal,
        "severity": severity,
        "contribution_score": score if detected else 0,
        "explanation": explanation,
    }


# ============================================================================
# DETECTION SYSTEM 6: Historical Pattern Matcher
# ============================================================================

def match_historical_patterns(
    today_profile: Dict[str, Any],
    historical_daily_aggregates: List[Dict[str, Any]],
    similarity_threshold: float = 0.75,
) -> Dict[str, Any]:
    """
    Finds historical days similar to today and checks their outcomes.
    """
    if not historical_daily_aggregates:
        result = _neutral_detector_result()
        result.update(
            {
                "similar_periods_found": 0,
                "outcomes": {
                    "pct_ended_in_decline": 0.0,
                    "pct_ended_in_crash": 0.0,
                    "avg_price_change_7d": 0.0,
                    "worst_outcome": 0.0,
                    "best_outcome": 0.0,
                },
                "pattern_match_confidence": 0.0,
                "insufficient_data": True,
            }
        )
        return result

    def _similarity(today: Dict[str, Any], hist: Dict[str, Any]) -> float:
        # Normalize article count difference
        a_today = max(1.0, float(today.get("article_count", 0)))
        a_hist = max(1.0, float(hist.get("article_count", 0)))
        diff_articles = abs(a_today - a_hist) / max(a_today, a_hist)

        diff_anom = abs(
            float(today.get("avg_composite_anomaly", 0.0))
            - float(hist.get("avg_composite_anomaly", 0.0))
        )
        diff_reliab = abs(
            float(today.get("avg_source_reliability", 0.0))
            - float(hist.get("avg_source_credibility", 0.0))
        )

        v_today = max(1e-9, float(today.get("volume_ratio", 1.0)))
        v_hist = max(1e-9, float(hist.get("volume_ratio", 1.0)))
        diff_vol = abs(v_today - v_hist) / max(v_today, v_hist)

        weighted_diff = (
            diff_articles * 0.20
            + diff_anom * 0.30
            + diff_reliab * 0.30
            + diff_vol * 0.20
        )
        return max(0.0, 1.0 - weighted_diff)

    matches: List[Tuple[float, Dict[str, Any]]] = []
    for day in historical_daily_aggregates:
        sim = _similarity(today_profile, day)
        if sim >= similarity_threshold:
            matches.append((sim, day))

    # If too few matches, relax the threshold once
    if len(matches) < 5:
        matches = []
        for day in historical_daily_aggregates:
            sim = _similarity(today_profile, day)
            if sim >= 0.65:
                matches.append((sim, day))

    if not matches:
        result = _neutral_detector_result()
        result.update(
            {
                "similar_periods_found": 0,
                "outcomes": {
                    "pct_ended_in_decline": 0.0,
                    "pct_ended_in_crash": 0.0,
                    "avg_price_change_7d": 0.0,
                    "worst_outcome": 0.0,
                    "best_outcome": 0.0,
                },
                "pattern_match_confidence": 0.0,
                "insufficient_data": True,
            }
        )
        return result

    # Analyze outcomes on matched days
    decline_count = 0
    crash_count = 0
    changes_7d: List[float] = []
    worst = 0.0
    best = 0.0

    for _, day in matches:
        chg7 = float(day.get("price_change_7d", 0.0))
        changes_7d.append(chg7)
        worst = min(worst, chg7)
        best = max(best, chg7)
        if chg7 < -2.0:
            decline_count += 1
        if chg7 < -5.0:
            crash_count += 1

    total = len(matches)
    pct_decline = decline_count / total if total > 0 else 0.0
    pct_crash = crash_count / total if total > 0 else 0.0
    avg_change = _safe_mean(changes_7d)

    if pct_crash > 0.5:
        detected = True
        severity = "HIGH"
        score = 10
    elif pct_decline > 0.6:
        detected = True
        severity = "MEDIUM"
        score = 7
    else:
        detected = False
        severity = "LOW"
        score = 0

    pattern_match_confidence = min(1.0, total / 180.0)

    return {
        "detected": detected,
        "similar_periods_found": total,
        "outcomes": {
            "pct_ended_in_decline": pct_decline,
            "pct_ended_in_crash": pct_crash,
            "avg_price_change_7d": avg_change,
            "worst_outcome": worst,
            "best_outcome": best,
        },
        "pattern_match_confidence": pattern_match_confidence,
        "severity": severity,
        "contribution_score": score if detected else 0,
        "insufficient_data": False,
    }


# ============================================================================
# DETECTION SYSTEM 7: Pump-and-Dump Composite Scorer
# ============================================================================

def calculate_pump_and_dump_score(
    volume_result: Dict[str, Any],
    reliability_result: Dict[str, Any],
    velocity_result: Dict[str, Any],
    divergence_result: Dict[str, Any],
    coherence_result: Dict[str, Any],
    pattern_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Aggregates all detection system results into final risk score.
    """
    breakdown = {
        "volume_anomaly": int(volume_result.get("contribution_score", 0)),
        "source_reliability_divergence": int(
            reliability_result.get("contribution_score", 0)
        ),
        "news_velocity_anomaly": int(velocity_result.get("contribution_score", 0)),
        "news_price_divergence": int(divergence_result.get("contribution_score", 0)),
        "cross_stream_incoherence": int(coherence_result.get("contribution_score", 0)),
        "historical_pattern_match": int(pattern_result.get("contribution_score", 0)),
    }

    total_score = sum(breakdown.values())

    if total_score <= 30:
        risk_level = "LOW"
    elif 31 <= total_score <= 55:
        risk_level = "MEDIUM"
    elif 56 <= total_score <= 75:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    # Special rule: Type C divergence auto-elevates to at least HIGH
    if divergence_result.get("divergence_type") == "C" and total_score < 56:
        total_score = max(total_score, 56)
        risk_level = "HIGH"

    # Identify primary risk factors (top 3 contributors)
    sorted_factors = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)
    primary_risk_factors = [
        name for name, score in sorted_factors if score > 0
    ][:3]

    return {
        "pump_and_dump_score": int(total_score),
        "risk_level": risk_level,
        "primary_risk_factors": primary_risk_factors,
        "detection_breakdown": breakdown,
    }


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def behavioral_anomaly_detection_node(
    state: StockAnalysisState,
) -> StockAnalysisState:
    """
    Node 9B: Behavioral Anomaly Detection

    Orchestrates all 7 detection systems and produces final behavioral risk assessment.

    Execution flow:
    1. Extract price, news, sentiment, Node 8 and Node 9A outputs from state
    2. Query database for historical aggregates and baselines
    3. Run each detector independently
    4. Combine into composite pump-and-dump score and risk_level
    5. Write detailed behavioral_anomaly_detection into state
    """
    start_time = datetime.now()
    ticker = state.get("ticker", "UNKNOWN")

    try:
        logger.info(f"Node 9B: Starting behavioral anomaly detection for {ticker}")

        raw_price_data = state.get("raw_price_data")
        if isinstance(raw_price_data, pd.DataFrame) and not raw_price_data.empty:
            df = raw_price_data.sort_values("date")
        else:
            df = None

        # Latest price & volume
        current_price: Optional[float] = None
        yesterday_close: Optional[float] = None
        current_volume: Optional[float] = None
        historical_volumes: List[float] = []

        if df is not None and len(df) >= 2:
            current_close_row = df.iloc[-1]
            prev_close_row = df.iloc[-2]
            current_price = float(current_close_row["close"])
            yesterday_close = float(prev_close_row["close"])
            current_volume = float(current_close_row["volume"])
            historical_volumes = [float(v) for v in df["volume"].tolist()]

        price_change_1d = _price_change_pct(yesterday_close, current_price)

        # Sentiment (Node 5, possibly adjusted by Node 8)
        aggregated_sentiment = state.get("aggregated_sentiment") or 0.0
        sentiment_signal = state.get("sentiment_signal")
        sentiment_confidence = state.get("sentiment_confidence") or 0.0

        raw_sentiment_scores: List[Dict[str, Any]] = state.get(
            "raw_sentiment_scores", []
        )

        # Market context (Node 6)
        market_context = state.get("market_context") or {}

        # Node 8 output
        news_impact_verification = state.get("news_impact_verification", {})
        historical_corr = float(
            news_impact_verification.get("historical_correlation", 0.0)
        )
        news_accuracy_score = float(
            news_impact_verification.get("news_accuracy_score", 0.0)
        )  # 0-100
        historical_avg_reliability = news_accuracy_score / 100.0 if news_accuracy_score else 0.0
        source_reliability = news_impact_verification.get("source_reliability", {})

        # Node 9A outputs
        content_summary = state.get("content_analysis_summary") or {}
        cleaned_stock_news = state.get("cleaned_stock_news", [])
        cleaned_market_news = state.get("cleaned_market_news", [])
        cleaned_related_news = state.get("cleaned_related_company_news", [])

        # Fetch metadata from Node 2 (enables true daily velocity comparison)
        news_metadata = state.get("news_fetch_metadata") or {}
        newly_fetched_count = news_metadata.get("newly_fetched_count")
        fetch_window_days = news_metadata.get("fetch_window_days", 1)

        # Total articles in state (6-month batch) — used by other detectors
        today_article_count = (
            len(cleaned_stock_news)
            + len(cleaned_market_news)
            + len(cleaned_related_news)
        )

        # Average composite anomaly for stock stream
        stock_anomaly_scores: List[float] = []
        for article in cleaned_stock_news:
            val = article.get("composite_anomaly_score")
            if isinstance(val, (int, float)):
                stock_anomaly_scores.append(float(val))
        stock_avg_anomaly = _safe_mean(stock_anomaly_scores)

        # Per-stream sentiment: derive from raw_sentiment_scores (per-article type tags)
        # produced by Node 5. avg_impact in news_type_effectiveness is a price-impact %,
        # NOT a -1..+1 sentiment score, so we never use it here.
        raw_scores_all: List[Dict[str, Any]] = raw_sentiment_scores or []
        stock_scores_list = [
            float(s["sentiment_score"])
            for s in raw_scores_all
            if s.get("type") == "stock" and isinstance(s.get("sentiment_score"), (int, float))
        ]
        market_scores_list = [
            float(s["sentiment_score"])
            for s in raw_scores_all
            if s.get("type") == "market" and isinstance(s.get("sentiment_score"), (int, float))
        ]
        related_scores_list = [
            float(s["sentiment_score"])
            for s in raw_scores_all
            if s.get("type") == "related" and isinstance(s.get("sentiment_score"), (int, float))
        ]

        # Fall back to aggregated_sentiment for any stream with no data
        stock_stream_sentiment = _safe_mean(stock_scores_list) if stock_scores_list else aggregated_sentiment
        market_stream_sentiment = _safe_mean(market_scores_list) if market_scores_list else aggregated_sentiment
        related_stream_sentiment = _safe_mean(related_scores_list) if related_scores_list else aggregated_sentiment

        # Database queries for baselines and historical aggregates
        historical_daily_aggregates = get_historical_daily_aggregates(ticker, days=180)
        volume_baseline = get_volume_baseline(ticker, days=30)
        article_count_baseline = get_article_count_baseline(ticker, days=180)

        # Prepare volumes for detector (fall back to DB baseline if needed)
        if not historical_volumes and volume_baseline is not None:
            historical_volumes = [volume_baseline] * 30

        # 1) Volume anomaly
        volume_result = detect_volume_anomaly(
            current_volume=current_volume or 0.0,
            historical_volumes=historical_volumes,
            baseline_days=30,
        )

        # 2) Source reliability divergence
        reliability_result = detect_source_reliability_divergence(
            todays_articles=cleaned_stock_news,
            source_reliability_dict=source_reliability,
            historical_avg_reliability=historical_avg_reliability,
        )

        # 3) News velocity anomaly
        #
        # Three paths in descending preference:
        #
        #  A) Live API fetch  → use newly_fetched_count / fetch_window_days
        #     (true daily rate vs stored per-calendar-day baseline)
        #
        #  B) Cache / DB load (newly_fetched_count == 0) → no new articles
        #     arrived since the last run; velocity cannot spike right now.
        #     Report ratio = 0, detected = False.  Use the stored baseline
        #     from ticker_stats so the display is informative, not confusing.
        #
        #  C) Metadata missing entirely → fall back to batch comparison.
        #     This should never happen in production but is kept as a safety net.
        #
        # The stored daily_article_avg in news_fetch_metadata is the true
        # per-calendar-day mean (total_articles / days_since_oldest_article)
        # computed by compute_and_store_ticker_stats() in Node 2, so the
        # baseline unit matches the velocity unit in paths A and C.

        # Prefer the pre-computed baseline from Node 2's metadata (free — no DB hit).
        # Fall back to get_article_count_baseline() which uses the same ticker_stats cache.
        stored_daily_avg = news_metadata.get("daily_article_avg") or article_count_baseline

        if newly_fetched_count is not None and newly_fetched_count > 0 and fetch_window_days > 0:
            # Path A — live fetch
            velocity_article_count = newly_fetched_count
            velocity_window = fetch_window_days
            logger.info(
                f"Node 9B: Velocity (live fetch) — "
                f"{newly_fetched_count} articles / {fetch_window_days} day(s) = "
                f"{newly_fetched_count / max(1, fetch_window_days):.1f} articles/day "
                f"vs baseline {stored_daily_avg or 0:.2f}/day"
            )
            velocity_result = detect_news_velocity_anomaly(
                today_article_count=velocity_article_count,
                historical_daily_avg=stored_daily_avg,
                content_analysis_summary=content_summary,
                article_time_window_days=velocity_window,
            )
        elif newly_fetched_count == 0:
            # Path B — cache/DB hit: no new articles, velocity cannot spike
            logger.info(
                "Node 9B: Velocity (cache/DB hit) — "
                f"no new articles fetched; baseline = {stored_daily_avg or 0:.2f}/day. "
                "Velocity detection skipped."
            )
            velocity_result = _neutral_detector_result()
            velocity_result.update({
                "today_article_count": 0,
                "articles_per_day": 0.0,
                "time_window_days": 0,
                "historical_daily_avg": stored_daily_avg or 0.0,
                "velocity_ratio": 0.0,
                "high_anomaly_article_count": 0,
                "coordinated_keywords": False,
                "note": "No new articles fetched this run — velocity not applicable",
            })
        else:
            # Path C — metadata absent (safety net, should not occur in production)
            velocity_article_count = today_article_count
            velocity_window = 180
            logger.warning(
                "Node 9B: Velocity — news_fetch_metadata absent; "
                "falling back to batch comparison (less sensitive)"
            )
            velocity_result = detect_news_velocity_anomaly(
                today_article_count=velocity_article_count,
                historical_daily_avg=stored_daily_avg,
                content_analysis_summary=content_summary,
                article_time_window_days=velocity_window,
            )

        # 4) News-price divergence
        divergence_result = detect_news_price_divergence(
            sentiment_signal=sentiment_signal,
            sentiment_score=aggregated_sentiment,
            price_change=price_change_1d,
            todays_articles=cleaned_stock_news,
            source_reliability_dict=source_reliability,
        )

        # 5) Cross-stream coherence
        coherence_result = detect_cross_stream_incoherence(
            stock_sentiment=stock_stream_sentiment,
            market_sentiment=market_stream_sentiment,
            related_sentiment=related_stream_sentiment,
            stock_article_count=len(cleaned_stock_news),
            market_article_count=len(cleaned_market_news),
            related_article_count=len(cleaned_related_news),
            stock_avg_anomaly=stock_avg_anomaly,
        )

        # 6) Historical pattern matcher
        today_profile = {
            "article_count": len(cleaned_stock_news),
            "avg_composite_anomaly": stock_avg_anomaly,
            "avg_source_reliability": historical_avg_reliability,
            "volume_ratio": volume_result.get("volume_ratio", 1.0),
            "sentiment_direction": (
                "positive"
                if aggregated_sentiment > 0.2
                else "negative"
                if aggregated_sentiment < -0.2
                else "neutral"
            ),
            "price_direction": (
                "up"
                if price_change_1d > 0.5
                else "down"
                if price_change_1d < -0.5
                else "flat"
            ),
        }
        pattern_result = match_historical_patterns(
            today_profile=today_profile,
            historical_daily_aggregates=historical_daily_aggregates,
            similarity_threshold=0.75,
        )

        # 7) Composite score
        composite_result = calculate_pump_and_dump_score(
            volume_result=volume_result,
            reliability_result=reliability_result,
            velocity_result=velocity_result,
            divergence_result=divergence_result,
            coherence_result=coherence_result,
            pattern_result=pattern_result,
        )

        pump_and_dump_score = composite_result["pump_and_dump_score"]
        risk_level = composite_result["risk_level"]

        # Build user-facing alerts
        alerts: List[str] = []
        if volume_result.get("detected"):
            alerts.append(
                f"Volume {volume_result.get('volume_ratio', 1.0):.1f}x above 30-day baseline."
            )
        if reliability_result.get("detected"):
            divergence_pct = reliability_result.get("divergence", 0.0) * 100.0
            alerts.append(
                f"Average source reliability {divergence_pct:.0f}% below historical baseline."
            )
        if velocity_result.get("detected"):
            alerts.append(
                f"News velocity {velocity_result.get('velocity_ratio', 1.0):.1f}x normal with potential coordination."
            )
        if divergence_result.get("detected"):
            alerts.append(
                f"News-price divergence type {divergence_result.get('divergence_type')}: {divergence_result.get('explanation')}"
            )
        if coherence_result.get("detected"):
            alerts.append(coherence_result.get("explanation", "Cross-stream incoherence detected."))
        if pattern_result.get("detected"):
            pct_crash = (
                pattern_result.get("outcomes", {}).get("pct_ended_in_crash", 0.0) * 100.0
            )
            alerts.append(
                f"Similar historical patterns led to crashes {pct_crash:.0f}% of the time."
            )

        if risk_level == "CRITICAL":
            trading_recommendation = "DO_NOT_TRADE"
        elif risk_level == "HIGH":
            trading_recommendation = "CAUTION"
        else:
            trading_recommendation = "NORMAL"

        if risk_level in ["HIGH", "CRITICAL"]:
            behavioral_summary = "Multiple behavioral manipulation signals detected."
        elif risk_level == "MEDIUM":
            behavioral_summary = "Some unusual behavioral patterns detected; monitor closely."
        else:
            behavioral_summary = "No significant behavioral anomalies detected."

        elapsed = (datetime.now() - start_time).total_seconds()

        state["behavioral_anomaly_detection"] = {
            "risk_level": risk_level,
            "pump_and_dump_score": pump_and_dump_score,
            "trading_recommendation": trading_recommendation,
            "behavioral_summary": behavioral_summary,
            "volume_anomaly": volume_result,
            "source_reliability_divergence": reliability_result,
            "news_velocity_anomaly": velocity_result,
            "news_price_divergence": divergence_result,
            "cross_stream_coherence": coherence_result,
            "historical_pattern_match": pattern_result,
            "detection_breakdown": composite_result["detection_breakdown"],
            "primary_risk_factors": composite_result["primary_risk_factors"],
            "alerts": alerts,
            "execution_time": elapsed,
            "analysis_timestamp": datetime.now().isoformat(),
        }

        # Track execution time in node_execution_times
        state["node_execution_times"]["node_9b"] = elapsed

        logger.info(
            f"Node 9B: Completed for {ticker} in {elapsed:.2f}s "
            f"(risk_level={risk_level}, pump_and_dump_score={pump_and_dump_score})"
        )
        return state

    except Exception as e:
        # Never break the graph: log, record error, set safe defaults, always return state
        logger.error(f"Node 9B failed for {ticker}: {str(e)}")
        state["errors"].append(f"Node 9B failed: {str(e)}")

        elapsed = (datetime.now() - start_time).total_seconds()
        state["node_execution_times"]["node_9b"] = elapsed

        # Safe fallback behavioral_anomaly_detection
        state["behavioral_anomaly_detection"] = {
            "risk_level": "LOW",
            "pump_and_dump_score": 0,
            "trading_recommendation": "NORMAL",
            "behavioral_summary": "Behavioral anomaly detection failed; defaulting to LOW risk.",
            "volume_anomaly": _neutral_detector_result(),
            "source_reliability_divergence": _neutral_detector_result(),
            "news_velocity_anomaly": _neutral_detector_result(),
            "news_price_divergence": _neutral_detector_result(),
            "cross_stream_coherence": _neutral_detector_result(),
            "historical_pattern_match": _neutral_detector_result(),
            "detection_breakdown": {
                "volume_anomaly": 0,
                "source_reliability_divergence": 0,
                "news_velocity_anomaly": 0,
                "news_price_divergence": 0,
                "cross_stream_incoherence": 0,
                "historical_pattern_match": 0,
            },
            "primary_risk_factors": [],
            "alerts": ["Behavioral anomaly detection node encountered an error."],
            "execution_time": elapsed,
            "analysis_timestamp": datetime.now().isoformat(),
        }

        return state

