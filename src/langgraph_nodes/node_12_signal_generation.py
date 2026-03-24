"""
Node 12: Final Signal Generation

TWO JOBS:

Job 1 — Weighted Signal Combination (unchanged):
  Reads continuous scores from 4 analytical streams (technical, sentiment,
  market context, monte carlo), applies adaptive weights from Node 11, and
  emits a final BUY/SELL/HOLD recommendation with confidence.

  Continuous scoring on [-1, +1]:
    Technical:    confidence × sign(signal)      e.g. BUY 0.72  → +0.72
    Sentiment:    aggregated_sentiment directly   e.g. +0.61     → +0.61
    Market:       context_score × correlation     e.g. BUY 0.82  → +0.82
    Monte Carlo:  (probability_up - 0.5) × 2     e.g. p=0.68    → +0.36

  Thresholds:
    final_score > +0.15 → BUY
    final_score < -0.15 → SELL
    else                → HOLD
    raw_confidence = min(1.0, abs(final_score))

Job 2 — Signal-Conditioned Historical Pattern Matching (NEW):
  Builds a 4-dimensional signal vector for today:
    [tech_score, stock_sentiment, market_sentiment, related_sentiment]

  Joins Node 10's per-stream daily_results by date to form the same
  4-dimensional vectors for each historical trading day, then runs
  inverse-distance weighted kNN using Node 11's adaptive weights as
  dimension importances — creating a direct feedback loop:
    Node 8 learning → Node 11 weights → Job 2 similarity quality

  The empirical outcomes (prob_up, expected_return_7d, percentiles) from
  similar historical days are blended with Node 7's GBM Monte Carlo estimate.
  Blend weight is data-driven: job2_weight = min(0.70, days_found/25 × 0.70).

  Job 1 and Job 2 results are compared and stored as agreement_level
  ('strong', 'mild', 'disagree', 'neutral') — a transparency flag for Nodes
  13/14 only. It does NOT modify final_confidence.

  If backtest_results is None or fewer than MIN_SIMILAR_DAYS days pass the
  similarity threshold, Job 2 sets sufficient_data=False and the confidence
  multiplier stays 1.0 (no change). Job 1 always runs regardless.

Error handling:
  - Any missing stream scores 0.0; its name is added to streams_missing.
  - adaptive_weights None → equal weights 0.25 each.
  - All 4 streams missing → HOLD, confidence 0.0, error appended.
  - Job 2 failures are isolated; Job 1 output is never affected.

Runs AFTER: Node 11 (adaptive_weights)
Runs BEFORE: Node 13 (LLM beginner explanation)
"""

import math
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_node_logger

logger = get_node_logger("node_12")


# ============================================================================
# CONSTANTS
# ============================================================================

# Job 1 — signal thresholds
BUY_THRESHOLD: float = 0.15
SELL_THRESHOLD: float = -0.15
EQUAL_WEIGHT: float = 0.25

# Minimum weight floor so no stream is ever fully silenced.
# Anti-predictive streams (weight_source == "neutral_anti_predictive") are exempt.
MIN_STREAM_WEIGHT: float = 0.05

# Maps signal strings to directional signs on [-1, +1]
SIGNAL_TO_SCORE: Dict[str, float] = {
    "BUY":  1.0,
    "SELL": -1.0,
    "HOLD": 0.0,
}

# Maps Node 5 sentiment labels to trading signals
SENTIMENT_TO_SIGNAL: Dict[str, str] = {
    "POSITIVE": "BUY",
    "NEGATIVE": "SELL",
    "NEUTRAL":  "HOLD",
}

# Default equal-weights fallback when adaptive_weights is None
_EQUAL_WEIGHTS_FALLBACK: Dict[str, float] = {
    "technical_weight":    EQUAL_WEIGHT,
    "stock_news_weight":   EQUAL_WEIGHT,
    "market_news_weight":  EQUAL_WEIGHT,
    "related_news_weight": EQUAL_WEIGHT,
}

# Job 2 — historical pattern matching
MIN_SNAPSHOTS_FOR_JOB2: int = 20             # minimum historical snapshots to attempt
PRIMARY_SIMILARITY_THRESHOLD: float = 0.65   # strong-match threshold
SECONDARY_SIMILARITY_THRESHOLD: float = 0.50 # fallback threshold
MIN_SIMILAR_DAYS: int = 5                    # minimum matches to produce a prediction
MAX_SIMILAR_DAYS: int = 25                   # cap to avoid noise from weak matches

# Job 1/2 agreement — used only as a transparency flag, not to adjust confidence


# ============================================================================
# HELPER 1: TECHNICAL STREAM SCORING
# ============================================================================

def _score_technical(state: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Convert Node 4's continuous normalised score to a stream score on [-1, +1].

    Node 4 no longer emits a discrete BUY/SELL/HOLD string. Instead it writes
    ``normalized_score`` (0–100) into ``technical_indicators``. This function
    maps that value linearly:

        tech_score = (normalized_score - 50) / 50

    So score=100 → +1.0 (strong buy), score=0 → -1.0 (strong sell),
    score=50 → 0.0 (perfectly neutral).

    This is strictly more informative than the old BUY/SELL/HOLD × confidence
    approach — a mild but consistent bearish score of 44 now yields -0.12
    instead of 0.0 (HOLD).

    Also logs hold_low / hold_high and market_regime from Node 4 for debugging.

    Args:
        state: LangGraph state dict.

    Returns:
        (score, is_missing) — is_missing=True when normalized_score is absent.
    """
    ti: Optional[Dict[str, Any]] = state.get("technical_indicators")

    if ti is None:
        logger.warning("  Technical stream: technical_indicators missing → score=0.0")
        return 0.0, True

    # Prefer technical_alpha (regression output) — fall back to normalized_score
    alpha: Optional[float] = ti.get("technical_alpha")
    if alpha is not None:
        score = float(np.clip(float(alpha), -1.0, 1.0))
        source = "alpha"
    else:
        normalized_score: Optional[float] = ti.get("normalized_score")
        if normalized_score is None:
            logger.warning("  Technical stream: neither technical_alpha nor normalized_score found → score=0.0")
            return 0.0, True
        score = (float(normalized_score) - 50.0) / 50.0
        source = "normalized_score"

    hold_low: Optional[float] = ti.get("hold_low")
    hold_high: Optional[float] = ti.get("hold_high")
    market_regime: Optional[str] = ti.get("market_regime")
    ic_score: Optional[float] = ti.get("ic_score")
    regression_valid: Optional[bool] = ti.get("regression_valid")

    logger.debug(
        f"  Technical ({source}): score={score:+.3f} "
        f"(band=[{hold_low},{hold_high}] regime={market_regime} "
        f"regression={regression_valid} IC={ic_score})"
    )
    return score, False


# ============================================================================
# HELPER 2: SENTIMENT STREAM SCORING
# ============================================================================

def _score_sentiment(state: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Return the continuous sentiment score on [-1, +1].

    Node 5 writes aggregated_sentiment as a signed float already on [-1, +1].
    Node 8 may have adjusted the sentiment_analysis dict. We prefer Node 8's
    aggregated_sentiment when available.

    Args:
        state: LangGraph state dict.

    Returns:
        (score, is_missing) — is_missing=True when no valid score is found.
    """
    # Node 8 override has priority
    sentiment_analysis: Optional[Dict[str, Any]] = state.get("sentiment_analysis")
    sentiment_val: Optional[float] = None
    if sentiment_analysis is not None:
        sentiment_val = sentiment_analysis.get("aggregated_sentiment")

    # Fall back to Node 5's direct output
    if sentiment_val is None:
        sentiment_val = state.get("aggregated_sentiment")

    if sentiment_val is None:
        logger.warning("  Sentiment stream: no aggregated_sentiment → score=0.0")
        return 0.0, True

    score = float(sentiment_val)
    logger.debug(f"  Sentiment: aggregated={score:+.3f}")
    return score, False


# ============================================================================
# HELPER 3: MARKET CONTEXT STREAM SCORING
# ============================================================================

def _score_market(state: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Convert Node 6's market_context to a continuous score on [-1, +1].

    Reads from Node 6's actual sub-dict structure:
      market_regime              → SPY returns, VIX level, regime_label
      sector_industry_context    → sector_return_5d, relative_strength_label
      market_correlation_profile → market_correlation, beta_calculated
      news_sentiment_context     → market_news_sentiment

    Components and weights (sum to 1.0):
      SPY 5d return    40%  normalised by /5.0  (5% move → ±1.0)
      VIX fear         28%  (20 - vix) / 20     capped [-1, +0.3]
      Sector 5d return 22%  normalised by /3.0  (3% move → ±1.0)
      Market news      10%  already on [-1, +1]

    Composite is multiplied by abs(market_correlation) so a low-beta
    stock is less dragged by macro headwinds.

    Args:
        state: LangGraph state dict.

    Returns:
        (score, is_missing) — is_missing=True when market_context is absent.
    """
    market_context: Optional[Dict[str, Any]] = state.get("market_context")
    if market_context is None:
        logger.warning("  Market stream: no market_context → score=0.0")
        return 0.0, True

    corr_profile   = market_context.get("market_correlation_profile") or {}
    market_regime  = market_context.get("market_regime")              or {}
    sector_context = market_context.get("sector_industry_context")    or {}
    news_context   = market_context.get("news_sentiment_context")     or {}

    market_corr = float(corr_profile.get("market_correlation") or 1.0)
    if not math.isfinite(market_corr):
        market_corr = 1.0

    spy_return_5d    = float(market_regime.get("spy_return_5d")        or 0.0)
    vix_level        = float(market_regime.get("vix_level")            or 20.0)
    sector_return_5d = float(sector_context.get("sector_return_5d")    or 0.0)
    news_sentiment   = float(news_context.get("market_news_sentiment") or 0.0)

    spy_score    = max(-1.0, min(1.0,  spy_return_5d    / 5.0))
    vix_score    = max(-1.0, min(0.3, (20.0 - vix_level) / 20.0))
    sector_score = max(-1.0, min(1.0,  sector_return_5d / 3.0))

    raw = (
        0.40 * spy_score
        + 0.28 * vix_score
        + 0.22 * sector_score
        + 0.10 * news_sentiment
    )
    composite = max(-1.0, min(1.0, raw))
    score = composite * min(abs(market_corr), 1.0)

    logger.debug(
        f"  Market: SPY5d={spy_return_5d:+.2f}%(→{spy_score:+.3f}) "
        f"VIX={vix_level:.1f}(→{vix_score:+.3f}) "
        f"sector5d={sector_return_5d:+.2f}%(→{sector_score:+.3f}) "
        f"news={news_sentiment:+.3f} "
        f"→ composite={composite:+.3f} × corr={market_corr:.3f} = {score:+.3f} "
        f"| regime={market_regime.get('regime_label','?')} "
        f"rel={sector_context.get('relative_strength_label','?')} "
        f"beta={corr_profile.get('beta_calculated', 1.0):.2f}"
    )
    return score, False


# ============================================================================
# HELPER 4: MONTE CARLO STREAM SCORING
# ============================================================================

def _score_monte_carlo(state: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Convert Node 7's Monte Carlo probability_up to a continuous score on [-1, +1].

    Mapping: score = (probability_up - 0.5) × 2
      probability_up=1.0 → +1.0  (certain upside)
      probability_up=0.5 →  0.0  (coin-flip)
      probability_up=0.0 → -1.0  (certain downside)

    Args:
        state: LangGraph state dict.

    Returns:
        (score, is_missing) — is_missing=True when monte_carlo_results is absent.
    """
    mc: Optional[Dict[str, Any]] = state.get("monte_carlo_results")
    if mc is None:
        logger.warning("  Monte Carlo stream: no monte_carlo_results → score=0.0")
        return 0.0, True

    prob_up: Optional[float] = mc.get("probability_up")
    if prob_up is None:
        logger.warning("  Monte Carlo stream: probability_up missing → score=0.0")
        return 0.0, True

    score = (float(prob_up) - 0.5) * 2.0
    logger.debug(f"  Monte Carlo: probability_up={prob_up:.3f} → {score:+.3f}")
    return score, False


# ============================================================================
# HELPER 4b: RELATED SENTIMENT STREAM SCORING
# ============================================================================

def _score_related_sentiment(state: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Return the related-company news sentiment score on [-1, +1].

    Reads related_news_sentiment from Node 5's sentiment_analysis dict.
    This replaces Monte Carlo as the 4th stream in Job 1 — related news
    is a backtested stream in Node 10/11 with an earned weight, unlike
    GBM which has no predictive weight from backtesting.

    Args:
        state: LangGraph state dict.

    Returns:
        (score, is_missing)
    """
    sentiment_analysis: Optional[Dict[str, Any]] = state.get("sentiment_analysis")
    related_val: Optional[float] = None

    if sentiment_analysis is not None:
        related_val = sentiment_analysis.get("related_news_sentiment")

    if related_val is None:
        related_val = state.get("related_news_sentiment")

    if related_val is None:
        logger.warning("  Related news stream: no related_news_sentiment → score=0.0")
        return 0.0, True

    score = float(np.clip(float(related_val), -1.0, 1.0))
    logger.debug(f"  Related news: score={score:+.3f}")
    return score, False


# ============================================================================
# HELPER 5: SIGNAL CLASSIFICATION
# ============================================================================

def _classify_signal(final_score: float, threshold: float = BUY_THRESHOLD) -> str:
    """
    Convert a weighted final score to a BUY/SELL/HOLD label.

    The threshold is symmetric: > +threshold → BUY | < -threshold → SELL | else HOLD.
    Node 11 may supply a learnt threshold via adaptive_weights["signal_threshold"];
    if absent the module-level BUY_THRESHOLD (0.15) is used as the default.

    Args:
        final_score: Weighted sum of stream scores on [-1, +1].
        threshold:   Classification boundary; defaults to BUY_THRESHOLD (0.15).

    Returns:
        'BUY', 'SELL', or 'HOLD'.
    """
    if final_score > threshold:
        return "BUY"
    if final_score < -threshold:
        return "SELL"
    return "HOLD"


# ============================================================================
# HELPER 6: SIGNAL AGREEMENT COUNT
# ============================================================================

def _count_signal_agreement(
    stream_scores: Dict[str, Dict[str, float]],
    final_signal: str,
) -> int:
    """
    Count how many of the 4 streams agree with the final signal direction.

    Agreement:
      BUY  → stream raw_score > 0
      SELL → stream raw_score < 0
      HOLD → abs(stream raw_score) <= BUY_THRESHOLD

    Args:
        stream_scores: Dict of stream data including 'raw_score'.
        final_signal:  'BUY', 'SELL', or 'HOLD'.

    Returns:
        Integer 0–4.
    """
    count = 0
    for data in stream_scores.values():
        raw = data["raw_score"]
        if final_signal == "BUY" and raw > 0:
            count += 1
        elif final_signal == "SELL" and raw < 0:
            count += 1
        elif final_signal == "HOLD" and abs(raw) <= BUY_THRESHOLD:
            count += 1
    return count


# ============================================================================
# HELPER 6b: SIGNAL STRENGTH
# ============================================================================

def _compute_signal_strength(
    stream_scores: Dict[str, Dict[str, float]],
    signal_agreement: int,
) -> int:
    """
    Compute how loud and united the 4 streams are on a 0–100 scale.

    Two components:
      Intensity (70%): weighted average of abs(raw_score) across all streams.
        Each stream contributes its own weight × abs(raw_score).
        A stream scoring ±1.0 at full weight = maximum intensity.

      Agreement (30%): fraction of streams pointing the same direction.
        4/4 agree = 1.0, 3/4 = 0.75, 2/4 = 0.50, 1/4 = 0.25, 0/4 = 0.0

    Returns integer 0–100 for clean UI display.

    Args:
        stream_scores:    Dict of stream data including 'raw_score' and 'weight'.
        signal_agreement: Number of streams agreeing with the final signal (0–4).

    Returns:
        Integer 0–100.
    """
    intensity = sum(
        abs(data["raw_score"]) * data["weight"]
        for data in stream_scores.values()
    )
    agreement_ratio = signal_agreement / 4.0
    raw = (0.70 * intensity) + (0.30 * agreement_ratio)
    return int(round(min(1.0, raw) * 100))


# ============================================================================
# HELPER 6c: TRUSTWORTHINESS
# ============================================================================

def _compute_trustworthiness(
    stream_scores: Dict[str, Dict[str, float]],
    adaptive_weights: Dict[str, Any],
    pattern_prediction: Dict[str, Any],
    w_tech: float,
    w_sent: float,
    w_market: float,
    w_related: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute how trustworthy the final recommendation is (0.0–1.0).

    Three components and their weights:
      Stream reliability  50%  — weighted hit rates from Node 11 backtesting.
                                  Each stream's hit_rate × its adaptive weight.
                                  Falls back to PRIOR (0.55) per stream when
                                  Node 11 has insufficient data (flagged by
                                  'fallback_equal_weights' or missing hit_rate
                                  or sample_count < MIN_SAMPLES_FOR_RELIABILITY).
      Stream agreement    30%  — fraction of streams agreeing with the signal.
                                  4/4 = 1.0 … 0/4 = 0.0
      Pattern backing     20%  — Job 2 empirical agreement with Job 1 signal.
                                  strong=1.0, mild=0.70, neutral=0.50,
                                  disagree=0.20, insufficient data=0.50 (neutral)

    PRIOR = 0.55: slightly above coin-flip — streams are expected to have
    some edge even before backtesting confirms it.

    Also returns a breakdown dict for UI display and Node 13/14 explanations.
    The 'insufficient_history' flag tells the UI to show "building track record"
    rather than displaying the numeric score as if it reflects real accuracy.

    Args:
        stream_scores:      Dict of stream data with 'raw_score' and 'contribution'.
        adaptive_weights:   Node 11 output dict (hit_rates, sample_counts, weights).
        pattern_prediction: Job 2 output dict with 'agreement_with_job1'.
        w_tech / w_sent / w_market / w_related: Normalized adaptive weights.

    Returns:
        (score, breakdown_dict)
    """
    PRIOR: float = 0.55
    MIN_SAMPLES_FOR_RELIABILITY: int = 10

    # --- Component 1: stream reliability ---
    def _hit_rate(key: str) -> Tuple[float, bool]:
        val = adaptive_weights.get(key)
        n = int(adaptive_weights.get(key.replace("hit_rate", "sample_count"), 0) or 0)
        if val is None:
            return PRIOR, True
        # Bayesian blend — same formula as Node 11: pull toward PRIOR when n is small
        blended = (float(val) * n + PRIOR * 10) / (n + 10)
        using_prior = n < MIN_SAMPLES_FOR_RELIABILITY
        return round(blended, 4), using_prior

    tech_hr,    tech_prior    = _hit_rate("technical_hit_rate")
    sent_hr,    sent_prior    = _hit_rate("stock_news_hit_rate")
    market_hr,  market_prior  = _hit_rate("market_news_hit_rate")
    related_hr, related_prior = _hit_rate("related_news_hit_rate")

    reliability_score: float = (
        tech_hr    * w_tech
        + sent_hr    * w_sent
        + market_hr  * w_market
        + related_hr * w_related
    )
    # Only count a stream as "using prior" when it actually carries weight.
    # Neutralised/anti-predictive streams have None hit_rate (prior=True) but
    # weight=0 — they don't influence the signal, so they must not hide the
    # real backtested accuracy of the streams that do contribute.
    _WEIGHT_THRESHOLD = 0.01
    weighted_prior_exposure = (
        (tech_prior    * w_tech)    +
        (sent_prior    * w_sent)    +
        (market_prior  * w_market)  +
        (related_prior * w_related)
    )
    using_prior: bool = weighted_prior_exposure > 0.30  # 30% threshold for "using prior"

    # --- Component 2: stream agreement ---
    agreement_count = sum(
        1 for data in stream_scores.values()
        if (data["raw_score"] > 0 and data["contribution"] > 0)
        or (data["raw_score"] < 0 and data["contribution"] < 0)
        or (abs(data["raw_score"]) <= BUY_THRESHOLD)
    )
    agreement_ratio: float = agreement_count / 4.0

    # --- Component 3: pattern backing ---
    agreement_level: str = pattern_prediction.get("agreement_with_job1", "neutral")
    pattern_map: Dict[str, float] = {
        "strong":  1.00,
        "mild":    0.70,
        "neutral": 0.50,
        "disagree": 0.20,
    }
    pattern_score: float = (
        0.50  # neutral — no Job 2 data is not a penalty
        if not pattern_prediction.get("sufficient_data")
        else pattern_map.get(agreement_level, 0.50)
    )

    # --- Final weighted score ---
    score = round(float(np.clip(
        0.50 * reliability_score
        + 0.30 * agreement_ratio
        + 0.20 * pattern_score,
        0.0, 1.0,
    )), 4)

    breakdown: Dict[str, Any] = {
        "reliability_score":   round(reliability_score, 4),
        "agreement_score":     round(agreement_ratio,   4),
        "pattern_score":       round(pattern_score,     4),
        "stream_hit_rates": {
            "technical":    {"hit_rate": tech_hr,    "using_prior": tech_prior},
            "sentiment":    {"hit_rate": sent_hr,    "using_prior": sent_prior},
            "market":       {"hit_rate": market_hr,  "using_prior": market_prior},
            "related_news": {"hit_rate": related_hr, "using_prior": related_prior},
        },
        "using_prior":          using_prior,
        "insufficient_history": using_prior,
    }

    return score, breakdown


# ============================================================================
# HELPER 7: RISK SUMMARY (enhanced with full Node 9B data)
# ============================================================================

def _build_risk_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate risk information from Nodes 9A and 9B into a single dict.

    trading_safe is True when neither behavioral risk is HIGH/CRITICAL
    nor pump_and_dump_score exceeds 70. Node 12 always runs regardless —
    blocking logic lives in the workflow's conditional edges.

    Full Node 9B fields (primary_risk_factors, alerts, detection_breakdown,
    behavioral_summary, trading_recommendation) are passed through so that
    Nodes 13 and 14 can use them for detailed explanations.

    Args:
        state: LangGraph state dict.

    Returns:
        Consolidated risk dict.
    """
    behavioral: Dict[str, Any] = state.get("behavioral_anomaly_detection") or {}
    content: Dict[str, Any] = state.get("content_analysis_summary") or {}

    risk_level: str = behavioral.get("risk_level", "UNKNOWN")
    pump_score: int = int(behavioral.get("pump_and_dump_score", 0) or 0)
    content_risk: str = content.get("early_risk_level", "UNKNOWN")

    trading_safe: bool = (
        risk_level not in ("HIGH", "CRITICAL") and pump_score < 70
    )

    return {
        # Core fields (backward-compatible)
        "overall_risk_level":     risk_level,
        "pump_and_dump_score":    pump_score,
        "behavioral_risk":        risk_level,
        "content_risk":           content_risk,
        "trading_safe":           trading_safe,
        # Full Node 9B pass-through for Nodes 13/14
        "behavioral_summary":     behavioral.get("behavioral_summary", ""),
        "trading_recommendation": behavioral.get("trading_recommendation", "NORMAL"),
        "primary_risk_factors":   behavioral.get("primary_risk_factors", []),
        "alerts":                 behavioral.get("alerts", []),
        "detection_breakdown":    behavioral.get("detection_breakdown", {}),
    }


# ============================================================================
# HELPER 8: PRICE TARGETS
# ============================================================================

def _build_price_targets(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect price forecast information from Node 7's Monte Carlo output.

    price_range (a tuple written by Node 7) takes priority for lower/upper.
    Falls back to confidence_95 dict inside monte_carlo_results.

    Args:
        state: LangGraph state dict.

    Returns:
        Dict with current_price, forecasted_price, range, expected_return_pct.
    """
    mc: Dict[str, Any] = state.get("monte_carlo_results") or {}
    price_range: Optional[tuple] = state.get("price_range")

    if price_range is not None and len(price_range) >= 2:
        lower = price_range[0]
        upper = price_range[1]
    else:
        ci = mc.get("confidence_95") or {}
        lower = ci.get("lower")
        upper = ci.get("upper")

    return {
        "current_price":       mc.get("current_price"),
        "forecasted_price":    state.get("forecasted_price"),
        "price_range_lower":   lower,
        "price_range_upper":   upper,
        "expected_return_pct": mc.get("expected_return"),
    }


# ============================================================================
# HELPER 9: SIGNED TECHNICAL SCORE (for historical snapshot building)
# ============================================================================

def _signed_tech_score(record: Dict[str, Any]) -> float:
    """
    Convert a Node 10 technical daily_result record to a signed score [-1, +1].

    BUY  + confidence → +confidence
    SELL + confidence → -confidence
    HOLD             →  0.0

    Args:
        record: Dict from Node 10's technical stream daily_results.

    Returns:
        Signed score on [-1, +1].
    """
    signal: str = record.get("signal", "HOLD") or "HOLD"
    confidence: float = float(record.get("confidence") or 0.0)
    if signal == "BUY":
        return confidence
    if signal == "SELL":
        return -confidence
    return 0.0


# ============================================================================
# HELPER 10: BUILD HISTORICAL SNAPSHOTS FROM NODE 10 DAILY RESULTS
# ============================================================================

def _build_historical_snapshots(
    backtest_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Join Node 10's four per-stream daily_results lists by date into unified
    4-dimensional signal snapshots.

    Each snapshot has the same dimensions used for Job 1's adaptive weights:
        tech_score:        signed technical confidence [-1, +1]
        stock_sentiment:   daily stock-news sentiment  [-1, +1]
        market_sentiment:  daily market-news sentiment [-1, +1]
        related_sentiment: daily related-news sentiment[-1, +1]
        actual_change_7d:  actual 7-day price change (%)

    Only dates with at least one data point and a valid actual_change_7d
    are included. Missing streams default to 0.0 (neutral) for that day.

    Args:
        backtest_results: Dict produced by Node 10's backtesting_node().

    Returns:
        List of snapshot dicts, one per trading day.
    """
    technical_stream  = backtest_results.get("technical")    or {}
    stock_stream      = backtest_results.get("stock_news")   or {}
    market_stream     = backtest_results.get("market_news")  or {}
    related_stream    = backtest_results.get("related_news") or {}

    tech_daily:    List[Dict] = technical_stream.get("daily_results",  []) or []
    stock_daily:   List[Dict] = stock_stream.get("daily_results",      []) or []
    market_daily:  List[Dict] = market_stream.get("daily_results",     []) or []
    related_daily: List[Dict] = related_stream.get("daily_results",    []) or []

    # Build date-keyed lookups for O(1) join
    tech_by_date:    Dict[str, Dict] = {r["date"]: r for r in tech_daily    if "date" in r}
    stock_by_date:   Dict[str, Dict] = {r["date"]: r for r in stock_daily   if "date" in r}
    market_by_date:  Dict[str, Dict] = {r["date"]: r for r in market_daily  if "date" in r}
    related_by_date: Dict[str, Dict] = {r["date"]: r for r in related_daily if "date" in r}

    all_dates = (
        set(tech_by_date)
        | set(stock_by_date)
        | set(market_by_date)
        | set(related_by_date)
    )

    snapshots: List[Dict[str, Any]] = []
    for date in sorted(all_dates):
        tech_rec    = tech_by_date.get(date)
        stock_rec   = stock_by_date.get(date)
        market_rec  = market_by_date.get(date)
        related_rec = related_by_date.get(date)

        # actual_change_7d: take from first available record
        source = tech_rec or stock_rec or market_rec or related_rec
        if source is None or source.get("actual_change_7d") is None:
            continue

        snapshots.append({
            "date":              date,
            "tech_score":        _signed_tech_score(tech_rec)                     if tech_rec    else 0.0,
            "stock_sentiment":   float(stock_rec.get("sentiment_score")   or 0.0) if stock_rec   else 0.0,
            "market_sentiment":  float(market_rec.get("sentiment_score")  or 0.0) if market_rec  else 0.0,
            "related_sentiment": float(related_rec.get("sentiment_score") or 0.0) if related_rec else 0.0,
            "actual_change_7d":  float(source["actual_change_7d"]),
        })

    logger.debug(f"  Built {len(snapshots)} historical snapshots from Node 10 daily_results")
    return snapshots


# ============================================================================
# HELPER 11: BUILD TODAY'S SIGNAL VECTOR
# ============================================================================

def _get_today_vector(
    state: Dict[str, Any],
    tech_score: float,
) -> Dict[str, float]:
    """
    Build today's 4-dimensional signal vector for the Job 2 similarity search.

    Dimensions match exactly what _build_historical_snapshots produces:
        tech_score, stock_sentiment, market_sentiment, related_sentiment

    Prefers per-stream scores from sentiment_analysis (Node 8 output).
    Falls back to aggregated_sentiment for stock_sentiment, 0.0 for the
    others when per-stream breakdown is unavailable.

    Args:
        state:      LangGraph state dict.
        tech_score: Already-computed technical score from _score_technical().

    Returns:
        Dict with four float dimensions.
    """
    sentiment_analysis: Dict[str, Any] = state.get("sentiment_analysis") or {}

    stock_sent:   Optional[float] = sentiment_analysis.get("stock_news_sentiment")
    market_sent:  Optional[float] = sentiment_analysis.get("market_news_sentiment")
    related_sent: Optional[float] = sentiment_analysis.get("related_news_sentiment")

    agg_sent: float = float(state.get("aggregated_sentiment") or 0.0)

    return {
        "tech_score":        tech_score,
        "stock_sentiment":   float(stock_sent)   if stock_sent   is not None else agg_sent,
        "market_sentiment":  float(market_sent)  if market_sent  is not None else 0.0,
        "related_sentiment": float(related_sent) if related_sent is not None else 0.0,
    }


# ============================================================================
# HELPER 12: COMPUTE WEIGHTED SIMILARITY
# ============================================================================

def _compute_weighted_similarity(
    today: Dict[str, float],
    hist: Dict[str, float],
    w_tech: float,
    w_stock: float,
    w_market: float,
    w_related: float,
) -> float:
    """
    Compute adaptive-weighted similarity between today's signal vector and a
    historical day's signal vector.

    Per-dimension similarity = 1.0 - abs(today - hist) / 2.0
    (All dimensions on [-1, +1] → max difference = 2.0 → similarity in [0, 1])

    Total similarity is the weighted sum using Node 11's adaptive weights as
    dimension importances: streams proven more accurate in backtesting carry
    more weight when deciding how similar two days are.

    Args:
        today:     Today's 4-dimensional signal vector.
        hist:      Historical day's 4-dimensional signal vector.
        w_tech / w_stock / w_market / w_related: Adaptive weights.

    Returns:
        Similarity score in [0, 1].
    """
    def _dim(a: float, b: float) -> float:
        return 1.0 - abs(a - b) / 2.0

    return (
        _dim(today["tech_score"],        hist["tech_score"])         * w_tech
        + _dim(today["stock_sentiment"],   hist["stock_sentiment"])   * w_stock
        + _dim(today["market_sentiment"],  hist["market_sentiment"])  * w_market
        + _dim(today["related_sentiment"], hist["related_sentiment"]) * w_related
    )


# ============================================================================
# HELPER 13: FIND SIMILAR HISTORICAL DAYS (inverse-distance weighted kNN)
# ============================================================================

def _find_similar_days(
    today: Dict[str, float],
    snapshots: List[Dict[str, Any]],
    w_tech: float,
    w_stock: float,
    w_market: float,
    w_related: float,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Find historical days most similar to today using inverse-distance weighted kNN.

    Steps:
    1. Compute adaptive-weighted similarity for every historical snapshot.
    2. Try PRIMARY_SIMILARITY_THRESHOLD (0.65); fall back to SECONDARY (0.50)
       if fewer than MIN_SIMILAR_DAYS qualify.
    3. Keep at most MAX_SIMILAR_DAYS by descending similarity.
    4. Assign inverse-distance weights (idw_weight) so more similar days
       contribute proportionally more to the prediction statistics.

    Args:
        today:     Today's 4-dimensional signal vector.
        snapshots: Historical snapshots from _build_historical_snapshots().
        w_tech / w_stock / w_market / w_related: Adaptive weights.

    Returns:
        (similar_days_with_idw_weight, threshold_used)
        similar_days is empty when fewer than MIN_SIMILAR_DAYS qualify.
    """
    if not snapshots:
        return [], 0.0

    scored: List[Dict[str, Any]] = []
    for snap in snapshots:
        sim = _compute_weighted_similarity(
            today, snap, w_tech, w_stock, w_market, w_related
        )
        scored.append({**snap, "similarity": round(sim, 4)})

    scored.sort(key=lambda x: x["similarity"], reverse=True)

    # Try primary threshold first
    threshold = PRIMARY_SIMILARITY_THRESHOLD
    pool = [s for s in scored if s["similarity"] >= threshold]

    if len(pool) < MIN_SIMILAR_DAYS:
        threshold = SECONDARY_SIMILARITY_THRESHOLD
        pool = [s for s in scored if s["similarity"] >= threshold]

    if len(pool) < MIN_SIMILAR_DAYS:
        logger.debug(
            f"  Job 2: only {len(pool)} days above secondary threshold "
            f"{threshold:.2f} (need {MIN_SIMILAR_DAYS}) → insufficient"
        )
        return [], threshold

    pool = pool[:MAX_SIMILAR_DAYS]

    # Inverse-distance weights (sum to 1.0)
    total_sim = sum(d["similarity"] for d in pool)
    for d in pool:
        d["idw_weight"] = d["similarity"] / total_sim if total_sim > 0 else 1.0 / len(pool)

    logger.debug(
        f"  Job 2: {len(pool)} similar days (threshold={threshold:.2f}), "
        f"top sim={pool[0]['similarity']:.3f}"
    )
    return pool, threshold


# ============================================================================
# HELPER 14: BUILD PATTERN PREDICTION FROM SIMILAR DAYS
# ============================================================================

def _build_pattern_prediction(
    similar_days: List[Dict[str, Any]],
    threshold_used: float,
) -> Dict[str, Any]:
    """
    Compute empirical prediction statistics from similar historical days.

    Uses inverse-distance weights (idw_weight) so more similar days
    contribute proportionally more to prob_up and expected_return_7d.
    Percentiles are computed from the raw sorted pool for interpretability.

    Args:
        similar_days:   List of similar-day dicts (each has idw_weight).
        threshold_used: Similarity threshold that filtered the pool.

    Returns:
        pattern_prediction dict. sufficient_data=False when list is empty.
    """
    if not similar_days:
        return {
            "sufficient_data":    False,
            "reason":             "No similar historical days found above threshold",
            "similar_days_found": 0,
        }

    weights: List[float] = [d["idw_weight"]       for d in similar_days]
    changes: List[float] = [d["actual_change_7d"]  for d in similar_days]

    # Weighted probability of upward move
    prob_up: float = sum(
        w for d, w in zip(similar_days, weights)
        if d["actual_change_7d"] > 0
    )

    # Weighted expected return
    expected_return: float = sum(
        d["actual_change_7d"] * w
        for d, w in zip(similar_days, weights)
    )

    # Percentiles from sorted pool
    sorted_changes = sorted(changes)
    n = len(sorted_changes)
    worst_case = sorted_changes[max(0,   int(n * 0.10))]   # 10th percentile
    median_ret = sorted_changes[n // 2]
    best_case  = sorted_changes[min(n-1, int(n * 0.90))]   # 90th percentile

    # Top-5 detail for Nodes 13/14 explanations
    top5 = sorted(similar_days, key=lambda x: x["similarity"], reverse=True)[:5]
    detail: List[Dict[str, Any]] = [
        {
            "date":             d["date"],
            "similarity_score": d["similarity"],
            "actual_change_7d": round(d["actual_change_7d"], 2),
            "direction": (
                "UP"   if d["actual_change_7d"] > 0
                else "DOWN" if d["actual_change_7d"] < 0
                else "FLAT"
            ),
        }
        for d in top5
    ]

    return {
        "sufficient_data":           True,
        "similar_days_found":        len(similar_days),
        "similarity_threshold_used": round(threshold_used, 2),
        "prob_up":                   round(prob_up,         4),
        "prob_down":                 round(1.0 - prob_up,   4),
        "expected_return_7d":        round(expected_return, 4),
        "worst_case_7d":             round(worst_case,      4),
        "best_case_7d":              round(best_case,       4),
        "median_return_7d":          round(median_ret,      4),
        "similar_days_detail":       detail,
    }


# ============================================================================
# HELPER 15: BUILD PREDICTION GRAPH DATA
# ============================================================================

def _build_prediction_graph_data(
    pattern_prediction: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build prediction graph parameters for the dashboard (Node 15).

    Blends Job 2's empirical return with Node 7's GBM expected return.
    The blend weight is earned proportionally by how many similar historical
    days Job 2 found — more evidence = higher empirical weight:

        job2_weight = min(0.70, similar_days_found / MAX_SIMILAR_DAYS × 0.70)
        gbm_weight  = 1.0 - job2_weight

    Examples:
        5  similar days → Job2 14%,  GBM 86%
        15 similar days → Job2 42%,  GBM 58%
        25 similar days → Job2 70%,  GBM 30%

    When Job 2 has insufficient data, falls back to GBM-only.

    Args:
        pattern_prediction: Output of _build_pattern_prediction().
        state:              LangGraph state dict (for Node 7 data).

    Returns:
        prediction_graph_data dict for Node 15.
    """
    mc: Dict[str, Any] = state.get("monte_carlo_results") or {}
    gbm_expected: float = float(mc.get("expected_return") or 0.0)

    ci = mc.get("confidence_95") or {}
    current_price = float(mc.get("current_price") or 0.0)
    if current_price > 0:
        lower_pct = ((float(ci.get("lower", current_price)) - current_price) / current_price) * 100
        upper_pct = ((float(ci.get("upper", current_price)) - current_price) / current_price) * 100
    else:
        lower_pct = -5.0
        upper_pct =  5.0

    if not pattern_prediction.get("sufficient_data"):
        return {
            "blended_expected_return":   round(gbm_expected, 4),
            "gbm_expected_return":       round(gbm_expected, 4),
            "empirical_expected_return": None,
            "job2_blend_weight":         0.0,
            "gbm_blend_weight":          1.0,
            "similar_days_used":         0,
            "gbm_spread_lower":          round(lower_pct, 4),
            "gbm_spread_upper":          round(upper_pct, 4),
            "empirical_lower":           None,
            "empirical_upper":           None,
            "data_source":               "gbm_only",
        }

    similar_count = int(pattern_prediction.get("similar_days_found", 0))
    job2_weight   = min(0.70, (similar_count / MAX_SIMILAR_DAYS) * 0.70)
    gbm_weight    = 1.0 - job2_weight

    emp_return: float = float(pattern_prediction.get("expected_return_7d", 0.0))
    blended_return = job2_weight * emp_return + gbm_weight * gbm_expected

    logger.debug(
        f"  Graph blend: {similar_count} days → "
        f"Job2={job2_weight:.0%} emp={emp_return:+.2f}% + "
        f"GBM={gbm_weight:.0%} gbm={gbm_expected:+.2f}% "
        f"= {blended_return:+.2f}%"
    )

    return {
        "blended_expected_return":   round(blended_return,  4),
        "gbm_expected_return":       round(gbm_expected,    4),
        "empirical_expected_return": round(emp_return,      4),
        "job2_blend_weight":         round(job2_weight,     4),
        "gbm_blend_weight":          round(gbm_weight,      4),
        "similar_days_used":         similar_count,
        "gbm_spread_lower":          round(lower_pct,       4),
        "gbm_spread_upper":          round(upper_pct,       4),
        "empirical_lower":           pattern_prediction.get("worst_case_7d"),
        "empirical_upper":           pattern_prediction.get("best_case_7d"),
        "data_source":               "blended",
    }


# ============================================================================
# HELPER 16: JOB 1 / JOB 2 AGREEMENT CHECK
# ============================================================================

def _check_job_agreement(
    final_signal: str,
    pattern_prediction: Dict[str, Any],
) -> str:
    """
    Compare Job 1's signal with Job 2's empirical probability.

    Returns an agreement_level string used as a transparency flag in Nodes 13/14.
    Does NOT return a confidence multiplier — final_confidence is Job 1's
    authoritative output and must not be modified by correlated data.

    Agreement thresholds:
      BUY  + prob_up >= 0.65 → strong | >= 0.55 → mild | < 0.40 → disagree
      SELL + prob_up <= 0.35 → strong | <= 0.45 → mild | > 0.60 → disagree
      HOLD + prob_up in [0.40, 0.60] → mild | outside → disagree

    Args:
        final_signal:       Job 1 signal ('BUY', 'SELL', 'HOLD').
        pattern_prediction: Output of _build_pattern_prediction().

    Returns:
        agreement_level: 'strong' | 'mild' | 'disagree' | 'neutral'
    """
    if not pattern_prediction.get("sufficient_data"):
        return "neutral"

    prob_up: float = float(pattern_prediction.get("prob_up", 0.5))

    if final_signal == "BUY":
        if prob_up >= 0.65:   return "strong"
        if prob_up >= 0.55:   return "mild"
        if prob_up < 0.40:    return "disagree"
        return "neutral"

    if final_signal == "SELL":
        if prob_up <= 0.35:   return "strong"
        if prob_up <= 0.45:   return "mild"
        if prob_up > 0.60:    return "disagree"
        return "neutral"

    # HOLD
    if 0.40 <= prob_up <= 0.60:          return "mild"
    if prob_up > 0.65 or prob_up < 0.35: return "disagree"
    return "neutral"


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def signal_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 12: Final Signal Generation

    Orchestrates Job 1 (weighted signal combination) and Job 2 (signal-
    conditioned historical pattern matching), adjusts final_confidence
    based on inter-method agreement, and packages everything into
    signal_components for Nodes 13, 14, and 15.

    Output fields written to state:
        final_signal        — 'BUY' | 'SELL' | 'HOLD'
        final_confidence    — float 0.0–1.0 (adjusted by Job 1/2 agreement)
        signal_components   — full breakdown dict for Nodes 13/14/15

    Runs AFTER: Node 11 (adaptive_weights)
    Runs BEFORE: Node 13 (LLM beginner explanation)

    Args:
        state: LangGraph state dict.

    Returns:
        Updated state.
    """
    start_time = datetime.now()
    ticker: str = state.get("ticker", "UNKNOWN")

    logger.info(f"Node 12: Generating final signal for {ticker}")

    try:
        # ==================================================================
        # STEP 1: Resolve adaptive weights
        # ==================================================================
        adaptive_weights: Dict[str, Any] = (
            state.get("adaptive_weights") or _EQUAL_WEIGHTS_FALLBACK
        )

        # Read all 4 backtested stream weights from Node 11.
        # Monte Carlo is NOT backtested — it has no earned weight slot.
        # Apply MIN_STREAM_WEIGHT floor before normalising — except for
        # anti-predictive streams that earned their 0.0 through backtesting.
        _weight_sources = adaptive_weights.get("weight_sources") or {}

        def _floored(raw: float, stream_key: str) -> float:
            if _weight_sources.get(stream_key) == "neutral_anti_predictive":
                return raw  # earned its 0.0 — do not restore
            return max(raw, MIN_STREAM_WEIGHT)

        w_tech_raw    = _floored(float(adaptive_weights.get("technical_weight",    EQUAL_WEIGHT)), "technical")
        w_sent_raw    = _floored(float(adaptive_weights.get("stock_news_weight",   EQUAL_WEIGHT)), "stock_news")
        w_market_raw  = _floored(float(adaptive_weights.get("market_news_weight",  EQUAL_WEIGHT)), "market_news")
        w_related_raw = _floored(float(adaptive_weights.get("related_news_weight", EQUAL_WEIGHT)), "related_news")

        _w_total = w_tech_raw + w_sent_raw + w_market_raw + w_related_raw
        if _w_total > 0:
            w_tech    = w_tech_raw    / _w_total
            w_sent    = w_sent_raw    / _w_total
            w_market  = w_market_raw  / _w_total
            w_related = w_related_raw / _w_total
        else:
            w_tech = w_sent = w_market = w_related = EQUAL_WEIGHT

        # Learnt threshold from Node 11's threshold optimisation; falls back
        # to the module-level BUY_THRESHOLD (0.15) when not present.
        signal_threshold: float = float(
            adaptive_weights.get("signal_threshold", BUY_THRESHOLD)
        )

        logger.info(
            f"  Weights (4-stream, MC excluded from signal, threshold=±{signal_threshold}) — "
            f"tech={w_tech:.3f}, sent={w_sent:.3f}, "
            f"market={w_market:.3f}, related={w_related:.3f}"
        )

        # ==================================================================
        # JOB 1 — STEP 2: Score each stream on [-1, +1]
        # ==================================================================
        streams_missing: List[str] = []

        tech_score,     tech_missing    = _score_technical(state)
        sent_score,     sent_missing    = _score_sentiment(state)
        mkt_score,      mkt_missing     = _score_market(state)
        related_score,  related_missing = _score_related_sentiment(state)

        if tech_missing:    streams_missing.append("technical")
        if sent_missing:    streams_missing.append("sentiment")
        if mkt_missing:     streams_missing.append("market")
        if related_missing: streams_missing.append("related_news")

        # Monte Carlo (Node 7) is excluded from Job 1 signal scoring.
        # GBM probability_up near-random over 7 days adds noise not signal.
        # Node 7 outputs remain in Job 2 prediction graph and price_targets.

        # ==================================================================
        # JOB 1 — STEP 3: All 4 streams missing → cannot produce a signal
        # ==================================================================
        if len(streams_missing) == 4:
            logger.error("Node 12: All 4 upstream streams unavailable — defaulting to HOLD")
            state.setdefault("errors", []).append(
                "Node 12: All upstream data unavailable"
            )
            _empty_pattern: Dict[str, Any] = {
                "sufficient_data": False,
                "reason": "No stream data available for pattern matching",
            }
            state["final_signal"]     = "HOLD"
            state["final_confidence"] = 0.0
            state["signal_strength"]  = 0
            state["trustworthiness"]  = 0.0
            state["signal_components"] = {
                "final_score":      0.0,
                "final_signal":     "HOLD",
                "final_confidence": 0.0,
                "stream_scores": {
                    "technical":    {"raw_score": 0.0, "weight": w_tech,    "contribution": 0.0},
                    "sentiment":    {"raw_score": 0.0, "weight": w_sent,    "contribution": 0.0},
                    "market":       {"raw_score": 0.0, "weight": w_market,  "contribution": 0.0},
                    "related_news": {"raw_score": 0.0, "weight": w_related, "contribution": 0.0},
                },
                "risk_summary":          _build_risk_summary(state),
                "price_targets":         _build_price_targets(state),
                "backtest_context": {
                    "hold_threshold_pct":   adaptive_weights.get("hold_threshold_pct"),
                    "streams_reliable":     int(adaptive_weights.get("streams_reliable", 0)),
                    "weights_are_fallback": bool(adaptive_weights.get("fallback_equal_weights", False)),
                },
                "signal_agreement":      0,
                "streams_missing":       streams_missing,
                "pattern_prediction":    _empty_pattern,
                "prediction_graph_data": {"data_source": "unavailable"},
                "signal_strength":       0,
                "trustworthiness":       0.0,
                "trustworthiness_breakdown": {
                    "insufficient_history": True,
                    "using_prior":          True,
                },
            }
            state.setdefault("node_execution_times", {})["node_12"] = (
                datetime.now() - start_time
            ).total_seconds()
            return state

        # ==================================================================
        # JOB 1 — STEP 4: Redistribute missing-stream weights, then score
        # ==================================================================
        # A stream marked missing has no data this run (0 articles / no node
        # output). Its Node-11 weight is dead weight at score=0.  Re-normalise
        # across the streams that DO have data so signal capacity is preserved.
        # Anti-predictive streams that earned 0.0 weight are NOT redistributed
        # — they are correctly zero regardless of whether data is present.
        _raw_weights: Dict[str, float] = {
            "technical":    w_tech,
            "sentiment":    w_sent,
            "market":       w_market,
            "related_news": w_related,
        }
        _scores: Dict[str, float] = {
            "technical":    tech_score,
            "sentiment":    sent_score,
            "market":       mkt_score,
            "related_news": related_score,
        }
        _missing_set = set(streams_missing)  # streams with no data this run

        # Sum weights of streams that have data (or have an earned 0-weight).
        # Missing streams contribute their weight to be redistributed.
        _present_weight_sum = sum(
            w for key, w in _raw_weights.items() if key not in _missing_set
        )

        if 0.0 < _present_weight_sum < 1.0 - 1e-6:
            # At least one stream is missing and at least one is present.
            _scale = 1.0 / _present_weight_sum
            _eff_weights: Dict[str, float] = {
                key: (w * _scale if key not in _missing_set else 0.0)
                for key, w in _raw_weights.items()
            }
            logger.info(
                f"  Weight redistribution (missing={sorted(_missing_set)}): "
                + ", ".join(
                    f"{k}:{_raw_weights[k]:.3f}→{_eff_weights[k]:.3f}"
                    for k in _raw_weights
                    if abs(_eff_weights[k] - _raw_weights[k]) > 0.001
                )
            )
        else:
            _eff_weights = _raw_weights  # nothing to redistribute

        ew_tech    = _eff_weights["technical"]
        ew_sent    = _eff_weights["sentiment"]
        ew_market  = _eff_weights["market"]
        ew_related = _eff_weights["related_news"]

        tech_contrib    = tech_score    * ew_tech
        sent_contrib    = sent_score    * ew_sent
        mkt_contrib     = mkt_score     * ew_market
        related_contrib = related_score * ew_related

        final_score: float = tech_contrib + sent_contrib + mkt_contrib + related_contrib
        final_signal: str     = _classify_signal(final_score, threshold=signal_threshold)
        raw_confidence: float = min(1.0, abs(final_score))

        logger.info(
            f"  Scores — tech={tech_score:+.3f}, sent={sent_score:+.3f}, "
            f"market={mkt_score:+.3f}, related={related_score:+.3f}"
        )
        logger.info(
            f"  Contributions — tech={tech_contrib:+.4f}, sent={sent_contrib:+.4f}, "
            f"market={mkt_contrib:+.4f}, related={related_contrib:+.4f}"
        )
        logger.info(
            f"  Job 1 → {final_signal} "
            f"(score={final_score:+.4f}, confidence={raw_confidence:.4f})"
        )

        # ==================================================================
        # JOB 1 — STEP 5: Stream scores breakdown (use effective weights)
        # ==================================================================
        stream_scores: Dict[str, Dict[str, float]] = {
            "technical":    {"raw_score": tech_score,    "weight": ew_tech,    "contribution": tech_contrib},
            "sentiment":    {"raw_score": sent_score,    "weight": ew_sent,    "contribution": sent_contrib},
            "market":       {"raw_score": mkt_score,     "weight": ew_market,  "contribution": mkt_contrib},
            "related_news": {"raw_score": related_score, "weight": ew_related, "contribution": related_contrib},
        }

        signal_agreement: int = _count_signal_agreement(stream_scores, final_signal)
        logger.info(
            f"  Agreement: {signal_agreement}/4 streams, "
            f"missing: {streams_missing or 'none'}"
        )

        dominant_sign = np.sign(final_score) if final_score != 0 else 0.0
        agreeing = sum(1 for s in [tech_score, sent_score, mkt_score, related_score]
                       if np.sign(s) == dominant_sign and s != 0)
        stream_agreement_score = round(agreeing / 4.0, 4)
        stream_confidence = round(stream_agreement_score * abs(final_score), 4)
        active_streams = sum(1 for w in [w_tech, w_sent, w_market, w_related] if w > 0.05)

        # ==================================================================
        # JOB 2 — Build historical snapshots from Node 10 daily_results
        # ==================================================================
        pattern_prediction: Dict[str, Any]
        threshold_used: float = 0.0

        backtest_results: Optional[Dict[str, Any]] = state.get("backtest_results")

        if backtest_results is not None:
            try:
                snapshots = _build_historical_snapshots(backtest_results)

                if len(snapshots) >= MIN_SNAPSHOTS_FOR_JOB2:
                    today_vector = _get_today_vector(state, tech_score)
                    similar_days, threshold_used = _find_similar_days(
                        today_vector, snapshots,
                        w_tech, w_sent, w_market, w_related,
                    )
                    pattern_prediction = _build_pattern_prediction(
                        similar_days, threshold_used
                    )
                else:
                    logger.info(
                        f"  Job 2: only {len(snapshots)} snapshots "
                        f"(need {MIN_SNAPSHOTS_FOR_JOB2}) → skipped"
                    )
                    pattern_prediction = {
                        "sufficient_data":    False,
                        "reason":             f"Only {len(snapshots)} snapshots available",
                        "similar_days_found": 0,
                    }

            except Exception as job2_err:
                logger.warning(f"  Job 2 failed (non-fatal): {job2_err}")
                pattern_prediction = {
                    "sufficient_data": False,
                    "reason":          f"Job 2 error: {job2_err}",
                }
        else:
            logger.info("  Job 2: no backtest_results → skipped")
            pattern_prediction = {
                "sufficient_data": False,
                "reason":          "backtest_results not in state",
            }

        if pattern_prediction.get("sufficient_data"):
            logger.info(
                f"  Job 2 → {pattern_prediction['similar_days_found']} similar days, "
                f"prob_up={pattern_prediction['prob_up']:.3f}, "
                f"expected_7d={pattern_prediction['expected_return_7d']:+.2f}%"
            )

        # ==================================================================
        # JOB 2 — Agreement check (transparency flag only)
        # ==================================================================
        agreement_level = _check_job_agreement(final_signal, pattern_prediction)
        pattern_prediction["agreement_with_job1"] = agreement_level

        JOB2_MULTIPLIER = {
            "strong":   1.00,
            "mild":     1.00,
            "neutral":  0.95,
            "disagree": 0.80,
        }

        if agreement_level == "disagree":
            logger.warning(
                f"  Job1/Job2 disagree: {final_signal} vs "
                f"prob_up={pattern_prediction.get('prob_up', 0.5):.2f} "
                f"— flagged for Nodes 13/14"
            )
        elif agreement_level in ("strong", "mild"):
            logger.info(f"  Job1/Job2 {agreement_level} agreement")

        # ==================================================================
        # SIGNAL STRENGTH AND TRUSTWORTHINESS
        # ==================================================================
        signal_strength: int = _compute_signal_strength(stream_scores, signal_agreement)

        trustworthiness: float
        trustworthiness_breakdown: Dict[str, Any]
        trustworthiness, trustworthiness_breakdown = _compute_trustworthiness(
            stream_scores=stream_scores,
            adaptive_weights=adaptive_weights,
            pattern_prediction=pattern_prediction,
            w_tech=w_tech,
            w_sent=w_sent,
            w_market=w_market,
            w_related=w_related,
        )

        # Apply Job 2 disagreement penalty to trustworthiness
        if pattern_prediction.get("sufficient_data"):
            j2_mult = JOB2_MULTIPLIER.get(agreement_level, 1.0)
            trustworthiness = round(float(np.clip(trustworthiness * j2_mult, 0.0, 1.0)), 4)
            trustworthiness_breakdown["job2_multiplier_applied"] = j2_mult

        logger.info(
            f"  Signal strength: {signal_strength}/100 | "
            f"Trustworthiness: {trustworthiness:.3f}"
            + (
                " [using prior — insufficient history]"
                if trustworthiness_breakdown["using_prior"]
                else ""
            )
        )

        # ==================================================================
        # JOB 2 — Prediction graph calibration for Node 15
        # ==================================================================
        prediction_graph_data = _build_prediction_graph_data(
            pattern_prediction, state
        )

        logger.debug(
            f"  Graph data: source={prediction_graph_data['data_source']}, "
            f"blended={prediction_graph_data['blended_expected_return']:+.2f}%"
        )

        # ==================================================================
        # STEP 6: Assemble signal_components and write to state
        # ==================================================================
        signal_components: Dict[str, Any] = {
            "final_score":           round(final_score,      6),
            "final_signal":          final_signal,
            "final_confidence":      round(trustworthiness,  6),
            "stream_agreement_score": stream_agreement_score,
            "stream_confidence":      stream_confidence,
            "active_streams":         active_streams,
            "stream_scores":         stream_scores,
            "risk_summary":          _build_risk_summary(state),
            "price_targets":         _build_price_targets(state),
            "backtest_context": {
                "hold_threshold_pct":   adaptive_weights.get("hold_threshold_pct"),
                "streams_reliable":     int(adaptive_weights.get("streams_reliable", 0)),
                "weights_are_fallback": bool(adaptive_weights.get("fallback_equal_weights", False)),
            },
            "signal_agreement":           signal_agreement,
            "streams_missing":            streams_missing,
            "pattern_prediction":         pattern_prediction,
            "prediction_graph_data":      prediction_graph_data,
            "signal_strength":            signal_strength,
            "trustworthiness":            trustworthiness,
            "trustworthiness_breakdown":  trustworthiness_breakdown,
        }

        state["final_signal"]      = final_signal
        state["final_confidence"]  = trustworthiness
        state["signal_strength"]   = signal_strength
        state["trustworthiness"]   = trustworthiness
        state["signal_components"] = signal_components

        execution_time = (datetime.now() - start_time).total_seconds()
        state.setdefault("node_execution_times", {})["node_12"] = execution_time
        logger.info(f"Node 12 completed in {execution_time:.3f}s → {final_signal}")

        return state

    except Exception as e:
        logger.error(f"Node 12 failed for {ticker}: {e}")
        logger.exception("Full traceback:")

        state.setdefault("errors", []).append(f"Node 12 (signal generation) failed: {e}")
        state["final_signal"]      = "HOLD"
        state["final_confidence"]  = 0.0
        state["signal_strength"]   = 0
        state["trustworthiness"]   = 0.0
        state["signal_components"] = None
        state.setdefault("node_execution_times", {})["node_12"] = (
            datetime.now() - start_time
        ).total_seconds()

        return state
