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
  similar historical days are blended 60/40 with Node 7's GBM Monte Carlo
  estimate to produce calibrated prediction graph parameters.

  Job 1 and Job 2 results are compared: when they agree, final_confidence
  is boosted; when they disagree, it is penalised and flagged for Nodes 13/14.

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

# Job 2 — blending empirical with Node 7 GBM
JOB2_BLEND_WEIGHT: float = 0.60
GBM_BLEND_WEIGHT: float = 0.40

# Job 1/2 agreement — confidence adjustment multipliers
STRONG_AGREEMENT_BOOST: float = 1.15
MILD_AGREEMENT_BOOST: float = 1.05
DISAGREEMENT_PENALTY: float = 0.80


# ============================================================================
# HELPER 1: TECHNICAL STREAM SCORING
# ============================================================================

def _score_technical(state: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Convert Node 4's technical signal to a continuous score on [-1, +1].

    Uses the signal direction (BUY=+1, SELL=-1, HOLD=0) scaled by the
    reported confidence, so a high-confidence SELL scores more negative
    than a low-confidence one.

    Args:
        state: LangGraph state dict.

    Returns:
        (score, is_missing) — is_missing=True when either field is absent.
    """
    signal: Optional[str] = state.get("technical_signal")
    confidence: Optional[float] = state.get("technical_confidence")

    if signal is None or confidence is None:
        logger.warning("  Technical stream: missing signal or confidence → score=0.0")
        return 0.0, True

    sign = SIGNAL_TO_SCORE.get(signal.upper(), 0.0)
    score = sign * float(confidence)
    logger.debug(f"  Technical: {signal} × {confidence:.3f} = {score:+.3f}")
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
    Convert Node 6's market context to a continuous score on [-1, +1].

    Score = context_direction × market_correlation.
    A perfectly correlated market with a BUY signal gives +1.0; a completely
    uncorrelated market (correlation=0) contributes nothing.

    Args:
        state: LangGraph state dict.

    Returns:
        (score, is_missing) — is_missing=True when market_context is absent.
    """
    market_context: Optional[Dict[str, Any]] = state.get("market_context")
    if market_context is None:
        logger.warning("  Market stream: no market_context → score=0.0")
        return 0.0, True

    context_signal: str = market_context.get("context_signal", "HOLD")
    market_corr: Optional[float] = market_context.get("market_correlation")
    if market_corr is None:
        market_corr = 1.0  # treat as fully correlated when unknown

    context_score = SIGNAL_TO_SCORE.get(context_signal.upper(), 0.0)
    score = context_score * float(market_corr)
    logger.debug(
        f"  Market: {context_signal} × correlation={market_corr:.3f} = {score:+.3f}"
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
# HELPER 5: SIGNAL CLASSIFICATION
# ============================================================================

def _classify_signal(final_score: float) -> str:
    """
    Convert a weighted final score to a BUY/SELL/HOLD label.

    Thresholds: > +0.15 → BUY | < -0.15 → SELL | else → HOLD

    Args:
        final_score: Weighted sum of stream scores.

    Returns:
        'BUY', 'SELL', or 'HOLD'.
    """
    if final_score > BUY_THRESHOLD:
        return "BUY"
    if final_score < SELL_THRESHOLD:
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
    Build prediction graph calibration parameters for the dashboard (Node 15).

    Blends Job 2's empirical return estimate (60%) with Node 7's GBM expected
    return (40%) to produce a signal-conditioned forecast center. The GBM
    confidence interval spread is preserved as the band shape; no new
    simulation is run.

    When Job 2 has insufficient data, falls back to GBM-only parameters.

    Args:
        pattern_prediction: Output of _build_pattern_prediction().
        state:              LangGraph state dict (for Node 7 data).

    Returns:
        prediction_graph_data dict for Node 15.
    """
    mc: Dict[str, Any] = state.get("monte_carlo_results") or {}
    gbm_expected: float = float(mc.get("expected_return") or 0.0)

    # GBM spread as percentage relative to current price
    ci = mc.get("confidence_95") or {}
    current_price = float(mc.get("current_price") or 0.0)
    if current_price > 0:
        lower_pct = ((float(ci.get("lower", current_price)) - current_price) / current_price) * 100
        upper_pct = ((float(ci.get("upper", current_price)) - current_price) / current_price) * 100
    else:
        lower_pct = -5.0
        upper_pct = 5.0

    if pattern_prediction.get("sufficient_data"):
        emp_return: float = float(pattern_prediction.get("expected_return_7d", 0.0))
        blended_return = JOB2_BLEND_WEIGHT * emp_return + GBM_BLEND_WEIGHT * gbm_expected

        return {
            "blended_expected_return":   round(blended_return, 4),
            "gbm_expected_return":       round(gbm_expected,   4),
            "empirical_expected_return": round(emp_return,     4),
            "gbm_spread_lower":          round(lower_pct,      4),
            "gbm_spread_upper":          round(upper_pct,      4),
            "empirical_lower":           pattern_prediction.get("worst_case_7d"),
            "empirical_upper":           pattern_prediction.get("best_case_7d"),
            "data_source":               "blended",
        }

    return {
        "blended_expected_return":   round(gbm_expected, 4),
        "gbm_expected_return":       round(gbm_expected, 4),
        "empirical_expected_return": None,
        "gbm_spread_lower":          round(lower_pct, 4),
        "gbm_spread_upper":          round(upper_pct, 4),
        "empirical_lower":           None,
        "empirical_upper":           None,
        "data_source":               "gbm_only",
    }


# ============================================================================
# HELPER 16: JOB 1 / JOB 2 AGREEMENT CHECK
# ============================================================================

def _check_job_agreement(
    final_signal: str,
    pattern_prediction: Dict[str, Any],
) -> Tuple[str, float]:
    """
    Compare Job 1's directional signal with Job 2's empirical probability.

    When both methods agree, confidence is boosted. When they disagree, it
    is penalised without overriding the Job 1 recommendation itself.
    The penalty and flag are passed through signal_components for Nodes 13/14.

    Agreement thresholds:
      BUY  + prob_up >= 0.65 → strong | >= 0.55 → mild | < 0.40 → disagree
      SELL + prob_up <= 0.35 → strong | <= 0.45 → mild | > 0.60 → disagree
      HOLD + prob_up in [0.40, 0.60] → mild | outside → disagree

    Args:
        final_signal:       Job 1 signal ('BUY', 'SELL', 'HOLD').
        pattern_prediction: Output of _build_pattern_prediction().

    Returns:
        (agreement_level, confidence_multiplier)
        agreement_level: 'strong' | 'mild' | 'disagree' | 'neutral'
    """
    if not pattern_prediction.get("sufficient_data"):
        return "neutral", 1.0

    prob_up: float = float(pattern_prediction.get("prob_up", 0.5))

    if final_signal == "BUY":
        if prob_up >= 0.65:
            return "strong",   STRONG_AGREEMENT_BOOST
        if prob_up >= 0.55:
            return "mild",     MILD_AGREEMENT_BOOST
        if prob_up < 0.40:
            return "disagree", DISAGREEMENT_PENALTY
        return "neutral", 1.0

    if final_signal == "SELL":
        if prob_up <= 0.35:
            return "strong",   STRONG_AGREEMENT_BOOST
        if prob_up <= 0.45:
            return "mild",     MILD_AGREEMENT_BOOST
        if prob_up > 0.60:
            return "disagree", DISAGREEMENT_PENALTY
        return "neutral", 1.0

    # HOLD
    if 0.40 <= prob_up <= 0.60:
        return "mild", MILD_AGREEMENT_BOOST
    if prob_up > 0.65 or prob_up < 0.35:
        return "disagree", DISAGREEMENT_PENALTY
    return "neutral", 1.0


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
        w_tech:   float = float(adaptive_weights.get("technical_weight",    EQUAL_WEIGHT))
        w_sent:   float = float(adaptive_weights.get("stock_news_weight",   EQUAL_WEIGHT))
        w_market: float = float(adaptive_weights.get("market_news_weight",  EQUAL_WEIGHT))
        w_mc:     float = float(adaptive_weights.get("related_news_weight", EQUAL_WEIGHT))

        logger.info(
            f"  Weights — tech={w_tech:.3f}, sent={w_sent:.3f}, "
            f"market={w_market:.3f}, mc={w_mc:.3f}"
        )

        # ==================================================================
        # JOB 1 — STEP 2: Score each stream on [-1, +1]
        # ==================================================================
        streams_missing: List[str] = []

        tech_score,  tech_missing  = _score_technical(state)
        sent_score,  sent_missing  = _score_sentiment(state)
        mkt_score,   mkt_missing   = _score_market(state)
        mc_score,    mc_missing    = _score_monte_carlo(state)

        if tech_missing:  streams_missing.append("technical")
        if sent_missing:  streams_missing.append("sentiment")
        if mkt_missing:   streams_missing.append("market")
        if mc_missing:    streams_missing.append("monte_carlo")

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
            state["signal_components"] = {
                "final_score":      0.0,
                "final_signal":     "HOLD",
                "final_confidence": 0.0,
                "stream_scores": {
                    "technical":   {"raw_score": 0.0, "weight": w_tech,   "contribution": 0.0},
                    "sentiment":   {"raw_score": 0.0, "weight": w_sent,   "contribution": 0.0},
                    "market":      {"raw_score": 0.0, "weight": w_market, "contribution": 0.0},
                    "monte_carlo": {"raw_score": 0.0, "weight": w_mc,     "contribution": 0.0},
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
            }
            state.setdefault("node_execution_times", {})["node_12"] = (
                datetime.now() - start_time
            ).total_seconds()
            return state

        # ==================================================================
        # JOB 1 — STEP 4: Weighted contributions and final score
        # ==================================================================
        tech_contrib = tech_score * w_tech
        sent_contrib = sent_score * w_sent
        mkt_contrib  = mkt_score  * w_market
        mc_contrib   = mc_score   * w_mc

        final_score: float    = tech_contrib + sent_contrib + mkt_contrib + mc_contrib
        final_signal: str     = _classify_signal(final_score)
        raw_confidence: float = min(1.0, abs(final_score))

        logger.info(
            f"  Scores — tech={tech_score:+.3f}, sent={sent_score:+.3f}, "
            f"market={mkt_score:+.3f}, mc={mc_score:+.3f}"
        )
        logger.info(
            f"  Contributions — tech={tech_contrib:+.4f}, sent={sent_contrib:+.4f}, "
            f"market={mkt_contrib:+.4f}, mc={mc_contrib:+.4f}"
        )
        logger.info(
            f"  Job 1 → {final_signal} "
            f"(score={final_score:+.4f}, confidence={raw_confidence:.4f})"
        )

        # ==================================================================
        # JOB 1 — STEP 5: Stream scores breakdown
        # ==================================================================
        stream_scores: Dict[str, Dict[str, float]] = {
            "technical":   {"raw_score": tech_score, "weight": w_tech,   "contribution": tech_contrib},
            "sentiment":   {"raw_score": sent_score, "weight": w_sent,   "contribution": sent_contrib},
            "market":      {"raw_score": mkt_score,  "weight": w_market, "contribution": mkt_contrib},
            "monte_carlo": {"raw_score": mc_score,   "weight": w_mc,     "contribution": mc_contrib},
        }

        signal_agreement: int = _count_signal_agreement(stream_scores, final_signal)
        logger.info(
            f"  Agreement: {signal_agreement}/4 streams, "
            f"missing: {streams_missing or 'none'}"
        )

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
                        w_tech, w_sent, w_market, w_mc,
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
        # JOB 2 — Agreement check → adjusted confidence
        # ==================================================================
        agreement_level, confidence_multiplier = _check_job_agreement(
            final_signal, pattern_prediction
        )
        adjusted_confidence: float = min(
            1.0, max(0.0, raw_confidence * confidence_multiplier)
        )

        if agreement_level != "neutral":
            logger.info(
                f"  Job1/Job2 {agreement_level} (×{confidence_multiplier:.2f}) "
                f"{raw_confidence:.4f} → {adjusted_confidence:.4f}"
            )
        if agreement_level == "disagree":
            logger.warning(
                f"  Disagreement: {final_signal} vs Job2 prob_up="
                f"{pattern_prediction.get('prob_up', 0.5):.2f}"
            )

        pattern_prediction["agreement_with_job1"]   = agreement_level
        pattern_prediction["confidence_multiplier"] = round(confidence_multiplier, 4)

        # ==================================================================
        # JOB 2 — Prediction graph calibration for Node 15
        # ==================================================================
        prediction_graph_data = _build_prediction_graph_data(pattern_prediction, state)

        logger.debug(
            f"  Graph data: source={prediction_graph_data['data_source']}, "
            f"blended={prediction_graph_data['blended_expected_return']:+.2f}%"
        )

        # ==================================================================
        # STEP 6: Assemble signal_components and write to state
        # ==================================================================
        signal_components: Dict[str, Any] = {
            "final_score":           round(final_score,          6),
            "final_signal":          final_signal,
            "final_confidence":      round(adjusted_confidence,  6),
            "stream_scores":         stream_scores,
            "risk_summary":          _build_risk_summary(state),
            "price_targets":         _build_price_targets(state),
            "backtest_context": {
                "hold_threshold_pct":   adaptive_weights.get("hold_threshold_pct"),
                "streams_reliable":     int(adaptive_weights.get("streams_reliable", 0)),
                "weights_are_fallback": bool(adaptive_weights.get("fallback_equal_weights", False)),
            },
            "signal_agreement":      signal_agreement,
            "streams_missing":       streams_missing,
            "pattern_prediction":    pattern_prediction,
            "prediction_graph_data": prediction_graph_data,
        }

        state["final_signal"]      = final_signal
        state["final_confidence"]  = round(adjusted_confidence, 6)
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
        state["signal_components"] = None
        state.setdefault("node_execution_times", {})["node_12"] = (
            datetime.now() - start_time
        ).total_seconds()

        return state
