"""
Node 12: Final Signal Generation

Reads signals from Nodes 4, 5, 6, 7 and the adaptive weights from Node 11,
converts each stream to a continuous score on [-1, +1], applies weighted
combination, and emits a final BUY/SELL/HOLD recommendation.

Continuous scoring (vs. naive binary 0/100 in the build guide):
- Technical:    confidence * sign(signal)    → e.g. BUY 0.72  → +0.72
- Sentiment:    aggregated_sentiment directly → e.g. +0.61     → +0.61
- Market:       context_score * market_correlation
                → e.g. BUY context, corr=0.82 → +0.82
- Monte Carlo:  (probability_up - 0.5) * 2  → e.g. p=0.68   → +0.36

Thresholds:
  final_score > +0.15 → BUY
  final_score < -0.15 → SELL
  else                → HOLD
  final_confidence = min(1.0, abs(final_score))

Error handling:
- Any missing stream scores 0.0; its name is added to streams_missing.
- adaptive_weights None → equal weights 0.25 each.
- All 4 streams missing → HOLD, confidence 0.0, error appended.

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

BUY_THRESHOLD: float = 0.15      # final_score above this → BUY
SELL_THRESHOLD: float = -0.15    # final_score below this → SELL
EQUAL_WEIGHT: float = 0.25       # equal weight for each stream (fallback)

# Maps signal strings to their directional sign on [-1, +1]
SIGNAL_TO_SCORE: Dict[str, float] = {
    "BUY":  1.0,
    "SELL": -1.0,
    "HOLD": 0.0,
}

# Maps Node 5 sentiment labels to trading signals (Node 5 does NOT emit BUY/SELL/HOLD)
SENTIMENT_TO_SIGNAL: Dict[str, str] = {
    "POSITIVE": "BUY",
    "NEGATIVE": "SELL",
    "NEUTRAL":  "HOLD",
}

# Default equal-weights dict used when adaptive_weights is None
_EQUAL_WEIGHTS_FALLBACK: Dict[str, float] = {
    "technical_weight":    EQUAL_WEIGHT,
    "stock_news_weight":   EQUAL_WEIGHT,
    "market_news_weight":  EQUAL_WEIGHT,
    "related_news_weight": EQUAL_WEIGHT,
}


# ============================================================================
# HELPER 1: STREAM SCORING
# ============================================================================

def _score_technical(state: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Convert Node 4's technical signal to a continuous score on [-1, +1].

    Uses the signal direction (BUY=+1, SELL=-1, HOLD=0) scaled by the
    reported confidence, so a high-confidence SELL scores more negative
    than a low-confidence one.

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


def _score_sentiment(state: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Return the continuous sentiment score on [-1, +1].

    Node 5 writes aggregated_sentiment as a signed float already on [-1, +1].
    Node 8 may have adjusted it and written the result into sentiment_analysis.
    We prefer Node 8's adjusted value when available.

    Returns:
        (score, is_missing) — is_missing=True when no valid score can be found.
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


def _score_market(state: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Convert Node 6's market context to a continuous score on [-1, +1].

    Score = context_direction × market_correlation
    A perfectly correlated market with a BUY signal gives +1.0; a completely
    uncorrelated market (correlation=0) contributes nothing regardless of signal.

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
        market_corr = 1.0   # treat as fully correlated when unknown

    context_score = SIGNAL_TO_SCORE.get(context_signal.upper(), 0.0)
    score = context_score * float(market_corr)
    logger.debug(
        f"  Market: {context_signal} × correlation={market_corr:.3f} = {score:+.3f}"
    )
    return score, False


def _score_monte_carlo(state: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Convert Node 7's Monte Carlo probability_up to a continuous score on [-1, +1].

    Mapping: score = (probability_up - 0.5) * 2
      probability_up=1.0 → +1.0 (certain upside)
      probability_up=0.5 → 0.0  (coin-flip)
      probability_up=0.0 → -1.0 (certain downside)

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
# HELPER 2: SIGNAL CLASSIFICATION
# ============================================================================

def _classify_signal(final_score: float) -> str:
    """
    Convert a weighted final score to a BUY/SELL/HOLD label.

    Thresholds:  > +0.15 → BUY  |  < -0.15 → SELL  |  else → HOLD
    """
    if final_score > BUY_THRESHOLD:
        return "BUY"
    if final_score < SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


# ============================================================================
# HELPER 3: SIGNAL AGREEMENT
# ============================================================================

def _count_signal_agreement(
    stream_scores: Dict[str, Dict[str, float]],
    final_signal: str,
) -> int:
    """
    Count how many of the 4 streams agree with the final signal direction.

    Agreement definition:
      BUY  → stream raw_score > 0
      SELL → stream raw_score < 0
      HOLD → abs(stream raw_score) <= BUY_THRESHOLD
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
# HELPER 4: RISK SUMMARY
# ============================================================================

def _build_risk_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate risk information from Nodes 9A and 9B into a single dict.

    trading_safe is True when neither behavioral risk is HIGH/CRITICAL
    nor pump_and_dump_score exceeds 70.  Node 12 always runs regardless of
    this flag — blocking logic lives in the workflow's conditional edges.
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
        "overall_risk_level": risk_level,
        "pump_and_dump_score": pump_score,
        "behavioral_risk": risk_level,
        "content_risk": content_risk,
        "trading_safe": trading_safe,
    }


# ============================================================================
# HELPER 5: PRICE TARGETS
# ============================================================================

def _build_price_targets(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect price forecast information from Nodes 7's Monte Carlo output.

    price_range (a tuple written by Node 7) takes priority for lower/upper.
    Falls back to confidence_95 dict inside monte_carlo_results.
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
        "current_price":      mc.get("current_price"),
        "forecasted_price":   state.get("forecasted_price"),
        "price_range_lower":  lower,
        "price_range_upper":  upper,
        "expected_return_pct": mc.get("expected_return"),
    }


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def signal_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 12: Final Signal Generation

    Reads continuous scores from 4 analytical streams, applies adaptive weights
    from Node 11, and produces a single weighted recommendation.

    Output fields written to state:
        final_signal      — 'BUY' | 'SELL' | 'HOLD'
        final_confidence  — float 0.0–1.0  (= min(1.0, abs(final_score)))
        signal_components — full breakdown dict for Nodes 13/14

    Runs AFTER: Node 11 (adaptive_weights)
    Runs BEFORE: Node 13 (LLM beginner explanation)

    Args:
        state: LangGraph state dict. Required upstream fields:
            technical_signal, technical_confidence       (Node 4)
            aggregated_sentiment, sentiment_analysis     (Nodes 5/8)
            market_context                               (Node 6)
            monte_carlo_results                          (Node 7)
            adaptive_weights                             (Node 11)

    Returns:
        Updated state with final_signal, final_confidence, signal_components.
    """
    start_time = datetime.now()
    ticker = state.get("ticker", "UNKNOWN")

    logger.info(f"Node 12: Generating final signal for {ticker}")

    try:
        # ------------------------------------------------------------------
        # 1. Resolve adaptive weights (fall back to equal if Node 11 failed)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 2. Score each stream on [-1, +1]
        # ------------------------------------------------------------------
        streams_missing: List[str] = []

        tech_score,   tech_missing   = _score_technical(state)
        sent_score,   sent_missing   = _score_sentiment(state)
        mkt_score,    mkt_missing    = _score_market(state)
        mc_score,     mc_missing     = _score_monte_carlo(state)

        if tech_missing:   streams_missing.append("technical")
        if sent_missing:   streams_missing.append("sentiment")
        if mkt_missing:    streams_missing.append("market")
        if mc_missing:     streams_missing.append("monte_carlo")

        # ------------------------------------------------------------------
        # 3. All 4 streams missing — cannot generate a meaningful signal
        # ------------------------------------------------------------------
        if len(streams_missing) == 4:
            logger.error("Node 12: All 4 upstream streams unavailable — defaulting to HOLD")
            state.setdefault("errors", []).append(
                "Node 12: All upstream data unavailable"
            )
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
                "risk_summary":    _build_risk_summary(state),
                "price_targets":   _build_price_targets(state),
                "backtest_context": {
                    "hold_threshold_pct": adaptive_weights.get("hold_threshold_pct"),
                    "streams_reliable":   int(adaptive_weights.get("streams_reliable", 0)),
                    "weights_are_fallback": bool(adaptive_weights.get("fallback_equal_weights", False)),
                },
                "signal_agreement": 0,
                "streams_missing":  streams_missing,
            }
            state.setdefault("node_execution_times", {})["node_12"] = (
                datetime.now() - start_time
            ).total_seconds()
            return state

        # ------------------------------------------------------------------
        # 4. Compute weighted contributions and final score
        # ------------------------------------------------------------------
        tech_contrib   = tech_score   * w_tech
        sent_contrib   = sent_score   * w_sent
        mkt_contrib    = mkt_score    * w_market
        mc_contrib     = mc_score     * w_mc

        final_score: float = tech_contrib + sent_contrib + mkt_contrib + mc_contrib
        final_signal: str  = _classify_signal(final_score)
        final_confidence: float = min(1.0, abs(final_score))

        logger.info(
            f"  Scores — tech={tech_score:+.3f}, sent={sent_score:+.3f}, "
            f"market={mkt_score:+.3f}, mc={mc_score:+.3f}"
        )
        logger.info(
            f"  Contributions — tech={tech_contrib:+.4f}, sent={sent_contrib:+.4f}, "
            f"market={mkt_contrib:+.4f}, mc={mc_contrib:+.4f}"
        )
        logger.info(
            f"  Final: {final_signal} (score={final_score:+.4f}, confidence={final_confidence:.4f})"
        )

        # ------------------------------------------------------------------
        # 5. Assemble stream_scores breakdown
        # ------------------------------------------------------------------
        stream_scores: Dict[str, Dict[str, float]] = {
            "technical":   {"raw_score": tech_score,   "weight": w_tech,   "contribution": tech_contrib},
            "sentiment":   {"raw_score": sent_score,   "weight": w_sent,   "contribution": sent_contrib},
            "market":      {"raw_score": mkt_score,    "weight": w_market, "contribution": mkt_contrib},
            "monte_carlo": {"raw_score": mc_score,     "weight": w_mc,     "contribution": mc_contrib},
        }

        signal_agreement: int = _count_signal_agreement(stream_scores, final_signal)
        logger.info(
            f"  Agreement: {signal_agreement}/4 streams, missing: {streams_missing or 'none'}"
        )

        # ------------------------------------------------------------------
        # 6. Build full signal_components dict
        # ------------------------------------------------------------------
        signal_components: Dict[str, Any] = {
            "final_score":      round(final_score,      6),
            "final_signal":     final_signal,
            "final_confidence": round(final_confidence, 6),
            "stream_scores":    stream_scores,
            "risk_summary":     _build_risk_summary(state),
            "price_targets":    _build_price_targets(state),
            "backtest_context": {
                "hold_threshold_pct": adaptive_weights.get("hold_threshold_pct"),
                "streams_reliable":   int(adaptive_weights.get("streams_reliable", 0)),
                "weights_are_fallback": bool(adaptive_weights.get("fallback_equal_weights", False)),
            },
            "signal_agreement": signal_agreement,
            "streams_missing":  streams_missing,
        }

        # ------------------------------------------------------------------
        # 7. Write to state
        # ------------------------------------------------------------------
        state["final_signal"]     = final_signal
        state["final_confidence"] = round(final_confidence, 6)
        state["signal_components"] = signal_components

        execution_time = (datetime.now() - start_time).total_seconds()
        state.setdefault("node_execution_times", {})["node_12"] = execution_time
        logger.info(f"Node 12 completed in {execution_time:.3f}s")

        return state

    except Exception as e:
        logger.error(f"Node 12 failed: {e}")
        logger.exception("Full traceback:")

        state.setdefault("errors", []).append(f"Node 12 (signal generation) failed: {e}")
        state["final_signal"]      = "HOLD"
        state["final_confidence"]  = 0.0
        state["signal_components"] = None
        state.setdefault("node_execution_times", {})["node_12"] = (
            datetime.now() - start_time
        ).total_seconds()

        return state
