"""
Node 11: Adaptive Weights Calculation

Receives raw accuracy metrics from Node 10 (Backtesting) and computes
optimal signal weights for each of the 4 analysis streams.

Responsibilities deliberately NOT done by Node 10:
1. Apply 70/30 recency weighting:
       weighted_accuracy = RECENCY_WEIGHT * recent_accuracy + FULL_WEIGHT * full_accuracy
2. Guard against statistically insufficient or insignificant streams —
   assign neutral weight (0.5) when the signal stream cannot be trusted
3. Normalize: weight_i = weighted_accuracy_i / sum(all_weighted_accuracies)

Neutral weight (0.5) is used when:
- Stream data is None (Node 10 failed for that stream)
- is_sufficient is False  (fewer than MIN_SIGNALS_FOR_SUFFICIENCY directional signals)
- is_significant is False (p-value ≥ 0.05 in binomial test against 0.5 baseline)

Equal-weight fallback (0.25 each) is used when:
- backtest_results is entirely None (Node 10 itself failed)
- An unrecoverable error occurs inside this node

The 4 streams and their output weight keys:
- 'technical'  → 'technical_weight'
- 'stock_news'  → 'stock_news_weight'
- 'market_news' → 'market_news_weight'
- 'related_news' → 'related_news_weight'

Runs AFTER: Node 10 (backtesting)
Runs BEFORE: Node 12 (final signal generation)
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from src.utils.logger import get_node_logger

logger = get_node_logger("node_11")


# ============================================================================
# CONSTANTS
# ============================================================================

RECENCY_WEIGHT: float = 0.7       # proportion given to recent_accuracy (last 60 days)
FULL_WEIGHT: float = 0.3          # proportion given to full_accuracy (all 180 days)
NEUTRAL_ACCURACY: float = 0.5     # neutral (coin-flip) fallback for unreliable streams
EQUAL_WEIGHT: float = 0.25        # equal weight across 4 streams when backtest failed


# Stream keys and their corresponding output weight field names
STREAM_KEYS: list = [
    ("technical",   "technical_weight"),
    ("stock_news",  "stock_news_weight"),
    ("market_news", "market_news_weight"),
    ("related_news", "related_news_weight"),
]


# ============================================================================
# HELPER 1: SINGLE-STREAM WEIGHTED ACCURACY
# ============================================================================

def _compute_weighted_accuracy(
    stream_metrics: Optional[Dict[str, Any]],
    stream_name: str,
) -> Tuple[float, str]:
    """
    Compute the recency-weighted accuracy value for a single stream.

    This is where Node 11 applies the 70/30 recency weighting that Node 10
    deliberately left undone (to keep the two responsibilities cleanly separated).

    Returns neutral weight (NEUTRAL_ACCURACY = 0.5) when the stream cannot
    be trusted — i.e. when it is missing, has too few directional signals,
    or failed the binomial significance test.

    When recent_accuracy is None (Node 10 had fewer than 5 recent days), we
    fall back to full_accuracy only — no 70/30 split, but still valid data.

    Args:
        stream_metrics: Metrics dict produced by Node 10's calculate_stream_metrics(),
                        or None if that stream failed.
        stream_name: Human-readable label for logging.

    Returns:
        Tuple (weighted_accuracy, source_label) where source_label is one of:
            'calculated'            — full 70/30 recency weighting applied
            'full_accuracy_only'    — recent_accuracy was None; used full_accuracy only
            'neutral_no_data'       — stream_metrics is None
            'neutral_insufficient'  — is_sufficient is False (< 20 directional signals)
            'neutral_insignificant' — is_significant is False (p-value ≥ 0.05)
    """
    # No data at all for this stream
    if stream_metrics is None:
        logger.warning(f"  {stream_name}: No data → neutral weight ({NEUTRAL_ACCURACY:.0%})")
        return NEUTRAL_ACCURACY, "neutral_no_data"

    # Not enough directional signals — accuracy estimate is unreliable
    if not stream_metrics.get("is_sufficient", False):
        n = stream_metrics.get("signal_count", 0)
        logger.warning(
            f"  {stream_name}: Insufficient signals ({n}) → neutral weight ({NEUTRAL_ACCURACY:.0%})"
        )
        return NEUTRAL_ACCURACY, "neutral_insufficient"

    # Statistically insignificant — stream performs no better than random
    if not stream_metrics.get("is_significant", False):
        p = stream_metrics.get("p_value", 1.0)
        logger.warning(
            f"  {stream_name}: Not significant (p={p:.3f}) → neutral weight ({NEUTRAL_ACCURACY:.0%})"
        )
        return NEUTRAL_ACCURACY, "neutral_insignificant"

    full_accuracy: float = float(stream_metrics.get("full_accuracy", NEUTRAL_ACCURACY))
    recent_accuracy: Optional[float] = stream_metrics.get("recent_accuracy")

    # recent_accuracy can be None when fewer than MIN_RECENT_DAYS_FOR_ACCURACY
    # recent days were available — fall back to full_accuracy only
    if recent_accuracy is None:
        logger.info(
            f"  {stream_name}: No recent data → full_accuracy only ({full_accuracy:.1%})"
        )
        return full_accuracy, "full_accuracy_only"

    # Standard 70/30 recency weighting
    weighted = RECENCY_WEIGHT * float(recent_accuracy) + FULL_WEIGHT * full_accuracy
    logger.info(
        f"  {stream_name}: full={full_accuracy:.1%}, recent={float(recent_accuracy):.1%}, "
        f"weighted={weighted:.1%} (70/30)"
    )
    return weighted, "calculated"


# ============================================================================
# HELPER 2: FULL WEIGHT CALCULATION
# ============================================================================

def calculate_adaptive_weights(
    backtest_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate normalized adaptive weights from Node 10's backtest results.

    Process:
    1. For each of the 4 streams call _compute_weighted_accuracy to get the
       recency-weighted accuracy (or neutral 0.5 when the stream cannot be trusted).
    2. Normalize: weight_i = weighted_accuracy_i / sum(all_weighted_accuracies).
    3. Validate: weights must sum to 1.0 ± 0.001; if not, fall back to equal weights.

    Why even unreliable streams contribute to the sum:
    All 4 streams get a weighted_accuracy (at minimum NEUTRAL_ACCURACY = 0.5),
    so the denominator is always > 0. Neutral streams will receive equal
    relative weight (since they all share the same value), which is the
    fairest default when we have no evidence that one is better than another.

    Args:
        backtest_results: Dict produced by Node 10's backtesting_node().
            Expected keys: 'technical', 'stock_news', 'market_news',
                           'related_news', 'hold_threshold_pct', 'sample_period_days'

    Returns:
        Dict containing:
            technical_weight, stock_news_weight, market_news_weight, related_news_weight
            weighted_accuracies  — per-stream weighted accuracy before normalization
            weight_sources       — how each weight was derived (label from _compute_weighted_accuracy)
            streams_reliable     — count of streams with source == 'calculated' or 'full_accuracy_only'
            hold_threshold_pct   — passed through from Node 10 for dashboard
            sample_period_days   — passed through from Node 10
    """
    weighted_accuracies: Dict[str, float] = {}
    weight_sources: Dict[str, str] = {}

    logger.info("Computing per-stream weighted accuracies:")
    for stream_key, _ in STREAM_KEYS:
        stream_metrics = backtest_results.get(stream_key)
        w_acc, source = _compute_weighted_accuracy(stream_metrics, stream_key)
        weighted_accuracies[stream_key] = w_acc
        weight_sources[stream_key] = source

    total = sum(weighted_accuracies.values())

    # Normalize
    weights: Dict[str, float] = {}
    for stream_key, weight_key in STREAM_KEYS:
        weights[weight_key] = round(weighted_accuracies[stream_key] / total, 6)

    # Sanity check — should always hold; log and fall back if it doesn't
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.001:
        logger.error(
            f"Weights do not sum to 1.0 (got {weight_sum:.6f}) — falling back to equal weights"
        )
        weights = {wk: EQUAL_WEIGHT for _, wk in STREAM_KEYS}

    streams_reliable = sum(
        1 for src in weight_sources.values() if src in ("calculated", "full_accuracy_only")
    )

    return {
        **weights,
        "weighted_accuracies": {k: round(v, 4) for k, v in weighted_accuracies.items()},
        "weight_sources": weight_sources,
        "streams_reliable": streams_reliable,
        "hold_threshold_pct": backtest_results.get("hold_threshold_pct"),
        "sample_period_days": backtest_results.get("sample_period_days", 180),
    }


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def adaptive_weights_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 11: Adaptive Weights Calculation

    Reads Node 10's backtest_results, applies 70/30 recency weighting,
    guards against statistically unreliable streams, normalizes, and writes
    adaptive_weights to the state for Node 12 to consume.

    Output: state['adaptive_weights'] = {
        'technical_weight':   float,   # e.g. 0.231
        'stock_news_weight':  float,   # e.g. 0.273  ← typically highest (Node 8 boost)
        'market_news_weight': float,   # e.g. 0.244
        'related_news_weight': float,  # e.g. 0.252
        'weighted_accuracies': {       # pre-normalization values for dashboard
            'technical':    float,
            'stock_news':   float,
            'market_news':  float,
            'related_news': float,
        },
        'weight_sources': {            # how each weight was determined
            'technical':    str,       # 'calculated' | 'full_accuracy_only' |
            'stock_news':   str,       # 'neutral_no_data' | 'neutral_insufficient' |
            'market_news':  str,       # 'neutral_insignificant'
            'related_news': str,
        },
        'streams_reliable':  int,      # number of non-neutral streams (0–4)
        'hold_threshold_pct': float,   # passed through from Node 10
        'sample_period_days': int,     # passed through from Node 10
        'fallback_equal_weights': bool # True if error forced equal weights
    }

    Runs AFTER: Node 10 (backtesting)
    Runs BEFORE: Node 12 (final signal generation)

    Args:
        state: LangGraph state dict containing:
            - ticker: stock ticker symbol
            - backtest_results: dict from Node 10 (or None)

    Returns:
        Updated state with 'adaptive_weights' populated
    """
    start_time = datetime.now()
    ticker = state.get("ticker", "UNKNOWN")

    logger.info(f"Node 11: Calculating adaptive weights for {ticker}")

    try:
        backtest_results = state.get("backtest_results")

        # Node 10 failed entirely — fall back to equal weights
        if backtest_results is None:
            logger.warning(
                f"Node 11: No backtest_results for {ticker} — using equal weights (0.25 each)"
            )
            state["adaptive_weights"] = {
                "technical_weight":    EQUAL_WEIGHT,
                "stock_news_weight":   EQUAL_WEIGHT,
                "market_news_weight":  EQUAL_WEIGHT,
                "related_news_weight": EQUAL_WEIGHT,
                "weighted_accuracies": {},
                "weight_sources": {},
                "streams_reliable": 0,
                "hold_threshold_pct": None,
                "sample_period_days": 180,
                "fallback_equal_weights": True,
            }
            state.setdefault("node_execution_times", {})["node_11"] = (
                datetime.now() - start_time
            ).total_seconds()
            return state

        # Main calculation
        adaptive_weights = calculate_adaptive_weights(backtest_results)
        adaptive_weights["fallback_equal_weights"] = False

        state["adaptive_weights"] = adaptive_weights

        # Execution tracking
        execution_time = (datetime.now() - start_time).total_seconds()
        state.setdefault("node_execution_times", {})["node_11"] = execution_time

        logger.info(f"Node 11 completed in {execution_time:.3f}s")
        logger.info(
            f"  Weights: technical={adaptive_weights['technical_weight']:.1%}, "
            f"stock_news={adaptive_weights['stock_news_weight']:.1%}, "
            f"market_news={adaptive_weights['market_news_weight']:.1%}, "
            f"related_news={adaptive_weights['related_news_weight']:.1%}"
        )
        logger.info(f"  Reliable streams: {adaptive_weights['streams_reliable']}/4")

        return state

    except Exception as e:
        logger.error(f"Node 11 failed: {e}")
        logger.exception("Full traceback:")

        state.setdefault("errors", []).append(f"Node 11 (adaptive weights) failed: {e}")

        # Fallback: equal weights so Node 12 can still run
        state["adaptive_weights"] = {
            "technical_weight":    EQUAL_WEIGHT,
            "stock_news_weight":   EQUAL_WEIGHT,
            "market_news_weight":  EQUAL_WEIGHT,
            "related_news_weight": EQUAL_WEIGHT,
            "weighted_accuracies": {},
            "weight_sources": {},
            "streams_reliable": 0,
            "hold_threshold_pct": None,
            "sample_period_days": 180,
            "fallback_equal_weights": True,
        }
        state.setdefault("node_execution_times", {})["node_11"] = (
            datetime.now() - start_time
        ).total_seconds()

        return state
