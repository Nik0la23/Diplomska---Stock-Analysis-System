"""
Node 11: Adaptive Weights Calculation

Receives raw accuracy metrics from Node 10 (Backtesting) and computes
optimal signal weights for each of the 4 analysis streams.

Responsibilities deliberately NOT done by Node 10:
1. Apply 70/30 recency weighting:
       weighted_accuracy = RECENCY_WEIGHT * recent_accuracy + FULL_WEIGHT * full_accuracy
2. Apply Bayesian smoothing to accuracy estimates using sample size
   (total_days_evaluated / recent_days_evaluated from Node 10):
       smoothed_accuracy = (raw_accuracy * n + 0.5 * PRIOR_STRENGTH) / (n + PRIOR_STRENGTH)
   A stream with 21 evaluated days (barely sufficient) gets pulled toward 0.5;
   a stream with 180 days keeps its measured accuracy.
3. Guard against statistically insufficient or insignificant streams —
   assign neutral weight (0.0) when the signal stream cannot be trusted,
   so it drops out of normalization entirely.
4. Normalize: weight_i = weighted_accuracy_i / sum(all_weighted_accuracies)
5. Apply Node 8 per-stream learning adjustments (capped to [0.30, 0.90]):
   - Uses news_type_effectiveness per stream type (stock / market / related)
     and historical_correlation for a per-type multiplier
   - Falls back to the global learning_adjustment when Node 8 type data is thin
   - Technical stream is always excluded from adjustment.

Neutral weight (0.0) is used when:
- Stream data is None (Node 10 failed for that stream)
- is_sufficient is False    (fewer than MIN_SIGNALS_FOR_SUFFICIENCY directional signals)
- is_anti_predictive is True (p_value_less < 0.05: stream is reliably WORSE than random)

Near-random streams (not significant AND not anti-predictive) still participate
in weight normalization at their measured Bayesian-smoothed accuracy. This allows
the system to weight streams relative to each other even when none passes the
strict p < 0.05 significance threshold — common for volatile or regime-changing
stocks over a 180-day lookback.

Equal-weight fallback (0.25 each) is used when:
- backtest_results is entirely None (Node 10 itself failed)
- All 4 streams return 0.0 (total == 0, cannot normalize)
- An unrecoverable error occurs inside this node

The 4 streams and their output weight keys:
- 'technical'   → 'technical_weight'
- 'stock_news'  → 'stock_news_weight'
- 'market_news' → 'market_news_weight'
- 'related_news' → 'related_news_weight'

Runs AFTER: Node 10 (backtesting)
Runs BEFORE: Node 12 (final signal generation)
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from src.langgraph_nodes.node_10_backtesting import RECENT_PERIOD_DAYS
from src.utils.logger import get_node_logger

logger = get_node_logger("node_11")


# ============================================================================
# CONSTANTS
# ============================================================================

RECENCY_WEIGHT: float = 0.7       # proportion given to recent_accuracy (last 60 days)
FULL_WEIGHT: float = 0.3          # proportion given to full_accuracy (all 180 days)
NEUTRAL_ACCURACY: float = 0.0     # anti-predictive streams contribute nothing to normalization
EQUAL_WEIGHT: float = 0.25        # equal weight across 4 streams when backtest failed

# Bayesian smoothing: pull accuracy toward 0.5 when sample is small.
# Equivalent to adding PRIOR_STRENGTH virtual observations at 50% accuracy.
# Low n (e.g. 5 recent days)  → significant pull toward 0.5
# High n (e.g. 180 days)      → accuracy stays close to measured value
PRIOR_STRENGTH: int = 10

# Node 8 learning_adjustment integration (per-stream capped boost)
NEWS_STREAM_KEYS: tuple = ("stock_news", "market_news", "related_news")  # technical excluded
MIN_ADJUSTED_ACCURACY: float = 0.30   # floor: news still contributes even if sources unreliable
MAX_ADJUSTED_ACCURACY: float = 0.90   # ceiling: no news stream dominates unrealistically
DEFAULT_LEARNING_ADJUSTMENT: float = 1.0  # neutral when Node 8 data unavailable

# Per-stream adjustment derived from Node 8's news_type_effectiveness
STREAM_TO_NEWS_TYPE: dict = {
    "stock_news":   "stock",
    "market_news":  "market",
    "related_news": "related",
}
MIN_NEWS_TYPE_SAMPLE: int = 10    # minimum Node 8 events required for per-type adjustment
MIN_ADJ_MULTIPLIER: float = 0.5   # lower bound for any per-stream multiplier
MAX_ADJ_MULTIPLIER: float = 2.0   # upper bound for any per-stream multiplier


# Stream keys and their corresponding output weight field names
STREAM_KEYS: list = [
    ("technical",    "technical_weight"),
    ("stock_news",   "stock_news_weight"),
    ("market_news",  "market_news_weight"),
    ("related_news", "related_news_weight"),
]

# Signal threshold optimisation — learn the ±t that maximises directional
# accuracy on historical data instead of using the hardcoded ±0.15 default.
THRESHOLD_CANDIDATES: list = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
THRESHOLD_HOLDOUT_DAYS: int = 30       # last 30 days held out for validation
MIN_DAYS_FOR_THRESHOLD_OPT: int = 20   # skip if train set is too thin
THRESHOLD_MIN_TEST_ACCURACY: float = 0.45  # fallback if test set underperforms
DEFAULT_SIGNAL_THRESHOLD: float = 0.15     # safe fallback


# ============================================================================
# HELPER 1: BAYESIAN ACCURACY SMOOTHING
# ============================================================================

def _bayesian_smooth(raw_accuracy: float, n: int) -> float:
    """
    Apply Bayesian smoothing to a raw accuracy estimate.

    Pulls accuracy toward 0.5 (the random baseline) proportionally to how
    thin the sample is. A stream with 5 evaluated days should not be trusted
    as much as one with 180 days, even if both pass is_significant.

    Formula:
        smoothed = (raw_accuracy * n + 0.5 * PRIOR_STRENGTH) / (n + PRIOR_STRENGTH)

    Examples (PRIOR_STRENGTH = 10):
        n=5,   acc=0.80 → smoothed = (4.00 + 5.0) / 15  = 0.600  (strongly pulled)
        n=21,  acc=0.65 → smoothed = (13.65 + 5.0) / 31 = 0.601  (noticeably pulled)
        n=60,  acc=0.65 → smoothed = (39.0 + 5.0) / 70  = 0.629  (mild pull)
        n=180, acc=0.65 → smoothed = (117.0 + 5.0) / 190 = 0.642 (near unchanged)

    Args:
        raw_accuracy: Measured accuracy in [0, 1].
        n: Number of observations (days evaluated) the accuracy is based on.

    Returns:
        Smoothed accuracy in [0, 1], always closer to 0.5 than raw_accuracy
        when n < infinity.
    """
    if n <= 0:
        return 0.5
    return (raw_accuracy * n + 0.5 * PRIOR_STRENGTH) / (n + PRIOR_STRENGTH)


# ============================================================================
# HELPER 2: SINGLE-STREAM WEIGHTED ACCURACY
# ============================================================================

def _compute_weighted_accuracy(
    stream_metrics: Optional[Dict[str, Any]],
    stream_name: str,
) -> Tuple[float, str]:
    """
    Compute the recency-weighted, Bayesian-smoothed accuracy for a single stream.

    Steps:
    1. Reliability guard: return 0.0 for missing, insufficient, or insignificant streams.
    2. Bayesian smoothing: pull full_accuracy and recent_accuracy toward 0.5
       using total_days_evaluated and recent_days_evaluated from Node 10.
    3. Recency weighting: 70% recent + 30% full (or full-only when recent is None).

        Returns NEUTRAL_ACCURACY (0.0) when the stream cannot be trusted — it will
    contribute nothing to the normalization and receive zero weight.

    Near-random streams (p_value >= 0.05 but p_value_less >= 0.05) are NOT zeroed
    out. They still participate in normalization at their measured accuracy, which
    naturally gives them proportional weight relative to better-performing streams.
    Bayesian smoothing pulls their accuracy toward 0.5, so they have little
    influence when competing against a clearly better stream.

    Args:
        stream_metrics: Metrics dict produced by Node 10's calculate_stream_metrics(),
                        or None if that stream failed.
        stream_name: Human-readable label for logging.

    Returns:
        Tuple (weighted_accuracy, source_label) where source_label is one of:
            'calculated'              — full 70/30 recency weighting applied
            'full_accuracy_only'      — recent_accuracy was None; used full_accuracy only
            'neutral_no_data'         — stream_metrics is None
            'neutral_insufficient'    — is_sufficient is False (< 20 directional signals)
            'neutral_anti_predictive' — is_anti_predictive is True (p_value_less < 0.05)
    """
    if stream_metrics is None:
        logger.warning(f"  {stream_name}: No data → neutral weight ({NEUTRAL_ACCURACY:.0%})")
        return NEUTRAL_ACCURACY, "neutral_no_data"

    if not stream_metrics.get("is_sufficient", False):
        n = stream_metrics.get("signal_count", 0)
        logger.warning(
            f"  {stream_name}: Insufficient signals ({n}) → neutral weight ({NEUTRAL_ACCURACY:.0%})"
        )
        return NEUTRAL_ACCURACY, "neutral_insufficient"

    if stream_metrics.get("is_anti_predictive", False):
        p_less = stream_metrics.get("p_value_less", 1.0)
        logger.warning(
            f"  {stream_name}: Anti-predictive (p_less={p_less:.3f}) "
            f"→ neutral weight ({NEUTRAL_ACCURACY:.0%})"
        )
        return NEUTRAL_ACCURACY, "neutral_anti_predictive"

    # Log whether the stream is statistically significant or just near-random
    is_sig = stream_metrics.get("is_significant", False)
    p_val  = stream_metrics.get("p_value", 1.0)
    if not is_sig:
        logger.info(
            f"  {stream_name}: Near-random (p={p_val:.3f}) but not anti-predictive "
            f"— contributing at measured accuracy"
        )

    # Prefer directional_accuracy — it is comparable across technical
    # (which has HOLD days) and sentiment streams (which never produce HOLD).
    # Fall back to full_accuracy for cached results that predate this field.
    _dir_acc = stream_metrics.get("directional_accuracy")
    raw_full: float = float(_dir_acc) if _dir_acc is not None else float(
        stream_metrics.get("full_accuracy", NEUTRAL_ACCURACY)
    )
    # Use signal_count (BUY+SELL only) as the Bayesian smoothing denominator
    # when directional_accuracy is available — consistent sample size.
    total_days: int = int(
        stream_metrics.get("signal_count", 0) if _dir_acc is not None
        else stream_metrics.get("total_days_evaluated", 0)
    )
    smoothed_full = _bayesian_smooth(raw_full, total_days)

    _dir_recent = stream_metrics.get("recent_directional_accuracy")
    raw_recent: Optional[float] = (
        float(_dir_recent) if _dir_recent is not None
        else stream_metrics.get("recent_accuracy")
    )
    # Recent sample size: count recent BUY+SELL days from daily_results
    if _dir_recent is not None:
        recent_days: int = sum(
            1 for r in stream_metrics.get("daily_results", [])
            if r.get("days_ago", 999) <= RECENT_PERIOD_DAYS
            and r.get("signal") in ("BUY", "SELL")
        )
    else:
        recent_days: int = int(stream_metrics.get("recent_days_evaluated", 0))

    accuracy_source = "directional" if _dir_acc is not None else "full"

    if raw_recent is None:
        logger.info(
            f"  {stream_name} ({accuracy_source}): No recent data → full_accuracy only "
            f"raw={raw_full:.1%} → smoothed={smoothed_full:.1%} (n={total_days})"
        )
        return smoothed_full, "full_accuracy_only"

    smoothed_recent = _bayesian_smooth(float(raw_recent), recent_days)

    weighted = RECENCY_WEIGHT * smoothed_recent + FULL_WEIGHT * smoothed_full
    logger.info(
        f"  {stream_name} ({accuracy_source}): "
        f"full={raw_full:.1%}→{smoothed_full:.1%} (n={total_days}), "
        f"recent={raw_recent:.1%}→{smoothed_recent:.1%} (n={recent_days}), "
        f"weighted={weighted:.1%} (70/30)"
    )
    return weighted, "calculated"


# ============================================================================
# HELPER 3: PER-STREAM NODE 8 ADJUSTMENTS
# ============================================================================

def _compute_per_stream_adjustments(
    news_type_effectiveness: Dict[str, Any],
    historical_correlation: float,
    fallback_adjustment: float,
) -> Dict[str, float]:
    """
    Compute per-news-stream learning adjustment multipliers from Node 8's
    granular news_type_effectiveness data and historical_correlation.

    Node 8 measures accuracy per news type (stock / market / related) over 6 months,
    and the overall Pearson correlation between sentiment and price movements. This
    lets us differentiate adjustments per stream instead of applying one global
    multiplier to all three news streams.

    Formula (mirrors Node 8's global learning_adjustment formula but per-type):
        multiplier = (type_accuracy / 0.5) * (historical_correlation * 2.0)
        clamped to [MIN_ADJ_MULTIPLIER, MAX_ADJ_MULTIPLIER]

    Examples:
        accuracy=0.5, corr=0.5 → (1.0) * (1.0) = 1.0  (neutral, no change)
        accuracy=0.7, corr=0.7 → (1.4) * (1.4) = 1.96  (strong boost)
        accuracy=0.3, corr=0.3 → (0.6) * (0.6) = 0.36 → clamped to 0.5

    Falls back to global fallback_adjustment when Node 8 type data has fewer
    than MIN_NEWS_TYPE_SAMPLE events.

    Args:
        news_type_effectiveness: From Node 8's news_impact_verification.
            e.g. {'stock': {'accuracy_rate': 0.68, 'sample_size': 45}, ...}
        historical_correlation: Node 8's normalized Pearson correlation [0, 1].
        fallback_adjustment: Node 8's global learning_adjustment — used when
            per-type sample is insufficient.

    Returns:
        Dict mapping each NEWS_STREAM_KEY to a multiplier in
        [MIN_ADJ_MULTIPLIER, MAX_ADJ_MULTIPLIER].
    """
    adjustments: Dict[str, float] = {}

    for stream_key in NEWS_STREAM_KEYS:
        news_type = STREAM_TO_NEWS_TYPE[stream_key]
        type_eff = news_type_effectiveness.get(news_type, {})
        sample_size = int(type_eff.get("sample_size", 0))

        if sample_size >= MIN_NEWS_TYPE_SAMPLE:
            accuracy_rate = float(type_eff.get("accuracy_rate", 0.5))
            multiplier = (accuracy_rate / 0.5) * (historical_correlation * 2.0)
            clamped = max(MIN_ADJ_MULTIPLIER, min(MAX_ADJ_MULTIPLIER, multiplier))
            adjustments[stream_key] = clamped
            logger.info(
                f"  [{stream_key}] per-type adj: "
                f"accuracy={accuracy_rate:.2%}, corr={historical_correlation:.3f} "
                f"→ multiplier={clamped:.3f} (n={sample_size})"
            )
        else:
            adjustments[stream_key] = fallback_adjustment
            logger.info(
                f"  [{stream_key}] fallback adj={fallback_adjustment:.3f} "
                f"(Node 8 type data n={sample_size} < {MIN_NEWS_TYPE_SAMPLE})"
            )

    return adjustments


# ============================================================================
# HELPER 4: SIGNAL THRESHOLD OPTIMISATION
# ============================================================================

def _optimize_signal_threshold(
    backtest_results: Dict[str, Any],
    weights: Dict[str, Any],
) -> float:
    """
    Find the combined-score classification threshold that maximises directional
    accuracy, validated on a held-out recent window to prevent in-sample bias.

    Algorithm:
    1. Reconstruct weighted combined_score for every historical day using the
       just-computed adaptive weights.
    2. Split: train = days_ago > THRESHOLD_HOLDOUT_DAYS, test = rest.
    3. For each candidate threshold, measure directional accuracy on TRAIN.
    4. Pick the best threshold, then confirm it on TEST.
       If test accuracy < THRESHOLD_MIN_TEST_ACCURACY, fall back to default.
    5. Return DEFAULT_SIGNAL_THRESHOLD when data is too thin.

    Args:
        backtest_results: Node 10 output dict.
        weights:          Adaptive weights just computed by calculate_adaptive_weights().

    Returns:
        Best validated threshold from THRESHOLD_CANDIDATES, or DEFAULT_SIGNAL_THRESHOLD.
    """
    w_tech    = float(weights.get("technical_weight",    EQUAL_WEIGHT))
    w_sent    = float(weights.get("stock_news_weight",   EQUAL_WEIGHT))
    w_market  = float(weights.get("market_news_weight",  EQUAL_WEIGHT))
    w_related = float(weights.get("related_news_weight", EQUAL_WEIGHT))

    # --- helpers to extract scores from Node 10 daily_results records ---
    def _tech_score(r: Dict[str, Any]) -> float:
        sig  = r.get("signal", "HOLD") or "HOLD"
        conf = float(r.get("confidence") or 0.0)
        return conf if sig == "BUY" else (-conf if sig == "SELL" else 0.0)

    def _sent_score(r: Dict[str, Any]) -> float:
        return float(r.get("sentiment_score") or 0.0)

    # --- build per-date lookups ---
    tech_daily    = (backtest_results.get("technical")    or {}).get("daily_results") or []
    stock_daily   = (backtest_results.get("stock_news")   or {}).get("daily_results") or []
    market_daily  = (backtest_results.get("market_news")  or {}).get("daily_results") or []
    related_daily = (backtest_results.get("related_news") or {}).get("daily_results") or []

    tech_by_date    = {r["date"]: r for r in tech_daily    if "date" in r}
    stock_by_date   = {r["date"]: r for r in stock_daily   if "date" in r}
    market_by_date  = {r["date"]: r for r in market_daily  if "date" in r}
    related_by_date = {r["date"]: r for r in related_daily if "date" in r}

    all_dates = (
        set(tech_by_date)
        | set(stock_by_date)
        | set(market_by_date)
        | set(related_by_date)
    )

    # --- reconstruct combined score + actual change for every date ---
    combined_data: List[Dict[str, Any]] = []
    for date in all_dates:
        tech_r    = tech_by_date.get(date)
        stock_r   = stock_by_date.get(date)
        market_r  = market_by_date.get(date)
        related_r = related_by_date.get(date)

        anchor = tech_r or stock_r or market_r or related_r
        if anchor is None or anchor.get("actual_change_7d") is None:
            continue

        score = (
            (_tech_score(tech_r)    if tech_r    else 0.0) * w_tech
          + (_sent_score(stock_r)   if stock_r   else 0.0) * w_sent
          + (_sent_score(market_r)  if market_r  else 0.0) * w_market
          + (_sent_score(related_r) if related_r else 0.0) * w_related
        )
        combined_data.append({
            "score":    score,
            "change":   float(anchor["actual_change_7d"]),
            "days_ago": int(anchor.get("days_ago", 999)),
        })

    # --- split train / test ---
    train = [d for d in combined_data if d["days_ago"] > THRESHOLD_HOLDOUT_DAYS]
    test  = [d for d in combined_data if d["days_ago"] <= THRESHOLD_HOLDOUT_DAYS]

    if len(train) < MIN_DAYS_FOR_THRESHOLD_OPT:
        logger.info(
            f"  Threshold optimisation: only {len(train)} train days "
            f"(< {MIN_DAYS_FOR_THRESHOLD_OPT}) — using default {DEFAULT_SIGNAL_THRESHOLD}"
        )
        return DEFAULT_SIGNAL_THRESHOLD

    # --- find best threshold on train set ---
    best_threshold = DEFAULT_SIGNAL_THRESHOLD
    best_train_acc = 0.0
    best_n         = 0

    for t in THRESHOLD_CANDIDATES:
        directional = [d for d in train if abs(d["score"]) > t]
        if not directional:
            continue
        correct = sum(
            1 for d in directional
            if (d["score"] > 0 and d["change"] > 0)
            or (d["score"] < 0 and d["change"] < 0)
        )
        acc = correct / len(directional)
        # prefer higher accuracy; tie-break toward lower threshold (more signals)
        if acc > best_train_acc or (acc == best_train_acc and len(directional) > best_n):
            best_threshold = t
            best_train_acc = acc
            best_n         = len(directional)

    # --- validate on test set ---
    if test:
        test_dir = [d for d in test if abs(d["score"]) > best_threshold]
        if test_dir:
            test_correct = sum(
                1 for d in test_dir
                if (d["score"] > 0 and d["change"] > 0)
                or (d["score"] < 0 and d["change"] < 0)
            )
            test_acc = test_correct / len(test_dir)

            if test_acc < THRESHOLD_MIN_TEST_ACCURACY:
                logger.warning(
                    f"  Threshold optimisation: best_threshold={best_threshold} "
                    f"failed validation (test_acc={test_acc:.1%} < "
                    f"{THRESHOLD_MIN_TEST_ACCURACY:.0%}) — falling back to "
                    f"{DEFAULT_SIGNAL_THRESHOLD}"
                )
                return DEFAULT_SIGNAL_THRESHOLD

            logger.info(
                f"  Threshold optimisation: best_threshold=±{best_threshold} "
                f"train_acc={best_train_acc:.1%} (n={best_n}) "
                f"test_acc={test_acc:.1%} (n={len(test_dir)}) ✓"
            )
        else:
            logger.info(
                f"  Threshold optimisation: no test signals at threshold={best_threshold} "
                f"— accepting train result"
            )
    else:
        logger.info(
            f"  Threshold optimisation: no test data — accepting train result "
            f"threshold=±{best_threshold} train_acc={best_train_acc:.1%}"
        )

    return best_threshold


# ============================================================================
# HELPER 5: FULL WEIGHT CALCULATION
# ============================================================================

def calculate_adaptive_weights(
    backtest_results: Dict[str, Any],
    per_stream_adjustments: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Calculate normalized adaptive weights from Node 10's backtest results.

    Process:
    1. For each of the 4 streams call _compute_weighted_accuracy to get the
       Bayesian-smoothed, recency-weighted accuracy (or 0.0 for unreliable streams).
    2. Apply per-stream Node 8 adjustments (capped to [MIN_ADJUSTED_ACCURACY,
       MAX_ADJUSTED_ACCURACY]) to the 3 news streams only. Technical stream is
       excluded. Neutral streams (0.0) are left unchanged — adjusting them would
       erroneously re-introduce them into the normalization.
    3. Normalize: weight_i = weighted_accuracy_i / sum(all_weighted_accuracies).
    4. If total == 0 (all streams neutral), fall back to equal weights.
    5. Validate: weights must sum to 1.0 ± 0.001; if not, fall back to equal weights.

    Args:
        backtest_results: Dict produced by Node 10's backtesting_node().
        per_stream_adjustments: Per-stream multipliers from _compute_per_stream_adjustments().
            Only news streams are adjusted; technical is always excluded.

    Returns:
        Dict with weights and diagnostic fields for Node 12 and dashboard.
    """
    weighted_accuracies: Dict[str, float] = {}
    weight_sources: Dict[str, str] = {}

    logger.info("Computing per-stream weighted accuracies (with Bayesian smoothing):")
    for stream_key, _ in STREAM_KEYS:
        stream_metrics = backtest_results.get(stream_key)
        w_acc, source = _compute_weighted_accuracy(stream_metrics, stream_key)
        weighted_accuracies[stream_key] = w_acc
        weight_sources[stream_key] = source

    # Apply per-stream Node 8 adjustments to news streams only.
    # Skip neutral streams (0.0) — they are already excluded.
    applied_adjustments: Dict[str, float] = {}
    if per_stream_adjustments:
        logger.info("Applying per-stream Node 8 adjustments to news streams:")
        for stream_key in NEWS_STREAM_KEYS:
            adj = per_stream_adjustments.get(stream_key, DEFAULT_LEARNING_ADJUSTMENT)
            applied_adjustments[stream_key] = adj
            if weight_sources[stream_key] in ("calculated", "full_accuracy_only"):
                raw = weighted_accuracies[stream_key]
                if adj != DEFAULT_LEARNING_ADJUSTMENT:
                    adjusted = max(MIN_ADJUSTED_ACCURACY, min(MAX_ADJUSTED_ACCURACY, raw * adj))
                    weighted_accuracies[stream_key] = adjusted
                    logger.info(f"  {stream_key}: {raw:.3f} → {adjusted:.3f} (×{adj:.3f})")
                else:
                    logger.info(f"  {stream_key}: {raw:.3f} unchanged (neutral multiplier)")
            else:
                logger.debug(
                    f"  {stream_key}: skipped (neutral stream, source={weight_sources[stream_key]})"
                )

    total = sum(weighted_accuracies.values())

    # Guard: all streams neutral (0.0) → division by zero, fall back to equal weights
    if total <= 0.0:
        logger.warning(
            "All 4 streams returned 0.0 — cannot normalize, falling back to equal weights"
        )
        return {
            **{wk: EQUAL_WEIGHT for _, wk in STREAM_KEYS},
            "weighted_accuracies": {k: 0.0 for k, _ in STREAM_KEYS},
            "weight_sources": weight_sources,
            "streams_reliable": 0,
            "hold_threshold_pct": backtest_results.get("hold_threshold_pct"),
            "sample_period_days": backtest_results.get("sample_period_days", 180),
            "per_stream_adjustments": applied_adjustments,
            "fallback_equal_weights": True,
            "technical_hit_rate":        None,
            "stock_news_hit_rate":        None,
            "market_news_hit_rate":       None,
            "related_news_hit_rate":      None,
            "technical_sample_count":     0,
            "stock_news_sample_count":    0,
            "market_news_sample_count":   0,
            "related_news_sample_count":  0,
            "signal_threshold":           DEFAULT_SIGNAL_THRESHOLD,
        }

    # Normalize
    weights: Dict[str, float] = {}
    for stream_key, weight_key in STREAM_KEYS:
        weights[weight_key] = round(weighted_accuracies[stream_key] / total, 6)

    # Sanity check
    weight_sum = sum(weights.values())
    used_fallback = False
    if abs(weight_sum - 1.0) > 0.001:
        logger.error(
            f"Weights do not sum to 1.0 (got {weight_sum:.6f}) — falling back to equal weights"
        )
        weights = {wk: EQUAL_WEIGHT for _, wk in STREAM_KEYS}
        used_fallback = True

    streams_reliable = sum(
        1 for src in weight_sources.values()
        if src in ("calculated", "full_accuracy_only")
    )

    # Expose per-stream hit rates and sample counts for Node 12 trustworthiness.
    # Only streams that were actually used get a real value; neutralised streams
    # get None / 0 so Node 12's PRIOR = 0.55 fallback triggers cleanly.
    hit_rates: Dict[str, Any] = {}
    sample_counts: Dict[str, int] = {}
    for stream_key, _ in STREAM_KEYS:
        stream_metrics = backtest_results.get(stream_key) or {}
        weight_source  = weight_sources.get(stream_key, "neutral_no_data")
        if weight_source in ("calculated", "full_accuracy_only"):
            dir_acc  = stream_metrics.get("directional_accuracy")
            full_acc = stream_metrics.get("full_accuracy", 0.5)
            hit_rates[f"{stream_key}_hit_rate"] = float(
                dir_acc if dir_acc is not None else full_acc
            )
            sample_counts[f"{stream_key}_sample_count"] = int(
                stream_metrics.get("signal_count")
                or stream_metrics.get("total_days_evaluated")
                or 0
            )
        else:
            hit_rates[f"{stream_key}_hit_rate"]         = None
            sample_counts[f"{stream_key}_sample_count"] = 0

    return {
        **weights,
        **hit_rates,
        **sample_counts,
        "weighted_accuracies": {k: round(v, 6) for k, v in weighted_accuracies.items()},
        "weight_sources": weight_sources,
        "streams_reliable": streams_reliable,
        "hold_threshold_pct": backtest_results.get("hold_threshold_pct"),
        "sample_period_days": backtest_results.get("sample_period_days", 180),
        "per_stream_adjustments": {k: round(v, 4) for k, v in applied_adjustments.items()},
        "fallback_equal_weights": used_fallback,
    }


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def adaptive_weights_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 11: Adaptive Weights Calculation

    Reads Node 10's backtest_results, applies Bayesian-smoothed 70/30 recency
    weighting, guards against statistically unreliable streams (weight = 0.0,
    dropped from normalization), applies per-stream Node 8 adjustments using
    news_type_effectiveness and historical_correlation, normalizes, and writes
    adaptive_weights to state for Node 12.

    Node 10 data consumed:
    - full_accuracy, recent_accuracy          — directional accuracy per stream
    - total_days_evaluated, recent_days_evaluated — used for Bayesian smoothing
    - is_sufficient, is_significant, p_value  — reliability guard
    - hold_threshold_pct, sample_period_days  — passed through

    Node 8 data consumed:
    - learning_adjustment       — global fallback multiplier [0.5, 2.0]
    - news_type_effectiveness   — per-type accuracy for stock / market / related
    - historical_correlation    — normalized Pearson correlation [0, 1]

    Output: state['adaptive_weights'] = {
        'technical_weight':    float,
        'stock_news_weight':   float,
        'market_news_weight':  float,
        'related_news_weight': float,
        'weighted_accuracies': dict,        # post-smoothing, post-adjustment, pre-normalization
        'weight_sources':      dict,        # how each weight was derived
        'streams_reliable':    int,         # 0–4
        'hold_threshold_pct':  float,       # from Node 10
        'sample_period_days':  int,         # from Node 10
        'per_stream_adjustments': dict,     # per-stream Node 8 multipliers used
        'fallback_equal_weights': bool,
        'learning_adjustment_applied': float,
        'historical_correlation': float,
        'technical_hit_rate':       float | None,  # raw directional accuracy for Node 12 trustworthiness
        'stock_news_hit_rate':      float | None,  # None when stream was neutralised → Node 12 uses PRIOR
        'market_news_hit_rate':     float | None,
        'related_news_hit_rate':    float | None,
        'technical_sample_count':   int,           # number of evaluated signals; < 30 → Node 12 uses PRIOR
        'stock_news_sample_count':  int,
        'market_news_sample_count': int,
        'related_news_sample_count': int,
    }

    Runs AFTER: Node 10 (backtesting)
    Runs BEFORE: Node 12 (final signal generation)
    """
    start_time = datetime.now()
    ticker = state.get("ticker", "UNKNOWN")

    logger.info(f"Node 11: Calculating adaptive weights for {ticker}")

    try:
        backtest_results = state.get("backtest_results")

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
                "per_stream_adjustments": {},
                "fallback_equal_weights": True,
                "learning_adjustment_applied": DEFAULT_LEARNING_ADJUSTMENT,
                "historical_correlation": 0.5,
                "technical_hit_rate":        None,
                "stock_news_hit_rate":        None,
                "market_news_hit_rate":       None,
                "related_news_hit_rate":      None,
                "technical_sample_count":     0,
                "stock_news_sample_count":    0,
                "market_news_sample_count":   0,
                "related_news_sample_count":  0,
                "signal_threshold":           DEFAULT_SIGNAL_THRESHOLD,
            }
            state.setdefault("node_execution_times", {})["node_11"] = (
                datetime.now() - start_time
            ).total_seconds()
            return state

        # ====================================================================
        # Extract Node 8 data
        # ====================================================================

        niv = state.get("news_impact_verification") or {}

        learning_adjustment: float = float(
            niv.get("learning_adjustment", DEFAULT_LEARNING_ADJUSTMENT)
        )
        news_type_effectiveness: Dict[str, Any] = niv.get("news_type_effectiveness") or {}
        historical_correlation: float = float(niv.get("historical_correlation", 0.5))

        node8_source = "from Node 8" if niv else "default — Node 8 not available"
        logger.info(f"  Node 8 learning_adjustment: {learning_adjustment:.3f} ({node8_source})")
        logger.info(f"  Node 8 historical_correlation: {historical_correlation:.3f}")
        logger.info(
            f"  Node 8 news_type_effectiveness types: "
            f"{list(news_type_effectiveness.keys()) or 'none'}"
        )

        # ====================================================================
        # Compute per-stream adjustments
        # ====================================================================

        logger.info("Computing per-stream Node 8 adjustments:")
        per_stream_adjustments = _compute_per_stream_adjustments(
            news_type_effectiveness,
            historical_correlation,
            learning_adjustment,
        )

        # ====================================================================
        # Main calculation
        # ====================================================================

        adaptive_weights = calculate_adaptive_weights(backtest_results, per_stream_adjustments)
        adaptive_weights["learning_adjustment_applied"] = round(learning_adjustment, 4)
        adaptive_weights["historical_correlation"] = round(historical_correlation, 4)

        # Learn the optimal signal classification threshold from backtested data.
        # Uses train/test split to prevent in-sample bias.
        logger.info("Optimising signal classification threshold:")
        signal_threshold = _optimize_signal_threshold(backtest_results, adaptive_weights)
        adaptive_weights["signal_threshold"] = signal_threshold

        state["adaptive_weights"] = adaptive_weights

        execution_time = (datetime.now() - start_time).total_seconds()
        state.setdefault("node_execution_times", {})["node_11"] = execution_time

        logger.info(f"Node 11 completed in {execution_time:.3f}s")
        logger.info(
            f"  Weights: technical={adaptive_weights['technical_weight']:.1%}, "
            f"stock_news={adaptive_weights['stock_news_weight']:.1%}, "
            f"market_news={adaptive_weights['market_news_weight']:.1%}, "
            f"related_news={adaptive_weights['related_news_weight']:.1%}"
        )
        logger.info(
            f"  Signal threshold: ±{adaptive_weights['signal_threshold']} "
            f"(learnt from back-test, validated on last {THRESHOLD_HOLDOUT_DAYS} days)"
        )
        logger.info(f"  Reliable streams: {adaptive_weights['streams_reliable']}/4")
        if adaptive_weights.get("fallback_equal_weights"):
            logger.warning("  WARNING: Equal-weight fallback was triggered")

        return state

    except Exception as e:
        logger.error(f"Node 11 failed: {e}")
        logger.exception("Full traceback:")

        state.setdefault("errors", []).append(f"Node 11 (adaptive weights) failed: {e}")

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
            "per_stream_adjustments": {},
            "fallback_equal_weights": True,
            "learning_adjustment_applied": DEFAULT_LEARNING_ADJUSTMENT,
            "historical_correlation": 0.5,
            "technical_hit_rate":        None,
            "stock_news_hit_rate":        None,
            "market_news_hit_rate":       None,
            "related_news_hit_rate":      None,
            "technical_sample_count":     0,
            "stock_news_sample_count":    0,
            "market_news_sample_count":   0,
            "related_news_sample_count":  0,
            "signal_threshold":           DEFAULT_SIGNAL_THRESHOLD,
        }
        state.setdefault("node_execution_times", {})["node_11"] = (
            datetime.now() - start_time
        ).total_seconds()

        return state
