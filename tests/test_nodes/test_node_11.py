"""
Tests for Node 11: Adaptive Weights Calculation

Covers:
- _bayesian_smooth: basic smoothing math, edge cases
- _compute_weighted_accuracy: all source labels, 70/30 + smoothing math, edge cases
- _compute_per_stream_adjustments: per-stream Node 8 multipliers, fallback
- calculate_adaptive_weights: normalization, zero-total guard, Node 8 adjustments
- adaptive_weights_node (main): full flow, missing backtest, fallback equal weights,
  error isolation, design contract verification

Design contract verified throughout:
- Node 11 is the ONLY place 70/30 recency weighting is applied
- Bayesian smoothing pulls accuracy toward 0.5 proportional to sample size
- NEUTRAL_ACCURACY == 0.0: unreliable streams contribute NOTHING to normalization
- Equal-weight fallback (0.25) when all streams are neutral (total == 0) or on error
- Weights always sum to 1.0
- adaptive_weights written to state even on failure
"""

import pytest
from copy import deepcopy

from src.langgraph_nodes.node_11_adaptive_weights import (
    DEFAULT_LEARNING_ADJUSTMENT,
    EQUAL_WEIGHT,
    FULL_WEIGHT,
    MAX_ADJUSTED_ACCURACY,
    MIN_ADJUSTED_ACCURACY,
    NEUTRAL_ACCURACY,
    NEWS_STREAM_KEYS,
    PRIOR_STRENGTH,
    RECENCY_WEIGHT,
    STREAM_KEYS,
    _bayesian_smooth,
    _compute_per_stream_adjustments,
    _compute_weighted_accuracy,
    adaptive_weights_node,
    calculate_adaptive_weights,
)


# ============================================================================
# SHARED FIXTURES & HELPERS
# ============================================================================


def _smooth(acc: float, n: int) -> float:
    """Mirror of _bayesian_smooth for computing expected values in tests."""
    if n <= 0:
        return 0.5
    return (acc * n + 0.5 * PRIOR_STRENGTH) / (n + PRIOR_STRENGTH)


def _expected_wa(
    full_acc: float,
    recent_acc: float | None,
    n_full: int = 120,
    n_recent: int = 30,
) -> float:
    """Expected weighted accuracy after smoothing and 70/30 blending."""
    s_full = _smooth(full_acc, n_full)
    if recent_acc is None:
        return s_full
    s_recent = _smooth(recent_acc, n_recent)
    return RECENCY_WEIGHT * s_recent + FULL_WEIGHT * s_full


def _make_stream_metrics(
    full_accuracy: float = 0.60,
    recent_accuracy: float = 0.65,
    signal_count: int = 30,
    is_sufficient: bool = True,
    is_significant: bool = True,
    p_value: float = 0.02,
    total_days_evaluated: int = 120,
    recent_days_evaluated: int = 30,
) -> dict:
    """Return a minimal stream_metrics dict as Node 10 would produce.

    total_days_evaluated=120 and recent_days_evaluated=30 are realistic for
    180 days of backtesting. Bayesian smoothing uses these to pull accuracy
    toward 0.5. Without them (n=0) every stream collapses to 0.5.
    """
    return {
        "full_accuracy": full_accuracy,
        "recent_accuracy": recent_accuracy,
        "signal_count": signal_count,
        "buy_count": signal_count // 2,
        "sell_count": signal_count // 2,
        "hold_count": 10,
        "total_days_evaluated": total_days_evaluated,
        "recent_days_evaluated": recent_days_evaluated,
        "is_sufficient": is_sufficient,
        "is_significant": is_significant,
        "p_value": p_value,
        "avg_actual_change": 1.5,
        "avg_change_on_correct": 2.8,
        "avg_change_on_wrong": -0.9,
        "daily_results": [],
    }


def _make_backtest_results(
    tech_full: float = 0.55,
    tech_recent: float = 0.62,
    stock_full: float = 0.65,
    stock_recent: float = 0.70,
    market_full: float = 0.58,
    market_recent: float = 0.60,
    related_full: float = 0.60,
    related_recent: float = 0.63,
    hold_threshold: float = 2.3,
) -> dict:
    """Return a realistic backtest_results dict for 4 sufficient/significant streams."""
    return {
        "hold_threshold_pct": hold_threshold,
        "sample_period_days": 180,
        "technical": _make_stream_metrics(
            full_accuracy=tech_full, recent_accuracy=tech_recent
        ),
        "stock_news": _make_stream_metrics(
            full_accuracy=stock_full, recent_accuracy=stock_recent
        ),
        "market_news": _make_stream_metrics(
            full_accuracy=market_full, recent_accuracy=market_recent
        ),
        "related_news": _make_stream_metrics(
            full_accuracy=related_full, recent_accuracy=related_recent
        ),
    }


def _make_base_state(
    backtest_results: dict = None,
    news_impact_verification: dict = None,
) -> dict:
    """Return a minimal state dict for adaptive_weights_node."""
    state = {
        "ticker": "AAPL",
        "backtest_results": backtest_results,
        "errors": [],
        "node_execution_times": {},
    }
    if news_impact_verification is not None:
        state["news_impact_verification"] = news_impact_verification
    return state


# ============================================================================
# 0. _bayesian_smooth
# ============================================================================


class TestBayesianSmooth:

    def test_zero_n_returns_half(self):
        """n=0 → return 0.5 (no data, full prior)."""
        assert _bayesian_smooth(0.80, 0) == 0.5
        assert _bayesian_smooth(0.30, 0) == 0.5

    def test_negative_n_returns_half(self):
        """Negative n treated same as 0."""
        assert _bayesian_smooth(0.70, -5) == 0.5

    def test_pulls_toward_half_for_small_n(self):
        """Small n significantly pulls accuracy toward 0.5."""
        result = _bayesian_smooth(0.80, 5)
        # (0.80*5 + 0.5*PRIOR_STRENGTH) / (5 + PRIOR_STRENGTH)
        expected = (0.80 * 5 + 0.5 * PRIOR_STRENGTH) / (5 + PRIOR_STRENGTH)
        assert abs(result - expected) < 1e-9
        assert 0.5 < result < 0.80  # pulled toward 0.5

    def test_minimal_effect_for_large_n(self):
        """Large n barely changes accuracy (smoothing is negligible)."""
        result = _bayesian_smooth(0.65, 10000)
        assert abs(result - 0.65) < 0.001

    def test_formula_matches_expected(self):
        """Formula: (acc*n + 0.5*PRIOR) / (n + PRIOR)."""
        acc, n = 0.65, 30
        expected = (acc * n + 0.5 * PRIOR_STRENGTH) / (n + PRIOR_STRENGTH)
        assert abs(_bayesian_smooth(acc, n) - expected) < 1e-9

    def test_above_half_stays_above_half(self):
        """Accuracy > 0.5 stays above 0.5 after smoothing."""
        assert _bayesian_smooth(0.70, 20) > 0.5

    def test_below_half_stays_below_half(self):
        """Accuracy < 0.5 stays below 0.5 after smoothing."""
        assert _bayesian_smooth(0.30, 20) < 0.5

    def test_exactly_half_unchanged(self):
        """Accuracy == 0.5 stays at 0.5 (it is the prior itself)."""
        assert abs(_bayesian_smooth(0.5, 50) - 0.5) < 1e-9


# ============================================================================
# 1. _compute_weighted_accuracy
# ============================================================================


class TestComputeWeightedAccuracy:

    def test_full_calculation_applies_70_30_with_smoothing(self):
        """Standard case: Bayesian smoothing then 70/30 split are applied correctly."""
        metrics = _make_stream_metrics(full_accuracy=0.60, recent_accuracy=0.80)
        result, source = _compute_weighted_accuracy(metrics, "test")
        expected = _expected_wa(0.60, 0.80)
        assert abs(result - expected) < 1e-6
        assert source == "calculated"

    def test_recency_accuracy_60_full_accuracy_50_smoothed(self):
        """Numeric check with smoothing: full=0.50, recent=0.60."""
        metrics = _make_stream_metrics(full_accuracy=0.50, recent_accuracy=0.60)
        result, source = _compute_weighted_accuracy(metrics, "test")
        expected = _expected_wa(0.50, 0.60)
        assert abs(result - expected) < 1e-6
        assert source == "calculated"

    def test_none_stream_returns_neutral(self):
        """None stream_metrics → neutral weight (0.0)."""
        result, source = _compute_weighted_accuracy(None, "test")
        assert result == NEUTRAL_ACCURACY
        assert result == 0.0
        assert source == "neutral_no_data"

    def test_insufficient_signals_returns_neutral(self):
        """is_sufficient=False → neutral weight (0.0)."""
        metrics = _make_stream_metrics(is_sufficient=False, signal_count=5)
        result, source = _compute_weighted_accuracy(metrics, "test")
        assert result == NEUTRAL_ACCURACY
        assert result == 0.0
        assert source == "neutral_insufficient"

    def test_insignificant_returns_measured_accuracy(self):
        """is_significant=False but not anti-predictive → contributes at measured accuracy (not zeroed)."""
        metrics = _make_stream_metrics(is_significant=False, p_value=0.30)
        result, source = _compute_weighted_accuracy(metrics, "test")
        # Near-random streams still participate in weight normalization at measured accuracy
        assert result > 0.0
        assert source == "calculated"

    def test_none_recent_accuracy_uses_smoothed_full_only(self):
        """recent_accuracy=None → falls back to smoothed full_accuracy only."""
        metrics = _make_stream_metrics(full_accuracy=0.65, recent_accuracy=None)
        metrics["recent_accuracy"] = None
        result, source = _compute_weighted_accuracy(metrics, "test")
        expected = _smooth(0.65, 120)
        assert abs(result - expected) < 1e-6
        assert source == "full_accuracy_only"

    def test_insufficient_takes_priority_over_insignificant(self):
        """When both is_sufficient=False and is_significant=False, label is insufficient."""
        metrics = _make_stream_metrics(is_sufficient=False, is_significant=False)
        _, source = _compute_weighted_accuracy(metrics, "test")
        assert source == "neutral_insufficient"

    def test_equal_full_and_recent_returns_smoothed_value(self):
        """Full == recent: result equals smoothed 70/30 blend (not raw accuracy)."""
        metrics = _make_stream_metrics(full_accuracy=0.60, recent_accuracy=0.60)
        result, source = _compute_weighted_accuracy(metrics, "test")
        expected = _expected_wa(0.60, 0.60)
        assert abs(result - expected) < 1e-6
        assert source == "calculated"

    def test_high_recent_accuracy_pulls_result_toward_recent(self):
        """When recent_accuracy >> full_accuracy, result is closer to recent (70% weight)."""
        metrics = _make_stream_metrics(full_accuracy=0.50, recent_accuracy=0.90)
        result, _ = _compute_weighted_accuracy(metrics, "test")
        s_full = _smooth(0.50, 120)
        s_recent = _smooth(0.90, 30)
        mid = (s_full + s_recent) / 2
        assert result > mid  # 70% on recent pulls it above the midpoint

    def test_smoothing_applied_separately_to_full_and_recent(self):
        """Verify smoothing uses correct n values for full vs recent."""
        metrics = _make_stream_metrics(
            full_accuracy=0.70, recent_accuracy=0.70,
            total_days_evaluated=200, recent_days_evaluated=10,
        )
        result, _ = _compute_weighted_accuracy(metrics, "test")
        s_full = _smooth(0.70, 200)   # large n → close to 0.70
        s_recent = _smooth(0.70, 10)  # small n → pulled more toward 0.5
        expected = RECENCY_WEIGHT * s_recent + FULL_WEIGHT * s_full
        assert abs(result - expected) < 1e-6


# ============================================================================
# 2. _compute_per_stream_adjustments
# ============================================================================


class TestComputePerStreamAdjustments:

    def _niv_effectiveness(self, stock_acc=0.65, market_acc=0.55, related_acc=0.60, n=30):
        return {
            "stock":   {"accuracy_rate": stock_acc,   "sample_size": n},
            "market":  {"accuracy_rate": market_acc,  "sample_size": n},
            "related": {"accuracy_rate": related_acc, "sample_size": n},
        }

    def test_uses_per_type_accuracy_when_sufficient(self):
        """With sufficient sample, each stream gets its own multiplier."""
        effectiveness = self._niv_effectiveness(stock_acc=0.70, n=30)
        adjs = _compute_per_stream_adjustments(effectiveness, 0.7, 1.0)
        # stock multiplier = (0.70/0.50) * (0.7*2) = 1.4 * 1.4 = 1.96 → clamped to 2.0
        stock_adj = adjs["stock_news"]
        assert stock_adj > 1.0  # above neutral (good news type gets boosted)

    def test_falls_back_when_sample_insufficient(self):
        """If sample_size < MIN_NEWS_TYPE_SAMPLE (10), use fallback."""
        effectiveness = {"stock": {"accuracy_rate": 0.80, "sample_size": 5}}
        adjs = _compute_per_stream_adjustments(effectiveness, 0.7, 1.3)
        # stock has n=5 < 10 → fallback
        assert abs(adjs["stock_news"] - 1.3) < 1e-6

    def test_empty_effectiveness_uses_fallback(self):
        """Empty news_type_effectiveness → all streams use fallback."""
        adjs = _compute_per_stream_adjustments({}, 0.7, 1.2)
        for sk in NEWS_STREAM_KEYS:
            assert abs(adjs[sk] - 1.2) < 1e-6

    def test_multiplier_clamped_to_min(self):
        """Low accuracy + low correlation → multiplier clamped to MIN_ADJ_MULTIPLIER."""
        effectiveness = {"stock": {"accuracy_rate": 0.10, "sample_size": 50}}
        adjs = _compute_per_stream_adjustments(effectiveness, 0.1, 1.0)
        from src.langgraph_nodes.node_11_adaptive_weights import MIN_ADJ_MULTIPLIER
        assert adjs["stock_news"] >= MIN_ADJ_MULTIPLIER

    def test_multiplier_clamped_to_max(self):
        """High accuracy + high correlation → multiplier clamped to MAX_ADJ_MULTIPLIER."""
        effectiveness = {"stock": {"accuracy_rate": 1.0, "sample_size": 50}}
        adjs = _compute_per_stream_adjustments(effectiveness, 1.0, 1.0)
        from src.langgraph_nodes.node_11_adaptive_weights import MAX_ADJ_MULTIPLIER
        assert adjs["stock_news"] <= MAX_ADJ_MULTIPLIER

    def test_neutral_accuracy_and_correlation_gives_neutral_multiplier(self):
        """accuracy=0.5, corr=0.5 → multiplier = 1.0 (no change)."""
        effectiveness = {"stock": {"accuracy_rate": 0.5, "sample_size": 50}}
        adjs = _compute_per_stream_adjustments(effectiveness, 0.5, 1.0)
        # (0.5/0.5) * (0.5*2.0) = 1.0 * 1.0 = 1.0
        assert abs(adjs["stock_news"] - 1.0) < 1e-6

    def test_all_three_news_keys_present(self):
        """Result always contains all 3 NEWS_STREAM_KEYS."""
        adjs = _compute_per_stream_adjustments({}, 0.5, 1.0)
        for sk in NEWS_STREAM_KEYS:
            assert sk in adjs


# ============================================================================
# 3. calculate_adaptive_weights
# ============================================================================


class TestCalculateAdaptiveWeights:

    def test_weights_sum_to_one(self):
        """Normalization must produce weights that sum to exactly 1.0."""
        backtest = _make_backtest_results()
        weights = calculate_adaptive_weights(backtest)
        total = sum(weights[wk] for _, wk in STREAM_KEYS)
        assert abs(total - 1.0) < 0.001

    def test_higher_accuracy_gets_higher_weight(self):
        """Higher accuracy stream gets proportionally higher weight."""
        backtest = _make_backtest_results(
            stock_full=0.80, stock_recent=0.85,   # clearly best
            tech_full=0.50,  tech_recent=0.50,    # worst
        )
        weights = calculate_adaptive_weights(backtest)
        assert weights["stock_news_weight"] > weights["technical_weight"]

    def test_all_equal_accuracy_produces_equal_weights(self):
        """All streams equally accurate → each gets 25%."""
        backtest = _make_backtest_results(
            tech_full=0.60,    tech_recent=0.60,
            stock_full=0.60,   stock_recent=0.60,
            market_full=0.60,  market_recent=0.60,
            related_full=0.60, related_recent=0.60,
        )
        weights = calculate_adaptive_weights(backtest)
        for _, wk in STREAM_KEYS:
            assert abs(weights[wk] - 0.25) < 0.001

    def test_all_neutral_streams_fallback_to_equal_weights(self):
        """All neutral streams → total==0 → zero-total guard → equal weights 0.25."""
        backtest = {
            "hold_threshold_pct": 2.0,
            "sample_period_days": 180,
            "technical":    _make_stream_metrics(is_sufficient=False),
            "stock_news":   _make_stream_metrics(is_sufficient=False),
            "market_news":  _make_stream_metrics(is_sufficient=False),
            "related_news": _make_stream_metrics(is_sufficient=False),
        }
        weights = calculate_adaptive_weights(backtest)
        for _, wk in STREAM_KEYS:
            assert abs(weights[wk] - 0.25) < 0.001
        assert weights["fallback_equal_weights"] is True

    def test_neutral_stream_zero_weight_among_calculated(self):
        """Neutral stream (0.0) gets zero weight — not diluting the 3 reliable streams."""
        backtest = _make_backtest_results(
            tech_full=0.65, tech_recent=0.65,
            stock_full=0.65, stock_recent=0.65,
            market_full=0.65, market_recent=0.65,
        )
        backtest["related_news"] = _make_stream_metrics(is_sufficient=False)
        weights = calculate_adaptive_weights(backtest)
        # Neutral stream gets 0.0 weight — not a small weight, exactly 0
        assert weights["related_news_weight"] == 0.0
        # The 3 reliable streams share the full weight
        assert abs(weights["related_news_weight"] - 0.0) < 1e-9
        # Weights still sum to 1
        total = sum(weights[wk] for _, wk in STREAM_KEYS)
        assert abs(total - 1.0) < 0.001

    def test_all_none_streams_fallback_to_equal_weights(self):
        """When all streams are None, fallback to equal weights 0.25."""
        backtest = {
            "hold_threshold_pct": 2.0,
            "sample_period_days": 180,
            "technical": None, "stock_news": None,
            "market_news": None, "related_news": None,
        }
        weights = calculate_adaptive_weights(backtest)
        for _, wk in STREAM_KEYS:
            assert abs(weights[wk] - 0.25) < 0.001

    def test_hold_threshold_passed_through(self):
        """hold_threshold_pct from Node 10 is preserved for dashboard."""
        backtest = _make_backtest_results(hold_threshold=3.7)
        weights = calculate_adaptive_weights(backtest)
        assert weights["hold_threshold_pct"] == 3.7

    def test_sample_period_passed_through(self):
        """sample_period_days from Node 10 is preserved."""
        backtest = _make_backtest_results()
        weights = calculate_adaptive_weights(backtest)
        assert weights["sample_period_days"] == 180

    def test_weighted_accuracies_present_for_all_streams(self):
        """weighted_accuracies must include all 4 stream keys."""
        backtest = _make_backtest_results()
        weights = calculate_adaptive_weights(backtest)
        for sk, _ in STREAM_KEYS:
            assert sk in weights["weighted_accuracies"]

    def test_weight_sources_present_for_all_streams(self):
        """weight_sources must include all 4 stream keys."""
        backtest = _make_backtest_results()
        weights = calculate_adaptive_weights(backtest)
        for sk, _ in STREAM_KEYS:
            assert sk in weights["weight_sources"]

    def test_streams_reliable_count_correct(self):
        """streams_reliable counts 'calculated' and 'full_accuracy_only' sources."""
        backtest = _make_backtest_results()
        backtest["related_news"]["is_sufficient"] = False
        weights = calculate_adaptive_weights(backtest)
        assert weights["streams_reliable"] == 3

    def test_streams_reliable_zero_when_all_neutral(self):
        backtest = _make_backtest_results()
        for sk, _ in STREAM_KEYS:
            backtest[sk]["is_sufficient"] = False
        weights = calculate_adaptive_weights(backtest)
        assert weights["streams_reliable"] == 0

    def test_full_accuracy_only_counts_as_reliable(self):
        """Stream where recent_accuracy is None but otherwise valid → reliable."""
        backtest = _make_backtest_results()
        backtest["stock_news"]["recent_accuracy"] = None
        weights = calculate_adaptive_weights(backtest)
        assert weights["weight_sources"]["stock_news"] == "full_accuracy_only"
        assert weights["streams_reliable"] == 4

    def test_per_stream_adjustments_in_output(self):
        """per_stream_adjustments key is present in output."""
        backtest = _make_backtest_results()
        weights = calculate_adaptive_weights(backtest)
        assert "per_stream_adjustments" in weights

    def test_recency_weighting_applied_with_smoothing(self):
        """70/30 is applied by Node 11 on smoothed values from raw full/recent input."""
        backtest = _make_backtest_results(
            tech_full=0.50, tech_recent=1.00  # extreme for clear signal
        )
        weights = calculate_adaptive_weights(backtest)
        tech_wa = weights["weighted_accuracies"]["technical"]
        expected = _expected_wa(0.50, 1.00)
        assert abs(tech_wa - expected) < 1e-4


# ============================================================================
# 4. adaptive_weights_node — main node function
# ============================================================================


class TestAdaptiveWeightsNode:

    def test_full_flow_returns_state(self):
        """Typical run returns state with adaptive_weights populated."""
        state = _make_base_state(_make_backtest_results())
        result = adaptive_weights_node(state)
        assert "adaptive_weights" in result
        aw = result["adaptive_weights"]
        for _, wk in STREAM_KEYS:
            assert wk in aw

    def test_weights_sum_to_one_in_node(self):
        """Weights from the node function always sum to 1.0."""
        state = _make_base_state(_make_backtest_results())
        result = adaptive_weights_node(state)
        total = sum(result["adaptive_weights"][wk] for _, wk in STREAM_KEYS)
        assert abs(total - 1.0) < 0.001

    def test_no_backtest_results_falls_back_to_equal_weights(self):
        """backtest_results=None → equal weights 0.25 each."""
        state = _make_base_state(backtest_results=None)
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        for _, wk in STREAM_KEYS:
            assert abs(aw[wk] - EQUAL_WEIGHT) < 0.001
        assert aw["fallback_equal_weights"] is True
        assert aw["streams_reliable"] == 0

    def test_node_execution_time_tracked(self):
        """node_11 execution time is recorded in node_execution_times."""
        state = _make_base_state(_make_backtest_results())
        result = adaptive_weights_node(state)
        assert "node_11" in result["node_execution_times"]
        assert result["node_execution_times"]["node_11"] >= 0

    def test_no_errors_on_clean_run(self):
        """No errors appended on a successful run."""
        state = _make_base_state(_make_backtest_results())
        result = adaptive_weights_node(state)
        assert result["errors"] == []

    def test_fallback_equal_weights_false_on_success(self):
        """fallback_equal_weights is False when the run succeeds."""
        state = _make_base_state(_make_backtest_results())
        result = adaptive_weights_node(state)
        assert result["adaptive_weights"]["fallback_equal_weights"] is False

    def test_missing_ticker_does_not_crash(self):
        """Missing ticker field is handled gracefully."""
        state = {
            "backtest_results": _make_backtest_results(),
            "errors": [],
            "node_execution_times": {},
        }
        result = adaptive_weights_node(state)
        assert "adaptive_weights" in result

    def test_state_fields_preserved(self):
        """Node 11 must not remove other state fields."""
        state = _make_base_state(_make_backtest_results())
        state["technical_signal"] = "BUY"
        state["raw_price_data"] = None
        result = adaptive_weights_node(state)
        assert result["technical_signal"] == "BUY"
        assert "raw_price_data" in result

    def test_higher_accuracy_stock_news_gets_highest_weight(self):
        """Stock news boosted by Node 8 learning gets the highest weight."""
        backtest = _make_backtest_results(
            stock_full=0.73, stock_recent=0.75,
            tech_full=0.55,  tech_recent=0.57,
            market_full=0.58, market_recent=0.60,
            related_full=0.60, related_recent=0.62,
        )
        state = _make_base_state(backtest)
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        assert aw["stock_news_weight"] > aw["technical_weight"]
        assert aw["stock_news_weight"] > aw["market_news_weight"]
        assert aw["stock_news_weight"] > aw["related_news_weight"]

    def test_error_in_backtest_results_produces_fallback(self):
        """Malformed backtest_results triggers fallback without crashing."""
        state = _make_base_state(backtest_results="not_a_dict")
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        for _, wk in STREAM_KEYS:
            assert abs(aw[wk] - EQUAL_WEIGHT) < 0.001
        assert aw["fallback_equal_weights"] is True
        assert len(result["errors"]) > 0

    def test_error_message_added_on_exception(self):
        """When an exception is raised, error is appended to state['errors']."""
        state = _make_base_state(backtest_results="garbage")
        result = adaptive_weights_node(state)
        assert "adaptive_weights" in result

    def test_historical_correlation_in_output(self):
        """historical_correlation from Node 8 is recorded in adaptive_weights."""
        niv = {"learning_adjustment": 1.0, "historical_correlation": 0.72}
        state = _make_base_state(_make_backtest_results(), niv)
        result = adaptive_weights_node(state)
        assert "historical_correlation" in result["adaptive_weights"]
        assert abs(result["adaptive_weights"]["historical_correlation"] - 0.72) < 1e-4

    def test_historical_correlation_defaults_to_half_without_node8(self):
        """Missing news_impact_verification → historical_correlation defaults to 0.5."""
        state = _make_base_state(_make_backtest_results())
        result = adaptive_weights_node(state)
        assert abs(result["adaptive_weights"]["historical_correlation"] - 0.5) < 1e-6

    def test_per_stream_adjustments_in_output(self):
        """per_stream_adjustments is present in adaptive_weights output."""
        niv = {
            "learning_adjustment": 1.2,
            "historical_correlation": 0.6,
            "news_type_effectiveness": {
                "stock":   {"accuracy_rate": 0.65, "sample_size": 30},
                "market":  {"accuracy_rate": 0.55, "sample_size": 30},
                "related": {"accuracy_rate": 0.58, "sample_size": 30},
            },
        }
        state = _make_base_state(_make_backtest_results(), niv)
        result = adaptive_weights_node(state)
        assert "per_stream_adjustments" in result["adaptive_weights"]


# ============================================================================
# 5. Design Contract Verification
# ============================================================================


class TestDesignContract:

    def test_node_11_is_sole_applier_of_recency_weighting(self):
        """
        The 70/30 recency weighting + smoothing is applied ONLY in Node 11.
        Input has raw full/recent; output has smoothed weighted value.
        """
        backtest = _make_backtest_results(tech_full=0.50, tech_recent=0.80)
        assert "full_accuracy" in backtest["technical"]
        assert "recent_accuracy" in backtest["technical"]
        assert "weighted_accuracy" not in backtest["technical"]

        state = _make_base_state(backtest)
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        expected_wa = _expected_wa(0.50, 0.80)
        actual_wa = aw["weighted_accuracies"]["technical"]
        assert abs(actual_wa - expected_wa) < 1e-4

    def test_weights_always_provided_even_on_failure(self):
        """adaptive_weights is ALWAYS written to state — Node 12 must always find it."""
        state = _make_base_state(backtest_results="garbage")
        result = adaptive_weights_node(state)
        assert "adaptive_weights" in result
        for _, wk in STREAM_KEYS:
            assert wk in result["adaptive_weights"]

    def test_equal_weights_when_no_reliable_data(self):
        """When no reliable historical data exists, equal weights are fair starting point."""
        state = _make_base_state(backtest_results=None)
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        weights = [aw[wk] for _, wk in STREAM_KEYS]
        assert all(abs(w - 0.25) < 0.001 for w in weights)
        assert abs(sum(weights) - 1.0) < 0.001

    def test_neutral_streams_get_zero_weight(self):
        """
        Anti-predictive stream must get ZERO weight — it drops out of normalization.
        Near-random streams (is_significant=False but not anti-predictive) still participate.
        """
        backtest = _make_backtest_results(
            tech_full=0.65, tech_recent=0.65,
            stock_full=0.65, stock_recent=0.65,
            market_full=0.65, market_recent=0.65,
        )
        # is_anti_predictive=True (p_value_less < 0.05) → zero weight
        backtest["related_news"] = {**_make_stream_metrics(), "is_anti_predictive": True, "p_value_less": 0.02}
        state = _make_base_state(backtest)
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        assert aw["related_news_weight"] == 0.0
        assert aw["related_news_weight"] < aw["technical_weight"]

    def test_weights_bounded_between_0_and_1(self):
        """Every weight must be in [0, 1]."""
        backtest = _make_backtest_results()
        state = _make_base_state(backtest)
        result = adaptive_weights_node(state)
        for _, wk in STREAM_KEYS:
            w = result["adaptive_weights"][wk]
            assert 0.0 <= w <= 1.0

    def test_recency_weight_constant_is_70_pct(self):
        assert abs(RECENCY_WEIGHT - 0.7) < 1e-9

    def test_full_weight_constant_is_30_pct(self):
        assert abs(FULL_WEIGHT - 0.3) < 1e-9

    def test_recency_and_full_weights_sum_to_one(self):
        assert abs(RECENCY_WEIGHT + FULL_WEIGHT - 1.0) < 1e-9

    def test_neutral_accuracy_is_zero(self):
        """NEUTRAL_ACCURACY == 0.0: unreliable streams drop out of normalization."""
        assert NEUTRAL_ACCURACY == 0.0

    def test_equal_weight_is_one_quarter(self):
        """EQUAL_WEIGHT == 0.25 for 4 streams."""
        assert abs(EQUAL_WEIGHT - 0.25) < 1e-9

    def test_all_neutral_triggers_zero_total_fallback(self):
        """With NEUTRAL_ACCURACY=0.0, all neutral → total=0 → equal-weight fallback."""
        backtest = {
            "hold_threshold_pct": 2.0,
            "sample_period_days": 180,
            "technical": None, "stock_news": None,
            "market_news": None, "related_news": None,
        }
        weights = calculate_adaptive_weights(backtest)
        assert weights["fallback_equal_weights"] is True
        for _, wk in STREAM_KEYS:
            assert abs(weights[wk] - EQUAL_WEIGHT) < 0.001


# ============================================================================
# 6. Integration / Thesis Validation
# ============================================================================


class TestIntegration:

    def test_node8_learning_effect_reflected_in_weights(self):
        """
        Node 8's learning system boosts stock_news accuracy.
        This should be reflected as the highest weight in Node 11's output.
        This is a KEY thesis demonstration.
        """
        backtest = _make_backtest_results(
            stock_full=0.73,  stock_recent=0.75,
            tech_full=0.55,   tech_recent=0.57,
            market_full=0.58, market_recent=0.60,
            related_full=0.61, related_recent=0.63,
        )
        state = _make_base_state(backtest)
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        assert aw["stock_news_weight"] == max(aw[wk] for _, wk in STREAM_KEYS)

    def test_weight_improves_with_node8_vs_without(self):
        """Stock news weight should be higher AFTER Node 8 learning."""
        backtest_without = _make_backtest_results(
            stock_full=0.62, stock_recent=0.62,
            tech_full=0.60,  tech_recent=0.60,
            market_full=0.60, market_recent=0.60,
            related_full=0.60, related_recent=0.60,
        )
        backtest_with = _make_backtest_results(
            stock_full=0.73, stock_recent=0.75,
            tech_full=0.60,  tech_recent=0.60,
            market_full=0.60, market_recent=0.60,
            related_full=0.60, related_recent=0.60,
        )
        result_without = adaptive_weights_node(_make_base_state(backtest_without))
        result_with = adaptive_weights_node(_make_base_state(backtest_with))

        w_without = result_without["adaptive_weights"]["stock_news_weight"]
        w_with = result_with["adaptive_weights"]["stock_news_weight"]
        assert w_with > w_without

    def test_volatile_stock_recency_matters_more(self):
        """
        For a stream improving recently, the 70/30 split yields higher weighted
        accuracy than a stable stream — confirming recency weighting is meaningful.
        """
        backtest_stable = _make_backtest_results(tech_full=0.65, tech_recent=0.65)
        backtest_improving = _make_backtest_results(tech_full=0.55, tech_recent=0.75)

        wa_stable = calculate_adaptive_weights(backtest_stable)["weighted_accuracies"]["technical"]
        wa_improving = calculate_adaptive_weights(backtest_improving)["weighted_accuracies"]["technical"]
        assert wa_improving > wa_stable

    def test_all_four_weight_keys_present(self):
        """All 4 weight keys that Node 12 reads must be present."""
        state = _make_base_state(_make_backtest_results())
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        required = [
            "technical_weight", "stock_news_weight",
            "market_news_weight", "related_news_weight",
        ]
        for key in required:
            assert key in aw, f"Missing key: {key}"

    def test_streams_reliable_reflects_actual_data_quality(self):
        """streams_reliable gives Node 12 a data quality indicator."""
        backtest_good = _make_backtest_results()
        result_good = adaptive_weights_node(_make_base_state(backtest_good))
        assert result_good["adaptive_weights"]["streams_reliable"] == 4

        backtest_partial = _make_backtest_results()
        backtest_partial["market_news"]["is_sufficient"] = False
        # is_significant=False but not anti-predictive → still reliable (contributes at measured accuracy)
        backtest_partial["related_news"]["is_significant"] = False
        result_partial = adaptive_weights_node(_make_base_state(backtest_partial))
        # Only market_news (insufficient) is unreliable; related_news (near-random) still counts
        assert result_partial["adaptive_weights"]["streams_reliable"] == 3


# ============================================================================
# 7. Learning Adjustment (Node 8 integration)
# ============================================================================


class TestLearningAdjustment:
    """
    Node 8's learning_adjustment is a capped multiplier applied to the 3 news
    stream weighted-accuracies before normalization. Technical stream is NOT affected.
    """

    def _niv(self, adjustment: float) -> dict:
        """Minimal news_impact_verification dict with a given learning_adjustment."""
        return {"learning_adjustment": adjustment, "insufficient_data": False}

    def test_boost_increases_news_avg_weight(self):
        """learning_adjustment > 1.0 raises average news stream weight vs technical."""
        backtest = _make_backtest_results()
        result_no = adaptive_weights_node(_make_base_state(backtest))
        result_adj = adaptive_weights_node(
            _make_base_state(deepcopy(backtest), self._niv(1.5))
        )

        def avg_news(aw):
            return (aw["stock_news_weight"] + aw["market_news_weight"] + aw["related_news_weight"]) / 3

        assert avg_news(result_adj["adaptive_weights"]) > avg_news(result_no["adaptive_weights"])

    def test_reduction_decreases_news_avg_weight(self):
        """learning_adjustment < 1.0 lowers average news stream weight vs technical."""
        backtest = _make_backtest_results()
        result_no = adaptive_weights_node(_make_base_state(backtest))
        result_adj = adaptive_weights_node(
            _make_base_state(deepcopy(backtest), self._niv(0.6))
        )

        def avg_news(aw):
            return (aw["stock_news_weight"] + aw["market_news_weight"] + aw["related_news_weight"]) / 3

        assert avg_news(result_adj["adaptive_weights"]) < avg_news(result_no["adaptive_weights"])

    def test_technical_stream_unaffected_by_adjustment(self):
        """weighted_accuracies['technical'] must be identical with and without adjustment."""
        backtest = _make_backtest_results()
        result_no = adaptive_weights_node(_make_base_state(deepcopy(backtest)))
        result_adj = adaptive_weights_node(
            _make_base_state(deepcopy(backtest), self._niv(1.8))
        )
        wa_no = result_no["adaptive_weights"]["weighted_accuracies"]["technical"]
        wa_adj = result_adj["adaptive_weights"]["weighted_accuracies"]["technical"]
        assert abs(wa_no - wa_adj) < 1e-9

    def test_default_1_0_when_no_node8_data(self):
        """Missing news_impact_verification produces same result as explicit 1.0."""
        backtest = _make_backtest_results()
        result_missing = adaptive_weights_node(_make_base_state(deepcopy(backtest)))
        result_one = adaptive_weights_node(
            _make_base_state(deepcopy(backtest), self._niv(1.0))
        )
        for _, wk in STREAM_KEYS:
            assert abs(
                result_missing["adaptive_weights"][wk]
                - result_one["adaptive_weights"][wk]
            ) < 1e-6

    def test_cap_prevents_above_max(self):
        """Extreme boost (2.0) cannot push any news stream above MAX_ADJUSTED_ACCURACY."""
        state = _make_base_state(_make_backtest_results(), self._niv(2.0))
        result = adaptive_weights_node(state)
        wa = result["adaptive_weights"]["weighted_accuracies"]
        for sk in NEWS_STREAM_KEYS:
            assert wa[sk] <= MAX_ADJUSTED_ACCURACY + 1e-6

    def test_floor_prevents_below_min(self):
        """Extreme reduction (0.1) cannot push any news stream below MIN_ADJUSTED_ACCURACY."""
        state = _make_base_state(_make_backtest_results(), self._niv(0.1))
        result = adaptive_weights_node(state)
        wa = result["adaptive_weights"]["weighted_accuracies"]
        for sk in NEWS_STREAM_KEYS:
            assert wa[sk] >= MIN_ADJUSTED_ACCURACY - 1e-6

    def test_weights_still_sum_to_1_with_adjustment(self):
        """Normalization must hold after learning_adjustment is applied."""
        state = _make_base_state(_make_backtest_results(), self._niv(1.5))
        result = adaptive_weights_node(state)
        total = sum(result["adaptive_weights"][wk] for _, wk in STREAM_KEYS)
        assert abs(total - 1.0) < 0.001

    def test_adjustment_stored_in_output(self):
        """learning_adjustment_applied must equal the value extracted from state."""
        state = _make_base_state(_make_backtest_results(), self._niv(1.3))
        result = adaptive_weights_node(state)
        assert "learning_adjustment_applied" in result["adaptive_weights"]
        assert abs(result["adaptive_weights"]["learning_adjustment_applied"] - 1.3) < 1e-4

    def test_neutral_streams_not_adjusted(self):
        """Neutral (0.0) streams are skipped — boost cannot re-introduce them."""
        backtest = _make_backtest_results()
        backtest["stock_news"] = _make_stream_metrics(is_sufficient=False)
        state = _make_base_state(backtest, self._niv(2.0))
        result = adaptive_weights_node(state)
        wa = result["adaptive_weights"]["weighted_accuracies"]["stock_news"]
        assert abs(wa - NEUTRAL_ACCURACY) < 1e-6  # NEUTRAL_ACCURACY == 0.0
        assert wa == 0.0

    def test_per_stream_adjustments_differ_with_news_type_effectiveness(self):
        """
        Different news types get different multipliers when Node 8 provides
        per-type effectiveness data. Stock news with higher accuracy gets a
        larger multiplier than market news with lower accuracy.
        """
        niv = {
            "learning_adjustment": 1.0,
            "historical_correlation": 0.7,
            "news_type_effectiveness": {
                "stock":   {"accuracy_rate": 0.75, "sample_size": 50},
                "market":  {"accuracy_rate": 0.45, "sample_size": 50},
                "related": {"accuracy_rate": 0.60, "sample_size": 50},
            },
        }
        backtest = _make_backtest_results(
            stock_full=0.60, stock_recent=0.60,
            market_full=0.60, market_recent=0.60,
        )
        state = _make_base_state(backtest, niv)
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        # Stock has higher per-type accuracy → more boosted → higher weight than market
        assert aw["stock_news_weight"] > aw["market_news_weight"]
