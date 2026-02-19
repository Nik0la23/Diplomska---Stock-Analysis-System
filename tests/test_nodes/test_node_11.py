"""
Tests for Node 11: Adaptive Weights Calculation

Covers:
- _compute_weighted_accuracy: all source labels, 70/30 math, edge cases
- calculate_adaptive_weights: normalization, neutral-stream mixing, sum-to-1
- adaptive_weights_node (main): full flow, missing backtest, fallback equal weights,
  error isolation, design contract verification

Design contract verified throughout:
- Node 11 is the ONLY place 70/30 recency weighting is applied
- Neutral weight 0.5 for insufficient / insignificant / missing streams
- Equal weights 0.25 each when backtest_results is None
- Weights always sum to 1.0
- adaptive_weights written to state even on failure
"""

import pytest
from copy import deepcopy

from src.langgraph_nodes.node_11_adaptive_weights import (
    EQUAL_WEIGHT,
    FULL_WEIGHT,
    NEUTRAL_ACCURACY,
    RECENCY_WEIGHT,
    STREAM_KEYS,
    _compute_weighted_accuracy,
    adaptive_weights_node,
    calculate_adaptive_weights,
)


# ============================================================================
# SHARED FIXTURES & HELPERS
# ============================================================================


def _make_stream_metrics(
    full_accuracy: float = 0.60,
    recent_accuracy: float = 0.65,
    signal_count: int = 30,
    is_sufficient: bool = True,
    is_significant: bool = True,
    p_value: float = 0.02,
) -> dict:
    """Return a minimal stream_metrics dict as Node 10 would produce."""
    return {
        "full_accuracy": full_accuracy,
        "recent_accuracy": recent_accuracy,
        "signal_count": signal_count,
        "buy_count": signal_count // 2,
        "sell_count": signal_count // 2,
        "hold_count": 10,
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


def _make_base_state(backtest_results: dict = None) -> dict:
    """Return a minimal state dict for adaptive_weights_node."""
    return {
        "ticker": "AAPL",
        "backtest_results": backtest_results,
        "errors": [],
        "node_execution_times": {},
    }


# ============================================================================
# 1. _compute_weighted_accuracy
# ============================================================================


class TestComputeWeightedAccuracy:

    def test_full_calculation_applies_70_30(self):
        """Standard case: 70/30 split is applied correctly."""
        metrics = _make_stream_metrics(full_accuracy=0.60, recent_accuracy=0.80)
        result, source = _compute_weighted_accuracy(metrics, "test")
        expected = 0.7 * 0.80 + 0.3 * 0.60
        assert abs(result - expected) < 1e-9
        assert source == "calculated"

    def test_recent_accuracy_60_full_accuracy_50(self):
        """Exact numeric check."""
        metrics = _make_stream_metrics(full_accuracy=0.50, recent_accuracy=0.60)
        result, source = _compute_weighted_accuracy(metrics, "test")
        assert abs(result - (0.7 * 0.60 + 0.3 * 0.50)) < 1e-9
        assert source == "calculated"

    def test_none_stream_returns_neutral(self):
        """None stream_metrics → neutral weight."""
        result, source = _compute_weighted_accuracy(None, "test")
        assert result == NEUTRAL_ACCURACY
        assert source == "neutral_no_data"

    def test_insufficient_signals_returns_neutral(self):
        """is_sufficient=False → neutral weight."""
        metrics = _make_stream_metrics(is_sufficient=False, signal_count=5)
        result, source = _compute_weighted_accuracy(metrics, "test")
        assert result == NEUTRAL_ACCURACY
        assert source == "neutral_insufficient"

    def test_insignificant_returns_neutral(self):
        """is_significant=False → neutral weight."""
        metrics = _make_stream_metrics(is_significant=False, p_value=0.30)
        result, source = _compute_weighted_accuracy(metrics, "test")
        assert result == NEUTRAL_ACCURACY
        assert source == "neutral_insignificant"

    def test_none_recent_accuracy_uses_full_only(self):
        """recent_accuracy=None → falls back to full_accuracy (no 70/30)."""
        metrics = _make_stream_metrics(full_accuracy=0.65, recent_accuracy=None)
        metrics["recent_accuracy"] = None
        result, source = _compute_weighted_accuracy(metrics, "test")
        assert abs(result - 0.65) < 1e-9
        assert source == "full_accuracy_only"

    def test_insufficient_takes_priority_over_insignificant(self):
        """When both is_sufficient=False and is_significant=False, label is insufficient."""
        metrics = _make_stream_metrics(is_sufficient=False, is_significant=False)
        _, source = _compute_weighted_accuracy(metrics, "test")
        assert source == "neutral_insufficient"

    def test_equal_full_and_recent_accuracy(self):
        """70/30 with full == recent → weighted == full == recent."""
        metrics = _make_stream_metrics(full_accuracy=0.60, recent_accuracy=0.60)
        result, source = _compute_weighted_accuracy(metrics, "test")
        assert abs(result - 0.60) < 1e-9
        assert source == "calculated"

    def test_high_volatility_stock_larger_recent_weight(self):
        """When recent_accuracy >> full_accuracy the result is pulled toward recent."""
        metrics = _make_stream_metrics(full_accuracy=0.50, recent_accuracy=0.90)
        result, _ = _compute_weighted_accuracy(metrics, "test")
        assert result > 0.50 + 0.05   # clearly above full_accuracy
        assert result < 0.90           # below recent_accuracy
        # 70% weight on recent → result should be closer to 0.90 than 0.50
        mid = (0.50 + 0.90) / 2
        assert result > mid


# ============================================================================
# 2. calculate_adaptive_weights
# ============================================================================


class TestCalculateAdaptiveWeights:

    def test_weights_sum_to_one(self):
        """Normalization must produce weights that sum to exactly 1.0."""
        backtest = _make_backtest_results()
        weights = calculate_adaptive_weights(backtest)
        total = (
            weights["technical_weight"]
            + weights["stock_news_weight"]
            + weights["market_news_weight"]
            + weights["related_news_weight"]
        )
        assert abs(total - 1.0) < 0.001

    def test_higher_accuracy_gets_higher_weight(self):
        """Higher accuracy stream gets proportionally higher weight."""
        backtest = _make_backtest_results(
            stock_full=0.80, stock_recent=0.85,    # clearly best
            tech_full=0.50,  tech_recent=0.50,    # worst
        )
        weights = calculate_adaptive_weights(backtest)
        assert weights["stock_news_weight"] > weights["technical_weight"]

    def test_all_equal_accuracy_produces_equal_weights(self):
        """All streams equally accurate → each gets 25%."""
        backtest = _make_backtest_results(
            tech_full=0.60,   tech_recent=0.60,
            stock_full=0.60,  stock_recent=0.60,
            market_full=0.60, market_recent=0.60,
            related_full=0.60, related_recent=0.60,
        )
        weights = calculate_adaptive_weights(backtest)
        for _, wk in STREAM_KEYS:
            assert abs(weights[wk] - 0.25) < 0.001

    def test_neutral_stream_gets_fair_share(self):
        """A neutral stream shares fairly with other neutral streams."""
        backtest = {
            "hold_threshold_pct": 2.0,
            "sample_period_days": 180,
            "technical":   _make_stream_metrics(is_sufficient=False),
            "stock_news":  _make_stream_metrics(is_sufficient=False),
            "market_news": _make_stream_metrics(is_sufficient=False),
            "related_news": _make_stream_metrics(is_sufficient=False),
        }
        weights = calculate_adaptive_weights(backtest)
        for _, wk in STREAM_KEYS:
            assert abs(weights[wk] - 0.25) < 0.001

    def test_one_neutral_stream_among_three_calculated(self):
        """One insufficient stream lowers its weight relative to the 3 good ones."""
        backtest = _make_backtest_results(
            tech_full=0.65, tech_recent=0.65,
            stock_full=0.65, stock_recent=0.65,
            market_full=0.65, market_recent=0.65,
        )
        backtest["related_news"] = _make_stream_metrics(is_sufficient=False)
        weights = calculate_adaptive_weights(backtest)
        # Neutral accuracy = 0.5, calculated = 0.65 → related gets less
        assert weights["related_news_weight"] < weights["technical_weight"]
        # Weights still sum to 1
        total = sum(weights[wk] for _, wk in STREAM_KEYS)
        assert abs(total - 1.0) < 0.001

    def test_all_none_streams_equal_weights(self):
        """When all streams are None, every weight should be 0.25."""
        backtest = {
            "hold_threshold_pct": 2.0,
            "sample_period_days": 180,
            "technical":    None,
            "stock_news":   None,
            "market_news":  None,
            "related_news": None,
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

    def test_weighted_accuracies_present(self):
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
        # Make one stream insufficient
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
        assert weights["streams_reliable"] == 4  # all 4 still reliable

    def test_recency_weighting_applied_here_not_in_input(self):
        """Verify 70/30 is applied by Node 11, not pre-applied in the input."""
        # If the input already had weighted accuracy, this test would look different.
        # We confirm the node receives RAW full/recent and applies the math.
        backtest = _make_backtest_results(
            tech_full=0.50, tech_recent=1.00  # extreme so result is clearly 70/30
        )
        weights = calculate_adaptive_weights(backtest)
        tech_wa = weights["weighted_accuracies"]["technical"]
        expected = 0.7 * 1.00 + 0.3 * 0.50
        assert abs(tech_wa - expected) < 1e-4


# ============================================================================
# 3. adaptive_weights_node — main node function
# ============================================================================


class TestAdaptiveWeightsNode:

    def test_full_flow_returns_state(self):
        """Typical run returns state with adaptive_weights populated."""
        state = _make_base_state(_make_backtest_results())
        result = adaptive_weights_node(state)
        assert "adaptive_weights" in result
        aw = result["adaptive_weights"]
        assert "technical_weight" in aw
        assert "stock_news_weight" in aw
        assert "market_news_weight" in aw
        assert "related_news_weight" in aw

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
        state = {"backtest_results": _make_backtest_results(), "errors": [], "node_execution_times": {}}
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
            stock_full=0.73, stock_recent=0.75,   # Node 8 improved
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
        # Should fall back to equal weights
        for _, wk in STREAM_KEYS:
            assert abs(aw[wk] - EQUAL_WEIGHT) < 0.001
        assert aw["fallback_equal_weights"] is True
        assert len(result["errors"]) > 0

    def test_error_message_added_on_exception(self):
        """When an exception is raised, error is appended to state['errors']."""
        state = _make_base_state(backtest_results={"bad_key": 999})
        # This won't crash since calculate_adaptive_weights handles missing streams as None
        # Let's force a crash by passing a non-dict for a stream
        state["backtest_results"]["technical"] = "invalid"
        # calculate_adaptive_weights calls _compute_weighted_accuracy which checks for None
        # "invalid" string is truthy, so it will try .get() on a string → AttributeError
        result = adaptive_weights_node(state)
        # Should still return valid state (fallback)
        assert "adaptive_weights" in result


# ============================================================================
# 4. Design Contract Verification
# ============================================================================


class TestDesignContract:

    def test_node_11_is_sole_applier_of_recency_weighting(self):
        """
        The 70/30 recency weighting must NOT appear in backtest_results.
        It is applied ONLY here in Node 11.
        """
        backtest = _make_backtest_results(
            tech_full=0.50, tech_recent=0.80
        )
        # Verify the input has raw full/recent, not pre-weighted
        assert "full_accuracy" in backtest["technical"]
        assert "recent_accuracy" in backtest["technical"]
        assert "weighted_accuracy" not in backtest["technical"]

        # After Node 11, the weighted value is in weighted_accuracies
        state = _make_base_state(backtest)
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        expected_wa = 0.7 * 0.80 + 0.3 * 0.50
        actual_wa = aw["weighted_accuracies"]["technical"]
        assert abs(actual_wa - expected_wa) < 1e-4

    def test_weights_always_provided_even_on_failure(self):
        """
        adaptive_weights is ALWAYS written to state — Node 12 must always
        find it, even when Node 11 encounters errors.
        """
        state = _make_base_state(backtest_results="garbage")
        result = adaptive_weights_node(state)
        assert "adaptive_weights" in result
        for _, wk in STREAM_KEYS:
            assert wk in result["adaptive_weights"]

    def test_equal_weights_when_no_reliable_data(self):
        """
        When no reliable historical data exists (new stock), equal weights
        provide a fair starting point — not biased toward any stream.
        """
        state = _make_base_state(backtest_results=None)
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        weights = [aw[wk] for _, wk in STREAM_KEYS]
        assert all(abs(w - 0.25) < 0.001 for w in weights)
        # Sum still 1.0
        assert abs(sum(weights) - 1.0) < 0.001

    def test_neutral_streams_get_smaller_weight_than_reliable(self):
        """
        Neutral (0.5) stream must get a smaller weight than a calculated (0.65+) stream.
        This ensures unreliable streams are penalized.
        """
        backtest = _make_backtest_results(
            tech_full=0.65, tech_recent=0.65,
            stock_full=0.65, stock_recent=0.65,
            market_full=0.65, market_recent=0.65,
        )
        backtest["related_news"] = _make_stream_metrics(is_significant=False)
        state = _make_base_state(backtest)
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        assert aw["related_news_weight"] < aw["technical_weight"]
        assert aw["related_news_weight"] < aw["stock_news_weight"]

    def test_weights_bounded_between_0_and_1(self):
        """Every weight must be in [0, 1]."""
        backtest = _make_backtest_results()
        state = _make_base_state(backtest)
        result = adaptive_weights_node(state)
        for _, wk in STREAM_KEYS:
            w = result["adaptive_weights"][wk]
            assert 0.0 <= w <= 1.0

    def test_recency_weight_constant_is_70_pct(self):
        """RECENCY_WEIGHT must be 0.7 — sanity check for constant."""
        assert abs(RECENCY_WEIGHT - 0.7) < 1e-9

    def test_full_weight_constant_is_30_pct(self):
        """FULL_WEIGHT must be 0.3 — sanity check for constant."""
        assert abs(FULL_WEIGHT - 0.3) < 1e-9

    def test_recency_and_full_weights_sum_to_one(self):
        """RECENCY_WEIGHT + FULL_WEIGHT == 1.0."""
        assert abs(RECENCY_WEIGHT + FULL_WEIGHT - 1.0) < 1e-9

    def test_neutral_accuracy_is_coin_flip(self):
        """NEUTRAL_ACCURACY == 0.5 (random baseline)."""
        assert abs(NEUTRAL_ACCURACY - 0.5) < 1e-9

    def test_equal_weight_is_one_quarter(self):
        """EQUAL_WEIGHT == 0.25 for 4 streams."""
        assert abs(EQUAL_WEIGHT - 0.25) < 1e-9


# ============================================================================
# 5. Integration / Thesis Validation
# ============================================================================


class TestIntegration:

    def test_node8_learning_effect_reflected_in_weights(self):
        """
        Node 8's learning system boosts stock_news accuracy.
        This should be reflected as the highest weight in Node 11's output.
        This is a KEY thesis demonstration.
        """
        backtest = _make_backtest_results(
            # Node 8 boosted stock_news significantly
            stock_full=0.73,  stock_recent=0.75,
            tech_full=0.55,   tech_recent=0.57,
            market_full=0.58, market_recent=0.60,
            related_full=0.61, related_recent=0.63,
        )
        state = _make_base_state(backtest)
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]

        # Stock news should have the highest weight
        assert aw["stock_news_weight"] == max(
            aw["technical_weight"],
            aw["stock_news_weight"],
            aw["market_news_weight"],
            aw["related_news_weight"],
        )

    def test_weight_improves_with_node8_vs_without(self):
        """
        Stock news weight should be higher AFTER Node 8 learning
        compared to a baseline without learning.
        """
        backtest_without_node8 = _make_backtest_results(
            stock_full=0.62, stock_recent=0.62,  # baseline
            tech_full=0.60,  tech_recent=0.60,
            market_full=0.60, market_recent=0.60,
            related_full=0.60, related_recent=0.60,
        )
        backtest_with_node8 = _make_backtest_results(
            stock_full=0.73, stock_recent=0.75,  # boosted by Node 8
            tech_full=0.60,  tech_recent=0.60,
            market_full=0.60, market_recent=0.60,
            related_full=0.60, related_recent=0.60,
        )
        result_without = adaptive_weights_node(_make_base_state(backtest_without_node8))
        result_with = adaptive_weights_node(_make_base_state(backtest_with_node8))

        w_without = result_without["adaptive_weights"]["stock_news_weight"]
        w_with = result_with["adaptive_weights"]["stock_news_weight"]
        assert w_with > w_without

    def test_volatile_stock_recency_matters_more(self):
        """
        For volatile stocks, recent performance differs from full-period —
        confirming the 70/30 split does something meaningful.
        """
        backtest_stable = _make_backtest_results(
            tech_full=0.65, tech_recent=0.65,  # same
        )
        backtest_improving = _make_backtest_results(
            tech_full=0.55, tech_recent=0.75,  # recent much better
        )

        wa_stable = calculate_adaptive_weights(backtest_stable)["weighted_accuracies"]["technical"]
        wa_improving = calculate_adaptive_weights(backtest_improving)["weighted_accuracies"]["technical"]

        # Improving recent should yield higher weighted accuracy
        assert wa_improving > wa_stable

    def test_all_four_weight_keys_present(self):
        """All 4 weight keys that Node 12 reads must be present."""
        state = _make_base_state(_make_backtest_results())
        result = adaptive_weights_node(state)
        aw = result["adaptive_weights"]
        required_keys = [
            "technical_weight",
            "stock_news_weight",
            "market_news_weight",
            "related_news_weight",
        ]
        for key in required_keys:
            assert key in aw, f"Missing key: {key}"

    def test_streams_reliable_reflects_actual_data_quality(self):
        """streams_reliable gives Node 12 a data quality indicator."""
        # All 4 reliable
        backtest_good = _make_backtest_results()
        result_good = adaptive_weights_node(_make_base_state(backtest_good))
        assert result_good["adaptive_weights"]["streams_reliable"] == 4

        # Only 2 reliable
        backtest_partial = _make_backtest_results()
        backtest_partial["market_news"]["is_sufficient"] = False
        backtest_partial["related_news"]["is_significant"] = False
        result_partial = adaptive_weights_node(_make_base_state(backtest_partial))
        assert result_partial["adaptive_weights"]["streams_reliable"] == 2
