"""
Tests for Node 12: Final Signal Generation

Covers:
- _score_technical:    continuous scoring, HOLD=0, missing fields
- _score_sentiment:    Node 8 override, fallback, missing
- _score_market:       direction × correlation, missing context
- _score_monte_carlo:  (prob_up - 0.5) * 2 mapping, missing
- signal_generation_node (main):
    strong BUY / SELL agreement, HOLD near-zero, dominant weight effect,
    missing individual streams, all streams missing, adaptive_weights=None,
    final_confidence clamped to [0, 1], stream_scores sum, signal_agreement
- risk_summary:  CRITICAL risk noted but signal still generated
- Design contracts:
    continuous scoring (not binary), weights used as-is, Node 8 boost effect,
    hold_threshold_pct and streams_reliable passed through unchanged
- Integration:  full realistic state with all nodes populated
"""

import pytest
from copy import deepcopy

from src.langgraph_nodes.node_12_signal_generation import (
    BUY_THRESHOLD,
    EQUAL_WEIGHT,
    SELL_THRESHOLD,
    SIGNAL_TO_SCORE,
    _classify_signal,
    _count_signal_agreement,
    _build_risk_summary,
    _build_price_targets,
    _score_technical,
    _score_sentiment,
    _score_market,
    _score_monte_carlo,
    signal_generation_node,
)


# ============================================================================
# SHARED FIXTURES & HELPERS
# ============================================================================


def _make_technical(signal: str = "BUY", confidence: float = 0.72) -> dict:
    return {"technical_signal": signal, "technical_confidence": confidence}


def _make_sentiment(
    aggregated: float = 0.61,
    signal: str = "POSITIVE",
    confidence: float = 0.80,
) -> dict:
    return {
        "aggregated_sentiment": aggregated,
        "sentiment_signal": signal,
        "sentiment_confidence": confidence,
        "sentiment_analysis": None,
    }


def _make_market_context(
    context_signal: str = "BUY",
    correlation: float = 0.82,
    market_trend: str = "BULLISH",
    sector_trend: str = "BULLISH",
) -> dict:
    return {
        "market_context": {
            "context_signal": context_signal,
            "market_trend": market_trend,
            "sector_trend": sector_trend,
            "market_correlation": correlation,
            "sector_performance": 1.5,
        }
    }


def _make_monte_carlo(probability_up: float = 0.68) -> dict:
    return {
        "monte_carlo_results": {
            "probability_up": probability_up,
            "mean_forecast": 195.0,
            "current_price": 185.0,
            "expected_return": 5.4,
            "confidence_95": {"lower": 172.0, "upper": 218.0},
        },
        "forecasted_price": 195.0,
        "price_range": (172.0, 218.0),
    }


def _make_adaptive_weights(
    tech: float = 0.258,
    stock: float = 0.297,
    market: float = 0.244,
    related: float = 0.201,
    hold_threshold_pct: float = 1.4,
    streams_reliable: int = 4,
    fallback: bool = False,
) -> dict:
    return {
        "adaptive_weights": {
            "technical_weight":    tech,
            "stock_news_weight":   stock,
            "market_news_weight":  market,
            "related_news_weight": related,
            "weighted_accuracies": {
                "technical": tech, "stock_news": stock,
                "market_news": market, "related_news": related,
            },
            "weight_sources": {
                "technical": "calculated", "stock_news": "calculated",
                "market_news": "calculated", "related_news": "calculated",
            },
            "streams_reliable": streams_reliable,
            "hold_threshold_pct": hold_threshold_pct,
            "sample_period_days": 180,
            "fallback_equal_weights": fallback,
        }
    }


def _make_behavioral(
    risk_level: str = "LOW",
    pump_score: int = 10,
    overall_risk: float = 12.0,
) -> dict:
    return {
        "behavioral_anomaly_detection": {
            "risk_level": risk_level,
            "pump_and_dump_score": pump_score,
            "overall_risk_score": overall_risk,
        }
    }


def _make_content(early_risk: str = "LOW") -> dict:
    return {
        "content_analysis_summary": {
            "early_risk_level": early_risk,
        }
    }


def _make_full_state(**overrides) -> dict:
    """
    Build a complete realistic state with all 4 streams populated.
    Keyword arguments override individual top-level state keys.
    """
    state = {
        "ticker": "AAPL",
        "errors": [],
        "node_execution_times": {},
    }
    state.update(_make_technical())
    state.update(_make_sentiment())
    state.update(_make_market_context())
    state.update(_make_monte_carlo())
    state.update(_make_adaptive_weights())
    state.update(_make_behavioral())
    state.update(_make_content())
    state.update(overrides)
    return state


# ============================================================================
# 1. _score_technical
# ============================================================================


class TestScoreTechnical:

    def test_buy_signal_positive_score(self):
        """BUY with confidence=0.72 → +0.72."""
        s = _make_technical(signal="BUY", confidence=0.72)
        score, missing = _score_technical(s)
        assert abs(score - 0.72) < 1e-9
        assert missing is False

    def test_sell_signal_negative_score(self):
        """SELL with confidence=0.61 → -0.61."""
        s = _make_technical(signal="SELL", confidence=0.61)
        score, missing = _score_technical(s)
        assert abs(score - (-0.61)) < 1e-9
        assert missing is False

    def test_hold_signal_zero_score(self):
        """HOLD regardless of confidence → 0.0 (direction is neutral)."""
        s = _make_technical(signal="HOLD", confidence=0.55)
        score, missing = _score_technical(s)
        assert score == 0.0
        assert missing is False

    def test_missing_signal_returns_missing(self):
        """technical_signal=None → (0.0, True)."""
        s = {"technical_signal": None, "technical_confidence": 0.70}
        score, missing = _score_technical(s)
        assert score == 0.0
        assert missing is True

    def test_missing_confidence_returns_missing(self):
        """technical_confidence=None → (0.0, True)."""
        s = {"technical_signal": "BUY", "technical_confidence": None}
        score, missing = _score_technical(s)
        assert score == 0.0
        assert missing is True

    def test_both_fields_absent_returns_missing(self):
        """Empty state → (0.0, True)."""
        score, missing = _score_technical({})
        assert score == 0.0
        assert missing is True

    def test_score_is_product_of_confidence_and_sign(self):
        """Verify exact arithmetic: score = sign * confidence."""
        s = _make_technical(signal="SELL", confidence=0.85)
        score, _ = _score_technical(s)
        assert abs(score - (-0.85)) < 1e-9


# ============================================================================
# 2. _score_sentiment
# ============================================================================


class TestScoreSentiment:

    def test_positive_aggregated_sentiment(self):
        """aggregated_sentiment=+0.61 → +0.61 (no conversion needed)."""
        s = _make_sentiment(aggregated=0.61)
        score, missing = _score_sentiment(s)
        assert abs(score - 0.61) < 1e-9
        assert missing is False

    def test_negative_aggregated_sentiment(self):
        """aggregated_sentiment=-0.45 → -0.45."""
        s = _make_sentiment(aggregated=-0.45)
        score, missing = _score_sentiment(s)
        assert abs(score - (-0.45)) < 1e-9
        assert missing is False

    def test_node8_override_takes_priority(self):
        """sentiment_analysis dict with aggregated_sentiment overrides Node 5 value."""
        s = _make_sentiment(aggregated=0.10)   # Node 5 value
        s["sentiment_analysis"] = {"aggregated_sentiment": 0.75}   # Node 8 override
        score, missing = _score_sentiment(s)
        assert abs(score - 0.75) < 1e-9
        assert missing is False

    def test_node8_none_falls_back_to_aggregated(self):
        """sentiment_analysis=None → falls back to state['aggregated_sentiment']."""
        s = _make_sentiment(aggregated=0.30)
        s["sentiment_analysis"] = None
        score, missing = _score_sentiment(s)
        assert abs(score - 0.30) < 1e-9
        assert missing is False

    def test_node8_missing_key_falls_back(self):
        """sentiment_analysis exists but lacks aggregated_sentiment → fallback."""
        s = _make_sentiment(aggregated=0.42)
        s["sentiment_analysis"] = {"some_other_key": 0.99}
        score, missing = _score_sentiment(s)
        assert abs(score - 0.42) < 1e-9
        assert missing is False

    def test_both_none_returns_missing(self):
        """No aggregated_sentiment anywhere → (0.0, True)."""
        s = {"aggregated_sentiment": None, "sentiment_analysis": None}
        score, missing = _score_sentiment(s)
        assert score == 0.0
        assert missing is True

    def test_empty_state_returns_missing(self):
        score, missing = _score_sentiment({})
        assert score == 0.0
        assert missing is True


# ============================================================================
# 3. _score_market
# ============================================================================


class TestScoreMarket:

    def test_buy_context_scaled_by_correlation(self):
        """BUY context + correlation=0.82 → +0.82."""
        s = _make_market_context(context_signal="BUY", correlation=0.82)
        score, missing = _score_market(s)
        assert abs(score - 0.82) < 1e-9
        assert missing is False

    def test_hold_context_zero_score(self):
        """HOLD context → 0.0 regardless of correlation."""
        s = _make_market_context(context_signal="HOLD", correlation=0.82)
        score, missing = _score_market(s)
        assert score == 0.0
        assert missing is False

    def test_sell_context_negative_score(self):
        """SELL context + correlation=0.60 → -0.60."""
        s = _make_market_context(context_signal="SELL", correlation=0.60)
        score, missing = _score_market(s)
        assert abs(score - (-0.60)) < 1e-9
        assert missing is False

    def test_market_context_none_returns_missing(self):
        """market_context=None → (0.0, True)."""
        score, missing = _score_market({"market_context": None})
        assert score == 0.0
        assert missing is True

    def test_missing_correlation_defaults_to_one(self):
        """market_correlation absent → treated as 1.0 (full correlation)."""
        s = {"market_context": {"context_signal": "BUY"}}
        score, missing = _score_market(s)
        assert abs(score - 1.0) < 1e-9
        assert missing is False

    def test_correlation_none_defaults_to_one(self):
        """market_correlation=None → treated as 1.0."""
        s = {"market_context": {"context_signal": "SELL", "market_correlation": None}}
        score, missing = _score_market(s)
        assert abs(score - (-1.0)) < 1e-9
        assert missing is False

    def test_score_is_product_of_direction_and_correlation(self):
        """Verify: score = context_direction × correlation."""
        s = _make_market_context(context_signal="BUY", correlation=0.55)
        score, _ = _score_market(s)
        assert abs(score - 0.55) < 1e-9


# ============================================================================
# 4. _score_monte_carlo
# ============================================================================


class TestScoreMonteCarlo:

    def test_probability_above_half_positive_score(self):
        """probability_up=0.68 → (0.68-0.5)*2 = +0.36."""
        s = _make_monte_carlo(probability_up=0.68)
        score, missing = _score_monte_carlo(s)
        assert abs(score - 0.36) < 1e-9
        assert missing is False

    def test_probability_exactly_half_zero_score(self):
        """probability_up=0.50 → 0.0 (coin-flip)."""
        s = _make_monte_carlo(probability_up=0.50)
        score, missing = _score_monte_carlo(s)
        assert abs(score) < 1e-9
        assert missing is False

    def test_probability_below_half_negative_score(self):
        """probability_up=0.25 → (0.25-0.5)*2 = -0.50."""
        s = _make_monte_carlo(probability_up=0.25)
        score, missing = _score_monte_carlo(s)
        assert abs(score - (-0.50)) < 1e-9
        assert missing is False

    def test_probability_one_gives_plus_one(self):
        """probability_up=1.0 → +1.0 (certain upside)."""
        s = _make_monte_carlo(probability_up=1.0)
        score, missing = _score_monte_carlo(s)
        assert abs(score - 1.0) < 1e-9
        assert missing is False

    def test_probability_zero_gives_minus_one(self):
        """probability_up=0.0 → -1.0 (certain downside)."""
        s = _make_monte_carlo(probability_up=0.0)
        score, missing = _score_monte_carlo(s)
        assert abs(score - (-1.0)) < 1e-9
        assert missing is False

    def test_monte_carlo_results_none_returns_missing(self):
        """monte_carlo_results=None → (0.0, True)."""
        score, missing = _score_monte_carlo({"monte_carlo_results": None})
        assert score == 0.0
        assert missing is True

    def test_probability_up_missing_returns_missing(self):
        """probability_up key absent → (0.0, True)."""
        s = {"monte_carlo_results": {"mean_forecast": 200.0}}
        score, missing = _score_monte_carlo(s)
        assert score == 0.0
        assert missing is True


# ============================================================================
# 5. signal_generation_node (main integration tests)
# ============================================================================


class TestSignalGenerationNode:

    def test_all_streams_agree_buy(self):
        """
        All 4 streams pointing strongly BUY → final_signal='BUY', high confidence.
        Technical BUY 0.90, sentiment +0.80, market BUY corr=0.90, MC prob=0.75.
        """
        state = _make_full_state(
            technical_signal="BUY", technical_confidence=0.90,
            aggregated_sentiment=0.80,
        )
        state["market_context"]["context_signal"] = "BUY"
        state["market_context"]["market_correlation"] = 0.90
        state["monte_carlo_results"]["probability_up"] = 0.75
        result = signal_generation_node(state)

        assert result["final_signal"] == "BUY"
        assert result["final_confidence"] > 0.0
        sc = result["signal_components"]
        assert sc["final_signal"] == "BUY"
        assert sc["signal_agreement"] == 4

    def test_all_streams_agree_sell(self):
        """All 4 streams pointing strongly SELL → final_signal='SELL'."""
        state = _make_full_state(
            technical_signal="SELL", technical_confidence=0.85,
            aggregated_sentiment=-0.75,
        )
        state["market_context"]["context_signal"] = "SELL"
        state["market_context"]["market_correlation"] = 0.85
        state["monte_carlo_results"]["probability_up"] = 0.20
        result = signal_generation_node(state)

        assert result["final_signal"] == "SELL"
        assert result["final_confidence"] > 0.0
        assert result["signal_components"]["signal_agreement"] == 4

    def test_balanced_streams_produce_hold(self):
        """
        Neutral/balanced signals → final_score near 0 → HOLD.
        Technical HOLD, sentiment≈0, market HOLD, MC prob=0.50.
        """
        state = _make_full_state(
            technical_signal="HOLD", technical_confidence=0.60,
            aggregated_sentiment=0.00,
        )
        state["market_context"]["context_signal"] = "HOLD"
        state["monte_carlo_results"]["probability_up"] = 0.50
        result = signal_generation_node(state)

        assert result["final_signal"] == "HOLD"
        # confidence should be very low
        assert result["final_confidence"] < BUY_THRESHOLD

    def test_dominant_stock_news_weight_drives_buy(self):
        """
        Very high stock_news_weight (0.70) + strong positive sentiment → BUY
        even when other streams are neutral.
        """
        state = _make_full_state(
            technical_signal="HOLD", technical_confidence=0.50,
            aggregated_sentiment=0.90,
        )
        state["market_context"]["context_signal"] = "HOLD"
        state["monte_carlo_results"]["probability_up"] = 0.50
        # Set extreme stock_news_weight
        state["adaptive_weights"]["stock_news_weight"]   = 0.70
        state["adaptive_weights"]["technical_weight"]    = 0.10
        state["adaptive_weights"]["market_news_weight"]  = 0.10
        state["adaptive_weights"]["related_news_weight"] = 0.10

        result = signal_generation_node(state)
        # 0.70 * 0.90 = 0.63 >> BUY_THRESHOLD → BUY
        assert result["final_signal"] == "BUY"

    def test_final_confidence_at_most_one(self):
        """final_confidence is always in [0.0, 1.0]."""
        state = _make_full_state(
            technical_signal="BUY", technical_confidence=1.0,
            aggregated_sentiment=1.0,
        )
        state["market_context"]["context_signal"] = "BUY"
        state["market_context"]["market_correlation"] = 1.0
        state["monte_carlo_results"]["probability_up"] = 1.0
        result = signal_generation_node(state)

        assert 0.0 <= result["final_confidence"] <= 1.0

    def test_final_confidence_at_least_zero(self):
        """final_confidence is never negative."""
        state = _make_full_state(
            technical_signal="SELL", technical_confidence=0.95,
            aggregated_sentiment=-0.95,
        )
        result = signal_generation_node(state)
        assert result["final_confidence"] >= 0.0

    def test_stream_contributions_sum_to_final_score(self):
        """sum(contributions) == final_score within floating-point tolerance."""
        state = _make_full_state()
        result = signal_generation_node(state)

        sc = result["signal_components"]
        total = sum(d["contribution"] for d in sc["stream_scores"].values())
        assert abs(total - sc["final_score"]) < 1e-9

    def test_signal_agreement_count_correct_buy(self):
        """For a BUY result, agreement = number of streams with raw_score > 0."""
        state = _make_full_state(
            technical_signal="BUY", technical_confidence=0.80,
            aggregated_sentiment=0.70,
        )
        state["market_context"]["context_signal"] = "BUY"
        state["market_context"]["market_correlation"] = 0.80
        state["monte_carlo_results"]["probability_up"] = 0.70
        result = signal_generation_node(state)

        sc = result["signal_components"]
        manual_count = sum(
            1 for d in sc["stream_scores"].values() if d["raw_score"] > 0
        )
        assert sc["signal_agreement"] == manual_count

    def test_critical_behavioral_risk_noted_but_signal_still_generated(self):
        """
        CRITICAL behavioral anomaly should be noted in risk_summary
        but Node 12 still produces a signal (blocking lives in workflow edges).
        """
        state = _make_full_state()
        state.update(_make_behavioral(risk_level="CRITICAL", pump_score=90))

        result = signal_generation_node(state)

        assert result["final_signal"] in ("BUY", "SELL", "HOLD")
        rs = result["signal_components"]["risk_summary"]
        assert rs["overall_risk_level"] == "CRITICAL"
        assert rs["trading_safe"] is False

    def test_node_writes_execution_time(self):
        """node_12 execution time must be recorded in node_execution_times."""
        state = _make_full_state()
        result = signal_generation_node(state)
        assert "node_12" in result["node_execution_times"]
        assert result["node_execution_times"]["node_12"] >= 0.0

    def test_all_required_keys_in_signal_components(self):
        """signal_components contains all keys required by Nodes 13/14."""
        state = _make_full_state()
        result = signal_generation_node(state)
        sc = result["signal_components"]

        required_top = {
            "final_score", "final_signal", "final_confidence",
            "stream_scores", "risk_summary", "price_targets",
            "backtest_context", "signal_agreement", "streams_missing",
        }
        assert required_top.issubset(sc.keys())

        required_stream_keys = {"raw_score", "weight", "contribution"}
        for stream_name in ("technical", "sentiment", "market", "monte_carlo"):
            assert stream_name in sc["stream_scores"]
            assert required_stream_keys.issubset(sc["stream_scores"][stream_name].keys())

        required_risk = {
            "overall_risk_level", "pump_and_dump_score",
            "behavioral_risk", "content_risk", "trading_safe",
        }
        assert required_risk.issubset(sc["risk_summary"].keys())

        required_price = {
            "current_price", "forecasted_price",
            "price_range_lower", "price_range_upper", "expected_return_pct",
        }
        assert required_price.issubset(sc["price_targets"].keys())

        required_backtest = {
            "hold_threshold_pct", "streams_reliable", "weights_are_fallback",
        }
        assert required_backtest.issubset(sc["backtest_context"].keys())


# ============================================================================
# 6. Error handling and missing data
# ============================================================================


class TestErrorHandling:

    def test_missing_technical_stream_graceful(self):
        """technical=None → score 0.0, 'technical' in streams_missing, node runs."""
        state = _make_full_state(
            technical_signal=None,
            technical_confidence=None,
        )
        result = signal_generation_node(state)

        assert result["final_signal"] in ("BUY", "SELL", "HOLD")
        sc = result["signal_components"]
        assert "technical" in sc["streams_missing"]
        assert sc["stream_scores"]["technical"]["raw_score"] == 0.0

    def test_missing_sentiment_stream_graceful(self):
        """aggregated_sentiment=None, sentiment_analysis=None → sentiment missing."""
        state = _make_full_state(
            aggregated_sentiment=None,
            sentiment_analysis=None,
        )
        result = signal_generation_node(state)

        assert result["final_signal"] in ("BUY", "SELL", "HOLD")
        sc = result["signal_components"]
        assert "sentiment" in sc["streams_missing"]

    def test_missing_market_stream_graceful(self):
        """market_context=None → market missing."""
        state = _make_full_state()
        state["market_context"] = None
        result = signal_generation_node(state)

        assert result["final_signal"] in ("BUY", "SELL", "HOLD")
        sc = result["signal_components"]
        assert "market" in sc["streams_missing"]

    def test_missing_monte_carlo_stream_graceful(self):
        """monte_carlo_results=None → monte_carlo missing."""
        state = _make_full_state()
        state["monte_carlo_results"] = None
        result = signal_generation_node(state)

        assert result["final_signal"] in ("BUY", "SELL", "HOLD")
        sc = result["signal_components"]
        assert "monte_carlo" in sc["streams_missing"]

    def test_all_streams_missing_returns_hold_zero_confidence(self):
        """
        All 4 streams None → final_signal='HOLD', final_confidence=0.0,
        error appended to state['errors'].
        """
        state = {
            "ticker": "AAPL",
            "errors": [],
            "node_execution_times": {},
            "technical_signal": None,
            "technical_confidence": None,
            "aggregated_sentiment": None,
            "sentiment_analysis": None,
            "market_context": None,
            "monte_carlo_results": None,
            "adaptive_weights": None,
        }
        result = signal_generation_node(state)

        assert result["final_signal"] == "HOLD"
        assert result["final_confidence"] == 0.0
        assert any("All upstream data" in e for e in result["errors"])

    def test_adaptive_weights_none_uses_equal_weights(self):
        """
        adaptive_weights=None → silently falls back to equal weights (0.25 each),
        node still produces a valid signal.
        """
        state = _make_full_state(adaptive_weights=None)
        result = signal_generation_node(state)

        assert result["final_signal"] in ("BUY", "SELL", "HOLD")
        sc = result["signal_components"]
        # Every stream weight should equal EQUAL_WEIGHT when fallback is used
        for data in sc["stream_scores"].values():
            assert abs(data["weight"] - EQUAL_WEIGHT) < 1e-9

    def test_missing_behavioral_anomaly_doesnt_crash(self):
        """behavioral_anomaly_detection=None → risk_summary uses UNKNOWN defaults."""
        state = _make_full_state(behavioral_anomaly_detection=None)
        result = signal_generation_node(state)

        rs = result["signal_components"]["risk_summary"]
        assert rs["overall_risk_level"] == "UNKNOWN"

    def test_missing_content_analysis_doesnt_crash(self):
        """content_analysis_summary=None → content_risk='UNKNOWN'."""
        state = _make_full_state(content_analysis_summary=None)
        result = signal_generation_node(state)

        rs = result["signal_components"]["risk_summary"]
        assert rs["content_risk"] == "UNKNOWN"


# ============================================================================
# 7. Design contract verification
# ============================================================================


class TestDesignContract:

    def test_continuous_scoring_not_binary(self):
        """
        DESIGN CONTRACT: Scores are NOT binary (0 or 1).
        BUY with confidence=0.72 must score +0.72, not +1.0 or 100.
        """
        state = _make_full_state(
            technical_signal="BUY", technical_confidence=0.72,
        )
        result = signal_generation_node(state)
        sc = result["signal_components"]
        tech_raw = sc["stream_scores"]["technical"]["raw_score"]
        assert abs(tech_raw - 0.72) < 1e-9, (
            f"Expected +0.72 (continuous), got {tech_raw} (binary scoring detected)"
        )

    def test_weights_used_exactly_as_provided(self):
        """
        DESIGN CONTRACT: Weights from adaptive_weights are used as-is,
        without re-normalization or modification.
        """
        # Deliberately unequal but already-normalized weights
        state = _make_full_state()
        state["adaptive_weights"]["technical_weight"]    = 0.40
        state["adaptive_weights"]["stock_news_weight"]   = 0.30
        state["adaptive_weights"]["market_news_weight"]  = 0.20
        state["adaptive_weights"]["related_news_weight"] = 0.10

        result = signal_generation_node(state)
        sc = result["signal_components"]
        assert abs(sc["stream_scores"]["technical"]["weight"]   - 0.40) < 1e-9
        assert abs(sc["stream_scores"]["sentiment"]["weight"]   - 0.30) < 1e-9
        assert abs(sc["stream_scores"]["market"]["weight"]      - 0.20) < 1e-9
        assert abs(sc["stream_scores"]["monte_carlo"]["weight"] - 0.10) < 1e-9

    def test_node8_boost_effect_on_final_score(self):
        """
        DESIGN CONTRACT: Higher stock_news_weight gives sentiment more influence.
        Same 4 stream values, doubled stock_news_weight → sentiment contribution doubles.
        """
        base_state = _make_full_state(aggregated_sentiment=0.80)
        # Run with low stock_news_weight
        low_state = deepcopy(base_state)
        low_state["adaptive_weights"]["stock_news_weight"]   = 0.10
        low_state["adaptive_weights"]["technical_weight"]    = 0.40
        low_state["adaptive_weights"]["market_news_weight"]  = 0.30
        low_state["adaptive_weights"]["related_news_weight"] = 0.20
        low_result = signal_generation_node(low_state)
        low_sent_contrib = low_result["signal_components"]["stream_scores"]["sentiment"]["contribution"]

        # Run with high stock_news_weight
        high_state = deepcopy(base_state)
        high_state["adaptive_weights"]["stock_news_weight"]  = 0.60
        high_state["adaptive_weights"]["technical_weight"]   = 0.20
        high_state["adaptive_weights"]["market_news_weight"] = 0.10
        high_state["adaptive_weights"]["related_news_weight"]= 0.10
        high_result = signal_generation_node(high_state)
        high_sent_contrib = high_result["signal_components"]["stream_scores"]["sentiment"]["contribution"]

        assert high_sent_contrib > low_sent_contrib, (
            "Higher stock_news_weight must produce larger sentiment contribution"
        )

    def test_hold_threshold_pct_passed_through(self):
        """
        DESIGN CONTRACT: hold_threshold_pct from adaptive_weights is relayed
        unchanged into signal_components['backtest_context'].
        """
        state = _make_full_state()
        state["adaptive_weights"]["hold_threshold_pct"] = 3.7
        result = signal_generation_node(state)
        bc = result["signal_components"]["backtest_context"]
        assert abs(bc["hold_threshold_pct"] - 3.7) < 1e-9

    def test_streams_reliable_passed_through(self):
        """
        DESIGN CONTRACT: streams_reliable count is relayed unchanged.
        """
        state = _make_full_state()
        state["adaptive_weights"]["streams_reliable"] = 3
        result = signal_generation_node(state)
        bc = result["signal_components"]["backtest_context"]
        assert bc["streams_reliable"] == 3

    def test_fallback_flag_passed_through(self):
        """
        DESIGN CONTRACT: fallback_equal_weights flag is relayed unchanged.
        """
        state = _make_full_state()
        state["adaptive_weights"]["fallback_equal_weights"] = True
        result = signal_generation_node(state)
        bc = result["signal_components"]["backtest_context"]
        assert bc["weights_are_fallback"] is True

    def test_hold_signal_confidence_is_zero_score_abs(self):
        """
        DESIGN CONTRACT: final_confidence = abs(final_score), not 1 - |score|
        or some other formula.
        """
        state = _make_full_state(
            technical_signal="HOLD", technical_confidence=0.55,
            aggregated_sentiment=0.0,
        )
        state["market_context"]["context_signal"] = "HOLD"
        state["monte_carlo_results"]["probability_up"] = 0.50
        result = signal_generation_node(state)

        sc = result["signal_components"]
        assert abs(result["final_confidence"] - abs(sc["final_score"])) < 1e-9


# ============================================================================
# 8. Integration — full realistic state
# ============================================================================


class TestIntegration:

    def test_full_realistic_buy_scenario(self):
        """
        Full state with all nodes populated, bullish across all streams.
        Verifies the node produces a coherent BUY recommendation end-to-end.
        """
        state = _make_full_state(
            technical_signal="BUY", technical_confidence=0.75,
            aggregated_sentiment=0.65,
        )
        state["market_context"]["context_signal"] = "BUY"
        state["market_context"]["market_correlation"] = 0.78
        state["monte_carlo_results"]["probability_up"] = 0.72

        result = signal_generation_node(state)

        assert result["final_signal"] == "BUY"
        assert 0.0 < result["final_confidence"] <= 1.0
        assert result["signal_components"] is not None
        assert result["final_signal"] == result["signal_components"]["final_signal"]
        assert result["final_confidence"] == result["signal_components"]["final_confidence"]

    def test_full_realistic_sell_scenario(self):
        """Bearish across all streams → SELL."""
        state = _make_full_state(
            technical_signal="SELL", technical_confidence=0.80,
            aggregated_sentiment=-0.70,
        )
        state["market_context"]["context_signal"] = "SELL"
        state["market_context"]["market_correlation"] = 0.85
        state["monte_carlo_results"]["probability_up"] = 0.22

        result = signal_generation_node(state)
        assert result["final_signal"] == "SELL"

    def test_streams_missing_list_empty_when_all_present(self):
        """When all 4 streams have data, streams_missing should be empty."""
        state = _make_full_state()
        result = signal_generation_node(state)
        assert result["signal_components"]["streams_missing"] == []

    def test_price_targets_populated_from_monte_carlo(self):
        """price_targets in signal_components reflects the Monte Carlo output."""
        state = _make_full_state()
        result = signal_generation_node(state)
        pt = result["signal_components"]["price_targets"]
        assert pt["forecasted_price"] == state["forecasted_price"]
        assert pt["price_range_lower"] == state["price_range"][0]
        assert pt["price_range_upper"] == state["price_range"][1]

    def test_signal_components_final_signal_matches_state(self):
        """signal_components['final_signal'] must equal state['final_signal']."""
        state = _make_full_state()
        result = signal_generation_node(state)
        assert result["signal_components"]["final_signal"] == result["final_signal"]
        assert result["signal_components"]["final_confidence"] == result["final_confidence"]
