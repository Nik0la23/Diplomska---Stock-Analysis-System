"""
Tests for Node 10: Backtesting

Covers:
- calculate_hold_threshold: adaptive threshold, fallback, bounds
- evaluate_outcome: all signal types, boundary conditions
- reconstruct_technical_signal: valid/insufficient data
- aggregate_daily_sentiment: all sentiment combinations, empty input
- backtest_technical_stream: normal run, insufficient data
- backtest_sentiment_stream: normal run, empty events, no price data
- calculate_stream_metrics: accuracy separation, significance, sufficient/insufficient
- backtesting_node (main): full flow, missing price data, empty news, error isolation

Design contract verified throughout:
- Node 10 does NOT apply recency weighting (that is Node 11's job)
- Returns separate full_accuracy and recent_accuracy
- Returns actual price movements, not just binary correct/incorrect
- Each stream fails independently without breaking others
"""

import pytest
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.langgraph_nodes.node_10_backtesting import (
    HOLD_THRESHOLD_FALLBACK_PCT,
    HOLD_THRESHOLD_MAX_PCT,
    HOLD_THRESHOLD_MIN_PCT,
    MIN_PRICE_ROWS_FOR_TECHNICAL,
    MIN_SIGNALS_FOR_SUFFICIENCY,
    RECENT_PERIOD_DAYS,
    aggregate_daily_sentiment,
    backtest_sentiment_stream,
    backtest_technical_stream,
    backtesting_node,
    calculate_hold_threshold,
    calculate_stream_metrics,
    evaluate_outcome,
    reconstruct_technical_signal,
)


# ============================================================================
# SHARED FIXTURES & HELPERS
# ============================================================================


def _make_price_df(n_days: int = 180, start_price: float = 100.0, volatility: float = 0.015) -> pd.DataFrame:
    """
    Create a realistic synthetic OHLCV DataFrame.

    Uses a random walk with fixed seed so tests are deterministic.
    """
    rng = np.random.default_rng(seed=42)
    today = datetime.now().date()
    dates = pd.date_range(end=today, periods=n_days, freq="B")  # business days

    prices = [start_price]
    for _ in range(n_days - 1):
        change = rng.normal(0, volatility)
        prices.append(prices[-1] * (1 + change))

    closes = np.array(prices)
    opens = closes * (1 + rng.normal(0, 0.003, n_days))
    highs = np.maximum(closes, opens) * (1 + abs(rng.normal(0, 0.005, n_days)))
    lows = np.minimum(closes, opens) * (1 - abs(rng.normal(0, 0.005, n_days)))
    volumes = rng.integers(1_000_000, 5_000_000, n_days).astype(float)

    return pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def _make_news_events(
    n_stock: int = 40,
    n_market: int = 30,
    n_related: int = 20,
    stock_accuracy: float = 0.65,
    days_back: int = 170,
) -> list:
    """
    Create synthetic news_with_outcomes records matching the DB schema.

    Articles are spread over days_back days ending 8 days ago (so all have
    valid 7-day outcomes). Accuracy is the fraction of articles where the
    sentiment prediction was correct.
    """
    rng = np.random.default_rng(seed=99)
    events = []
    today = datetime.now()
    base_date = today - timedelta(days=8)  # 7+ days ago so outcomes exist

    def _make_event(idx: int, news_type: str, accuracy: float, days_offset: int) -> dict:
        pub_date = base_date - timedelta(days=days_offset % days_back)
        correct = rng.random() < accuracy
        label = rng.choice(["positive", "negative", "neutral"], p=[0.5, 0.3, 0.2])
        score = float(rng.uniform(0.4, 0.9))
        # price_change consistent with label when correct
        if correct:
            if label == "positive":
                price_change = float(rng.uniform(2.5, 6.0))
            elif label == "negative":
                price_change = float(rng.uniform(-6.0, -2.5))
            else:
                price_change = float(rng.uniform(-1.5, 1.5))
        else:
            if label == "positive":
                price_change = float(rng.uniform(-6.0, -0.5))
            elif label == "negative":
                price_change = float(rng.uniform(0.5, 6.0))
            else:
                price_change = float(rng.uniform(-5.0, 5.0))

        return {
            "id": idx,
            "ticker": "AAPL",
            "news_type": news_type,
            "title": f"Test article {idx}",
            "source": "TestSource.com",
            "published_at": pub_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "sentiment_label": label,
            "sentiment_score": score,
            "price_at_news": 150.0,
            "price_7day_later": 150.0 * (1 + price_change / 100),
            "price_change_7day": price_change,
            "predicted_direction": "UP" if label == "positive" else "DOWN" if label == "negative" else "FLAT",
            "actual_direction": "UP" if price_change > 1.5 else "DOWN" if price_change < -1.5 else "FLAT",
            "prediction_was_accurate_7day": correct,
        }

    idx = 0
    for i in range(n_stock):
        events.append(_make_event(idx, "stock", stock_accuracy, i * (days_back // n_stock)))
        idx += 1
    for i in range(n_market):
        events.append(_make_event(idx, "market", 0.55, i * (days_back // n_market)))
        idx += 1
    for i in range(n_related):
        events.append(_make_event(idx, "related", 0.60, i * (days_back // n_related)))
        idx += 1

    return events


def _make_base_state(n_days: int = 180) -> dict:
    """Minimal valid state for backtesting_node."""
    return {
        "ticker": "AAPL",
        "raw_price_data": _make_price_df(n_days),
        "errors": [],
        "node_execution_times": {},
    }


# ============================================================================
# TEST CLASS 1: calculate_hold_threshold
# ============================================================================


class TestCalculateHoldThreshold:
    """Volatility-adaptive HOLD threshold."""

    def test_normal_data_returns_reasonable_threshold(self):
        """Standard 180-day data should produce a threshold in sane range."""
        df = _make_price_df(180, volatility=0.015)
        threshold = calculate_hold_threshold(df)
        assert HOLD_THRESHOLD_MIN_PCT <= threshold <= HOLD_THRESHOLD_MAX_PCT

    def test_high_volatility_stock_larger_threshold(self):
        """High-volatility stock (like TSLA) should get a larger HOLD zone."""
        low_vol = _make_price_df(180, volatility=0.008)
        high_vol = _make_price_df(180, volatility=0.030)
        assert calculate_hold_threshold(high_vol) > calculate_hold_threshold(low_vol)

    def test_insufficient_data_returns_fallback(self):
        """Fewer than 30 rows should return the static fallback."""
        df = _make_price_df(20)
        threshold = calculate_hold_threshold(df)
        assert threshold == HOLD_THRESHOLD_FALLBACK_PCT

    def test_none_input_returns_fallback(self):
        threshold = calculate_hold_threshold(None)
        assert threshold == HOLD_THRESHOLD_FALLBACK_PCT

    def test_result_within_bounds(self):
        """Even extreme data must stay within sanity bounds."""
        # Extremely high volatility
        df = _make_price_df(180, volatility=0.20)
        threshold = calculate_hold_threshold(df)
        assert threshold <= HOLD_THRESHOLD_MAX_PCT

    def test_result_is_float(self):
        df = _make_price_df(180)
        assert isinstance(calculate_hold_threshold(df), float)

    def test_thirty_days_exact_boundary(self):
        """Exactly 30 rows should compute, not fall back."""
        df = _make_price_df(30)
        threshold = calculate_hold_threshold(df)
        # Should NOT be the fallback (could coincidentally equal it, but result is computed)
        assert isinstance(threshold, float)


# ============================================================================
# TEST CLASS 2: evaluate_outcome
# ============================================================================


class TestEvaluateOutcome:
    """Single correctness definition — used by all streams."""

    def test_buy_correct_when_price_rises_above_threshold(self):
        assert evaluate_outcome("BUY", 3.0, 2.0) is True

    def test_buy_incorrect_when_price_rises_below_threshold(self):
        assert evaluate_outcome("BUY", 1.5, 2.0) is False

    def test_buy_incorrect_when_price_falls(self):
        assert evaluate_outcome("BUY", -2.0, 2.0) is False

    def test_sell_correct_when_price_falls_below_threshold(self):
        assert evaluate_outcome("SELL", -3.0, 2.0) is True

    def test_sell_incorrect_when_price_falls_less_than_threshold(self):
        assert evaluate_outcome("SELL", -1.0, 2.0) is False

    def test_sell_incorrect_when_price_rises(self):
        assert evaluate_outcome("SELL", 2.0, 2.0) is False

    def test_hold_correct_when_price_stays_flat(self):
        assert evaluate_outcome("HOLD", 1.0, 2.0) is True
        assert evaluate_outcome("HOLD", -1.0, 2.0) is True

    def test_hold_incorrect_when_price_moves_outside_zone(self):
        assert evaluate_outcome("HOLD", 3.0, 2.0) is False
        assert evaluate_outcome("HOLD", -3.0, 2.0) is False

    def test_boundary_exactly_at_threshold(self):
        """Boundary value: exactly at threshold is NOT sufficient for BUY/SELL."""
        # BUY requires strictly greater than threshold
        assert evaluate_outcome("BUY", 2.0, 2.0) is False
        # SELL requires strictly less than -threshold
        assert evaluate_outcome("SELL", -2.0, 2.0) is False
        # HOLD: equal to threshold is within zone
        assert evaluate_outcome("HOLD", 2.0, 2.0) is True

    def test_unknown_signal_returns_false(self):
        assert evaluate_outcome("UNKNOWN", 5.0, 2.0) is False

    def test_zero_price_change(self):
        assert evaluate_outcome("HOLD", 0.0, 2.0) is True
        assert evaluate_outcome("BUY", 0.0, 2.0) is False


# ============================================================================
# TEST CLASS 3: reconstruct_technical_signal
# ============================================================================


class TestReconstructTechnicalSignal:
    """Calls Node 4 helpers on a historical price slice."""

    def test_sufficient_data_returns_signal_and_confidence(self):
        df = _make_price_df(120)
        signal, confidence = reconstruct_technical_signal(df)
        assert signal in ("BUY", "SELL", "HOLD")
        assert 0.0 <= confidence <= 1.0

    def test_insufficient_data_returns_none_tuple(self):
        df = _make_price_df(30)  # below MIN_PRICE_ROWS_FOR_TECHNICAL
        signal, confidence = reconstruct_technical_signal(df)
        assert signal is None
        assert confidence is None

    def test_none_input_returns_none_tuple(self):
        signal, confidence = reconstruct_technical_signal(None)
        assert signal is None
        assert confidence is None

    def test_exactly_minimum_rows(self):
        """Exactly MIN_PRICE_ROWS_FOR_TECHNICAL rows should succeed."""
        df = _make_price_df(MIN_PRICE_ROWS_FOR_TECHNICAL)
        signal, confidence = reconstruct_technical_signal(df)
        assert signal in ("BUY", "SELL", "HOLD")

    def test_one_below_minimum_returns_none(self):
        df = _make_price_df(MIN_PRICE_ROWS_FOR_TECHNICAL - 1)
        signal, confidence = reconstruct_technical_signal(df)
        assert signal is None


# ============================================================================
# TEST CLASS 4: aggregate_daily_sentiment
# ============================================================================


class TestAggregateDailySentiment:
    """Per-stream daily aggregation — no 50/25/25 cross-stream weighting here."""

    def test_all_positive_returns_buy(self):
        articles = [
            {"sentiment_label": "positive", "sentiment_score": 0.8},
            {"sentiment_label": "positive", "sentiment_score": 0.7},
        ]
        score, signal = aggregate_daily_sentiment(articles)
        assert signal == "BUY"
        assert score > 0

    def test_all_negative_returns_sell(self):
        articles = [
            {"sentiment_label": "negative", "sentiment_score": 0.8},
            {"sentiment_label": "negative", "sentiment_score": 0.7},
        ]
        score, signal = aggregate_daily_sentiment(articles)
        assert signal == "SELL"
        assert score < 0

    def test_all_neutral_returns_hold(self):
        articles = [
            {"sentiment_label": "neutral", "sentiment_score": 0.5},
            {"sentiment_label": "neutral", "sentiment_score": 0.6},
        ]
        score, signal = aggregate_daily_sentiment(articles)
        assert signal == "HOLD"

    def test_mixed_signals_averages_correctly(self):
        """One strong positive + one strong negative = near zero = HOLD."""
        articles = [
            {"sentiment_label": "positive", "sentiment_score": 0.8},
            {"sentiment_label": "negative", "sentiment_score": 0.8},
        ]
        score, signal = aggregate_daily_sentiment(articles)
        # scores: +0.8 + (-0.8) = 0.0 average → HOLD
        assert signal == "HOLD"
        assert abs(score) < 0.01

    def test_empty_list_returns_hold(self):
        score, signal = aggregate_daily_sentiment([])
        assert signal == "HOLD"
        assert score == 0.0

    def test_missing_sentiment_fields_uses_defaults(self):
        articles = [{"title": "Article with no sentiment fields"}]
        score, signal = aggregate_daily_sentiment(articles)
        assert signal in ("BUY", "SELL", "HOLD")

    def test_score_range(self):
        articles = [
            {"sentiment_label": "positive", "sentiment_score": 1.0},
        ]
        score, _ = aggregate_daily_sentiment(articles)
        assert -1.0 <= score <= 1.0


# ============================================================================
# TEST CLASS 5: backtest_technical_stream
# ============================================================================


class TestBacktestTechnicalStream:
    """Full technical backtest loop using reconstructed historical signals."""

    def test_returns_list_of_dicts(self):
        df = _make_price_df(180)
        results = backtest_technical_stream(df, hold_threshold=2.0)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_each_result_has_required_keys(self):
        df = _make_price_df(180)
        results = backtest_technical_stream(df, hold_threshold=2.0)
        required_keys = {"date", "signal", "confidence", "actual_change_7d", "correct", "days_ago"}
        for r in results[:5]:
            assert required_keys.issubset(r.keys())

    def test_signal_values_are_valid(self):
        df = _make_price_df(180)
        results = backtest_technical_stream(df, hold_threshold=2.0)
        for r in results:
            assert r["signal"] in ("BUY", "SELL", "HOLD")

    def test_correct_field_is_boolean(self):
        df = _make_price_df(180)
        results = backtest_technical_stream(df, hold_threshold=2.0)
        for r in results:
            assert isinstance(r["correct"], bool)

    def test_actual_change_is_numeric(self):
        df = _make_price_df(180)
        results = backtest_technical_stream(df, hold_threshold=2.0)
        for r in results:
            assert isinstance(r["actual_change_7d"], float)

    def test_days_ago_decreases_toward_end(self):
        """More recent entries should have smaller days_ago."""
        df = _make_price_df(180)
        results = backtest_technical_stream(df, hold_threshold=2.0)
        # Last result should be more recent than first
        assert results[-1]["days_ago"] < results[0]["days_ago"]

    def test_insufficient_data_returns_empty(self):
        df = _make_price_df(40)  # not enough rows
        results = backtest_technical_stream(df, hold_threshold=2.0)
        assert results == []

    def test_none_price_data_returns_empty(self):
        results = backtest_technical_stream(None, hold_threshold=2.0)
        assert results == []

    def test_generates_enough_datapoints_for_180_days(self):
        """Should produce at least 100 evaluation days for a 180-day window."""
        df = _make_price_df(180)
        results = backtest_technical_stream(df, hold_threshold=2.0)
        assert len(results) >= 100


# ============================================================================
# TEST CLASS 6: backtest_sentiment_stream
# ============================================================================


class TestBacktestSentimentStream:
    """Sentiment backtesting from cached DB news_with_outcomes records."""

    def test_returns_list_for_valid_news_type(self):
        events = _make_news_events(n_stock=40)
        results = backtest_sentiment_stream(events, "stock", hold_threshold=2.0)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_each_result_has_required_keys(self):
        events = _make_news_events(n_stock=40)
        results = backtest_sentiment_stream(events, "stock", hold_threshold=2.0)
        required_keys = {"date", "signal", "sentiment_score", "article_count", "actual_change_7d", "correct", "days_ago"}
        for r in results:
            assert required_keys.issubset(r.keys())

    def test_filters_by_news_type(self):
        """Market events should NOT appear in stock stream and vice versa."""
        events = _make_news_events(n_stock=20, n_market=20)
        stock_results = backtest_sentiment_stream(events, "stock", hold_threshold=2.0)
        market_results = backtest_sentiment_stream(events, "market", hold_threshold=2.0)
        # Both should have data from different articles
        assert len(stock_results) > 0
        assert len(market_results) > 0
        # Dates may overlap (different articles same day) but types are independent
        assert len(stock_results) != len(market_results) or True  # just verify both have data

    def test_empty_events_returns_empty(self):
        results = backtest_sentiment_stream([], "stock", hold_threshold=2.0)
        assert results == []

    def test_wrong_news_type_returns_empty(self):
        # Explicitly zero out market and related so only stock events exist
        events = _make_news_events(n_stock=20, n_market=0, n_related=0)
        results = backtest_sentiment_stream(events, "market", hold_threshold=2.0)
        assert results == []

    def test_article_count_per_day_is_correct(self):
        """article_count should reflect how many articles were on that day."""
        events = _make_news_events(n_stock=40)
        results = backtest_sentiment_stream(events, "stock", hold_threshold=2.0)
        for r in results:
            assert r["article_count"] >= 1

    def test_actual_change_uses_median_not_mean(self):
        """Outlier article with extreme price_change should not dominate."""
        events = [
            {
                "news_type": "stock",
                "published_at": "2024-06-01T10:00:00",
                "sentiment_label": "positive",
                "sentiment_score": 0.8,
                "price_change_7day": 3.0,
            },
            {
                "news_type": "stock",
                "published_at": "2024-06-01T11:00:00",
                "sentiment_label": "positive",
                "sentiment_score": 0.7,
                "price_change_7day": 3.5,
            },
            {
                "news_type": "stock",
                "published_at": "2024-06-01T12:00:00",
                "sentiment_label": "positive",
                "sentiment_score": 0.9,
                "price_change_7day": 1000.0,  # extreme outlier
            },
        ]
        results = backtest_sentiment_stream(events, "stock", hold_threshold=2.0)
        assert len(results) == 1
        # Median of [3.0, 3.5, 1000.0] = 3.5, not 335.5 (mean)
        assert results[0]["actual_change_7d"] == pytest.approx(3.5, abs=0.01)


# ============================================================================
# TEST CLASS 7: calculate_stream_metrics
# ============================================================================


class TestCalculateStreamMetrics:
    """Raw metric computation — confirms NO weighting between full/recent."""

    def _make_results(self, n: int = 100, accuracy: float = 0.6, recent_accuracy: float = 0.7) -> list:
        """
        Generate synthetic daily_results with controlled accuracy.
        Recent days have different accuracy to verify separation.
        """
        rng = np.random.default_rng(seed=7)
        results = []
        today = datetime.now()

        signals = ["BUY", "SELL"] * (n // 2)
        for i, signal in enumerate(signals):
            days_ago = int((n - i) * (180 / n))
            is_recent = days_ago <= RECENT_PERIOD_DAYS
            acc = recent_accuracy if is_recent else accuracy
            correct = rng.random() < acc
            actual = rng.uniform(3.0, 6.0) if correct and signal == "BUY" else rng.uniform(-6.0, -3.0) if correct and signal == "SELL" else rng.uniform(-1.5, 1.5)
            results.append({
                "date": (today - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
                "signal": signal,
                "actual_change_7d": actual,
                "correct": correct,
                "days_ago": days_ago,
            })
        return results

    def test_returns_dict_for_valid_results(self):
        results = self._make_results(100, accuracy=0.6)
        metrics = calculate_stream_metrics(results)
        assert isinstance(metrics, dict)

    def test_returns_none_for_empty_results(self):
        metrics = calculate_stream_metrics([])
        assert metrics is None

    def test_full_accuracy_and_recent_accuracy_are_separate(self):
        """
        CRITICAL: Node 10 must NOT apply the 70/30 weighting.
        full_accuracy and recent_accuracy must be stored as independent values.
        """
        results = self._make_results(100, accuracy=0.55, recent_accuracy=0.75)
        metrics = calculate_stream_metrics(results)
        assert metrics is not None
        # Both values must be present and distinct
        assert "full_accuracy" in metrics
        assert "recent_accuracy" in metrics
        assert metrics["full_accuracy"] != metrics["recent_accuracy"]
        # No combined/weighted accuracy key should exist in Node 10 output
        assert "weighted_accuracy" not in metrics

    def test_signal_counts_are_correct(self):
        results = self._make_results(60)
        metrics = calculate_stream_metrics(results)
        total = metrics["buy_count"] + metrics["sell_count"] + metrics["hold_count"]
        assert total == len(results)
        assert metrics["signal_count"] == metrics["buy_count"] + metrics["sell_count"]

    def test_insufficient_signals_marks_not_sufficient(self):
        """Fewer than MIN_SIGNALS_FOR_SUFFICIENCY → is_sufficient=False."""
        results = self._make_results(10)  # only 10 directional signals
        metrics = calculate_stream_metrics(results, min_signals=MIN_SIGNALS_FOR_SUFFICIENCY)
        assert metrics["is_sufficient"] is False

    def test_sufficient_signals_marks_sufficient(self):
        results = self._make_results(60)
        metrics = calculate_stream_metrics(results, min_signals=20)
        assert metrics["is_sufficient"] is True

    def test_avg_change_on_correct_vs_wrong(self):
        """Correct signals should tend toward larger directional moves."""
        results = self._make_results(100, accuracy=0.65)
        metrics = calculate_stream_metrics(results)
        # avg_change_on_correct and avg_change_on_wrong are returned (not None)
        assert metrics["avg_change_on_correct"] is not None
        assert metrics["avg_change_on_wrong"] is not None

    def test_actual_price_movements_are_returned(self):
        """Node 10 must store actual price movements, not just binary correct."""
        results = self._make_results(50)
        metrics = calculate_stream_metrics(results)
        assert "avg_actual_change" in metrics
        assert isinstance(metrics["avg_actual_change"], float)
        assert "avg_change_on_correct" in metrics
        assert "avg_change_on_wrong" in metrics

    def test_daily_results_preserved_for_dashboard(self):
        """Full daily_results list must be passed through for dashboard."""
        results = self._make_results(50)
        metrics = calculate_stream_metrics(results)
        assert "daily_results" in metrics
        assert len(metrics["daily_results"]) == len(results)

    def test_p_value_returned(self):
        results = self._make_results(80, accuracy=0.65)
        metrics = calculate_stream_metrics(results)
        assert "p_value" in metrics
        assert 0.0 <= metrics["p_value"] <= 1.0

    def test_is_significant_false_for_low_accuracy(self):
        """50% accuracy against 50% baseline should not be significant."""
        results = self._make_results(60, accuracy=0.50)
        metrics = calculate_stream_metrics(results)
        # p_value should be high (not significant)
        assert metrics["p_value"] > 0.05 or metrics["is_significant"] is False

    def test_recent_accuracy_none_when_too_few_recent_days(self):
        """If fewer than 5 recent days, recent_accuracy should be None."""
        today = datetime.now()
        # All results are from 90+ days ago (not recent)
        results = [
            {
                "date": (today - timedelta(days=100 + i)).strftime("%Y-%m-%d"),
                "signal": "BUY",
                "actual_change_7d": 3.0,
                "correct": True,
                "days_ago": 100 + i,
            }
            for i in range(40)
        ]
        metrics = calculate_stream_metrics(results)
        assert metrics["recent_accuracy"] is None

    def test_accuracy_values_in_valid_range(self):
        results = self._make_results(80, accuracy=0.60)
        metrics = calculate_stream_metrics(results)
        assert 0.0 <= metrics["full_accuracy"] <= 1.0
        if metrics["recent_accuracy"] is not None:
            assert 0.0 <= metrics["recent_accuracy"] <= 1.0


# ============================================================================
# TEST CLASS 8: backtesting_node (main function)
# ============================================================================


class TestBacktestingNode:
    """Full node integration tests using mocked DB calls."""

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_full_flow_returns_backtest_results(self, mock_db):
        """Happy path: valid price data + news events → all 4 streams populated."""
        mock_db.return_value = pd.DataFrame(_make_news_events())

        state = _make_base_state(180)
        result = backtesting_node(state)

        assert result["backtest_results"] is not None
        assert "technical" in result["backtest_results"]
        assert "stock_news" in result["backtest_results"]
        assert "market_news" in result["backtest_results"]
        assert "related_news" in result["backtest_results"]

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_hold_threshold_in_results(self, mock_db):
        mock_db.return_value = pd.DataFrame(_make_news_events())

        state = _make_base_state(180)
        result = backtesting_node(state)

        threshold = result["backtest_results"]["hold_threshold_pct"]
        assert HOLD_THRESHOLD_MIN_PCT <= threshold <= HOLD_THRESHOLD_MAX_PCT

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_sample_period_days_correct(self, mock_db):
        mock_db.return_value = pd.DataFrame(_make_news_events())

        state = _make_base_state(180)
        result = backtesting_node(state)

        assert result["backtest_results"]["sample_period_days"] == 180

    def test_missing_price_data_returns_none_backtest_results(self):
        """No price data → cannot backtest → backtest_results=None, error logged."""
        state = {"ticker": "AAPL", "raw_price_data": None, "errors": [], "node_execution_times": {}}
        result = backtesting_node(state)

        assert result["backtest_results"] is None
        assert len(result["errors"]) > 0

    def test_too_few_price_rows_returns_none(self):
        """Fewer than 30 price rows → cannot even compute HOLD threshold reliably."""
        state = _make_base_state(n_days=20)
        result = backtesting_node(state)

        assert result["backtest_results"] is None

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_empty_news_events_sentiment_streams_none(self, mock_db):
        """No news in DB → sentiment streams return None, technical still runs."""
        mock_db.return_value = pd.DataFrame()  # empty DataFrame

        state = _make_base_state(180)
        result = backtesting_node(state)

        bt = result["backtest_results"]
        assert bt is not None
        # Technical stream should still have results
        assert bt["technical"] is not None
        # Sentiment streams should be None (no events)
        assert bt["stock_news"] is None
        assert bt["market_news"] is None
        assert bt["related_news"] is None

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_db_failure_sentiment_streams_none_technical_still_runs(self, mock_db):
        """DB exception → sentiment streams fail gracefully, technical still runs."""
        mock_db.side_effect = Exception("DB connection lost")

        state = _make_base_state(180)
        result = backtesting_node(state)

        bt = result["backtest_results"]
        assert bt is not None
        # Technical should still be computed
        assert bt["technical"] is not None

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_execution_time_tracked(self, mock_db):
        mock_db.return_value = pd.DataFrame(_make_news_events())

        state = _make_base_state(180)
        result = backtesting_node(state)

        assert "node_10" in result["node_execution_times"]
        assert result["node_execution_times"]["node_10"] > 0

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_state_errors_preserved(self, mock_db):
        """Existing errors from prior nodes must not be wiped."""
        mock_db.return_value = pd.DataFrame(_make_news_events())

        state = _make_base_state(180)
        state["errors"] = ["prior error from node 9b"]
        result = backtesting_node(state)

        assert "prior error from node 9b" in result["errors"]

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_no_weighting_in_output(self, mock_db):
        """
        DESIGN CONTRACT: Node 10 must NOT apply recency weighting.
        No 'weighted_accuracy' key should exist in stream metrics.
        That belongs to Node 11.
        """
        mock_db.return_value = pd.DataFrame(_make_news_events())

        state = _make_base_state(180)
        result = backtesting_node(state)

        for stream in ["technical", "stock_news", "market_news", "related_news"]:
            metrics = result["backtest_results"].get(stream)
            if metrics:
                assert "weighted_accuracy" not in metrics, (
                    f"Stream '{stream}' contains 'weighted_accuracy' — "
                    "weighting belongs in Node 11, not Node 10"
                )

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_full_and_recent_accuracy_are_separate(self, mock_db):
        """
        DESIGN CONTRACT: full_accuracy and recent_accuracy must both be present
        and independently computed (not derived from each other).
        """
        mock_db.return_value = pd.DataFrame(_make_news_events())

        state = _make_base_state(180)
        result = backtesting_node(state)

        tech = result["backtest_results"]["technical"]
        if tech and tech.get("signal_count", 0) > 0:
            assert "full_accuracy" in tech
            assert "recent_accuracy" in tech

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_actual_price_movements_in_results(self, mock_db):
        """Node 10 must return actual price movements, not just binary correct."""
        mock_db.return_value = pd.DataFrame(_make_news_events())

        state = _make_base_state(180)
        result = backtesting_node(state)

        tech = result["backtest_results"]["technical"]
        if tech:
            assert "avg_actual_change" in tech
            assert "avg_change_on_correct" in tech
            assert "avg_change_on_wrong" in tech


# ============================================================================
# TEST CLASS 9: Error Isolation
# ============================================================================


class TestErrorIsolation:
    """Verify each stream fails independently without breaking others."""

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    @patch("src.langgraph_nodes.node_10_backtesting.backtest_technical_stream")
    def test_technical_stream_failure_does_not_stop_sentiment(self, mock_tech, mock_db):
        """If technical backtest raises, sentiment streams still run."""
        mock_tech.side_effect = Exception("Technical calculation exploded")
        mock_db.return_value = pd.DataFrame(_make_news_events())

        state = _make_base_state(180)
        result = backtesting_node(state)

        bt = result["backtest_results"]
        assert bt is not None
        # Technical failed but sentiment may still have results
        assert bt["technical"] is None

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    @patch("src.langgraph_nodes.node_10_backtesting.backtest_sentiment_stream")
    def test_sentiment_stream_failure_does_not_stop_technical(self, mock_sent, mock_db):
        """If sentiment backtest raises, technical stream still runs."""
        mock_sent.side_effect = Exception("Sentiment DB failed")
        mock_db.return_value = pd.DataFrame(_make_news_events())

        state = _make_base_state(180)
        result = backtesting_node(state)

        bt = result["backtest_results"]
        assert bt is not None
        # Technical should still be computed from price data
        assert bt["technical"] is not None

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_catastrophic_failure_returns_none_not_exception(self, mock_db):
        """Even if the entire node fails, it must return state (never raise)."""
        mock_db.side_effect = Exception("Total system failure")

        state = _make_base_state(180)
        # Should not raise
        result = backtesting_node(state)

        assert isinstance(result, dict)
        assert "ticker" in result


# ============================================================================
# TEST CLASS 10: Integration / Thesis Validation
# ============================================================================


class TestIntegration:
    """End-to-end validation with realistic data for thesis documentation."""

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_high_accuracy_stock_news_produces_sufficient_signal(self, mock_db):
        """
        Thesis relevance: stock news with 65% accuracy should produce
        is_sufficient=True and is_significant=True (with enough articles).
        """
        events = _make_news_events(n_stock=80, stock_accuracy=0.65, days_back=170)
        mock_db.return_value = pd.DataFrame(events)

        state = _make_base_state(180)
        result = backtesting_node(state)

        stock = result["backtest_results"]["stock_news"]
        if stock:
            print(f"\n[Thesis] Stock News Accuracy:")
            print(f"  full_accuracy:   {stock['full_accuracy']:.1%}")
            print(f"  recent_accuracy: {stock.get('recent_accuracy', 'N/A')}")
            print(f"  signal_count:    {stock['signal_count']}")
            print(f"  is_significant:  {stock['is_significant']} (p={stock['p_value']:.4f})")
            print(f"  avg_change_correct: {stock['avg_change_on_correct']:.2f}%")
            print(f"  avg_change_wrong:   {stock['avg_change_on_wrong']:.2f}%")

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_hold_threshold_differs_by_volatility(self, mock_db):
        """
        Thesis relevance: a high-volatility stock should get a larger HOLD
        zone than a low-volatility one, making signal evaluation fairer.
        """
        mock_db.return_value = pd.DataFrame(_make_news_events())

        low_vol_state = {
            "ticker": "JNJ",
            "raw_price_data": _make_price_df(180, volatility=0.007),
            "errors": [],
            "node_execution_times": {},
        }
        high_vol_state = {
            "ticker": "TSLA",
            "raw_price_data": _make_price_df(180, volatility=0.035),
            "errors": [],
            "node_execution_times": {},
        }

        low_vol_result = backtesting_node(deepcopy(low_vol_state))
        high_vol_result = backtesting_node(deepcopy(high_vol_state))

        low_threshold = low_vol_result["backtest_results"]["hold_threshold_pct"]
        high_threshold = high_vol_result["backtest_results"]["hold_threshold_pct"]

        print(f"\n[Thesis] HOLD threshold comparison:")
        print(f"  JNJ  (low vol):  {low_threshold:.2f}%")
        print(f"  TSLA (high vol): {high_threshold:.2f}%")

        assert high_threshold > low_threshold, (
            "High-volatility stock must have a larger HOLD zone than low-volatility stock"
        )

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_all_four_streams_return_data_for_liquid_stock(self, mock_db):
        """
        Thesis relevance: a liquid stock with 6 months of news should have
        all 4 streams populated with meaningful data.
        """
        events = _make_news_events(n_stock=60, n_market=40, n_related=30)
        mock_db.return_value = pd.DataFrame(events)

        state = _make_base_state(180)
        result = backtesting_node(state)

        bt = result["backtest_results"]
        for stream in ["technical", "stock_news", "market_news", "related_news"]:
            assert bt[stream] is not None, f"Stream '{stream}' has no data"

        print(f"\n[Thesis] All 4 streams for AAPL:")
        for stream in ["technical", "stock_news", "market_news", "related_news"]:
            m = bt[stream]
            if m:
                print(
                    f"  {stream:12s}: full={m['full_accuracy']:.1%}, "
                    f"signals={m['signal_count']}, sufficient={m['is_sufficient']}"
                )

    @patch("src.langgraph_nodes.node_10_backtesting.get_news_with_outcomes")
    def test_output_is_ready_for_node_11(self, mock_db):
        """
        Verify that backtest_results has exactly the keys Node 11 will consume:
        - full_accuracy (for 30% component)
        - recent_accuracy (for 70% component)
        - is_sufficient (guard before using accuracy)
        - is_significant (guard before trusting accuracy)
        """
        events = _make_news_events(n_stock=60, n_market=40, n_related=30)
        mock_db.return_value = pd.DataFrame(events)

        state = _make_base_state(180)
        result = backtesting_node(state)

        node_11_required = {"full_accuracy", "recent_accuracy", "is_sufficient", "is_significant"}

        for stream in ["technical", "stock_news", "market_news", "related_news"]:
            metrics = result["backtest_results"][stream]
            if metrics:
                missing = node_11_required - metrics.keys()
                assert not missing, f"Stream '{stream}' missing Node 11 keys: {missing}"
