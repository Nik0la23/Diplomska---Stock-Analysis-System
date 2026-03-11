"""
Tests for Node 4: Technical Analysis

Covers:
1. RSI calculation and interpretation
2. MACD calculation and crossover detection
3. Bollinger Bands calculation and position
4. Moving averages and trend detection
5. Volume analysis
6. ADX calculation
7. build_feature_matrix: shape, column names, NaN warmup
8. fit_ic_regression: IC range, required keys, insufficient data
9. calculate_technical_score: regression path, fallback path, alpha range
10. technical_analysis_node: full node output with new IC/alpha keys
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.langgraph_nodes.node_04_technical_analysis import (
    analyze_volume,
    build_feature_matrix,
    calculate_adx,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
    calculate_technical_score,
    fit_ic_regression,
    technical_analysis_node,
)


# ============================================================================
# SHARED FIXTURES
# ============================================================================


def _make_ohlcv(n: int, start: float = 100.0, trend: float = 0.001,
                vol: float = 0.015, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with n rows."""
    rng = np.random.default_rng(seed=seed)
    dates = pd.date_range(end=datetime.now(), periods=n, freq="B")
    returns = rng.normal(trend, vol, n)
    closes = start * np.exp(np.cumsum(returns))
    opens  = closes * (1 + rng.normal(0, 0.003, n))
    highs  = np.maximum(closes, opens) * (1 + abs(rng.normal(0, 0.005, n)))
    lows   = np.minimum(closes, opens) * (1 - abs(rng.normal(0, 0.005, n)))
    volumes = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "date":   [d.strftime("%Y-%m-%d") for d in dates],
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    })


@pytest.fixture
def sample_price_data_100_days():
    return _make_ohlcv(100)


@pytest.fixture
def bullish_price_data():
    """Strong linear uptrend."""
    prices = np.linspace(100, 150, 100)
    dates  = pd.date_range(end=datetime.now(), periods=100, freq="B")
    rng = np.random.default_rng(seed=123)
    return pd.DataFrame({
        "date":   [d.strftime("%Y-%m-%d") for d in dates],
        "open":   prices * 0.99,
        "high":   prices * 1.01,
        "low":    prices * 0.99,
        "close":  prices,
        "volume": rng.integers(2_000_000, 6_000_000, 100).astype(float),
    })


@pytest.fixture
def bearish_price_data():
    """Strong linear downtrend."""
    prices = np.linspace(150, 100, 100)
    dates  = pd.date_range(end=datetime.now(), periods=100, freq="B")
    rng = np.random.default_rng(seed=456)
    return pd.DataFrame({
        "date":   [d.strftime("%Y-%m-%d") for d in dates],
        "open":   prices * 1.01,
        "high":   prices * 1.02,
        "low":    prices * 0.99,
        "close":  prices,
        "volume": rng.integers(2_000_000, 6_000_000, 100).astype(float),
    })


@pytest.fixture
def insufficient_price_data():
    """Only 20 rows — below Node 4's 50-row minimum."""
    prices = np.linspace(100, 105, 20)
    dates  = pd.date_range(end=datetime.now(), periods=20, freq="B")
    return pd.DataFrame({
        "date":   [d.strftime("%Y-%m-%d") for d in dates],
        "open":   prices * 0.99,
        "high":   prices * 1.01,
        "low":    prices * 0.99,
        "close":  prices,
        "volume": [1_000_000.0] * 20,
    })


@pytest.fixture
def price_data_180_days():
    """180 days with a detectable trend — enough for IC regression."""
    return _make_ohlcv(180, trend=0.001)


# ============================================================================
# 1. RSI
# ============================================================================


class TestCalculateRsi:

    def test_returns_float_in_valid_range(self, sample_price_data_100_days):
        rsi = calculate_rsi(sample_price_data_100_days, period=14)
        assert rsi is not None
        assert isinstance(rsi, float)
        assert 0.0 <= rsi <= 100.0

    def test_uptrend_rsi_above_50(self, bullish_price_data):
        rsi = calculate_rsi(bullish_price_data, period=14)
        assert rsi is not None
        assert rsi > 50.0

    def test_downtrend_rsi_below_50(self, bearish_price_data):
        rsi = calculate_rsi(bearish_price_data, period=14)
        assert rsi is not None
        assert rsi < 50.0

    def test_insufficient_data_returns_none(self):
        df = _make_ohlcv(5)
        rsi = calculate_rsi(df, period=14)
        assert rsi is None


# ============================================================================
# 2. MACD
# ============================================================================


class TestCalculateMacd:

    def test_returns_required_keys(self, sample_price_data_100_days):
        macd = calculate_macd(sample_price_data_100_days)
        assert macd is not None
        assert {"macd", "signal", "histogram", "crossover"} <= macd.keys()
        assert macd["crossover"] in ("bullish", "bearish", "none")

    def test_bullish_trend_macd_above_signal(self, bullish_price_data):
        macd = calculate_macd(bullish_price_data)
        assert macd is not None
        assert macd["macd"] > macd["signal"] or macd["crossover"] == "bullish"

    def test_bearish_trend_macd_below_signal(self, bearish_price_data):
        macd = calculate_macd(bearish_price_data)
        assert macd is not None
        assert macd["macd"] < macd["signal"] or macd["crossover"] == "bearish"

    def test_insufficient_data_returns_none(self, insufficient_price_data):
        assert calculate_macd(insufficient_price_data) is None


# ============================================================================
# 3. Bollinger Bands
# ============================================================================


class TestCalculateBollingerBands:

    def test_returns_required_keys(self, sample_price_data_100_days):
        bb = calculate_bollinger_bands(sample_price_data_100_days)
        assert bb is not None
        assert {"upper_band", "middle_band", "lower_band", "bandwidth", "position"} <= bb.keys()

    def test_band_ordering(self, sample_price_data_100_days):
        bb = calculate_bollinger_bands(sample_price_data_100_days)
        assert bb is not None
        assert bb["upper_band"] > bb["middle_band"] > bb["lower_band"]

    def test_position_valid_values(self, sample_price_data_100_days):
        bb = calculate_bollinger_bands(sample_price_data_100_days)
        assert bb["position"] in ("upper", "middle", "lower")

    def test_insufficient_data_returns_none(self):
        assert calculate_bollinger_bands(_make_ohlcv(5)) is None


# ============================================================================
# 4. Moving Averages
# ============================================================================


class TestCalculateMovingAverages:

    def test_returns_required_keys(self, sample_price_data_100_days):
        ma = calculate_moving_averages(sample_price_data_100_days)
        assert ma is not None
        assert {"sma_20", "sma_50", "current_price", "trend", "golden_cross", "death_cross"} <= ma.keys()

    def test_trend_valid_values(self, sample_price_data_100_days):
        ma = calculate_moving_averages(sample_price_data_100_days)
        assert ma["trend"] in ("strong_uptrend", "uptrend", "downtrend", "strong_downtrend", "neutral")

    def test_golden_cross_in_uptrend(self, bullish_price_data):
        ma = calculate_moving_averages(bullish_price_data)
        assert ma is not None
        assert ma["golden_cross"] is True
        assert ma["death_cross"] is False
        assert ma["trend"] in ("uptrend", "strong_uptrend")

    def test_death_cross_in_downtrend(self, bearish_price_data):
        ma = calculate_moving_averages(bearish_price_data)
        assert ma is not None
        assert ma["death_cross"] is True
        assert ma["golden_cross"] is False
        assert ma["trend"] in ("downtrend", "strong_downtrend")

    def test_insufficient_data_returns_none(self, insufficient_price_data):
        assert calculate_moving_averages(insufficient_price_data) is None


# ============================================================================
# 5. Volume Analysis
# ============================================================================


class TestAnalyzeVolume:

    def test_returns_required_keys(self, sample_price_data_100_days):
        vol = analyze_volume(sample_price_data_100_days)
        assert vol is not None
        assert {"current_volume", "average_volume", "volume_ratio", "volume_signal"} <= vol.keys()
        assert vol["volume_signal"] in ("high", "normal", "low")

    def test_high_volume_detection(self, sample_price_data_100_days):
        df = sample_price_data_100_days.copy()
        df.loc[df.index[-1], "volume"] = df["volume"].mean() * 3.0
        vol = analyze_volume(df)
        assert vol["volume_signal"] == "high"
        assert vol["volume_ratio"] > 1.5

    def test_low_volume_detection(self, sample_price_data_100_days):
        df = sample_price_data_100_days.copy()
        df.loc[df.index[-1], "volume"] = df["volume"].mean() * 0.2
        vol = analyze_volume(df)
        assert vol["volume_signal"] == "low"
        assert vol["volume_ratio"] < 0.5

    def test_no_volume_column_returns_none(self):
        df = _make_ohlcv(50)[["date", "open", "high", "low", "close"]]
        assert analyze_volume(df) is None


# ============================================================================
# 6. ADX
# ============================================================================


class TestCalculateAdx:

    def test_returns_positive_float(self, sample_price_data_100_days):
        adx = calculate_adx(sample_price_data_100_days)
        assert adx is not None
        assert isinstance(adx, float)
        assert adx >= 0.0

    def test_strong_trend_high_adx(self, bullish_price_data):
        adx = calculate_adx(bullish_price_data)
        assert adx is not None
        assert adx > 20.0  # linear trend should produce strong ADX

    def test_insufficient_data_returns_none(self, insufficient_price_data):
        assert calculate_adx(insufficient_price_data) is None


# ============================================================================
# 7. build_feature_matrix
# ============================================================================


class TestBuildFeatureMatrix:

    EXPECTED_COLUMNS = {
        "rsi", "macd_hist", "bb_pct_b", "sma20_slope_5d",
        "volume_ratio", "adx", "mom_5d", "mom_20d",
    }

    def test_returns_dataframe_with_8_columns(self, sample_price_data_100_days):
        feat = build_feature_matrix(sample_price_data_100_days)
        assert isinstance(feat, pd.DataFrame)
        assert set(feat.columns) == self.EXPECTED_COLUMNS

    def test_output_length_matches_input(self, sample_price_data_100_days):
        feat = build_feature_matrix(sample_price_data_100_days)
        assert len(feat) == len(sample_price_data_100_days)

    def test_first_rows_contain_nan(self, sample_price_data_100_days):
        feat = build_feature_matrix(sample_price_data_100_days)
        # RSI needs 14 warmup rows — first few should have NaN
        assert feat["rsi"].iloc[:14].isna().any()

    def test_later_rows_mostly_valid(self, price_data_180_days):
        feat = build_feature_matrix(price_data_180_days)
        # After warmup, majority of rows should be non-NaN for RSI
        valid_rsi = feat["rsi"].dropna()
        assert len(valid_rsi) > 100

    def test_bb_pct_b_clipped_to_0_1(self, sample_price_data_100_days):
        feat = build_feature_matrix(sample_price_data_100_days)
        valid = feat["bb_pct_b"].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_volume_ratio_positive(self, sample_price_data_100_days):
        feat = build_feature_matrix(sample_price_data_100_days)
        valid = feat["volume_ratio"].dropna()
        assert (valid >= 0.0).all()


# ============================================================================
# 8. fit_ic_regression
# ============================================================================


class TestFitIcRegression:

    REQUIRED_KEYS = {"ic_score", "predicted_return_pct", "return_std", "n_training_samples"}

    def test_returns_dict_with_required_keys(self, price_data_180_days):
        result = fit_ic_regression(price_data_180_days)
        assert result is not None
        assert self.REQUIRED_KEYS <= result.keys()

    def test_ic_in_valid_range(self, price_data_180_days):
        result = fit_ic_regression(price_data_180_days)
        assert result is not None
        assert -1.0 <= result["ic_score"] <= 1.0

    def test_n_training_samples_reasonable(self, price_data_180_days):
        result = fit_ic_regression(price_data_180_days)
        assert result is not None
        # 180 rows - warmup (~20) - forward window (7) - test split = should be ≥60
        assert result["n_training_samples"] >= 60

    def test_insufficient_data_returns_none(self, insufficient_price_data):
        result = fit_ic_regression(insufficient_price_data)
        assert result is None

    def test_returns_none_below_60_rows(self):
        df = _make_ohlcv(50)
        result = fit_ic_regression(df)
        assert result is None

    def test_predicted_return_is_finite(self, price_data_180_days):
        result = fit_ic_regression(price_data_180_days)
        assert result is not None
        assert np.isfinite(result["predicted_return_pct"])

    def test_return_std_positive(self, price_data_180_days):
        result = fit_ic_regression(price_data_180_days)
        assert result is not None
        assert result["return_std"] > 0.0


# ============================================================================
# 9. calculate_technical_score
# ============================================================================


class TestCalculateTechnicalScore:

    REQUIRED_KEYS = {
        "technical_alpha", "predicted_return_pct", "ic_score",
        "regression_valid", "n_training_samples",
        "normalized_score", "market_regime", "hold_low", "hold_high",
    }

    def test_regression_path_with_sufficient_data(self, price_data_180_days):
        rsi  = calculate_rsi(price_data_180_days)
        macd = calculate_macd(price_data_180_days)
        bb   = calculate_bollinger_bands(price_data_180_days)
        ma   = calculate_moving_averages(price_data_180_days)
        vol  = analyze_volume(price_data_180_days)
        adx  = calculate_adx(price_data_180_days)

        result = calculate_technical_score(rsi, macd, bb, ma, vol,
                                           price_data=price_data_180_days, adx=adx)
        assert result["regression_valid"] is True

    def test_technical_alpha_in_range(self, price_data_180_days):
        rsi = calculate_rsi(price_data_180_days)
        macd = calculate_macd(price_data_180_days)
        bb   = calculate_bollinger_bands(price_data_180_days)
        ma   = calculate_moving_averages(price_data_180_days)
        vol  = analyze_volume(price_data_180_days)
        adx  = calculate_adx(price_data_180_days)

        result = calculate_technical_score(rsi, macd, bb, ma, vol,
                                           price_data=price_data_180_days, adx=adx)
        alpha = result["technical_alpha"]
        assert -1.0 <= alpha <= 1.0

    def test_fallback_when_price_data_none(self, sample_price_data_100_days):
        rsi  = calculate_rsi(sample_price_data_100_days)
        macd = calculate_macd(sample_price_data_100_days)
        bb   = calculate_bollinger_bands(sample_price_data_100_days)
        ma   = calculate_moving_averages(sample_price_data_100_days)
        vol  = analyze_volume(sample_price_data_100_days)

        result = calculate_technical_score(rsi, macd, bb, ma, vol,
                                           price_data=None, adx=None)
        assert result["regression_valid"] is False
        assert -1.0 <= result["technical_alpha"] <= 1.0

    def test_fallback_when_insufficient_data(self, insufficient_price_data):
        # With <50 rows the node would fail, but calculate_technical_score is called
        # with pre-computed indicator values even on short data
        result = calculate_technical_score(None, None, None, None, None,
                                           price_data=insufficient_price_data, adx=None)
        assert result["regression_valid"] is False

    def test_normalized_score_derived_from_alpha(self, price_data_180_days):
        rsi  = calculate_rsi(price_data_180_days)
        macd = calculate_macd(price_data_180_days)
        bb   = calculate_bollinger_bands(price_data_180_days)
        ma   = calculate_moving_averages(price_data_180_days)
        vol  = analyze_volume(price_data_180_days)

        result = calculate_technical_score(rsi, macd, bb, ma, vol,
                                           price_data=price_data_180_days)
        alpha = result["technical_alpha"]
        expected_ns = float(np.clip((alpha + 1.0) * 50.0, 0.0, 100.0))
        assert abs(result["normalized_score"] - expected_ns) < 1e-9

    def test_all_required_keys_present(self, price_data_180_days):
        rsi  = calculate_rsi(price_data_180_days)
        macd = calculate_macd(price_data_180_days)
        bb   = calculate_bollinger_bands(price_data_180_days)
        ma   = calculate_moving_averages(price_data_180_days)
        vol  = analyze_volume(price_data_180_days)

        result = calculate_technical_score(rsi, macd, bb, ma, vol,
                                           price_data=price_data_180_days)
        assert self.REQUIRED_KEYS <= result.keys()

    def test_ic_score_present_and_finite(self, price_data_180_days):
        rsi  = calculate_rsi(price_data_180_days)
        macd = calculate_macd(price_data_180_days)
        bb   = calculate_bollinger_bands(price_data_180_days)
        ma   = calculate_moving_averages(price_data_180_days)
        vol  = analyze_volume(price_data_180_days)

        result = calculate_technical_score(rsi, macd, bb, ma, vol,
                                           price_data=price_data_180_days)
        assert np.isfinite(result["ic_score"])


# ============================================================================
# 10. technical_analysis_node (main node)
# ============================================================================


class TestTechnicalAnalysisNode:

    def test_node_success_with_sufficient_data(self, price_data_180_days):
        state = {"ticker": "AAPL", "raw_price_data": price_data_180_days}
        result = technical_analysis_node(state)

        assert result["technical_indicators"] is not None
        assert "node_4" in result["node_execution_times"]

    def test_node_returns_technical_alpha(self, price_data_180_days):
        state = {"ticker": "AAPL", "raw_price_data": price_data_180_days}
        result = technical_analysis_node(state)
        ti = result["technical_indicators"]
        assert "technical_alpha" in ti
        assert -1.0 <= ti["technical_alpha"] <= 1.0

    def test_node_returns_ic_score(self, price_data_180_days):
        state = {"ticker": "AAPL", "raw_price_data": price_data_180_days}
        result = technical_analysis_node(state)
        ti = result["technical_indicators"]
        assert "ic_score" in ti
        assert np.isfinite(ti["ic_score"])

    def test_node_returns_regression_valid(self, price_data_180_days):
        state = {"ticker": "AAPL", "raw_price_data": price_data_180_days}
        result = technical_analysis_node(state)
        ti = result["technical_indicators"]
        assert "regression_valid" in ti
        assert isinstance(ti["regression_valid"], bool)

    def test_node_regression_true_with_180_days(self, price_data_180_days):
        state = {"ticker": "AAPL", "raw_price_data": price_data_180_days}
        result = technical_analysis_node(state)
        ti = result["technical_indicators"]
        assert ti["regression_valid"] is True

    def test_node_returns_normalized_score_legacy(self, price_data_180_days):
        state = {"ticker": "AAPL", "raw_price_data": price_data_180_days}
        result = technical_analysis_node(state)
        ti = result["technical_indicators"]
        assert "normalized_score" in ti
        assert 0.0 <= ti["normalized_score"] <= 100.0

    def test_node_error_with_no_price_data(self):
        state = {"ticker": "AAPL", "raw_price_data": None}
        result = technical_analysis_node(state)
        assert result["technical_indicators"] is None
        assert len(result.get("errors", [])) > 0
        assert "Node 4" in result["errors"][0]

    def test_node_error_with_insufficient_data(self, insufficient_price_data):
        state = {"ticker": "AAPL", "raw_price_data": insufficient_price_data}
        result = technical_analysis_node(state)
        assert result["technical_indicators"] is None
        assert len(result.get("errors", [])) > 0

    def test_node_summary_contains_alpha(self, price_data_180_days):
        state = {"ticker": "AAPL", "raw_price_data": price_data_180_days}
        result = technical_analysis_node(state)
        ti = result["technical_indicators"]
        summary = ti.get("technical_summary", "")
        assert "alpha" in summary.lower()

    def test_node_summary_mentions_ic(self, price_data_180_days):
        state = {"ticker": "AAPL", "raw_price_data": price_data_180_days}
        result = technical_analysis_node(state)
        ti = result["technical_indicators"]
        summary = ti.get("technical_summary", "")
        assert "ic=" in summary.lower() or "IC=" in summary

    def test_node_execution_time_recorded(self, price_data_180_days):
        state = {"ticker": "AAPL", "raw_price_data": price_data_180_days}
        result = technical_analysis_node(state)
        assert "node_4" in result["node_execution_times"]
        assert result["node_execution_times"]["node_4"] >= 0.0

    def test_node_preserves_raw_indicators(self, price_data_180_days):
        state = {"ticker": "AAPL", "raw_price_data": price_data_180_days}
        result = technical_analysis_node(state)
        ti = result["technical_indicators"]
        assert "rsi" in ti
        assert "macd" in ti
        assert "bollinger_bands" in ti
        assert "moving_averages" in ti
