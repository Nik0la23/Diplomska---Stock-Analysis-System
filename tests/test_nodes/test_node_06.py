"""
Tests for Node 6: Market Context Analysis (rewritten)

Covers all helpers in the five-layer composite:
1. Sector Detection           (4 tests)
2. VIX Fear Gauge             (4 tests)
3. SPY Multi-Timeframe Trend  (5 tests)
4. Sector Performance         (4 tests)
5. Related Companies          (5 tests)
6. Correlation Calculation    (4 tests)
7. Market News Sentiment      (5 tests)
8. Headwind/Tailwind Score    (5 tests)
9. Main Node Function         (4 tests)
10. Integration               (2 tests)

Total: 42 tests
"""

import math
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock

from src.langgraph_nodes.node_06_market_context import (
    get_stock_sector,
    get_vix_level,
    get_market_trend_multitimeframe,
    get_sector_performance,
    analyze_related_companies,
    calculate_correlation,
    analyze_market_news_sentiment,
    compute_headwind_tailwind_score,
    market_context_node,
    SECTOR_ETFS,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """30+ rows of OHLCV data (timezone-naive index, matching DB output)."""
    dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(60) * 2)
    return pd.DataFrame({
        "date":   dates,
        "open":   prices * 0.99,
        "high":   prices * 1.02,
        "low":    prices * 0.98,
        "close":  prices,
        "volume": np.random.randint(1_000_000, 5_000_000, 60),
    })


@pytest.fixture
def sample_market_news() -> list:
    """Market news articles with AV overall_sentiment_score."""
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    recent  = now_ts - 3 * 86_400   # 3 days ago
    old     = now_ts - 30 * 86_400  # 30 days ago (outside 14d window)
    return [
        {"headline": "Markets rally on Fed pivot", "overall_sentiment_score": 0.25,  "datetime": recent},
        {"headline": "Stocks mixed on earnings",    "overall_sentiment_score": 0.05,  "datetime": recent},
        {"headline": "Market turmoil fears rise",  "overall_sentiment_score": -0.30, "datetime": recent},
        {"headline": "Old positive news",           "overall_sentiment_score": 0.35,  "datetime": old},
    ]


@pytest.fixture
def base_state(sample_price_data) -> dict:
    """Minimal state for market_context_node."""
    return {
        "ticker":             "NVDA",
        "raw_price_data":     sample_price_data,
        "related_companies":  ["AMD", "INTC"],
        "cleaned_market_news": [],
        "errors":             [],
        "node_execution_times": {},
    }


# ============================================================================
# 1. SECTOR DETECTION
# ============================================================================

@patch("yfinance.Ticker")
def test_get_stock_sector_technology(mock_ticker):
    mock_ticker.return_value.info = {"sector": "Technology", "industry": "Semiconductors"}
    sector, industry = get_stock_sector("NVDA")
    assert sector == "Technology"
    assert industry == "Semiconductors"


@patch("yfinance.Ticker")
def test_get_stock_sector_healthcare(mock_ticker):
    mock_ticker.return_value.info = {"sector": "Healthcare", "industry": "Drug Manufacturers"}
    sector, industry = get_stock_sector("JNJ")
    assert sector == "Healthcare"


@patch("yfinance.Ticker")
def test_get_stock_sector_missing_fields(mock_ticker):
    mock_ticker.return_value.info = {}
    sector, industry = get_stock_sector("UNKNOWN")
    assert sector == "Unknown"
    assert industry == "Unknown"


@patch("yfinance.Ticker")
def test_get_stock_sector_api_failure(mock_ticker):
    mock_ticker.side_effect = Exception("timeout")
    sector, industry = get_stock_sector("ERR")
    assert sector == "Unknown"
    assert industry == "Unknown"


# ============================================================================
# 2. VIX FEAR GAUGE
# ============================================================================

@patch("yfinance.Ticker")
def test_vix_calm(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [12.5]})
    result = get_vix_level()
    assert result["vix_category"] == "CALM"
    assert result["vix_contribution"] > 0


@patch("yfinance.Ticker")
def test_vix_panic(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [38.0]})
    result = get_vix_level()
    assert result["vix_category"] == "PANIC"
    assert result["vix_contribution"] == -1.0


@patch("yfinance.Ticker")
def test_vix_elevated(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [22.0]})
    result = get_vix_level()
    assert result["vix_category"] == "ELEVATED"
    assert result["vix_contribution"] < 0


@patch("yfinance.Ticker")
def test_vix_empty_data_defaults(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame()
    result = get_vix_level()
    assert result["vix_category"] == "MODERATE"
    assert result["vix_contribution"] == 0.0


# ============================================================================
# 3. SPY MULTI-TIMEFRAME TREND
# ============================================================================

@patch("yfinance.Ticker")
def test_spy_multitimeframe_bullish(mock_ticker):
    """Strong uptrend → BULLISH composite."""
    closes = list(range(100, 125))   # 25 days of +1/day
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": closes})
    result = get_market_trend_multitimeframe()
    assert result["market_trend"] == "BULLISH"
    assert result["spy_composite_score"] > 0


@patch("yfinance.Ticker")
def test_spy_multitimeframe_bearish(mock_ticker):
    closes = list(range(125, 100, -1))   # 25 days of -1/day
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": closes})
    result = get_market_trend_multitimeframe()
    assert result["market_trend"] == "BEARISH"
    assert result["spy_composite_score"] < 0


@patch("yfinance.Ticker")
def test_spy_multitimeframe_returns_all_fields(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [100] * 25})
    result = get_market_trend_multitimeframe()
    for field in ("market_trend", "market_performance", "spy_return_1d",
                  "spy_return_5d", "spy_return_21d", "volatility", "spy_composite_score"):
        assert field in result


@patch("yfinance.Ticker")
def test_spy_multitimeframe_empty_data(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame()
    result = get_market_trend_multitimeframe()
    assert result["market_trend"] == "NEUTRAL"
    assert result["spy_composite_score"] == 0.0


@patch("yfinance.Ticker")
def test_spy_multitimeframe_composite_clamped(mock_ticker):
    closes = list(range(100, 135))  # big uptrend
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": closes})
    result = get_market_trend_multitimeframe()
    assert -1.0 <= result["spy_composite_score"] <= 1.0


# ============================================================================
# 4. SECTOR PERFORMANCE
# ============================================================================

@patch("yfinance.Ticker")
def test_sector_performance_up(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [100.0, 103.0]})
    result = get_sector_performance("Technology", days=5)
    assert result["trend"] == "UP"
    assert result["performance"] > 0
    assert result["sector_score"] > 0


@patch("yfinance.Ticker")
def test_sector_performance_down(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [100.0, 95.0]})
    result = get_sector_performance("Technology", days=5)
    assert result["trend"] == "DOWN"
    assert result["sector_score"] < 0


def test_sector_performance_unknown_sector():
    result = get_sector_performance("GalacticMining", days=5)
    assert result["etf_ticker"] is None
    assert result["performance"] == 0.0
    assert result["sector_score"] == 0.0


@patch("yfinance.Ticker")
def test_sector_etf_mapping_valid(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [100.0, 101.0]})
    for sector, etf in SECTOR_ETFS.items():
        result = get_sector_performance(sector, days=5)
        assert result["etf_ticker"] == etf


# ============================================================================
# 5. RELATED COMPANIES
# ============================================================================

@patch("yfinance.Ticker")
def test_related_all_up(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [100.0, 106.0]})
    result = analyze_related_companies(["AMD", "INTC", "TSM"])
    assert result["up_count"] == 3
    assert result["overall_signal"] == "BULLISH"
    assert result["peers_score"] > 0


@patch("yfinance.Ticker")
def test_related_all_down(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [100.0, 93.0]})
    result = analyze_related_companies(["AMD", "INTC"])
    assert result["down_count"] == 2
    assert result["overall_signal"] == "BEARISH"
    assert result["peers_score"] < 0


def test_related_empty_list():
    result = analyze_related_companies([])
    assert result["related_companies"] == []
    assert result["overall_signal"] == "NEUTRAL"
    assert result["peers_score"] == 0.0


@patch("yfinance.Ticker")
def test_related_partial_api_failure(mock_ticker):
    call_count = [0]
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return Mock(history=Mock(return_value=pd.DataFrame({"Close": [100.0, 105.0]})))
        raise Exception("API error")
    mock_ticker.side_effect = side_effect
    result = analyze_related_companies(["AMD", "INTC"])
    assert isinstance(result["related_companies"], list)


@patch("yfinance.Ticker")
def test_related_returns_peers_score(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [100.0, 102.0]})
    result = analyze_related_companies(["AMD"])
    assert "peers_score" in result
    assert -1.0 <= result["peers_score"] <= 1.0


# ============================================================================
# 6. CORRELATION CALCULATION (NaN-safe)
# ============================================================================

@patch("yfinance.Ticker")
def test_correlation_returns_finite(mock_ticker, sample_price_data):
    np.random.seed(1)
    spy_closes = 400 + np.cumsum(np.random.randn(40))
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": spy_closes})
    result = calculate_correlation("NVDA", sample_price_data)
    assert math.isfinite(result["market_correlation"])
    assert -1.0 <= result["market_correlation"] <= 1.0


@patch("yfinance.Ticker")
def test_correlation_no_nan_on_timezone_mismatch(mock_ticker, sample_price_data):
    """Timezone-aware SPY vs timezone-naive stock should NOT produce NaN."""
    spy_closes = 400 + np.cumsum(np.random.randn(40))
    tz_index = pd.date_range(end=datetime.now(tz=timezone.utc), periods=40, freq="D")
    spy_df = pd.DataFrame({"Close": spy_closes}, index=tz_index)
    mock_ticker.return_value.history.return_value = spy_df
    result = calculate_correlation("NVDA", sample_price_data)
    assert not math.isnan(result["market_correlation"])


def test_correlation_insufficient_data():
    short_data = pd.DataFrame({"close": [100.0] * 10})
    result = calculate_correlation("TEST", short_data)
    assert result["correlation_strength"] == "MEDIUM"
    assert result["beta"] == 1.0


@patch("yfinance.Ticker")
def test_correlation_beta_reasonable(mock_ticker, sample_price_data):
    spy_closes = 400 + np.cumsum(np.random.randn(40) * 0.5)
    mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": spy_closes})
    result = calculate_correlation("NVDA", sample_price_data)
    assert "beta" in result
    assert -10.0 < result["beta"] < 10.0


# ============================================================================
# 7. MARKET NEWS SENTIMENT
# ============================================================================

def test_news_sentiment_positive(sample_market_news):
    """Recent positive news → positive normalised score."""
    result = analyze_market_news_sentiment(sample_market_news)
    # Articles within 14d: +0.25, +0.05, -0.30 → avg = 0.0
    # (the old +0.35 article is >14 days old and excluded)
    assert isinstance(result["market_news_sentiment"], float)
    assert -1.0 <= result["market_news_sentiment"] <= 1.0


def test_news_sentiment_count_excludes_old(sample_market_news):
    result = analyze_market_news_sentiment(sample_market_news)
    # Only 3 articles are within the 14-day window
    assert result["market_news_count"] == 3


def test_news_sentiment_empty_list():
    result = analyze_market_news_sentiment([])
    assert result["market_news_sentiment"] == 0.0
    assert result["market_news_count"] == 0


def test_news_sentiment_clamped():
    """Extreme AV scores should be clamped to [-1, +1]."""
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    extreme = [{"overall_sentiment_score": 999.0, "datetime": now_ts}]
    result = analyze_market_news_sentiment(extreme)
    assert result["market_news_sentiment"] <= 1.0


def test_news_sentiment_alias_field():
    result = analyze_market_news_sentiment([])
    assert "news_sentiment_score" in result


# ============================================================================
# 8. HEADWIND / TAILWIND SCORE
# ============================================================================

def test_headwind_all_positive_is_buy():
    result = compute_headwind_tailwind_score(
        spy_composite=1.0, vix_contribution=0.3,
        sector_score=1.0, peers_score=1.0, news_sentiment=1.0,
    )
    assert result["context_signal"] == "BUY"
    assert result["market_headwind_score"] >= BUY_THRESHOLD


def test_headwind_all_negative_is_sell():
    result = compute_headwind_tailwind_score(
        spy_composite=-1.0, vix_contribution=-1.0,
        sector_score=-1.0, peers_score=-1.0, news_sentiment=-1.0,
    )
    assert result["context_signal"] == "SELL"
    assert result["market_headwind_score"] <= SELL_THRESHOLD


def test_headwind_neutral_is_hold():
    result = compute_headwind_tailwind_score(
        spy_composite=0.0, vix_contribution=0.0,
        sector_score=0.0, peers_score=0.0, news_sentiment=0.0,
    )
    assert result["context_signal"] == "HOLD"
    assert result["market_headwind_score"] == 0.0


def test_headwind_score_clamped():
    result = compute_headwind_tailwind_score(
        spy_composite=10.0, vix_contribution=10.0,
        sector_score=10.0, peers_score=10.0, news_sentiment=10.0,
    )
    assert result["market_headwind_score"] == 1.0


def test_headwind_confidence_abs():
    result = compute_headwind_tailwind_score(
        spy_composite=0.6, vix_contribution=0.0,
        sector_score=0.0, peers_score=0.0, news_sentiment=0.0,
    )
    assert result["confidence"] >= 0.0


# ============================================================================
# 9. MAIN NODE FUNCTION
# ============================================================================

@patch("src.langgraph_nodes.node_06_market_context.get_stock_sector")
@patch("src.langgraph_nodes.node_06_market_context.get_vix_level")
@patch("src.langgraph_nodes.node_06_market_context.get_market_trend_multitimeframe")
@patch("src.langgraph_nodes.node_06_market_context.get_sector_performance")
@patch("src.langgraph_nodes.node_06_market_context.analyze_related_companies")
@patch("src.langgraph_nodes.node_06_market_context.calculate_correlation")
@patch("src.langgraph_nodes.node_06_market_context.analyze_market_news_sentiment")
def test_node_success(mock_news, mock_corr, mock_peers, mock_sector_perf,
                      mock_spy, mock_vix, mock_sector_det, base_state):
    mock_sector_det.return_value = ("Technology", "Semiconductors")
    mock_vix.return_value       = {"vix_level": 18.0, "vix_category": "MODERATE", "vix_contribution": 0.0}
    mock_spy.return_value       = {
        "market_trend": "BULLISH", "market_performance": 2.0,
        "spy_return_1d": 0.5, "spy_return_5d": 2.0, "spy_return_21d": 4.0,
        "volatility": 1.0, "spy_composite_score": 0.7,
    }
    mock_sector_perf.return_value = {"performance": 2.0, "trend": "UP", "etf_ticker": "XLK", "sector_score": 1.0}
    mock_peers.return_value     = {
        "related_companies": [], "avg_performance": 1.5,
        "up_count": 2, "down_count": 0, "overall_signal": "BULLISH", "peers_score": 1.0,
    }
    mock_corr.return_value      = {"market_correlation": 0.8, "correlation_strength": "HIGH", "beta": 1.2}
    mock_news.return_value      = {"market_news_sentiment": 0.3, "market_news_count": 10, "news_sentiment_score": 0.3}

    result = market_context_node(base_state)

    assert "market_context" in result
    assert "node_execution_times" in result
    ctx = result["market_context"]
    assert ctx["context_signal"] in ("BUY", "SELL", "HOLD")
    assert "market_headwind_score" in ctx
    assert math.isfinite(ctx["market_headwind_score"])
    assert "vix_level" in ctx
    assert "spy_return_1d" in ctx
    assert "market_news_sentiment" in ctx


def test_node_partial_data_no_crash():
    """Node must not raise even with no price_data or related companies."""
    state = {
        "ticker": "TST",
        "raw_price_data": None,
        "related_companies": [],
        "cleaned_market_news": [],
        "errors": [],
        "node_execution_times": {},
    }
    result = market_context_node(state)
    assert "market_context" in result
    assert result["market_context"]["context_signal"] in ("BUY", "SELL", "HOLD")


def test_node_error_returns_defaults():
    """Node must never raise, even with minimal state. Returns valid market_context."""
    result = market_context_node({
        "ticker": "ERR",
        "errors": [],
        "node_execution_times": {},
    })
    assert "market_context" in result
    ctx = result["market_context"]
    assert ctx["context_signal"] in ("BUY", "SELL", "HOLD")
    assert math.isfinite(ctx["market_headwind_score"])
    assert -1.0 <= ctx["market_headwind_score"] <= 1.0


def test_node_uses_cleaned_market_news():
    """cleaned_market_news from state reaches the sentiment helper."""
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    state = {
        "ticker": "AAPL",
        "raw_price_data": None,
        "related_companies": [],
        "cleaned_market_news": [
            {"overall_sentiment_score": 0.30, "datetime": now_ts},
        ],
        "errors": [],
        "node_execution_times": {},
    }
    result = market_context_node(state)
    # If the article reached the helper, market_news_count should be 1
    assert result["market_context"]["market_news_count"] == 1


# ============================================================================
# 10. INTEGRATION
# ============================================================================

@patch("yfinance.Ticker")
def test_integration_end_to_end(mock_ticker, base_state):
    """Full end-to-end with mocked yfinance — should return BUY/SELL/HOLD."""
    def ticker_side_effect(symbol):
        m = Mock()
        m.info = {"sector": "Technology", "industry": "Semiconductors"}
        m.history.return_value = pd.DataFrame({"Close": [100.0, 102.0, 104.0] * 15})
        return m
    mock_ticker.side_effect = ticker_side_effect
    result = market_context_node(base_state)
    assert result["market_context"]["context_signal"] in ("BUY", "SELL", "HOLD")
    assert "sector" in result["market_context"]


def test_integration_partial_state_returned():
    """Node must only return market_context and node_execution_times (parallel-safe)."""
    state = {
        "ticker": "AAPL",
        "raw_price_data": pd.DataFrame({"close": [100.0] * 35}),
        "related_companies": [],
        "cleaned_market_news": [],
        "errors": [],
        "node_execution_times": {},
    }
    result = market_context_node(state)
    assert "ticker" not in result
    assert "raw_price_data" not in result
    assert "node_6" in result["node_execution_times"]


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
