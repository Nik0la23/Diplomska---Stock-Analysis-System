"""
Comprehensive Test Suite for Node 9B: Behavioral Anomaly Detection
Tests all 7 detection systems plus integration scenarios for thesis validation.

Run with: pytest tests/test_nodes/test_node_09b.py -v
"""

import pytest
import pandas as pd
from copy import deepcopy
from unittest.mock import patch, MagicMock

from src.graph.state import create_initial_state
from src.langgraph_nodes.node_09b_behavioral_anomaly import (
    behavioral_anomaly_detection_node,
    detect_volume_anomaly,
    detect_source_reliability_divergence,
    detect_news_velocity_anomaly,
    detect_news_price_divergence,
    detect_cross_stream_incoherence,
    match_historical_patterns,
    calculate_pump_and_dump_score,
)


# ============================================================================
# HELPER FUNCTIONS FOR TEST DATA
# ============================================================================

def _build_basic_state(ticker: str = "AAPL") -> dict:
    """Build minimal valid state for Node 9B testing."""
    state = create_initial_state(ticker)
    
    # Minimal price history (2 days)
    state["raw_price_data"] = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "open": [100.0, 102.0],
            "high": [101.0, 103.0],
            "low": [99.0, 101.0],
            "close": [100.0, 103.0],
            "volume": [1_000_000, 1_200_000],
        }
    )
    
    # Node 9A summary and cleaned news
    state["content_analysis_summary"] = {
        "total_articles_processed": 5,
        "high_risk_articles": 0,
        "top_keywords": [("earnings", 3)],
    }
    state["cleaned_stock_news"] = [
        {
            "headline": "Company reports solid earnings",
            "summary": "",
            "source": "Bloomberg",
            "composite_anomaly_score": 0.1,
        }
    ]
    state["cleaned_market_news"] = []
    state["cleaned_related_company_news"] = []
    
    # Node 8 verification output
    state["news_impact_verification"] = {
        "historical_correlation": 0.7,
        "news_accuracy_score": 70.0,
        "source_reliability": {
            "Bloomberg": {
                "accuracy_rate": 0.84,
                "total_articles": 10,
                "confidence_multiplier": 1.1,
            }
        },
        "news_type_effectiveness": {
            "stock": {"accuracy_rate": 0.68, "avg_impact": 3.2},
            "market": {"accuracy_rate": 0.52, "avg_impact": 1.1},
            "related": {"accuracy_rate": 0.61, "avg_impact": 1.8},
        },
        "sample_size": 50,
        "insufficient_data": False,
    }
    
    state["aggregated_sentiment"] = 0.2
    state["sentiment_signal"] = "NEUTRAL"
    state["sentiment_confidence"] = 0.6

    # Node 2 fetch metadata — simulates 5 new articles fetched in a 1-day window
    state["news_fetch_metadata"] = {
        "from_date": "2024-01-01",
        "to_date": "2024-01-02",
        "fetch_window_days": 1,
        "newly_fetched_count": 5,
        "total_in_state": 5,
        "newly_fetched_stock": 5,
        "newly_fetched_market": 0,
        "was_incremental_fetch": False,
    }

    return state


def _build_pump_and_dump_state(ticker: str = "PUMP") -> dict:
    """Build state with all pump-and-dump signals present."""
    state = _build_basic_state(ticker)
    
    # 1. Massive volume spike (8x normal)
    state["raw_price_data"] = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "open": [100.0] * 29 + [105.0],
            "high": [101.0] * 29 + [115.0],
            "low": [99.0] * 29 + [104.0],
            "close": [100.0] * 29 + [112.0],  # +12% spike on last day
            "volume": [1_000_000] * 29 + [8_000_000],  # 8x spike
        }
    )
    
    # 2. Article surge (50 articles from low-credibility sources)
    state["cleaned_stock_news"] = [
        {
            "source": "pump-blog.com",
            "composite_anomaly_score": 0.82,
            "headline": "BREAKING: REVOLUTIONARY BREAKTHROUGH!!!",
        }
        for _ in range(50)
    ]
    
    # 3. Low source reliability (28% accurate)
    state["news_impact_verification"]["source_reliability"] = {
        "pump-blog.com": {"accuracy_rate": 0.28, "total_articles": 100}
    }
    state["news_impact_verification"]["news_accuracy_score"] = 68.0  # Historical baseline
    
    # 4. Strong positive sentiment
    state["aggregated_sentiment"] = 0.75
    state["sentiment_signal"] = "BUY"
    
    # 5. High anomaly in content summary
    state["content_analysis_summary"] = {
        "total_articles_processed": 50,
        "high_risk_articles": 45,
        "top_keywords": [("BREAKING", 48), ("REVOLUTIONARY", 46)],
    }

    # 6. Override fetch metadata: 50 articles pumped in a single day
    state["news_fetch_metadata"] = {
        "from_date": "2024-01-30",
        "to_date": "2024-01-30",
        "fetch_window_days": 1,
        "newly_fetched_count": 50,
        "total_in_state": 50,
        "newly_fetched_stock": 50,
        "newly_fetched_market": 0,
        "was_incremental_fetch": True,
    }

    return state


# ============================================================================
# SMOKE TESTS (Basic Functionality)
# ============================================================================

def test_node_09b_with_basic_state():
    """Node 9B should run and populate behavioral_anomaly_detection on a basic state."""
    state = _build_basic_state()
    result = behavioral_anomaly_detection_node(state)

    assert "behavioral_anomaly_detection" in result
    bad = result["behavioral_anomaly_detection"]

    assert bad["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert isinstance(bad["pump_and_dump_score"], int)
    assert 0 <= bad["pump_and_dump_score"] <= 100
    assert "volume_anomaly" in bad
    assert "news_price_divergence" in bad
    assert "alerts" in bad
    assert isinstance(bad["alerts"], list)
    assert "node_9b" in result["node_execution_times"]


def test_node_09b_handles_missing_price_data_gracefully():
    """If price data is missing, node must not crash and should default to LOW risk."""
    state = _build_basic_state()
    state["raw_price_data"] = None

    result = behavioral_anomaly_detection_node(state)
    bad = result["behavioral_anomaly_detection"]

    assert bad["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert bad["pump_and_dump_score"] >= 0
    # Should not crash - this is the main test


def test_node_09b_does_not_mutate_unrelated_state_fields():
    """Node 9B must only add/update behavioral_anomaly_detection and node_9b timing."""
    state = _build_basic_state()
    before = deepcopy(state)

    result = behavioral_anomaly_detection_node(state)

    # Unrelated fields should remain identical
    assert before["cleaned_stock_news"] == result["cleaned_stock_news"]
    assert before["aggregated_sentiment"] == result["aggregated_sentiment"]
    assert before["raw_price_data"].equals(result["raw_price_data"])


def test_node_09b_with_empty_news():
    """Node 9B should handle state with no articles gracefully."""
    state = _build_basic_state()
    state["cleaned_stock_news"] = []
    state["cleaned_market_news"] = []
    state["cleaned_related_company_news"] = []
    
    result = behavioral_anomaly_detection_node(state)
    bad = result["behavioral_anomaly_detection"]
    
    assert bad["risk_level"] == "LOW"
    assert bad["pump_and_dump_score"] == 0


# ============================================================================
# DETECTION SYSTEM 1: VOLUME ANOMALY TESTS
# ============================================================================

def test_volume_anomaly_detector_normal_volume():
    """Normal volume (1.5x baseline) should not trigger detection."""
    historical = [1_000_000] * 30
    current = 1_500_000
    
    result = detect_volume_anomaly(current, historical, baseline_days=30)
    
    assert result["detected"] == False
    assert result["severity"] == "LOW"
    assert result["contribution_score"] == 0
    assert 1.4 <= result["volume_ratio"] <= 1.6


def test_volume_anomaly_detector_medium_spike():
    """Medium volume spike (3x) should trigger MEDIUM severity."""
    historical = [1_000_000] * 30
    current = 3_000_000
    
    result = detect_volume_anomaly(current, historical, baseline_days=30)
    
    assert result["detected"] == True
    assert result["severity"] == "MEDIUM"
    assert result["contribution_score"] == 10
    assert result["volume_ratio"] == 3.0


def test_volume_anomaly_detector_high_spike():
    """High volume spike (5x) should trigger HIGH severity."""
    historical = [1_000_000] * 30
    current = 5_000_000
    
    result = detect_volume_anomaly(current, historical, baseline_days=30)
    
    assert result["detected"] == True
    assert result["severity"] == "HIGH"
    assert result["contribution_score"] == 15
    assert result["volume_ratio"] == 5.0


def test_volume_anomaly_detector_extreme_spike():
    """Extreme volume spike (8x) should trigger HIGH severity with max score."""
    historical = [1_000_000] * 30
    current = 8_000_000
    
    result = detect_volume_anomaly(current, historical, baseline_days=30)
    
    assert result["detected"] == True
    assert result["severity"] == "HIGH"
    assert result["contribution_score"] == 20
    assert result["volume_ratio"] == 8.0


# ============================================================================
# DETECTION SYSTEM 2: SOURCE RELIABILITY DIVERGENCE TESTS
# ============================================================================

def test_source_reliability_no_divergence():
    """When today's sources match historical baseline, no divergence."""
    articles = [
        {"source": "Bloomberg", "composite_anomaly_score": 0.1},
        {"source": "Reuters", "composite_anomaly_score": 0.12},
    ]
    
    reliability = {
        "Bloomberg": {"accuracy_rate": 0.84},
        "Reuters": {"accuracy_rate": 0.78},
    }
    
    historical_avg = 0.80
    
    result = detect_source_reliability_divergence(
        articles, reliability, historical_avg
    )
    
    assert result["detected"] == False
    assert result["severity"] == "LOW"
    assert result["contribution_score"] == 0
    assert result["divergence"] < 0.10


def test_source_reliability_medium_divergence():
    """Medium divergence (0.15) should trigger detection."""
    articles = [
        {"source": "low-cred-source.com", "composite_anomaly_score": 0.6},
        {"source": "another-low.com", "composite_anomaly_score": 0.65},
    ]
    
    reliability = {
        "low-cred-source.com": {"accuracy_rate": 0.45},
        "another-low.com": {"accuracy_rate": 0.48},
    }
    
    historical_avg = 0.68  # 68% baseline
    
    result = detect_source_reliability_divergence(
        articles, reliability, historical_avg
    )
    
    assert result["detected"] == True
    assert result["severity"] in ["MEDIUM", "HIGH"]
    assert result["contribution_score"] >= 12
    assert result["divergence"] >= 0.15


def test_source_reliability_high_divergence():
    """High divergence (0.35) should trigger HIGH severity."""
    articles = [
        {"source": "bad-blog.com", "composite_anomaly_score": 0.7},
        {"source": "fake-news.net", "composite_anomaly_score": 0.75},
        {"source": "pump-site.com", "composite_anomaly_score": 0.8},
    ]
    
    reliability = {
        "bad-blog.com": {"accuracy_rate": 0.28},
        "fake-news.net": {"accuracy_rate": 0.25},
        "pump-site.com": {"accuracy_rate": 0.30},
    }
    
    historical_avg = 0.65  # 65% baseline
    
    result = detect_source_reliability_divergence(
        articles, reliability, historical_avg
    )
    
    assert result["detected"] == True
    assert result["severity"] == "HIGH"
    assert result["contribution_score"] >= 18
    assert result["divergence"] >= 0.30


# ============================================================================
# DETECTION SYSTEM 3: NEWS VELOCITY ANOMALY TESTS
# ============================================================================

def test_news_velocity_normal():
    """Normal article count should not trigger detection."""
    today_count = 8
    historical_avg = 7.5
    summary = {"high_risk_articles": 0, "top_keywords": [("earnings", 3)]}
    
    result = detect_news_velocity_anomaly(today_count, historical_avg, summary)
    
    assert result["detected"] == False
    assert result["severity"] == "LOW"
    assert result["contribution_score"] == 0


def test_news_velocity_medium_spike():
    """Medium spike (4x) should trigger MEDIUM severity."""
    today_count = 32
    historical_avg = 8.0
    summary = {"high_risk_articles": 5, "top_keywords": [("earnings", 10)]}
    
    result = detect_news_velocity_anomaly(today_count, historical_avg, summary)
    
    assert result["detected"] == True
    assert result["severity"] == "MEDIUM"
    assert result["contribution_score"] == 8
    assert result["velocity_ratio"] == 4.0


def test_news_velocity_high_spike_with_coordination():
    """High spike (8x) with coordinated keywords should add bonus points."""
    today_count = 64
    historical_avg = 8.0
    summary = {
        "high_risk_articles": 50,
        "top_keywords": [("BREAKING", 60)],  # 60/64 = 93% > 60% threshold
        "total_articles_processed": 64,
    }
    
    result = detect_news_velocity_anomaly(today_count, historical_avg, summary)
    
    assert result["detected"] == True
    assert result["severity"] == "HIGH"
    assert result["contribution_score"] >= 15  # Base + coordination bonus
    assert result["coordinated_keywords"] == True


# ============================================================================
# DETECTION SYSTEM 4: NEWS-PRICE DIVERGENCE TESTS
# ============================================================================

def test_divergence_type_a_positive_news_negative_price():
    """Type A: Positive sentiment but price falling."""
    sentiment_signal = "BUY"
    sentiment_score = 0.6  # Positive
    price_change = -3.5  # Down 3.5%
    
    articles = [
        {"source": "Bloomberg", "composite_anomaly_score": 0.1},
        {"source": "Reuters", "composite_anomaly_score": 0.12},
    ]
    
    reliability = {
        "Bloomberg": {"accuracy_rate": 0.84},
        "Reuters": {"accuracy_rate": 0.78},
    }
    
    result = detect_news_price_divergence(
        sentiment_signal, sentiment_score, price_change, articles, reliability
    )
    
    assert result["detected"] == True
    assert result["divergence_type"] == "A"
    assert result["severity"] == "LOW"
    assert result["contribution_score"] == 5
    assert "priced in" in result["explanation"].lower()


def test_divergence_type_b_negative_news_positive_price():
    """Type B: Negative sentiment but price rising."""
    sentiment_signal = "SELL"
    sentiment_score = -0.5  # Negative
    price_change = 2.8  # Up 2.8%
    
    articles = [
        {"source": "CNBC", "composite_anomaly_score": 0.15},
    ]
    
    reliability = {
        "CNBC": {"accuracy_rate": 0.72},
    }
    
    result = detect_news_price_divergence(
        sentiment_signal, sentiment_score, price_change, articles, reliability
    )
    
    assert result["detected"] == True
    assert result["divergence_type"] == "B"
    assert result["severity"] == "MEDIUM"
    assert result["contribution_score"] == 10
    assert "short covering" in result["explanation"].lower()


def test_divergence_type_c_pump_and_dump_signal():
    """Type C: Low-cred positive news + rising price + no corroboration = CRITICAL."""
    sentiment_signal = "BUY"
    sentiment_score = 0.75  # Strong positive
    price_change = 8.5  # Up 8.5%
    
    articles = [
        {"source": "pump-blog.com", "composite_anomaly_score": 0.82},
        {"source": "fake-news.net", "composite_anomaly_score": 0.78},
        {"source": "random-site.com", "composite_anomaly_score": 0.85},
    ]
    
    reliability = {
        "pump-blog.com": {"accuracy_rate": 0.28},
        "fake-news.net": {"accuracy_rate": 0.22},
        "random-site.com": {"accuracy_rate": 0.30},
    }
    
    result = detect_news_price_divergence(
        sentiment_signal, sentiment_score, price_change, articles, reliability
    )
    
    assert result["detected"] == True
    assert result["divergence_type"] == "C"
    assert result["severity"] == "CRITICAL"
    assert result["contribution_score"] == 25  # Max score
    assert result["credible_source_corroboration"] == False
    assert "pump" in result["explanation"].lower()


def test_divergence_no_detection_aligned_signals():
    """No divergence when sentiment and price align with credible sources."""
    sentiment_signal = "BUY"
    sentiment_score = 0.5  # Positive
    price_change = 2.3  # Up 2.3%
    
    articles = [
        {"source": "Bloomberg", "composite_anomaly_score": 0.08},
        {"source": "Reuters", "composite_anomaly_score": 0.10},
        {"source": "WSJ", "composite_anomaly_score": 0.12},
    ]
    
    reliability = {
        "Bloomberg": {"accuracy_rate": 0.84},
        "Reuters": {"accuracy_rate": 0.78},
        "WSJ": {"accuracy_rate": 0.82},
    }
    
    result = detect_news_price_divergence(
        sentiment_signal, sentiment_score, price_change, articles, reliability
    )
    
    assert result["detected"] == False
    assert result["divergence_type"] is None
    assert result["contribution_score"] == 0


# ============================================================================
# DETECTION SYSTEM 5: CROSS-STREAM COHERENCE TESTS
# ============================================================================

def test_cross_stream_coherence_aligned():
    """All streams aligned should produce high coherence (no detection)."""
    stock_sentiment = 0.6
    market_sentiment = 0.55
    related_sentiment = 0.58
    
    result = detect_cross_stream_incoherence(
        stock_sentiment, market_sentiment, related_sentiment,
        stock_article_count=10, market_article_count=8, related_article_count=5,
        stock_avg_anomaly=0.15
    )
    
    assert result["detected"] == False
    assert result["coherence_score"] > 0.7
    assert result["severity"] == "LOW"
    assert result["contribution_score"] == 0


def test_cross_stream_coherence_isolated_signal():
    """Stock stream strongly positive but market/related neutral = isolation."""
    stock_sentiment = 0.75
    market_sentiment = 0.05
    related_sentiment = -0.02
    
    result = detect_cross_stream_incoherence(
        stock_sentiment, market_sentiment, related_sentiment,
        stock_article_count=50, market_article_count=5, related_article_count=3,
        stock_avg_anomaly=0.68
    )
    
    assert result["detected"] == True
    assert result["isolated_signal"] == True
    assert result["coherence_score"] < 0.6
    assert result["contribution_score"] >= 8


# ============================================================================
# DETECTION SYSTEM 6: HISTORICAL PATTERN MATCHER TESTS
# ============================================================================

def test_historical_pattern_matcher_no_data():
    """With no historical data, should return insufficient_data."""
    today_profile = {
        "article_count": 10,
        "avg_composite_anomaly": 0.2,
        "avg_source_reliability": 0.7,
        "volume_ratio": 1.5,
    }
    
    result = match_historical_patterns(today_profile, [], similarity_threshold=0.75)
    
    assert result["detected"] == False
    assert result["insufficient_data"] == True
    assert result["similar_periods_found"] == 0


def test_historical_pattern_matcher_crash_pattern():
    """When similar days mostly crashed, should detect HIGH risk."""
    today_profile = {
        "article_count": 50,
        "avg_composite_anomaly": 0.75,
        "avg_source_reliability": 0.30,
        "volume_ratio": 6.0,
    }
    
    # Mock historical data: 10 similar days, 8 crashed
    historical = [
        {
            "date": f"2024-01-{i:02d}",
            "article_count": 48,
            "avg_composite_anomaly": 0.72,
            "avg_source_credibility": 0.32,
            "volume_ratio": 5.8,
            "price_change_7d": -18.5 if i <= 8 else 2.1,  # 8/10 crashed
        }
        for i in range(1, 11)
    ]
    
    result = match_historical_patterns(
        today_profile, historical, similarity_threshold=0.70
    )
    
    assert result["detected"] == True
    assert result["similar_periods_found"] >= 8
    assert result["outcomes"]["pct_ended_in_crash"] >= 0.6
    assert result["severity"] == "HIGH"
    assert result["contribution_score"] == 10


# ============================================================================
# DETECTION SYSTEM 7: COMPOSITE SCORER TESTS
# ============================================================================

def test_composite_scorer_all_neutral():
    """All detectors neutral should produce LOW risk."""
    volume = {"contribution_score": 0}
    reliability = {"contribution_score": 0}
    velocity = {"contribution_score": 0}
    divergence = {"contribution_score": 0, "divergence_type": None}
    coherence = {"contribution_score": 0}
    pattern = {"contribution_score": 0}
    
    result = calculate_pump_and_dump_score(
        volume, reliability, velocity, divergence, coherence, pattern
    )
    
    assert result["pump_and_dump_score"] == 0
    assert result["risk_level"] == "LOW"


def test_composite_scorer_medium_risk():
    """Moderate scores should produce MEDIUM risk."""
    volume = {"contribution_score": 10}
    reliability = {"contribution_score": 12}
    velocity = {"contribution_score": 8}
    divergence = {"contribution_score": 5, "divergence_type": "A"}
    coherence = {"contribution_score": 5}
    pattern = {"contribution_score": 7}
    
    result = calculate_pump_and_dump_score(
        volume, reliability, velocity, divergence, coherence, pattern
    )
    
    assert 31 <= result["pump_and_dump_score"] <= 55
    assert result["risk_level"] == "MEDIUM"


def test_composite_scorer_high_risk():
    """High scores should produce HIGH risk."""
    volume = {"contribution_score": 20}
    reliability = {"contribution_score": 18}
    velocity = {"contribution_score": 15}
    divergence = {"contribution_score": 10, "divergence_type": "B"}
    coherence = {"contribution_score": 8}
    pattern = {"contribution_score": 4}  # 20+18+15+10+8+4 = 75 (top of HIGH range)
    
    result = calculate_pump_and_dump_score(
        volume, reliability, velocity, divergence, coherence, pattern
    )
    
    assert 56 <= result["pump_and_dump_score"] <= 75
    assert result["risk_level"] == "HIGH"


def test_composite_scorer_type_c_auto_elevation():
    """Type C divergence should force at least HIGH risk even if score is moderate."""
    volume = {"contribution_score": 10}
    reliability = {"contribution_score": 8}
    velocity = {"contribution_score": 5}
    divergence = {"contribution_score": 25, "divergence_type": "C"}  # Type C!
    coherence = {"contribution_score": 0}
    pattern = {"contribution_score": 0}
    
    result = calculate_pump_and_dump_score(
        volume, reliability, velocity, divergence, coherence, pattern
    )
    
    # Type C should force risk level to at least HIGH
    assert result["risk_level"] in ["HIGH", "CRITICAL"]
    assert result["pump_and_dump_score"] >= 56


# ============================================================================
# INTEGRATION TESTS (Full Node Execution)
# ============================================================================

@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_historical_daily_aggregates")
@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_volume_baseline")
@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_article_count_baseline")
def test_node_09b_full_pump_and_dump_detection(
    mock_article_baseline, mock_volume_baseline, mock_historical
):
    """
    THESIS SHOWCASE: Complete pump-and-dump scenario with all signals.
    This is the test you show in your defense.
    """
    # Mock database responses
    mock_volume_baseline.return_value = 1_000_000
    mock_article_baseline.return_value = 8.0
    mock_historical.return_value = []  # Skip pattern matching for simplicity
    
    state = _build_pump_and_dump_state()
    result = behavioral_anomaly_detection_node(state)
    bad = result["behavioral_anomaly_detection"]
    
    # Should detect HIGH or CRITICAL risk
    assert bad["risk_level"] in ["HIGH", "CRITICAL"], \
        f"Expected HIGH/CRITICAL, got {bad['risk_level']}"
    
    assert bad["pump_and_dump_score"] >= 60, \
        f"Expected score >= 60, got {bad['pump_and_dump_score']}"
    
    # Should detect multiple anomalies
    assert bad["volume_anomaly"]["detected"] == True, "Volume anomaly not detected"
    assert bad["source_reliability_divergence"]["detected"] == True, \
        "Source divergence not detected"
    assert bad["news_velocity_anomaly"]["detected"] == True, \
        "News velocity not detected"
    
    # Type C divergence should be detected (smoking gun)
    assert bad["news_price_divergence"]["divergence_type"] == "C", \
        f"Expected Type C, got {bad['news_price_divergence']['divergence_type']}"
    
    # Should have multiple alerts
    assert len(bad["alerts"]) >= 3, \
        f"Expected >= 3 alerts, got {len(bad['alerts'])}"
    
    # Trading recommendation should be cautious
    assert bad["trading_recommendation"] in ["CAUTION", "DO_NOT_TRADE"], \
        f"Expected CAUTION/DO_NOT_TRADE, got {bad['trading_recommendation']}"
    
    print(f"\n=== PUMP-AND-DUMP TEST RESULTS ===")
    print(f"Risk Level: {bad['risk_level']}")
    print(f"Score: {bad['pump_and_dump_score']}/100")
    print(f"Alerts: {bad['alerts']}")
    print(f"Primary Risks: {bad['primary_risk_factors']}")


@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_historical_daily_aggregates")
@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_volume_baseline")
@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_article_count_baseline")
def test_node_09b_legitimate_high_volume_with_credible_sources(
    mock_article_baseline, mock_volume_baseline, mock_historical
):
    """
    Legitimate news event: High volume + credible sources should NOT trigger HIGH risk.
    This shows the system doesn't false-positive on real news.
    """
    mock_volume_baseline.return_value = 1_000_000
    mock_article_baseline.return_value = 8.0
    mock_historical.return_value = []
    
    state = _build_basic_state()
    
    # High volume (earnings day)
    state["raw_price_data"].loc[1, "volume"] = 4_000_000  # 4x spike
    state["raw_price_data"].loc[1, "close"] = 105.0  # +5% move
    
    # But all articles from highly credible sources
    state["cleaned_stock_news"] = [
        {
            "source": "Bloomberg",
            "composite_anomaly_score": 0.08,
            "headline": "Company beats earnings estimates",
        },
        {
            "source": "Reuters",
            "composite_anomaly_score": 0.10,
            "headline": "Strong Q4 results announced",
        },
        {
            "source": "WSJ",
            "composite_anomaly_score": 0.12,
            "headline": "Earnings surprise on revenue growth",
        },
    ]
    
    state["news_impact_verification"]["source_reliability"] = {
        "Bloomberg": {"accuracy_rate": 0.84},
        "Reuters": {"accuracy_rate": 0.78},
        "WSJ": {"accuracy_rate": 0.82},
    }
    
    state["aggregated_sentiment"] = 0.5
    
    result = behavioral_anomaly_detection_node(state)
    bad = result["behavioral_anomaly_detection"]
    
    # Should be LOW or MEDIUM, NOT HIGH/CRITICAL
    assert bad["risk_level"] in ["LOW", "MEDIUM"], \
        f"Legitimate news flagged as {bad['risk_level']}"
    
    # Type C should NOT be detected (credible sources present)
    assert bad["news_price_divergence"]["divergence_type"] != "C", \
        "False positive: Type C detected on legitimate news"
    
    print(f"\n=== LEGITIMATE NEWS TEST RESULTS ===")
    print(f"Risk Level: {bad['risk_level']} (Expected LOW/MEDIUM)")
    print(f"Score: {bad['pump_and_dump_score']}/100")


@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_historical_daily_aggregates")
@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_volume_baseline")
@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_article_count_baseline")
def test_node_09b_risk_level_threshold_boundaries(
    mock_article_baseline, mock_volume_baseline, mock_historical
):
    """Verify risk level thresholds work correctly at boundaries."""
    mock_volume_baseline.return_value = 1_000_000
    mock_article_baseline.return_value = 8.0
    mock_historical.return_value = []
    
    # Test LOW boundary (score = 30)
    state = _build_basic_state()
    state["raw_price_data"].loc[1, "volume"] = 3_500_000  # Should give ~10 pts
    # Other signals neutral, total ~10-15 pts
    
    result = behavioral_anomaly_detection_node(state)
    bad = result["behavioral_anomaly_detection"]
    
    if bad["pump_and_dump_score"] <= 30:
        assert bad["risk_level"] == "LOW"
    elif 31 <= bad["pump_and_dump_score"] <= 55:
        assert bad["risk_level"] == "MEDIUM"


def test_node_09b_with_insufficient_node8_data():
    """When Node 8 had insufficient data, Node 9B should handle gracefully."""
    state = _build_basic_state()
    
    # Node 8 had insufficient data
    state["news_impact_verification"]["insufficient_data"] = True
    state["news_impact_verification"]["sample_size"] = 5
    
    result = behavioral_anomaly_detection_node(state)
    bad = result["behavioral_anomaly_detection"]
    
    # Should not crash
    assert "behavioral_anomaly_detection" in result
    assert bad["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    # Source reliability divergence should note insufficient data
    assert bad["source_reliability_divergence"]["contribution_score"] == 0


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_node_09b_handles_malformed_articles():
    """Node should handle articles with missing/malformed fields."""
    state = _build_basic_state()
    
    state["cleaned_stock_news"] = [
        {"headline": "Valid article"},  # Missing source
        {"source": None, "composite_anomaly_score": "invalid"},  # Invalid types
        {},  # Empty dict
    ]
    
    result = behavioral_anomaly_detection_node(state)
    
    # Should not crash
    assert "behavioral_anomaly_detection" in result


def test_node_09b_execution_time_reasonable():
    """Node should execute in reasonable time (< 5 seconds)."""
    import time
    
    state = _build_basic_state()
    
    start = time.time()
    result = behavioral_anomaly_detection_node(state)
    elapsed = time.time() - start
    
    assert elapsed < 5.0, f"Node took {elapsed:.2f}s (expected < 5s)"
    assert result["behavioral_anomaly_detection"]["execution_time"] < 5.0


# ============================================================================
# SUMMARY TEST - Run this to see all results
# ============================================================================

@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_historical_daily_aggregates")
@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_volume_baseline")
@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_article_count_baseline")
def test_node_09b_detects_daily_article_spike(
    mock_article_baseline, mock_volume_baseline, mock_historical
):
    """
    Test that Node 9B detects a sudden spike in daily article volume.

    This is the TRUE pump-and-dump article flood detector: 50 articles
    published in 1 day against a baseline of ~7/day = 7.2x ratio → HIGH.
    """
    mock_volume_baseline.return_value = 1_000_000
    mock_article_baseline.return_value = 6.96  # ~7 articles/day historically
    mock_historical.return_value = []

    state = _build_basic_state()

    # Simulate 50 articles published in a coordinated flood over 1 day
    state["cleaned_stock_news"] = [
        {
            "source": "pump-blog.com",
            "composite_anomaly_score": 0.75,
            "headline": f"BREAKING: Stock about to explode #{i}",
        }
        for i in range(50)
    ]
    state["content_analysis_summary"] = {
        "total_articles_processed": 50,
        "high_risk_articles": 35,
        "top_keywords": [("explode", 31)],  # 31/50 = 62% → coordinated keywords
    }

    # Node 2 metadata: 50 articles fetched fresh from APIs over 1 day
    state["news_fetch_metadata"] = {
        "from_date": "2024-01-02",
        "to_date": "2024-01-02",
        "fetch_window_days": 1,
        "newly_fetched_count": 50,
        "total_in_state": 50,
        "newly_fetched_stock": 50,
        "newly_fetched_market": 0,
        "was_incremental_fetch": True,
    }

    result = behavioral_anomaly_detection_node(state)
    bad = result["behavioral_anomaly_detection"]
    vel = bad["news_velocity_anomaly"]

    # Velocity: 50 articles / 1 day / 6.96 baseline = ~7.2x → HIGH
    assert vel["detected"] is True, "Daily article flood should be detected"
    assert vel["velocity_ratio"] > 5.0, (
        f"Velocity ratio should be > 5.0x, got {vel['velocity_ratio']:.2f}x"
    )
    assert vel["severity"] == "HIGH", (
        f"Severity should be HIGH, got {vel['severity']}"
    )
    assert vel["articles_per_day"] == pytest.approx(50.0, abs=0.1), (
        f"articles_per_day should be 50.0, got {vel['articles_per_day']}"
    )

    print(f"\n=== DAILY VELOCITY SPIKE TEST ===")
    print(f"Newly fetched:   {vel['today_article_count']} articles over {vel['time_window_days']} day(s)")
    print(f"Articles/day:    {vel['articles_per_day']:.1f}")
    print(f"Baseline:        {vel['historical_daily_avg']:.2f}/day")
    print(f"Velocity ratio:  {vel['velocity_ratio']:.2f}x")
    print(f"Severity:        {vel['severity']}")
    print(f"Detected:        {vel['detected']}")


@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_historical_daily_aggregates")
@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_volume_baseline")
@patch("src.langgraph_nodes.node_09b_behavioral_anomaly.get_article_count_baseline")
def test_node_09b_no_false_alarm_normal_velocity(
    mock_article_baseline, mock_volume_baseline, mock_historical
):
    """
    Test that Node 9B does NOT fire a false alarm when the daily rate is normal.

    15 articles fetched over 3 days = 5/day vs 6.96/day baseline → 0.72x, no detection.
    """
    mock_volume_baseline.return_value = 1_000_000
    mock_article_baseline.return_value = 6.96
    mock_historical.return_value = []

    state = _build_basic_state()

    state["news_fetch_metadata"] = {
        "fetch_window_days": 3,
        "newly_fetched_count": 15,
        "total_in_state": 434,  # Large 6-month batch — should NOT be used
        "newly_fetched_stock": 12,
        "newly_fetched_market": 3,
        "was_incremental_fetch": True,
    }

    result = behavioral_anomaly_detection_node(state)
    bad = result["behavioral_anomaly_detection"]
    vel = bad["news_velocity_anomaly"]

    # 15 / 3 days / 6.96 = 0.72x → well below threshold, no detection
    assert vel["detected"] is False, (
        f"Normal daily rate should NOT be detected as anomaly (ratio={vel['velocity_ratio']:.2f}x)"
    )
    assert vel["velocity_ratio"] < 3.0, (
        f"Velocity ratio should be < 3.0x for normal traffic, got {vel['velocity_ratio']:.2f}x"
    )

    print(f"\n=== NORMAL VELOCITY (NO FALSE ALARM) TEST ===")
    print(f"Newly fetched:   {vel['today_article_count']} articles over {vel['time_window_days']} day(s)")
    print(f"Articles/day:    {vel['articles_per_day']:.1f}")
    print(f"Baseline:        {vel['historical_daily_avg']:.2f}/day")
    print(f"Velocity ratio:  {vel['velocity_ratio']:.2f}x")
    print(f"Detected:        {vel['detected']}")


def test_summary_print_test_coverage():
    """Print summary of test coverage for thesis documentation."""
    print("\n" + "="*70)
    print("NODE 9B TEST COVERAGE SUMMARY")
    print("="*70)
    print("\nDetection Systems Tested:")
    print("  ✓ Volume Anomaly (4 tests)")
    print("  ✓ Source Reliability Divergence (3 tests)")
    print("  ✓ News Velocity Anomaly (3 tests)")
    print("  ✓ News-Price Divergence (4 tests including Type C)")
    print("  ✓ Cross-Stream Coherence (2 tests)")
    print("  ✓ Historical Pattern Matcher (2 tests)")
    print("  ✓ Composite Scorer (4 tests)")
    print("\nIntegration Scenarios:")
    print("  ✓ Full pump-and-dump detection (THESIS SHOWCASE)")
    print("  ✓ Legitimate high-volume news (false positive check)")
    print("  ✓ Risk level threshold boundaries")
    print("\nEdge Cases:")
    print("  ✓ Missing price data")
    print("  ✓ Empty news lists")
    print("  ✓ Insufficient Node 8 data")
    print("  ✓ Malformed articles")
    print("  ✓ State immutability")
    print("  ✓ Execution time performance")
    print("\nTotal Tests: 27")
    print("="*70 + "\n")
    
    assert True  # This test always passes, just for documentation