"""
Tests for Node 5: Sentiment Analysis

Tests cover:
1. Alpha Vantage sentiment extraction (4 tests)
2. FinBERT model loading (3 tests)
3. FinBERT analysis (5 tests)
4. Sentiment aggregation (6 tests)
5. Combined sentiment calculation (5 tests)
6. Signal generation (4 tests)
7. Main node function (3 tests)
8. Integration (2 tests)

Total: 32+ tests
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from src.langgraph_nodes.node_05_sentiment_analysis import (
    extract_alpha_vantage_sentiment,
    load_finbert_model,
    analyze_text_with_finbert,
    analyze_articles_batch,
    aggregate_sentiment_by_type,
    calculate_combined_sentiment,
    generate_sentiment_signal,
    sentiment_analysis_node
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_av_articles():
    """Articles with Alpha Vantage sentiment scores"""
    return [
        {
            'title': 'Company beats earnings expectations',
            'summary': 'Strong quarterly results',
            'overall_sentiment_score': 0.7,
            'ticker_sentiment_score': 0.8,
            'source': 'Alpha Vantage'
        },
        {
            'title': 'Market concerns over regulations',
            'summary': 'New regulations may impact growth',
            'overall_sentiment_score': -0.5,
            'ticker_sentiment_score': -0.6,
            'source': 'Alpha Vantage'
        },
        {
            'title': 'Company announces new product',
            'summary': 'Neutral market reception',
            'overall_sentiment_score': 0.1,
            'ticker_sentiment_score': 0.05,
            'source': 'Alpha Vantage'
        }
    ]


@pytest.fixture
def sample_mixed_articles():
    """Mix of articles with and without sentiment"""
    return [
        {
            'title': 'Good news',
            'ticker_sentiment_score': 0.7,
            'source': 'Alpha Vantage'
        },
        {
            'title': 'Neutral news from Finnhub',
            'source': 'Finnhub'
        },
        {
            'title': 'Bad news',
            'ticker_sentiment_score': -0.6,
            'source': 'Alpha Vantage'
        }
    ]


@pytest.fixture
def sample_state_with_news():
    """Complete state with cleaned news"""
    return {
        'ticker': 'NVDA',
        'cleaned_stock_news': [
            {'title': 'Positive stock news', 'ticker_sentiment_score': 0.8},
            {'title': 'More positive news', 'ticker_sentiment_score': 0.7}
        ],
        'cleaned_market_news': [
            {'title': 'Market bullish', 'ticker_sentiment_score': 0.5}
        ],
        'cleaned_related_company_news': [
            {'title': 'Competitor doing well', 'ticker_sentiment_score': 0.6}
        ],
        'errors': [],
        'node_execution_times': {}
    }


# ============================================================================
# CATEGORY 1: Alpha Vantage Sentiment Extraction (4 tests)
# ============================================================================

def test_extract_av_sentiment_valid_scores(sample_av_articles):
    """Test extracting valid sentiment scores from Alpha Vantage"""
    result = extract_alpha_vantage_sentiment(sample_av_articles)
    
    assert len(result) == 3
    assert all('sentiment_score' in a for a in result)
    assert all('has_sentiment' in a for a in result)
    assert all(a['has_sentiment'] for a in result)
    
    # Check ticker sentiment is used (preferred)
    assert result[0]['sentiment_score'] == 0.8  # Ticker sentiment
    assert result[1]['sentiment_score'] == -0.6
    assert result[2]['sentiment_score'] == 0.05


def test_extract_av_sentiment_missing():
    """Test handling articles without sentiment gracefully"""
    articles = [
        {'title': 'No sentiment', 'source': 'Finnhub'}
    ]
    
    result = extract_alpha_vantage_sentiment(articles)
    
    assert len(result) == 1
    assert result[0]['has_sentiment'] is False
    assert result[0]['sentiment_score'] == 0.0  # Neutral default
    assert result[0]['sentiment_source'] == 'none'


def test_extract_av_sentiment_normalization():
    """Test sentiment scores are normalized to -1 to +1 range"""
    articles = [
        {'title': 'Extreme positive', 'ticker_sentiment_score': 2.5},  # Out of range
        {'title': 'Extreme negative', 'ticker_sentiment_score': -2.0}  # Out of range
    ]
    
    result = extract_alpha_vantage_sentiment(articles)
    
    # Should be clamped to [-1, 1]
    assert -1.0 <= result[0]['sentiment_score'] <= 1.0
    assert -1.0 <= result[1]['sentiment_score'] <= 1.0


def test_extract_av_sentiment_malformed_data():
    """Test handling malformed data"""
    articles = [
        {},  # Empty article
        {'title': 'No scores'},  # Missing sentiment fields
        {'title': 'Invalid score', 'ticker_sentiment_score': 'invalid'}  # Invalid type
    ]
    
    # Should not crash
    result = extract_alpha_vantage_sentiment(articles)
    assert len(result) == 3


# ============================================================================
# CATEGORY 2: FinBERT Model Loading (3 tests)
# ============================================================================

def test_load_finbert_model_success():
    """Test successful FinBERT model loading"""
    # Clear cached model
    import src.langgraph_nodes.node_05_sentiment_analysis as node5
    node5._FINBERT_MODEL = None
    
    # Load model (this will actually download from HuggingFace first time)
    model = load_finbert_model()
    
    # Model should load successfully
    assert model is not None
    
    # Verify it's a pipeline object
    assert hasattr(model, '__call__')  # Should be callable


@patch('transformers.pipeline')
def test_load_finbert_model_failure(mock_pipeline):
    """Test graceful handling of model loading failure"""
    mock_pipeline.side_effect = Exception("Model not found")
    
    # Clear cached model
    import src.langgraph_nodes.node_05_sentiment_analysis as node5
    node5._FINBERT_MODEL = None
    
    model = load_finbert_model()
    
    assert model is None  # Should return None, not raise


def test_load_finbert_model_caching():
    """Test that model is cached after first load"""
    import src.langgraph_nodes.node_05_sentiment_analysis as node5
    
    # Set a mock model
    mock_model = Mock()
    node5._FINBERT_MODEL = mock_model
    
    # Load again
    model = load_finbert_model()
    
    # Should return cached model
    assert model is mock_model


# ============================================================================
# CATEGORY 3: FinBERT Analysis (5 tests)
# ============================================================================

def test_analyze_text_finbert_positive():
    """Test analyzing positive text with FinBERT"""
    mock_model = Mock()
    mock_model.return_value = [[
        {'label': 'positive', 'score': 0.95},
        {'label': 'neutral', 'score': 0.03},
        {'label': 'negative', 'score': 0.02}
    ]]
    
    result = analyze_text_with_finbert("Company beats earnings", mock_model)
    
    assert result['label'] == 'positive'
    assert result['sentiment'] > 0  # Positive sentiment
    assert 0 <= result['score'] <= 1


def test_analyze_text_finbert_negative():
    """Test analyzing negative text with FinBERT"""
    mock_model = Mock()
    mock_model.return_value = [[
        {'label': 'negative', 'score': 0.9},
        {'label': 'neutral', 'score': 0.07},
        {'label': 'positive', 'score': 0.03}
    ]]
    
    result = analyze_text_with_finbert("Company misses earnings", mock_model)
    
    assert result['label'] == 'negative'
    assert result['sentiment'] < 0  # Negative sentiment


def test_analyze_text_finbert_neutral():
    """Test analyzing neutral text with FinBERT"""
    mock_model = Mock()
    mock_model.return_value = [[
        {'label': 'neutral', 'score': 0.8},
        {'label': 'positive', 'score': 0.1},
        {'label': 'negative', 'score': 0.1}
    ]]
    
    result = analyze_text_with_finbert("Company releases statement", mock_model)
    
    assert result['label'] == 'neutral'
    assert result['sentiment'] == 0.0


def test_analyze_text_finbert_empty():
    """Test handling empty text"""
    mock_model = Mock()
    
    result = analyze_text_with_finbert("", mock_model)
    
    assert result['label'] == 'neutral'
    assert result['sentiment'] == 0.0


def test_analyze_text_finbert_model_none():
    """Test handling when model is None"""
    result = analyze_text_with_finbert("Some text", None)
    
    assert result['label'] == 'neutral'
    assert result['sentiment'] == 0.0


# ============================================================================
# CATEGORY 4: Sentiment Aggregation (6 tests)
# ============================================================================

def test_aggregate_sentiment_positive():
    """Test aggregating positive sentiment"""
    articles = [
        {'sentiment_score': 0.8, 'sentiment_label': 'positive'},
        {'sentiment_score': 0.7, 'sentiment_label': 'positive'},
        {'sentiment_score': 0.6, 'sentiment_label': 'positive'}
    ]
    
    result = aggregate_sentiment_by_type(articles, 'stock')
    
    assert result['article_count'] == 3
    assert result['average_sentiment'] > 0.6
    assert result['positive_count'] == 3
    assert result['sentiment_signal'] == 'BUY'
    assert result['confidence'] > 0.5


def test_aggregate_sentiment_negative():
    """Test aggregating negative sentiment"""
    articles = [
        {'sentiment_score': -0.7, 'sentiment_label': 'negative'},
        {'sentiment_score': -0.8, 'sentiment_label': 'negative'}
    ]
    
    result = aggregate_sentiment_by_type(articles, 'stock')
    
    assert result['average_sentiment'] < -0.6
    assert result['negative_count'] == 2
    assert result['sentiment_signal'] == 'SELL'


def test_aggregate_sentiment_mixed():
    """Test aggregating mixed sentiment"""
    articles = [
        {'sentiment_score': 0.5, 'sentiment_label': 'positive'},
        {'sentiment_score': -0.5, 'sentiment_label': 'negative'},
        {'sentiment_score': 0.05, 'sentiment_label': 'neutral'}
    ]
    
    result = aggregate_sentiment_by_type(articles, 'stock')
    
    assert result['article_count'] == 3
    assert result['positive_count'] == 1
    assert result['negative_count'] == 1
    assert result['neutral_count'] == 1
    # Average close to 0, likely HOLD
    assert result['sentiment_signal'] in ['BUY', 'SELL', 'HOLD']


def test_aggregate_sentiment_weighted():
    """Test weighted aggregation (recent articles matter more)"""
    # Recent article (first) is very negative, older ones positive
    articles = [
        {'sentiment_score': -0.8, 'sentiment_label': 'negative'},  # Recent (weight 1.0)
        {'sentiment_score': 0.3, 'sentiment_label': 'positive'},   # Older (weight 0.95)
        {'sentiment_score': 0.3, 'sentiment_label': 'positive'}    # Oldest (weight 0.90)
    ]
    
    result = aggregate_sentiment_by_type(articles, 'stock')
    
    # Weighted sentiment should be more negative than simple average
    assert result['weighted_sentiment'] < result['average_sentiment']


def test_aggregate_sentiment_empty():
    """Test handling empty article list"""
    result = aggregate_sentiment_by_type([], 'stock')
    
    assert result['article_count'] == 0
    assert result['average_sentiment'] == 0.0
    assert result['sentiment_signal'] == 'HOLD'
    assert result['confidence'] == 0.0


def test_aggregate_sentiment_confidence_calculation():
    """Test confidence calculation based on article count and consistency"""
    # Many consistent articles = high confidence
    articles_high_conf = [
        {'sentiment_score': 0.8, 'sentiment_label': 'positive'},
        {'sentiment_score': 0.7, 'sentiment_label': 'positive'},
        {'sentiment_score': 0.9, 'sentiment_label': 'positive'},
        {'sentiment_score': 0.75, 'sentiment_label': 'positive'},
        {'sentiment_score': 0.85, 'sentiment_label': 'positive'}
    ]
    
    result_high = aggregate_sentiment_by_type(articles_high_conf, 'stock')
    
    # Few inconsistent articles = lower confidence
    articles_low_conf = [
        {'sentiment_score': 0.3, 'sentiment_label': 'positive'},
        {'sentiment_score': -0.3, 'sentiment_label': 'negative'}
    ]
    
    result_low = aggregate_sentiment_by_type(articles_low_conf, 'stock')
    
    assert result_high['confidence'] > result_low['confidence']


# ============================================================================
# CATEGORY 5: Combined Sentiment (5 tests)
# ============================================================================

def test_combined_sentiment_weighted_50_25_25():
    """Test weighted combination (50/25/25)"""
    stock = {'weighted_sentiment': 0.8, 'confidence': 0.9, 'article_count': 10}
    market = {'weighted_sentiment': 0.4, 'confidence': 0.7, 'article_count': 5}
    related = {'weighted_sentiment': 0.6, 'confidence': 0.8, 'article_count': 5}
    
    result = calculate_combined_sentiment(stock, market, related)
    
    # Expected: 0.8*0.5 + 0.4*0.25 + 0.6*0.25 = 0.4 + 0.1 + 0.15 = 0.65
    expected = 0.65
    assert abs(result['combined_sentiment'] - expected) < 0.01
    assert result['sentiment_signal'] == 'BUY'  # > 0.2


def test_combined_sentiment_all_positive():
    """Test all positive signals"""
    stock = {'weighted_sentiment': 0.7, 'confidence': 0.8}
    market = {'weighted_sentiment': 0.6, 'confidence': 0.7}
    related = {'weighted_sentiment': 0.5, 'confidence': 0.7}
    
    result = calculate_combined_sentiment(stock, market, related)
    
    assert result['combined_sentiment'] > 0.5
    assert result['sentiment_signal'] == 'BUY'


def test_combined_sentiment_all_negative():
    """Test all negative signals"""
    stock = {'weighted_sentiment': -0.7, 'confidence': 0.8}
    market = {'weighted_sentiment': -0.5, 'confidence': 0.7}
    related = {'weighted_sentiment': -0.6, 'confidence': 0.7}
    
    result = calculate_combined_sentiment(stock, market, related)
    
    assert result['combined_sentiment'] < -0.5
    assert result['sentiment_signal'] == 'SELL'


def test_combined_sentiment_mixed_signals():
    """Test mixed signals"""
    stock = {'weighted_sentiment': 0.3, 'confidence': 0.7}
    market = {'weighted_sentiment': -0.2, 'confidence': 0.6}
    related = {'weighted_sentiment': 0.1, 'confidence': 0.6}
    
    result = calculate_combined_sentiment(stock, market, related)
    
    # Should have signal (likely HOLD or BUY given stock weight)
    assert result['sentiment_signal'] in ['BUY', 'SELL', 'HOLD']


def test_combined_sentiment_missing_data():
    """Test handling missing news types"""
    stock = {'weighted_sentiment': 0.5, 'confidence': 0.7}
    market = {'weighted_sentiment': 0.0, 'confidence': 0.0}  # No market news
    related = {'weighted_sentiment': 0.0, 'confidence': 0.0}  # No related news
    
    result = calculate_combined_sentiment(stock, market, related)
    
    # Should still work, weighted by stock only (50%)
    assert result['combined_sentiment'] == 0.25  # 0.5 * 0.5
    assert 'sentiment_signal' in result


# ============================================================================
# CATEGORY 6: Signal Generation (4 tests)
# ============================================================================

def test_generate_signal_buy():
    """Test BUY signal generation (>0.2)"""
    signal = generate_sentiment_signal(0.5, 0.8)
    assert signal == 'BUY'


def test_generate_signal_sell():
    """Test SELL signal generation (<-0.2)"""
    signal = generate_sentiment_signal(-0.5, 0.8)
    assert signal == 'SELL'


def test_generate_signal_hold():
    """Test HOLD signal generation"""
    signal = generate_sentiment_signal(0.1, 0.8)  # Between -0.2 and 0.2
    assert signal == 'HOLD'


def test_generate_signal_low_confidence():
    """Test low confidence results in HOLD"""
    signal = generate_sentiment_signal(0.8, 0.3)  # High sentiment but low confidence
    assert signal == 'HOLD'


# ============================================================================
# CATEGORY 7: Main Node Function (3 tests)
# ============================================================================

def test_node_function_success(sample_state_with_news):
    """Test full execution of sentiment analysis node"""
    result = sentiment_analysis_node(sample_state_with_news)
    
    # Check all required fields are present
    assert 'raw_sentiment_scores' in result
    assert 'aggregated_sentiment' in result
    assert 'sentiment_signal' in result
    assert 'sentiment_confidence' in result
    assert 'node_execution_times' in result
    assert 'node_5' in result['node_execution_times']
    
    # Check valid signal
    assert result['sentiment_signal'] in ['BUY', 'SELL', 'HOLD']
    
    # Check confidence range
    if result['sentiment_confidence'] is not None:
        assert 0.0 <= result['sentiment_confidence'] <= 1.0


def test_node_function_with_finbert_disabled():
    """Test execution with FinBERT disabled"""
    state = {
        'ticker': 'AAPL',
        'cleaned_stock_news': [
            {'title': 'Good news', 'ticker_sentiment_score': 0.8}
        ],
        'cleaned_market_news': [],
        'cleaned_related_company_news': [],
        'errors': [],
        'node_execution_times': {}
    }
    
    # Mock USE_FINBERT = False
    with patch('src.langgraph_nodes.node_05_sentiment_analysis.USE_FINBERT', False):
        result = sentiment_analysis_node(state)
    
    assert result['sentiment_signal'] in ['BUY', 'SELL', 'HOLD']


def test_node_function_error_handling():
    """Test error handling in main node"""
    invalid_state = {
        'ticker': 'TEST',
        # Missing required fields
        'errors': [],
        'node_execution_times': {}
    }
    
    result = sentiment_analysis_node(invalid_state)
    
    # Should not crash, should return partial state with errors
    assert 'errors' in result or 'sentiment_signal' in result


# ============================================================================
# CATEGORY 8: Integration (2 tests)
# ============================================================================

def test_integration_alpha_vantage_only():
    """Test end-to-end with Alpha Vantage sentiment only"""
    state = {
        'ticker': 'NVDA',
        'cleaned_stock_news': [
            {'title': 'Positive', 'ticker_sentiment_score': 0.8, 'summary': 'Good'},
            {'title': 'Negative', 'ticker_sentiment_score': -0.6, 'summary': 'Bad'}
        ],
        'cleaned_market_news': [
            {'title': 'Market up', 'ticker_sentiment_score': 0.5, 'summary': 'Bullish'}
        ],
        'cleaned_related_company_news': [],
        'errors': [],
        'node_execution_times': {}
    }
    
    result = sentiment_analysis_node(state)
    
    # Should complete successfully
    assert result['sentiment_signal'] in ['BUY', 'SELL', 'HOLD']
    assert len(result['raw_sentiment_scores']) == 3
    assert result['aggregated_sentiment'] is not None


def test_integration_parallel_execution_compatibility():
    """Test that node returns partial state suitable for parallel execution"""
    state = {
        'ticker': 'AAPL',
        'cleaned_stock_news': [{'title': 'News', 'ticker_sentiment_score': 0.5}],
        'cleaned_market_news': [],
        'cleaned_related_company_news': [],
        'errors': [],
        'node_execution_times': {}
    }
    
    result = sentiment_analysis_node(state)
    
    # Check that ONLY Node 5 fields are returned (partial state)
    expected_keys = {
        'raw_sentiment_scores', 
        'aggregated_sentiment', 
        'sentiment_signal', 
        'sentiment_confidence',
        'sentiment_credibility_summary',  # NEW field
        'node_execution_times'
    }
    
    # Should not contain other node fields
    assert 'ticker' not in result
    assert 'cleaned_stock_news' not in result
    assert 'raw_price_data' not in result
    
    # Should have execution time for node_5
    assert 'node_5' in result['node_execution_times']


# ============================================================================
# CREDIBILITY WEIGHTING TESTS (Node 5 Upgrade)
# ============================================================================

def test_credibility_weight_high_quality():
    """Bloomberg article should get weight near 1.0"""
    from src.langgraph_nodes.node_05_sentiment_analysis import calculate_credibility_weight
    
    article = {
        'source_credibility_score': 0.95,
        'composite_anomaly_score': 0.04,
        'relevance_score': 0.85
    }
    weight = calculate_credibility_weight(article)
    assert weight > 0.85, f"High quality article should have weight > 0.85, got {weight}"
    assert weight <= 1.0


def test_credibility_weight_low_quality():
    """Random blog should get weight near 0.2-0.3"""
    from src.langgraph_nodes.node_05_sentiment_analysis import calculate_credibility_weight
    
    article = {
        'source_credibility_score': 0.30,
        'composite_anomaly_score': 0.75,
        'relevance_score': 0.40
    }
    weight = calculate_credibility_weight(article)
    assert weight < 0.40, f"Low quality article should have weight < 0.40, got {weight}"
    assert weight >= 0.1  # Floor


def test_credibility_weight_missing_scores():
    """Articles without 9A scores should get moderate defaults"""
    from src.langgraph_nodes.node_05_sentiment_analysis import calculate_credibility_weight
    
    article = {}  # No scores at all
    weight = calculate_credibility_weight(article)
    assert 0.4 < weight < 0.7, f"Article with missing scores should have moderate weight, got {weight}"


def test_credibility_weight_floor():
    """Even worst articles should have minimum 0.1 weight"""
    from src.langgraph_nodes.node_05_sentiment_analysis import calculate_credibility_weight
    
    article = {
        'source_credibility_score': 0.0,
        'composite_anomaly_score': 1.0,
        'relevance_score': 0.0
    }
    weight = calculate_credibility_weight(article)
    assert weight >= 0.1, f"Weight should be floored at 0.1, got {weight}"
    assert weight <= 0.15  # Should be very close to floor


def test_weighted_aggregation_reduces_blog_influence():
    """Random blog sentiment should not dominate over Bloomberg"""
    articles = [
        {
            'sentiment_score': 0.9, 
            'source_credibility_score': 0.30,
            'composite_anomaly_score': 0.75, 
            'relevance_score': 0.4,
            'time_published': '20260214T120000',
            'sentiment_label': 'positive'
        },
        {
            'sentiment_score': -0.3, 
            'source_credibility_score': 0.95,
            'composite_anomaly_score': 0.04, 
            'relevance_score': 0.85,
            'time_published': '20260214T110000',
            'sentiment_label': 'negative'
        },
    ]
    result = aggregate_sentiment_by_type(articles, 'stock')
    
    # Bloomberg's -0.3 should pull the average below 0.3
    # Without weighting it would be (0.9 + -0.3)/2 = 0.30
    # With credibility weighting, Bloomberg should have more influence
    assert result['weighted_sentiment'] < 0.15, \
        f"Expected weighted sentiment < 0.15, got {result['weighted_sentiment']}"


def test_credibility_summary_in_output():
    """State output should include credibility summary"""
    state = {
        'ticker': 'TEST',
        'cleaned_stock_news': [
            {
                'title': 'High credibility news',
                'ticker_sentiment_score': 0.8,
                'source_credibility_score': 0.90,
                'composite_anomaly_score': 0.05,
                'relevance_score': 0.85
            },
            {
                'title': 'Low credibility news',
                'ticker_sentiment_score': 0.5,
                'source_credibility_score': 0.40,
                'composite_anomaly_score': 0.60,
                'relevance_score': 0.50
            }
        ],
        'cleaned_market_news': [],
        'cleaned_related_company_news': [],
        'errors': [],
        'node_execution_times': {}
    }
    
    result = sentiment_analysis_node(state)
    
    assert 'sentiment_credibility_summary' in result
    summary = result['sentiment_credibility_summary']
    assert 'avg_source_credibility' in summary
    assert 'high_credibility_articles' in summary
    assert 'medium_credibility_articles' in summary
    assert 'low_credibility_articles' in summary
    assert 'avg_composite_anomaly' in summary
    assert summary['credibility_weighted'] == True
    
    # Check counts
    assert summary['high_credibility_articles'] == 1  # 0.90 >= 0.8
    assert summary['low_credibility_articles'] == 1   # 0.40 < 0.5


def test_per_article_scores_include_credibility():
    """raw_sentiment_scores should include credibility_weight per article"""
    state = {
        'ticker': 'TEST',
        'cleaned_stock_news': [
            {
                'title': 'Test article',
                'source': 'Bloomberg',
                'ticker_sentiment_score': 0.7,
                'source_credibility_score': 0.95,
                'composite_anomaly_score': 0.04,
                'relevance_score': 0.85
            }
        ],
        'cleaned_market_news': [],
        'cleaned_related_company_news': [],
        'errors': [],
        'node_execution_times': {}
    }
    
    result = sentiment_analysis_node(state)
    
    assert len(result['raw_sentiment_scores']) > 0
    for score in result['raw_sentiment_scores']:
        assert 'credibility_weight' in score
        assert 'source_credibility_score' in score
        assert 'composite_anomaly_score' in score
        assert 'relevance_score' in score
        assert 'source' in score


def test_backward_compatibility_without_9a_scores():
    """Node 5 should still work if 9A scores are missing (graceful defaults)"""
    articles = [
        {
            'title': 'Test', 
            'summary': 'Test article', 
            'ticker_sentiment_score': 0.5
            # No 9A scores at all
        }
    ]
    state = {
        'ticker': 'TEST',
        'cleaned_stock_news': articles, 
        'cleaned_market_news': [],
        'cleaned_related_company_news': [], 
        'errors': [], 
        'node_execution_times': {}
    }
    
    result = sentiment_analysis_node(state)
    
    # Should work with default weights
    assert result['aggregated_sentiment'] is not None
    assert result['sentiment_signal'] in ['BUY', 'SELL', 'HOLD']
    assert 'sentiment_credibility_summary' in result
    
    # Check that default credibility was used
    summary = result['sentiment_credibility_summary']
    assert 0.4 < summary['avg_source_credibility'] < 0.6  # Should be near 0.5 default


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
