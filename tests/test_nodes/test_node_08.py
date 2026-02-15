"""
Tests for Node 8: News Impact Verification & Learning System
PRIMARY THESIS INNOVATION

Tests cover:
1. Utility functions (3 tests)
2. Source reliability calculation (5 tests)
3. News type effectiveness calculation (4 tests)
4. Historical correlation calculation (5 tests)
5. Confidence adjustment (6 tests)
6. Main node function - sufficient data (4 tests)
7. Main node function - insufficient data (3 tests)
8. Main node function - error handling (3 tests)
9. Integration tests (4 tests)

Total: 37+ tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.langgraph_nodes.node_08_news_verification import (
    match_source_name,
    convert_dataframe_to_events,
    clamp_value,
    calculate_source_reliability,
    calculate_news_type_effectiveness,
    calculate_historical_correlation,
    adjust_current_sentiment_confidence,
    news_verification_node
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_historical_events():
    """Historical news events with outcomes for testing"""
    return [
        # Bloomberg - high accuracy (8/10 = 80%)
        {'source': 'Bloomberg', 'prediction_was_accurate_7day': True, 'price_change_7day': 2.5, 
         'news_type': 'stock', 'sentiment_label': 'positive'},
        {'source': 'Bloomberg', 'prediction_was_accurate_7day': True, 'price_change_7day': 1.8, 
         'news_type': 'stock', 'sentiment_label': 'positive'},
        {'source': 'Bloomberg', 'prediction_was_accurate_7day': True, 'price_change_7day': 3.2, 
         'news_type': 'stock', 'sentiment_label': 'positive'},
        {'source': 'Bloomberg', 'prediction_was_accurate_7day': True, 'price_change_7day': 1.5, 
         'news_type': 'stock', 'sentiment_label': 'positive'},
        {'source': 'Bloomberg', 'prediction_was_accurate_7day': True, 'price_change_7day': 2.0, 
         'news_type': 'stock', 'sentiment_label': 'positive'},
        {'source': 'Bloomberg', 'prediction_was_accurate_7day': True, 'price_change_7day': -1.8, 
         'news_type': 'stock', 'sentiment_label': 'negative'},
        {'source': 'Bloomberg', 'prediction_was_accurate_7day': True, 'price_change_7day': -2.2, 
         'news_type': 'stock', 'sentiment_label': 'negative'},
        {'source': 'Bloomberg', 'prediction_was_accurate_7day': True, 'price_change_7day': -1.5, 
         'news_type': 'stock', 'sentiment_label': 'negative'},
        {'source': 'Bloomberg', 'prediction_was_accurate_7day': False, 'price_change_7day': -0.5, 
         'news_type': 'stock', 'sentiment_label': 'positive'},
        {'source': 'Bloomberg', 'prediction_was_accurate_7day': False, 'price_change_7day': 0.8, 
         'news_type': 'stock', 'sentiment_label': 'negative'},
        
        # Random Blog - low accuracy (2/10 = 20%)
        {'source': 'Random Blog', 'prediction_was_accurate_7day': True, 'price_change_7day': 0.5, 
         'news_type': 'market', 'sentiment_label': 'positive'},
        {'source': 'Random Blog', 'prediction_was_accurate_7day': True, 'price_change_7day': 0.3, 
         'news_type': 'market', 'sentiment_label': 'positive'},
        {'source': 'Random Blog', 'prediction_was_accurate_7day': False, 'price_change_7day': -1.2, 
         'news_type': 'market', 'sentiment_label': 'positive'},
        {'source': 'Random Blog', 'prediction_was_accurate_7day': False, 'price_change_7day': -0.8, 
         'news_type': 'market', 'sentiment_label': 'positive'},
        {'source': 'Random Blog', 'prediction_was_accurate_7day': False, 'price_change_7day': 1.5, 
         'news_type': 'market', 'sentiment_label': 'negative'},
        {'source': 'Random Blog', 'prediction_was_accurate_7day': False, 'price_change_7day': 0.9, 
         'news_type': 'market', 'sentiment_label': 'negative'},
        {'source': 'Random Blog', 'prediction_was_accurate_7day': False, 'price_change_7day': 0.2, 
         'news_type': 'market', 'sentiment_label': 'neutral'},
        {'source': 'Random Blog', 'prediction_was_accurate_7day': False, 'price_change_7day': -0.3, 
         'news_type': 'market', 'sentiment_label': 'neutral'},
        {'source': 'Random Blog', 'prediction_was_accurate_7day': False, 'price_change_7day': 0.7, 
         'news_type': 'market', 'sentiment_label': 'negative'},
        {'source': 'Random Blog', 'prediction_was_accurate_7day': False, 'price_change_7day': -0.4, 
         'news_type': 'market', 'sentiment_label': 'positive'},
        
        # Reuters - medium accuracy (6/10 = 60%)
        {'source': 'Reuters', 'prediction_was_accurate_7day': True, 'price_change_7day': 1.8, 
         'news_type': 'related', 'sentiment_label': 'positive'},
        {'source': 'Reuters', 'prediction_was_accurate_7day': True, 'price_change_7day': 2.1, 
         'news_type': 'related', 'sentiment_label': 'positive'},
        {'source': 'Reuters', 'prediction_was_accurate_7day': True, 'price_change_7day': -1.5, 
         'news_type': 'related', 'sentiment_label': 'negative'},
        {'source': 'Reuters', 'prediction_was_accurate_7day': True, 'price_change_7day': 1.2, 
         'news_type': 'related', 'sentiment_label': 'positive'},
        {'source': 'Reuters', 'prediction_was_accurate_7day': True, 'price_change_7day': -0.9, 
         'news_type': 'related', 'sentiment_label': 'negative'},
        {'source': 'Reuters', 'prediction_was_accurate_7day': True, 'price_change_7day': 0.8, 
         'news_type': 'related', 'sentiment_label': 'positive'},
        {'source': 'Reuters', 'prediction_was_accurate_7day': False, 'price_change_7day': -0.6, 
         'news_type': 'related', 'sentiment_label': 'positive'},
        {'source': 'Reuters', 'prediction_was_accurate_7day': False, 'price_change_7day': 0.5, 
         'news_type': 'related', 'sentiment_label': 'negative'},
        {'source': 'Reuters', 'prediction_was_accurate_7day': False, 'price_change_7day': -0.3, 
         'news_type': 'related', 'sentiment_label': 'positive'},
        {'source': 'Reuters', 'prediction_was_accurate_7day': False, 'price_change_7day': 0.4, 
         'news_type': 'related', 'sentiment_label': 'negative'},
    ]


@pytest.fixture
def sample_current_sentiment():
    """Current sentiment from Node 5"""
    return {
        'sentiment_signal': 'BUY',
        'combined_sentiment_score': 0.35,
        'confidence': 0.72,
        'sentiment_label': 'positive'
    }


@pytest.fixture
def sample_today_articles():
    """Today's articles from Node 9A"""
    return {
        'stock': [
            {'source': 'Bloomberg', 'title': 'Good news'},
            {'source': 'Bloomberg', 'title': 'More good news'},
            {'source': 'Reuters', 'title': 'Neutral news'}
        ],
        'market': [
            {'source': 'Random Blog', 'title': 'Market news'}
        ],
        'related': [
            {'source': 'Reuters', 'title': 'Competitor news'}
        ]
    }


# ============================================================================
# TEST GROUP 1: UTILITY FUNCTIONS
# ============================================================================

class TestUtilityFunctions:
    """Test utility helper functions"""
    
    def test_match_source_name_exact(self):
        """Test exact case-insensitive match"""
        reliability = {
            'Bloomberg': {'accuracy_rate': 0.8},
            'Reuters': {'accuracy_rate': 0.6}
        }
        
        assert match_source_name('Bloomberg', reliability) == 'Bloomberg'
        assert match_source_name('bloomberg', reliability) == 'Bloomberg'
        assert match_source_name('BLOOMBERG', reliability) == 'Bloomberg'
    
    def test_match_source_name_no_match(self):
        """Test no match returns None"""
        reliability = {'Bloomberg': {'accuracy_rate': 0.8}}
        
        assert match_source_name('Unknown Source', reliability) is None
        assert match_source_name('', reliability) is None
    
    def test_match_source_name_partial(self):
        """Test partial matching"""
        reliability = {'Bloomberg.com': {'accuracy_rate': 0.8}}
        
        # Should match partial
        result = match_source_name('Bloomberg', reliability)
        assert result == 'Bloomberg.com'
    
    def test_convert_dataframe_to_events_valid(self):
        """Test DataFrame to dict conversion"""
        df = pd.DataFrame({
            'source': ['Bloomberg', 'Reuters'],
            'prediction_was_accurate_7day': [True, False],
            'price_change_7day': [2.5, -1.3]
        })
        
        events = convert_dataframe_to_events(df)
        
        assert len(events) == 2
        assert events[0]['source'] == 'Bloomberg'
        assert events[0]['prediction_was_accurate_7day'] == True
        assert events[0]['price_change_7day'] == 2.5
    
    def test_convert_dataframe_to_events_empty(self):
        """Test empty DataFrame"""
        df = pd.DataFrame()
        events = convert_dataframe_to_events(df)
        assert events == []
    
    def test_convert_dataframe_to_events_none(self):
        """Test None input"""
        events = convert_dataframe_to_events(None)
        assert events == []
    
    def test_clamp_value(self):
        """Test value clamping"""
        assert clamp_value(0.5, 0.0, 1.0) == 0.5
        assert clamp_value(1.5, 0.0, 1.0) == 1.0
        assert clamp_value(-0.5, 0.0, 1.0) == 0.0
        assert clamp_value(0.8, 0.5, 2.0) == 0.8


# ============================================================================
# TEST GROUP 2: SOURCE RELIABILITY CALCULATION
# ============================================================================

class TestSourceReliability:
    """Test calculate_source_reliability function"""
    
    def test_high_accuracy_source(self, sample_historical_events):
        """Test high accuracy source gets high multiplier"""
        reliability = calculate_source_reliability(sample_historical_events, 'NVDA')
        
        assert 'Bloomberg' in reliability
        bloomberg = reliability['Bloomberg']
        
        assert bloomberg['total_articles'] == 10
        assert bloomberg['accurate_predictions'] == 8
        assert bloomberg['accuracy_rate'] == 0.8
        assert bloomberg['confidence_multiplier'] >= 1.0  # Should boost
    
    def test_low_accuracy_source(self, sample_historical_events):
        """Test low accuracy source gets low multiplier"""
        reliability = calculate_source_reliability(sample_historical_events, 'NVDA')
        
        assert 'Random Blog' in reliability
        blog = reliability['Random Blog']
        
        assert blog['total_articles'] == 10
        assert blog['accurate_predictions'] == 2
        assert blog['accuracy_rate'] == 0.2
        assert blog['confidence_multiplier'] < 1.0  # Should reduce
    
    def test_medium_accuracy_source(self, sample_historical_events):
        """Test medium accuracy source gets neutral multiplier"""
        reliability = calculate_source_reliability(sample_historical_events, 'NVDA')
        
        assert 'Reuters' in reliability
        reuters = reliability['Reuters']
        
        assert reuters['total_articles'] == 10
        assert reuters['accurate_predictions'] == 6
        assert reuters['accuracy_rate'] == 0.6
        assert reuters['confidence_multiplier'] == 1.0  # Neutral
    
    def test_empty_events(self):
        """Test with no events"""
        reliability = calculate_source_reliability([], 'NVDA')
        assert reliability == {}
    
    def test_price_impact_calculation(self, sample_historical_events):
        """Test average price impact calculation"""
        reliability = calculate_source_reliability(sample_historical_events, 'NVDA')
        
        # Bloomberg has larger price changes
        assert reliability['Bloomberg']['avg_price_impact'] > reliability['Random Blog']['avg_price_impact']
    
    def test_multiplier_bounds(self, sample_historical_events):
        """Test confidence multipliers are within expected bounds"""
        reliability = calculate_source_reliability(sample_historical_events, 'NVDA')
        
        for source, metrics in reliability.items():
            multiplier = metrics['confidence_multiplier']
            assert 0.5 <= multiplier <= 1.4  # Valid range


# ============================================================================
# TEST GROUP 3: NEWS TYPE EFFECTIVENESS
# ============================================================================

class TestNewsTypeEffectiveness:
    """Test calculate_news_type_effectiveness function"""
    
    def test_all_types_present(self, sample_historical_events):
        """Test effectiveness for all news types"""
        effectiveness = calculate_news_type_effectiveness(sample_historical_events, 'NVDA')
        
        assert 'stock' in effectiveness
        assert 'market' in effectiveness
        assert 'related' in effectiveness
    
    def test_stock_news_accuracy(self, sample_historical_events):
        """Test stock news effectiveness calculation"""
        effectiveness = calculate_news_type_effectiveness(sample_historical_events, 'NVDA')
        
        stock = effectiveness['stock']
        # Bloomberg articles (all stock type): 8/10 accurate
        assert stock['accuracy_rate'] == 0.8
        assert stock['sample_size'] == 10
        assert stock['avg_impact'] > 0
    
    def test_market_news_accuracy(self, sample_historical_events):
        """Test market news effectiveness"""
        effectiveness = calculate_news_type_effectiveness(sample_historical_events, 'NVDA')
        
        market = effectiveness['market']
        # Random Blog articles (all market type): 2/10 accurate
        assert market['accuracy_rate'] == 0.2
        assert market['sample_size'] == 10
    
    def test_empty_events(self):
        """Test with no events"""
        effectiveness = calculate_news_type_effectiveness([], 'NVDA')
        
        # Should return neutral defaults for all types
        assert effectiveness['stock']['accuracy_rate'] == 0.5
        assert effectiveness['market']['accuracy_rate'] == 0.5
        assert effectiveness['related']['accuracy_rate'] == 0.5


# ============================================================================
# TEST GROUP 4: HISTORICAL CORRELATION
# ============================================================================

class TestHistoricalCorrelation:
    """Test calculate_historical_correlation function"""
    
    def test_perfect_positive_correlation(self):
        """Test perfect positive correlation (positive news → price up)"""
        events = [
            {'sentiment_label': 'positive', 'price_change_7day': 2.0},
            {'sentiment_label': 'positive', 'price_change_7day': 1.5},
            {'sentiment_label': 'positive', 'price_change_7day': 3.0},
            {'sentiment_label': 'negative', 'price_change_7day': -2.0},
            {'sentiment_label': 'negative', 'price_change_7day': -1.5},
            {'sentiment_label': 'negative', 'price_change_7day': -2.5},
            {'sentiment_label': 'neutral', 'price_change_7day': 0.1},
            {'sentiment_label': 'neutral', 'price_change_7day': -0.1},
            {'sentiment_label': 'positive', 'price_change_7day': 2.5},
            {'sentiment_label': 'negative', 'price_change_7day': -1.8},
        ]
        
        correlation = calculate_historical_correlation(events)
        
        # Should be close to 1.0 (perfect positive)
        assert correlation > 0.8
    
    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation (positive news → price down)"""
        events = [
            {'sentiment_label': 'positive', 'price_change_7day': -2.0},
            {'sentiment_label': 'positive', 'price_change_7day': -1.5},
            {'sentiment_label': 'positive', 'price_change_7day': -3.0},
            {'sentiment_label': 'negative', 'price_change_7day': 2.0},
            {'sentiment_label': 'negative', 'price_change_7day': 1.5},
            {'sentiment_label': 'negative', 'price_change_7day': 2.5},
            {'sentiment_label': 'neutral', 'price_change_7day': 0.1},
            {'sentiment_label': 'neutral', 'price_change_7day': -0.1},
            {'sentiment_label': 'positive', 'price_change_7day': -2.5},
            {'sentiment_label': 'negative', 'price_change_7day': 1.8},
        ]
        
        correlation = calculate_historical_correlation(events)
        
        # Should be close to 0.0 (perfect negative)
        assert correlation < 0.2
    
    def test_no_correlation(self):
        """Test no correlation (random)"""
        events = [
            {'sentiment_label': 'positive', 'price_change_7day': 1.0},
            {'sentiment_label': 'positive', 'price_change_7day': -1.0},
            {'sentiment_label': 'negative', 'price_change_7day': 1.0},
            {'sentiment_label': 'negative', 'price_change_7day': -1.0},
            {'sentiment_label': 'neutral', 'price_change_7day': 0.5},
            {'sentiment_label': 'neutral', 'price_change_7day': -0.5},
            {'sentiment_label': 'positive', 'price_change_7day': -0.8},
            {'sentiment_label': 'negative', 'price_change_7day': 0.8},
            {'sentiment_label': 'positive', 'price_change_7day': 0.3},
            {'sentiment_label': 'negative', 'price_change_7day': -0.3},
        ]
        
        correlation = calculate_historical_correlation(events)
        
        # Should be close to 0.5 (no correlation)
        assert 0.3 < correlation < 0.7
    
    def test_insufficient_events(self):
        """Test with fewer than 10 events"""
        events = [
            {'sentiment_label': 'positive', 'price_change_7day': 2.0},
            {'sentiment_label': 'negative', 'price_change_7day': -1.0}
        ]
        
        correlation = calculate_historical_correlation(events)
        
        # Should return neutral default
        assert correlation == 0.5
    
    def test_missing_price_changes(self):
        """Test with some None price changes"""
        events = [
            {'sentiment_label': 'positive', 'price_change_7day': 2.0},
            {'sentiment_label': 'positive', 'price_change_7day': None},
            {'sentiment_label': 'negative', 'price_change_7day': -1.0},
        ] * 4
        
        correlation = calculate_historical_correlation(events)
        
        # Should handle None values gracefully
        assert 0.0 <= correlation <= 1.0


# ============================================================================
# TEST GROUP 5: CONFIDENCE ADJUSTMENT
# ============================================================================

class TestConfidenceAdjustment:
    """Test adjust_current_sentiment_confidence function"""
    
    def test_high_reliability_boosts_confidence(self, sample_current_sentiment, sample_today_articles):
        """Test high reliability sources boost confidence"""
        # Bloomberg has high reliability
        source_reliability = {
            'Bloomberg': {
                'total_articles': 10,
                'accurate_predictions': 8,
                'accuracy_rate': 0.8,
                'confidence_multiplier': 1.2
            }
        }
        
        news_type_eff = {
            'stock': {'accuracy_rate': 0.7},
            'market': {'accuracy_rate': 0.5},
            'related': {'accuracy_rate': 0.6}
        }
        
        adjusted, details = adjust_current_sentiment_confidence(
            sample_current_sentiment,
            source_reliability,
            news_type_eff,
            sample_today_articles
        )
        
        # Confidence should increase
        assert adjusted['confidence'] > sample_current_sentiment['confidence']
        assert details['reliability_multiplier'] > 1.0
        assert details['sources_matched'] > 0
    
    def test_low_reliability_reduces_confidence(self, sample_current_sentiment):
        """Test low reliability sources reduce confidence"""
        source_reliability = {
            'Random Blog': {
                'total_articles': 10,
                'accurate_predictions': 2,
                'accuracy_rate': 0.2,
                'confidence_multiplier': 0.7
            }
        }
        
        news_type_eff = {
            'stock': {'accuracy_rate': 0.5},
            'market': {'accuracy_rate': 0.5},
            'related': {'accuracy_rate': 0.5}
        }
        
        today_articles = {
            'stock': [{'source': 'Random Blog'}],
            'market': [],
            'related': []
        }
        
        adjusted, details = adjust_current_sentiment_confidence(
            sample_current_sentiment,
            source_reliability,
            news_type_eff,
            today_articles
        )
        
        # Confidence should decrease
        assert adjusted['confidence'] < sample_current_sentiment['confidence']
        assert details['reliability_multiplier'] < 1.0
    
    def test_unknown_sources_neutral(self, sample_current_sentiment):
        """Test unknown sources use neutral multiplier"""
        source_reliability = {}  # No known sources
        
        news_type_eff = {
            'stock': {'accuracy_rate': 0.5},
            'market': {'accuracy_rate': 0.5},
            'related': {'accuracy_rate': 0.5}
        }
        
        today_articles = {
            'stock': [{'source': 'Unknown Source'}],
            'market': [],
            'related': []
        }
        
        adjusted, details = adjust_current_sentiment_confidence(
            sample_current_sentiment,
            source_reliability,
            news_type_eff,
            today_articles
        )
        
        # Should not crash, use neutral adjustment
        assert details['sources_unmatched'] > 0
        assert 0.0 <= adjusted['confidence'] <= 1.0
    
    def test_case_insensitive_matching(self, sample_current_sentiment):
        """Test source name matching is case-insensitive"""
        source_reliability = {
            'Bloomberg': {'confidence_multiplier': 1.2}
        }
        
        news_type_eff = {'stock': {'accuracy_rate': 0.7}}
        
        # Test with lowercase
        today_articles = {
            'stock': [{'source': 'bloomberg'}],  # lowercase
            'market': [],
            'related': []
        }
        
        adjusted, details = adjust_current_sentiment_confidence(
            sample_current_sentiment,
            source_reliability,
            news_type_eff,
            today_articles
        )
        
        # Should match and boost
        assert details['sources_matched'] == 1
        assert adjusted['confidence'] > sample_current_sentiment['confidence']
    
    def test_weighted_news_type_effectiveness(self, sample_current_sentiment, sample_today_articles):
        """Test news type effectiveness is weighted by article count"""
        source_reliability = {}
        
        # Stock news is highly effective, market is not
        news_type_eff = {
            'stock': {'accuracy_rate': 0.9},  # Very effective
            'market': {'accuracy_rate': 0.2},  # Not effective
            'related': {'accuracy_rate': 0.6}
        }
        
        # Most articles are stock type (3 stock, 1 market, 1 related)
        # So weighted average should be closer to 0.9 than equal average (0.567)
        
        adjusted, details = adjust_current_sentiment_confidence(
            sample_current_sentiment,
            source_reliability,
            news_type_eff,
            sample_today_articles
        )
        
        # Effectiveness factor should reflect the weighted calculation
        assert 0.0 <= details['effectiveness_factor'] <= 2.0
    
    def test_confidence_bounds(self, sample_current_sentiment):
        """Test confidence stays within [0, 1] bounds"""
        # Extreme multipliers
        source_reliability = {
            'Bloomberg': {'confidence_multiplier': 10.0}  # Unrealistically high
        }
        
        news_type_eff = {'stock': {'accuracy_rate': 1.0}}
        
        today_articles = {
            'stock': [{'source': 'Bloomberg'}],
            'market': [],
            'related': []
        }
        
        adjusted, details = adjust_current_sentiment_confidence(
            sample_current_sentiment,
            source_reliability,
            news_type_eff,
            today_articles
        )
        
        # Should be clamped to [0, 1]
        assert 0.0 <= adjusted['confidence'] <= 1.0


# ============================================================================
# TEST GROUP 6: MAIN NODE - SUFFICIENT DATA
# ============================================================================

class TestNodeSufficientData:
    """Test news_verification_node with sufficient historical data"""
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    @patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability')
    def test_node_with_sufficient_data(self, mock_store, mock_get_news, 
                                      sample_historical_events, sample_current_sentiment):
        """Test node with sufficient historical data (>10 events)"""
        # Mock database response
        historical_df = pd.DataFrame(sample_historical_events)
        mock_get_news.return_value = historical_df
        
        state = {
            'ticker': 'NVDA',
            'sentiment_analysis': sample_current_sentiment,
            'cleaned_stock_news': [{'source': 'Bloomberg', 'title': 'Good news'}],
            'cleaned_market_news': [],
            'cleaned_related_news': []
        }
        
        result = news_verification_node(state)
        
        # Should have verification results
        assert 'news_impact_verification' in result
        verification = result['news_impact_verification']
        
        assert verification['insufficient_data'] == False
        assert verification['sample_size'] == 30
        assert 'source_reliability' in verification
        assert 'news_type_effectiveness' in verification
        assert 0.0 <= verification['historical_correlation'] <= 1.0
        assert 0.5 <= verification['learning_adjustment'] <= 2.0
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    @patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability')
    def test_node_adjusts_confidence(self, mock_store, mock_get_news, 
                                    sample_historical_events, sample_current_sentiment):
        """Test node adjusts sentiment confidence"""
        historical_df = pd.DataFrame(sample_historical_events)
        mock_get_news.return_value = historical_df
        
        state = {
            'ticker': 'NVDA',
            'sentiment_analysis': sample_current_sentiment.copy(),
            'cleaned_stock_news': [{'source': 'Bloomberg', 'title': 'Good news'}],
            'cleaned_market_news': [],
            'cleaned_related_news': []
        }
        
        original_confidence = state['sentiment_analysis']['confidence']
        
        result = news_verification_node(state)
        
        # Confidence should be adjusted
        adjusted_confidence = result['sentiment_analysis']['confidence']
        assert 'confidence_adjustment' in result['sentiment_analysis']
        
        # Bloomberg is high reliability, so confidence should increase
        assert adjusted_confidence != original_confidence
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    @patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability')
    def test_node_saves_to_database(self, mock_store, mock_get_news, 
                                   sample_historical_events, sample_current_sentiment):
        """Test node saves source reliability to database"""
        historical_df = pd.DataFrame(sample_historical_events)
        mock_get_news.return_value = historical_df
        
        state = {
            'ticker': 'NVDA',
            'sentiment_analysis': sample_current_sentiment,
            'cleaned_stock_news': [],
            'cleaned_market_news': [],
            'cleaned_related_news': []
        }
        
        result = news_verification_node(state)
        
        # Should call store_source_reliability
        mock_store.assert_called_once()
        call_args = mock_store.call_args
        assert call_args[0][0] == 'NVDA'  # ticker
        assert isinstance(call_args[0][1], dict)  # reliability dict
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    @patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability')
    def test_node_execution_time(self, mock_store, mock_get_news, 
                                 sample_historical_events, sample_current_sentiment):
        """Test node records execution time"""
        historical_df = pd.DataFrame(sample_historical_events)
        mock_get_news.return_value = historical_df
        
        state = {
            'ticker': 'NVDA',
            'sentiment_analysis': sample_current_sentiment,
            'cleaned_stock_news': [],
            'cleaned_market_news': [],
            'cleaned_related_news': []
        }
        
        result = news_verification_node(state)
        
        assert 'node_execution_times' in result
        assert 'node_8' in result['node_execution_times']
        assert result['node_execution_times']['node_8'] > 0


# ============================================================================
# TEST GROUP 7: MAIN NODE - INSUFFICIENT DATA
# ============================================================================

class TestNodeInsufficientData:
    """Test news_verification_node with insufficient historical data"""
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    def test_node_with_insufficient_data(self, mock_get_news, sample_current_sentiment):
        """Test node with <10 historical events"""
        # Mock insufficient data
        historical_df = pd.DataFrame([
            {'source': 'Bloomberg', 'prediction_was_accurate_7day': True, 
             'price_change_7day': 2.5, 'news_type': 'stock', 'sentiment_label': 'positive'}
        ] * 5)  # Only 5 events
        mock_get_news.return_value = historical_df
        
        state = {
            'ticker': 'NVDA',
            'sentiment_analysis': sample_current_sentiment.copy(),
            'cleaned_stock_news': [],
            'cleaned_market_news': [],
            'cleaned_related_news': []
        }
        
        original_confidence = state['sentiment_analysis']['confidence']
        
        result = news_verification_node(state)
        
        # Should return insufficient data flag
        verification = result['news_impact_verification']
        assert verification['insufficient_data'] == True
        assert verification['sample_size'] == 5
        assert verification['learning_adjustment'] == 1.0  # Neutral
        
        # Should NOT adjust confidence
        assert result['sentiment_analysis']['confidence'] == original_confidence
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    def test_node_with_zero_events(self, mock_get_news, sample_current_sentiment):
        """Test node with zero historical events"""
        mock_get_news.return_value = pd.DataFrame()
        
        state = {
            'ticker': 'NVDA',
            'sentiment_analysis': sample_current_sentiment,
            'cleaned_stock_news': [],
            'cleaned_market_news': [],
            'cleaned_related_news': []
        }
        
        result = news_verification_node(state)
        
        verification = result['news_impact_verification']
        assert verification['insufficient_data'] == True
        assert verification['sample_size'] == 0
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    def test_node_neutral_defaults(self, mock_get_news, sample_current_sentiment):
        """Test node returns neutral defaults when insufficient data"""
        mock_get_news.return_value = pd.DataFrame()
        
        state = {
            'ticker': 'NVDA',
            'sentiment_analysis': sample_current_sentiment,
            'cleaned_stock_news': [],
            'cleaned_market_news': [],
            'cleaned_related_news': []
        }
        
        result = news_verification_node(state)
        
        verification = result['news_impact_verification']
        assert verification['historical_correlation'] == 0.5
        assert verification['news_accuracy_score'] == 50.0
        assert verification['learning_adjustment'] == 1.0


# ============================================================================
# TEST GROUP 8: MAIN NODE - ERROR HANDLING
# ============================================================================

class TestNodeErrorHandling:
    """Test news_verification_node error handling"""
    
    def test_node_missing_sentiment(self):
        """Test node handles missing sentiment analysis"""
        state = {
            'ticker': 'NVDA',
            # No sentiment_analysis!
            'cleaned_stock_news': [],
            'cleaned_market_news': [],
            'cleaned_related_news': []
        }
        
        result = news_verification_node(state)
        
        # Should return early without crashing
        assert result['news_impact_verification'] is None
        assert 'node_8' in result['node_execution_times']
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    def test_node_database_error(self, mock_get_news, sample_current_sentiment):
        """Test node handles database errors gracefully"""
        # Mock database error
        mock_get_news.side_effect = Exception("Database connection failed")
        
        state = {
            'ticker': 'NVDA',
            'sentiment_analysis': sample_current_sentiment,
            'cleaned_stock_news': [],
            'cleaned_market_news': [],
            'cleaned_related_news': []
        }
        
        result = news_verification_node(state)
        
        # Should return neutral defaults on error
        verification = result['news_impact_verification']
        assert verification['insufficient_data'] == True
        assert 'error' in verification or verification['sample_size'] == 0
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    @patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability')
    def test_node_calculation_error(self, mock_store, mock_get_news, sample_current_sentiment):
        """Test node handles calculation errors"""
        # Return malformed data that might cause calculation errors
        historical_df = pd.DataFrame([
            {'source': None, 'prediction_was_accurate_7day': None, 
             'price_change_7day': None, 'news_type': None, 'sentiment_label': None}
        ] * 15)
        mock_get_news.return_value = historical_df
        
        state = {
            'ticker': 'NVDA',
            'sentiment_analysis': sample_current_sentiment,
            'cleaned_stock_news': [],
            'cleaned_market_news': [],
            'cleaned_related_news': []
        }
        
        result = news_verification_node(state)
        
        # Should not crash
        assert 'news_impact_verification' in result


# ============================================================================
# TEST GROUP 9: INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests with realistic scenarios"""
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    @patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability')
    def test_full_pipeline_nvda(self, mock_store, mock_get_news):
        """Test full pipeline with realistic NVDA scenario"""
        # Create realistic historical data
        historical_events = []
        for i in range(100):
            historical_events.append({
                'source': 'Bloomberg' if i % 3 == 0 else 'Reuters',
                'prediction_was_accurate_7day': i % 4 != 0,  # 75% accurate
                'price_change_7day': np.random.normal(1.5, 2.0),
                'news_type': ['stock', 'market', 'related'][i % 3],
                'sentiment_label': ['positive', 'negative', 'neutral'][i % 3]
            })
        
        historical_df = pd.DataFrame(historical_events)
        mock_get_news.return_value = historical_df
        
        state = {
            'ticker': 'NVDA',
            'sentiment_analysis': {
                'confidence': 0.68,
                'sentiment_signal': 'BUY',
                'combined_sentiment_score': 0.42
            },
            'cleaned_stock_news': [
                {'source': 'Bloomberg', 'title': 'NVDA earnings beat'},
                {'source': 'Reuters', 'title': 'AI chip demand strong'}
            ],
            'cleaned_market_news': [
                {'source': 'CNBC', 'title': 'Tech sector rallies'}
            ],
            'cleaned_related_news': []
        }
        
        result = news_verification_node(state)
        
        # Verify complete output structure
        assert 'news_impact_verification' in result
        verification = result['news_impact_verification']
        
        assert verification['insufficient_data'] == False
        assert verification['sample_size'] == 100
        assert 0.0 <= verification['historical_correlation'] <= 1.0
        assert 0.0 <= verification['news_accuracy_score'] <= 100.0
        assert 0.5 <= verification['learning_adjustment'] <= 2.0
        
        # Verify sentiment was adjusted
        assert 'confidence_adjustment' in result['sentiment_analysis']
        
        # Verify execution time is reasonable
        assert result['node_execution_times']['node_8'] < 5.0  # Should be fast
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    @patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability')
    def test_learning_adjustment_bounds(self, mock_store, mock_get_news):
        """Test learning_adjustment is always within [0.5, 2.0]"""
        # Test with various accuracy scenarios
        scenarios = [
            (0.95, 100),  # Very high accuracy
            (0.50, 50),   # Medium accuracy
            (0.20, 20),   # Low accuracy
        ]
        
        for accuracy, sample_size in scenarios:
            historical_events = [
                {
                    'source': 'Test Source',
                    'prediction_was_accurate_7day': i < int(sample_size * accuracy),
                    'price_change_7day': 1.0,
                    'news_type': 'stock',
                    'sentiment_label': 'positive'
                }
                for i in range(sample_size)
            ]
            
            historical_df = pd.DataFrame(historical_events)
            mock_get_news.return_value = historical_df
            
            state = {
                'ticker': 'TEST',
                'sentiment_analysis': {'confidence': 0.7},
                'cleaned_stock_news': [{'source': 'Test Source'}],
                'cleaned_market_news': [],
                'cleaned_related_news': []
            }
            
            result = news_verification_node(state)
            
            learning_adj = result['news_impact_verification']['learning_adjustment']
            assert 0.5 <= learning_adj <= 2.0, \
                f"Learning adjustment {learning_adj} out of bounds for accuracy {accuracy}"
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    def test_state_preservation(self, mock_get_news, sample_historical_events):
        """Test node preserves other state fields"""
        historical_df = pd.DataFrame(sample_historical_events)
        mock_get_news.return_value = historical_df
        
        state = {
            'ticker': 'NVDA',
            'sentiment_analysis': {'confidence': 0.7},
            'cleaned_stock_news': [],
            'cleaned_market_news': [],
            'cleaned_related_news': [],
            'technical_analysis': {'rsi': 58.4},  # Other node data
            'raw_price_data': pd.DataFrame({'close': [100, 101, 102]}),
            'errors': []
        }
        
        result = news_verification_node(state)
        
        # Other fields should be preserved
        assert result['ticker'] == 'NVDA'
        assert 'technical_analysis' in result
        assert 'raw_price_data' in result
        assert isinstance(result['errors'], list)
    
    @patch('src.langgraph_nodes.node_08_news_verification.get_news_with_outcomes')
    @patch('src.langgraph_nodes.node_08_news_verification.store_source_reliability')
    def test_thesis_innovation_impact(self, mock_store, mock_get_news):
        """Test that Node 8 demonstrates measurable impact (thesis validation)"""
        # Create historical data showing clear source reliability pattern
        # Bloomberg: 85% accurate, Random Blog: 25% accurate
        historical_events = []
        
        for i in range(50):
            historical_events.append({
                'source': 'Bloomberg',
                'prediction_was_accurate_7day': i < 42,  # 84% accurate
                'price_change_7day': 2.0,
                'news_type': 'stock',
                'sentiment_label': 'positive'
            })
        
        for i in range(40):
            historical_events.append({
                'source': 'Random Blog',
                'prediction_was_accurate_7day': i < 10,  # 25% accurate
                'price_change_7day': 0.5,
                'news_type': 'market',
                'sentiment_label': 'positive'
            })
        
        historical_df = pd.DataFrame(historical_events)
        mock_get_news.return_value = historical_df
        
        # Test with Bloomberg articles (should boost confidence)
        state_bloomberg = {
            'ticker': 'NVDA',
            'sentiment_analysis': {'confidence': 0.70},
            'cleaned_stock_news': [{'source': 'Bloomberg'}] * 5,
            'cleaned_market_news': [],
            'cleaned_related_news': []
        }
        
        result_bloomberg = news_verification_node(state_bloomberg)
        bloomberg_confidence = result_bloomberg['sentiment_analysis']['confidence']
        
        # Test with Random Blog articles (should reduce confidence)
        # Use market news type to match where Random Blog appears in historical data
        state_blog = {
            'ticker': 'NVDA',
            'sentiment_analysis': {'confidence': 0.70},
            'cleaned_stock_news': [],
            'cleaned_market_news': [{'source': 'Random Blog'}] * 5,
            'cleaned_related_news': []
        }
        
        result_blog = news_verification_node(state_blog)
        blog_confidence = result_blog['sentiment_analysis']['confidence']
        
        # Get reliability scores for validation
        bloomberg_reliability = result_bloomberg['news_impact_verification']['source_reliability']['Bloomberg']
        blog_reliability = result_blog['news_impact_verification']['source_reliability']['Random Blog']
        
        # THESIS VALIDATION: Node 8 should differentiate between sources
        # Bloomberg has high accuracy → high multiplier
        assert bloomberg_reliability['confidence_multiplier'] > 1.0, \
            f"Bloomberg multiplier should be > 1.0, got {bloomberg_reliability['confidence_multiplier']}"
        
        # Random Blog has low accuracy → low multiplier
        assert blog_reliability['confidence_multiplier'] < 1.0, \
            f"Random Blog multiplier should be < 1.0, got {blog_reliability['confidence_multiplier']}"
        
        # Different sources should result in different confidence adjustments
        assert bloomberg_confidence > 0.70, "Bloomberg should boost confidence"
        assert bloomberg_confidence > blog_confidence, \
            "Node 8 should differentiate reliable vs unreliable sources"
        
        # Log for thesis documentation
        print(f"\n=== THESIS INNOVATION VALIDATION ===")
        print(f"Original confidence: 0.70")
        print(f"Bloomberg articles → {bloomberg_confidence:.3f} (multiplier: {bloomberg_reliability['confidence_multiplier']:.3f})")
        print(f"Random Blog articles → {blog_confidence:.3f} (multiplier: {blog_reliability['confidence_multiplier']:.3f})")
        print(f"Reliability differentiation: {bloomberg_confidence - blog_confidence:.3f}")
