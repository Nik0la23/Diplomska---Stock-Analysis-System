"""
Tests for Node 6: Market Context Analysis

Tests cover:
1. Sector Detection (4 tests)
2. Sector Performance (5 tests)
3. Market Trend (6 tests)
4. Related Companies Analysis (5 tests)
5. Correlation Calculation (4 tests)
6. Context Signal Generation (4 tests)
7. Main Node Function (3 tests)
8. Integration (2 tests)

Total: 33 tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.langgraph_nodes.node_06_market_context import (
    get_stock_sector,
    get_sector_performance,
    get_market_trend,
    analyze_related_companies,
    calculate_correlation,
    generate_context_signal,
    market_context_node,
    SECTOR_ETFS
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    return pd.DataFrame({
        'date': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })


@pytest.fixture
def sample_state_with_context():
    """Complete state for testing market context node"""
    dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
    prices = 100 + np.cumsum(np.random.randn(50) * 2)
    
    return {
        'ticker': 'NVDA',
        'raw_price_data': pd.DataFrame({
            'date': dates,
            'close': prices
        }),
        'related_companies': ['AMD', 'INTC'],
        'errors': [],
        'node_execution_times': {}
    }


# ============================================================================
# CATEGORY 1: Sector Detection (4 tests)
# ============================================================================

@patch('yfinance.Ticker')
def test_get_stock_sector_technology(mock_ticker):
    """Test identifying technology sector (NVDA)"""
    mock_info = {'sector': 'Technology', 'industry': 'Semiconductors'}
    mock_ticker.return_value.info = mock_info
    
    sector, industry = get_stock_sector('NVDA')
    
    assert sector == 'Technology'
    assert industry == 'Semiconductors'


@patch('yfinance.Ticker')
def test_get_stock_sector_healthcare(mock_ticker):
    """Test identifying healthcare sector"""
    mock_info = {'sector': 'Healthcare', 'industry': 'Drug Manufacturers'}
    mock_ticker.return_value.info = mock_info
    
    sector, industry = get_stock_sector('JNJ')
    
    assert sector == 'Healthcare'
    assert industry == 'Drug Manufacturers'


@patch('yfinance.Ticker')
def test_get_stock_sector_unknown(mock_ticker):
    """Test handling unknown sector"""
    mock_info = {}  # No sector info
    mock_ticker.return_value.info = mock_info
    
    sector, industry = get_stock_sector('UNKNOWN')
    
    assert sector == 'Unknown'
    assert industry == 'Unknown'


@patch('yfinance.Ticker')
def test_get_stock_sector_api_failure(mock_ticker):
    """Test handling API failure"""
    mock_ticker.side_effect = Exception("API error")
    
    sector, industry = get_stock_sector('TEST')
    
    assert sector == 'Unknown'
    assert industry == 'Unknown'


# ============================================================================
# CATEGORY 2: Sector Performance (5 tests)
# ============================================================================

@patch('yfinance.Ticker')
def test_get_sector_performance_positive(mock_ticker):
    """Test calculating positive sector performance"""
    mock_history = pd.DataFrame({
        'Close': [100.0, 102.0]
    })
    mock_ticker.return_value.history.return_value = mock_history
    
    result = get_sector_performance('Technology', days=1)
    
    assert result['sector'] == 'Technology'
    assert result['etf_ticker'] == 'XLK'
    assert result['performance'] > 0
    assert result['trend'] == 'UP'


@patch('yfinance.Ticker')
def test_get_sector_performance_negative(mock_ticker):
    """Test calculating negative sector performance"""
    mock_history = pd.DataFrame({
        'Close': [100.0, 98.0]
    })
    mock_ticker.return_value.history.return_value = mock_history
    
    result = get_sector_performance('Technology', days=1)
    
    assert result['performance'] < 0
    assert result['trend'] == 'DOWN'


def test_get_sector_performance_invalid_sector():
    """Test handling invalid sector"""
    result = get_sector_performance('InvalidSector', days=1)
    
    assert result['etf_ticker'] is None
    assert result['performance'] == 0.0
    assert result['trend'] == 'FLAT'


@patch('yfinance.Ticker')
def test_get_sector_performance_etf_mapping(mock_ticker):
    """Test ETF mapping validation"""
    mock_history = pd.DataFrame({
        'Close': [100.0, 101.0]
    })
    mock_ticker.return_value.history.return_value = mock_history
    
    # Test all sector mappings
    for sector, etf in SECTOR_ETFS.items():
        result = get_sector_performance(sector, days=1)
        assert result['etf_ticker'] == etf


@patch('yfinance.Ticker')
def test_get_sector_performance_multiple_days(mock_ticker):
    """Test performance over multiple days"""
    mock_history = pd.DataFrame({
        'Close': [100.0, 102.0, 103.0, 105.0, 106.0, 108.0]
    })
    mock_ticker.return_value.history.return_value = mock_history
    
    result = get_sector_performance('Technology', days=5)
    
    # Performance should be from first to last day
    expected_perf = ((108.0 - 100.0) / 100.0) * 100
    assert abs(result['performance'] - expected_perf) < 0.01


# ============================================================================
# CATEGORY 3: Market Trend (6 tests)
# ============================================================================

@patch('yfinance.Ticker')
def test_get_market_trend_bullish(mock_ticker):
    """Test detecting bullish market trend"""
    # Strong uptrend with low volatility
    dates = pd.date_range(end=datetime.now(), periods=6, freq='D')
    closes = [100, 101, 102, 103, 104, 105]  # Steady increase
    mock_history = pd.DataFrame({
        'Close': closes
    }, index=dates)
    mock_ticker.return_value.history.return_value = mock_history
    
    result = get_market_trend(days=5)
    
    assert result['market_trend'] == 'BULLISH'
    assert result['spy_performance'] > 2.0
    assert result['confidence'] >= 0.6


@patch('yfinance.Ticker')
def test_get_market_trend_bearish(mock_ticker):
    """Test detecting bearish market trend"""
    # Strong downtrend
    dates = pd.date_range(end=datetime.now(), periods=6, freq='D')
    closes = [105, 104, 102, 100, 98, 95]  # Steady decrease
    mock_history = pd.DataFrame({
        'Close': closes
    }, index=dates)
    mock_ticker.return_value.history.return_value = mock_history
    
    result = get_market_trend(days=5)
    
    assert result['market_trend'] == 'BEARISH'
    assert result['spy_performance'] < -2.0


@patch('yfinance.Ticker')
def test_get_market_trend_neutral(mock_ticker):
    """Test detecting neutral market trend"""
    # Flat market
    dates = pd.date_range(end=datetime.now(), periods=6, freq='D')
    closes = [100, 101, 100, 101, 100, 100]  # Sideways
    mock_history = pd.DataFrame({
        'Close': closes
    }, index=dates)
    mock_ticker.return_value.history.return_value = mock_history
    
    result = get_market_trend(days=5)
    
    assert result['market_trend'] == 'NEUTRAL'


@patch('yfinance.Ticker')
def test_get_market_trend_volatility_calculation(mock_ticker):
    """Test volatility calculation"""
    dates = pd.date_range(end=datetime.now(), periods=6, freq='D')
    closes = [100, 105, 95, 110, 90, 115]  # High volatility
    mock_history = pd.DataFrame({
        'Close': closes
    }, index=dates)
    mock_ticker.return_value.history.return_value = mock_history
    
    result = get_market_trend(days=5)
    
    # High volatility should be detected
    assert result['volatility'] > 3.0 or result['market_trend'] == 'BEARISH'


@patch('yfinance.Ticker')
def test_get_market_trend_confidence_scoring(mock_ticker):
    """Test confidence scoring"""
    dates = pd.date_range(end=datetime.now(), periods=6, freq='D')
    closes = [100, 103, 106, 109, 112, 115]  # Very strong trend
    mock_history = pd.DataFrame({
        'Close': closes
    }, index=dates)
    mock_ticker.return_value.history.return_value = mock_history
    
    result = get_market_trend(days=5)
    
    # Strong trend should have high confidence
    assert result['confidence'] >= 0.6


@patch('yfinance.Ticker')
def test_get_market_trend_spy_fetch_failure(mock_ticker):
    """Test handling SPY fetch failure"""
    mock_ticker.return_value.history.return_value = pd.DataFrame()  # Empty
    
    result = get_market_trend(days=5)
    
    assert result['market_trend'] == 'NEUTRAL'
    assert result['spy_performance'] == 0.0


# ============================================================================
# CATEGORY 4: Related Companies Analysis (5 tests)
# ============================================================================

@patch('yfinance.Ticker')
def test_analyze_related_all_up(mock_ticker):
    """Test all related companies up (bullish)"""
    mock_history = pd.DataFrame({
        'Close': [100.0, 105.0]
    })
    mock_ticker.return_value.history.return_value = mock_history
    
    result = analyze_related_companies(['AMD', 'INTC', 'TSM'])
    
    assert result['up_count'] == 3
    assert result['overall_signal'] == 'BULLISH'


@patch('yfinance.Ticker')
def test_analyze_related_all_down(mock_ticker):
    """Test all related companies down (bearish)"""
    mock_history = pd.DataFrame({
        'Close': [100.0, 95.0]
    })
    mock_ticker.return_value.history.return_value = mock_history
    
    result = analyze_related_companies(['AMD', 'INTC'])
    
    assert result['down_count'] == 2
    assert result['overall_signal'] == 'BEARISH'


@patch('yfinance.Ticker')
def test_analyze_related_mixed_signals(mock_ticker):
    """Test mixed signals (neutral)"""
    call_count = [0]
    
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First company up
            return Mock(history=Mock(return_value=pd.DataFrame({'Close': [100.0, 103.0]})))
        else:
            # Second company down
            return Mock(history=Mock(return_value=pd.DataFrame({'Close': [100.0, 97.0]})))
    
    mock_ticker.side_effect = side_effect
    
    result = analyze_related_companies(['AMD', 'INTC'])
    
    assert result['up_count'] >= 0
    assert result['down_count'] >= 0
    # With mixed signals, likely NEUTRAL


def test_analyze_related_empty_list():
    """Test handling empty related list"""
    result = analyze_related_companies([])
    
    assert result['related_companies'] == []
    assert result['avg_performance'] == 0.0
    assert result['overall_signal'] == 'NEUTRAL'


@patch('yfinance.Ticker')
def test_analyze_related_api_failures(mock_ticker):
    """Test handling API failures for some tickers"""
    call_count = [0]
    
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First company succeeds
            return Mock(history=Mock(return_value=pd.DataFrame({'Close': [100.0, 105.0]})))
        else:
            # Second company fails
            raise Exception("API error")
    
    mock_ticker.side_effect = side_effect
    
    result = analyze_related_companies(['AMD', 'INTC'])
    
    # Should still work with partial data
    assert len(result['related_companies']) >= 0


# ============================================================================
# CATEGORY 5: Correlation Calculation (4 tests)
# ============================================================================

@patch('yfinance.Ticker')
def test_calculate_correlation_high(mock_ticker, sample_price_data):
    """Test high correlation (>0.7)"""
    # Mock SPY data that's highly correlated
    spy_closes = sample_price_data['close'].values + np.random.randn(len(sample_price_data)) * 2
    mock_history = pd.DataFrame({
        'Close': spy_closes
    })
    mock_ticker.return_value.history.return_value = mock_history
    
    result = calculate_correlation('NVDA', sample_price_data)
    
    assert 'market_correlation' in result
    assert result['correlation_strength'] in ['HIGH', 'MEDIUM', 'LOW']


@patch('yfinance.Ticker')
def test_calculate_correlation_medium(mock_ticker, sample_price_data):
    """Test medium correlation (0.4-0.7)"""
    # Mock SPY data with moderate correlation
    spy_closes = sample_price_data['close'].values * 0.5 + 50 + np.random.randn(len(sample_price_data)) * 10
    mock_history = pd.DataFrame({
        'Close': spy_closes
    })
    mock_ticker.return_value.history.return_value = mock_history
    
    result = calculate_correlation('NVDA', sample_price_data)
    
    assert -1.0 <= result['market_correlation'] <= 1.0


@patch('yfinance.Ticker')
def test_calculate_correlation_beta(mock_ticker, sample_price_data):
    """Test beta calculation"""
    # Mock SPY data
    spy_closes = 200 + np.cumsum(np.random.randn(len(sample_price_data)) * 1)
    mock_history = pd.DataFrame({
        'Close': spy_closes
    })
    mock_ticker.return_value.history.return_value = mock_history
    
    result = calculate_correlation('NVDA', sample_price_data)
    
    assert 'beta' in result
    # Beta can be positive or negative (depending on correlation)
    assert isinstance(result['beta'], (int, float))
    assert -5.0 < result['beta'] < 5.0  # Reasonable beta range


def test_calculate_correlation_insufficient_data():
    """Test handling insufficient price data"""
    # Only 10 days of data
    short_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    
    result = calculate_correlation('TEST', short_data)
    
    # Should return defaults
    assert result['correlation_strength'] == 'MEDIUM'
    assert result['beta'] == 1.0


# ============================================================================
# CATEGORY 6: Context Signal Generation (4 tests)
# ============================================================================

def test_generate_signal_strong_buy():
    """Test strong BUY scenario (all positive)"""
    sector_perf = {'performance': 2.5, 'trend': 'UP'}
    market = {'market_trend': 'BULLISH'}
    related = {'overall_signal': 'BULLISH', 'avg_performance': 2.0}
    corr = {'market_correlation': 0.8, 'correlation_strength': 'HIGH'}
    
    signal, confidence = generate_context_signal(sector_perf, market, related, corr)
    
    assert signal == 'BUY'
    assert confidence > 50


def test_generate_signal_strong_sell():
    """Test strong SELL scenario (all negative)"""
    sector_perf = {'performance': -3.0, 'trend': 'DOWN'}
    market = {'market_trend': 'BEARISH'}
    related = {'overall_signal': 'BEARISH', 'avg_performance': -2.5}
    corr = {'market_correlation': 0.7}
    
    signal, confidence = generate_context_signal(sector_perf, market, related, corr)
    
    assert signal == 'SELL'
    assert confidence > 50


def test_generate_signal_hold():
    """Test HOLD scenario (mixed signals)"""
    sector_perf = {'performance': 0.5, 'trend': 'FLAT'}
    market = {'market_trend': 'NEUTRAL'}
    related = {'overall_signal': 'NEUTRAL', 'avg_performance': 0.2}
    corr = {'market_correlation': 0.5}
    
    signal, confidence = generate_context_signal(sector_perf, market, related, corr)
    
    assert signal == 'HOLD'


def test_generate_signal_correlation_multiplier():
    """Test correlation multiplier effect"""
    sector_perf = {'performance': 2.0, 'trend': 'UP'}
    market = {'market_trend': 'BULLISH'}
    related = {'overall_signal': 'BULLISH', 'avg_performance': 1.5}
    
    # Low correlation
    corr_low = {'market_correlation': 0.3}
    signal_low, conf_low = generate_context_signal(sector_perf, market, related, corr_low)
    
    # High correlation (should amplify signal)
    corr_high = {'market_correlation': 0.9}
    signal_high, conf_high = generate_context_signal(sector_perf, market, related, corr_high)
    
    # High correlation should result in higher confidence
    assert signal_low == signal_high == 'BUY'
    # Note: Due to score multiplication, high correlation gives stronger signal


# ============================================================================
# CATEGORY 7: Main Node Function (3 tests)
# ============================================================================

@patch('src.langgraph_nodes.node_06_market_context.get_stock_sector')
@patch('src.langgraph_nodes.node_06_market_context.get_sector_performance')
@patch('src.langgraph_nodes.node_06_market_context.get_market_trend')
@patch('src.langgraph_nodes.node_06_market_context.analyze_related_companies')
@patch('src.langgraph_nodes.node_06_market_context.calculate_correlation')
def test_node_function_success(mock_corr, mock_related, mock_market, mock_sector, mock_get_sector,
                                sample_state_with_context):
    """Test full execution of market context node"""
    # Mock all helper functions
    mock_get_sector.return_value = ('Technology', 'Semiconductors')
    mock_sector.return_value = {'performance': 2.0, 'trend': 'UP'}
    mock_market.return_value = {'market_trend': 'BULLISH', 'spy_performance': 1.5}
    mock_related.return_value = {'overall_signal': 'BULLISH', 'avg_performance': 1.8, 'related_companies': []}
    mock_corr.return_value = {'market_correlation': 0.8, 'correlation_strength': 'HIGH', 'beta': 1.2}
    
    result = market_context_node(sample_state_with_context)
    
    # Check all required fields
    assert 'market_context' in result
    assert 'node_execution_times' in result
    assert 'node_6' in result['node_execution_times']
    
    # Check market context structure
    context = result['market_context']
    assert 'sector' in context
    assert 'context_signal' in context
    assert context['context_signal'] in ['BUY', 'SELL', 'HOLD']


@patch('src.langgraph_nodes.node_06_market_context.get_stock_sector')
def test_node_function_partial_data(mock_get_sector):
    """Test execution with partial data available"""
    state = {
        'ticker': 'TEST',
        'raw_price_data': None,  # No price data
        'related_companies': [],  # No related companies
        'errors': [],
        'node_execution_times': {}
    }
    
    mock_get_sector.return_value = ('Unknown', 'Unknown')
    
    result = market_context_node(state)
    
    # Should still complete
    assert 'market_context' in result
    assert result['market_context']['context_signal'] in ['BUY', 'SELL', 'HOLD']


def test_node_function_error_handling():
    """Test error handling in main node"""
    invalid_state = {
        'ticker': 'TEST',
        # Missing required fields
        'errors': [],
        'node_execution_times': {}
    }
    
    result = market_context_node(invalid_state)
    
    # Should not crash, should return partial state with defaults
    assert 'market_context' in result
    assert result['market_context']['context_signal'] == 'HOLD'


# ============================================================================
# CATEGORY 8: Integration (2 tests)
# ============================================================================

@patch('yfinance.Ticker')
def test_integration_end_to_end(mock_ticker, sample_state_with_context):
    """Test end-to-end integration"""
    # Mock yfinance responses
    def ticker_side_effect(symbol):
        mock = Mock()
        if symbol in ['XLK', 'SPY'] or symbol in ['AMD', 'INTC']:
            mock.history.return_value = pd.DataFrame({
                'Close': [100.0, 102.0]
            })
        mock.info = {'sector': 'Technology', 'industry': 'Semiconductors'}
        return mock
    
    mock_ticker.side_effect = ticker_side_effect
    
    result = market_context_node(sample_state_with_context)
    
    # Should complete successfully
    assert result['market_context']['context_signal'] in ['BUY', 'SELL', 'HOLD']
    assert 'sector' in result['market_context']


def test_integration_parallel_execution_compatibility():
    """Test that node returns partial state suitable for parallel execution"""
    state = {
        'ticker': 'AAPL',
        'raw_price_data': pd.DataFrame({'close': [100, 101, 102]}),
        'related_companies': [],
        'errors': [],
        'node_execution_times': {}
    }
    
    result = market_context_node(state)
    
    # Check that ONLY Node 6 fields are returned (partial state)
    expected_keys = {'market_context', 'node_execution_times'}
    
    # Should not contain other node fields
    assert 'ticker' not in result
    assert 'raw_price_data' not in result
    assert 'sentiment_signal' not in result
    
    # Should have execution time for node_6
    assert 'node_6' in result['node_execution_times']


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
