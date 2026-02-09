"""
Unit Tests for Node 1: Price Data Fetching
Tests data fetching, caching, fallback mechanisms, and error handling.
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graph.state import create_initial_state
from src.langgraph_nodes.node_01_data_fetching import (
    fetch_price_data_node,
    fetch_from_polygon,
    fetch_from_yfinance,
    validate_price_data
)


class TestNode01PriceDataFetching:
    """Test suite for Node 1: Price Data Fetching"""
    
    def test_fetch_price_data_valid_ticker(self):
        """Test fetching data for a valid ticker (AAPL)"""
        state = create_initial_state('AAPL')
        result = fetch_price_data_node(state)
        
        # Should have price data
        assert result['raw_price_data'] is not None
        assert isinstance(result['raw_price_data'], pd.DataFrame)
        
        # Should have minimum rows
        assert len(result['raw_price_data']) >= 50
        
        # Should have required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in result['raw_price_data'].columns for col in required_cols)
        
        # Should track execution time
        assert 'node_1' in result['node_execution_times']
        assert result['node_execution_times']['node_1'] > 0
        
        # Should not have errors
        assert len(result['errors']) == 0
    
    def test_fetch_price_data_invalid_ticker(self):
        """Test handling of invalid ticker"""
        state = create_initial_state('INVALID123')
        result = fetch_price_data_node(state)
        
        # Should return None for price data
        assert result['raw_price_data'] is None
        
        # Should log error
        assert len(result['errors']) > 0
        assert 'Failed to fetch price data' in result['errors'][0]
        
        # Should still track execution time
        assert 'node_1' in result['node_execution_times']
    
    def test_cache_hit_performance(self):
        """Test that cache improves performance"""
        ticker = 'MSFT'
        
        # First call (fetch from API)
        state1 = create_initial_state(ticker)
        result1 = fetch_price_data_node(state1)
        time1 = result1['node_execution_times']['node_1']
        
        # Second call (should use cache)
        state2 = create_initial_state(ticker)
        result2 = fetch_price_data_node(state2)
        time2 = result2['node_execution_times']['node_1']
        
        # Cache hit should be faster
        assert time2 < time1, f"Cache should be faster: {time2}s vs {time1}s"
        
        # Both should return data
        assert result1['raw_price_data'] is not None
        assert result2['raw_price_data'] is not None
        
        # Data should be the same (sort both by date first since order may vary)
        df1_sorted = result1['raw_price_data'].sort_values('date').reset_index(drop=True)
        df2_sorted = result2['raw_price_data'].sort_values('date').reset_index(drop=True)
        pd.testing.assert_frame_equal(df1_sorted, df2_sorted)
    
    def test_validate_price_data_valid(self):
        """Test data validation with valid data"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=60),
            'open': [100.0] * 60,
            'high': [105.0] * 60,
            'low': [95.0] * 60,
            'close': [102.0] * 60,
            'volume': [1000000] * 60
        })
        
        assert validate_price_data(df, 'TEST', min_rows=50) is True
    
    def test_validate_price_data_insufficient_rows(self):
        """Test data validation with insufficient rows"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30),
            'open': [100.0] * 30,
            'high': [105.0] * 30,
            'low': [95.0] * 30,
            'close': [102.0] * 30,
            'volume': [1000000] * 30
        })
        
        assert validate_price_data(df, 'TEST', min_rows=50) is False
    
    def test_validate_price_data_missing_columns(self):
        """Test data validation with missing columns"""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=60),
            'close': [100.0] * 60
            # Missing open, high, low, volume
        })
        
        assert validate_price_data(df, 'TEST', min_rows=50) is False
    
    def test_validate_price_data_empty(self):
        """Test data validation with empty dataframe"""
        df = pd.DataFrame()
        assert validate_price_data(df, 'TEST', min_rows=50) is False
    
    def test_validate_price_data_none(self):
        """Test data validation with None"""
        assert validate_price_data(None, 'TEST', min_rows=50) is False
    
    def test_state_preservation_on_error(self):
        """Test that state is preserved even when node fails"""
        state = create_initial_state('INVALID123')
        state['some_other_field'] = 'preserved'
        
        result = fetch_price_data_node(state)
        
        # Should preserve other state fields
        assert result['ticker'] == 'INVALID123'
        assert 'some_other_field' in result
        assert result['some_other_field'] == 'preserved'
        
        # Should still update execution time
        assert 'node_1' in result['node_execution_times']
    
    def test_yfinance_fallback(self):
        """Test that yfinance fallback works"""
        # This test assumes Polygon might fail or be unavailable
        df = fetch_from_yfinance('AAPL', days=90)
        
        if df is not None:
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert all(col in df.columns for col in ['date', 'open', 'high', 'low', 'close', 'volume'])
    
    def test_data_format_consistency(self):
        """Test that returned data has consistent format"""
        state = create_initial_state('GOOGL')
        result = fetch_price_data_node(state)
        
        if result['raw_price_data'] is not None:
            df = result['raw_price_data']
            
            # Check column types
            assert pd.api.types.is_datetime64_any_dtype(df['date']) or isinstance(df['date'].iloc[0], datetime)
            assert pd.api.types.is_numeric_dtype(df['open'])
            assert pd.api.types.is_numeric_dtype(df['high'])
            assert pd.api.types.is_numeric_dtype(df['low'])
            assert pd.api.types.is_numeric_dtype(df['close'])
            assert pd.api.types.is_numeric_dtype(df['volume'])
            
            # Check data is sorted by date
            assert df['date'].is_monotonic_increasing or df['date'].is_monotonic_decreasing


# ============================================================================
# Integration Tests
# ============================================================================

class TestNode01Integration:
    """Integration tests for Node 1 with real APIs"""
    
    @pytest.mark.integration
    def test_full_flow_multiple_tickers(self):
        """Test full flow with multiple different tickers"""
        tickers = ['AAPL', 'MSFT', 'TSLA']
        
        for ticker in tickers:
            state = create_initial_state(ticker)
            result = fetch_price_data_node(state)
            
            # Should successfully fetch data for major tickers
            assert result['raw_price_data'] is not None, f"Failed to fetch data for {ticker}"
            assert len(result['raw_price_data']) >= 50, f"Insufficient data for {ticker}"
            assert len(result['errors']) == 0, f"Unexpected errors for {ticker}"


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )


if __name__ == '__main__':
    # Run tests with: python -m pytest tests/test_nodes/test_node_01.py -v
    pytest.main([__file__, '-v'])
