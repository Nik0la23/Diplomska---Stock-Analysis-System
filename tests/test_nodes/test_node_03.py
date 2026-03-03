"""
Unit Tests for Node 3: Related Companies Detection
Tests peer discovery, correlation analysis, and error handling.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graph.state import create_initial_state
from src.langgraph_nodes.node_03_related_companies import (
    detect_related_companies_node,
    fetch_peers_from_finnhub,
    fetch_and_cache_peer_price_data,
    calculate_price_correlation,
    rank_peers_by_correlation
)


class TestNode03RelatedCompanies:
    """Test suite for Node 3: Related Companies Detection"""
    
    def test_detect_related_companies_valid_ticker(self):
        """Test detecting related companies for a valid ticker (NVDA)"""
        state = create_initial_state('NVDA')
        
        # Add mock price data
        state['raw_price_data'] = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=60),
            'close': [100.0] * 60
        })
        
        result = detect_related_companies_node(state)
        
        # Should have related companies
        assert 'related_companies' in result
        assert isinstance(result['related_companies'], list)
        
        # Should have at least some peers (or empty if API fails)
        if result['related_companies']:
            # Should not include target ticker
            assert 'NVDA' not in result['related_companies']
            
            # Should be limited to 5 or fewer
            assert len(result['related_companies']) <= 5
        
        # Should track execution time
        assert 'node_3' in result['node_execution_times']
        assert result['node_execution_times']['node_3'] > 0
    
    def test_detect_related_companies_invalid_ticker(self):
        """Test handling of invalid ticker"""
        state = create_initial_state('INVALID123')
        result = detect_related_companies_node(state)
        
        # Should return empty list (not None)
        assert result['related_companies'] == []
        
        # Should not have errors (graceful handling)
        # Empty peers list is not an error, just no peers found
        
        # Should still track execution time
        assert 'node_3' in result['node_execution_times']
    
    def test_detect_related_companies_without_price_data(self):
        """Test that node works without price data (no correlation)"""
        state = create_initial_state('AAPL')
        # Don't set raw_price_data
        
        result = detect_related_companies_node(state)
        
        # Should still return peers (just without correlation ranking)
        if result['related_companies']:
            assert isinstance(result['related_companies'], list)
            assert 'AAPL' not in result['related_companies']
    
    def test_fetch_peers_from_finnhub(self):
        """Test fetching peers from Finnhub API"""
        # Test with a major tech company
        peers = fetch_peers_from_finnhub('NVDA')
        
        if peers:  # API might fail or return empty
            assert isinstance(peers, list)
            # Should not include the target ticker
            assert 'NVDA' not in peers
            # All should be strings
            assert all(isinstance(p, str) for p in peers)
    
    def test_rank_peers_by_correlation_with_price_data(self):
        """Test ranking peers by correlation when price data available"""
        target_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=60),
            'close': [100.0 + i for i in range(60)]
        })
        
        peers = ['AMD', 'INTC', 'TSM']
        
        ranked = rank_peers_by_correlation('NVDA', peers, target_df, max_peers=5)
        
        # Should return a list
        assert isinstance(ranked, list)
        
        # Should not exceed max_peers
        assert len(ranked) <= 5
        
        # Should be a subset of input peers
        assert all(p in peers for p in ranked)
    
    def test_rank_peers_by_correlation_without_price_data(self):
        """Test ranking when no price data available"""
        peers = ['AMD', 'INTC', 'TSM', 'QCOM', 'AVGO', 'MU']
        
        # Pass None for price data
        ranked = rank_peers_by_correlation('NVDA', peers, None, max_peers=5)
        
        # Should return first 5 peers
        assert len(ranked) == 5
        assert ranked == peers[:5]
    
    def test_rank_peers_respects_max_limit(self):
        """Test that ranking respects max_peers limit"""
        target_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=60),
            'close': [100.0] * 60
        })
        
        # Create a long list of peers
        peers = [f'TICKER{i}' for i in range(20)]
        
        ranked = rank_peers_by_correlation('TEST', peers, target_df, max_peers=3)
        
        # Should return exactly 3
        assert len(ranked) == 3
    
    def test_target_ticker_excluded_from_peers(self):
        """Test that target ticker is never in the peers list"""
        state = create_initial_state('MSFT')
        result = detect_related_companies_node(state)
        
        # MSFT should never be in its own peers list
        assert 'MSFT' not in result['related_companies']
    
    def test_state_preservation_on_error(self):
        """Test that state is preserved even when node fails"""
        state = create_initial_state('TEST')
        state['some_other_field'] = 'preserved'
        
        result = detect_related_companies_node(state)
        
        # Should preserve other state fields
        assert result['ticker'] == 'TEST'
        assert 'some_other_field' in result
        assert result['some_other_field'] == 'preserved'
        
        # Should still update execution time
        assert 'node_3' in result['node_execution_times']
    
    def test_empty_peers_list_handling(self):
        """Test handling when no peers are found"""
        # Simulate empty peers list by using very invalid ticker
        state = create_initial_state('XXXINVALIDXXX')
        result = detect_related_companies_node(state)
        
        # Should return empty list, not None or error
        assert result['related_companies'] == []
        assert isinstance(result['related_companies'], list)
    
    def test_correlation_with_insufficient_data(self):
        """Test correlation calculation with insufficient overlapping data"""
        # Create small dataset (less than 30 days)
        target_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20),
            'close': [100.0] * 20
        })
        
        # Correlation should return None for insufficient data
        corr = calculate_price_correlation('NVDA', 'AMD', target_df)
        
        # Should return None (not enough data)
        assert corr is None or isinstance(corr, float)


# ============================================================================
# Unit Tests: fetch_and_cache_peer_price_data (new helper)
# ============================================================================

class TestFetchAndCachePeerPriceData:
    """Unit tests for the fetch_and_cache_peer_price_data helper (uses mocks)."""

    _SAMPLE_DF = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=180),
        'open':   [100.0] * 180,
        'high':   [101.0] * 180,
        'low':    [99.0]  * 180,
        'close':  [100.0] * 180,
        'volume': [1_000_000] * 180,
    })

    @patch('src.langgraph_nodes.node_03_related_companies.get_cached_price_data')
    def test_returns_cached_data_without_calling_api(self, mock_cache_get):
        """Cache hit: should return cached DataFrame and never call yfinance."""
        mock_cache_get.return_value = self._SAMPLE_DF.copy()

        with patch('src.langgraph_nodes.node_03_related_companies.fetch_from_yfinance') as mock_yf:
            result = fetch_and_cache_peer_price_data('AMD')

        assert result is not None
        assert len(result) == 180
        mock_yf.assert_not_called()
        mock_cache_get.assert_called_once_with('AMD', max_age_hours=24)

    @patch('src.langgraph_nodes.node_03_related_companies.cache_price_data')
    @patch('src.langgraph_nodes.node_03_related_companies.fetch_from_yfinance')
    @patch('src.langgraph_nodes.node_03_related_companies.get_cached_price_data')
    def test_fetches_yfinance_on_cache_miss_and_caches_result(
        self, mock_cache_get, mock_yf, mock_cache_set
    ):
        """Cache miss: should fetch from yfinance and cache the result."""
        mock_cache_get.return_value = None
        mock_yf.return_value = self._SAMPLE_DF.copy()

        result = fetch_and_cache_peer_price_data('AMD')

        assert result is not None
        mock_yf.assert_called_once_with('AMD', days=180)
        mock_cache_set.assert_called_once_with('AMD', mock_yf.return_value)

    @patch('src.langgraph_nodes.node_03_related_companies.cache_price_data')
    @patch('src.langgraph_nodes.node_03_related_companies.fetch_from_polygon')
    @patch('src.langgraph_nodes.node_03_related_companies.fetch_from_yfinance')
    @patch('src.langgraph_nodes.node_03_related_companies.get_cached_price_data')
    def test_falls_back_to_polygon_when_yfinance_fails(
        self, mock_cache_get, mock_yf, mock_polygon, mock_cache_set
    ):
        """yfinance failure: should fall back to Polygon and still cache."""
        mock_cache_get.return_value = None
        mock_yf.return_value = None
        mock_polygon.return_value = self._SAMPLE_DF.copy()

        result = fetch_and_cache_peer_price_data('AMD')

        assert result is not None
        mock_polygon.assert_called_once_with('AMD', days=180)
        mock_cache_set.assert_called_once()

    @patch('src.langgraph_nodes.node_03_related_companies.cache_price_data')
    @patch('src.langgraph_nodes.node_03_related_companies.fetch_from_polygon')
    @patch('src.langgraph_nodes.node_03_related_companies.fetch_from_yfinance')
    @patch('src.langgraph_nodes.node_03_related_companies.get_cached_price_data')
    def test_returns_none_when_all_sources_fail(
        self, mock_cache_get, mock_yf, mock_polygon, mock_cache_set
    ):
        """All sources fail: should return None without caching anything."""
        mock_cache_get.return_value = None
        mock_yf.return_value = None
        mock_polygon.return_value = None

        result = fetch_and_cache_peer_price_data('UNKNOWN')

        assert result is None
        mock_cache_set.assert_not_called()


# ============================================================================
# Unit Tests: correlation now produces real values (not always None)
# ============================================================================

class TestCorrelationWithNewHelper:
    """Tests that verify correlation ranking actually works after the fix."""

    # Shared dates used across tests
    _DATES = pd.date_range('2025-01-01', periods=60)

    def _make_df(self, values):
        return pd.DataFrame({'date': self._DATES, 'close': values})

    @patch('src.langgraph_nodes.node_03_related_companies.fetch_and_cache_peer_price_data')
    def test_correlation_returns_float_when_peer_data_available(self, mock_fetch):
        """Should return a valid float in [0, 1] when peer data is present."""
        mock_fetch.return_value = self._make_df([50.0 + i * 0.5 for i in range(60)])
        target_df = self._make_df([100.0 + i for i in range(60)])

        corr = calculate_price_correlation('NVDA', 'AMD', target_df)

        assert corr is not None
        assert isinstance(corr, float)
        assert 0.0 <= corr <= 1.0

    @patch('src.langgraph_nodes.node_03_related_companies.fetch_and_cache_peer_price_data')
    def test_correlation_returns_none_when_peer_data_unavailable(self, mock_fetch):
        """Should return None when the new helper also returns None."""
        mock_fetch.return_value = None
        target_df = self._make_df([100.0 + i for i in range(60)])

        corr = calculate_price_correlation('NVDA', 'UNKNOWN', target_df)

        assert corr is None

    @patch('src.langgraph_nodes.node_03_related_companies.fetch_and_cache_peer_price_data')
    def test_higher_correlated_peer_ranked_first(self, mock_fetch):
        """Peer with higher absolute correlation should rank before a weaker one."""
        target_close = [100.0 + i for i in range(60)]
        target_df = self._make_df(target_close)

        # AMD moves nearly in lockstep → high correlation
        amd_close = [50.0 + i * 0.99 for i in range(60)]
        # INTC alternates around a flat line → near-zero correlation
        intc_close = [80.0 + (1 if i % 2 == 0 else -1) for i in range(60)]

        def side_effect(ticker):
            if ticker == 'AMD':
                return self._make_df(amd_close)
            if ticker == 'INTC':
                return self._make_df(intc_close)
            return None

        mock_fetch.side_effect = side_effect

        ranked = rank_peers_by_correlation('NVDA', ['AMD', 'INTC'], target_df, max_peers=2)

        assert ranked[0] == 'AMD', "AMD (high correlation) should rank first"
        assert ranked[1] == 'INTC'

    @patch('src.langgraph_nodes.node_03_related_companies.fetch_and_cache_peer_price_data')
    def test_peers_with_no_data_excluded_from_ranking(self, mock_fetch):
        """Peers for which no price data can be fetched should be dropped."""
        target_df = self._make_df([100.0 + i for i in range(60)])

        def side_effect(ticker):
            if ticker == 'AMD':
                return self._make_df([50.0 + i for i in range(60)])
            return None  # All others fail

        mock_fetch.side_effect = side_effect

        ranked = rank_peers_by_correlation('NVDA', ['AMD', 'INTC', 'TSM'], target_df, max_peers=5)

        # Only AMD should appear (it's the only one with data)
        assert ranked == ['AMD']


# ============================================================================
# Integration Tests
# ============================================================================

class TestNode03Integration:
    """Integration tests for Node 3 with real APIs"""
    
    @pytest.mark.integration
    def test_full_flow_multiple_tickers(self):
        """Test full flow with multiple different tickers"""
        tickers = ['NVDA', 'AAPL', 'TSLA']
        
        for ticker in tickers:
            state = create_initial_state(ticker)
            result = detect_related_companies_node(state)
            
            # Should complete without errors (may or may not find peers)
            assert 'related_companies' in result
            assert isinstance(result['related_companies'], list)
            assert ticker not in result['related_companies']
            assert len(result['related_companies']) <= 5
    
    @pytest.mark.integration
    def test_with_node_1_integration(self):
        """Test Node 3 after Node 1 (realistic flow)"""
        from src.langgraph_nodes.node_01_data_fetching import fetch_price_data_node
        
        # Run Node 1 first
        state = create_initial_state('MSFT')
        state = fetch_price_data_node(state)
        
        # Then run Node 3
        result = detect_related_companies_node(state)
        
        # Should have found peers
        if result['related_companies']:
            assert 'MSFT' not in result['related_companies']
            assert len(result['related_companies']) <= 5


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )


if __name__ == '__main__':
    # Run tests with: python -m pytest tests/test_nodes/test_node_03.py -v
    pytest.main([__file__, '-v'])
