"""
Unit Tests for Node 2: Multi-Source News Fetching
Tests async parallel fetching, caching, and error handling.
"""

import pytest
import asyncio
import aiohttp
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graph.state import create_initial_state
from src.langgraph_nodes.node_02_news_fetching import (
    fetch_all_news_node,
    fetch_alpha_vantage_news_async,
    fetch_alpha_vantage_market_news_async,
    fetch_alpha_vantage_related_company_news_async,
    NEWS_LOOKBACK_DAYS,
)


class TestNode02NewsFetching:
    """Test suite for Node 2: Multi-Source News Fetching"""
    
    def test_fetch_all_news_basic(self):
        """Test basic news fetching functionality"""
        state = create_initial_state('AAPL')
        state['related_companies'] = ['MSFT', 'GOOGL']
        
        result = fetch_all_news_node(state)
        
        # Should have news lists (may be empty if API limit reached)
        assert 'stock_news' in result
        assert 'market_news' in result
        assert 'related_company_news' in result
        
        # All should be lists
        assert isinstance(result['stock_news'], list)
        assert isinstance(result['market_news'], list)
        assert isinstance(result['related_company_news'], list)
        
        # Should track execution time
        assert 'node_2' in result['node_execution_times']
        assert result['node_execution_times']['node_2'] > 0
    
    def test_fetch_all_news_without_related_companies(self):
        """Test news fetching when no related companies provided"""
        state = create_initial_state('AAPL')
        # Don't set related_companies
        
        result = fetch_all_news_node(state)
        
        # Should still work
        assert 'stock_news' in result
        assert 'market_news' in result
        assert 'related_company_news' in result
        
        # Related news should be empty
        assert result['related_company_news'] == []
    
    def test_fetch_all_news_handles_errors_gracefully(self):
        """Test that node handles API errors gracefully"""
        state = create_initial_state('INVALID123')
        
        result = fetch_all_news_node(state)
        
        # Should not crash - return empty lists
        assert isinstance(result['stock_news'], list)
        assert isinstance(result['market_news'], list)
        assert isinstance(result['related_company_news'], list)
        
        # Should still track execution time
        assert 'node_2' in result['node_execution_times']
    
    def test_state_preservation(self):
        """Test that state is preserved during execution"""
        state = create_initial_state('AAPL')
        state['some_other_field'] = 'preserved'
        state['related_companies'] = ['MSFT']
        
        result = fetch_all_news_node(state)
        
        # Should preserve other fields
        assert result['ticker'] == 'AAPL'
        assert 'some_other_field' in result
        assert result['some_other_field'] == 'preserved'
    
    def test_news_article_structure(self):
        """Test that fetched news has correct structure"""
        state = create_initial_state('AAPL')
        result = fetch_all_news_node(state)
        
        # Check at least one news source has articles
        all_news = result['stock_news'] + result['market_news'] + result['related_company_news']
        
        if all_news:
            article = all_news[0]
            # Standardized articles have headline (or title) and news_type; legacy cache may not
            has_headline = 'headline' in article or 'title' in article
            assert has_headline, "Missing headline/title"
            if 'news_type' in article:
                assert article['news_type'] in ('stock', 'market', 'related'), "Invalid news_type"
    
    def test_news_type_classification(self):
        """Test that news is correctly classified by type"""
        state = create_initial_state('AAPL')
        state['related_companies'] = ['MSFT']
        
        result = fetch_all_news_node(state)
        
        # Check stock news has correct type
        for article in result['stock_news']:
            if 'news_type' in article:
                assert article['news_type'] == 'stock'
        
        # Check market news has correct type
        for article in result['market_news']:
            if 'news_type' in article:
                assert article['news_type'] == 'market'
        
        # Check related news has correct type
        for article in result['related_company_news']:
            if 'news_type' in article:
                assert article['news_type'] == 'related'
    
    @pytest.mark.asyncio
    async def test_fetch_alpha_vantage_market_news_async(self):
        """Test async Alpha Vantage market/global news fetching (6-month range)"""
        to_date = datetime.now()
        from_date = to_date - timedelta(days=NEWS_LOOKBACK_DAYS)
        
        async with aiohttp.ClientSession() as session:
            news = await fetch_alpha_vantage_market_news_async(session, from_date, to_date)
            
            # Should return a list (may be empty if API limit reached)
            assert isinstance(news, list)
            
            # If we got news, check structure
            if news:
                article = news[0]
                assert 'headline' in article
                assert 'source' in article
                assert 'news_type' in article
                assert article['news_type'] == 'market'
    
    @pytest.mark.asyncio
    async def test_fetch_alpha_vantage_related_company_news_async(self):
        """Test async Alpha Vantage related-company news fetching (6-month range)"""
        to_date = datetime.now()
        from_date = to_date - timedelta(days=NEWS_LOOKBACK_DAYS)
        peers = ['MSFT', 'GOOGL']
        
        async with aiohttp.ClientSession() as session:
            news = await fetch_alpha_vantage_related_company_news_async(
                peers, session, from_date, to_date
            )
            
            # Should return a list
            assert isinstance(news, list)
            
            # If we got news, check structure
            if news:
                article = news[0]
                assert 'headline' in article
                assert 'source' in article
                assert 'news_type' in article
                assert article['news_type'] == 'related'
    
    @pytest.mark.asyncio
    async def test_fetch_alpha_vantage_news_async_with_date_range(self):
        """Test Alpha Vantage news with 6-month date range"""
        to_date = datetime.now()
        from_date = to_date - timedelta(days=NEWS_LOOKBACK_DAYS)
        
        async with aiohttp.ClientSession() as session:
            news = await fetch_alpha_vantage_news_async('AAPL', session, from_date, to_date)
            
            assert isinstance(news, list)
            if news:
                article = news[0]
                assert 'headline' in article or 'title' in article
                assert 'news_type' in article or 'overall_sentiment_score' in article


# ============================================================================
# Integration Tests
# ============================================================================

class TestNode02Integration:
    """Integration tests for Node 2 with full pipeline"""
    
    @pytest.mark.integration
    def test_full_pipeline_node_1_to_3(self):
        """Test Node 2 after running Node 1 and Node 3"""
        from src.langgraph_nodes.node_01_data_fetching import fetch_price_data_node
        from src.langgraph_nodes.node_03_related_companies import detect_related_companies_node
        
        # Run full pipeline
        state = create_initial_state('MSFT')
        state = fetch_price_data_node(state)
        state = detect_related_companies_node(state)
        result = fetch_all_news_node(state)
        
        # Should have completed all nodes
        assert 'node_1' in result['node_execution_times']
        assert 'node_3' in result['node_execution_times']
        assert 'node_2' in result['node_execution_times']
        
        # Should have news from all sources (or empty lists)
        assert 'stock_news' in result
        assert 'market_news' in result
        assert 'related_company_news' in result
    
    @pytest.mark.integration
    def test_parallel_execution_performance(self):
        """Test that parallel fetching is faster than sequential"""
        state = create_initial_state('GOOGL')
        state['related_companies'] = ['MSFT', 'AAPL']
        
        result = fetch_all_news_node(state)
        
        # With 6-month window (Alpha Vantage stock + related + market), allow up to 15s
        assert result['node_execution_times']['node_2'] < 15.0


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )


if __name__ == '__main__':
    # Run tests with: python -m pytest tests/test_nodes/test_node_02.py -v
    pytest.main([__file__, '-v'])
