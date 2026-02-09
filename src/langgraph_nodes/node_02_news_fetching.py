"""
Node 2: Multi-Source News Fetching
Fetches news from 3 sources in parallel: stock-specific, market-wide, and related companies.

Runs AFTER: Node 3 (needs related companies list)
Runs BEFORE: Node 9A (early anomaly detection)
Can run in PARALLEL with: Nothing
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import time

from src.utils.config import FINNHUB_API_KEY
from src.database.db_manager import cache_news, get_cached_news

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTION: Fetch Company News from Finnhub (Async)
# ============================================================================

async def fetch_company_news_async(
    ticker: str,
    from_date: str,
    to_date: str,
    session: aiohttp.ClientSession
) -> List[Dict[str, Any]]:
    """
    Fetch company-specific news from Finnhub API (async).
    
    Args:
        ticker: Stock ticker symbol
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        session: aiohttp session for requests
        
    Returns:
        List of news articles with fields: headline, summary, source, url, datetime
        Empty list if fetch fails
        
    Example:
        >>> async with aiohttp.ClientSession() as session:
        >>>     news = await fetch_company_news_async('AAPL', '2024-01-01', '2024-01-07', session)
        >>>     print(len(news))
    """
    try:
        url = f"https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': ticker,
            'from': from_date,
            'to': to_date,
            'token': FINNHUB_API_KEY
        }
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.warning(f"Finnhub company news returned status {response.status} for {ticker}")
                return []
            
            data = await response.json()
            
            if not data or not isinstance(data, list):
                logger.warning(f"No news data returned for {ticker}")
                return []
            
            # Standardize news format
            articles = []
            for item in data:
                articles.append({
                    'headline': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', 'finnhub'),
                    'url': item.get('url', ''),
                    'datetime': item.get('datetime', 0),
                    'category': item.get('category', 'general'),
                    'ticker': ticker,
                    'news_type': 'stock'  # Mark as stock-specific news
                })
            
            logger.info(f"Fetched {len(articles)} company news articles for {ticker}")
            return articles
            
    except Exception as e:
        logger.error(f"Failed to fetch company news for {ticker}: {str(e)}")
        return []


# ============================================================================
# HELPER FUNCTION: Fetch Market News from Finnhub (Async)
# ============================================================================

async def fetch_market_news_async(
    category: str,
    session: aiohttp.ClientSession
) -> List[Dict[str, Any]]:
    """
    Fetch general market news from Finnhub API (async).
    
    Args:
        category: News category ('general', 'forex', 'crypto', 'merger')
        session: aiohttp session for requests
        
    Returns:
        List of market news articles
        Empty list if fetch fails
        
    Example:
        >>> async with aiohttp.ClientSession() as session:
        >>>     news = await fetch_market_news_async('general', session)
        >>>     print(len(news))
    """
    try:
        url = f"https://finnhub.io/api/v1/news"
        params = {
            'category': category,
            'token': FINNHUB_API_KEY
        }
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.warning(f"Finnhub market news returned status {response.status}")
                return []
            
            data = await response.json()
            
            if not data or not isinstance(data, list):
                logger.warning(f"No market news data returned")
                return []
            
            # Standardize news format
            articles = []
            for item in data:
                articles.append({
                    'headline': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', 'finnhub'),
                    'url': item.get('url', ''),
                    'datetime': item.get('datetime', 0),
                    'category': item.get('category', category),
                    'ticker': None,  # Market news not ticker-specific
                    'news_type': 'market'  # Mark as market news
                })
            
            logger.info(f"Fetched {len(articles)} market news articles")
            return articles
            
    except Exception as e:
        logger.error(f"Failed to fetch market news: {str(e)}")
        return []


# ============================================================================
# HELPER FUNCTION: Fetch Related Companies News (Async)
# ============================================================================

async def fetch_related_companies_news_async(
    related_tickers: List[str],
    from_date: str,
    to_date: str,
    session: aiohttp.ClientSession,
    max_articles_per_ticker: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch news for multiple related companies (async, parallel).
    
    Args:
        related_tickers: List of related company tickers
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        session: aiohttp session for requests
        max_articles_per_ticker: Limit articles per ticker
        
    Returns:
        Aggregated list of news from all related companies
        
    Example:
        >>> async with aiohttp.ClientSession() as session:
        >>>     news = await fetch_related_companies_news_async(['AMD', 'INTC'], '2024-01-01', '2024-01-07', session)
        >>>     print(len(news))
    """
    try:
        if not related_tickers:
            logger.info("No related companies to fetch news for")
            return []
        
        logger.info(f"Fetching news for {len(related_tickers)} related companies")
        
        # Fetch news for all related companies in parallel
        tasks = []
        for ticker in related_tickers:
            task = fetch_company_news_async(ticker, from_date, to_date, session)
            tasks.append(task)
            
            # Rate limiting: small delay between launches
            await asyncio.sleep(0.1)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        all_articles = []
        for ticker, result in zip(related_tickers, results):
            if isinstance(result, list):
                # Limit articles per ticker to avoid overwhelming
                articles = result[:max_articles_per_ticker]
                for article in articles:
                    article['news_type'] = 'related'  # Mark as related company news
                all_articles.extend(articles)
            else:
                logger.warning(f"Failed to fetch news for related company {ticker}: {result}")
        
        logger.info(f"Fetched {len(all_articles)} total articles from related companies")
        return all_articles
        
    except Exception as e:
        logger.error(f"Failed to fetch related companies news: {str(e)}")
        return []


# ============================================================================
# HELPER FUNCTION: Check Cache for Recent News
# ============================================================================

def check_news_cache(ticker: str, news_type: str, max_age_hours: int = 6) -> Optional[List[Dict[str, Any]]]:
    """
    Check if we have recent cached news.
    
    Args:
        ticker: Stock ticker (or None for market news)
        news_type: 'stock', 'market', or 'related'
        max_age_hours: Maximum age of cached news
        
    Returns:
        Cached news articles or None
    """
    try:
        cached = get_cached_news(ticker, news_type=news_type, max_age_hours=max_age_hours)
        if cached:
            logger.info(f"Using cached {news_type} news for {ticker or 'market'} ({len(cached)} articles)")
        return cached
    except Exception as e:
        logger.debug(f"Cache check failed: {str(e)}")
        return None


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def fetch_all_news_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2: Multi-Source News Fetching
    
    Execution flow:
    1. Check cache for recent news (< 6 hours)
    2. If not cached, fetch from 3 sources in parallel:
       - Stock-specific news (target company)
       - Market-wide news (general market)
       - Related companies news (competitors)
    3. Aggregate and standardize all news
    4. Cache results for future use
    5. Update state with 3 news lists
    
    Args:
        state: LangGraph state containing 'ticker' and 'related_companies'
        
    Returns:
        Updated state with news lists populated
    """
    start_time = datetime.now()
    ticker = state['ticker']
    related_companies = state.get('related_companies', [])
    
    try:
        logger.info(f"Node 2: Starting news fetching for {ticker}")
        
        if not FINNHUB_API_KEY:
            logger.error("Node 2: Finnhub API key not configured")
            state['errors'].append("Node 2: Finnhub API key missing")
            state['stock_news'] = []
            state['market_news'] = []
            state['related_company_news'] = []
            elapsed = (datetime.now() - start_time).total_seconds()
            state['node_execution_times']['node_2'] = elapsed
            return state
        
        # ====================================================================
        # STEP 1: Define Date Range (match price data or default 6 months)
        # ====================================================================
        to_date = datetime.now()
        
        # If we have price data from Node 1, use its date range
        if state.get('raw_price_data') is not None:
            price_df = state['raw_price_data']
            if not price_df.empty and 'date' in price_df.columns:
                # Use the date range from price data
                from_date = price_df['date'].min()
                # Convert to datetime if it's a Timestamp
                if hasattr(from_date, 'to_pydatetime'):
                    from_date = from_date.to_pydatetime()
                logger.info(f"Node 2: Using date range from price data")
            else:
                # Default to 180 days (6 months)
                from_date = to_date - timedelta(days=180)
                logger.info(f"Node 2: Price data empty, defaulting to 180 days")
        else:
            # Default to 180 days (6 months)
            from_date = to_date - timedelta(days=180)
            logger.info(f"Node 2: No price data, defaulting to 180 days")
        
        from_str = from_date.strftime('%Y-%m-%d')
        to_str = to_date.strftime('%Y-%m-%d')
        
        logger.info(f"Node 2: Fetching news from {from_str} to {to_str}")
        
        # ====================================================================
        # STEP 2: Check Cache (only use if has articles)
        # ====================================================================
        cached_stock_news = check_news_cache(ticker, 'stock', max_age_hours=6)
        cached_market_news = check_news_cache(None, 'market', max_age_hours=6)
        
        # If both are cached AND have articles, skip API calls
        has_cached_stock = cached_stock_news is not None and len(cached_stock_news) > 0
        has_cached_market = cached_market_news is not None and len(cached_market_news) > 0
        
        if has_cached_stock and has_cached_market:
            logger.info("Node 2: Using all cached news (cache hit)")
            state['stock_news'] = cached_stock_news
            state['market_news'] = cached_market_news
            state['related_company_news'] = []  # Related news not typically cached separately
            elapsed = (datetime.now() - start_time).total_seconds()
            state['node_execution_times']['node_2'] = elapsed
            return state
        
        # ====================================================================
        # STEP 3: Fetch News from 3 Sources in Parallel (Async)
        # ====================================================================
        async def fetch_all_async():
            """Async function to fetch all news in parallel"""
            async with aiohttp.ClientSession() as session:
                # Launch 3 tasks in parallel
                task_stock = fetch_company_news_async(ticker, from_str, to_str, session)
                task_market = fetch_market_news_async('general', session)
                task_related = fetch_related_companies_news_async(
                    related_companies, from_str, to_str, session
                )
                
                # Wait for all 3 to complete
                stock_news, market_news, related_news = await asyncio.gather(
                    task_stock,
                    task_market,
                    task_related,
                    return_exceptions=True
                )
                
                # Handle exceptions
                if isinstance(stock_news, Exception):
                    logger.error(f"Stock news fetch failed: {stock_news}")
                    stock_news = []
                
                if isinstance(market_news, Exception):
                    logger.error(f"Market news fetch failed: {market_news}")
                    market_news = []
                
                if isinstance(related_news, Exception):
                    logger.error(f"Related news fetch failed: {related_news}")
                    related_news = []
                
                return stock_news, market_news, related_news
        
        # Run async tasks
        stock_news, market_news, related_news = asyncio.run(fetch_all_async())
        
        # ====================================================================
        # STEP 4: Cache Results
        # ====================================================================
        try:
            if stock_news:
                cache_news(ticker, 'stock', stock_news)
                logger.info(f"Node 2: Cached {len(stock_news)} stock news articles")
            
            if market_news:
                cache_news(None, 'market', market_news)
                logger.info(f"Node 2: Cached {len(market_news)} market news articles")
                
        except Exception as e:
            logger.warning(f"Node 2: Failed to cache news: {str(e)}")
            # Continue anyway - caching failure is not critical
        
        # ====================================================================
        # STEP 5: Update State
        # ====================================================================
        state['stock_news'] = stock_news
        state['market_news'] = market_news
        state['related_company_news'] = related_news
        
        elapsed = (datetime.now() - start_time).total_seconds()
        state['node_execution_times']['node_2'] = elapsed
        
        total_articles = len(stock_news) + len(market_news) + len(related_news)
        logger.info(f"Node 2: Successfully fetched {total_articles} total articles in {elapsed:.2f}s")
        logger.info(f"Node 2:   Stock news: {len(stock_news)}")
        logger.info(f"Node 2:   Market news: {len(market_news)}")
        logger.info(f"Node 2:   Related news: {len(related_news)}")
        
        return state
        
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Node 2: Unexpected error for {ticker}: {str(e)}")
        state['errors'].append(f"Node 2: Unexpected error - {str(e)}")
        state['stock_news'] = []
        state['market_news'] = []
        state['related_company_news'] = []
        elapsed = (datetime.now() - start_time).total_seconds()
        state['node_execution_times']['node_2'] = elapsed
        return state
