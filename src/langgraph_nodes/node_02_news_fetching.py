"""
Node 2: Multi-Source News Fetching
Fetches news from multiple sources in parallel using async operations.

Data Sources (optimized for free tier):
1. Alpha Vantage (Primary - stock news + built-in sentiment, 10-day coverage)
2. Finnhub (Supplement - market news + real-time updates, 3-day coverage)

Runs AFTER: Node 1 (price data), Node 3 (related companies)
Runs BEFORE: Node 5 (sentiment), Node 6 (market context)
Can run in PARALLEL with: Node 4 (technical indicators)
"""

import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from src.utils.config import ALPHA_VANTAGE_API_KEY, FINNHUB_API_KEY
from src.database.db_manager import cache_news, get_cached_news

logger = logging.getLogger(__name__)

# Cache duration for news (6 hours = reasonable for news freshness)
NEWS_CACHE_HOURS = 6


# ============================================================================
# HELPER FUNCTION: Fetch from Alpha Vantage (Primary)
# ============================================================================

async def fetch_alpha_vantage_news_async(
    ticker: str,
    session: aiohttp.ClientSession
) -> List[Dict[str, Any]]:
    """
    Fetch stock news + sentiment from Alpha Vantage API (async).
    
    Provides:
    - Stock-specific news (10-day coverage)
    - Built-in overall sentiment analysis
    - Ticker-specific sentiment scores
    
    Args:
        ticker: Stock ticker symbol
        session: aiohttp session for requests
        
    Returns:
        List of news articles with sentiment data
        Empty list if fetch fails
    """
    try:
        if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == 'your_alpha_vantage_key_here':
            logger.warning("Alpha Vantage API key not configured")
            return []
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': ALPHA_VANTAGE_API_KEY,
            'limit': 200,  # Get more articles
            'sort': 'LATEST'
        }
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"Alpha Vantage API error: {response.status}")
                return []
            
            data = await response.json()
            
            # Check for errors or rate limits
            if 'Note' in data or 'Error Message' in data:
                logger.warning(f"Alpha Vantage: {data.get('Note') or data.get('Error Message')}")
                return []
            
            if 'feed' not in data:
                logger.warning(f"Alpha Vantage: No feed in response for {ticker}")
                return []
            
            feed = data['feed']
            
            # Standardize to our format
            articles = []
            for item in feed:
                # Parse publish time
                time_str = item.get('time_published', '')
                if time_str:
                    try:
                        pub_time = datetime.strptime(time_str, '%Y%m%dT%H%M%S')
                        timestamp = int(pub_time.timestamp())
                    except:
                        timestamp = int(datetime.now().timestamp())
                else:
                    timestamp = int(datetime.now().timestamp())
                
                # Extract ticker-specific sentiment
                ticker_sentiment = None
                ticker_relevance = 0.0
                if 'ticker_sentiment' in item:
                    for ts in item['ticker_sentiment']:
                        if ts.get('ticker') == ticker:
                            ticker_sentiment = ts.get('ticker_sentiment_score', 0.0)
                            ticker_relevance = ts.get('relevance_score', 0.0)
                            break
                
                article = {
                    'headline': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', ''),
                    'url': item.get('url', ''),
                    'datetime': timestamp,
                    'image': item.get('banner_image', ''),
                    'news_type': 'stock',  # Tag as stock-specific news
                    # Alpha Vantage specific fields
                    'overall_sentiment_score': item.get('overall_sentiment_score', 0.0),
                    'overall_sentiment_label': item.get('overall_sentiment_label', 'Neutral'),
                    'ticker_sentiment_score': ticker_sentiment,
                    'ticker_relevance_score': ticker_relevance
                }
                articles.append(article)
            
            logger.info(f"Alpha Vantage: Fetched {len(articles)} news articles for {ticker}")
            return articles
            
    except Exception as e:
        logger.error(f"Failed to fetch Alpha Vantage news for {ticker}: {str(e)}")
        return []


# ============================================================================
# HELPER FUNCTION: Fetch Market News from Finnhub (Supplement)
# ============================================================================

async def fetch_finnhub_market_news_async(
    category: str,
    session: aiohttp.ClientSession
) -> List[Dict[str, Any]]:
    """
    Fetch general market news from Finnhub API (async).
    Supplements Alpha Vantage with broader market context.
    
    Args:
        category: News category ('general', 'forex', 'crypto', 'merger')
        session: aiohttp session for requests
        
    Returns:
        List of market news articles
        Empty list if fetch fails
    """
    try:
        if not FINNHUB_API_KEY or FINNHUB_API_KEY == 'your_finnhub_key_here':
            logger.warning("Finnhub API key not configured")
            return []
        
        url = "https://finnhub.io/api/v1/news"
        params = {
            'category': category,
            'token': FINNHUB_API_KEY
        }
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"Finnhub market news error: {response.status}")
                return []
            
            data = await response.json()
            
            if not data:
                return []
            
            # Standardize to our format
            articles = []
            for item in data:
                article = {
                    'headline': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', ''),
                    'url': item.get('url', ''),
                    'datetime': item.get('datetime', 0),
                    'image': item.get('image', ''),
                    'news_type': 'market',  # Tag as market news
                    'category': item.get('category', category)
                }
                articles.append(article)
            
            logger.info(f"Finnhub: Fetched {len(articles)} market news articles (category: {category})")
            return articles
            
    except Exception as e:
        logger.error(f"Failed to fetch Finnhub market news: {str(e)}")
        return []


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def fetch_all_news_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2: Multi-Source News Fetching
    
    Execution flow:
    1. Check cache for recent news (< 6 hours)
    2. If cached, return cached news
    3. Fetch stock news from Alpha Vantage (primary, with sentiment)
    4. Fetch market news from Finnhub (supplement, broader context)
    5. Combine and deduplicate articles
    6. Cache results for future use
    7. Update state with all news articles
    
    Args:
        state: LangGraph state containing 'ticker' and optionally 'raw_price_data'
        
    Returns:
        Updated state with 'stock_news' and 'market_news' populated
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    try:
        logger.info(f"Node 2: Starting news fetching for {ticker}")
        
        # ====================================================================
        # STEP 1: Define Date Range (match price data if available)
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
                # Default to 10 days (Alpha Vantage coverage)
                from_date = to_date - timedelta(days=10)
                logger.info(f"Node 2: Price data empty, defaulting to 10 days")
        else:
            # Default to 10 days (Alpha Vantage coverage)
            from_date = to_date - timedelta(days=10)
            logger.info(f"Node 2: No price data, defaulting to 10 days")
        
        from_str = from_date.strftime('%Y-%m-%d')
        to_str = to_date.strftime('%Y-%m-%d')
        
        logger.info(f"Node 2: Fetching news from {from_str} to {to_str}")
        
        # ====================================================================
        # STEP 2: Check Cache (only use if has articles)
        # ====================================================================
        cached_stock_news = get_cached_news(ticker, 'stock', max_age_hours=NEWS_CACHE_HOURS)
        cached_market_news = get_cached_news(ticker, 'market', max_age_hours=NEWS_CACHE_HOURS)
        
        # Check if cache has actual articles (not just empty list)
        has_cached_stock = cached_stock_news is not None and len(cached_stock_news) > 0
        has_cached_market = cached_market_news is not None and len(cached_market_news) > 0
        
        if has_cached_stock and has_cached_market:
            logger.info(f"Node 2: Using cached news for {ticker}")
            logger.info(f"  - Stock news: {len(cached_stock_news)} articles")
            logger.info(f"  - Market news: {len(cached_market_news)} articles")
            
            state['stock_news'] = cached_stock_news
            state['market_news'] = cached_market_news
            state['related_company_news'] = []  # No separate related news with Alpha Vantage
            
            elapsed = (datetime.now() - start_time).total_seconds()
            state['node_execution_times']['node_2'] = elapsed
            logger.info(f"Node 2: Completed (cache hit) in {elapsed:.2f}s")
            return state
        
        # ====================================================================
        # STEP 3: Fetch News from Multiple Sources (Async Parallel)
        # ====================================================================
        logger.info(f"Node 2: Cache miss - fetching fresh news from APIs")
        
        async def fetch_all():
            """Async wrapper to fetch all news in parallel"""
            async with aiohttp.ClientSession() as session:
                tasks = [
                    # Primary: Alpha Vantage stock news with sentiment
                    fetch_alpha_vantage_news_async(ticker, session),
                    # Supplement: Finnhub market news
                    fetch_finnhub_market_news_async('general', session)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
        
        # Run async tasks
        results = asyncio.run(fetch_all())
        
        # Extract results
        stock_news = results[0] if isinstance(results[0], list) else []
        market_news = results[1] if isinstance(results[1], list) else []
        
        # ====================================================================
        # STEP 4: Filter by Date Range (if needed)
        # ====================================================================
        from_timestamp = int(from_date.timestamp())
        to_timestamp = int(to_date.timestamp())
        
        stock_news = [
            article for article in stock_news
            if from_timestamp <= article.get('datetime', 0) <= to_timestamp
        ]
        
        market_news = [
            article for article in market_news
            if from_timestamp <= article.get('datetime', 0) <= to_timestamp
        ]
        
        # ====================================================================
        # STEP 5: Cache Results
        # ====================================================================
        try:
            if stock_news:
                cache_news(ticker, 'stock', stock_news)
                logger.info(f"Node 2: Cached {len(stock_news)} stock news articles")
            
            if market_news:
                cache_news(ticker, 'market', market_news)
                logger.info(f"Node 2: Cached {len(market_news)} market news articles")
        except Exception as e:
            logger.warning(f"Node 2: Failed to cache news: {str(e)}")
            # Continue anyway - caching failure is not critical
        
        # ====================================================================
        # STEP 6: Update State
        # ====================================================================
        state['stock_news'] = stock_news
        state['market_news'] = market_news
        state['related_company_news'] = []  # Alpha Vantage doesn't separate related news
        
        total_articles = len(stock_news) + len(market_news)
        elapsed = (datetime.now() - start_time).total_seconds()
        state['node_execution_times']['node_2'] = elapsed
        
        logger.info(f"Node 2: Successfully fetched {total_articles} total articles in {elapsed:.2f}s")
        logger.info(f"Node 2:   Stock news: {len(stock_news)} (Alpha Vantage with sentiment)")
        logger.info(f"Node 2:   Market news: {len(market_news)} (Finnhub supplement)")
        
        return state
        
    except Exception as e:
        logger.error(f"Node 2: Critical error for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        state['errors'].append(f"Node 2: {str(e)}")
        state['stock_news'] = []
        state['market_news'] = []
        state['related_company_news'] = []
        
        elapsed = (datetime.now() - start_time).total_seconds()
        state['node_execution_times']['node_2'] = elapsed
        
        return state
