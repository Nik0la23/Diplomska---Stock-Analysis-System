"""
Node 2: Multi-Source News Fetching
Fetches news from multiple sources in parallel using async operations.

Data Sources (6-month historical window):
1. Alpha Vantage (Primary - stock news + built-in sentiment, date range: 6 months)
2. Finnhub (Company news with date range: 6 months; market news as supplement)

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
from src.database.db_manager import (
    cache_news,
    compute_and_store_ticker_stats,
    get_cached_news,
    get_latest_news_date,
    get_news_for_ticker,
)

logger = logging.getLogger(__name__)

# Cache duration for news (6 hours = reasonable for news freshness)
NEWS_CACHE_HOURS = 6

# News lookback: 6 months (align with Node 1 price data window)
NEWS_LOOKBACK_DAYS = 180

# Overlap when doing incremental fetch (re-fetch last N days for late-arriving articles)
NEWS_OVERLAP_DAYS = 3


# ============================================================================
# HELPER FUNCTION: Fetch from Alpha Vantage (Primary)
# ============================================================================

async def fetch_alpha_vantage_news_async(
    ticker: str,
    session: aiohttp.ClientSession,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Fetch stock news + sentiment from Alpha Vantage API (async).
    
    Provides:
    - Stock-specific news (6-month coverage when date range supplied)
    - Built-in overall sentiment analysis
    - Ticker-specific sentiment scores
    
    Args:
        ticker: Stock ticker symbol
        session: aiohttp session for requests
        from_date: Start of date range (6 months back). Optional.
        to_date: End of date range (today). Optional.
        
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
            'limit': 1000,  # Max for historical range (was 200)
            'sort': 'LATEST'
        }
        # Add date range for 6-month historical pull (format: YYYYMMDDTHHMMSS)
        if from_date is not None and to_date is not None:
            params['time_from'] = from_date.strftime('%Y%m%dT0000')
            params['time_to'] = to_date.strftime('%Y%m%dT2359')
            logger.info(f"Alpha Vantage: Requesting news from {params['time_from']} to {params['time_to']}")
        
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
# HELPER FUNCTION: Fetch Company News from Finnhub (with date range)
# ============================================================================

async def fetch_finnhub_company_news_async(
    ticker: str,
    from_date: datetime,
    to_date: datetime,
    session: aiohttp.ClientSession
) -> List[Dict[str, Any]]:
    """
    Fetch company-specific news from Finnhub API (async) for a 6-month window.
    
    Args:
        ticker: Stock ticker symbol
        from_date: Start of date range
        to_date: End of date range
        session: aiohttp session for requests
        
    Returns:
        List of news articles (ticker-specific, with date range)
        Empty list if fetch fails
    """
    try:
        if not FINNHUB_API_KEY or FINNHUB_API_KEY == 'your_finnhub_key_here':
            logger.warning("Finnhub API key not configured")
            return []
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': ticker,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': FINNHUB_API_KEY
        }
        
        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"Finnhub company news error: {response.status}")
                return []
            
            data = await response.json()
            
            if not data:
                logger.info(f"Finnhub: No company news for {ticker} in date range")
                return []
            
            articles = []
            for item in data:
                # Finnhub company-news returns datetime as Unix timestamp (seconds)
                ts = item.get('datetime', 0)
                if isinstance(ts, (int, float)):
                    datetime_val = ts
                else:
                    datetime_val = int(datetime.now().timestamp())
                
                article = {
                    'headline': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('source', ''),
                    'url': item.get('url', ''),
                    'datetime': datetime_val,
                    'image': item.get('image', ''),
                    'news_type': 'stock',  # Company-specific = stock
                    'category': item.get('category', 'company')
                }
                articles.append(article)
            
            logger.info(f"Finnhub: Fetched {len(articles)} company news articles for {ticker} (6-month range)")
            return articles
            
    except Exception as e:
        logger.error(f"Failed to fetch Finnhub company news for {ticker}: {str(e)}")
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
    Supplements with broader market context (recent articles; no date range in API).
    
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
        
        to_date = datetime.now()
        
        # ====================================================================
        # STEP 1: Same-day cache (hybrid) — avoid API if we have fresh data
        # ====================================================================
        cached_stock_news = get_cached_news(ticker, 'stock', max_age_hours=NEWS_CACHE_HOURS)
        cached_market_news = get_cached_news(ticker, 'market', max_age_hours=NEWS_CACHE_HOURS)
        has_cached_stock = cached_stock_news is not None and len(cached_stock_news) > 0
        has_cached_market = cached_market_news is not None and len(cached_market_news) > 0
        
        if has_cached_stock and has_cached_market:
            logger.info(f"Node 2: Using cached news for {ticker}")
            state['stock_news'] = cached_stock_news
            state['market_news'] = cached_market_news
            state['related_company_news'] = []
            # Read stored velocity baseline (no DB scan needed — TTL-cached)
            from src.database.db_manager import get_ticker_stats
            _cached_stats = get_ticker_stats(ticker)
            state['news_fetch_metadata'] = {
                'from_date': None,
                'to_date': to_date.isoformat(),
                'fetch_window_days': 0,
                'newly_fetched_count': 0,  # No new articles - cache hit
                'total_in_state': len(cached_stock_news) + len(cached_market_news),
                'newly_fetched_stock': 0,
                'newly_fetched_market': 0,
                'was_cache_hit': True,
                'daily_article_avg': _cached_stats['daily_article_avg'] if _cached_stats else None,
            }
            elapsed = (datetime.now() - start_time).total_seconds()
            state['node_execution_times']['node_2'] = elapsed
            logger.info(f"Node 2: Completed (cache hit) in {elapsed:.2f}s")
            return state
        
        # ====================================================================
        # STEP 2: Multi-day gap — check DB for latest news date, load if current
        # ====================================================================
        latest_news_date = get_latest_news_date(ticker)
        today = to_date.date() if hasattr(to_date, 'date') else to_date
        
        if latest_news_date is not None:
            latest_d = latest_news_date.date() if hasattr(latest_news_date, 'date') else latest_news_date
            days_since = (today - latest_d).days
            if days_since <= 1:
                stock_from_db = get_news_for_ticker(ticker, 'stock', days=NEWS_LOOKBACK_DAYS)
                market_from_db = get_news_for_ticker(ticker, 'market', days=NEWS_LOOKBACK_DAYS)
                if stock_from_db or market_from_db:
                    logger.info(f"Node 2: News current for {ticker}, loaded from DB (no API) — stock: {len(stock_from_db)}, market: {len(market_from_db)}")
                    state['stock_news'] = stock_from_db
                    state['market_news'] = market_from_db
                    state['related_company_news'] = []
                    from src.database.db_manager import get_ticker_stats
                    _db_stats = get_ticker_stats(ticker)
                    state['news_fetch_metadata'] = {
                        'from_date': None,
                        'to_date': to_date.isoformat(),
                        'fetch_window_days': 0,
                        'newly_fetched_count': 0,  # No new articles - loaded from DB
                        'total_in_state': len(stock_from_db) + len(market_from_db),
                        'newly_fetched_stock': 0,
                        'newly_fetched_market': 0,
                        'was_db_load': True,
                        'daily_article_avg': _db_stats['daily_article_avg'] if _db_stats else None,
                    }
                    elapsed = (datetime.now() - start_time).total_seconds()
                    state['node_execution_times']['node_2'] = elapsed
                    return state
        
        # ====================================================================
        # STEP 3: Define date range — full 6 months or incremental + overlap
        # ====================================================================
        if latest_news_date is None:
            from_date = to_date - timedelta(days=NEWS_LOOKBACK_DAYS)
            logger.info(f"Node 2: First run for {ticker}, fetching 6-month window")
        else:
            latest_d = latest_news_date.date() if hasattr(latest_news_date, 'date') else latest_news_date
            from_date = (datetime.combine(latest_d, datetime.min.time()) - timedelta(days=NEWS_OVERLAP_DAYS))
            logger.info(f"Node 2: Incremental fetch for {ticker} from {from_date.date()} (incl. {NEWS_OVERLAP_DAYS}-day overlap)")
        
        from_str = from_date.strftime('%Y-%m-%d')
        to_str = to_date.strftime('%Y-%m-%d')
        logger.info(f"Node 2: Fetching news from {from_str} to {to_str}")
        
        # ====================================================================
        # STEP 4: Fetch News from Multiple Sources (Async Parallel)
        # ====================================================================
        
        async def fetch_all():
            """Async wrapper to fetch all news in parallel with date range"""
            async with aiohttp.ClientSession() as session:
                tasks = [
                    # Alpha Vantage: stock news + sentiment (with time_from, time_to)
                    fetch_alpha_vantage_news_async(ticker, session, from_date, to_date),
                    # Finnhub: company news for ticker (with from, to)
                    fetch_finnhub_company_news_async(ticker, from_date, to_date, session),
                    # Finnhub: general market news (recent; filtered by date below)
                    fetch_finnhub_market_news_async('general', session)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
        
        # Run async tasks
        results = asyncio.run(fetch_all())
        
        # Extract results: merge Alpha Vantage + Finnhub company into stock_news
        av_news = results[0] if isinstance(results[0], list) else []
        finnhub_company_news = results[1] if isinstance(results[1], list) else []
        market_news = results[2] if isinstance(results[2], list) else []
        
        # Merge stock news: Alpha Vantage (has sentiment) + Finnhub company (6-month)
        # Deduplicate by headline to avoid same story from both sources
        seen_headlines = set()
        stock_news = []
        for article in av_news + finnhub_company_news:
            key = (article.get('headline', '') or '').strip() or article.get('url', '')
            if key and key not in seen_headlines:
                seen_headlines.add(key)
                stock_news.append(article)
        logger.info(f"Node 2: Stock news merged: {len(av_news)} Alpha Vantage + {len(finnhub_company_news)} Finnhub company → {len(stock_news)} total")

        # Track how many articles were newly fetched from APIs (before date filtering)
        newly_fetched_stock_count = len(av_news) + len(finnhub_company_news)
        newly_fetched_market_count = len(market_news)
        newly_fetched_total = newly_fetched_stock_count + newly_fetched_market_count
        logger.info(f"Node 2: Newly fetched from APIs: {newly_fetched_total} articles ({newly_fetched_stock_count} stock + {newly_fetched_market_count} market)")
        
        # ====================================================================
        # Filter by Date Range
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
        # Cache Results (INSERT OR IGNORE — new articles only)
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
        
        # After incremental fetch, load full 6 months from DB for state
        if latest_news_date is not None and (to_date - from_date).days < (NEWS_LOOKBACK_DAYS - 10):
            stock_news = get_news_for_ticker(ticker, 'stock', days=NEWS_LOOKBACK_DAYS)
            market_news = get_news_for_ticker(ticker, 'market', days=NEWS_LOOKBACK_DAYS)
            logger.info(f"Node 2: Loaded full 6-month series from DB — stock: {len(stock_news)}, market: {len(market_news)}")

        # ====================================================================
        # Persist velocity baseline (ticker_stats) — live API fetch only.
        # Cache and DB-load paths skip this; their stats are already fresh.
        # ====================================================================
        try:
            daily_avg = compute_and_store_ticker_stats(ticker)
            if daily_avg is not None:
                logger.info(f"Node 2: Velocity baseline stored — {daily_avg:.2f} articles/day for {ticker}")
        except Exception as _stats_err:
            logger.warning(f"Node 2: Could not update ticker_stats for {ticker}: {_stats_err}")

        # ====================================================================
        # Update State
        # ====================================================================
        state['stock_news'] = stock_news
        state['market_news'] = market_news
        state['related_company_news'] = []  # Alpha Vantage doesn't separate related news

        # Add metadata about this fetch for Node 9B velocity detection
        fetch_window_days = max(1, (to_date - from_date).days + 1)
        state['news_fetch_metadata'] = {
            'from_date': from_date.isoformat(),
            'to_date': to_date.isoformat(),
            'fetch_window_days': fetch_window_days,
            'newly_fetched_count': newly_fetched_total,   # Articles fetched from APIs this run
            'total_in_state': len(stock_news) + len(market_news),  # Full 6-month batch
            'newly_fetched_stock': newly_fetched_stock_count,
            'newly_fetched_market': newly_fetched_market_count,
            'was_incremental_fetch': latest_news_date is not None,
            'daily_article_avg': daily_avg,  # True per-calendar-day baseline (from ticker_stats)
        }

        total_articles = len(stock_news) + len(market_news)
        elapsed = (datetime.now() - start_time).total_seconds()
        state['node_execution_times']['node_2'] = elapsed

        logger.info(f"Node 2: Successfully fetched {total_articles} total articles in {elapsed:.2f}s (6-month window)")
        logger.info(f"Node 2:   Stock news: {len(stock_news)} (Alpha Vantage + Finnhub company, 6-month range)")
        logger.info(f"Node 2:   Market news: {len(market_news)} (Finnhub general, filtered by date)")
        logger.info(f"Node 2:   Newly fetched: {newly_fetched_total} articles over {fetch_window_days} days")
        
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
