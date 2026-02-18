""" 
Node 2: Multi-Source News Fetching
Fetches news from Alpha Vantage in parallel using async operations.

Data Sources (6-month historical window):
1. Alpha Vantage (Primary - stock, related-company, and market/global news + built-in sentiment)
   - Stock news: target ticker
   - Related-company news: peers from Node 3 (Finnhub)
   - Market/global news: financial_markets topic

Runs AFTER: Node 1 (price data), Node 3 (related companies)
Runs BEFORE: Node 5 (sentiment), Node 6 (market context)
Can run in PARALLEL with: Node 4 (technical indicators)
"""

import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from src.utils.config import ALPHA_VANTAGE_API_KEY
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
# HELPER FUNCTIONS: Alpha Vantage (Primary for ALL News)
# ============================================================================

# Size of each time window for Alpha Vantage chunked fetching.
# 6 months / 30 days = 6 calls per stream (stock, market, related each make 6 calls).
_AV_WINDOW_DAYS = 30


def _parse_av_timestamp(time_str: str) -> int:
    """Parse Alpha Vantage time_published string to Unix timestamp."""
    if time_str:
        try:
            return int(datetime.strptime(time_str, "%Y%m%dT%H%M%S").timestamp())
        except Exception:
            pass
    return int(datetime.now().timestamp())


async def _av_window_fetch(
    session: aiohttp.ClientSession,
    base_params: Dict[str, Any],
    from_date: Optional[datetime],
    to_date: Optional[datetime],
    label: str,
    window_days: int = _AV_WINDOW_DAYS,
) -> List[Dict[str, Any]]:
    """
    Fetch Alpha Vantage NEWS_SENTIMENT by slicing the date range into fixed
    monthly windows and making one API call per window (sort=RELEVANCE).

    Strategy
    --------
    - Divide [from_date, to_date] into N windows of `window_days` each.
    - For a 6-month window with window_days=30 this is exactly 6 calls.
    - Each call uses sort=RELEVANCE so the 1 000-article limit returns the
      most important articles from that period rather than an arbitrary cut-off.
    - Duplicates across windows are removed by URL.

    Returns the raw feed items (not yet parsed into our article format).
    """
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "your_alpha_vantage_key_here":
        logger.warning(f"Alpha Vantage API key not configured ({label})")
        return []

    effective_to = to_date if to_date is not None else datetime.now()
    effective_from = from_date if from_date is not None else (effective_to - timedelta(days=180))

    # Build windows from newest to oldest so logs read naturally
    windows: List[tuple] = []
    win_to = effective_to
    while win_to > effective_from:
        win_from = max(win_to - timedelta(days=window_days), effective_from)
        windows.append((win_from, win_to))
        win_to = win_from

    url = "https://www.alphavantage.co/query"
    seen_urls: set = set()
    all_items: List[Dict[str, Any]] = []

    for idx, (win_from, win_to) in enumerate(windows, start=1):
        params = {
            **base_params,
            "function": "NEWS_SENTIMENT",
            "apikey": ALPHA_VANTAGE_API_KEY,
            "limit": 1000,
            "sort": "RELEVANCE",
            "time_from": win_from.strftime("%Y%m%dT%H%M"),
            "time_to": win_to.strftime("%Y%m%dT%H%M"),
        }

        logger.info(
            f"AV ({label}) window {idx}/{len(windows)}: "
            f"{win_from.strftime('%Y-%m-%d')} → {win_to.strftime('%Y-%m-%d')}"
        )

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"AV API error ({label}) window {idx}: {response.status}")
                    continue
                data = await response.json()
        except Exception as exc:
            logger.error(f"AV request failed ({label}) window {idx}: {exc}")
            continue

        if "Note" in data or "Error Message" in data:
            logger.warning(
                f"AV ({label}) window {idx}: "
                f"{data.get('Note') or data.get('Error Message')}"
            )
            continue

        feed = data.get("feed", [])
        if not feed:
            logger.info(f"AV ({label}) window {idx}: empty feed")
            continue

        new_this_window = 0
        for item in feed:
            item_url = item.get("url", "")
            if item_url and item_url in seen_urls:
                continue
            if item_url:
                seen_urls.add(item_url)
            item["_ts"] = _parse_av_timestamp(item.get("time_published", ""))
            all_items.append(item)
            new_this_window += 1

        logger.info(
            f"AV ({label}) window {idx}: "
            f"{len(feed)} raw → {new_this_window} new, running total={len(all_items)}"
        )

    logger.info(
        f"AV ({label}): fetch complete — {len(all_items)} articles "
        f"across {len(windows)} windows"
    )
    return all_items


async def fetch_alpha_vantage_news_async(
    ticker: str,
    session: aiohttp.ClientSession,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch ticker-specific stock news + sentiment from Alpha Vantage (paginated).

    Walks backwards through the full 6-month window by repeating the 1 000-article
    call with a sliding time_to until from_date is reached.
    """
    raw_items = await _av_window_fetch(
        session=session,
        base_params={"tickers": ticker},
        from_date=from_date,
        to_date=to_date,
        label=f"stock:{ticker}",
    )

    articles: List[Dict[str, Any]] = []
    for item in raw_items:
        ticker_sentiment = None
        ticker_relevance = 0.0
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker:
                ticker_sentiment = ts.get("ticker_sentiment_score", 0.0)
                ticker_relevance = ts.get("relevance_score", 0.0)
                break

        articles.append({
            "headline": item.get("title", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
            "datetime": item["_ts"],
            "image": item.get("banner_image", ""),
            "news_type": "stock",
            "overall_sentiment_score": item.get("overall_sentiment_score", 0.0),
            "overall_sentiment_label": item.get("overall_sentiment_label", "Neutral"),
            "ticker_sentiment_score": ticker_sentiment,
            "ticker_relevance_score": ticker_relevance,
        })

    logger.info(f"Alpha Vantage: {len(articles)} stock news articles for {ticker}")
    return articles


async def fetch_alpha_vantage_market_news_async(
    session: aiohttp.ClientSession,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    topics: str = "financial_markets",
) -> List[Dict[str, Any]]:
    """
    Fetch broad market / global news from Alpha Vantage (paginated).

    Uses the topics filter (e.g. 'financial_markets') to capture macro stories.
    Paginates through the full 6-month window.
    """
    raw_items = await _av_window_fetch(
        session=session,
        base_params={"topics": topics},
        from_date=from_date,
        to_date=to_date,
        label=f"market:{topics}",
    )

    articles: List[Dict[str, Any]] = []
    for item in raw_items:
        articles.append({
            "headline": item.get("title", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
            "datetime": item["_ts"],
            "image": item.get("banner_image", ""),
            "news_type": "market",
            "overall_sentiment_score": item.get("overall_sentiment_score", 0.0),
            "overall_sentiment_label": item.get("overall_sentiment_label", "Neutral"),
        })

    logger.info(f"Alpha Vantage: {len(articles)} market/global news articles")
    return articles


async def fetch_alpha_vantage_related_company_news_async(
    peers: List[str],
    session: aiohttp.ClientSession,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    per_peer_limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Fetch top N most relevant articles for each peer concurrently.

    One API call per peer (sort=RELEVANCE, limit=per_peer_limit) over the full
    6-month window.  Passing all tickers comma-separated to AV returns 0 results;
    individual calls per ticker work reliably.

    For 5 peers this is exactly 5 concurrent API calls.
    """
    if not peers:
        return []

    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "your_alpha_vantage_key_here":
        logger.warning("Alpha Vantage API key not configured for related company news")
        return []

    url = "https://www.alphavantage.co/query"

    async def _fetch_one(peer: str) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "function": "NEWS_SENTIMENT",
            "tickers": peer.upper(),
            "apikey": ALPHA_VANTAGE_API_KEY,
            "limit": per_peer_limit,
            "sort": "RELEVANCE",
        }
        if from_date is not None:
            params["time_from"] = from_date.strftime("%Y%m%dT%H%M")
        if to_date is not None:
            params["time_to"] = to_date.strftime("%Y%m%dT%H%M")

        logger.info(f"AV (related:{peer}): fetching top {per_peer_limit} articles")
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"AV API error (related:{peer}): {response.status}")
                    return []
                data = await response.json()
        except Exception as exc:
            logger.error(f"AV request failed (related:{peer}): {exc}")
            return []

        if "Note" in data or "Error Message" in data:
            logger.warning(f"AV (related:{peer}): {data.get('Note') or data.get('Error Message')}")
            return []

        feed = data.get("feed", [])
        peer_articles: List[Dict[str, Any]] = []
        for item in feed:
            ts = _parse_av_timestamp(item.get("time_published", ""))
            peer_articles.append({
                "headline": item.get("title", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "datetime": ts,
                "image": item.get("banner_image", ""),
                "news_type": "related",
                "related_ticker": peer.upper(),
                "overall_sentiment_score": item.get("overall_sentiment_score", 0.0),
                "overall_sentiment_label": item.get("overall_sentiment_label", "Neutral"),
            })

        logger.info(f"AV (related:{peer}): got {len(peer_articles)} articles")
        return peer_articles

    # Run all peer fetches concurrently
    results = await asyncio.gather(*[_fetch_one(p) for p in peers], return_exceptions=True)

    seen_urls: set = set()
    all_articles: List[Dict[str, Any]] = []
    for result in results:
        if not isinstance(result, list):
            continue
        for article in result:
            article_url = article.get("url", "")
            if article_url and article_url in seen_urls:
                continue
            if article_url:
                seen_urls.add(article_url)
            all_articles.append(article)

    logger.info(
        f"Alpha Vantage: {len(all_articles)} related-company news articles "
        f"({per_peer_limit}/peer × {len(peers)} peers)"
    )
    return all_articles


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def fetch_all_news_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2: Multi-Source News Fetching (Alpha Vantage primary)
    
    Execution flow:
    1. Check cache for recent news (< 6 hours)
    2. If cached, return cached news
    3. Fetch stock news from Alpha Vantage (ticker-specific, with sentiment)
    4. Fetch market/global news from Alpha Vantage (topics-based)
    5. Fetch related-company news from Alpha Vantage using peers from Node 3
    6. Cache results for future use
    7. Update state with all news articles
    
    Args:
        state: LangGraph state containing 'ticker' and optionally 'raw_price_data'
        
    Returns:
        Updated state with 'stock_news', 'market_news', and 'related_company_news' populated
    """
    start_time = datetime.now()
    ticker = state['ticker']
    related_companies = state.get('related_companies', []) or []
    
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
                    logger.info(
                        f"Node 2: News current for {ticker}, loaded from DB (no API) — "
                        f"stock: {len(stock_from_db)}, market: {len(market_from_db)}"
                    )
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
        # STEP 4: Fetch News from Alpha Vantage (Async Parallel)
        # ====================================================================
        
        async def fetch_all():
            """Async wrapper to fetch all news in parallel with date range"""
            async with aiohttp.ClientSession() as session:
                tasks = [
                    # Alpha Vantage: stock news + sentiment (with time_from, time_to)
                    fetch_alpha_vantage_news_async(ticker, session, from_date, to_date),
                    # Alpha Vantage: market/global news via topics
                    fetch_alpha_vantage_market_news_async(session, from_date, to_date),
                    # Alpha Vantage: related-company news for peers from Node 3
                    fetch_alpha_vantage_related_company_news_async(
                        related_companies, session, from_date, to_date
                    ),
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
        
        # Run async tasks
        results = asyncio.run(fetch_all())
        
        stock_news = results[0] if isinstance(results[0], list) else []
        market_news = results[1] if isinstance(results[1], list) else []
        related_company_news = results[2] if isinstance(results[2], list) else []
        
        # Track how many articles were newly fetched from APIs (before date filtering)
        newly_fetched_stock_count = len(stock_news)
        newly_fetched_market_count = len(market_news)
        newly_fetched_related_count = len(related_company_news)
        newly_fetched_total = (
            newly_fetched_stock_count + newly_fetched_market_count + newly_fetched_related_count
        )
        logger.info(
            f"Node 2: Newly fetched from Alpha Vantage: {newly_fetched_total} articles "
            f"({newly_fetched_stock_count} stock + {newly_fetched_market_count} market "
            f"+ {newly_fetched_related_count} related)"
        )
        
        # ====================================================================
        # Filter by Date Range
        # ====================================================================
        from_timestamp = int(from_date.timestamp())
        to_timestamp = int(to_date.timestamp())
        
        stock_news = [
            article
            for article in stock_news
            if from_timestamp <= article.get("datetime", 0) <= to_timestamp
        ]
        
        market_news = [
            article
            for article in market_news
            if from_timestamp <= article.get("datetime", 0) <= to_timestamp
        ]
        
        related_company_news = [
            article
            for article in related_company_news
            if from_timestamp <= article.get("datetime", 0) <= to_timestamp
        ]
        
        # ====================================================================
        # Cache Results (INSERT OR IGNORE — new articles only)
        # ====================================================================
        try:
            if stock_news:
                cache_news(ticker, "stock", stock_news)
                logger.info(f"Node 2: Cached {len(stock_news)} stock news articles")
            if market_news:
                cache_news(ticker, "market", market_news)
                logger.info(f"Node 2: Cached {len(market_news)} market news articles")
            if related_company_news:
                cache_news(ticker, "related", related_company_news)
                logger.info(
                    f"Node 2: Cached {len(related_company_news)} related-company news articles"
                )
        except Exception as e:
            logger.warning(f"Node 2: Failed to cache news: {str(e)}")
        
        # After incremental fetch, load full 6 months from DB for state
        if latest_news_date is not None and (to_date - from_date).days < (NEWS_LOOKBACK_DAYS - 10):
            stock_news = get_news_for_ticker(ticker, "stock", days=NEWS_LOOKBACK_DAYS)
            market_news = get_news_for_ticker(ticker, "market", days=NEWS_LOOKBACK_DAYS)
            related_company_news = get_news_for_ticker(
                ticker, "related", days=NEWS_LOOKBACK_DAYS
            )
            logger.info(
                "Node 2: Loaded full 6-month series from DB — "
                f"stock: {len(stock_news)}, market: {len(market_news)}, "
                f"related: {len(related_company_news)}"
            )

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
        state["stock_news"] = stock_news
        state["market_news"] = market_news
        state["related_company_news"] = related_company_news

        # Add metadata about this fetch for Node 9B velocity detection
        fetch_window_days = max(1, (to_date - from_date).days + 1)
        state["news_fetch_metadata"] = {
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "fetch_window_days": fetch_window_days,
            "newly_fetched_count": newly_fetched_total,
            "total_in_state": len(stock_news) + len(market_news) + len(related_company_news),
            "newly_fetched_stock": newly_fetched_stock_count,
            "newly_fetched_market": newly_fetched_market_count,
            "newly_fetched_related": newly_fetched_related_count,
            "was_incremental_fetch": latest_news_date is not None,
            "daily_article_avg": daily_avg,
        }

        total_articles = len(stock_news) + len(market_news) + len(related_company_news)
        elapsed = (datetime.now() - start_time).total_seconds()
        state["node_execution_times"]["node_2"] = elapsed
        
        logger.info(
            f"Node 2: Successfully fetched {total_articles} total articles in {elapsed:.2f}s "
            f"(6-month window)"
        )
        logger.info(
            f"Node 2:   Stock news: {len(stock_news)} (Alpha Vantage ticker-specific, 6-month range)"
        )
        logger.info(
            f"Node 2:   Market/global news: {len(market_news)} (Alpha Vantage topics-based)"
        )
        logger.info(
            f"Node 2:   Related-company news: {len(related_company_news)} "
            f"(Alpha Vantage peers from Node 3)"
        )
        logger.info(
            f"Node 2:   Newly fetched: {newly_fetched_total} articles over {fetch_window_days} days"
        )
        
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
