"""
Node 1: Price Data Fetching
Fetches historical OHLCV (Open, High, Low, Close, Volume) price data.

Data Sources (in priority order):
1. Cache (24 hours)
2. yfinance (Primary - NO API key, fast, reliable)
3. Polygon.io (Backup - requires API key)

Runs AFTER: Nothing (first node)
Runs BEFORE: All other nodes
Can run in PARALLEL with: Nothing
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from polygon import RESTClient
import yfinance as yf

from src.utils.config import POLYGON_API_KEY, DATABASE_PATH, CACHE_HOURS
from src.database.db_manager import get_cached_price_data, cache_price_data

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTION: Fetch from yfinance (Primary)
# ============================================================================

def fetch_from_yfinance(ticker: str, days: int = 180) -> Optional[pd.DataFrame]:
    """
    Fetch price data from yfinance (Primary source - NO API key needed).
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        days: Number of days of historical data (default: 180 for 6 months)
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
        None if fetch fails
        
    Example:
        >>> df = fetch_from_yfinance('AAPL', days=180)
        >>> print(df.columns)
        Index(['date', 'open', 'high', 'low', 'close', 'volume'])
    """
    try:
        logger.info(f"Fetching {days} days of price data from yfinance for {ticker}")
        
        # Calculate period (yfinance uses period parameter)
        # Map days to yfinance period
        if days <= 7:
            period = '1mo'
        elif days <= 90:
            period = '3mo'
        elif days <= 180:
            period = '6mo'
        elif days <= 365:
            period = '1y'
        else:
            period = '2y'
        
        # Fetch data using Ticker object for better reliability
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period)
        
        if df.empty:
            logger.warning(f"yfinance: No data returned for {ticker}")
            return None
        
        # Standardize column names to match our schema
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Keep only required columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"yfinance: Fetched {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {str(e)}")
        return None


# ============================================================================
# HELPER FUNCTION: Fetch from Polygon (Backup)
# ============================================================================

    """
    Fetch price data from Polygon.io API (Backup source).
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        days: Number of days of historical data (default: 180 for 6 months)
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
        None if fetch fails
        
    Example:
        >>> df = fetch_from_polygon('AAPL', days=180)
        >>> print(df.columns)
        Index(['date', 'open', 'high', 'low', 'close', 'volume'])
    """
    try:
        if not POLYGON_API_KEY or POLYGON_API_KEY == 'your_polygon_api_key_here':
            logger.warning("Polygon API key not configured, skipping")
            return None
            
        logger.info(f"Fetching {days} days of price data from Polygon for {ticker}")
        
        # Initialize Polygon client
        client = RESTClient(POLYGON_API_KEY)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for Polygon API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Fetch aggregates (daily bars)
        aggs = []
        for agg in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=from_date,
            to=to_date,
            limit=50000
        ):
            aggs.append(agg)
        
        if not aggs:
            logger.warning(f"No data returned from Polygon for {ticker}")
            return None
        
        # Convert to DataFrame
        data = []
        for agg in aggs:
            data.append({
                'date': datetime.fromtimestamp(agg.timestamp / 1000),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Polygon: Fetched {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Polygon fetch failed for {ticker}: {str(e)}")
        return None




# ============================================================================
# HELPER FUNCTION: Validate Data Quality
# ============================================================================

def validate_price_data(df: pd.DataFrame, ticker: str, min_rows: int = 50) -> bool:
    """
    Validate that price data meets quality requirements.
    
    Args:
        df: Price data DataFrame
        ticker: Stock ticker for logging
        min_rows: Minimum required rows
        
    Returns:
        True if data is valid, False otherwise
    """
    if df is None or df.empty:
        logger.warning(f"Validation failed for {ticker}: Empty dataframe")
        return False
    
    if len(df) < min_rows:
        logger.warning(f"Validation failed for {ticker}: Only {len(df)} rows (need {min_rows})")
        return False
    
    # Check for required columns
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        logger.warning(f"Validation failed for {ticker}: Missing required columns")
        return False
    
    # Check for null values
    null_counts = df[required_cols].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Validation warning for {ticker}: {null_counts.sum()} null values found")
    
    logger.info(f"Data validation passed for {ticker}: {len(df)} rows")
    return True


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def fetch_price_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 1: Price Data Fetching
    
    Execution flow:
    1. Check SQLite cache for recent data (< 24h old)
    2. If cached, return cached data
    3. If not cached, try yfinance (Primary - NO API key, fast, reliable)
    4. If yfinance fails, fall back to Polygon.io (requires API key)
    5. Validate data quality (min 50 days)
    6. Cache results for future use
    7. Update state with price data
    
    Args:
        state: LangGraph state containing 'ticker'
        
    Returns:
        Updated state with 'raw_price_data' populated
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    try:
        logger.info(f"Node 1: Starting price data fetch for {ticker}")
        
        # ====================================================================
        # STEP 1: Check Cache (80% cache hit rate)
        # ====================================================================
        cached_df = get_cached_price_data(ticker, max_age_hours=CACHE_HOURS)
        
        if cached_df is not None and validate_price_data(cached_df, ticker):
            logger.info(f"Node 1: Using cached data for {ticker} ({len(cached_df)} rows)")
            state['raw_price_data'] = cached_df
            elapsed = (datetime.now() - start_time).total_seconds()
            state['node_execution_times']['node_1'] = elapsed
            logger.info(f"Node 1: Completed (cache hit) in {elapsed:.2f}s")
            return state
        
        # ====================================================================
        # STEP 2: Fetch from yfinance (Primary Source - NO API key needed)
        # ====================================================================
        df = fetch_from_yfinance(ticker, days=180)
        
        # ====================================================================
        # STEP 3: Fallback to Polygon if yfinance fails
        # ====================================================================
        if df is None or not validate_price_data(df, ticker):
            logger.warning(f"Node 1: yfinance failed for {ticker}, falling back to Polygon.io")
            df = fetch_from_polygon(ticker, days=180)
        
        # ====================================================================
        # STEP 4: Validate Final Result
        # ====================================================================
        if df is None or not validate_price_data(df, ticker):
            logger.error(f"Node 1: All price sources failed for {ticker}")
            state['errors'].append(f"Node 1: Failed to fetch price data for {ticker}")
            state['raw_price_data'] = None
            elapsed = (datetime.now() - start_time).total_seconds()
            state['node_execution_times']['node_1'] = elapsed
            return state
        
        # ====================================================================
        # STEP 5: Cache for Future Use
        # ====================================================================
        try:
            cache_price_data(ticker, df)
            logger.info(f"Node 1: Cached {len(df)} rows for {ticker}")
        except Exception as e:
            logger.warning(f"Node 1: Failed to cache data for {ticker}: {str(e)}")
            # Continue anyway - caching failure is not critical
        
        # ====================================================================
        # STEP 6: Update State
        # ====================================================================
        state['raw_price_data'] = df
        elapsed = (datetime.now() - start_time).total_seconds()
        state['node_execution_times']['node_1'] = elapsed
        
        logger.info(f"Node 1: Successfully fetched {len(df)} rows for {ticker} in {elapsed:.2f}s")
        return state
        
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Node 1: Unexpected error for {ticker}: {str(e)}")
        state['errors'].append(f"Node 1: Unexpected error - {str(e)}")
        state['raw_price_data'] = None
        elapsed = (datetime.now() - start_time).total_seconds()
        state['node_execution_times']['node_1'] = elapsed
        return state
