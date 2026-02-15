"""
Database Manager - SQLite Operations
Handles all database CRUD operations with proper error handling and caching.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = "data/stock_prices.db"


# ============================================================================
# CONNECTION MANAGEMENT
# ============================================================================

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Get database connection with context manager support.
    
    Args:
        db_path: Path to SQLite database file
    
    Returns:
        SQLite connection object
    
    Example:
        >>> with get_connection() as conn:
        >>>     cursor = conn.cursor()
        >>>     cursor.execute("SELECT * FROM price_data LIMIT 1")
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column name access
    return conn


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_database(schema_path: str = "src/database/schema.sql", db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initialize database with schema from SQL file.
    
    Creates all tables, indexes, and views as specified in schema.
    Safe to run multiple times (uses IF NOT EXISTS).
    
    Args:
        schema_path: Path to schema.sql file
        db_path: Path to SQLite database file
    
    Raises:
        FileNotFoundError: If schema file doesn't exist
        sqlite3.Error: If database creation fails
    
    Example:
        >>> init_database()
        >>> # Database created at data/stock_prices.db
    """
    schema_file = Path(schema_path)
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    # Read schema SQL
    with open(schema_file, 'r') as f:
        schema_sql = f.read()
    
    # Execute schema
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.executescript(schema_sql)
            conn.commit()
            logger.info(f"Database initialized successfully at {db_path}")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise


# ============================================================================
# PRICE DATA OPERATIONS
# ============================================================================

def cache_price_data(ticker: str, df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Cache price data to database.
    
    Uses INSERT OR REPLACE to handle duplicates.
    
    Args:
        ticker: Stock ticker symbol
        df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
        db_path: Path to database
    
    Example:
        >>> df = pd.DataFrame({'date': [...], 'close': [...]})
        >>> cache_price_data('AAPL', df)
    """
    if df is None or df.empty:
        logger.warning(f"Attempted to cache empty price data for {ticker}")
        return
    
    try:
        with get_connection(db_path) as conn:
            # Prepare data for insertion
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO price_data 
                    (ticker, date, open, high, low, close, volume, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    ticker,
                    row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else str(row['date']),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume'])
                ))
            conn.commit()
            logger.info(f"Cached {len(df)} price records for {ticker}")
    except Exception as e:
        logger.error(f"Failed to cache price data for {ticker}: {str(e)}")


def get_cached_price_data(
    ticker: str, 
    max_age_hours: int = 24, 
    db_path: str = DEFAULT_DB_PATH
) -> Optional[pd.DataFrame]:
    """
    Retrieve cached price data from database.
    
    Returns data only if it was cached within max_age_hours.
    
    Args:
        ticker: Stock ticker symbol
        max_age_hours: Maximum age of cached data in hours (default: 24)
        db_path: Path to database
    
    Returns:
        DataFrame with price data or None if not cached or too old
    
    Example:
        >>> df = get_cached_price_data('AAPL', max_age_hours=24)
        >>> if df is not None:
        >>>     print(f"Got {len(df)} cached records")
    """
    try:
        with get_connection(db_path) as conn:
            query = """
                SELECT date, open, high, low, close, volume
                FROM price_data
                WHERE ticker = ?
                AND created_at > datetime('now', ?)
                ORDER BY date DESC
            """
            params = (ticker, f'-{max_age_hours} hours')
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                logger.debug(f"No cached price data for {ticker}")
                return None
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"Retrieved {len(df)} cached price records for {ticker}")
            return df
            
    except Exception as e:
        logger.error(f"Failed to retrieve cached price data for {ticker}: {str(e)}")
        return None


def get_latest_price_date(
    ticker: str,
    db_path: str = DEFAULT_DB_PATH
) -> Optional[datetime]:
    """
    Get the most recent date of stored price data for a ticker.
    
    Used for incremental fetch: only fetch from (latest_date - overlap) to today.
    
    Args:
        ticker: Stock ticker symbol
        db_path: Path to database
    
    Returns:
        datetime of latest stored date, or None if no data for ticker
    
    Example:
        >>> latest = get_latest_price_date('NVDA')
        >>> if latest: fetch_from_yfinance('NVDA', from_date=latest - timedelta(days=3))
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(date) AS latest_date
                FROM price_data
                WHERE ticker = ?
            """, (ticker,))
            row = cursor.fetchone()
            if not row or not row['latest_date']:
                return None
            return datetime.strptime(str(row['latest_date'])[:10], '%Y-%m-%d')
    except Exception as e:
        logger.error(f"Failed to get latest price date for {ticker}: {str(e)}")
        return None


def get_price_data_for_ticker(
    ticker: str,
    days: int = 180,
    db_path: str = DEFAULT_DB_PATH
) -> Optional[pd.DataFrame]:
    """
    Get all stored price data for a ticker in the last N days (by date, not created_at).
    
    Used after incremental fetch to load the full series for the pipeline.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of calendar days of history (default: 180)
        db_path: Path to database
    
    Returns:
        DataFrame with date, open, high, low, close, volume or None if empty
    """
    try:
        with get_connection(db_path) as conn:
            query = """
                SELECT date, open, high, low, close, volume
                FROM price_data
                WHERE ticker = ?
                AND date >= date('now', ?)
                ORDER BY date ASC
            """
            params = (ticker, f'-{days} days')
            df = pd.read_sql_query(query, conn, params=params)
            if df.empty:
                return None
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"Retrieved {len(df)} price records for {ticker} (last {days} days)")
            return df
    except Exception as e:
        logger.error(f"Failed to get price data for {ticker}: {str(e)}")
        return None


# ============================================================================
# NEWS OPERATIONS
# ============================================================================

def cache_news(
    ticker: str, 
    news_type: str, 
    articles: List[Dict], 
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Cache news articles to database.
    
    Saves Alpha Vantage sentiment data if available (overall_sentiment_label, overall_sentiment_score).
    This enables the background script to evaluate news outcomes without re-analyzing sentiment.
    
    Args:
        ticker: Stock ticker symbol
        news_type: 'stock', 'market', or 'related'
        articles: List of article dictionaries with keys:
                  ['title', 'description', 'url', 'source', 'publishedAt']
                  Optional Alpha Vantage fields:
                  ['overall_sentiment_label', 'overall_sentiment_score']
        db_path: Path to database
    
    Example:
        >>> # Alpha Vantage article with sentiment
        >>> articles = [{
        ...     'title': 'News',
        ...     'url': 'http://...',
        ...     'overall_sentiment_label': 'Bullish',
        ...     'overall_sentiment_score': 0.75
        ... }]
        >>> cache_news('AAPL', 'stock', articles)
    """
    if not articles:
        logger.debug(f"No articles to cache for {ticker} ({news_type})")
        return
    
    try:
        with get_connection(db_path) as conn:
            for article in articles:
                # Extract source name (handle both dict and string)
                source_name = article.get('source', 'Unknown')
                if isinstance(source_name, dict):
                    source_name = source_name.get('name', 'Unknown')
                
                # Handle field name variations from different APIs
                # Node 2 (Alpha Vantage/Finnhub) uses: headline, summary, datetime
                # Some other sources might use: title, description, publishedAt
                title = article.get('headline') or article.get('title', '')
                description = article.get('summary') or article.get('description', '')
                
                # Convert datetime (Unix timestamp) to ISO string for database
                published_at = article.get('publishedAt', '')
                if not published_at and 'datetime' in article:
                    dt_val = article['datetime']
                    if isinstance(dt_val, (int, float)):
                        # Unix timestamp to ISO string
                        published_at = datetime.fromtimestamp(dt_val).isoformat()
                    else:
                        published_at = str(dt_val)
                
                # Extract Alpha Vantage sentiment if available
                # Alpha Vantage provides: 'Bearish', 'Somewhat-Bearish', 'Neutral', 
                # 'Somewhat-Bullish', 'Bullish'
                sentiment_label = article.get('overall_sentiment_label')
                sentiment_score = article.get('overall_sentiment_score')
                
                # Normalize Alpha Vantage labels to our standard format
                # 'Bearish', 'Somewhat-Bearish' → 'negative'
                # 'Neutral' → 'neutral'
                # 'Bullish', 'Somewhat-Bullish' → 'positive'
                if sentiment_label:
                    sentiment_label_lower = sentiment_label.lower()
                    if 'bearish' in sentiment_label_lower:
                        normalized_label = 'negative'
                    elif 'bullish' in sentiment_label_lower:
                        normalized_label = 'positive'
                    else:
                        normalized_label = 'neutral'
                else:
                    normalized_label = None
                
                conn.execute("""
                    INSERT OR IGNORE INTO news_articles
                    (ticker, news_type, title, description, url, source, published_at, 
                     sentiment_label, sentiment_score, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    ticker,
                    news_type,
                    title,
                    description,
                    article.get('url', ''),
                    source_name,
                    published_at,
                    normalized_label,  # Alpha Vantage sentiment normalized
                    sentiment_score if sentiment_score is not None else None
                ))
            conn.commit()
            
            # Log how many had sentiment data
            with_sentiment = sum(1 for a in articles if a.get('overall_sentiment_label'))
            logger.info(f"Cached {len(articles)} {news_type} news articles for {ticker} "
                       f"({with_sentiment} with Alpha Vantage sentiment)")
    except Exception as e:
        logger.error(f"Failed to cache news for {ticker}: {str(e)}")


def get_cached_news(
    ticker: str, 
    news_type: str, 
    max_age_hours: int = 24, 
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict]:
    """
    Retrieve cached news articles from database.
    
    Args:
        ticker: Stock ticker symbol
        news_type: 'stock', 'market', or 'related'
        max_age_hours: Maximum age in hours
        db_path: Path to database
    
    Returns:
        List of article dictionaries
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT title, description, url, source, published_at
                FROM news_articles
                WHERE ticker = ?
                AND news_type = ?
                AND fetched_at > datetime('now', ?)
                ORDER BY published_at DESC
            """, (ticker, news_type, f'-{max_age_hours} hours'))
            
            articles = []
            for row in cursor.fetchall():
                articles.append({
                    'title': row['title'],
                    'description': row['description'],
                    'url': row['url'],
                    'source': {'name': row['source']},
                    'publishedAt': row['published_at']
                })
            
            logger.info(f"Retrieved {len(articles)} cached {news_type} articles for {ticker}")
            return articles
            
    except Exception as e:
        logger.error(f"Failed to retrieve cached news: {str(e)}")
        return []


def get_latest_news_date(
    ticker: str,
    db_path: str = DEFAULT_DB_PATH
) -> Optional[datetime]:
    """
    Get the most recent published_at date of stored news for a ticker (any type).
    
    Used for incremental fetch: only fetch from (latest_date - overlap) to today.
    
    Args:
        ticker: Stock ticker symbol
        db_path: Path to database
    
    Returns:
        datetime of latest article published_at, or None if no news for ticker
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(published_at) AS latest_date
                FROM news_articles
                WHERE ticker = ?
                AND published_at IS NOT NULL
                AND published_at != ''
            """, (ticker,))
            row = cursor.fetchone()
            if not row or not row['latest_date']:
                return None
            raw = str(row['latest_date'])
            return datetime.strptime(raw[:10], '%Y-%m-%d')
    except Exception as e:
        logger.error(f"Failed to get latest news date for {ticker}: {str(e)}")
        return None


def get_news_for_ticker(
    ticker: str,
    news_type: str,
    days: int = 180,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict]:
    """
    Get all stored news for a ticker in the last N days (by published_at).
    
    Returns articles in pipeline format (headline, summary, datetime, etc.)
    so downstream nodes work the same as with API-fetched news.
    
    Args:
        ticker: Stock ticker symbol
        news_type: 'stock', 'market', or 'related'
        days: Number of calendar days of history (default: 180)
        db_path: Path to database
    
    Returns:
        List of article dicts with headline, summary, url, source, datetime, etc.
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT title, description, url, source, published_at
                FROM news_articles
                WHERE ticker = ?
                AND news_type = ?
                AND published_at >= date('now', ?)
                ORDER BY published_at DESC
            """, (ticker, news_type, f'-{days} days'))
            articles = []
            for row in cursor.fetchall():
                pub = row['published_at']
                try:
                    if not pub:
                        ts = int(datetime.now().timestamp())
                    else:
                        s = str(pub).replace('Z', '').strip()
                        if 'T' in s:
                            ts = int(datetime.strptime(s[:19], '%Y-%m-%dT%H:%M:%S').timestamp())
                        else:
                            ts = int(datetime.strptime(s[:10], '%Y-%m-%d').timestamp())
                except (ValueError, TypeError):
                    ts = int(datetime.now().timestamp())
                src_name = row['source'] if isinstance(row['source'], str) else getattr(row['source'], 'get', lambda k, d=None: d)('name', 'Unknown')
                articles.append({
                    'headline': row['title'] or '',
                    'title': row['title'] or '',
                    'summary': row['description'] or '',
                    'description': row['description'] or '',
                    'url': row['url'] or '',
                    'source': {'name': src_name},
                    'publishedAt': pub,
                    'datetime': ts
                })
            logger.info(f"Retrieved {len(articles)} {news_type} articles for {ticker} (last {days} days)")
            return articles
    except Exception as e:
        logger.error(f"Failed to get news for {ticker} ({news_type}): {str(e)}")
        return []


# ============================================================================
# NEWS OUTCOMES (Critical for Node 8 Learning)
# ============================================================================

def get_news_with_outcomes(
    ticker: str, 
    days: int = 180, 
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """
    Get historical news articles with their outcomes.
    
    CRITICAL for Node 8 learning system.
    Returns news articles where we know what happened 7 days later.
    
    Args:
        ticker: Stock ticker symbol
        days: How many days of history to retrieve (default: 180 for 6 months)
        db_path: Path to database
    
    Returns:
        DataFrame with columns:
        - id, ticker, news_type, title, source, published_at
        - sentiment_label, sentiment_score
        - price_at_news, price_7day_later, price_change_7day
        - predicted_direction, actual_direction, prediction_was_accurate_7day
    
    Example:
        >>> df = get_news_with_outcomes('NVDA', days=180)
        >>> bloomberg_articles = df[df['source'] == 'Bloomberg.com']
        >>> accuracy = bloomberg_articles['prediction_was_accurate_7day'].mean()
        >>> print(f"Bloomberg accuracy: {accuracy:.1%}")
    """
    try:
        with get_connection(db_path) as conn:
            query = """
                SELECT * FROM news_with_outcomes
                WHERE ticker = ?
                AND published_at >= date('now', ?)
                ORDER BY published_at DESC
            """
            params = (ticker, f'-{days} days')
            df = pd.read_sql_query(query, conn, params=params)
            
            logger.info(f"Retrieved {len(df)} news with outcomes for {ticker}")
            return df
            
    except Exception as e:
        logger.error(f"Failed to get news with outcomes: {str(e)}")
        return pd.DataFrame()


# ============================================================================
# SOURCE RELIABILITY OPERATIONS
# ============================================================================

def store_source_reliability(
    ticker: str,
    source_reliability_data: Dict[str, Dict[str, float]],
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Store calculated source reliability scores.
    
    Used by Node 8 to persist learning results.
    
    Args:
        ticker: Stock ticker symbol
        source_reliability_data: Dict mapping source names to reliability metrics
            Example: {
                'Bloomberg.com': {
                    'total_articles': 20,
                    'accurate_predictions': 17,
                    'accuracy_rate': 0.85,
                    'avg_price_impact': 2.3,
                    'confidence_multiplier': 1.2
                }
            }
        db_path: Path to database
    """
    try:
        with get_connection(db_path) as conn:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
            
            for source_name, metrics in source_reliability_data.items():
                conn.execute("""
                    INSERT OR REPLACE INTO source_reliability
                    (ticker, source_name, analysis_date, total_articles, 
                     accurate_predictions, accuracy_rate, avg_price_impact, 
                     confidence_multiplier, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    ticker,
                    source_name,
                    analysis_date,
                    metrics['total_articles'],
                    metrics['accurate_predictions'],
                    metrics['accuracy_rate'],
                    metrics.get('avg_price_impact', 0.0),
                    metrics['confidence_multiplier']
                ))
            conn.commit()
            logger.info(f"Stored source reliability for {len(source_reliability_data)} sources")
    except Exception as e:
        logger.error(f"Failed to store source reliability: {str(e)}")


def get_source_reliability(
    ticker: str, 
    db_path: str = DEFAULT_DB_PATH
) -> Dict[str, Dict[str, float]]:
    """
    Get latest source reliability scores for a ticker.
    
    Used by Node 8 to adjust sentiment confidence.
    
    Args:
        ticker: Stock ticker symbol
        db_path: Path to database
    
    Returns:
        Dict mapping source names to metrics
    
    Example:
        >>> reliability = get_source_reliability('NVDA')
        >>> bloomberg_multiplier = reliability['Bloomberg.com']['confidence_multiplier']
        >>> # Use to adjust sentiment: confidence * bloomberg_multiplier
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT source_name, total_articles, accurate_predictions,
                       accuracy_rate, avg_price_impact, confidence_multiplier
                FROM source_reliability
                WHERE ticker = ?
                AND analysis_date = (
                    SELECT MAX(analysis_date) 
                    FROM source_reliability 
                    WHERE ticker = ?
                )
            """, (ticker, ticker))
            
            reliability = {}
            for row in cursor.fetchall():
                reliability[row['source_name']] = {
                    'total_articles': row['total_articles'],
                    'accurate_predictions': row['accurate_predictions'],
                    'accuracy_rate': row['accuracy_rate'],
                    'avg_price_impact': row['avg_price_impact'],
                    'confidence_multiplier': row['confidence_multiplier']
                }
            
            return reliability
            
    except Exception as e:
        logger.error(f"Failed to get source reliability: {str(e)}")
        return {}


# ============================================================================
# ANOMALY DETECTION STORAGE
# ============================================================================

def store_anomaly_results(
    ticker: str,
    early_anomaly: Dict[str, Any],
    behavioral_anomaly: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Store anomaly detection results for both phases.
    
    Args:
        ticker: Stock ticker symbol
        early_anomaly: Results from Node 9A
        behavioral_anomaly: Results from Node 9B
        db_path: Path to database
    """
    analysis_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        with get_connection(db_path) as conn:
            # Store early anomaly results
            if early_anomaly:
                conn.execute("""
                    INSERT INTO early_anomaly_results
                    (ticker, analysis_date, keyword_alerts, news_surge_detected,
                     suspicious_sources, coordinated_posting, filtered_news_count,
                     risk_level, warning_flags, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    ticker,
                    analysis_date,
                    json.dumps(early_anomaly.get('keyword_alerts', [])),
                    early_anomaly.get('news_surge_detected', False),
                    json.dumps(early_anomaly.get('suspicious_sources', [])),
                    early_anomaly.get('coordinated_posting', False),
                    early_anomaly.get('filtered_news_count', 0),
                    early_anomaly.get('early_risk_level', 'LOW'),
                    json.dumps(early_anomaly.get('early_warning_flags', []))
                ))
            
            # Store behavioral anomaly results
            if behavioral_anomaly:
                conn.execute("""
                    INSERT INTO behavioral_anomaly_results
                    (ticker, analysis_date, pump_and_dump_score, is_pump_and_dump,
                     price_anomalies, volume_anomalies, volatility_anomaly,
                     news_price_divergence, manipulation_probability, risk_level,
                     warning_flags, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    ticker,
                    analysis_date,
                    behavioral_anomaly.get('pump_and_dump_score', 0.0),
                    behavioral_anomaly.get('is_pump_and_dump', False),
                    json.dumps(behavioral_anomaly.get('price_anomalies', [])),
                    json.dumps(behavioral_anomaly.get('volume_anomalies', [])),
                    behavioral_anomaly.get('volatility_anomaly', False),
                    behavioral_anomaly.get('news_price_divergence', False),
                    behavioral_anomaly.get('manipulation_probability', 0.0),
                    behavioral_anomaly.get('behavioral_risk_level', 'LOW'),
                    json.dumps(behavioral_anomaly.get('behavioral_warning_flags', []))
                ))
            
            conn.commit()
            logger.info(f"Stored anomaly detection results for {ticker}")
    except Exception as e:
        logger.error(f"Failed to store anomaly results: {str(e)}")


# ============================================================================
# BACKTEST & WEIGHTS OPERATIONS
# ============================================================================

def store_backtest_results(
    ticker: str,
    results: Dict[str, Dict[str, float]],
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Store backtest results for all signal streams.
    
    Args:
        ticker: Stock ticker symbol
        results: Dict with keys 'technical', 'stock_news', 'market_news', 'related'
                 Each value is dict with 'accuracy', 'sample_size', 'backtest_period_days'
        db_path: Path to database
    """
    try:
        with get_connection(db_path) as conn:
            for signal_type, metrics in results.items():
                conn.execute("""
                    INSERT INTO backtest_results
                    (ticker, signal_type, accuracy, sample_size, backtest_period_days, created_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    ticker,
                    signal_type,
                    metrics['accuracy'],
                    metrics['sample_size'],
                    metrics['backtest_period_days']
                ))
            conn.commit()
            logger.info(f"Stored backtest results for {ticker}")
    except Exception as e:
        logger.error(f"Failed to store backtest results: {str(e)}")


def store_adaptive_weights(
    ticker: str,
    weights: Dict[str, float],
    explanation: str,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Store adaptive weights calculation.
    
    Args:
        ticker: Stock ticker symbol
        weights: Dict with 'technical_weight', 'stock_news_weight', 
                'market_news_weight', 'related_companies_weight'
        explanation: Human-readable explanation of weight distribution
        db_path: Path to database
    """
    try:
        with get_connection(db_path) as conn:
            date = datetime.now().strftime('%Y-%m-%d')
            
            conn.execute("""
                INSERT INTO weight_history
                (ticker, date, technical_weight, stock_news_weight,
                 market_news_weight, related_companies_weight, weight_explanation, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                ticker,
                date,
                weights['technical_weight'],
                weights['stock_news_weight'],
                weights['market_news_weight'],
                weights['related_companies_weight'],
                explanation
            ))
            conn.commit()
            logger.info(f"Stored adaptive weights for {ticker}")
    except Exception as e:
        logger.error(f"Failed to store adaptive weights: {str(e)}")


# ============================================================================
# FINAL SIGNAL STORAGE
# ============================================================================

def store_final_signal(
    ticker: str,
    signal_data: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Store final signal recommendation.
    
    Args:
        ticker: Stock ticker symbol
        signal_data: Dict with 'recommendation', 'confidence', 'strength',
                    'target_price', 'stop_loss', 'contributing_factors', 'risk_warnings'
        db_path: Path to database
    """
    try:
        with get_connection(db_path) as conn:
            date = datetime.now().strftime('%Y-%m-%d')
            
            conn.execute("""
                INSERT INTO final_signals
                (ticker, date, recommendation, confidence, strength,
                 target_price, stop_loss, contributing_factors, risk_warnings, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                ticker,
                date,
                signal_data['recommendation'],
                signal_data['confidence'],
                signal_data['strength'],
                signal_data.get('target_price'),
                signal_data.get('stop_loss'),
                json.dumps(signal_data.get('contributing_factors', [])),
                json.dumps(signal_data.get('risk_warnings', []))
            ))
            conn.commit()
            logger.info(f"Stored final signal for {ticker}: {signal_data['recommendation']}")
    except Exception as e:
        logger.error(f"Failed to store final signal: {str(e)}")


# ============================================================================
# NEWS OUTCOMES OPERATIONS (For Background Script)
# ============================================================================

def get_news_outcomes_pending(
    ticker: Optional[str] = None,
    limit: int = 500,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict]:
    """
    Get news articles that are 7+ days old and don't have outcomes yet.
    
    Used by background script to find articles needing evaluation.
    
    Args:
        ticker: Optional ticker symbol (if None, gets for ALL tickers)
        limit: Max articles to return (default: 500)
        db_path: Path to database
        
    Returns:
        List of article dictionaries with:
        - id, ticker, published_at, sentiment_label, sentiment_score
        
    Example:
        >>> pending = get_news_outcomes_pending(ticker='NVDA', limit=100)
        >>> print(f"Found {len(pending)} articles needing evaluation")
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            if ticker:
                # Single ticker
                cursor.execute("""
                    SELECT n.id, n.ticker, n.published_at, 
                           n.sentiment_label, n.sentiment_score
                    FROM news_articles n
                    LEFT JOIN news_outcomes no ON n.id = no.news_id
                    WHERE n.ticker = ?
                    AND no.id IS NULL
                    AND n.published_at <= date('now', '-7 days')
                    AND n.published_at IS NOT NULL
                    AND n.published_at != ''
                    ORDER BY n.published_at ASC
                    LIMIT ?
                """, (ticker, limit))
            else:
                # All tickers
                cursor.execute("""
                    SELECT n.id, n.ticker, n.published_at,
                           n.sentiment_label, n.sentiment_score
                    FROM news_articles n
                    LEFT JOIN news_outcomes no ON n.id = no.news_id
                    WHERE no.id IS NULL
                    AND n.published_at <= date('now', '-7 days')
                    AND n.published_at IS NOT NULL
                    AND n.published_at != ''
                    ORDER BY n.published_at ASC
                    LIMIT ?
                """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row['id'],
                    'ticker': row['ticker'],
                    'published_at': row['published_at'],
                    'sentiment_label': row['sentiment_label'],
                    'sentiment_score': row['sentiment_score']
                })
            
            logger.info(f"Found {len(results)} pending news outcomes"
                       f"{f' for {ticker}' if ticker else ''}")
            return results
            
    except Exception as e:
        logger.error(f"Failed to get pending news outcomes: {str(e)}")
        return []


def get_price_on_date(
    ticker: str,
    date_str: str,
    db_path: str = DEFAULT_DB_PATH
) -> Optional[float]:
    """
    Get closing price on a specific date.
    
    If no exact match (weekend/holiday), returns nearest PREVIOUS trading day.
    
    Args:
        ticker: Stock ticker symbol
        date_str: Date in format 'YYYY-MM-DD'
        db_path: Path to database
        
    Returns:
        Closing price (float) or None if no data available
        
    Example:
        >>> price = get_price_on_date('NVDA', '2025-09-15')
        >>> print(f"NVDA closed at ${price:.2f} on 2025-09-15")
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Get price on or before the date (handles weekends/holidays)
            cursor.execute("""
                SELECT close FROM price_data
                WHERE ticker = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            """, (ticker, date_str))
            
            row = cursor.fetchone()
            
            if row:
                return float(row['close'])
            
            logger.debug(f"No price data for {ticker} on or before {date_str}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to get price for {ticker} on {date_str}: {str(e)}")
        return None


def get_price_after_days(
    ticker: str,
    date_str: str,
    days: int,
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[Optional[float], Optional[str]]:
    """
    Get closing price N calendar days after a given date.
    
    Finds the nearest trading day on or after (date + days).
    Handles weekends and holidays automatically.
    
    Args:
        ticker: Stock ticker symbol
        date_str: Starting date in format 'YYYY-MM-DD'
        days: Number of calendar days to add
        db_path: Path to database
        
    Returns:
        Tuple of (closing_price, actual_date) or (None, None) if no data
        
    Example:
        >>> price, date = get_price_after_days('NVDA', '2025-09-15', 7)
        >>> print(f"7 days later: ${price:.2f} on {date}")
    """
    try:
        # Calculate target date
        base_date = datetime.strptime(date_str[:10], '%Y-%m-%d').date()
        target_date = base_date + timedelta(days=days)
        
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Get price on or after target date (next trading day)
            cursor.execute("""
                SELECT close, date FROM price_data
                WHERE ticker = ? AND date >= ?
                ORDER BY date ASC
                LIMIT 1
            """, (ticker, str(target_date)))
            
            row = cursor.fetchone()
            
            if row:
                return (float(row['close']), row['date'])
            
            logger.debug(f"No price data for {ticker} after {target_date}")
            return (None, None)
            
    except Exception as e:
        logger.error(f"Failed to get price for {ticker} after {days} days from {date_str}: {str(e)}")
        return (None, None)


def save_news_outcome(
    outcome_data: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Save a single news outcome evaluation.
    
    Used by background script to record prediction accuracy.
    
    Args:
        outcome_data: Dictionary with keys:
            - news_id: ID of the news article
            - ticker: Stock ticker symbol
            - price_at_news: Price when article was published
            - price_1day_later: Price 1 day later (optional)
            - price_3day_later: Price 3 days later (optional)
            - price_7day_later: Price 7 days later (required)
            - price_change_1day: % change 1 day (optional)
            - price_change_3day: % change 3 days (optional)
            - price_change_7day: % change 7 days (required)
            - predicted_direction: 'UP', 'DOWN', or 'FLAT'
            - actual_direction: 'UP', 'DOWN', or 'FLAT'
            - prediction_was_accurate_7day: Boolean
        db_path: Path to database
        
    Example:
        >>> outcome = {
        ...     'news_id': 123,
        ...     'ticker': 'NVDA',
        ...     'price_at_news': 135.00,
        ...     'price_7day_later': 139.50,
        ...     'price_change_7day': 3.33,
        ...     'predicted_direction': 'UP',
        ...     'actual_direction': 'UP',
        ...     'prediction_was_accurate_7day': True
        ... }
        >>> save_news_outcome(outcome)
    """
    try:
        with get_connection(db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO news_outcomes
                (news_id, ticker, price_at_news, price_1day_later,
                 price_3day_later, price_7day_later, price_change_1day,
                 price_change_3day, price_change_7day, predicted_direction,
                 actual_direction, prediction_was_accurate_7day, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                outcome_data['news_id'],
                outcome_data['ticker'],
                outcome_data['price_at_news'],
                outcome_data.get('price_1day_later'),
                outcome_data.get('price_3day_later'),
                outcome_data['price_7day_later'],
                outcome_data.get('price_change_1day'),
                outcome_data.get('price_change_3day'),
                outcome_data['price_change_7day'],
                outcome_data['predicted_direction'],
                outcome_data['actual_direction'],
                outcome_data['prediction_was_accurate_7day']
            ))
            conn.commit()
            
            logger.debug(f"Saved outcome for news_id {outcome_data['news_id']}: "
                        f"{outcome_data['predicted_direction']} vs {outcome_data['actual_direction']} "
                        f"({'✓' if outcome_data['prediction_was_accurate_7day'] else '✗'})")
            
    except Exception as e:
        logger.error(f"Failed to save news outcome for article {outcome_data.get('news_id')}: {str(e)}")


# ============================================================================
# NODE EXECUTION LOGGING
# ============================================================================

def log_node_execution(
    ticker: str,
    node_name: str,
    execution_time: float,
    success: bool,
    error_message: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Log node execution for performance monitoring.
    
    Args:
        ticker: Stock ticker symbol
        node_name: Name of the node (e.g., 'node_1', 'node_8')
        execution_time: Execution time in seconds
        success: Whether the node completed successfully
        error_message: Error message if failed
        db_path: Path to database
    """
    try:
        with get_connection(db_path) as conn:
            conn.execute("""
                INSERT INTO node_execution_logs
                (ticker, node_name, execution_time_seconds, success, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (ticker, node_name, execution_time, success, error_message))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log node execution: {str(e)}")


# ============================================================================
# DATABASE VERIFICATION
# ============================================================================

def verify_database(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Verify database structure and return statistics.
    
    Args:
        db_path: Path to database
    
    Returns:
        Dictionary with verification results:
        {
            'tables': List[str],
            'views': List[str],
            'indexes': List[str],
            'table_count': int,
            'view_count': int,
            'status': 'OK' or 'ERROR'
        }
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get all views
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
            views = [row[0] for row in cursor.fetchall()]
            
            # Get all indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
            indexes = [row[0] for row in cursor.fetchall()]
            
            return {
                'tables': tables,
                'views': views,
                'indexes': indexes,
                'table_count': len(tables),
                'view_count': len(views),
                'index_count': len(indexes),
                'status': 'OK'
            }
            
    except Exception as e:
        logger.error(f"Database verification failed: {str(e)}")
        return {
            'tables': [],
            'views': [],
            'indexes': [],
            'table_count': 0,
            'view_count': 0,
            'index_count': 0,
            'status': 'ERROR',
            'error': str(e)
        }
