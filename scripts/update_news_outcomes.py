"""
Background Script: News Outcomes Evaluator

Evaluates historical news articles to determine if their sentiment predictions
were accurate. Fills the news_outcomes table that Node 8 uses for learning.

This script checks what actually happened to stock prices 7 days after each
news article was published, then records whether the sentiment prediction
(positive → UP, negative → DOWN) was correct.

Runs separately from the main pipeline - can be run manually or via cron.

Usage:
    python -m scripts.update_news_outcomes                # All tickers
    python -m scripts.update_news_outcomes --ticker NVDA  # Single ticker
    python -m scripts.update_news_outcomes --limit 1000   # Custom limit
    
When to run:
    - FIRST TIME: After Nodes 1-2 have fetched 6 months of historical data
      (will backfill hundreds of outcomes at once)
    - ONGOING: Run daily to evaluate articles that have aged past 7 days
    - BEFORE TESTING NODE 8: Node 8 needs data in news_outcomes table to learn from
"""

import sys
import argparse
from datetime import datetime, timedelta
from typing import Dict, Optional, List

# Database operations
from src.database.db_manager import (
    get_news_outcomes_pending,
    get_price_on_date,
    get_price_after_days,
    save_news_outcome
)

# Logger
from src.utils.logger import get_node_logger

logger = get_node_logger("update_news_outcomes")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def determine_predicted_direction(sentiment_label: Optional[str]) -> str:
    """
    Convert sentiment label to predicted price direction.
    
    Handles both Alpha Vantage and FinBERT label formats:
    - Alpha Vantage: 'Bullish', 'Somewhat-Bullish', 'Neutral', 'Somewhat-Bearish', 'Bearish'
    - FinBERT: 'positive', 'negative', 'neutral'
    
    Args:
        sentiment_label: Sentiment label from news article
        
    Returns:
        'UP', 'DOWN', or 'FLAT'
        
    Example:
        >>> determine_predicted_direction('positive')
        'UP'
        >>> determine_predicted_direction('Bearish')
        'DOWN'
        >>> determine_predicted_direction('Neutral')
        'FLAT'
    """
    if not sentiment_label:
        return 'FLAT'
    
    label = sentiment_label.lower().strip()
    
    # Alpha Vantage format
    if label in ['bullish', 'somewhat-bullish', 'somewhat_bullish']:
        return 'UP'
    if label in ['bearish', 'somewhat-bearish', 'somewhat_bearish']:
        return 'DOWN'
    
    # FinBERT / normalized format
    if label == 'positive':
        return 'UP'
    if label == 'negative':
        return 'DOWN'
    
    # Neutral or unknown → FLAT
    return 'FLAT'


def determine_actual_direction(price_change_pct: float, threshold: float = 0.5) -> str:
    """
    Determine actual price direction from percentage change.
    
    Uses 0.5% threshold to distinguish meaningful movement from noise.
    
    Args:
        price_change_pct: Percentage price change (e.g., 2.5 for +2.5%)
        threshold: Minimum % change to count as UP/DOWN (default: 0.5)
        
    Returns:
        'UP' if change > threshold
        'DOWN' if change < -threshold
        'FLAT' if within ±threshold
        
    Example:
        >>> determine_actual_direction(3.5)
        'UP'
        >>> determine_actual_direction(-2.1)
        'DOWN'
        >>> determine_actual_direction(0.3)
        'FLAT'
    """
    if price_change_pct > threshold:
        return 'UP'
    elif price_change_pct < -threshold:
        return 'DOWN'
    return 'FLAT'


# ============================================================================
# CORE EVALUATION LOGIC
# ============================================================================

def evaluate_single_article(article: Dict) -> Optional[Dict]:
    """
    Evaluate a single article's prediction accuracy.
    
    Looks up what actually happened to the stock price 1, 3, and 7 days
    after the article was published, and checks if the sentiment prediction
    matched reality.
    
    Args:
        article: Dictionary with keys:
            - id: Article ID
            - ticker: Stock ticker
            - published_at: Publication date (ISO format)
            - sentiment_label: Sentiment label
            - sentiment_score: Sentiment confidence
            
    Returns:
        Outcome dictionary for database insertion, or None if price data unavailable
        
    Example:
        >>> article = {
        ...     'id': 123,
        ...     'ticker': 'NVDA',
        ...     'published_at': '2025-09-15',
        ...     'sentiment_label': 'positive'
        ... }
        >>> outcome = evaluate_single_article(article)
        >>> outcome['prediction_was_accurate_7day']
        True
    """
    ticker = article['ticker']
    published_at = article['published_at']
    article_id = article['id']
    
    # =========================================================================
    # Step 1: Parse published date
    # =========================================================================
    try:
        # Handle various date formats
        if 'T' in str(published_at):
            # ISO format: '2025-09-15T14:30:00'
            pub_date = datetime.fromisoformat(str(published_at).replace('Z', '')).date()
        else:
            # Simple format: '2025-09-15'
            pub_date = datetime.strptime(str(published_at)[:10], '%Y-%m-%d').date()
    except (ValueError, TypeError) as e:
        logger.warning(f"Cannot parse date '{published_at}' for article {article_id}: {str(e)}")
        return None
    
    # =========================================================================
    # Step 2: Get price at time of news
    # =========================================================================
    price_at_news = get_price_on_date(ticker, str(pub_date))
    
    if price_at_news is None:
        logger.debug(f"No price data for {ticker} on {pub_date}, skipping article {article_id}")
        return None
    
    # =========================================================================
    # Step 3: Get prices 1, 3, 7 days later
    # =========================================================================
    prices = {}
    
    for days in [1, 3, 7]:
        result = get_price_after_days(ticker, str(pub_date), days)
        if result and result[0] is not None:
            prices[days] = result[0]  # Extract price value
    
    # Must have 7-day price to evaluate (primary metric)
    if 7 not in prices:
        logger.debug(f"No 7-day price for {ticker} after {pub_date}, skipping article {article_id}")
        return None
    
    # =========================================================================
    # Step 4: Calculate price changes (percentage)
    # =========================================================================
    price_change_1day = None
    price_change_3day = None
    price_change_7day = None
    
    if 1 in prices:
        price_change_1day = ((prices[1] - price_at_news) / price_at_news) * 100
    
    if 3 in prices:
        price_change_3day = ((prices[3] - price_at_news) / price_at_news) * 100
    
    # 7-day is required
    price_change_7day = ((prices[7] - price_at_news) / price_at_news) * 100
    
    # =========================================================================
    # Step 5: Determine predicted vs actual directions
    # =========================================================================
    predicted_direction = determine_predicted_direction(article.get('sentiment_label'))
    actual_direction = determine_actual_direction(price_change_7day)
    
    # Check if prediction was accurate
    prediction_was_accurate_7day = (predicted_direction == actual_direction)
    
    # Also check 1-day and 3-day accuracy for completeness
    prediction_was_accurate_1day = None
    prediction_was_accurate_3day = None
    
    if price_change_1day is not None:
        actual_1day = determine_actual_direction(price_change_1day)
        prediction_was_accurate_1day = (predicted_direction == actual_1day)
    
    if price_change_3day is not None:
        actual_3day = determine_actual_direction(price_change_3day)
        prediction_was_accurate_3day = (predicted_direction == actual_3day)
    
    # =========================================================================
    # Step 6: Build outcome record
    # =========================================================================
    outcome = {
        'news_id': article_id,
        'ticker': ticker,
        'price_at_news': price_at_news,
        'price_1day_later': prices.get(1),
        'price_3day_later': prices.get(3),
        'price_7day_later': prices[7],
        'price_change_1day': price_change_1day,
        'price_change_3day': price_change_3day,
        'price_change_7day': price_change_7day,
        'predicted_direction': predicted_direction,
        'actual_direction': actual_direction,
        'prediction_was_accurate_1day': prediction_was_accurate_1day,
        'prediction_was_accurate_3day': prediction_was_accurate_3day,
        'prediction_was_accurate_7day': prediction_was_accurate_7day
    }
    
    logger.debug(f"Article {article_id}: {predicted_direction} vs {actual_direction} "
                f"({price_change_7day:+.2f}%) → {'✓' if prediction_was_accurate_7day else '✗'}")
    
    return outcome


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def run_evaluation(ticker: Optional[str] = None, limit: int = 500, verbose: bool = True) -> Dict:
    """
    Main evaluation function - processes pending articles and saves outcomes.
    
    Args:
        ticker: Optional ticker symbol (if None, evaluates all tickers)
        limit: Maximum articles to process per run (default: 500)
        verbose: Print progress messages (default: True)
        
    Returns:
        Dictionary with results:
        {
            'evaluated': int,    # Successfully evaluated
            'skipped': int,      # Skipped (missing price data)
            'accurate': int,     # Correct predictions
            'total': int,        # Total pending
            'accuracy_pct': float
        }
        
    Example:
        >>> results = run_evaluation(ticker='NVDA', limit=100)
        >>> print(f"Accuracy: {results['accuracy_pct']:.1f}%")
    """
    start_time = datetime.now()
    
    logger.info(f"Starting news outcomes evaluation"
               f"{f' for {ticker}' if ticker else ' for all tickers'}"
               f" (limit: {limit})")
    
    # =========================================================================
    # Step 1: Get pending articles needing evaluation
    # =========================================================================
    pending = get_news_outcomes_pending(ticker=ticker, limit=limit)
    
    if not pending:
        logger.info("No pending articles to evaluate")
        return {
            'evaluated': 0,
            'skipped': 0,
            'accurate': 0,
            'total': 0,
            'accuracy_pct': 0.0
        }
    
    logger.info(f"Found {len(pending)} articles needing evaluation")
    
    # =========================================================================
    # Step 2: Evaluate each article
    # =========================================================================
    evaluated = 0
    skipped = 0
    accurate = 0
    with_sentiment = 0   # Articles that had sentiment_label set (not NULL)
    
    for i, article in enumerate(pending, 1):
        try:
            # Progress reporting
            if verbose and i % 50 == 0:
                logger.info(f"Progress: {i}/{len(pending)} articles processed...")
            
            if article.get('sentiment_label'):
                with_sentiment += 1
            
            # Evaluate this article
            outcome = evaluate_single_article(article)
            
            if outcome is None:
                # Missing price data - skip
                skipped += 1
                continue
            
            # Save to database
            save_news_outcome(outcome)
            evaluated += 1
            
            if outcome['prediction_was_accurate_7day']:
                accurate += 1
                
        except Exception as e:
            logger.error(f"Failed to evaluate article {article.get('id', '?')}: {str(e)}")
            skipped += 1
            continue
    
    # =========================================================================
    # Step 3: Calculate summary statistics
    # =========================================================================
    accuracy_pct = (accurate / evaluated * 100) if evaluated > 0 else 0.0
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Evaluation complete in {execution_time:.2f}s:")
    logger.info(f"  Total found: {len(pending)}")
    logger.info(f"  Evaluated: {evaluated}")
    logger.info(f"  Skipped: {skipped} (missing price data)")
    logger.info(f"  With sentiment label: {with_sentiment}/{len(pending)} (rest predict as FLAT)")
    logger.info(f"  Accurate predictions: {accurate}/{evaluated} ({accuracy_pct:.1f}%)")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"News Outcomes Evaluation Results")
        print(f"{'='*60}")
        if ticker:
            print(f"Ticker: {ticker}")
        print(f"Total found: {len(pending)}")
        print(f"Evaluated: {evaluated}")
        print(f"Skipped: {skipped} (missing price data)")
        print(f"With sentiment label: {with_sentiment}/{len(pending)} (rest → predicted FLAT)")
        print(f"Accurate predictions: {accurate}/{evaluated} ({accuracy_pct:.1f}%)")
        print(f"Execution time: {execution_time:.2f}s")
        print(f"{'='*60}\n")
    
    return {
        'evaluated': evaluated,
        'skipped': skipped,
        'accurate': accurate,
        'total': len(pending),
        'with_sentiment': with_sentiment,
        'accuracy_pct': accuracy_pct,
        'execution_time': execution_time
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for the script"""
    parser = argparse.ArgumentParser(
        description='Evaluate news prediction outcomes for Node 8 learning system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backfill all tickers (first time)
  python -m scripts.update_news_outcomes
  
  # Single ticker
  python -m scripts.update_news_outcomes --ticker NVDA
  
  # Larger batch
  python -m scripts.update_news_outcomes --limit 1000
  
  # Quiet mode
  python -m scripts.update_news_outcomes --quiet
        """
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        default=None,
        help='Specific ticker to evaluate (default: all tickers)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=500,
        help='Max articles to process per run (default: 500)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output (log only)'
    )
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        results = run_evaluation(
            ticker=args.ticker,
            limit=args.limit,
            verbose=not args.quiet
        )
        
        # Exit with success
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        logger.exception("Full traceback:")
        print(f"\n\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
