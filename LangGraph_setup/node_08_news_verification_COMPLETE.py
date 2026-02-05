"""
Node 8: News Impact Verification & Learning System
CRITICAL THESIS INNOVATION

This node learns from historical news-price correlations to:
1. Calculate which news sources are reliable for each stock
2. Learn how different types of news impact price movement
3. Verify if sentiment predictions actually came true
4. Adjust confidence scores based on historical accuracy
5. Provide feedback loop for adaptive weighting

How It Works:
- Looks back 6 months of historical news + price data
- For each past news article, checks if price moved as predicted
- Calculates accuracy score per news source
- Adjusts current sentiment confidence based on source reliability
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import sqlite3
import json

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_historical_news_with_outcomes(ticker: str, days_back: int = 180) -> List[Dict]:
    """
    Fetch historical news articles with their price outcomes.
    
    Returns list of news events with:
    - Original news data (title, source, sentiment)
    - Price at time of news
    - Price 1, 3, 7 days later
    - Whether prediction was accurate
    """
    try:
        with sqlite3.connect('data/stock_prices.db') as conn:
            # Get all historical news with price outcomes
            query = """
            SELECT 
                n.id,
                n.ticker,
                n.news_type,
                n.title,
                n.source,
                n.published_at,
                n.sentiment_label,
                n.sentiment_score,
                no.price_at_news,
                no.price_1day_later,
                no.price_3day_later,
                no.price_7day_later,
                no.price_change_1day,
                no.price_change_3day,
                no.price_change_7day,
                no.prediction_was_accurate_7day,
                no.actual_direction
            FROM news_articles n
            LEFT JOIN news_outcomes no ON n.id = no.news_id
            WHERE n.ticker = ?
            AND n.published_at >= date('now', '-' || ? || ' days')
            AND n.is_filtered = 0
            AND no.id IS NOT NULL
            ORDER BY n.published_at DESC
            """
            
            cursor = conn.cursor()
            cursor.execute(query, (ticker, days_back))
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            logger.info(f"Retrieved {len(results)} historical news events with outcomes for {ticker}")
            return results
            
    except Exception as e:
        logger.error(f"Failed to fetch historical news outcomes: {str(e)}")
        return []


def calculate_source_reliability(news_events: List[Dict], ticker: str) -> Dict[str, Dict]:
    """
    Calculate reliability score for each news source based on historical accuracy.
    
    Returns:
    {
        'Bloomberg.com': {
            'total_articles': 20,
            'accurate_predictions': 17,
            'accuracy_rate': 0.85,
            'avg_price_impact': 2.3,  # Average % price change after their news
            'confidence_multiplier': 1.2  # How much to boost confidence
        },
        'random-blog.com': {
            'total_articles': 10,
            'accurate_predictions': 2,
            'accuracy_rate': 0.20,
            'avg_price_impact': -0.5,
            'confidence_multiplier': 0.5
        }
    }
    """
    source_stats = {}
    
    for event in news_events:
        source = event.get('source', 'unknown')
        
        if source not in source_stats:
            source_stats[source] = {
                'total_articles': 0,
                'accurate_predictions': 0,
                'price_changes': [],
                'articles': []
            }
        
        source_stats[source]['total_articles'] += 1
        source_stats[source]['articles'].append(event)
        
        # Check if prediction was accurate
        if event.get('prediction_was_accurate_7day'):
            source_stats[source]['accurate_predictions'] += 1
        
        # Track price change magnitude
        price_change = event.get('price_change_7day', 0)
        if price_change is not None:
            source_stats[source]['price_changes'].append(abs(price_change))
    
    # Calculate final metrics
    reliability_scores = {}
    
    for source, stats in source_stats.items():
        total = stats['total_articles']
        accurate = stats['accurate_predictions']
        
        if total == 0:
            continue
        
        accuracy_rate = accurate / total
        avg_price_impact = sum(stats['price_changes']) / len(stats['price_changes']) if stats['price_changes'] else 0
        
        # Calculate confidence multiplier
        # High accuracy (>80%) = boost confidence by 20%
        # Medium accuracy (50-80%) = keep confidence as is
        # Low accuracy (<50%) = reduce confidence
        if accuracy_rate >= 0.80:
            confidence_multiplier = 1.0 + (accuracy_rate - 0.80) * 2  # 1.0 to 1.4
        elif accuracy_rate >= 0.50:
            confidence_multiplier = 1.0
        else:
            confidence_multiplier = 0.5 + accuracy_rate  # 0.5 to 1.0
        
        reliability_scores[source] = {
            'total_articles': total,
            'accurate_predictions': accurate,
            'accuracy_rate': accuracy_rate,
            'avg_price_impact': avg_price_impact,
            'confidence_multiplier': confidence_multiplier
        }
    
    logger.info(f"Calculated reliability for {len(reliability_scores)} sources for {ticker}")
    
    return reliability_scores


def calculate_news_type_effectiveness(news_events: List[Dict], ticker: str) -> Dict[str, Dict]:
    """
    Calculate how effective each type of news is for predicting price movement.
    
    Types: 'stock' (company-specific), 'market' (market-wide), 'related' (competitors)
    
    Returns:
    {
        'stock': {
            'accuracy_rate': 0.68,
            'avg_impact': 3.2,
            'sample_size': 45
        },
        'market': {
            'accuracy_rate': 0.52,
            'avg_impact': 1.1,
            'sample_size': 30
        },
        'related': {
            'accuracy_rate': 0.61,
            'avg_impact': 1.8,
            'sample_size': 25
        }
    }
    """
    type_stats = {
        'stock': {'correct': 0, 'total': 0, 'price_changes': []},
        'market': {'correct': 0, 'total': 0, 'price_changes': []},
        'related': {'correct': 0, 'total': 0, 'price_changes': []}
    }
    
    for event in news_events:
        news_type = event.get('news_type', 'stock')
        
        if news_type in type_stats:
            type_stats[news_type]['total'] += 1
            
            if event.get('prediction_was_accurate_7day'):
                type_stats[news_type]['correct'] += 1
            
            price_change = event.get('price_change_7day', 0)
            if price_change is not None:
                type_stats[news_type]['price_changes'].append(abs(price_change))
    
    # Calculate final metrics
    effectiveness = {}
    
    for news_type, stats in type_stats.items():
        if stats['total'] == 0:
            effectiveness[news_type] = {
                'accuracy_rate': 0.5,  # Neutral default
                'avg_impact': 0.0,
                'sample_size': 0
            }
        else:
            effectiveness[news_type] = {
                'accuracy_rate': stats['correct'] / stats['total'],
                'avg_impact': sum(stats['price_changes']) / len(stats['price_changes']) if stats['price_changes'] else 0,
                'sample_size': stats['total']
            }
    
    logger.info(f"News type effectiveness for {ticker}: {effectiveness}")
    
    return effectiveness


def adjust_current_sentiment_confidence(
    current_sentiment: Dict,
    source_reliability: Dict[str, Dict],
    news_type_effectiveness: Dict[str, Dict],
    cleaned_news: List[Dict]
) -> Dict:
    """
    Adjust current sentiment analysis confidence based on learned patterns.
    
    Takes the sentiment from Node 5 and adjusts confidence scores based on:
    1. Source reliability (which websites are trustworthy)
    2. News type effectiveness (stock vs market vs related)
    3. Historical patterns for this stock
    """
    if not current_sentiment:
        return current_sentiment
    
    adjusted = current_sentiment.copy()
    
    # Get original confidence
    original_confidence = current_sentiment.get('confidence', 50)
    
    # Calculate weighted reliability for current news sources
    total_reliability = 0
    count = 0
    
    for article in cleaned_news:
        source = article.get('source', {}).get('name', 'unknown')
        if source in source_reliability:
            multiplier = source_reliability[source]['confidence_multiplier']
            total_reliability += multiplier
            count += 1
    
    avg_reliability_multiplier = total_reliability / count if count > 0 else 1.0
    
    # Get news type effectiveness
    # (Assuming we know which news type this sentiment came from)
    # For simplicity, use average of all types
    avg_news_effectiveness = sum(
        eff['accuracy_rate'] for eff in news_type_effectiveness.values()
    ) / len(news_type_effectiveness)
    
    # Adjust confidence
    # Formula: new_confidence = old_confidence * reliability_multiplier * effectiveness_factor
    effectiveness_factor = avg_news_effectiveness * 2  # Scale to ~1.0
    
    adjusted_confidence = original_confidence * avg_reliability_multiplier * effectiveness_factor
    adjusted_confidence = min(100, max(0, adjusted_confidence))  # Clamp to 0-100
    
    adjusted['confidence'] = adjusted_confidence
    adjusted['confidence_adjustment'] = {
        'original': original_confidence,
        'reliability_multiplier': avg_reliability_multiplier,
        'effectiveness_factor': effectiveness_factor,
        'final': adjusted_confidence
    }
    
    logger.info(f"Adjusted confidence from {original_confidence:.1f}% to {adjusted_confidence:.1f}%")
    
    return adjusted


def calculate_historical_correlation(news_events: List[Dict]) -> float:
    """
    Calculate overall correlation between news sentiment and price movement.
    
    Returns correlation coefficient (-1 to +1):
    - +1 = perfect positive correlation (positive news → price up)
    - 0 = no correlation
    - -1 = perfect negative correlation (positive news → price down)
    """
    if len(news_events) < 10:  # Need minimum sample size
        return 0.5  # Neutral
    
    sentiments = []
    price_changes = []
    
    for event in news_events:
        sentiment_label = event.get('sentiment_label', 'neutral')
        price_change = event.get('price_change_7day', 0)
        
        if price_change is None:
            continue
        
        # Convert sentiment to numeric
        if sentiment_label == 'positive':
            sentiment_numeric = 1
        elif sentiment_label == 'negative':
            sentiment_numeric = -1
        else:
            sentiment_numeric = 0
        
        sentiments.append(sentiment_numeric)
        price_changes.append(price_change)
    
    if len(sentiments) < 10:
        return 0.5
    
    # Calculate Pearson correlation
    try:
        df = pd.DataFrame({
            'sentiment': sentiments,
            'price_change': price_changes
        })
        correlation = df['sentiment'].corr(df['price_change'])
        
        if pd.isna(correlation):
            return 0.5
        
        # Normalize to 0-1 scale (where 0.5 is neutral)
        normalized = (correlation + 1) / 2  # -1→0, 0→0.5, +1→1
        
        return normalized
        
    except Exception as e:
        logger.warning(f"Correlation calculation failed: {str(e)}")
        return 0.5


def save_source_reliability_to_db(ticker: str, source_reliability: Dict, analysis_date: str):
    """Save source reliability scores to database for future reference"""
    try:
        with sqlite3.connect('data/stock_prices.db') as conn:
            cursor = conn.cursor()
            
            for source, stats in source_reliability.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO source_reliability 
                    (ticker, source_name, analysis_date, total_articles, 
                     accurate_predictions, accuracy_rate, avg_price_impact, 
                     confidence_multiplier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    source,
                    analysis_date,
                    stats['total_articles'],
                    stats['accurate_predictions'],
                    stats['accuracy_rate'],
                    stats['avg_price_impact'],
                    stats['confidence_multiplier']
                ))
            
            conn.commit()
            logger.info(f"Saved source reliability for {len(source_reliability)} sources")
            
    except Exception as e:
        logger.error(f"Failed to save source reliability: {str(e)}")


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def news_verification_node(state: 'StockAnalysisState') -> 'StockAnalysisState':
    """
    Node 8: News Impact Verification & Learning System
    
    This is the CRITICAL LEARNING NODE that makes your thesis innovative.
    
    What it does:
    1. Fetches 6 months of historical news + price data
    2. Calculates which news sources were accurate (Bloomberg vs random blogs)
    3. Calculates which news types are effective (stock vs market vs related)
    4. Adjusts current sentiment confidence based on learned patterns
    5. Provides feedback for adaptive weighting
    
    Runs AFTER parallel analysis nodes (4, 5, 6, 7)
    Runs BEFORE anomaly detection (9B)
    
    Args:
        state: LangGraph state
        
    Returns:
        Updated state with news_impact_verification results
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    try:
        logger.info(f"Node 8: Starting news verification for {ticker}")
        
        # Get current sentiment analysis (from Node 5)
        current_sentiment = state.get('sentiment_analysis')
        if not current_sentiment:
            logger.warning("No sentiment analysis available, skipping verification")
            state['news_impact_verification'] = None
            state['node_execution_times']['node_8'] = (datetime.now() - start_time).total_seconds()
            return state
        
        # Get cleaned news (from Node 9A)
        cleaned_stock_news = state.get('cleaned_stock_news', [])
        cleaned_market_news = state.get('cleaned_market_news', [])
        cleaned_related_news = state.get('cleaned_related_news', [])
        all_cleaned_news = cleaned_stock_news + cleaned_market_news + cleaned_related_news
        
        # STEP 1: Fetch historical news with outcomes
        logger.info("Fetching historical news events with price outcomes...")
        historical_events = get_historical_news_with_outcomes(ticker, days_back=180)
        
        if len(historical_events) < 10:
            logger.warning(f"Insufficient historical data ({len(historical_events)} events), using defaults")
            
            # Not enough data to learn - use neutral defaults
            state['news_impact_verification'] = {
                'historical_correlation': 0.5,
                'news_accuracy_score': 50.0,
                'verified_signal_strength': current_sentiment.get('confidence', 50),
                'learning_adjustment': 1.0,
                'sample_size': len(historical_events),
                'source_reliability': {},
                'news_type_effectiveness': {},
                'insufficient_data': True
            }
            state['node_execution_times']['node_8'] = (datetime.now() - start_time).total_seconds()
            return state
        
        # STEP 2: Calculate source reliability
        logger.info("Calculating source reliability...")
        source_reliability = calculate_source_reliability(historical_events, ticker)
        
        # STEP 3: Calculate news type effectiveness
        logger.info("Calculating news type effectiveness...")
        news_type_effectiveness = calculate_news_type_effectiveness(historical_events, ticker)
        
        # STEP 4: Calculate overall correlation
        logger.info("Calculating historical news-price correlation...")
        historical_correlation = calculate_historical_correlation(historical_events)
        
        # STEP 5: Adjust current sentiment confidence
        logger.info("Adjusting current sentiment confidence based on learned patterns...")
        adjusted_sentiment = adjust_current_sentiment_confidence(
            current_sentiment,
            source_reliability,
            news_type_effectiveness,
            all_cleaned_news
        )
        
        # Calculate overall news accuracy score (weighted average of all sources)
        if source_reliability:
            total_weight = sum(s['total_articles'] for s in source_reliability.values())
            weighted_accuracy = sum(
                s['accuracy_rate'] * s['total_articles'] 
                for s in source_reliability.values()
            )
            news_accuracy_score = (weighted_accuracy / total_weight) * 100 if total_weight > 0 else 50.0
        else:
            news_accuracy_score = 50.0
        
        # Calculate learning adjustment factor
        # This will be used by adaptive weighting (Node 11)
        # High accuracy sources = increase news weight
        # Low accuracy sources = decrease news weight
        learning_adjustment = (news_accuracy_score / 50.0) * historical_correlation * 2
        learning_adjustment = min(2.0, max(0.5, learning_adjustment))  # Clamp to 0.5-2.0
        
        # Build results
        results = {
            'historical_correlation': float(historical_correlation),
            'news_accuracy_score': float(news_accuracy_score),
            'verified_signal_strength': adjusted_sentiment.get('confidence', current_sentiment.get('confidence', 50)),
            'learning_adjustment': float(learning_adjustment),
            'sample_size': len(historical_events),
            'source_reliability': source_reliability,
            'news_type_effectiveness': news_type_effectiveness,
            'confidence_adjustment_details': adjusted_sentiment.get('confidence_adjustment', {}),
            'insufficient_data': False
        }
        
        logger.info(f"News verification complete:")
        logger.info(f"  - Historical correlation: {historical_correlation:.2f}")
        logger.info(f"  - News accuracy: {news_accuracy_score:.1f}%")
        logger.info(f"  - Verified confidence: {results['verified_signal_strength']:.1f}%")
        logger.info(f"  - Learning adjustment: {learning_adjustment:.2f}x")
        logger.info(f"  - Sample size: {len(historical_events)} events")
        
        # STEP 6: Save source reliability to database
        save_source_reliability_to_db(ticker, source_reliability, state['analysis_date'])
        
        # Update sentiment in state with adjusted confidence
        state['sentiment_analysis'] = adjusted_sentiment
        
        # Store verification results
        state['news_impact_verification'] = results
        state['node_execution_times']['node_8'] = (datetime.now() - start_time).total_seconds()
        
        return state
        
    except Exception as e:
        logger.error(f"Node 8 failed: {str(e)}")
        state['errors'].append(f"News verification failed: {str(e)}")
        
        # Failsafe: return neutral results
        state['news_impact_verification'] = {
            'historical_correlation': 0.5,
            'news_accuracy_score': 50.0,
            'verified_signal_strength': 50.0,
            'learning_adjustment': 1.0,
            'sample_size': 0,
            'source_reliability': {},
            'news_type_effectiveness': {},
            'error': str(e)
        }
        state['node_execution_times']['node_8'] = (datetime.now() - start_time).total_seconds()
        return state


# ============================================================================
# BACKGROUND TASK: Update Historical News Outcomes
# ============================================================================

def update_news_outcomes_background_task(ticker: str):
    """
    Background task to be run periodically (e.g., daily)
    
    For all historical news that don't have outcomes yet:
    1. Check if enough time has passed (7 days)
    2. Fetch the price 7 days after the news
    3. Determine if prediction was accurate
    4. Save to news_outcomes table
    
    This builds the historical dataset that Node 8 learns from.
    """
    try:
        with sqlite3.connect('data/stock_prices.db') as conn:
            cursor = conn.cursor()
            
            # Get news articles without outcomes that are old enough (7+ days)
            cursor.execute("""
                SELECT n.id, n.ticker, n.published_at, n.sentiment_label
                FROM news_articles n
                LEFT JOIN news_outcomes no ON n.id = no.news_id
                WHERE n.ticker = ?
                AND no.id IS NULL
                AND n.published_at <= date('now', '-7 days')
                AND n.is_filtered = 0
                LIMIT 100
            """, (ticker,))
            
            pending_news = cursor.fetchall()
            
            logger.info(f"Found {len(pending_news)} news articles needing outcome evaluation")
            
            for news_id, ticker, published_at, sentiment_label in pending_news:
                try:
                    # Get price data
                    published_date = datetime.fromisoformat(published_at).date()
                    
                    # Get price at time of news
                    cursor.execute("""
                        SELECT close FROM price_data
                        WHERE ticker = ? AND date = ?
                    """, (ticker, str(published_date)))
                    price_at_news_row = cursor.fetchone()
                    
                    if not price_at_news_row:
                        continue
                    
                    price_at_news = price_at_news_row[0]
                    
                    # Get prices 1, 3, 7 days later
                    prices = {}
                    for days in [1, 3, 7]:
                        future_date = published_date + timedelta(days=days)
                        cursor.execute("""
                            SELECT close FROM price_data
                            WHERE ticker = ? AND date >= ?
                            ORDER BY date ASC LIMIT 1
                        """, (ticker, str(future_date)))
                        
                        row = cursor.fetchone()
                        if row:
                            prices[f'price_{days}day'] = row[0]
                    
                    if 'price_7day' not in prices:
                        continue  # Can't evaluate without 7-day price
                    
                    # Calculate price changes
                    price_change_1day = ((prices.get('price_1day', price_at_news) - price_at_news) / price_at_news) * 100 if 'price_1day' in prices else None
                    price_change_3day = ((prices.get('price_3day', price_at_news) - price_at_news) / price_at_news) * 100 if 'price_3day' in prices else None
                    price_change_7day = ((prices['price_7day'] - price_at_news) / price_at_news) * 100
                    
                    # Determine actual direction
                    if price_change_7day > 0.5:
                        actual_direction = 'UP'
                    elif price_change_7day < -0.5:
                        actual_direction = 'DOWN'
                    else:
                        actual_direction = 'FLAT'
                    
                    # Determine if prediction was accurate
                    predicted_direction = 'UP' if sentiment_label == 'positive' else 'DOWN' if sentiment_label == 'negative' else 'FLAT'
                    prediction_accurate = (predicted_direction == actual_direction)
                    
                    # Insert outcome
                    cursor.execute("""
                        INSERT INTO news_outcomes 
                        (news_id, ticker, price_at_news, price_1day_later, price_3day_later, 
                         price_7day_later, price_change_1day, price_change_3day, price_change_7day,
                         predicted_direction, actual_direction, prediction_was_accurate_7day)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        news_id, ticker, price_at_news,
                        prices.get('price_1day'), prices.get('price_3day'), prices['price_7day'],
                        price_change_1day, price_change_3day, price_change_7day,
                        predicted_direction, actual_direction, prediction_accurate
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate outcome for news {news_id}: {str(e)}")
                    continue
            
            conn.commit()
            logger.info(f"Successfully evaluated {len(pending_news)} news outcomes for {ticker}")
            
    except Exception as e:
        logger.error(f"Background task failed: {str(e)}")


# ============================================================================
# UTILITY: Display Source Reliability Report
# ============================================================================

def generate_source_reliability_report(ticker: str, source_reliability: Dict[str, Dict]) -> str:
    """
    Generate human-readable report of source reliability.
    
    This can be displayed in the dashboard to show users which sources
    are trustworthy for this specific stock.
    """
    if not source_reliability:
        return "Insufficient data to generate reliability report."
    
    # Sort by accuracy rate
    sorted_sources = sorted(
        source_reliability.items(),
        key=lambda x: x[1]['accuracy_rate'],
        reverse=True
    )
    
    report = f"# News Source Reliability Report for {ticker}\n\n"
    report += f"Based on {sum(s[1]['total_articles'] for s in sorted_sources)} historical articles\n\n"
    
    report += "## Most Reliable Sources:\n\n"
    
    for source, stats in sorted_sources[:10]:  # Top 10
        accuracy_pct = stats['accuracy_rate'] * 100
        emoji = "🟢" if accuracy_pct >= 70 else "🟡" if accuracy_pct >= 50 else "🔴"
        
        report += f"{emoji} **{source}**\n"
        report += f"   - Accuracy: {accuracy_pct:.1f}% ({stats['accurate_predictions']}/{stats['total_articles']} correct)\n"
        report += f"   - Avg Price Impact: {stats['avg_price_impact']:.2f}%\n"
        report += f"   - Confidence Boost: {stats['confidence_multiplier']:.2f}x\n\n"
    
    if len(sorted_sources) > 10:
        report += f"\n... and {len(sorted_sources) - 10} more sources\n"
    
    return report
