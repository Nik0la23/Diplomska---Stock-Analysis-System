"""
Node 5: Sentiment Analysis

Analyzes sentiment from THREE cleaned news streams using:
1. Alpha Vantage built-in sentiment scores (primary)
2. FinBERT model for articles without sentiment (fallback)

CRITICAL: Uses cleaned_*_news from Node 9A, NOT raw news.

Dual-mode approach:
- Phase 1: Aggregate existing Alpha Vantage sentiment scores
- Phase 2: FinBERT infrastructure for articles lacking sentiment

Runs in PARALLEL with: Nodes 4, 6, 7
Runs AFTER: Node 9A (content analysis)
Runs BEFORE: Node 8 (verification)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np

# Sentiment configuration
from src.utils.sentiment_config import (
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    STOCK_NEWS_WEIGHT,
    MARKET_NEWS_WEIGHT,
    RELATED_NEWS_WEIGHT,
    USE_FINBERT,
    FINBERT_MODEL_NAME,
    FINBERT_DEVICE,
    USE_TICKER_SENTIMENT,
    normalize_sentiment_score,
    classify_sentiment,
    CREDIBILITY_SOURCE_WEIGHT,
    CREDIBILITY_ANOMALY_WEIGHT,
    CREDIBILITY_RELEVANCE_WEIGHT,
    CREDIBILITY_WEIGHT_FLOOR,
    CREDIBILITY_WEIGHT_CEILING,
    CREDIBILITY_CONFIDENCE_MIN,
    CREDIBILITY_CONFIDENCE_MAX
)

logger = logging.getLogger(__name__)

# Global variable for FinBERT model (load once, reuse)
_FINBERT_MODEL = None


# ============================================================================
# HELPER FUNCTION 1: Extract Alpha Vantage Sentiment
# ============================================================================

def extract_alpha_vantage_sentiment(articles: List[Dict]) -> List[Dict]:
    """
    Extract existing sentiment scores from Alpha Vantage articles.
    
    Alpha Vantage provides:
    - overall_sentiment_score: Overall article sentiment
    - ticker_sentiment_score: Ticker-specific sentiment
    
    Algorithm:
    1. For each article, check for sentiment scores
    2. Extract ticker_sentiment_score (preferred) or overall_sentiment_score
    3. Normalize to -1.0 to +1.0 range
    4. Mark articles with sentiment vs without
    
    Args:
        articles: List of news articles (may include Alpha Vantage and Finnhub)
        
    Returns:
        List of articles with 'sentiment_score' and 'has_sentiment' fields added
        
    Example:
        >>> articles = [
        ...     {'title': 'Good news', 'overall_sentiment_score': 0.5, 'ticker_sentiment_score': 0.7},
        ...     {'title': 'Bad news', 'source': 'finnhub'}
        ... ]
        >>> result = extract_alpha_vantage_sentiment(articles)
        >>> result[0]['sentiment_score']  # 0.7 (ticker sentiment)
        >>> result[1]['has_sentiment']     # False (no sentiment from Finnhub)
    """
    processed_articles = []
    
    for article in articles:
        article_copy = article.copy()
        
        # Try to extract sentiment from Alpha Vantage
        if USE_TICKER_SENTIMENT and 'ticker_sentiment_score' in article:
            # Prefer ticker-specific sentiment
            raw_score = article.get('ticker_sentiment_score', 0.0)
            sentiment_score = normalize_sentiment_score(raw_score)
            article_copy['sentiment_score'] = sentiment_score
            article_copy['has_sentiment'] = True
            article_copy['sentiment_source'] = 'alpha_vantage_ticker'
            
        elif 'overall_sentiment_score' in article:
            # Fallback to overall sentiment
            raw_score = article.get('overall_sentiment_score', 0.0)
            sentiment_score = normalize_sentiment_score(raw_score)
            article_copy['sentiment_score'] = sentiment_score
            article_copy['has_sentiment'] = True
            article_copy['sentiment_source'] = 'alpha_vantage_overall'
            
        else:
            # No sentiment available (likely Finnhub article)
            article_copy['sentiment_score'] = 0.0  # Neutral default
            article_copy['has_sentiment'] = False
            article_copy['sentiment_source'] = 'none'
        
        # Add sentiment label
        article_copy['sentiment_label'] = classify_sentiment(article_copy['sentiment_score'])
        
        processed_articles.append(article_copy)
    
    # Log statistics
    with_sentiment = sum(1 for a in processed_articles if a['has_sentiment'])
    without_sentiment = len(processed_articles) - with_sentiment
    
    logger.info(f"Sentiment extraction: {with_sentiment} with sentiment, {without_sentiment} without")
    
    return processed_articles


# ============================================================================
# HELPER FUNCTION 2: Load FinBERT Model
# ============================================================================

def load_finbert_model():
    """
    Load FinBERT model from HuggingFace.
    
    FinBERT is a BERT model fine-tuned for financial sentiment analysis.
    Model: ProsusAI/finbert
    
    Algorithm:
    1. Check if model already loaded (global cache)
    2. If not, load from HuggingFace transformers
    3. Cache in global variable for reuse
    4. Handle loading errors gracefully
    
    Returns:
        Transformers pipeline object or None if loading fails
        
    Example:
        >>> model = load_finbert_model()
        >>> if model:
        ...     result = model("Stock prices are rising")
        ...     print(result[0]['label'])  # 'positive'
    """
    global _FINBERT_MODEL
    
    # Return cached model if available
    if _FINBERT_MODEL is not None:
        logger.debug("Using cached FinBERT model")
        return _FINBERT_MODEL
    
    try:
        logger.info(f"Loading FinBERT model: {FINBERT_MODEL_NAME}")
        
        from transformers import pipeline
        
        # Load sentiment analysis pipeline with FinBERT
        _FINBERT_MODEL = pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL_NAME,
            device=FINBERT_DEVICE,  # -1 for CPU, 0 for GPU
            top_k=None  # Return all labels with scores
        )
        
        logger.info("FinBERT model loaded successfully")
        return _FINBERT_MODEL
        
    except Exception as e:
        logger.error(f"Failed to load FinBERT model: {str(e)}")
        logger.warning("Sentiment analysis will use Alpha Vantage scores only")
        return None


# ============================================================================
# HELPER FUNCTION 3: Analyze Text with FinBERT
# ============================================================================

def analyze_text_with_finbert(text: str, model) -> Dict[str, float]:
    """
    Analyze single text using FinBERT model.
    
    FinBERT returns: [{'label': 'positive', 'score': 0.95}, ...]
    
    Algorithm:
    1. Clean and truncate text (FinBERT has 512 token limit)
    2. Run through FinBERT model
    3. Get highest-confidence label
    4. Normalize to -1.0 to +1.0 scale
    
    Args:
        text: Text to analyze (title + description)
        model: FinBERT pipeline object
        
    Returns:
        {
            'label': 'positive' | 'negative' | 'neutral',
            'score': float,           # Raw confidence (0-1)
            'sentiment': float        # Normalized sentiment (-1 to +1)
        }
        
    Example:
        >>> model = load_finbert_model()
        >>> result = analyze_text_with_finbert("Company beats earnings", model)
        >>> result['label']      # 'positive'
        >>> result['sentiment']  # 0.85
    """
    if model is None:
        return {'label': 'neutral', 'score': 0.0, 'sentiment': 0.0}
    
    try:
        # Clean text
        text = text.strip()
        if not text:
            return {'label': 'neutral', 'score': 0.0, 'sentiment': 0.0}
        
        # Truncate to avoid token limit (rough estimate: 512 tokens ≈ 2000 chars)
        if len(text) > 2000:
            text = text[:2000]
        
        # Run FinBERT
        results = model(text)[0]  # Returns list of dicts
        
        # Find highest-confidence label
        best_result = max(results, key=lambda x: x['score'])
        label = best_result['label'].lower()
        confidence = best_result['score']
        
        # Normalize to -1 to +1
        if label == 'positive':
            sentiment = confidence  # 0 to +1
        elif label == 'negative':
            sentiment = -confidence  # -1 to 0
        else:  # neutral
            sentiment = 0.0
        
        return {
            'label': label,
            'score': confidence,
            'sentiment': sentiment
        }
        
    except Exception as e:
        logger.error(f"FinBERT analysis failed: {str(e)}")
        return {'label': 'neutral', 'score': 0.0, 'sentiment': 0.0}


# ============================================================================
# HELPER FUNCTION 4: Analyze Articles Batch
# ============================================================================

def analyze_articles_batch(
    articles: List[Dict], 
    use_finbert: bool = USE_FINBERT
) -> List[Dict]:
    """
    Process batch of articles, applying FinBERT to those without sentiment.
    
    Algorithm:
    1. First pass: Extract Alpha Vantage sentiment
    2. Identify articles without sentiment
    3. If use_finbert=True, analyze missing ones with FinBERT
    4. Return all articles with sentiment scores
    
    Args:
        articles: List of news articles
        use_finbert: Whether to use FinBERT for articles without sentiment
        
    Returns:
        List of articles with sentiment_score and sentiment_label
        
    Example:
        >>> articles = [
        ...     {'title': 'Good', 'ticker_sentiment_score': 0.8},  # Has sentiment
        ...     {'title': 'Bad', 'source': 'finnhub'}              # No sentiment
        ... ]
        >>> result = analyze_articles_batch(articles, use_finbert=True)
        >>> all('sentiment_score' in a for a in result)  # True
    """
    # Step 1: Extract existing Alpha Vantage sentiment
    processed = extract_alpha_vantage_sentiment(articles)
    
    # Count articles without sentiment
    missing_sentiment = [a for a in processed if not a['has_sentiment']]
    
    if not missing_sentiment:
        logger.info("All articles have sentiment scores from Alpha Vantage")
        return processed
    
    logger.info(f"{len(missing_sentiment)} articles lack sentiment scores")
    
    # Step 2: Apply FinBERT if enabled
    if use_finbert and missing_sentiment:
        logger.info("Loading FinBERT model for missing sentiment scores...")
        model = load_finbert_model()
        
        if model is None:
            logger.warning("FinBERT unavailable, using neutral sentiment for missing articles")
            return processed
        
        logger.info(f"Analyzing {len(missing_sentiment)} articles with FinBERT...")
        
        for article in processed:
            if not article['has_sentiment']:
                # Combine title and summary for analysis
                title = article.get('title', '')
                summary = article.get('summary', '') or article.get('description', '')
                text = f"{title}. {summary}"
                
                # Analyze with FinBERT
                result = analyze_text_with_finbert(text, model)
                
                # Update article
                article['sentiment_score'] = result['sentiment']
                article['sentiment_label'] = result['label']
                article['sentiment_source'] = 'finbert'
                article['has_sentiment'] = True
                article['finbert_confidence'] = result['score']
        
        logger.info("FinBERT analysis complete")
    
    return processed


# ============================================================================
# HELPER FUNCTION 5: Calculate Credibility Weight
# ============================================================================

def calculate_credibility_weight(article: Dict) -> float:
    """
    Calculate how much this article should influence the sentiment score.
    
    Uses Node 9A's scores:
    - source_credibility_score (most important, 50% of weight)
    - composite_anomaly_score (inverted - high anomaly = low weight, 30%)
    - relevance_score from Alpha Vantage (20%)
    
    Algorithm:
    1. Extract scores with safe defaults if missing
    2. Invert anomaly score (high anomaly = low quality)
    3. Weighted combination (50/30/20)
    4. Floor at 0.1, ceiling at 1.0
    
    Args:
        article: Article dictionary with optional Node 9A scores
        
    Returns:
        Weight from 0.1 (almost ignore) to 1.0 (full trust)
        
    Examples:
        Bloomberg article, low anomaly, high relevance → 0.93
        Random blog, high anomaly, low relevance → 0.22
        Unknown source, medium anomaly, medium relevance → 0.48
    """
    # Get scores (with safe defaults if missing)
    credibility = article.get('source_credibility_score', 0.5)    # default: unknown
    anomaly = article.get('composite_anomaly_score', 0.3)         # default: moderate
    relevance = article.get('relevance_score', 0.5)               # default: moderate
    
    # Invert anomaly (high anomaly = low weight)
    anomaly_quality = 1.0 - anomaly
    
    # Weighted combination
    weight = (
        credibility * CREDIBILITY_SOURCE_WEIGHT +       # Source reputation matters most
        anomaly_quality * CREDIBILITY_ANOMALY_WEIGHT +   # Content quality
        relevance * CREDIBILITY_RELEVANCE_WEIGHT         # Relevance to ticker
    )
    
    # Floor at 0.1 (never completely ignore — Node 9B needs to see patterns)
    # Cap at 1.0
    return max(CREDIBILITY_WEIGHT_FLOOR, min(CREDIBILITY_WEIGHT_CEILING, weight))


# ============================================================================
# HELPER FUNCTION 6: Aggregate Sentiment by Type
# ============================================================================

def aggregate_sentiment_by_type(articles: List[Dict], news_type: str) -> Dict[str, Any]:
    """
    Aggregate sentiment across articles of one type.
    
    Algorithm:
    1. Calculate average sentiment
    2. Count positive/negative/neutral
    3. Calculate weighted sentiment (recent articles weigh more)
    4. Calculate confidence based on article count and consistency
    
    Args:
        articles: List of articles (same type: stock, market, or related)
        news_type: Type label for logging ('stock', 'market', 'related')
        
    Returns:
        {
            'news_type': str,
            'article_count': int,
            'average_sentiment': float,           # Simple average
            'weighted_sentiment': float,          # Time-weighted average
            'positive_count': int,
            'negative_count': int,
            'neutral_count': int,
            'sentiment_signal': 'BUY'|'SELL'|'HOLD',
            'confidence': float
        }
        
    Example:
        >>> articles = [
        ...     {'sentiment_score': 0.8, 'sentiment_label': 'positive'},
        ...     {'sentiment_score': 0.6, 'sentiment_label': 'positive'}
        ... ]
        >>> result = aggregate_sentiment_by_type(articles, 'stock')
        >>> result['average_sentiment']  # 0.7
        >>> result['sentiment_signal']   # 'BUY'
    """
    if not articles:
        return {
            'news_type': news_type,
            'article_count': 0,
            'average_sentiment': 0.0,
            'weighted_sentiment': 0.0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'sentiment_signal': 'HOLD',
            'confidence': 0.0
        }
    
    # Extract sentiment scores
    sentiments = [a.get('sentiment_score', 0.0) for a in articles]
    
    # Calculate simple average
    average_sentiment = np.mean(sentiments) if sentiments else 0.0
    
    # Calculate weighted average with BOTH time decay AND credibility weighting
    # Assuming articles are sorted by time (newest first)
    combined_weights = []
    for i, article in enumerate(articles):
        time_weight = 0.95 ** i  # Exponential time decay
        credibility_weight = calculate_credibility_weight(article)
        combined_weight = time_weight * credibility_weight
        combined_weights.append(combined_weight)
    
    # Calculate weighted sentiment using combined weights
    if combined_weights and sum(combined_weights) > 0:
        weighted_sentiment = np.average(sentiments, weights=combined_weights)
    else:
        weighted_sentiment = 0.0
    
    # Count sentiment labels
    labels = [a.get('sentiment_label', 'neutral') for a in articles]
    positive_count = labels.count('positive')
    negative_count = labels.count('negative')
    neutral_count = labels.count('neutral')
    
    # Determine signal based on weighted sentiment
    if weighted_sentiment > BUY_THRESHOLD:
        signal = 'BUY'
    elif weighted_sentiment < SELL_THRESHOLD:
        signal = 'SELL'
    else:
        signal = 'HOLD'
    
    # Calculate confidence based on:
    # 1. Number of articles (more = higher confidence)
    # 2. Consistency of sentiment (all positive/negative = higher)
    # 3. Average source credibility (NEW: high-credibility sources = higher confidence)
    article_count = len(articles)
    consistency = max(positive_count, negative_count, neutral_count) / article_count if article_count > 0 else 0
    
    # Base confidence formula
    count_factor = min(article_count / 10.0, 1.0)  # Max at 10 articles
    base_confidence = (consistency * 0.7 + count_factor * 0.3)  # Weighted combination
    
    # Average credibility of sources used (NEW)
    avg_credibility = sum(
        article.get('source_credibility_score', 0.5) for article in articles
    ) / len(articles) if articles else 0.5
    
    # Adjust confidence: high-credibility sources = higher confidence
    # Scale: avg_credibility of 0.9 boosts confidence by ~10%
    #        avg_credibility of 0.3 reduces confidence by ~20%
    credibility_factor = CREDIBILITY_CONFIDENCE_MIN + (avg_credibility * (CREDIBILITY_CONFIDENCE_MAX - CREDIBILITY_CONFIDENCE_MIN))
    confidence = min(1.0, base_confidence * credibility_factor)
    
    logger.info(f"{news_type.capitalize()} sentiment: {weighted_sentiment:.3f} "
                f"({positive_count}+ {negative_count}- {neutral_count}~) → {signal}")
    
    return {
        'news_type': news_type,
        'article_count': article_count,
        'average_sentiment': float(average_sentiment),
        'weighted_sentiment': float(weighted_sentiment),
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'sentiment_signal': signal,
        'confidence': float(confidence)
    }


# ============================================================================
# HELPER FUNCTION 6: Calculate Combined Sentiment
# ============================================================================

def calculate_combined_sentiment(
    stock_sentiment: Dict[str, Any],
    market_sentiment: Dict[str, Any],
    related_sentiment: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate combined weighted sentiment from three news types.
    
    Weighting (from sentiment_config.py):
    - Stock news: 50%
    - Market news: 25%
    - Related companies: 25%
    
    Algorithm:
    1. Extract weighted sentiment from each type
    2. Apply type weights (50/25/25)
    3. Calculate combined sentiment
    4. Calculate combined confidence
    5. Generate final signal
    
    Args:
        stock_sentiment: Aggregated stock news sentiment
        market_sentiment: Aggregated market news sentiment
        related_sentiment: Aggregated related companies sentiment
        
    Returns:
        {
            'combined_sentiment': float,          # -1.0 to +1.0
            'sentiment_signal': 'BUY'|'SELL'|'HOLD',
            'confidence': float,                  # 0.0 to 1.0
            'breakdown': {
                'stock': {...},
                'market': {...},
                'related': {...}
            }
        }
        
    Example:
        >>> stock = {'weighted_sentiment': 0.7, 'confidence': 0.8}
        >>> market = {'weighted_sentiment': 0.3, 'confidence': 0.6}
        >>> related = {'weighted_sentiment': 0.5, 'confidence': 0.7}
        >>> result = calculate_combined_sentiment(stock, market, related)
        >>> result['combined_sentiment']  # 0.5*0.7 + 0.25*0.3 + 0.25*0.5 = 0.55
        >>> result['sentiment_signal']    # 'BUY' (> 0.2)
    """
    # Extract weighted sentiments
    stock_sent = stock_sentiment.get('weighted_sentiment', 0.0)
    market_sent = market_sentiment.get('weighted_sentiment', 0.0)
    related_sent = related_sentiment.get('weighted_sentiment', 0.0)
    
    # Apply type weights (50/25/25)
    combined_sentiment = (
        stock_sent * STOCK_NEWS_WEIGHT +
        market_sent * MARKET_NEWS_WEIGHT +
        related_sent * RELATED_NEWS_WEIGHT
    )
    
    # Extract confidences
    stock_conf = stock_sentiment.get('confidence', 0.0)
    market_conf = market_sentiment.get('confidence', 0.0)
    related_conf = related_sentiment.get('confidence', 0.0)
    
    # Combined confidence (weighted average)
    combined_confidence = (
        stock_conf * STOCK_NEWS_WEIGHT +
        market_conf * MARKET_NEWS_WEIGHT +
        related_conf * RELATED_NEWS_WEIGHT
    )
    
    # Generate signal
    signal = generate_sentiment_signal(combined_sentiment, combined_confidence)
    
    logger.info(f"Combined sentiment: {combined_sentiment:.3f} → {signal} "
                f"(confidence: {combined_confidence:.2f})")
    
    return {
        'combined_sentiment': float(combined_sentiment),
        'sentiment_signal': signal,
        'confidence': float(combined_confidence),
        'breakdown': {
            'stock': stock_sentiment,
            'market': market_sentiment,
            'related': related_sentiment
        }
    }


# ============================================================================
# HELPER FUNCTION 7: Generate Sentiment Signal
# ============================================================================

def generate_sentiment_signal(combined_sentiment: float, confidence: float) -> str:
    """
    Generate final sentiment signal based on combined sentiment and confidence.
    
    Thresholds:
    - BUY: sentiment > 0.2
    - SELL: sentiment < -0.2
    - HOLD: -0.2 to 0.2 (or low confidence)
    
    Algorithm:
    1. Check confidence threshold
    2. If low confidence, default to HOLD
    3. Otherwise, use sentiment thresholds
    
    Args:
        combined_sentiment: Combined weighted sentiment (-1.0 to +1.0)
        confidence: Combined confidence (0.0 to 1.0)
        
    Returns:
        Signal: 'BUY', 'SELL', or 'HOLD'
        
    Example:
        >>> generate_sentiment_signal(0.5, 0.8)   # 'BUY'
        >>> generate_sentiment_signal(-0.5, 0.8)  # 'SELL'
        >>> generate_sentiment_signal(0.5, 0.3)   # 'HOLD' (low confidence)
    """
    # Low confidence → HOLD
    if confidence < 0.5:
        logger.info(f"Low confidence ({confidence:.2f}) → HOLD")
        return 'HOLD'
    
    # High confidence → use thresholds
    if combined_sentiment > BUY_THRESHOLD:
        return 'BUY'
    elif combined_sentiment < SELL_THRESHOLD:
        return 'SELL'
    else:
        return 'HOLD'


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def sentiment_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 5: Sentiment Analysis
    
    Execution flow:
    1. Get cleaned news from state (from Node 9A)
    2. Extract/analyze sentiment for all articles (Alpha Vantage + FinBERT)
    3. Aggregate sentiment by type (stock, market, related)
    4. Calculate combined weighted sentiment (50/25/25)
    5. Generate sentiment signal (BUY/SELL/HOLD)
    6. Return partial state update (for parallel execution)
    
    Runs in PARALLEL with: Nodes 4, 6, 7
    Runs AFTER: Node 9A (content analysis)
    Runs BEFORE: Node 8 (verification)
    
    Args:
        state: LangGraph state containing cleaned news
        
    Returns:
        Partial state update with sentiment analysis results
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    try:
        logger.info(f"Node 5: Starting sentiment analysis for {ticker}")
        
        # ====================================================================
        # STEP 1: Get Cleaned News from Node 9A
        # ====================================================================
        stock_news = state.get('cleaned_stock_news', [])
        market_news = state.get('cleaned_market_news', [])
        related_news = state.get('cleaned_related_company_news', [])
        
        logger.info(f"Processing: {len(stock_news)} stock, {len(market_news)} market, "
                   f"{len(related_news)} related articles")
        
        # ====================================================================
        # STEP 2: Analyze Sentiment for All Articles
        # ====================================================================
        logger.info("Extracting/analyzing sentiment...")
        
        stock_analyzed = analyze_articles_batch(stock_news, use_finbert=USE_FINBERT)
        market_analyzed = analyze_articles_batch(market_news, use_finbert=USE_FINBERT)
        related_analyzed = analyze_articles_batch(related_news, use_finbert=USE_FINBERT)
        
        # ====================================================================
        # STEP 3: Aggregate Sentiment by Type
        # ====================================================================
        logger.info("Aggregating sentiment by news type...")
        
        stock_sentiment = aggregate_sentiment_by_type(stock_analyzed, 'stock')
        market_sentiment = aggregate_sentiment_by_type(market_analyzed, 'market')
        related_sentiment = aggregate_sentiment_by_type(related_analyzed, 'related')
        
        # ====================================================================
        # STEP 4: Calculate Combined Sentiment
        # ====================================================================
        logger.info("Calculating combined weighted sentiment (50/25/25)...")
        
        combined_result = calculate_combined_sentiment(
            stock_sentiment,
            market_sentiment,
            related_sentiment
        )
        
        combined_sentiment = combined_result['combined_sentiment']
        sentiment_signal = combined_result['sentiment_signal']
        sentiment_confidence = combined_result['confidence']
        
        # ====================================================================
        # STEP 5: Collect Per-Article Scores (with credibility data)
        # ====================================================================
        raw_sentiment_scores = []
        
        for article in stock_analyzed:
            raw_sentiment_scores.append({
                'type': 'stock',
                'title': article.get('title', ''),
                'source': article.get('source', ''),
                'sentiment_score': article.get('sentiment_score', 0.0),
                'sentiment_label': article.get('sentiment_label', 'neutral'),
                'sentiment_source': article.get('sentiment_source', 'none'),
                # NEW: Credibility-related fields for Node 8 and Node 13
                'credibility_weight': calculate_credibility_weight(article),
                'source_credibility_score': article.get('source_credibility_score', 0.5),
                'composite_anomaly_score': article.get('composite_anomaly_score', 0.3),
                'relevance_score': article.get('relevance_score', 0.5)
            })
        
        for article in market_analyzed:
            raw_sentiment_scores.append({
                'type': 'market',
                'title': article.get('title', ''),
                'source': article.get('source', ''),
                'sentiment_score': article.get('sentiment_score', 0.0),
                'sentiment_label': article.get('sentiment_label', 'neutral'),
                'sentiment_source': article.get('sentiment_source', 'none'),
                # NEW: Credibility-related fields for Node 8 and Node 13
                'credibility_weight': calculate_credibility_weight(article),
                'source_credibility_score': article.get('source_credibility_score', 0.5),
                'composite_anomaly_score': article.get('composite_anomaly_score', 0.3),
                'relevance_score': article.get('relevance_score', 0.5)
            })
        
        for article in related_analyzed:
            raw_sentiment_scores.append({
                'type': 'related',
                'title': article.get('title', ''),
                'source': article.get('source', ''),
                'sentiment_score': article.get('sentiment_score', 0.0),
                'sentiment_label': article.get('sentiment_label', 'neutral'),
                'sentiment_source': article.get('sentiment_source', 'none'),
                # NEW: Credibility-related fields for Node 8 and Node 13
                'credibility_weight': calculate_credibility_weight(article),
                'source_credibility_score': article.get('source_credibility_score', 0.5),
                'composite_anomaly_score': article.get('composite_anomaly_score', 0.3),
                'relevance_score': article.get('relevance_score', 0.5)
            })
        
        # ====================================================================
        # STEP 6: Calculate Credibility Summary
        # ====================================================================
        all_articles = stock_analyzed + market_analyzed + related_analyzed
        
        if all_articles:
            # Calculate average source credibility
            avg_source_credibility = sum(
                a.get('source_credibility_score', 0.5) for a in all_articles
            ) / len(all_articles)
            
            # Calculate average composite anomaly
            avg_composite_anomaly = sum(
                a.get('composite_anomaly_score', 0.3) for a in all_articles
            ) / len(all_articles)
            
            # Count articles by credibility tier
            high_credibility_articles = sum(
                1 for a in all_articles if a.get('source_credibility_score', 0.5) >= 0.8
            )
            medium_credibility_articles = sum(
                1 for a in all_articles 
                if 0.5 <= a.get('source_credibility_score', 0.5) < 0.8
            )
            low_credibility_articles = sum(
                1 for a in all_articles if a.get('source_credibility_score', 0.5) < 0.5
            )
        else:
            avg_source_credibility = 0.5
            avg_composite_anomaly = 0.3
            high_credibility_articles = 0
            medium_credibility_articles = 0
            low_credibility_articles = 0
        
        credibility_summary = {
            'avg_source_credibility': float(avg_source_credibility),
            'high_credibility_articles': high_credibility_articles,      # credibility >= 0.8
            'medium_credibility_articles': medium_credibility_articles,  # 0.5 to 0.8
            'low_credibility_articles': low_credibility_articles,        # below 0.5
            'avg_composite_anomaly': float(avg_composite_anomaly),
            'credibility_weighted': True                                 # flag that weighting was applied
        }
        
        # ====================================================================
        # STEP 7: Log Results
        # ====================================================================
        logger.info(f"Sentiment Analysis Results:")
        logger.info(f"  Combined Sentiment: {combined_sentiment:.3f}")
        logger.info(f"  Signal: {sentiment_signal}")
        logger.info(f"  Confidence: {sentiment_confidence:.2f}")
        logger.info(f"  Stock: {stock_sentiment['weighted_sentiment']:.3f} "
                   f"({stock_sentiment['article_count']} articles)")
        logger.info(f"  Market: {market_sentiment['weighted_sentiment']:.3f} "
                   f"({market_sentiment['article_count']} articles)")
        logger.info(f"  Related: {related_sentiment['weighted_sentiment']:.3f} "
                   f"({related_sentiment['article_count']} articles)")
        logger.info(f"  Credibility: {avg_source_credibility:.2f} avg "
                   f"({high_credibility_articles}H/{medium_credibility_articles}M/{low_credibility_articles}L)")
        
        # ====================================================================
        # STEP 8: Return Partial State Update (for parallel execution)
        # ====================================================================
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Node 5: Sentiment analysis completed in {elapsed:.2f}s")
        
        # Return only the fields this node updates
        return {
            'raw_sentiment_scores': raw_sentiment_scores,
            'aggregated_sentiment': combined_sentiment,
            'sentiment_signal': sentiment_signal,
            'sentiment_confidence': sentiment_confidence,
            'sentiment_credibility_summary': credibility_summary,  # NEW
            'node_execution_times': {'node_5': elapsed}
        }
        
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Node 5: Sentiment analysis failed for {ticker}: {str(e)}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Return only the fields this node updates (for parallel execution)
        return {
            'errors': [f"Node 5: Sentiment analysis failed - {str(e)}"],
            'raw_sentiment_scores': [],
            'aggregated_sentiment': None,
            'sentiment_signal': None,
            'sentiment_confidence': None,
            'sentiment_credibility_summary': None,
            'node_execution_times': {'node_5': elapsed}
        }
