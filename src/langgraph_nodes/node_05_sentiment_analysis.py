"""
Node 5: Sentiment Analysis

Analyzes sentiment from THREE cleaned news streams using:
1. Alpha Vantage built-in sentiment scores (primary)
2. FinBERT model for articles without sentiment (fallback)

CRITICAL: Uses cleaned_*_news from Node 9A, NOT raw news.

Dual-mode approach:
- Phase 1: Aggregate existing Alpha Vantage sentiment scores
- Phase 2: FinBERT for articles lacking AV sentiment (e.g. Finnhub articles)

Output philosophy:
- aggregated_sentiment (float) and sentiment_confidence are used by Node 8 and Node 12
- sentiment_breakdown is a rich narrative dict for Node 13 (LLM explainer):
  per-stream scores, dominant labels, top article headlines, credibility breakdown
- BUY/SELL/HOLD signal generation is NOT this node's job — that belongs to Node 12

Runs in PARALLEL with: Nodes 4, 6, 7
Runs AFTER: Node 9A (content analysis)
Runs BEFORE: Node 8 (verification)
"""

from datetime import datetime
from typing import Dict, List, Any
import logging
import numpy as np

# Sentiment configuration
from src.utils.sentiment_config import (
    STOCK_NEWS_WEIGHT,
    MARKET_NEWS_WEIGHT,
    RELATED_NEWS_WEIGHT,
    USE_FINBERT,
    FINBERT_MODEL_NAME,
    FINBERT_DEVICE,
    USE_TICKER_SENTIMENT,
    normalize_sentiment_score,
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

    Alpha Vantage provides per article:
    - overall_sentiment_score: Overall article sentiment (-1.0 to +1.0)
    - overall_sentiment_label: AV's own label (e.g. 'Bullish', 'Somewhat-Bullish',
      'Neutral', 'Somewhat-Bearish', 'Bearish')
    - ticker_sentiment_score: Ticker-specific score (stock articles only)
    - ticker_relevance_score: Ticker relevance (stock articles only)

    For AV articles the label is kept as-is from AV (not recalculated).
    For non-AV articles (e.g. Finnhub) with no score, FinBERT handles them
    later in analyze_articles_batch(); their label is set there.

    Algorithm:
    1. Prefer ticker_sentiment_score when available and USE_TICKER_SENTIMENT is True
    2. Fall back to overall_sentiment_score
    3. If neither present, mark has_sentiment=False (FinBERT fallback later)
    4. Preserve AV's own label for AV articles — do NOT recalculate

    Args:
        articles: List of news articles (may include Alpha Vantage and Finnhub)

    Returns:
        List of articles with 'sentiment_score', 'sentiment_label',
        'has_sentiment', and 'sentiment_source' fields added.

    Example:
        >>> articles = [
        ...     {'title': 'Good news', 'overall_sentiment_score': 0.5,
        ...      'overall_sentiment_label': 'Somewhat-Bullish', 'ticker_sentiment_score': 0.7},
        ...     {'title': 'Bad news', 'source': 'finnhub'}
        ... ]
        >>> result = extract_alpha_vantage_sentiment(articles)
        >>> result[0]['sentiment_score']   # 0.7 (ticker score)
        >>> result[0]['sentiment_label']   # 'Somewhat-Bullish' (kept from AV)
        >>> result[1]['has_sentiment']     # False (no AV sentiment)
    """
    processed_articles = []

    for article in articles:
        article_copy = article.copy()

        if USE_TICKER_SENTIMENT and 'ticker_sentiment_score' in article:
            # Prefer ticker-specific sentiment score
            raw_score = article.get('ticker_sentiment_score', 0.0)
            article_copy['sentiment_score'] = normalize_sentiment_score(raw_score)
            article_copy['has_sentiment'] = True
            article_copy['sentiment_source'] = 'alpha_vantage_ticker'
            # Use AV's overall label (no ticker-level label from AV — fall back to overall)
            article_copy['sentiment_label'] = article.get('overall_sentiment_label', 'Neutral')

        elif 'overall_sentiment_score' in article:
            # Fall back to overall sentiment score
            raw_score = article.get('overall_sentiment_score', 0.0)
            article_copy['sentiment_score'] = normalize_sentiment_score(raw_score)
            article_copy['has_sentiment'] = True
            article_copy['sentiment_source'] = 'alpha_vantage_overall'
            # Keep AV's label directly — no need to recalculate
            article_copy['sentiment_label'] = article.get('overall_sentiment_label', 'Neutral')

        else:
            # No AV sentiment (likely Finnhub) — FinBERT will fill this in later
            article_copy['sentiment_score'] = 0.0
            article_copy['has_sentiment'] = False
            article_copy['sentiment_source'] = 'none'
            article_copy['sentiment_label'] = 'Neutral'

        processed_articles.append(article_copy)

    with_sentiment = sum(1 for a in processed_articles if a['has_sentiment'])
    without_sentiment = len(processed_articles) - with_sentiment
    logger.info(f"Sentiment extraction: {with_sentiment} with AV sentiment, {without_sentiment} without")

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
    1. Calculate simple and credibility+time-weighted sentiment averages
    2. Count positive/negative/neutral labels
    3. Derive dominant_label (most frequent label)
    4. Select top_articles (top 3 by credibility_weight, for Node 13 narrative)
    5. Calculate confidence from consistency, article count, and avg credibility

    Note: No BUY/SELL/HOLD signal is generated here — that is Node 12's job.

    Args:
        articles:  List of articles of the same type (stock, market, or related)
        news_type: Label for logging ('stock', 'market', 'related')

    Returns:
        {
            'news_type': str,
            'article_count': int,
            'average_sentiment': float,    # Simple mean
            'weighted_sentiment': float,   # Time-decay + credibility weighted mean
            'positive_count': int,
            'negative_count': int,
            'neutral_count': int,
            'dominant_label': str,         # 'positive' | 'negative' | 'neutral'
            'top_articles': List[Dict],    # Top 3 by credibility_weight
            'confidence': float            # 0.0 to 1.0
        }

    Example:
        >>> articles = [
        ...     {'sentiment_score': 0.8, 'sentiment_label': 'Bullish',
        ...      'title': 'AAPL beats earnings', 'source': 'Bloomberg'},
        ...     {'sentiment_score': 0.6, 'sentiment_label': 'Somewhat-Bullish',
        ...      'title': 'Strong iPhone sales', 'source': 'Reuters'}
        ... ]
        >>> result = aggregate_sentiment_by_type(articles, 'stock')
        >>> result['average_sentiment']   # 0.7
        >>> result['dominant_label']      # 'positive' (both > 0)
        >>> result['top_articles']        # [{'title': ..., 'source': ..., ...}, ...]
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
            'dominant_label': 'neutral',
            'top_articles': [],
            'confidence': 0.0
        }

    sentiments = [a.get('sentiment_score', 0.0) for a in articles]

    average_sentiment = np.mean(sentiments) if sentiments else 0.0

    # Time-decay + credibility combined weights (articles assumed newest-first)
    combined_weights = []
    for i, article in enumerate(articles):
        time_weight = 0.95 ** i
        credibility_weight = calculate_credibility_weight(article)
        combined_weights.append(time_weight * credibility_weight)

    if combined_weights and sum(combined_weights) > 0:
        weighted_sentiment = np.average(sentiments, weights=combined_weights)
    else:
        weighted_sentiment = 0.0

    # Label counts — normalise AV labels to positive/negative/neutral for counting
    def _normalise_label(label: str) -> str:
        """Map any sentiment label variant to positive/negative/neutral."""
        label_lower = label.lower()
        if any(k in label_lower for k in ('bullish', 'positive')):
            return 'positive'
        if any(k in label_lower for k in ('bearish', 'negative')):
            return 'negative'
        return 'neutral'

    normalised_labels = [_normalise_label(a.get('sentiment_label', 'neutral')) for a in articles]
    positive_count = normalised_labels.count('positive')
    negative_count = normalised_labels.count('negative')
    neutral_count  = normalised_labels.count('neutral')

    # Dominant label: most frequent normalised label
    counts = {'positive': positive_count, 'negative': negative_count, 'neutral': neutral_count}
    dominant_label = max(counts, key=counts.get)

    # Top 3 articles by credibility weight (for Node 13 narrative)
    articles_with_cw = [
        (calculate_credibility_weight(a), a) for a in articles
    ]
    articles_with_cw.sort(key=lambda x: x[0], reverse=True)
    top_articles = [
        {
            'title': a.get('title') or a.get('headline', ''),
            'source': a.get('source', ''),
            'sentiment_score': float(a.get('sentiment_score', 0.0)),
            'sentiment_label': a.get('sentiment_label', 'Neutral'),
            'credibility_weight': round(cw, 3),
        }
        for cw, a in articles_with_cw[:3]
    ]

    # Confidence: consistency × count factor × credibility factor
    article_count = len(articles)
    consistency = max(positive_count, negative_count, neutral_count) / article_count
    count_factor = min(article_count / 10.0, 1.0)
    base_confidence = consistency * 0.7 + count_factor * 0.3

    avg_credibility = sum(
        a.get('source_credibility_score', 0.5) for a in articles
    ) / article_count

    credibility_factor = CREDIBILITY_CONFIDENCE_MIN + (
        avg_credibility * (CREDIBILITY_CONFIDENCE_MAX - CREDIBILITY_CONFIDENCE_MIN)
    )
    confidence = min(1.0, base_confidence * credibility_factor)

    logger.info(
        f"{news_type.capitalize()} sentiment: {weighted_sentiment:.3f} "
        f"({positive_count}+ {negative_count}- {neutral_count}~) dominant={dominant_label}"
    )

    return {
        'news_type': news_type,
        'article_count': article_count,
        'average_sentiment': float(average_sentiment),
        'weighted_sentiment': float(weighted_sentiment),
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'dominant_label': dominant_label,
        'top_articles': top_articles,
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
    Calculate combined weighted sentiment from three news streams.

    Weighting (from sentiment_config.py):
    - Stock news:   50%
    - Market news:  25%
    - Related news: 25%

    Note: BUY/SELL/HOLD signal generation is NOT done here — that is
    Node 12's responsibility. This function only aggregates the float
    scores and confidence values.

    Args:
        stock_sentiment:   Aggregated stock news sentiment dict
        market_sentiment:  Aggregated market news sentiment dict
        related_sentiment: Aggregated related company news sentiment dict

    Returns:
        {
            'combined_sentiment': float,   # -1.0 to +1.0
            'confidence': float,           # 0.0 to 1.0
        }

    Example:
        >>> stock = {'weighted_sentiment': 0.7, 'confidence': 0.8}
        >>> market = {'weighted_sentiment': 0.3, 'confidence': 0.6}
        >>> related = {'weighted_sentiment': 0.5, 'confidence': 0.7}
        >>> result = calculate_combined_sentiment(stock, market, related)
        >>> result['combined_sentiment']  # 0.5*0.7 + 0.25*0.3 + 0.25*0.5 = 0.55
    """
    stock_sent = stock_sentiment.get('weighted_sentiment', 0.0)
    market_sent = market_sentiment.get('weighted_sentiment', 0.0)
    related_sent = related_sentiment.get('weighted_sentiment', 0.0)

    combined_sentiment = (
        stock_sent * STOCK_NEWS_WEIGHT +
        market_sent * MARKET_NEWS_WEIGHT +
        related_sent * RELATED_NEWS_WEIGHT
    )

    stock_conf = stock_sentiment.get('confidence', 0.0)
    market_conf = market_sentiment.get('confidence', 0.0)
    related_conf = related_sentiment.get('confidence', 0.0)

    combined_confidence = (
        stock_conf * STOCK_NEWS_WEIGHT +
        market_conf * MARKET_NEWS_WEIGHT +
        related_conf * RELATED_NEWS_WEIGHT
    )

    logger.info(f"Combined sentiment: {combined_sentiment:.3f} (confidence: {combined_confidence:.2f})")

    return {
        'combined_sentiment': float(combined_sentiment),
        'confidence': float(combined_confidence),
    }


# ============================================================================
# HELPER FUNCTION 7: Build Sentiment Breakdown for Node 13
# ============================================================================

def build_sentiment_breakdown(
    stock_sentiment: Dict[str, Any],
    market_sentiment: Dict[str, Any],
    related_sentiment: Dict[str, Any],
    combined_sentiment: float,
    combined_confidence: float,
    all_articles: List[Dict],
) -> Dict[str, Any]:
    """
    Build a rich narrative dict for Node 13 (LLM explainer).

    Combines per-stream sentiment data with credibility stats and sentiment
    source mix so that Node 13 can tell a meaningful story rather than
    just repeating a number.

    Args:
        stock_sentiment:    Output of aggregate_sentiment_by_type for 'stock'
        market_sentiment:   Output of aggregate_sentiment_by_type for 'market'
        related_sentiment:  Output of aggregate_sentiment_by_type for 'related'
        combined_sentiment: Weighted average across all streams (-1.0 to +1.0)
        combined_confidence: Weighted confidence across all streams (0.0 to 1.0)
        all_articles:       Flat list of all analyzed articles (all three streams)

    Returns:
        {
            'stock':   { weighted_sentiment, article_count, positive_count,
                         negative_count, neutral_count, dominant_label,
                         top_articles },
            'market':  { ... same structure ... },
            'related': { ... same structure ... },
            'overall': {
                'combined_sentiment': float,
                'confidence': float,
                'credibility': {
                    'avg_source_credibility': float,
                    'high_credibility_articles': int,   # score >= 0.8
                    'medium_credibility_articles': int, # 0.5 to 0.8
                    'low_credibility_articles': int,    # < 0.5
                    'avg_composite_anomaly': float
                },
                'sentiment_source_mix': {
                    'alpha_vantage_ticker': int,
                    'alpha_vantage_overall': int,
                    'finbert': int,
                    'none': int
                }
            }
        }

    Example:
        >>> breakdown = build_sentiment_breakdown(stock, market, related,
        ...                                       0.43, 0.71, all_articles)
        >>> breakdown['stock']['dominant_label']     # 'positive'
        >>> breakdown['overall']['credibility']['high_credibility_articles']  # 6
    """
    # Per-stream data (drop internal fields not useful for narration)
    def _stream_dict(agg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'weighted_sentiment': agg.get('weighted_sentiment', 0.0),
            'article_count':      agg.get('article_count', 0),
            'positive_count':     agg.get('positive_count', 0),
            'negative_count':     agg.get('negative_count', 0),
            'neutral_count':      agg.get('neutral_count', 0),
            'dominant_label':     agg.get('dominant_label', 'neutral'),
            'top_articles':       agg.get('top_articles', []),
        }

    # Credibility stats across all articles
    if all_articles:
        avg_source_credibility = sum(
            a.get('source_credibility_score', 0.5) for a in all_articles
        ) / len(all_articles)

        avg_composite_anomaly = sum(
            a.get('composite_anomaly_score', 0.3) for a in all_articles
        ) / len(all_articles)

        high_cred   = sum(1 for a in all_articles if a.get('source_credibility_score', 0.5) >= 0.8)
        medium_cred = sum(1 for a in all_articles if 0.5 <= a.get('source_credibility_score', 0.5) < 0.8)
        low_cred    = sum(1 for a in all_articles if a.get('source_credibility_score', 0.5) < 0.5)
    else:
        avg_source_credibility = 0.5
        avg_composite_anomaly  = 0.3
        high_cred = medium_cred = low_cred = 0

    # How sentiment scores were obtained
    source_mix: Dict[str, int] = {
        'alpha_vantage_ticker':  0,
        'alpha_vantage_overall': 0,
        'finbert':               0,
        'none':                  0,
    }
    for a in all_articles:
        src = a.get('sentiment_source', 'none')
        if src in source_mix:
            source_mix[src] += 1
        else:
            source_mix['none'] += 1

    return {
        'stock':   _stream_dict(stock_sentiment),
        'market':  _stream_dict(market_sentiment),
        'related': _stream_dict(related_sentiment),
        'overall': {
            'combined_sentiment': float(combined_sentiment),
            'confidence':         float(combined_confidence),
            'credibility': {
                'avg_source_credibility':    round(float(avg_source_credibility), 3),
                'high_credibility_articles': high_cred,
                'medium_credibility_articles': medium_cred,
                'low_credibility_articles':  low_cred,
                'avg_composite_anomaly':     round(float(avg_composite_anomaly), 3),
            },
            'sentiment_source_mix': source_mix,
        },
    }


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def sentiment_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 5: Sentiment Analysis

    Execution flow:
    1. Get cleaned news from state (from Node 9A)
    2. Extract/analyse sentiment for each article (Alpha Vantage primary, FinBERT fallback)
    3. Aggregate sentiment by stream (stock, market, related)
    4. Calculate combined weighted sentiment (50 / 25 / 25)
    5. Build sentiment_breakdown for Node 13 (LLM narrative)
    6. Derive directional label (POSITIVE / NEGATIVE / NEUTRAL) for state
    7. Return partial state update (parallel execution safe)

    Runs in PARALLEL with: Nodes 4, 6, 7
    Runs AFTER:  Node 9A (content analysis — provides credibility scores)
    Runs BEFORE: Node 8  (verification — adjusts sentiment_confidence)

    Args:
        state: LangGraph state containing cleaned_*_news from Node 9A

    Returns:
        Partial state update:
            raw_sentiment_scores   — per-article list (for Node 8)
            aggregated_sentiment   — float -1.0..+1.0  (for Node 8 / Node 12)
            sentiment_signal       — 'POSITIVE'|'NEGATIVE'|'NEUTRAL' (for Node 8)
            sentiment_confidence   — float 0.0..1.0    (for Node 8 / Node 11)
            sentiment_breakdown    — rich dict for Node 13 narrative
    """
    start_time = datetime.now()
    ticker = state['ticker']

    try:
        logger.info(f"Node 5: Starting sentiment analysis for {ticker}")

        # ====================================================================
        # STEP 1: Get Cleaned News from Node 9A
        # ====================================================================
        stock_news   = state.get('cleaned_stock_news', [])
        market_news  = state.get('cleaned_market_news', [])
        related_news = state.get('cleaned_related_company_news', [])

        logger.info(
            f"Processing: {len(stock_news)} stock, {len(market_news)} market, "
            f"{len(related_news)} related articles"
        )

        # ====================================================================
        # STEP 2: Analyse Sentiment for All Articles
        # ====================================================================
        logger.info("Extracting/analysing sentiment (AV primary, FinBERT fallback)...")

        stock_analyzed   = analyze_articles_batch(stock_news,   use_finbert=USE_FINBERT)
        market_analyzed  = analyze_articles_batch(market_news,  use_finbert=USE_FINBERT)
        related_analyzed = analyze_articles_batch(related_news, use_finbert=USE_FINBERT)

        # ====================================================================
        # STEP 3: Aggregate Sentiment by Stream
        # ====================================================================
        logger.info("Aggregating sentiment by news stream...")

        stock_sentiment   = aggregate_sentiment_by_type(stock_analyzed,   'stock')
        market_sentiment  = aggregate_sentiment_by_type(market_analyzed,  'market')
        related_sentiment = aggregate_sentiment_by_type(related_analyzed, 'related')

        # ====================================================================
        # STEP 4: Calculate Combined Sentiment (50 / 25 / 25)
        # ====================================================================
        logger.info("Calculating combined weighted sentiment (50/25/25)...")

        combined_result    = calculate_combined_sentiment(stock_sentiment, market_sentiment, related_sentiment)
        combined_sentiment = combined_result['combined_sentiment']
        sentiment_confidence = combined_result['confidence']

        # ====================================================================
        # STEP 5: Build Narrative Breakdown for Node 13
        # ====================================================================
        all_articles = stock_analyzed + market_analyzed + related_analyzed

        sentiment_breakdown = build_sentiment_breakdown(
            stock_sentiment,
            market_sentiment,
            related_sentiment,
            combined_sentiment,
            sentiment_confidence,
            all_articles,
        )

        # ====================================================================
        # STEP 6: Derive Directional Label (for Node 8 compatibility)
        # ====================================================================
        if combined_sentiment > 0.0:
            sentiment_signal = 'POSITIVE'
        elif combined_sentiment < 0.0:
            sentiment_signal = 'NEGATIVE'
        else:
            sentiment_signal = 'NEUTRAL'

        # ====================================================================
        # STEP 7: Collect Per-Article Scores (for Node 8)
        # ====================================================================
        raw_sentiment_scores = []
        for news_type, analyzed in [('stock', stock_analyzed), ('market', market_analyzed), ('related', related_analyzed)]:
            for article in analyzed:
                raw_sentiment_scores.append({
                    'type':                   news_type,
                    'title':                  article.get('title') or article.get('headline', ''),
                    'source':                 article.get('source', ''),
                    'sentiment_score':        article.get('sentiment_score', 0.0),
                    'sentiment_label':        article.get('sentiment_label', 'Neutral'),
                    'sentiment_source':       article.get('sentiment_source', 'none'),
                    'credibility_weight':     calculate_credibility_weight(article),
                    'source_credibility_score': article.get('source_credibility_score', 0.5),
                    'composite_anomaly_score':  article.get('composite_anomaly_score', 0.3),
                    'relevance_score':          article.get('relevance_score', 0.5),
                })

        # ====================================================================
        # STEP 8: Log Summary
        # ====================================================================
        cred = sentiment_breakdown['overall']['credibility']
        logger.info("Sentiment Analysis Results:")
        logger.info(f"  Combined: {combined_sentiment:.3f} ({sentiment_signal}) confidence={sentiment_confidence:.2f}")
        logger.info(f"  Stock:   {stock_sentiment['weighted_sentiment']:.3f} dominant={stock_sentiment['dominant_label']} ({stock_sentiment['article_count']} articles)")
        logger.info(f"  Market:  {market_sentiment['weighted_sentiment']:.3f} dominant={market_sentiment['dominant_label']} ({market_sentiment['article_count']} articles)")
        logger.info(f"  Related: {related_sentiment['weighted_sentiment']:.3f} dominant={related_sentiment['dominant_label']} ({related_sentiment['article_count']} articles)")
        logger.info(f"  Credibility: avg={cred['avg_source_credibility']:.2f} H={cred['high_credibility_articles']} M={cred['medium_credibility_articles']} L={cred['low_credibility_articles']}")

        # ====================================================================
        # STEP 9: Return Partial State Update
        # ====================================================================
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Node 5: Completed in {elapsed:.2f}s")

        return {
            'raw_sentiment_scores': raw_sentiment_scores,
            'aggregated_sentiment': combined_sentiment,
            'sentiment_signal':     sentiment_signal,
            'sentiment_confidence': sentiment_confidence,
            'sentiment_breakdown':  sentiment_breakdown,
            'node_execution_times': {'node_5': elapsed},
        }

    except Exception as e:
        logger.error(f"Node 5: Sentiment analysis failed for {ticker}: {str(e)}")
        elapsed = (datetime.now() - start_time).total_seconds()
        return {
            'errors':               [f"Node 5: Sentiment analysis failed - {str(e)}"],
            'raw_sentiment_scores': [],
            'aggregated_sentiment': None,
            'sentiment_signal':     None,
            'sentiment_confidence': None,
            'sentiment_breakdown':  None,
            'node_execution_times': {'node_5': elapsed},
        }
