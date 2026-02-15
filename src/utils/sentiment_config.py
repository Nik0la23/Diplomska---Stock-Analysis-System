"""
Sentiment Analysis Configuration

Thresholds, weights, and model configuration for Node 5 (Sentiment Analysis).
"""

# ============================================================================
# SENTIMENT THRESHOLDS
# ============================================================================

# Signal generation thresholds
BUY_THRESHOLD = 0.2      # Sentiment > 0.2 → BUY signal
SELL_THRESHOLD = -0.2    # Sentiment < -0.2 → SELL signal
# Between -0.2 and 0.2 → HOLD signal

# Confidence thresholds
MIN_CONFIDENCE = 0.5     # Minimum confidence to generate non-HOLD signal
HIGH_CONFIDENCE = 0.8    # High confidence threshold


# ============================================================================
# SENTIMENT WEIGHTS (News Type Combination)
# ============================================================================

# Weighted combination of news types
STOCK_NEWS_WEIGHT = 0.50      # 50% - Stock-specific news (most important)
MARKET_NEWS_WEIGHT = 0.25     # 25% - Market-wide news
RELATED_NEWS_WEIGHT = 0.25    # 25% - Related companies news

# Ensure weights sum to 1.0
assert abs(STOCK_NEWS_WEIGHT + MARKET_NEWS_WEIGHT + RELATED_NEWS_WEIGHT - 1.0) < 0.001


# ============================================================================
# TIME DECAY FOR WEIGHTED SENTIMENT
# ============================================================================

# Recent articles matter more than older ones
ENABLE_TIME_DECAY = True
TIME_DECAY_FACTOR = 0.95  # Each day older reduces weight by 5%


# ============================================================================
# FINBERT CONFIGURATION
# ============================================================================

# FinBERT model from HuggingFace
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Enable FinBERT for articles without sentiment scores
USE_FINBERT = True  # Set to False to use Alpha Vantage sentiment only

# FinBERT device ('cpu' or 'cuda')
FINBERT_DEVICE = -1  # -1 for CPU, 0 for GPU

# Batch size for FinBERT processing
FINBERT_BATCH_SIZE = 8


# ============================================================================
# ALPHA VANTAGE SENTIMENT NORMALIZATION
# ============================================================================

# Alpha Vantage sentiment ranges
# overall_sentiment_score: typically -1.0 to +1.0 (but can vary)
# ticker_sentiment_score: typically -1.0 to +1.0
AV_MIN_SCORE = -1.0
AV_MAX_SCORE = 1.0

# Which Alpha Vantage score to use
USE_TICKER_SENTIMENT = True  # True = ticker_sentiment_score, False = overall_sentiment_score


# ============================================================================
# SENTIMENT CLASSIFICATION LABELS
# ============================================================================

LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
LABEL_NEUTRAL = "neutral"

# Label to normalized score mapping (for FinBERT)
LABEL_TO_SCORE = {
    LABEL_POSITIVE: 1.0,
    LABEL_NEGATIVE: -1.0,
    LABEL_NEUTRAL: 0.0
}


# ============================================================================
# CREDIBILITY WEIGHTING (Node 5 Upgrade)
# ============================================================================

# Credibility weight formula components (must sum to 1.0)
CREDIBILITY_SOURCE_WEIGHT = 0.50      # Source reputation (most important)
CREDIBILITY_ANOMALY_WEIGHT = 0.30     # Content quality (inverted anomaly)
CREDIBILITY_RELEVANCE_WEIGHT = 0.20   # Relevance to ticker

assert abs(CREDIBILITY_SOURCE_WEIGHT + CREDIBILITY_ANOMALY_WEIGHT + CREDIBILITY_RELEVANCE_WEIGHT - 1.0) < 0.001

# Credibility weight bounds
CREDIBILITY_WEIGHT_FLOOR = 0.1        # Minimum weight (never completely ignore)
CREDIBILITY_WEIGHT_CEILING = 1.0      # Maximum weight

# Confidence adjustment based on source credibility
CREDIBILITY_CONFIDENCE_MIN = 0.8      # Min multiplier for low credibility
CREDIBILITY_CONFIDENCE_MAX = 1.2      # Max multiplier for high credibility


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def normalize_sentiment_score(score: float) -> float:
    """
    Normalize sentiment score to -1.0 to +1.0 range.
    
    Args:
        score: Raw sentiment score
        
    Returns:
        Normalized score between -1.0 and +1.0
    """
    try:
        # Clamp to valid range
        return max(-1.0, min(1.0, float(score)))
    except (ValueError, TypeError):
        # Handle invalid types - return neutral
        return 0.0


def classify_sentiment(score: float) -> str:
    """
    Classify sentiment score into positive/negative/neutral.
    
    Args:
        score: Sentiment score (-1.0 to +1.0)
        
    Returns:
        Label: 'positive', 'negative', or 'neutral'
    """
    if score > 0.2:
        return LABEL_POSITIVE
    elif score < -0.2:
        return LABEL_NEGATIVE
    else:
        return LABEL_NEUTRAL
