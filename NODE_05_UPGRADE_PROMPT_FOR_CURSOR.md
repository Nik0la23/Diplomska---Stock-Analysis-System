# UPGRADE Node 5: Add Credibility-Weighted Sentiment

## What This Upgrade Does

Node 5 currently treats all articles equally when calculating sentiment. A Bloomberg article and a random blog have the same influence on the final sentiment score. This upgrade makes Node 5 use the credibility and anomaly scores that Node 9A already attached to every article, so that trustworthy sources have more influence than low-quality sources.

## IMPORTANT: What NOT To Change

- Do NOT remove or rewrite existing functions that work
- Do NOT change the FinBERT logic or Alpha Vantage sentiment extraction
- Do NOT change the signal generation thresholds (BUY > 0.2, SELL < -0.2)
- Do NOT change the news type weights (50% stock, 25% market, 25% related)
- Do NOT remove the time decay logic
- Do NOT change the state output field names
- Do NOT change how the node integrates with the parallel execution in workflow.py
- Keep all 32 existing tests passing

## What Node 9A Already Attaches to Every Article

Node 9A runs BEFORE Node 5 and adds these fields to every article dictionary:

```python
article = {
    # Original fields from Node 2 (Alpha Vantage)
    'title': 'NVDA beats earnings expectations',
    'source': 'Bloomberg',              # flat string
    'url': 'https://bloomberg.com/...',
    'summary': '...',
    'ticker_sentiment_score': 0.45,     # Alpha Vantage pre-computed
    'ticker_sentiment_label': 'Somewhat-Bullish',
    'relevance_score': 0.85,
    'overall_sentiment_score': 0.38,
    'time_published': '20260214T103000',
    
    # THESE ARE ADDED BY NODE 9A (already present when Node 5 receives them):
    'sensationalism_score': 0.0,        # 0-1, lower = better
    'urgency_score': 0.0,              # 0-1, lower = better
    'unverified_claims_score': 0.1,    # 0-1, lower = better
    'source_credibility_score': 0.95,  # 0-1, HIGHER = more trustworthy
    'complexity_score': 0.3,           # 0-1, informational only
    'composite_anomaly_score': 0.045,  # 0-1, lower = cleaner article
    'content_tags': {
        'keywords': ['earnings', 'revenue'],
        'topic': 'earnings_report',
        'temporal': 'past',
        'entities': ['NVDA']
    }
}
```

## The Upgrade: Credibility Weight Calculation

Add a new helper function that calculates a single `credibility_weight` (0-1) for each article. This weight is then used in the existing aggregation function.

### New Helper Function: `calculate_credibility_weight(article)`

```python
def calculate_credibility_weight(article: Dict) -> float:
    """
    Calculate how much this article should influence the sentiment score.
    
    Uses Node 9A's scores:
    - source_credibility_score (most important, 50% of weight)
    - composite_anomaly_score (inverted - high anomaly = low weight, 30%)
    - relevance_score from Alpha Vantage (20%)
    
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
        credibility * 0.50 +       # Source reputation matters most
        anomaly_quality * 0.30 +   # Content quality
        relevance * 0.20           # Relevance to ticker
    )
    
    # Floor at 0.1 (never completely ignore — Node 9B needs to see patterns)
    # Cap at 1.0
    return max(0.1, min(1.0, weight))
```

### Where To Apply The Weight

Modify the `aggregate_sentiment_by_type` function. Currently it does a simple average (or time-decayed average) of all articles' sentiment scores. The upgrade multiplies each article's contribution by its `credibility_weight`.

**Current logic (simplified):**
```python
# Current: all articles weighted equally (only time decay)
for i, article in enumerate(articles):
    score = article['sentiment_score']
    time_weight = 0.95 ** i
    total += score * time_weight
    weight_sum += time_weight

average = total / weight_sum
```

**Upgraded logic:**
```python
# Upgraded: articles weighted by credibility AND time decay
for i, article in enumerate(articles):
    score = article['sentiment_score']
    time_weight = 0.95 ** i
    credibility_weight = calculate_credibility_weight(article)
    combined_weight = time_weight * credibility_weight
    
    total += score * combined_weight
    weight_sum += combined_weight

average = total / weight_sum if weight_sum > 0 else 0.0
```

This means:
- Bloomberg (credibility 0.95, anomaly 0.04) gets combined_weight ≈ 0.93
- Random blog (credibility 0.30, anomaly 0.75) gets combined_weight ≈ 0.22
- Bloomberg has ~4x more influence than the random blog

### Confidence Calculation Update

The confidence calculation should also factor in the average credibility of sources. If most articles are from high-credibility sources, confidence should be higher.

Add this to the confidence calculation in `aggregate_sentiment_by_type`:

```python
# Average credibility of sources used
avg_credibility = sum(
    article.get('source_credibility_score', 0.5) for article in articles
) / len(articles) if articles else 0.5

# Adjust confidence: high-credibility sources = higher confidence
# Scale: avg_credibility of 0.9 boosts confidence by ~10%
#         avg_credibility of 0.3 reduces confidence by ~20%
credibility_factor = 0.8 + (avg_credibility * 0.4)  # Range: 0.8 to 1.2
adjusted_confidence = min(1.0, base_confidence * credibility_factor)
```

### Per-Article Output Enhancement

In the `raw_sentiment_scores` output, include the credibility weight so Node 8 and Node 13 can see why certain articles had more influence:

```python
# For each article in raw_sentiment_scores, add:
{
    'title': article['title'],
    'source': article['source'],
    'sentiment_score': 0.45,
    'sentiment_label': 'positive',
    'type': 'stock',                          # existing field
    'credibility_weight': 0.93,               # NEW
    'source_credibility_score': 0.95,         # NEW (pass through from 9A)
    'composite_anomaly_score': 0.045,         # NEW (pass through from 9A)
    'relevance_score': 0.85,                  # NEW (pass through from AV)
}
```

### State Output Addition

Add a new field to the sentiment_analysis output so Node 8 knows the credibility breakdown:

```python
state['sentiment_analysis'] = {
    # All existing fields stay the same:
    'sentiment_signal': 'BUY',
    'combined_sentiment_score': 0.35,
    'confidence': 0.72,
    'stock_sentiment': {...},
    'market_sentiment': {...},
    'related_sentiment': {...},
    'raw_sentiment_scores': [...],
    
    # NEW field for Node 8:
    'credibility_summary': {
        'avg_source_credibility': 0.78,
        'high_credibility_articles': 8,      # credibility >= 0.8
        'medium_credibility_articles': 5,    # 0.5 to 0.8
        'low_credibility_articles': 2,       # below 0.5
        'avg_composite_anomaly': 0.12,
        'credibility_weighted': True          # flag that weighting was applied
    }
}
```

## Practical Example: Before vs After

### Input: 5 articles about NVDA

```
Article 1: Bloomberg — "NVDA beats earnings" — sentiment +0.8, credibility 0.95, anomaly 0.04
Article 2: Reuters — "NVDA revenue up 30%" — sentiment +0.6, credibility 0.90, anomaly 0.05
Article 3: Random blog — "NVDA TO THE MOON!!!" — sentiment +0.9, credibility 0.30, anomaly 0.75
Article 4: Seeking Alpha — "NVDA outlook mixed" — sentiment -0.1, credibility 0.55, anomaly 0.20
Article 5: CNBC — "NVDA AI demand strong" — sentiment +0.5, credibility 0.80, anomaly 0.08
```

### BEFORE upgrade (current — equal weighting):
```
Simple average = (0.8 + 0.6 + 0.9 + (-0.1) + 0.5) / 5 = 0.54
The random blog's +0.9 "TO THE MOON" inflates the score
```

### AFTER upgrade (credibility-weighted):
```
Credibility weights:
  Bloomberg:      0.95×0.50 + (1-0.04)×0.30 + 0.85×0.20 = 0.475 + 0.288 + 0.170 = 0.933
  Reuters:        0.90×0.50 + (1-0.05)×0.30 + 0.80×0.20 = 0.450 + 0.285 + 0.160 = 0.895
  Random blog:    0.30×0.50 + (1-0.75)×0.30 + 0.40×0.20 = 0.150 + 0.075 + 0.080 = 0.305
  Seeking Alpha:  0.55×0.50 + (1-0.20)×0.30 + 0.60×0.20 = 0.275 + 0.240 + 0.120 = 0.635
  CNBC:           0.80×0.50 + (1-0.08)×0.30 + 0.75×0.20 = 0.400 + 0.276 + 0.150 = 0.826

Weighted calculation:
  (0.8×0.933) + (0.6×0.895) + (0.9×0.305) + (-0.1×0.635) + (0.5×0.826)
  = 0.746 + 0.537 + 0.275 + (-0.064) + 0.413
  = 1.907

  ÷ (0.933 + 0.895 + 0.305 + 0.635 + 0.826)
  = 1.907 / 3.594
  = 0.531

Result: 0.531 vs 0.54 (small difference here because most sources agree)
BUT: the random blog's influence dropped from 20% (1/5) to 8.5% (0.305/3.594)
```

### Where The Difference Is Huge:

If 4 random blogs say +0.9 and 1 Bloomberg says -0.3:

**Before:** (0.9+0.9+0.9+0.9+(-0.3))/5 = +0.66 → BUY signal (misleading!)

**After:** blogs get ~0.3 weight each, Bloomberg gets ~0.93 weight:
(0.9×0.3×4 + (-0.3)×0.93) / (0.3×4 + 0.93) = (1.08 - 0.279) / 2.13 = +0.376 → HOLD signal (much safer)

## New Tests To Add

Add these tests to the EXISTING test file (don't replace existing tests):

```python
# Test: Credibility weight calculation
def test_credibility_weight_high_quality():
    """Bloomberg article should get weight near 1.0"""
    article = {
        'source_credibility_score': 0.95,
        'composite_anomaly_score': 0.04,
        'relevance_score': 0.85
    }
    weight = calculate_credibility_weight(article)
    assert weight > 0.85

def test_credibility_weight_low_quality():
    """Random blog should get weight near 0.2-0.3"""
    article = {
        'source_credibility_score': 0.30,
        'composite_anomaly_score': 0.75,
        'relevance_score': 0.40
    }
    weight = calculate_credibility_weight(article)
    assert weight < 0.40

def test_credibility_weight_missing_scores():
    """Articles without 9A scores should get moderate defaults"""
    article = {}  # No scores at all
    weight = calculate_credibility_weight(article)
    assert 0.4 < weight < 0.7  # Moderate default

def test_credibility_weight_floor():
    """Even worst articles should have minimum 0.1 weight"""
    article = {
        'source_credibility_score': 0.0,
        'composite_anomaly_score': 1.0,
        'relevance_score': 0.0
    }
    weight = calculate_credibility_weight(article)
    assert weight >= 0.1

def test_weighted_aggregation_reduces_blog_influence():
    """Random blog sentiment should not dominate over Bloomberg"""
    articles = [
        {'sentiment_score': 0.9, 'source_credibility_score': 0.30,
         'composite_anomaly_score': 0.75, 'relevance_score': 0.4,
         'time_published': '20260214T120000'},
        {'sentiment_score': -0.3, 'source_credibility_score': 0.95,
         'composite_anomaly_score': 0.04, 'relevance_score': 0.85,
         'time_published': '20260214T110000'},
    ]
    result = aggregate_sentiment_by_type(articles, 'stock')
    # Bloomberg's -0.3 should pull the average below 0.3
    # Without weighting it would be (0.9 + -0.3)/2 = 0.30
    assert result['average_sentiment'] < 0.15

def test_credibility_summary_in_output():
    """State output should include credibility summary"""
    state = build_test_state_with_scored_articles()
    result = sentiment_analysis_node(state)
    assert 'credibility_summary' in result['sentiment_analysis']
    assert 'avg_source_credibility' in result['sentiment_analysis']['credibility_summary']
    assert result['sentiment_analysis']['credibility_summary']['credibility_weighted'] == True

def test_per_article_scores_include_credibility():
    """raw_sentiment_scores should include credibility_weight per article"""
    state = build_test_state_with_scored_articles()
    result = sentiment_analysis_node(state)
    for score in result['sentiment_analysis']['raw_sentiment_scores']:
        assert 'credibility_weight' in score

def test_backward_compatibility_without_9a_scores():
    """Node 5 should still work if 9A scores are missing (graceful defaults)"""
    articles = [
        {'title': 'Test', 'summary': 'Test article', 'sentiment_score': 0.5}
        # No 9A scores at all
    ]
    state = {'cleaned_stock_news': articles, 'cleaned_market_news': [],
             'cleaned_related_company_news': [], 'ticker': 'TEST',
             'errors': [], 'node_execution_times': {}}
    result = sentiment_analysis_node(state)
    assert result['sentiment_analysis'] is not None
    # Should work with default weights
```

## Summary of Changes

1. **Add** `calculate_credibility_weight()` function — new helper
2. **Modify** `aggregate_sentiment_by_type()` — multiply each article's contribution by credibility_weight alongside existing time decay
3. **Modify** confidence calculation — factor in average source credibility
4. **Modify** per-article output in `raw_sentiment_scores` — include credibility_weight, source_credibility_score, composite_anomaly_score, relevance_score
5. **Add** `credibility_summary` to the state output
6. **Add** 8 new tests (keep all 32 existing tests passing)
7. **Update** `sentiment_config.py` if needed — add credibility weight config values

## Files to Modify
- `src/langgraph_nodes/node_05_sentiment_analysis.py` — main changes
- `src/utils/sentiment_config.py` — add credibility weight defaults if needed
- `tests/test_nodes/test_node_05.py` — add new tests

## Files NOT to Modify
- `src/langgraph_nodes/node_09a_content_analysis.py` — no changes
- `src/graph/workflow.py` — no changes (same state fields, just enriched)
- Any other node files
