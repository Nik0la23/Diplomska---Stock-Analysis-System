# Node 9B: Behavioral Anomaly Detection & Fraud Pattern Recognition
## Complete Build Roadmap for Cursor

**File:** `src/langgraph_nodes/node_09b_behavioral_anomaly.py`  
**Estimated Time:** 8-10 hours  
**Runs AFTER:** Node 8 (News Verification & Learning System)  
**Runs BEFORE:** Node 10 (Backtesting)  
**Parallel with:** Nothing (sequential)  

---

## What This Node Does — In Plain English

Node 9B is the **behavioral detective**. While Node 9A looked at article content (clickbait, sensationalism, source credibility), Node 9B looks at **market behavior patterns** and asks: "Does the way the market is moving match what the news says? Does this look like past manipulation schemes?"

**The Core Question:** "Given everything we know — article volume, source reliability, price movement, volume spikes, sentiment direction — have we seen this exact pattern before, and was it manipulation?"

Node 9B has **7 detection systems** that each look for different behavioral anomalies:

1. **Volume Anomaly Detector** — Is trading volume unusually high?
2. **Source Reliability Divergence Detector** — Are today's sources less reliable than usual for this stock?
3. **News Velocity Anomaly Detector** — Are articles appearing faster than normal?
4. **News-Price Divergence Detector** — Does price movement match what credible sources predicted?
5. **Cross-Stream Coherence Detector** — Do stock/market/related news streams align or conflict?
6. **Historical Pattern Matcher** — Have we seen this behavioral fingerprint before, and what happened?
7. **Pump-and-Dump Composite Scorer** — Combines all signals into a final risk score

**Output:** Risk level (LOW/MEDIUM/HIGH/CRITICAL), pump-and-dump score (0-100), and specific alerts for the user.

---

## What Node 9B Does NOT Do

- Does NOT fetch any data from APIs
- Does NOT recalculate anything from previous nodes
- Does NOT modify any previous analysis results
- Does NOT filter or remove articles (Node 9A did that)
- Does NOT call any LLM
- Does NOT make buy/sell recommendations (that's Node 12's job)

Everything Node 9B needs is already in state or in the database. It only READS and produces NEW behavioral risk assessment.

---

## Critical Architectural Position

Node 9B sits at a critical junction in the workflow:

```
Node 8 (Learning) → NODE 9B (Behavioral Detection) → Node 10 (Backtesting) → 
Node 11 (Weights) → Conditional Edge (reads 9B output) → Node 12 or HALT
```

**Why this position matters:**

1. **After Node 8** — Needs source reliability data and learning adjustment from Node 8
2. **Before Node 10** — Node 10's backtest shouldn't include manipulated data patterns
3. **Before Conditional Edge** — The risk router reads 9B's output to decide halt vs continue
4. **Before Node 12** — Node 12 needs 9B's risk assessment for the final signal

If Node 9B detects CRITICAL risk, the conditional edge skips Node 12 entirely and routes straight to Node 13 for a warning-only output.

---

## What Node 9B Receives From State

When Node 9B runs, all previous nodes have completed. Here's what's available:

```python
# From Node 1 (Price Data)
state['raw_price_data'] = pd.DataFrame({
    'date': [...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})
state['current_price'] = 143.50

# From Node 4 (Technical Analysis)
state['technical_analysis'] = {
    'technical_signal': 'BUY' | 'SELL' | 'HOLD',
    'confidence': 62.0,
    'rsi': 58.4,
    'macd_signal': 'bullish_crossover' | 'bearish_crossover' | 'neutral',
    'volume_ratio': 1.2,  # vs 20-day average
    # ... other indicators
}

# From Node 5 (Sentiment Analysis) — already adjusted by Node 8
state['sentiment_analysis'] = {
    'sentiment_signal': 'BUY' | 'SELL' | 'HOLD',
    'confidence': 71.0,  # already adjusted by Node 8
    'combined_sentiment_score': 0.31,  # -1 to +1
    'sentiment_label': 'positive' | 'negative' | 'neutral',
    'stock_news_sentiment': 0.45,
    'market_news_sentiment': 0.22,
    'related_news_sentiment': 0.18,
    'article_count': 10,
    'raw_sentiment_scores': [  # per-article breakdown
        {
            'title': 'NVDA beats earnings',
            'source': 'Bloomberg',
            'sentiment_score': 0.45,
            'sentiment_label': 'positive',
            'type': 'stock',
            'credibility_weight': 0.93,
            'source_credibility_score': 0.95,  # from Node 9A
            'composite_anomaly_score': 0.045,  # from Node 9A
        },
        # ... more articles
    ]
}

# From Node 6 (Market Context)
state['market_context'] = {
    'context_signal': 'BUY' | 'SELL' | 'HOLD',
    'confidence': 65.0,
    'sector_performance': 1.8,  # % change
    'market_trend': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
    'spy_5day_change': 1.5,
    'correlation_with_market': 0.82,
    'related_companies_performance': {
        'AMD': 2.1,
        'INTC': 0.8,
        'TSM': 1.9
    }
}

# From Node 8 (News Verification & Learning) — CRITICAL INPUT
state['news_impact_verification'] = {
    'historical_correlation': 0.72,
    'news_accuracy_score': 68.0,  # weighted avg across sources
    'verified_signal_strength': 71.0,
    'learning_adjustment': 1.2,  # 0.5-2.0
    'source_reliability': {
        'Bloomberg': {
            'accuracy_rate': 0.844,
            'total_articles': 45,
            'confidence_multiplier': 1.088,
            'avg_price_impact': 2.3
        },
        'Reuters': {
            'accuracy_rate': 0.789,
            'total_articles': 38,
            'confidence_multiplier': 1.0,
            'avg_price_impact': 1.9
        },
        'random-blog.com': {
            'accuracy_rate': 0.286,
            'total_articles': 28,
            'confidence_multiplier': 0.786,
            'avg_price_impact': 0.4
        }
        # ... more sources
    },
    'news_type_effectiveness': {
        'stock': {'accuracy_rate': 0.68, 'avg_impact': 3.2, 'sample_size': 45},
        'market': {'accuracy_rate': 0.52, 'avg_impact': 1.1, 'sample_size': 30},
        'related': {'accuracy_rate': 0.61, 'avg_impact': 1.8, 'sample_size': 25}
    },
    'sample_size': 350,
    'insufficient_data': False
}

# From Node 9A (Content Analysis) — CRITICAL INPUT
state['content_analysis_summary'] = {
    'total_articles_processed': 150,
    'articles_by_type': {
        'stock': 50,
        'market': 100,
        'related': 0
    },
    'average_scores': {
        'sensationalism': 0.021,
        'urgency': 0.002,
        'unverified_claims': 0.130,
        'source_credibility': 0.631,
        'composite_anomaly': 0.149
    },
    'high_risk_articles': 0,  # composite > 0.7
    'high_risk_percentage': 0.0,
    'top_keywords': [
        ('earnings', 25),
        ('revenue', 19),
        ('profit', 14)
    ],
    'source_credibility_distribution': {
        'high': 19,    # 0.8-1.0
        'medium': 84,  # 0.5-0.8
        'low': 47      # 0.0-0.5
    }
}

# From Node 9A — Cleaned news with embedded scores
state['cleaned_stock_news'] = [
    {
        'headline': '...',
        'summary': '...',
        'source': 'Bloomberg',
        'published_at': '2026-02-14T10:30:00',
        'url': '...',
        # Scores from Node 9A
        'sensationalism_score': 0.02,
        'urgency_score': 0.01,
        'unverified_claims_score': 0.05,
        'source_credibility_score': 0.95,
        'composite_anomaly_score': 0.045,
        'content_tags': {
            'keywords': ['earnings', 'revenue'],
            'topic': 'earnings_report',
            'temporal': 'past'
        }
    },
    # ... more articles
]
state['cleaned_market_news'] = [...]
state['cleaned_related_company_news'] = [...]

# Basic state fields
state['ticker'] = 'NVDA'
state['analysis_date'] = '2026-02-14'
state['errors'] = []
state['node_execution_times'] = {}
```

---

## Database Functions Needed

Node 9B needs to query historical data for pattern matching. Add these to `src/database/db_manager.py`:

### 1. `get_historical_daily_aggregates(ticker, days=180)`

Returns daily aggregates for pattern matching:

```python
def get_historical_daily_aggregates(ticker: str, days: int = 180) -> List[Dict]:
    """
    Get daily aggregated metrics for historical pattern matching.
    
    Returns list of dicts, one per day:
    [
        {
            'date': '2025-08-15',
            'article_count': 8,
            'avg_composite_anomaly': 0.12,
            'avg_source_credibility': 0.68,
            'volume': 45000000,
            'volume_ratio': 1.1,  # vs 20-day avg
            'price_change_1d': 0.8,
            'price_change_7d': 2.1,
            'sentiment_avg': 0.28,
            'sentiment_label': 'positive'
        },
        # ... 180 entries
    ]
    """
```

**Implementation:** Join `news_articles`, `news_outcomes`, and `price_data` tables, group by date.

### 2. `get_volume_baseline(ticker, days=30)`

Returns average daily volume for the past N days:

```python
def get_volume_baseline(ticker: str, days: int = 30) -> float:
    """
    Calculate average daily volume over the past N days.
    Used as baseline for volume anomaly detection.
    """
```

### 3. `get_article_count_baseline(ticker, days=180)`

Returns average daily article count:

```python
def get_article_count_baseline(ticker: str, days: int = 180) -> float:
    """
    Calculate average number of articles per day over past N days.
    Used as baseline for news velocity detection.
    """
```

If these functions don't exist, ADD them to `db_manager.py` before building Node 9B.

---

## Implementation Plan — The 7 Detection Systems

### Detection System 1: Volume Anomaly Detector

**Purpose:** Detect abnormal trading volume that could indicate coordinated activity.

**Function Signature:**
```python
def detect_volume_anomaly(
    current_volume: float,
    historical_volumes: List[float],
    baseline_days: int = 30
) -> Dict[str, Any]:
    """
    Detects volume anomalies by comparing current volume to historical baseline.
    
    Args:
        current_volume: Today's trading volume
        historical_volumes: List of daily volumes (most recent N days)
        baseline_days: How many days to use for baseline calculation
    
    Returns:
        {
            'detected': bool,
            'volume_ratio': float,  # current / baseline
            'severity': 'LOW' | 'MEDIUM' | 'HIGH',
            'contribution_score': int,  # 0-20
            'baseline_volume': float,
            'current_volume': float
        }
    """
```

**Algorithm:**
```
1. Calculate baseline = mean(historical_volumes[-baseline_days:])
2. Calculate volume_ratio = current_volume / baseline
3. Determine severity:
   - ratio < 2.0: LOW (0 points)
   - 2.0 <= ratio < 4.0: MEDIUM (10 points)
   - 4.0 <= ratio < 6.0: HIGH (15 points)
   - ratio >= 6.0: HIGH (20 points)
4. Set detected = True if ratio >= 2.0
```

**Correlation with 9A:**
If volume_ratio is high AND average composite_anomaly_score from 9A is high (>0.6), increase contribution score by 50%.

---

### Detection System 2: Source Reliability Divergence Detector

**Purpose:** Detect when today's sources are significantly less reliable than historical baseline for this stock.

**Function Signature:**
```python
def detect_source_reliability_divergence(
    todays_articles: List[Dict],
    source_reliability_dict: Dict[str, Dict],
    historical_avg_reliability: float
) -> Dict[str, Any]:
    """
    Detects when today's news sources are less reliable than usual.
    
    Args:
        todays_articles: List of articles from cleaned_stock_news
        source_reliability_dict: From Node 8's news_impact_verification
        historical_avg_reliability: Baseline from Node 8
    
    Returns:
        {
            'detected': bool,
            'today_avg_accuracy': float,
            'historical_avg_accuracy': float,
            'divergence': float,  # difference
            'severity': 'LOW' | 'MEDIUM' | 'HIGH',
            'contribution_score': int,  # 0-20
            'unreliable_source_count': int,
            'total_article_count': int
        }
    """
```

**Algorithm:**
```
1. For each article in todays_articles:
   - Look up article['source'] in source_reliability_dict
   - If found: get accuracy_rate
   - If not found: use default 0.50
   - Weight by article count (if source appears 5 times, count 5x)

2. Calculate weighted average accuracy across all today's articles:
   today_avg_accuracy = sum(accuracy × count) / total_articles

3. Get historical_avg_accuracy from Node 8's news_accuracy_score / 100

4. Calculate divergence = historical_avg_accuracy - today_avg_accuracy

5. Determine severity:
   - divergence < 0.10: LOW (0 points)
   - 0.10 <= divergence < 0.20: MEDIUM (12 points)
   - 0.20 <= divergence < 0.30: HIGH (18 points)
   - divergence >= 0.30: HIGH (20 points)

6. Set detected = True if divergence >= 0.15
```

**Key Insight:** A source being unreliable isn't itself suspicious (Node 8 handles that). But TODAY specifically being dominated by unreliable sources MORE than usual is suspicious.

---

### Detection System 3: News Velocity Anomaly Detector

**Purpose:** Detect abnormal spikes in article volume that could indicate coordinated posting.

**Function Signature:**
```python
def detect_news_velocity_anomaly(
    today_article_count: int,
    historical_daily_avg: float,
    content_analysis_summary: Dict
) -> Dict[str, Any]:
    """
    Detects abnormal spikes in news article volume.
    
    Args:
        today_article_count: Total articles today (all 3 streams)
        historical_daily_avg: Average articles per day (from database)
        content_analysis_summary: From Node 9A
    
    Returns:
        {
            'detected': bool,
            'today_article_count': int,
            'historical_daily_avg': float,
            'velocity_ratio': float,  # today / baseline
            'severity': 'LOW' | 'MEDIUM' | 'HIGH',
            'contribution_score': int,  # 0-15
            'high_anomaly_article_count': int,  # articles with composite > 0.6
            'coordinated_keywords': bool  # same keywords across many articles
        }
    """
```

**Algorithm:**
```
1. Calculate velocity_ratio = today_article_count / historical_daily_avg

2. Check content_analysis_summary for:
   - high_risk_article_count (composite > 0.7)
   - top_keywords concentration

3. Check for coordination:
   - If top keyword appears in > 60% of articles: coordinated = True
   - If multiple sensational keywords repeat: coordinated = True

4. Determine severity:
   - ratio < 3.0: LOW (0 points)
   - 3.0 <= ratio < 5.0: MEDIUM (8 points)
   - 5.0 <= ratio < 8.0: HIGH (12 points)
   - ratio >= 8.0: HIGH (15 points)

5. If coordinated keywords detected: add 3 bonus points

6. Set detected = True if ratio >= 3.0
```

---

### Detection System 4: News-Price Divergence Detector

**Purpose:** Detect when price movement doesn't match what credible sources predicted (Type C manipulation).

**Function Signature:**
```python
def detect_news_price_divergence(
    sentiment_signal: str,  # 'BUY' | 'SELL' | 'HOLD'
    sentiment_score: float,  # -1 to +1
    price_change: float,  # % change since yesterday or recent
    todays_articles: List[Dict],
    source_reliability_dict: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Detects divergence between news sentiment and actual price movement.
    
    Returns:
        {
            'detected': bool,
            'divergence_type': 'A' | 'B' | 'C' | None,
            'sentiment_direction': 'positive' | 'negative' | 'neutral',
            'price_direction': 'up' | 'down' | 'flat',
            'credible_source_corroboration': bool,
            'severity': 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
            'contribution_score': int,  # 0-25
            'explanation': str
        }
    """
```

**Algorithm:**
```
1. Determine sentiment_direction:
   - sentiment_score > 0.2: 'positive'
   - sentiment_score < -0.2: 'negative'
   - else: 'neutral'

2. Determine price_direction:
   - price_change > 0.5%: 'up'
   - price_change < -0.5%: 'down'
   - else: 'flat'

3. Check for credible source corroboration:
   - Count articles from sources with accuracy_rate > 0.70
   - If >= 3 credible articles: corroboration = True

4. Identify divergence type:

   TYPE A (Positive news, negative price):
   - sentiment = positive AND price = down
   - Severity: LOW (5 points)
   - Explanation: "Market expected more" or "Already priced in"

   TYPE B (Negative news, positive price):
   - sentiment = negative AND price = up
   - Severity: MEDIUM (10 points)
   - Explanation: "Short covering" or "Market expected worse"

   TYPE C (MANIPULATION — low-credibility positive, price rising):
   - sentiment = positive
   - price = up
   - credible_source_corroboration = False
   - avg composite_anomaly > 0.6
   - Severity: CRITICAL (25 points)
   - Explanation: "Price moving on unreliable sources only — pump signal"

5. Set detected = True if divergence type exists
```

**Type C is the smoking gun** — this is your strongest pump-and-dump indicator.

---

### Detection System 5: Cross-Stream Coherence Detector

**Purpose:** Detect when stock news diverges from market/related news in suspicious ways.

**Function Signature:**
```python
def detect_cross_stream_incoherence(
    stock_sentiment: float,
    market_sentiment: float,
    related_sentiment: float,
    stock_article_count: int,
    market_article_count: int,
    related_article_count: int,
    stock_avg_anomaly: float
) -> Dict[str, Any]:
    """
    Detects when news streams don't align as expected.
    
    Returns:
        {
            'detected': bool,
            'coherence_score': float,  # 0-1 (1 = perfect alignment)
            'stock_stream_sentiment': float,
            'market_stream_sentiment': float,
            'related_stream_sentiment': float,
            'isolated_signal': bool,  # stock diverges significantly
            'severity': 'LOW' | 'MEDIUM' | 'HIGH',
            'contribution_score': int,  # 0-10
            'explanation': str
        }
    """
```

**Algorithm:**
```
1. Calculate sentiment differences:
   stock_market_diff = abs(stock_sentiment - market_sentiment)
   stock_related_diff = abs(stock_sentiment - related_sentiment)

2. Calculate coherence_score:
   coherence = 1.0 - (stock_market_diff + stock_related_diff) / 4.0
   # Range: 0 (completely incoherent) to 1 (perfect alignment)

3. Check for isolation:
   - If stock_sentiment > 0.4 AND market < 0.1 AND related < 0.1:
     isolated_signal = True
   - If stock has 50+ articles but market/related < 10 articles:
     isolated_signal = True

4. Determine severity:
   - coherence > 0.7: LOW (0 points)
   - 0.5 <= coherence <= 0.7: MEDIUM (5 points)
   - coherence < 0.5: HIGH (8 points)
   - If isolated_signal AND stock_avg_anomaly > 0.6: add 2 bonus points

5. Set detected = True if coherence < 0.6 OR isolated_signal = True
```

**Key Insight:** Legitimate company-specific events (earnings, M&A) create isolated signals, but they come from credible sources. Isolated signals from low-credibility sources are manipulation.

---

### Detection System 6: Historical Pattern Matcher

**Purpose:** Find past days with similar behavioral profiles and check their outcomes.

**Function Signature:**
```python
def match_historical_patterns(
    today_profile: Dict,
    historical_daily_aggregates: List[Dict],
    similarity_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Finds historical days similar to today and checks their outcomes.
    
    Args:
        today_profile: {
            'article_count': int,
            'avg_composite_anomaly': float,
            'avg_source_reliability': float,
            'volume_ratio': float,
            'sentiment_direction': str,
            'price_direction': str
        }
        historical_daily_aggregates: From database query
        similarity_threshold: Minimum similarity to count as match (0-1)
    
    Returns:
        {
            'detected': bool,  # True if found manipulation pattern
            'similar_periods_found': int,
            'similarity_scores': List[float],  # for matched days
            'outcomes': {
                'pct_ended_in_decline': float,
                'pct_ended_in_crash': float,  # decline > 5%
                'avg_price_change_7d': float,
                'worst_outcome': float,
                'best_outcome': float
            },
            'pattern_match_confidence': float,  # 0-1
            'severity': 'LOW' | 'MEDIUM' | 'HIGH',
            'contribution_score': int,  # 0-10
            'explanation': str
        }
    """
```

**Algorithm:**
```
1. Build today_profile from current state:
   {
       'article_count': len(cleaned_stock_news),
       'avg_composite_anomaly': mean([a['composite_anomaly_score'] for a in articles]),
       'avg_source_reliability': weighted_avg_from_node8,
       'volume_ratio': current_volume / baseline_volume,
       'sentiment_direction': 'positive' | 'negative',
       'price_direction': 'up' | 'down' | 'flat'
   }

2. For each historical_day in historical_daily_aggregates:
   Calculate similarity score using weighted Euclidean distance:
   
   differences = [
       abs(today.article_count - hist.article_count) / max(today, hist),
       abs(today.avg_composite_anomaly - hist.avg_composite_anomaly),
       abs(today.avg_source_reliability - hist.avg_source_reliability),
       abs(today.volume_ratio - hist.volume_ratio) / max(today, hist)
   ]
   
   # Weight the differences
   weighted_diff = (
       differences[0] * 0.20 +  # article count
       differences[1] * 0.30 +  # anomaly score (most important)
       differences[2] * 0.30 +  # source reliability (most important)
       differences[3] * 0.20    # volume ratio
   )
   
   similarity = 1.0 - weighted_diff

3. Filter historical days where similarity > similarity_threshold

4. If fewer than 5 similar days found:
   - Try lowering threshold to 0.65
   - If still < 5: return insufficient_data

5. For matched days, analyze outcomes:
   - Count how many had price_change_7d < -2% (decline)
   - Count how many had price_change_7d < -5% (crash)
   - Calculate average 7-day price change
   - Track best/worst outcomes

6. Determine if it's a manipulation pattern:
   - If pct_ended_in_crash > 0.50: detected = True, HIGH severity (10 points)
   - If pct_ended_in_decline > 0.60: detected = True, MEDIUM severity (7 points)
   - Else: detected = False (0 points)

7. Calculate pattern_match_confidence = similar_periods_found / 180
```

**This is the most sophisticated detector** — it learns from the full 180-day history what similar situations led to.

---

### Detection System 7: Pump-and-Dump Composite Scorer

**Purpose:** Combine all detection system scores into a single risk assessment.

**Function Signature:**
```python
def calculate_pump_and_dump_score(
    volume_result: Dict,
    reliability_result: Dict,
    velocity_result: Dict,
    divergence_result: Dict,
    coherence_result: Dict,
    pattern_result: Dict
) -> Dict[str, Any]:
    """
    Aggregates all detection system results into final risk score.
    
    Returns:
        {
            'pump_and_dump_score': int,  # 0-100
            'risk_level': 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
            'primary_risk_factors': List[str],  # top 3 contributing factors
            'detection_breakdown': {
                'volume_anomaly': int,
                'source_reliability_divergence': int,
                'news_velocity_anomaly': int,
                'news_price_divergence': int,
                'cross_stream_incoherence': int,
                'historical_pattern_match': int
            }
        }
    """
```

**Algorithm:**
```
1. Sum contribution scores from all detectors:
   total_score = (
       volume_result['contribution_score'] +        # max 20
       reliability_result['contribution_score'] +   # max 20
       velocity_result['contribution_score'] +      # max 15
       divergence_result['contribution_score'] +    # max 25
       coherence_result['contribution_score'] +     # max 10
       pattern_result['contribution_score']         # max 10
   )
   # Total possible: 100 points

2. Determine risk_level based on total_score:
   - 0-30: LOW
   - 31-55: MEDIUM
   - 56-75: HIGH
   - 76-100: CRITICAL

3. Identify primary risk factors (top 3 contributors):
   Sort detectors by contribution_score descending, take top 3

4. Special rule — Type C divergence auto-elevates to HIGH:
   If divergence_type == 'C' AND total_score < 56:
       total_score = max(total_score, 56)  # force to HIGH
       risk_level = 'HIGH'
```

---

## Main Node Function

**Function Signature:**
```python
def behavioral_anomaly_detection_node(state: Dict) -> Dict:
    """
    Main node function for behavioral anomaly detection.
    
    Orchestrates all 7 detection systems and produces final risk assessment.
    
    Args:
        state: Full StockAnalysisState dictionary
    
    Returns:
        Updated state with behavioral_anomaly_detection field populated
    """
```

**Implementation Steps:**

```python
1. Extract required data from state:
   - ticker
   - current_price, raw_price_data
   - technical_analysis (for volume_ratio)
   - sentiment_analysis
   - news_impact_verification (Node 8 output)
   - content_analysis_summary (Node 9A output)
   - cleaned_stock_news, cleaned_market_news, cleaned_related_company_news
   - market_context

2. Query database for historical data:
   - historical_daily_aggregates = get_historical_daily_aggregates(ticker, 180)
   - volume_baseline = get_volume_baseline(ticker, 30)
   - article_count_baseline = get_article_count_baseline(ticker, 180)

3. Calculate current metrics needed:
   - current_volume = raw_price_data['volume'].iloc[-1]
   - price_change_1d = ((current_price - yesterday_close) / yesterday_close) * 100
   - today_article_count = len(cleaned_stock_news) + len(cleaned_market_news) + len(cleaned_related_company_news)

4. Run all 7 detection systems:
   volume_result = detect_volume_anomaly(...)
   reliability_result = detect_source_reliability_divergence(...)
   velocity_result = detect_news_velocity_anomaly(...)
   divergence_result = detect_news_price_divergence(...)
   coherence_result = detect_cross_stream_incoherence(...)
   pattern_result = match_historical_patterns(...)

5. Calculate composite score:
   composite_result = calculate_pump_and_dump_score(
       volume_result,
       reliability_result,
       velocity_result,
       divergence_result,
       coherence_result,
       pattern_result
   )

6. Build alerts list:
   alerts = []
   if volume_result['detected']:
       alerts.append(f"Volume {volume_result['volume_ratio']:.1f}x above normal")
   if reliability_result['detected']:
       alerts.append(f"Source reliability {reliability_result['divergence']*100:.0f}% below baseline")
   # ... add alert for each detected anomaly

7. Determine trading_recommendation:
   if risk_level == 'CRITICAL':
       trading_recommendation = 'DO_NOT_TRADE'
   elif risk_level == 'HIGH':
       trading_recommendation = 'CAUTION'
   else:
       trading_recommendation = 'NORMAL'

8. Build behavioral_summary:
   if risk_level in ['HIGH', 'CRITICAL']:
       summary = "Multiple coordinated manipulation signals detected"
   elif risk_level == 'MEDIUM':
       summary = "Some unusual behavioral patterns detected"
   else:
       summary = "No significant behavioral anomalies detected"

9. Update state:
   state['behavioral_anomaly_detection'] = {
       'risk_level': composite_result['risk_level'],
       'pump_and_dump_score': composite_result['pump_and_dump_score'],
       'volume_anomaly': volume_result,
       'source_reliability_divergence': reliability_result,
       'news_velocity_anomaly': velocity_result,
       'news_price_divergence': divergence_result,
       'cross_stream_coherence': coherence_result,
       'historical_pattern_match': pattern_result,
       'alerts': alerts,
       'behavioral_summary': summary,
       'trading_recommendation': trading_recommendation,
       'execution_time': elapsed_time
   }

10. Track execution time:
    state['node_execution_times']['node_9b'] = elapsed_time

11. Handle errors gracefully:
    - If any detector fails, log error but continue with others
    - If database queries fail, use neutral defaults
    - Never crash — always produce some output

12. Return updated state
```

---

## Error Handling & Edge Cases

### Case 1: Insufficient Historical Data
```python
if len(historical_daily_aggregates) < 30:
    # Not enough history for pattern matching
    pattern_result = {
        'detected': False,
        'similar_periods_found': 0,
        'insufficient_data': True,
        'contribution_score': 0
    }
    # Continue with other detectors
```

### Case 2: Missing Node 8 Data
```python
if state.get('news_impact_verification', {}).get('insufficient_data', True):
    # Node 8 didn't have enough data to learn
    # Use neutral defaults for source reliability
    reliability_result = {
        'detected': False,
        'severity': 'LOW',
        'contribution_score': 0,
        'insufficient_data': True
    }
```

### Case 3: No Articles Today
```python
if today_article_count == 0:
    # No news to analyze
    return state with all detectors showing 'detected': False
    # Set risk_level = 'LOW', pump_and_dump_score = 0
```

### Case 4: Database Query Fails
```python
try:
    historical_data = get_historical_daily_aggregates(ticker, 180)
except Exception as e:
    logger.error(f"Database query failed: {e}")
    historical_data = []
    # Pattern matcher will handle empty data gracefully
```

### Case 5: Price Data Unavailable
```python
if raw_price_data is None or len(raw_price_data) < 2:
    # Can't calculate price changes or volume ratios
    # Set volume and divergence detectors to neutral
    volume_result = neutral_result()
    divergence_result = neutral_result()
```

---

## Output Structure — What Goes in State

```python
state['behavioral_anomaly_detection'] = {
    # Overall risk assessment
    'risk_level': 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
    'pump_and_dump_score': 0-100,  # int
    'trading_recommendation': 'NORMAL' | 'CAUTION' | 'DO_NOT_TRADE',
    'behavioral_summary': str,  # one-line summary
    
    # Individual detection results
    'volume_anomaly': {
        'detected': bool,
        'volume_ratio': float,
        'severity': str,
        'contribution_score': int,
        'baseline_volume': float,
        'current_volume': float
    },
    
    'source_reliability_divergence': {
        'detected': bool,
        'today_avg_accuracy': float,
        'historical_avg_accuracy': float,
        'divergence': float,
        'severity': str,
        'contribution_score': int,
        'unreliable_source_count': int,
        'total_article_count': int
    },
    
    'news_velocity_anomaly': {
        'detected': bool,
        'today_article_count': int,
        'historical_daily_avg': float,
        'velocity_ratio': float,
        'severity': str,
        'contribution_score': int,
        'high_anomaly_article_count': int,
        'coordinated_keywords': bool
    },
    
    'news_price_divergence': {
        'detected': bool,
        'divergence_type': 'A' | 'B' | 'C' | None,
        'sentiment_direction': str,
        'price_direction': str,
        'credible_source_corroboration': bool,
        'severity': str,
        'contribution_score': int,
        'explanation': str
    },
    
    'cross_stream_coherence': {
        'detected': bool,
        'coherence_score': float,
        'stock_stream_sentiment': float,
        'market_stream_sentiment': float,
        'related_stream_sentiment': float,
        'isolated_signal': bool,
        'severity': str,
        'contribution_score': int,
        'explanation': str
    },
    
    'historical_pattern_match': {
        'detected': bool,
        'similar_periods_found': int,
        'outcomes': {
            'pct_ended_in_decline': float,
            'pct_ended_in_crash': float,
            'avg_price_change_7d': float,
            'worst_outcome': float,
            'best_outcome': float
        },
        'pattern_match_confidence': float,
        'severity': str,
        'contribution_score': int,
        'insufficient_data': bool
    },
    
    # Composite results
    'detection_breakdown': {
        'volume_anomaly': int,
        'source_reliability_divergence': int,
        'news_velocity_anomaly': int,
        'news_price_divergence': int,
        'cross_stream_incoherence': int,
        'historical_pattern_match': int
    },
    
    'primary_risk_factors': List[str],  # top 3 contributors
    
    # User-facing alerts
    'alerts': List[str],  # specific warnings
    
    # Metadata
    'execution_time': float,
    'analysis_timestamp': str
}
```

---

## Testing Strategy

### Test 1: Clean Data (No Anomalies)
```python
# Normal trading day — low volume, credible sources, aligned streams
state = build_clean_state(
    volume_ratio=1.1,
    source_reliability=0.68,
    article_count=8,
    sentiment_aligned=True
)
result = behavioral_anomaly_detection_node(state)
assert result['behavioral_anomaly_detection']['risk_level'] == 'LOW'
assert result['behavioral_anomaly_detection']['pump_and_dump_score'] < 30
```

### Test 2: Volume Spike Only
```python
# High volume but everything else normal
state = build_state_with_volume_spike(volume_ratio=4.5)
result = behavioral_anomaly_detection_node(state)
assert result['behavioral_anomaly_detection']['volume_anomaly']['detected'] == True
assert result['behavioral_anomaly_detection']['risk_level'] in ['LOW', 'MEDIUM']
# Volume alone shouldn't trigger HIGH risk
```

### Test 3: Type C Divergence (Pump Signal)
```python
# Low-cred sources, positive sentiment, price rising, no corroboration
state = build_type_c_scenario(
    sentiment='positive',
    price_direction='up',
    avg_source_reliability=0.32,
    credible_sources=0
)
result = behavioral_anomaly_detection_node(state)
assert result['behavioral_anomaly_detection']['news_price_divergence']['divergence_type'] == 'C'
assert result['behavioral_anomaly_detection']['risk_level'] in ['HIGH', 'CRITICAL']
assert result['behavioral_anomaly_detection']['pump_and_dump_score'] > 60
```

### Test 4: Combined Signals (Full Pump-and-Dump)
```python
# Multiple anomalies together
state = build_pump_and_dump_scenario(
    volume_ratio=6.0,
    source_reliability=0.28,
    article_count_ratio=8.0,
    divergence_type='C',
    isolated_signal=True
)
result = behavioral_anomaly_detection_node(state)
assert result['behavioral_anomaly_detection']['risk_level'] == 'CRITICAL'
assert result['behavioral_anomaly_detection']['pump_and_dump_score'] > 75
assert result['behavioral_anomaly_detection']['trading_recommendation'] == 'DO_NOT_TRADE'
```

### Test 5: Historical Pattern Match
```python
# Create historical data with known outcomes
historical_data = create_mock_history_with_crashes(crash_rate=0.75)
state = build_state_with_history(historical_data)
result = behavioral_anomaly_detection_node(state)
assert result['behavioral_anomaly_detection']['historical_pattern_match']['detected'] == True
assert result['behavioral_anomaly_detection']['historical_pattern_match']['outcomes']['pct_ended_in_crash'] > 0.6
```

### Test 6: Insufficient Data (Graceful Degradation)
```python
# Node 8 had insufficient data
state = build_state_with_insufficient_node8_data()
result = behavioral_anomaly_detection_node(state)
# Should still work, just with neutral reliability scores
assert 'behavioral_anomaly_detection' in result
assert result['behavioral_anomaly_detection']['source_reliability_divergence']['insufficient_data'] == True
```

### Test 7: Missing Price Data
```python
state = build_state_without_price_data()
result = behavioral_anomaly_detection_node(state)
# Should not crash
assert 'behavioral_anomaly_detection' in result
assert result['behavioral_anomaly_detection']['volume_anomaly']['detected'] == False
```

### Test 8: Cross-Stream Incoherence
```python
# Stock news bullish, market/related neutral
state = build_incoherent_streams(
    stock_sentiment=0.75,
    market_sentiment=0.05,
    related_sentiment=-0.02,
    stock_anomaly=0.68
)
result = behavioral_anomaly_detection_node(state)
assert result['behavioral_anomaly_detection']['cross_stream_coherence']['detected'] == True
assert result['behavioral_anomaly_detection']['cross_stream_coherence']['isolated_signal'] == True
```

### Test 9: Execution Time
```python
# Performance test
state = build_realistic_state()
result = behavioral_anomaly_detection_node(state)
assert result['behavioral_anomaly_detection']['execution_time'] < 2.0
# Target: under 2 seconds
```

### Test 10: State Immutability
```python
# Verify Node 9B doesn't modify other state fields
state_before = deepcopy(state)
result = behavioral_anomaly_detection_node(state)
# Check that only 'behavioral_anomaly_detection' was added
assert state_before['sentiment_analysis'] == result['sentiment_analysis']
assert state_before['cleaned_stock_news'] == result['cleaned_stock_news']
```

---

## Integration with Conditional Edge

The conditional edge in `workflow.py` reads Node 9B's output:

```python
def should_halt_for_risk(state: StockAnalysisState) -> str:
    """
    Conditional edge: Determine if critical risk requires halting.
    
    Returns:
        'halt' → Skip Node 12, go straight to Node 13 (warning only)
        'continue' → Proceed to Node 12 (normal signal generation)
    """
    early = state.get('early_anomaly_detection', {})
    behavioral = state.get('behavioral_anomaly_detection', {})
    
    pump_dump_score = behavioral.get('pump_and_dump_score', 0)
    early_risk = early.get('early_risk_level', 'LOW')
    behavioral_risk = behavioral.get('risk_level', 'LOW')
    
    # Critical conditions that trigger halt
    if pump_dump_score > 75:
        logger.warning(f"HALT: Pump-and-dump detected (score: {pump_dump_score})")
        return 'halt'
    
    if behavioral_risk == 'CRITICAL':
        logger.warning(f"HALT: Critical behavioral risk")
        return 'halt'
    
    if early_risk == 'HIGH' and behavioral_risk == 'HIGH':
        logger.warning(f"HALT: Combined high risk from both phases")
        return 'halt'
    
    # Safe to continue
    return 'continue'
```

---

## Success Criteria Checklist

- [ ] All 7 detection systems implemented as independent functions
- [ ] Each detector returns proper dict structure with detected, severity, contribution_score
- [ ] Pump-and-dump composite scorer aggregates correctly (0-100 scale)
- [ ] Risk levels calculated correctly: LOW (0-30), MEDIUM (31-55), HIGH (56-75), CRITICAL (76-100)
- [ ] Volume anomaly detector works with historical baseline
- [ ] Source reliability divergence uses Node 8 data correctly
- [ ] News velocity compares to historical article count baseline
- [ ] Type C divergence (pump signal) detected correctly
- [ ] Cross-stream coherence checks all 3 news streams
- [ ] Historical pattern matcher finds similar days and checks outcomes
- [ ] Main node function orchestrates all detectors
- [ ] Error handling for missing data (Node 8 insufficient, no price data, etc.)
- [ ] State output matches expected structure
- [ ] Never modifies existing state fields (only adds behavioral_anomaly_detection)
- [ ] Execution time < 2 seconds
- [ ] All 10+ tests pass
- [ ] Graceful degradation when data is missing
- [ ] Database functions added to db_manager
- [ ] Type hints on all functions
- [ ] Comprehensive docstrings
- [ ] Logger used for all operations
- [ ] Follows all 16 project rules

---

## Files to Create

1. **`src/langgraph_nodes/node_09b_behavioral_anomaly.py`** (main implementation, ~1200 lines)
   - All 7 detection system functions
   - Composite scorer
   - Main node function
   - Helper utilities

2. **`tests/test_nodes/test_node_09b.py`** (comprehensive tests, ~800 lines)
   - Unit tests for each detector
   - Integration tests
   - Edge case tests
   - Mock data builders

3. **`scripts/test_node_09b_integration.py`** (end-to-end validation, ~300 lines)
   - Full pipeline test: Node 1 → ... → Node 9B
   - Multiple test scenarios (clean, pump, crash)
   - Performance benchmarks

---

## Files to Modify

1. **`src/graph/state.py`**
   - Add `behavioral_anomaly_detection: Optional[Dict[str, Any]]` to StockAnalysisState
   - Update create_initial_state() to initialize as None

2. **`src/database/db_manager.py`**
   - Add `get_historical_daily_aggregates(ticker, days)`
   - Add `get_volume_baseline(ticker, days)`
   - Add `get_article_count_baseline(ticker, days)`

3. **`src/graph/workflow.py`** (if not already done)
   - Add Node 9B to the graph after Node 8
   - Ensure conditional edge reads from Node 9B

---

## Estimated Time Breakdown

- Database functions (3 new functions): 1-2 hours
- Detection System 1 (Volume Anomaly): 1 hour
- Detection System 2 (Source Reliability): 1 hour
- Detection System 3 (News Velocity): 1 hour
- Detection System 4 (News-Price Divergence): 1.5 hours
- Detection System 5 (Cross-Stream Coherence): 1 hour
- Detection System 6 (Historical Pattern Matcher): 2 hours
- Detection System 7 (Composite Scorer): 1 hour
- Main node function + error handling: 1.5 hours
- Testing (10+ comprehensive tests): 2-3 hours
- Integration testing + validation: 1-2 hours
- Documentation + cleanup: 1 hour

**Total: 8-10 hours**

---

## Key Architectural Reminders

1. **Node 9B is READ-ONLY** — it never modifies cleaned_stock_news, sentiment_analysis, or any other existing state
2. **Node 9B is ADDITIVE** — it only adds the behavioral_anomaly_detection field
3. **Node 9B depends on Node 8** — source reliability data is critical
4. **Node 9B feeds the conditional edge** — pump_and_dump_score determines halt vs continue
5. **Node 9B enriches Node 12** — risk summary uses 9B's alerts
6. **Type C divergence is the smoking gun** — this is your strongest manipulation signal
7. **Historical pattern matching is optional but powerful** — if insufficient data, other detectors still work

---

## Final Pre-Build Checklist

Before starting implementation, verify:

- [ ] Node 8 is complete and producing source_reliability data
- [ ] Node 9A is complete and producing cleaned news with embedded scores
- [ ] Database has news_outcomes table populated (via background script)
- [ ] Database has price_data table with volume history
- [ ] State definition includes all required fields from Nodes 1-8
- [ ] You understand the difference between Node 8 (learning) and 9B (detection)
- [ ] You understand why 9B runs after Node 8 but before Node 10
- [ ] Conditional edge logic is clear (when to halt vs continue)

---

**You're ready to build Node 9B — the behavioral detective that catches pump-and-dump schemes! 🕵️**
