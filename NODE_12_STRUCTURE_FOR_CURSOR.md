# Node 12: Final Signal Generation + Historical Pattern Prediction
## Complete Build Roadmap for Cursor

**File:** `src/langgraph_nodes/node_12_signal_generation.py`
**Estimated Time:** 5-6 hours
**Runs AFTER:** Node 11 (adaptive weights)
**Runs BEFORE:** Nodes 13, 14 (LLM explanations)
**Only runs if:** Conditional edge says 'continue' (not 'halt')

---

## What This Node Does

Node 12 is the brain that produces the final output. It has TWO jobs:

**Job 1 — Weighted Signal Combination (simple math):**
Take the 4 signals (technical, stock news, market context, related companies), multiply each by its adaptive weight from Node 11, and produce a BUY/SELL/HOLD recommendation with confidence.

**Job 2 — Historical Pattern Prediction (the intelligence):**
Take today's full picture (technical indicators + sentiment + market + related), find past days from the last 180 days where ALL signals looked similar to today, and report what happened after those days. Use Node 11's adaptive weights to make the similarity search smarter — signals that are more accurate for this stock should matter more when finding "similar" days.

**The user-facing output should sound like:**
> "We recommend BUY with 72% confidence. In the last 180 days, we found 18 days where conditions were similar to today. In those cases, the price went up 78% of the time, with an average gain of +2.4% over 7 days. The expected range is -0.8% to +5.1%."

---

## What Node 12 Does NOT Do

- Does NOT fetch any data from APIs or databases
- Does NOT recalculate anything that previous nodes already computed
- Does NOT call any LLM (that's Node 13 and 14's job)
- Does NOT modify any previous analysis results

Everything Node 12 needs is already in the state from previous nodes. It only READS from state and WRITES its results.

---

## State Available to Node 12

When Node 12 runs, the following data is already in the state:

```python
# From Node 4 (Technical Analysis)
state['technical_analysis'] = {
    'technical_signal': 'BUY' | 'SELL' | 'HOLD',
    'confidence': 62.0,              # 0-100
    'rsi': 58.4,
    'macd_signal': 'bullish_crossover' | 'bearish_crossover' | 'neutral',
    'bollinger_position': 'middle' | 'upper' | 'lower',
    'sma_20': 145.30,
    'sma_50': 141.20,
    'sma_trend': 'golden_cross' | 'death_cross' | 'neutral',
    'volume_ratio': 1.2,             # vs 20-day average
    # ... other indicators
}

# From Node 5 (Sentiment Analysis) — already adjusted by Node 8
state['sentiment_analysis'] = {
    'sentiment_signal': 'BUY' | 'SELL' | 'HOLD',
    'confidence': 71.0,              # already adjusted by Node 8
    'combined_sentiment_score': 0.31, # -1 to +1
    'sentiment_label': 'positive' | 'negative' | 'neutral',
    'stock_news_sentiment': 0.45,
    'market_news_sentiment': 0.22,
    'related_news_sentiment': 0.18,
    'article_count': 10,
    'confidence_adjustment': {        # from Node 8
        'original': 70.0,
        'reliability_multiplier': 1.01,
        'final': 71.0
    }
}

# From Node 6 (Market Context)
state['market_context'] = {
    'context_signal': 'BUY' | 'SELL' | 'HOLD',
    'confidence': 65.0,
    'sector_performance': 1.8,       # % change
    'market_trend': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
    'spy_5day_change': 1.5,          # % change
    'correlation_with_market': 0.82,
    'beta': 1.3,
    'related_companies_performance': {
        'AMD': 2.1,
        'INTC': 0.8,
        'TSM': 1.9
    }
}

# From Node 7 (Monte Carlo)
state['monte_carlo'] = {
    'median_price_30d': 148.0,
    'current_price': 143.5,
    'confidence_intervals': {
        '68': {'low': 138.0, 'high': 155.0},
        '95': {'low': 130.0, 'high': 162.0}
    },
    'probability_of_increase': 0.68,
    'expected_return_30d': 3.1        # %
}

# From Node 8 (News Verification & Learning)
state['news_impact_verification'] = {
    'historical_correlation': 0.72,   # 0-1, how well news predicts price
    'news_accuracy_score': 68.0,      # weighted avg accuracy across sources
    'verified_signal_strength': 71.0,
    'learning_adjustment': 1.2,       # 0.5-2.0, for Node 11
    'source_reliability': {
        'Bloomberg.com': {'accuracy_rate': 0.85, 'total_articles': 45, 'confidence_multiplier': 1.10},
        'Reuters': {'accuracy_rate': 0.78, 'total_articles': 38, 'confidence_multiplier': 1.0},
        'CNBC': {'accuracy_rate': 0.72, 'total_articles': 52, 'confidence_multiplier': 1.0},
        'SeekingAlpha': {'accuracy_rate': 0.48, 'total_articles': 28, 'confidence_multiplier': 0.98},
        # ...
    },
    'news_type_effectiveness': {
        'stock': {'accuracy_rate': 0.68, 'avg_impact': 3.2, 'sample_size': 45},
        'market': {'accuracy_rate': 0.52, 'avg_impact': 1.1, 'sample_size': 30},
        'related': {'accuracy_rate': 0.61, 'avg_impact': 1.8, 'sample_size': 25}
    }
}

# From Node 9A (Early Anomaly Detection)
state['early_anomaly_detection'] = {
    'early_risk_level': 'LOW' | 'MEDIUM' | 'HIGH',
    'alerts': [...],
    'filtered_article_count': 2
}

# From Node 9B (Behavioral Anomaly Detection)
state['behavioral_anomaly_detection'] = {
    'risk_level': 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
    'pump_and_dump_score': 32,        # 0-100
    'price_anomaly': False,
    'volume_anomaly': True,           # flagged but not critical
    'volatility_anomaly': False,
    'news_price_divergence': False,
    'alerts': [...]
}

# From Node 10 (Backtesting)
state['backtest_results'] = {
    'technical_accuracy': 55.0,
    'stock_news_accuracy': 65.0,
    'market_news_accuracy': 58.0,
    'related_companies_accuracy': 60.0,
    'sample_size': 120,
    'historical_daily_snapshots': [...]  # SEE SECTION BELOW
}

# From Node 11 (Adaptive Weights)
state['adaptive_weights'] = {
    'technical_weight': 0.231,
    'stock_news_weight': 0.273,
    'market_news_weight': 0.244,
    'related_weight': 0.252,
    'weights_explanation': "Stock news highest due to Node 8 learning..."
}
```

---

## CRITICAL: What Node 10 Must Pass for Job 2

For the pattern prediction to work, Node 10 needs to pass `historical_daily_snapshots` in the state. This is a list of daily summaries for the last 180 days. Node 10 already loads all this data for backtesting — it just needs to package it instead of throwing it away.

**Node 10 must include this in `state['backtest_results']`:**

```python
state['backtest_results']['historical_daily_snapshots'] = [
    {
        'date': '2025-08-15',
        # Technical indicators on that day
        'rsi': 55.2,
        'macd_signal': 'bullish_crossover',
        'bollinger_position': 'middle',
        'sma_trend': 'neutral',
        'volume_ratio': 1.1,
        # Sentiment on that day
        'sentiment_score': 0.28,
        'sentiment_label': 'positive',
        'article_count': 6,
        # Market context on that day
        'sector_performance': 1.2,
        'market_trend': 'BULLISH',
        'spy_change': 0.8,
        # What ACTUALLY happened after that day
        'price_at_date': 135.50,
        'price_change_1d': 0.8,    # %
        'price_change_3d': 1.4,    # %
        'price_change_7d': 2.1,    # %
        'actual_direction_7d': 'UP' # UP if >0.5%, DOWN if <-0.5%, FLAT otherwise
    },
    # ... 180 entries, one per trading day
]
```

**Important:** If Node 10 cannot produce these snapshots yet, Node 12 should still work. Job 2 (pattern prediction) should gracefully return `insufficient_data: True` and Node 12 falls back to Job 1 only. The system still works, it just doesn't have the pattern prediction yet.

---

## Implementation Plan

### Function 1: `combine_weighted_signals()` — Job 1

**Purpose:** Combine 4 signals using adaptive weights into BUY/SELL/HOLD.

**Input:**
- `technical_analysis` dict from state
- `sentiment_analysis` dict from state
- `market_context` dict from state
- `adaptive_weights` dict from state

**Logic:**

1. Extract each signal's score (0-100 scale):
   - If signal is BUY, use its confidence as the score
   - If signal is SELL, use `100 - confidence` as the score (inverted)
   - If signal is HOLD, use 50

2. Calculate weighted score:
```
weighted_score = (tech_score × tech_weight) + (news_score × news_weight) 
               + (market_score × market_weight) + (related_score × related_weight)
```

3. Determine recommendation:
   - weighted_score > 60 → BUY
   - weighted_score < 40 → SELL
   - else → HOLD

4. Confidence = how far the score is from the threshold:
   - BUY: confidence = weighted_score (already 60-100 range)
   - SELL: confidence = 100 - weighted_score (inverted to positive)
   - HOLD: confidence = 50 (neutral)

**Output:**
```python
{
    'recommendation': 'BUY',
    'confidence': 72.2,
    'weighted_score': 72.2,
    'signal_breakdown': {
        'technical': {'signal': 'BUY', 'score': 80, 'weight': 0.231, 'weighted_contribution': 18.48},
        'stock_news': {'signal': 'BUY', 'score': 90, 'weight': 0.273, 'weighted_contribution': 24.57},
        'market_context': {'signal': 'HOLD', 'score': 50, 'weight': 0.244, 'weighted_contribution': 12.20},
        'related': {'signal': 'BUY', 'score': 70, 'weight': 0.252, 'weighted_contribution': 17.64}
    }
}
```

---

### Function 2: `find_similar_historical_days()` — Job 2

**Purpose:** Find past days where the full picture looked like today and report what happened.

**Input:**
- `current_snapshot` — today's technical + sentiment + market signals (built from state)
- `historical_daily_snapshots` — list from Node 10's backtest results
- `adaptive_weights` — from Node 11

**Logic:**

1. Build today's snapshot from state:
```python
current_snapshot = {
    'rsi': state['technical_analysis']['rsi'],
    'macd_signal': state['technical_analysis']['macd_signal'],
    'bollinger_position': state['technical_analysis']['bollinger_position'],
    'sma_trend': state['technical_analysis']['sma_trend'],
    'volume_ratio': state['technical_analysis']['volume_ratio'],
    'sentiment_score': state['sentiment_analysis']['combined_sentiment_score'],
    'article_count': state['sentiment_analysis']['article_count'],
    'sector_performance': state['market_context']['sector_performance'],
    'market_trend': state['market_context']['market_trend'],
    'spy_change': state['market_context']['spy_5day_change']
}
```

2. For each historical day, calculate WEIGHTED similarity:

```
Similarity has 4 components, one per signal stream:

TECHNICAL similarity (0-1):
  - RSI difference: 1 - (abs(today_rsi - hist_rsi) / 100)
  - MACD match: 1.0 if same signal, 0.5 if neutral, 0.0 if opposite
  - Bollinger match: 1.0 if same position, 0.5 if adjacent, 0.0 if opposite
  - SMA trend match: 1.0 if same, 0.0 if opposite
  - Average these 4 sub-scores → technical_similarity (0-1)

NEWS similarity (0-1):
  - Sentiment difference: 1 - abs(today_sentiment - hist_sentiment)
    (scores are -1 to +1, so max diff is 2, normalize to 0-1)
  - Volume similarity: 1 - abs(today_articles - hist_articles) / max(today_articles, hist_articles)
  - Average → news_similarity (0-1)

MARKET similarity (0-1):
  - Sector diff: 1 - (abs(today_sector - hist_sector) / 10)
    (cap at 10% diff = completely different)
  - Market trend match: 1.0 if same, 0.5 if one is NEUTRAL, 0.0 if opposite
  - Average → market_similarity (0-1)

RELATED similarity (0-1):
  - Use sector performance as proxy since related companies track sector
  - Same calculation as market sector diff
  - → related_similarity (0-1)

WEIGHTED TOTAL:
  similarity = (technical_similarity × tech_weight)
             + (news_similarity × news_weight)
             + (market_similarity × market_weight)
             + (related_similarity × related_weight)
```

3. Filter: keep historical days with similarity > 0.65 (threshold)
   - If fewer than 5 days pass threshold, lower to 0.55
   - If still fewer than 5, return `sufficient_data: False`
   - Cap at 30 most similar days to avoid noise

4. Calculate prediction from similar days:
   - Count UP / DOWN / FLAT outcomes
   - Calculate average price change at 1d, 3d, 7d
   - Calculate median, 10th percentile (worst case), 90th percentile (best case)
   - Determine predicted direction: UP if >60% went up, DOWN if >60% went down, else NEUTRAL

**Output:**
```python
{
    'sufficient_data': True,
    'similar_days_found': 18,
    'similarity_threshold_used': 0.65,
    'predicted_direction': 'UP',
    'probability_up': 0.78,
    'probability_down': 0.11,
    'probability_flat': 0.11,
    'expected_move_1d': 0.8,
    'expected_move_3d': 1.4,
    'expected_move_7d': 2.4,
    'median_move_7d': 2.1,
    'best_case_7d': 5.1,       # 90th percentile
    'worst_case_7d': -0.8,     # 10th percentile
    'similar_days_detail': [    # for transparency and Node 13/14
        {
            'date': '2025-10-15',
            'similarity_score': 0.82,
            'price_change_7d': 3.2,
            'direction': 'UP'
        },
        # ... top 5 most similar for explanation purposes
    ]
}
```

---

### Function 3: `build_risk_summary()`

**Purpose:** Consolidate all risk information from Nodes 9A and 9B into a clean summary.

**Input:**
- `early_anomaly_detection` from state
- `behavioral_anomaly_detection` from state

**Logic:**
1. Combine risk levels from both phases
2. List all active alerts
3. Calculate overall risk level:
   - If either is CRITICAL → CRITICAL
   - If both are HIGH → HIGH
   - If either is HIGH → MEDIUM
   - If either is MEDIUM → LOW-MEDIUM
   - Else → LOW

**Output:**
```python
{
    'overall_risk_level': 'LOW',
    'pump_and_dump_risk': 32,
    'early_detection_risk': 'LOW',
    'behavioral_risk': 'LOW',
    'active_alerts': ['Volume 40% above normal - monitor closely'],
    'trading_safe': True
}
```

---

### Function 4: `build_target_and_stop_loss()`

**Purpose:** Set price targets using Monte Carlo data and pattern prediction.

**Input:**
- `monte_carlo` from state
- `pattern_prediction` from Job 2
- `recommendation` from Job 1

**Logic:**

1. Target price:
   - If pattern prediction has sufficient data: use current_price × (1 + expected_move_7d/100)
   - Else: use Monte Carlo median_price_30d
   - Cross-reference: if Monte Carlo and pattern prediction agree on direction, boost confidence

2. Stop loss:
   - Use Monte Carlo 95% confidence lower bound as maximum risk
   - Or current_price × (1 + worst_case_7d/100) from pattern prediction
   - Take the TIGHTER (less risky) of the two

3. Risk/reward ratio:
   - potential_gain = target_price - current_price
   - potential_loss = current_price - stop_loss
   - ratio = potential_gain / potential_loss

**Output:**
```python
{
    'current_price': 143.50,
    'target_price': 147.00,
    'target_source': 'pattern_prediction',   # or 'monte_carlo'
    'expected_return': 2.4,                   # %
    'stop_loss': 142.35,
    'max_downside': -0.8,                     # %
    'risk_reward_ratio': 3.0,
    'monte_carlo_agrees': True,               # both point same direction
    'time_horizon': '7 days'
}
```

---

### Function 5: `signal_generation_node(state)` — Main Node Function

**Purpose:** Orchestrate all the above functions and produce the complete output.

**Steps:**
1. Start timer
2. Extract all needed data from state
3. Run Job 1: `combine_weighted_signals()` → recommendation + confidence
4. Run Job 2: `find_similar_historical_days()` → pattern prediction
5. Run `build_risk_summary()` → consolidated risks
6. Run `build_target_and_stop_loss()` → price targets
7. Build contributing factors list (what drove the decision)
8. Package everything into `state['final_signal']`
9. Record execution time

**The complete output stored in `state['final_signal']`:**

```python
state['final_signal'] = {
    # Job 1: The recommendation
    'recommendation': 'BUY',
    'confidence': 72.2,
    'weighted_score': 72.2,
    'signal_breakdown': {
        'technical': {'signal': 'BUY', 'score': 80, 'weight': 0.231, 'weighted_contribution': 18.48},
        'stock_news': {'signal': 'BUY', 'score': 90, 'weight': 0.273, 'weighted_contribution': 24.57},
        'market_context': {'signal': 'HOLD', 'score': 50, 'weight': 0.244, 'weighted_contribution': 12.20},
        'related': {'signal': 'BUY', 'score': 70, 'weight': 0.252, 'weighted_contribution': 17.64}
    },

    # Job 2: The pattern prediction
    'pattern_prediction': {
        'sufficient_data': True,
        'similar_days_found': 18,
        'predicted_direction': 'UP',
        'probability_up': 0.78,
        'probability_down': 0.11,
        'probability_flat': 0.11,
        'expected_move_1d': 0.8,
        'expected_move_3d': 1.4,
        'expected_move_7d': 2.4,
        'median_move_7d': 2.1,
        'best_case_7d': 5.1,
        'worst_case_7d': -0.8,
        'similar_days_detail': [...]
    },

    # Price targets
    'target_price': 147.00,
    'stop_loss': 142.35,
    'expected_return': 2.4,
    'risk_reward_ratio': 3.0,
    'time_horizon': '7 days',

    # Risk summary
    'risk_summary': {
        'overall_risk_level': 'LOW',
        'pump_and_dump_risk': 32,
        'active_alerts': ['Volume 40% above normal'],
        'trading_safe': True
    },

    # Contributing factors (for Node 13/14 explanations)
    'contributing_factors': [
        'Strong positive sentiment from reliable sources (Bloomberg 85% accurate, Reuters 78%)',
        'Bullish technical setup (RSI 58.4, golden cross formation)',
        'Favorable sector performance (+1.8%) and market trend',
        'Historical pattern match: 18 similar days found, 78% resulted in price increase',
        'Monte Carlo simulation supports upward movement (68% probability of increase)'
    ],

    # What could go wrong (for Node 13/14 risk section)
    'risk_factors': [
        'Volume 40% above normal - could precede larger moves in either direction',
        'High market correlation (0.82) - vulnerable to broad market pullback',
        'Pattern prediction worst case: -0.8% in 7 days'
    ],

    # Metadata
    'analysis_timestamp': '2026-02-14T15:30:00',
    'ticker': 'NVDA',
    'current_price': 143.50
}
```

---

## Edge Cases and Failsafes

### Insufficient Historical Snapshots
If `historical_daily_snapshots` is empty or missing (Node 10 hasn't been updated yet):
- Skip Job 2 entirely
- Set `pattern_prediction = {'sufficient_data': False, 'reason': 'Historical snapshots not available'}`
- Node 12 still works — it just doesn't have the pattern prediction
- Node 13 explains without the "similar days" section

### Not Enough Similar Days Found
If fewer than 5 similar days pass the similarity threshold:
- First try lowering threshold from 0.65 to 0.55
- If still insufficient, return `sufficient_data: False`
- The recommendation from Job 1 still stands
- Confidence may be slightly lower since pattern prediction can't confirm

### All Signals Disagree
If technical says BUY, news says SELL, market says HOLD, related says SELL:
- The weighted score will land near 50 → HOLD recommendation
- Confidence will be low (near 50%)
- Contributing factors should explicitly note the disagreement
- Add to risk_factors: "Signals are conflicting — high uncertainty"

### Missing State Data
If any previous node failed and its data is None:
- Use default neutral values (signal=HOLD, score=50, weight=equal)
- Log which node data was missing
- Add to risk_factors: "Incomplete analysis — [node name] data unavailable"
- Never crash — always produce some output

### Pattern Prediction Disagrees with Signal Combination
If Job 1 says BUY but Job 2 says historically price went DOWN 60% of the time:
- Do NOT override Job 1's recommendation
- Report BOTH in the output
- Add to risk_factors: "Historical pattern suggests caution — similar conditions led to price decline 60% of the time"
- Reduce confidence by 10-15%
- Let the user decide (Node 13 will explain the disagreement)

---

## What Node 13 Does With This Output

Node 13 receives all of `state['final_signal']` and uses Claude to generate something like:

> "NVDA is showing bullish signals across most indicators. The stock's RSI is at 58, with a recent golden cross formation, suggesting upward momentum with room to grow.
>
> Recent news from Bloomberg and Reuters — which have been 85% and 78% accurate historically for NVDA — is positive, focused on AI chip demand and a new data center contract.
>
> The broader tech sector is up 1.8% this week, and competitors AMD and Intel are also performing well, creating a supportive environment.
>
> Looking at the past 6 months, we found 18 days where market conditions were similar to today. In those cases, NVDA's price went up 78% of the time, with an average gain of 2.4% over the following week. The expected range is -0.8% to +5.1%.
>
> Our Monte Carlo simulation also supports this, showing a 68% probability of price increase over the next 30 days.
>
> One thing to watch: trading volume has been 40% above normal, which sometimes precedes larger moves in either direction.
>
> Overall recommendation: BUY with 72% confidence. Expected target: $147 within 7 days. Consider a stop-loss at $142.35 to manage risk."

---

## Testing Strategy

```python
# Test 1: All signals agree (BUY)
state = build_mock_state(
    technical='BUY', news='BUY', market='BUY', related='BUY',
    weights={'tech': 0.25, 'news': 0.25, 'market': 0.25, 'related': 0.25}
)
result = signal_generation_node(state)
assert result['final_signal']['recommendation'] == 'BUY'
assert result['final_signal']['confidence'] > 70

# Test 2: Mixed signals — news dominant
state = build_mock_state(
    technical='HOLD', news='BUY', market='HOLD', related='BUY',
    weights={'tech': 0.15, 'news': 0.40, 'market': 0.20, 'related': 0.25}
)
result = signal_generation_node(state)
assert result['final_signal']['recommendation'] == 'BUY'  # news weight dominates

# Test 3: All signals disagree
state = build_mock_state(
    technical='BUY', news='SELL', market='HOLD', related='SELL',
    weights={'tech': 0.25, 'news': 0.25, 'market': 0.25, 'related': 0.25}
)
result = signal_generation_node(state)
assert result['final_signal']['recommendation'] == 'HOLD'
assert result['final_signal']['confidence'] < 60

# Test 4: Pattern prediction with sufficient data
state = build_mock_state_with_history(similar_days=20, pct_up=0.80)
result = signal_generation_node(state)
assert result['final_signal']['pattern_prediction']['sufficient_data'] == True
assert result['final_signal']['pattern_prediction']['probability_up'] > 0.7

# Test 5: Pattern prediction with insufficient data
state = build_mock_state_with_history(similar_days=2)
result = signal_generation_node(state)
assert result['final_signal']['pattern_prediction']['sufficient_data'] == False
# Job 1 should still work fine
assert result['final_signal']['recommendation'] in ['BUY', 'SELL', 'HOLD']

# Test 6: Pattern prediction disagrees with signal
state = build_mock_state(
    technical='BUY', news='BUY', market='BUY', related='BUY',
    pattern_history_direction='DOWN'  # history says down
)
result = signal_generation_node(state)
assert result['final_signal']['recommendation'] == 'BUY'  # doesn't override
assert 'Historical pattern suggests caution' in str(result['final_signal']['risk_factors'])
assert result['final_signal']['confidence'] < 70  # reduced due to disagreement

# Test 7: Missing state data (node failure)
state = build_mock_state(technical=None, news='BUY', market='BUY', related='BUY')
result = signal_generation_node(state)
assert result['final_signal']['recommendation'] in ['BUY', 'SELL', 'HOLD']
assert 'Incomplete analysis' in str(result['final_signal']['risk_factors'])

# Test 8: Weighted similarity check
# For NVDA where news_weight=0.40, a day with matching sentiment
# should score higher similarity than a day with matching RSI
state = build_mock_state(weights={'tech': 0.15, 'news': 0.40, 'market': 0.20, 'related': 0.25})
# Day A: same sentiment, different RSI
# Day B: different sentiment, same RSI
# Day A should have higher similarity score
```

---

## Success Criteria

- [ ] Combines all 4 signal streams correctly using adaptive weights
- [ ] Weights are applied correctly (higher weight = more influence)
- [ ] BUY/SELL/HOLD thresholds work (>60, <40, else HOLD)
- [ ] Pattern prediction finds similar historical days
- [ ] Similarity calculation uses adaptive weights (not equal)
- [ ] Handles insufficient historical data gracefully
- [ ] Handles missing state data from failed nodes
- [ ] Handles disagreement between Job 1 and Job 2
- [ ] Target price and stop loss calculated from Monte Carlo + pattern prediction
- [ ] Risk summary consolidates 9A and 9B alerts
- [ ] Contributing factors list is comprehensive
- [ ] Risk factors list includes all warnings
- [ ] Output structure matches what Node 13/14 expect
- [ ] Execution time < 1 second (no API calls, just math)
- [ ] Never crashes — always produces output even with partial data

---

## Dependency on Node 10 Update

**IMPORTANT:** For Job 2 to work, Node 10 must be updated to include `historical_daily_snapshots` in its output. This is a small addition to Node 10 — it already loads all the historical data for backtesting, it just needs to package each day's indicators + outcomes into the list described above.

If you build Node 12 before updating Node 10, that's fine. Just have Job 2 check if snapshots exist:

```python
snapshots = state.get('backtest_results', {}).get('historical_daily_snapshots', [])
if len(snapshots) < 30:
    pattern_prediction = {'sufficient_data': False, 'reason': 'Snapshots not available yet'}
else:
    pattern_prediction = find_similar_historical_days(current_snapshot, snapshots, weights)
```

This way Node 12 works immediately with Job 1 only, and Job 2 activates automatically once Node 10 is updated.
