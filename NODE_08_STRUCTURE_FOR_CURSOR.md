# Node 8: News Impact Verification & Learning System
## Complete Build Roadmap for Cursor
## ⭐ PRIMARY THESIS INNOVATION ⭐

**File:** `src/langgraph_nodes/node_08_news_verification.py`
**Estimated Time:** 6-8 hours
**Runs AFTER:** Nodes 4, 5, 6, 7 (parallel analysis layer)
**Runs BEFORE:** Node 9B (behavioral anomaly detection)
**Parallel with:** Nothing (sequential)

---

## What This Node Does — In Plain English

Node 8 is the learning system. It looks at the last 6 months of history and asks:

1. "When Bloomberg said NVDA was going up, did it actually go up?" → Source reliability
2. "When stock-specific news was positive, was it more accurate than market news?" → News type effectiveness
3. "Overall, does news actually predict price movement for this stock?" → Historical correlation
4. "Given what we learned, how much should we trust today's sentiment score?" → Confidence adjustment

Then it adjusts the sentiment confidence that Node 5 produced. If today's news mostly comes from sources that have been historically accurate for this stock, confidence goes up. If it comes from unreliable sources, confidence goes down.

It also outputs a `learning_adjustment` factor (0.5 to 2.0) that Node 11 will use to influence how much weight the news signal gets in the final recommendation.

---

## Reference File

There is a complete reference implementation at:
`LangGraph_setup/node_08_news_verification_COMPLETE.py` (699 lines)

**Use it as a reference for the algorithm logic, but DO NOT copy-paste it directly.** It has several issues that must be fixed (listed below). Understand the logic, then implement it following the project rules.

---

## CRITICAL: Project Rules That Apply

The reference file violates project rules. The new implementation must follow them:

### Rule 1: Never Talk to Database Directly
The reference file uses `sqlite3.connect('data/stock_prices.db')` directly. This is WRONG.

```python
# ❌ WRONG (reference file does this):
import sqlite3
with sqlite3.connect('data/stock_prices.db') as conn:
    cursor.execute("SELECT ...")

# ✅ CORRECT (use db_manager):
from src.database.db_manager import get_news_with_outcomes, save_source_reliability
results = get_news_with_outcomes(ticker, days_back=180)
```

All database operations must go through `src/database/db_manager.py`. If the needed functions don't exist in db_manager yet, ADD them there first, then call them from Node 8.

### Rule 2: Use the Logger
```python
from src.utils.logger import get_node_logger
logger = get_node_logger("node_08")
```

### Rule 3: Nodes Are Independent
Node 8 reads from state. It does NOT import any other node.

---

## What Node 8 Receives From State

When Node 8 runs, the parallel analysis layer (Nodes 4, 5, 6, 7) has completed. The state contains:

```python
# From Node 5 (Sentiment Analysis) — THE MAIN INPUT
state['sentiment_analysis'] = {
    'sentiment_signal': 'BUY',
    'combined_sentiment_score': 0.35,        # -1.0 to +1.0
    'confidence': 0.72,                       # 0.0 to 1.0
    'sentiment_label': 'positive',
    'stock_sentiment': {
        'average_sentiment': 0.45,
        'article_count': 8,
        'signal': 'BUY',
        'confidence': 0.75
    },
    'market_sentiment': {...},
    'related_sentiment': {...},
    'raw_sentiment_scores': [                 # per-article breakdown
        {
            'title': 'NVDA beats earnings',
            'source': 'Bloomberg',             # FLAT STRING
            'sentiment_score': 0.45,
            'sentiment_label': 'positive',
            'type': 'stock',
            'credibility_weight': 0.93,        # from Node 5 upgrade
            'source_credibility_score': 0.95,  # from Node 9A
            'composite_anomaly_score': 0.045,  # from Node 9A
            'relevance_score': 0.85            # from Alpha Vantage
        },
        # ... more articles
    ],
    'credibility_summary': {                   # from Node 5 upgrade
        'avg_source_credibility': 0.78,
        'high_credibility_articles': 8,
        'medium_credibility_articles': 5,
        'low_credibility_articles': 2,
        'credibility_weighted': True
    }
}

# From Node 9A — cleaned news lists (articles with scores attached)
state['cleaned_stock_news'] = [...]    # articles with 9A scores embedded
state['cleaned_market_news'] = [...]
state['cleaned_related_company_news'] = [...]

# Basic state fields
state['ticker'] = 'NVDA'
state['analysis_date'] = '2026-02-14'  # set during graph initialization
state['errors'] = []
state['node_execution_times'] = {}
```

---

## What Node 8 Reads From Database

Node 8 queries the database for HISTORICAL data (not today's data). This is the learning dataset:

```sql
-- news_articles table joined with news_outcomes table
-- Each row = one historical article + what happened to the price after

SELECT
    n.id,
    n.ticker,
    n.news_type,           -- 'stock', 'market', 'related'
    n.title,
    n.source,              -- flat string: 'Bloomberg', 'Reuters', etc.
    n.published_at,        -- ISO datetime string
    n.sentiment_label,     -- 'positive', 'negative', 'neutral'
    n.sentiment_score,     -- -1.0 to +1.0
    no.price_at_news,      -- price when article was published
    no.price_1day_later,
    no.price_3day_later,
    no.price_7day_later,
    no.price_change_1day,  -- % change
    no.price_change_3day,
    no.price_change_7day,
    no.prediction_was_accurate_7day,  -- boolean
    no.actual_direction    -- 'UP', 'DOWN', 'FLAT'
FROM news_articles n
JOIN news_outcomes no ON n.id = no.news_id
WHERE n.ticker = ?
AND n.published_at >= date('now', '-180 days')
ORDER BY n.published_at DESC
```

**IMPORTANT:** This query must be implemented as a function in `src/database/db_manager.py`, NOT inside Node 8.

---

## db_manager Functions Needed

Add these functions to `src/database/db_manager.py` BEFORE building Node 8:

### 1. `get_news_with_outcomes(ticker, days_back=180)`
Runs the query above. Returns list of dicts.

### 2. `save_source_reliability(ticker, source_reliability, analysis_date)`
Saves the calculated source reliability scores to the `source_reliability` table.

```sql
INSERT OR REPLACE INTO source_reliability
(ticker, source_name, analysis_date, total_articles,
 accurate_predictions, accuracy_rate, avg_price_impact,
 confidence_multiplier)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
```

### 3. `get_news_outcomes_pending(ticker, limit=100)`
For the background task. Gets news articles that are 7+ days old and don't have outcomes yet.

### 4. `save_news_outcome(news_id, ticker, outcome_data)`
Saves a single news outcome record.

### 5. `get_price_on_date(ticker, date)`
Gets the closing price on a specific date (or nearest trading day).

### 6. `get_price_after_days(ticker, date, days)`
Gets the closing price N days after a given date (finds nearest trading day).

---

## Implementation Plan

### Function 1: `calculate_source_reliability(news_events, ticker)`

**Purpose:** For each news source, calculate how often its sentiment correctly predicted price movement.

**Input:** List of historical news events with outcomes (from db_manager)

**Logic:**
1. Group events by source name
2. For each source, count total articles and accurate predictions
3. Calculate accuracy_rate = accurate / total
4. Calculate confidence_multiplier:
   - accuracy >= 0.80 → multiplier = 1.0 + (accuracy - 0.80) × 2 (range: 1.0 to 1.4)
   - accuracy >= 0.50 → multiplier = 1.0 (neutral)
   - accuracy < 0.50 → multiplier = 0.5 + accuracy (range: 0.5 to 1.0)
5. Calculate avg_price_impact = mean of absolute 7-day price changes

**Output:**
```python
{
    'Bloomberg': {
        'total_articles': 45,
        'accurate_predictions': 38,
        'accuracy_rate': 0.844,
        'avg_price_impact': 2.3,
        'confidence_multiplier': 1.09
    },
    'random-blog.com': {
        'total_articles': 12,
        'accurate_predictions': 4,
        'accuracy_rate': 0.333,
        'avg_price_impact': 1.1,
        'confidence_multiplier': 0.833
    }
}
```

**Important:** The source name format must match what's in the database. Alpha Vantage stores source as a flat string like "Bloomberg", "Motley Fool", "CNBC". Node 9A's articles also have `source` as a flat string. Make sure the matching is case-insensitive or normalized.

---

### Function 2: `calculate_news_type_effectiveness(news_events, ticker)`

**Purpose:** Calculate how effective each news TYPE is at predicting price movement.

**Input:** Same historical events list

**Logic:**
1. Group events by `news_type` field ('stock', 'market', 'related')
2. For each type, count total and accurate predictions
3. Calculate accuracy_rate and avg_price_impact

**Output:**
```python
{
    'stock': {'accuracy_rate': 0.68, 'avg_impact': 3.2, 'sample_size': 45},
    'market': {'accuracy_rate': 0.52, 'avg_impact': 1.1, 'sample_size': 30},
    'related': {'accuracy_rate': 0.61, 'avg_impact': 1.8, 'sample_size': 25}
}
```

---

### Function 3: `calculate_historical_correlation(news_events)`

**Purpose:** Calculate overall correlation between news sentiment and price movement for this stock.

**Input:** Historical events list
**Minimum sample:** 10 events. If fewer, return 0.5 (neutral).

**Logic:**
1. Convert sentiment_label to numeric: positive=1, negative=-1, neutral=0
2. Pair with price_change_7day values
3. Calculate Pearson correlation using pandas
4. Normalize to 0-1 scale: (correlation + 1) / 2
   - 0.0 = perfect negative (positive news → price drops)
   - 0.5 = no correlation
   - 1.0 = perfect positive (positive news → price rises)

**Output:** Float between 0.0 and 1.0

---

### Function 4: `adjust_current_sentiment_confidence(current_sentiment, source_reliability, news_type_effectiveness, today_articles)`

**Purpose:** Adjust Node 5's sentiment confidence based on what we learned from history.

**Input:**
- `current_sentiment`: dict from state['sentiment_analysis']
- `source_reliability`: output from Function 1
- `news_type_effectiveness`: output from Function 2
- `today_articles`: list from state['cleaned_stock_news'] + market + related

**Logic:**
1. For each of today's articles, look up its source in source_reliability
2. Get the confidence_multiplier for that source
3. Calculate the average multiplier across all matched articles
4. If no articles match any known source, use 1.0 (no adjustment)

**Source name matching fix:** Today's articles store the source name as:
```python
article.get('source', 'unknown')  # flat string
```
NOT as `article.get('source', {}).get('name', 'unknown')` (nested dict — this is what the reference file does INCORRECTLY on line 268).

Always match using the flat string. Normalize both sides: strip whitespace, apply consistent casing.

```python
# ✅ CORRECT matching:
source_name = str(article.get('source', 'unknown')).strip()

# Then look up in reliability dict (try case-insensitive):
matched_source = None
for known_source in source_reliability:
    if known_source.lower() == source_name.lower():
        matched_source = known_source
        break
```

5. Calculate news type effectiveness factor:
   - Instead of averaging all types equally (reference file's lazy approach on line 279), weight by how many articles of each type are in today's news:
   ```python
   type_counts = {'stock': len(stock_news), 'market': len(market_news), 'related': len(related_news)}
   total = sum(type_counts.values())
   
   if total > 0:
       weighted_effectiveness = sum(
           news_type_effectiveness[t]['accuracy_rate'] * (type_counts[t] / total)
           for t in type_counts if t in news_type_effectiveness
       )
   else:
       weighted_effectiveness = 0.5
   
   effectiveness_factor = weighted_effectiveness * 2  # Scale to ~1.0
   ```

6. Final formula:
   ```python
   adjusted_confidence = original_confidence * avg_reliability_multiplier * effectiveness_factor
   adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
   ```

**Output:** Updated sentiment dict with adjusted confidence + details:
```python
{
    # All original sentiment fields preserved
    'confidence': 0.78,  # adjusted from 0.72
    'confidence_adjustment': {
        'original': 0.72,
        'reliability_multiplier': 1.05,
        'effectiveness_factor': 1.03,
        'final': 0.78,
        'sources_matched': 6,
        'sources_unmatched': 2
    }
}
```

---

### Function 5: `news_verification_node(state)` — Main Node Function

**Purpose:** Orchestrate all learning functions and update state.

**Steps:**

1. Start timer
2. Get ticker from state
3. Get current sentiment from state (from Node 5)
   - If no sentiment available, return early with neutral defaults
4. Get cleaned news lists from state (from Node 9A)
5. **Query database** for historical news with outcomes (via db_manager)
   - If fewer than 10 historical events, return `insufficient_data: True` with neutral defaults
6. Calculate source reliability (Function 1)
7. Calculate news type effectiveness (Function 2)
8. Calculate historical correlation (Function 3)
9. Adjust current sentiment confidence (Function 4)
10. Calculate overall news accuracy score (weighted average across all sources)
11. Calculate learning_adjustment factor for Node 11:
    ```python
    learning_adjustment = (news_accuracy_score / 50.0) * historical_correlation * 2
    learning_adjustment = max(0.5, min(2.0, learning_adjustment))
    ```
12. Build results dict
13. Save source reliability to database (via db_manager)
14. Update state:
    - `state['sentiment_analysis']` = adjusted sentiment (overwrites Node 5's version)
    - `state['news_impact_verification']` = full results dict
15. Record execution time

**Insufficient data handling:** If fewer than 10 historical events exist (new stock, fresh database), return:
```python
state['news_impact_verification'] = {
    'historical_correlation': 0.5,
    'news_accuracy_score': 50.0,
    'verified_signal_strength': current_sentiment.get('confidence', 0.5),
    'learning_adjustment': 1.0,
    'sample_size': len(historical_events),
    'source_reliability': {},
    'news_type_effectiveness': {},
    'insufficient_data': True
}
```
Do NOT adjust sentiment confidence. Leave Node 5's output untouched.

**Error handling:** If anything crashes, log the error, add to state['errors'], and return neutral defaults (same as insufficient data but with an 'error' field). Never crash the pipeline.

---

## Complete Output Structure

```python
state['news_impact_verification'] = {
    'historical_correlation': 0.72,          # 0-1
    'news_accuracy_score': 68.0,             # 0-100
    'verified_signal_strength': 0.78,        # adjusted confidence
    'learning_adjustment': 1.2,              # 0.5-2.0, for Node 11
    'sample_size': 350,                      # historical articles used
    'source_reliability': {
        'Bloomberg': {
            'total_articles': 45,
            'accurate_predictions': 38,
            'accuracy_rate': 0.844,
            'avg_price_impact': 2.3,
            'confidence_multiplier': 1.09
        },
        # ... more sources
    },
    'news_type_effectiveness': {
        'stock': {'accuracy_rate': 0.68, 'avg_impact': 3.2, 'sample_size': 45},
        'market': {'accuracy_rate': 0.52, 'avg_impact': 1.1, 'sample_size': 30},
        'related': {'accuracy_rate': 0.61, 'avg_impact': 1.8, 'sample_size': 25}
    },
    'confidence_adjustment_details': {
        'original': 0.72,
        'reliability_multiplier': 1.05,
        'effectiveness_factor': 1.03,
        'final': 0.78,
        'sources_matched': 6,
        'sources_unmatched': 2
    },
    'insufficient_data': False
}

# Also updates sentiment_analysis in state:
state['sentiment_analysis']['confidence'] = 0.78  # adjusted
state['sentiment_analysis']['confidence_adjustment'] = {...}  # details
```

---

## Background Task: `scripts/update_news_outcomes.py`

This is a SEPARATE script, not part of Node 8's execution. It runs daily (cron job or manual) to build the historical learning dataset.

**What it does:**
1. Query db_manager for news articles that are 7+ days old and don't have outcomes yet
2. For each article:
   - Get the closing price on the day the article was published
   - Get the closing price 1, 3, and 7 days later
   - Calculate % price changes
   - Determine actual direction: UP if change > 0.5%, DOWN if < -0.5%, else FLAT
   - Determine predicted direction from sentiment_label: positive→UP, negative→DOWN, neutral→FLAT
   - Check if prediction was accurate (predicted == actual)
   - Save to news_outcomes table via db_manager

**Important:** This script also uses db_manager, not direct sqlite3 calls.

**This script must be built alongside Node 8.** Without it, the news_outcomes table stays empty and Node 8 always returns `insufficient_data: True`.

---

## Database Tables Required

These tables must exist in the schema. If they don't, add them to `src/database/schema.sql`:

### news_outcomes
```sql
CREATE TABLE IF NOT EXISTS news_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    price_at_news REAL,
    price_1day_later REAL,
    price_3day_later REAL,
    price_7day_later REAL,
    price_change_1day REAL,
    price_change_3day REAL,
    price_change_7day REAL,
    predicted_direction TEXT,       -- 'UP', 'DOWN', 'FLAT'
    actual_direction TEXT,          -- 'UP', 'DOWN', 'FLAT'
    prediction_was_accurate_7day BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (news_id) REFERENCES news_articles(id),
    UNIQUE(news_id)
);
```

### source_reliability
```sql
CREATE TABLE IF NOT EXISTS source_reliability (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    source_name TEXT NOT NULL,
    analysis_date TEXT NOT NULL,
    total_articles INTEGER,
    accurate_predictions INTEGER,
    accuracy_rate REAL,
    avg_price_impact REAL,
    confidence_multiplier REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, source_name, analysis_date)
);
```

### news_articles table must include these columns
Verify the existing `news_articles` table has:
- `news_type` (TEXT) — 'stock', 'market', 'related'
- `source` (TEXT) — flat string source name
- `sentiment_label` (TEXT) — 'positive', 'negative', 'neutral'
- `sentiment_score` (REAL) — -1.0 to +1.0

If `is_filtered` column exists in the schema, it should be ignored. Node 9A does not filter articles — it scores them. The historical query should NOT filter by `is_filtered`. Remove that condition from the query.

---

## Fixes From Reference File

The reference implementation (`node_08_news_verification_COMPLETE.py`) is useful for understanding the algorithm but has issues. Here's what to fix:

### Fix 1: No Direct Database Access
Reference uses `sqlite3.connect()` directly (lines 43, 363, 565). Use db_manager functions instead.

### Fix 2: Source Name Matching
Reference line 268 does:
```python
source = article.get('source', {}).get('name', 'unknown')
```
This assumes a nested dict. Alpha Vantage and Node 9A store source as a flat string. Use:
```python
source = str(article.get('source', 'unknown')).strip()
```

### Fix 3: News Type Effectiveness Averaging
Reference lines 277-281 averages all news types equally regardless of how many articles of each type exist today. Weight by actual article count instead (described in Function 4 above).

### Fix 4: `is_filtered` Column
Reference queries filter by `n.is_filtered = 0` (line 68). Since Node 9A does NOT filter articles (it only scores them), this column may not exist or may always be 0. Remove this filter condition from the query. Instead, optionally use the `composite_anomaly_score` if it's stored in the database — but only if Node 2 saves it when caching articles.

### Fix 5: `analysis_date` Reference
Reference line 518 uses `state['analysis_date']`. Make sure this field exists in the state. It should be set during graph initialization. If it doesn't exist, use `datetime.now().strftime('%Y-%m-%d')` as fallback.

### Fix 6: Missing `credibility_weight` Integration
The reference file doesn't use Node 9A's credibility scores at all. The upgraded Node 5 now provides `credibility_weight` per article and `credibility_summary`. Node 8 can optionally use `source_credibility_score` from Node 9A to supplement its own learned reliability — but this is a nice-to-have, not a requirement for the first build.

---

## Testing Strategy

### Test 1: Sufficient Historical Data
```python
# Mock 100+ historical events with outcomes
# Verify source reliability calculated correctly
# Verify confidence adjusted
state = build_mock_state_with_sentiment()
# Mock db_manager to return 100 historical events
result = news_verification_node(state)
assert result['news_impact_verification']['insufficient_data'] == False
assert len(result['news_impact_verification']['source_reliability']) > 0
assert result['news_impact_verification']['sample_size'] >= 100
```

### Test 2: Insufficient Historical Data
```python
# Mock fewer than 10 historical events
result = news_verification_node(state)
assert result['news_impact_verification']['insufficient_data'] == True
assert result['news_impact_verification']['learning_adjustment'] == 1.0
# Sentiment confidence should NOT be adjusted
assert result['sentiment_analysis']['confidence'] == original_confidence
```

### Test 3: Source Reliability Calculation
```python
# Create mock events: Bloomberg 8/10 correct, random blog 2/10 correct
events = create_mock_events('Bloomberg', 10, 8) + create_mock_events('random-blog', 10, 2)
reliability = calculate_source_reliability(events, 'NVDA')
assert reliability['Bloomberg']['accuracy_rate'] == 0.8
assert reliability['random-blog']['accuracy_rate'] == 0.2
assert reliability['Bloomberg']['confidence_multiplier'] > 1.0
assert reliability['random-blog']['confidence_multiplier'] < 1.0
```

### Test 4: Confidence Adjustment Direction
```python
# High-credibility sources in today's news → confidence should increase
# Low-credibility sources in today's news → confidence should decrease
result_high = adjust_with_reliable_sources(original_confidence=0.7)
result_low = adjust_with_unreliable_sources(original_confidence=0.7)
assert result_high > 0.7
assert result_low < 0.7
```

### Test 5: Source Name Matching
```python
# Verify matching works with various source formats
# 'Bloomberg' should match 'Bloomberg' in reliability dict
# 'bloomberg' (lowercase) should also match
# 'Unknown Source' should not crash, just skip
```

### Test 6: News Type Effectiveness
```python
events = create_mock_events_by_type('stock', 20, 14) + \
         create_mock_events_by_type('market', 20, 10)
effectiveness = calculate_news_type_effectiveness(events, 'NVDA')
assert effectiveness['stock']['accuracy_rate'] == 0.7
assert effectiveness['market']['accuracy_rate'] == 0.5
assert effectiveness['stock']['accuracy_rate'] > effectiveness['market']['accuracy_rate']
```

### Test 7: Historical Correlation
```python
# All positive sentiment + all positive price changes → high correlation
# Random mix → correlation near 0.5
high_corr = calculate_historical_correlation(perfectly_correlated_events)
assert high_corr > 0.7
random_corr = calculate_historical_correlation(random_events)
assert 0.3 < random_corr < 0.7
```

### Test 8: Error Handling
```python
# Database unavailable → should not crash
# Missing sentiment in state → should return early with defaults
# Malformed historical data → should skip bad records
```

### Test 9: Learning Adjustment Bounds
```python
# Verify learning_adjustment is always between 0.5 and 2.0
result = news_verification_node(state)
adj = result['news_impact_verification']['learning_adjustment']
assert 0.5 <= adj <= 2.0
```

### Test 10: Background Task
```python
# Verify update_news_outcomes correctly evaluates pending articles
# Mock a news article from 10 days ago with known prices
# Run the task
# Verify outcome record created with correct direction and accuracy
```

---

## Success Criteria

- [ ] All database access goes through db_manager (no direct sqlite3)
- [ ] Uses project logger (get_node_logger)
- [ ] Queries historical data successfully
- [ ] Calculates source reliability per source
- [ ] Calculates news type effectiveness per type
- [ ] Calculates historical correlation
- [ ] Adjusts sentiment confidence based on source reliability
- [ ] Source name matching works (flat string, case-insensitive)
- [ ] Handles insufficient data gracefully (< 10 events → neutral defaults)
- [ ] Handles missing sentiment in state (returns early)
- [ ] Handles database errors (logs + neutral defaults)
- [ ] learning_adjustment always between 0.5 and 2.0
- [ ] Saves source reliability to database
- [ ] Background task script created and functional
- [ ] news_outcomes and source_reliability tables exist in schema
- [ ] All tests pass
- [ ] Execution time < 2 seconds
- [ ] State output matches expected structure

---

## Files to Create

1. `src/langgraph_nodes/node_08_news_verification.py` — main node
2. `scripts/update_news_outcomes.py` — background task

## Files to Modify

3. `src/database/db_manager.py` — add 6 new functions listed above
4. `src/database/schema.sql` — add news_outcomes and source_reliability tables (if not present)

## Files NOT to Modify

- Node 5, Node 9A, or any other node
- workflow.py (Node 8 is already in the workflow as sequential after parallel layer)
- Any config files

## Reference Files (Read, Don't Copy)

- `LangGraph_setup/node_08_news_verification_COMPLETE.py` — algorithm reference
- `LangGraph_setup/NEWS_LEARNING_SYSTEM_GUIDE.md` — conceptual explanation
