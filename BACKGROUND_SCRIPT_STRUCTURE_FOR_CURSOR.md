# Build: Background Script — News Outcomes Evaluator
## `scripts/update_news_outcomes.py`

**Estimated Time:** 2-3 hours
**This is NOT a node.** It's a standalone script that runs separately from the pipeline.

---

## What This Script Does — In Plain English

Every news article has a sentiment prediction (positive/negative/neutral). This script checks if that prediction was correct by looking at what the stock price actually did 7 days later.

```
Input:  news_articles table (articles with sentiment, from Node 2)
      + price_data table (historical prices, from Node 1)

Output: news_outcomes table (was each article's prediction correct?)
```

Example of what it produces for ONE article:

```
Article: "NVDA earnings beat expectations"
  Published: September 15, 2025
  Sentiment: positive
  
  Price on Sept 15: $135.00
  Price on Sept 16 (1 day later): $136.20 → +0.89%
  Price on Sept 18 (3 days later): $137.80 → +2.07%
  Price on Sept 22 (7 days later): $139.50 → +3.33%
  
  Predicted direction: UP (because sentiment was positive)
  Actual direction: UP (because price went up more than 0.5%)
  Prediction was accurate: TRUE ✅
```

It does this for EVERY article that is 7+ days old and hasn't been evaluated yet.

---

## When To Run This Script

**First time (today):** Run it after Nodes 1-2 have fetched 6 months of historical data. It will backfill hundreds of outcomes at once since most articles are already months old.

**Ongoing:** Run daily (manually or via cron). It picks up any new articles that have aged past 7 days and evaluates them.

**Before testing Node 8:** Node 8 reads from `news_outcomes`. If this table is empty, Node 8 has nothing to learn from.

---

## How It Works

### Step 1: Find Articles Needing Evaluation

Query the database for articles that:
- Are 7+ days old (enough time has passed to check the outcome)
- Don't already have an outcome in `news_outcomes` (haven't been evaluated yet)

```sql
SELECT n.id, n.ticker, n.published_at, n.sentiment_label, n.sentiment_score
FROM news_articles n
LEFT JOIN news_outcomes no ON n.id = no.news_id
WHERE no.id IS NULL
AND n.published_at <= date('now', '-7 days')
ORDER BY n.published_at ASC
```

### Step 2: For Each Article, Look Up Prices

For each pending article:
1. Get the closing price on the day the article was published
2. Get the closing price 1 trading day later
3. Get the closing price 3 trading days later
4. Get the closing price 7 trading days later

**Important — Trading Days vs Calendar Days:**
Markets are closed on weekends and holidays. If an article was published on Friday, "1 day later" is Monday, not Saturday. The query should find the NEXT available trading day:

```sql
-- Get price on or after a specific date (finds next trading day)
SELECT close, date FROM price_data
WHERE ticker = ? AND date >= ?
ORDER BY date ASC
LIMIT 1
```

**If no price data exists** for a particular date (stock was halted, data gap, etc.), skip that article and move on. Don't crash.

### Step 3: Calculate Price Changes

```python
price_change_1day = ((price_1day - price_at_news) / price_at_news) * 100
price_change_3day = ((price_3day - price_at_news) / price_at_news) * 100
price_change_7day = ((price_7day - price_at_news) / price_at_news) * 100
```

### Step 4: Determine Directions

**Actual direction** (what the price actually did):
```python
if price_change_7day > 0.5:     # went up more than 0.5%
    actual_direction = 'UP'
elif price_change_7day < -0.5:  # went down more than 0.5%
    actual_direction = 'DOWN'
else:                           # stayed within ±0.5%
    actual_direction = 'FLAT'
```

**Predicted direction** (what the sentiment implied):
```python
if sentiment_label == 'positive' or sentiment_label == 'Bullish' or sentiment_label == 'Somewhat-Bullish':
    predicted_direction = 'UP'
elif sentiment_label == 'negative' or sentiment_label == 'Bearish' or sentiment_label == 'Somewhat-Bearish':
    predicted_direction = 'DOWN'
else:  # neutral
    predicted_direction = 'FLAT'
```

**IMPORTANT — Alpha Vantage Sentiment Labels:**
Alpha Vantage uses labels like `Bullish`, `Somewhat-Bullish`, `Neutral`, `Somewhat-Bearish`, `Bearish`. FinBERT uses `positive`, `negative`, `neutral`. The script must handle BOTH formats. Check what your Node 2 / Node 5 actually stores in the `sentiment_label` column of `news_articles`.

### Step 5: Check If Prediction Was Accurate

```python
prediction_accurate = (predicted_direction == actual_direction)
```

Simple: if the predicted direction matches the actual direction, it was accurate.

### Step 6: Save to Database

```sql
INSERT INTO news_outcomes
(news_id, ticker, price_at_news, price_1day_later, price_3day_later,
 price_7day_later, price_change_1day, price_change_3day, price_change_7day,
 predicted_direction, actual_direction, prediction_was_accurate_7day)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

---

## Project Rules

### Rule 1: Use db_manager for ALL Database Access

```python
# ❌ WRONG:
import sqlite3
conn = sqlite3.connect('data/stock_prices.db')

# ✅ CORRECT:
from src.database.db_manager import (
    get_news_outcomes_pending,
    get_price_on_date,
    get_price_after_days,
    save_news_outcome
)
```

### Rule 2: These db_manager Functions Need To Be Created First

Before building this script, add these to `src/database/db_manager.py`:

**1. `get_news_outcomes_pending(ticker=None, limit=500)`**
```python
def get_news_outcomes_pending(ticker=None, limit=500, db_path=DEFAULT_DB_PATH):
    """
    Get news articles that are 7+ days old and don't have outcomes yet.
    
    Args:
        ticker: Optional - if None, get for ALL tickers
        limit: Max articles to process per run
    
    Returns:
        List of dicts with: id, ticker, published_at, sentiment_label, sentiment_score
    """
```
If ticker is None, process all tickers. This way one script run can backfill everything.

**2. `get_price_on_date(ticker, date)`**
```python
def get_price_on_date(ticker, date, db_path=DEFAULT_DB_PATH):
    """
    Get closing price on a specific date.
    If no exact match (weekend/holiday), returns nearest PREVIOUS trading day.
    
    Returns: float (closing price) or None if no data
    """
```

**3. `get_price_after_days(ticker, date, days)`**
```python
def get_price_after_days(ticker, date, days, db_path=DEFAULT_DB_PATH):
    """
    Get closing price N calendar days after a given date.
    Finds the nearest trading day on or after (date + days).
    
    Returns: tuple (price, actual_date) or (None, None) if no data
    """
```

**4. `save_news_outcome(outcome_data)`**
```python
def save_news_outcome(outcome_data, db_path=DEFAULT_DB_PATH):
    """
    Save a single news outcome record.
    
    Args:
        outcome_data: dict with keys:
            news_id, ticker, price_at_news, price_1day_later,
            price_3day_later, price_7day_later, price_change_1day,
            price_change_3day, price_change_7day, predicted_direction,
            actual_direction, prediction_was_accurate_7day
    """
```

### Rule 3: Use the Logger

```python
from src.utils.logger import get_node_logger
logger = get_node_logger("update_news_outcomes")
```

---

## Script Structure

```python
"""
Background Script: News Outcomes Evaluator

Evaluates historical news articles to determine if their sentiment
predictions were accurate. Fills the news_outcomes table that Node 8
uses for learning.

Usage:
    python -m scripts.update_news_outcomes              # All tickers
    python -m scripts.update_news_outcomes --ticker NVDA # Single ticker
    python -m scripts.update_news_outcomes --limit 1000  # Custom limit
"""

import argparse
from datetime import datetime, timedelta
from src.database.db_manager import (
    get_news_outcomes_pending,
    get_price_on_date,
    get_price_after_days,
    save_news_outcome
)
from src.utils.logger import get_node_logger

logger = get_node_logger("update_news_outcomes")


def determine_predicted_direction(sentiment_label: str) -> str:
    """
    Convert sentiment label to predicted price direction.
    Handles both Alpha Vantage and FinBERT label formats.
    """
    if not sentiment_label:
        return 'FLAT'
    
    label = sentiment_label.lower().strip()
    
    # Alpha Vantage format
    if label in ['bullish', 'somewhat-bullish', 'somewhat_bullish']:
        return 'UP'
    if label in ['bearish', 'somewhat-bearish', 'somewhat_bearish']:
        return 'DOWN'
    
    # FinBERT format
    if label == 'positive':
        return 'UP'
    if label == 'negative':
        return 'DOWN'
    
    # Neutral / unknown
    return 'FLAT'


def determine_actual_direction(price_change_pct: float) -> str:
    """
    Determine actual price direction from % change.
    UP if > +0.5%, DOWN if < -0.5%, else FLAT.
    """
    if price_change_pct > 0.5:
        return 'UP'
    elif price_change_pct < -0.5:
        return 'DOWN'
    return 'FLAT'


def evaluate_single_article(article: dict) -> dict:
    """
    Evaluate a single article's prediction accuracy.
    
    Returns outcome dict or None if price data unavailable.
    """
    ticker = article['ticker']
    published_at = article['published_at']
    
    # Parse published date
    # Handle various date formats from Alpha Vantage
    try:
        if 'T' in str(published_at):
            pub_date = datetime.fromisoformat(str(published_at).replace('Z', '')).date()
        else:
            pub_date = datetime.strptime(str(published_at)[:10], '%Y-%m-%d').date()
    except (ValueError, TypeError) as e:
        logger.warning(f"Cannot parse date '{published_at}' for article {article['id']}: {e}")
        return None
    
    # Get price at time of news
    price_at_news = get_price_on_date(ticker, str(pub_date))
    if price_at_news is None:
        logger.debug(f"No price data for {ticker} on {pub_date}, skipping")
        return None
    
    # Get prices 1, 3, 7 days later
    prices = {}
    for days in [1, 3, 7]:
        result = get_price_after_days(ticker, str(pub_date), days)
        if result and result[0] is not None:
            prices[days] = result[0]  # price value
    
    # Must have 7-day price to evaluate
    if 7 not in prices:
        logger.debug(f"No 7-day price for {ticker} after {pub_date}, skipping")
        return None
    
    # Calculate price changes
    price_change_1day = ((prices[1] - price_at_news) / price_at_news * 100) if 1 in prices else None
    price_change_3day = ((prices[3] - price_at_news) / price_at_news * 100) if 3 in prices else None
    price_change_7day = ((prices[7] - price_at_news) / price_at_news * 100)
    
    # Determine directions
    predicted = determine_predicted_direction(article.get('sentiment_label', ''))
    actual = determine_actual_direction(price_change_7day)
    accurate = (predicted == actual)
    
    return {
        'news_id': article['id'],
        'ticker': ticker,
        'price_at_news': price_at_news,
        'price_1day_later': prices.get(1),
        'price_3day_later': prices.get(3),
        'price_7day_later': prices[7],
        'price_change_1day': price_change_1day,
        'price_change_3day': price_change_3day,
        'price_change_7day': price_change_7day,
        'predicted_direction': predicted,
        'actual_direction': actual,
        'prediction_was_accurate_7day': accurate
    }


def run_evaluation(ticker=None, limit=500):
    """
    Main evaluation function.
    
    Args:
        ticker: Optional - evaluate single ticker, or None for all
        limit: Max articles to process
    """
    logger.info(f"Starting news outcomes evaluation"
                f"{f' for {ticker}' if ticker else ' for all tickers'}"
                f" (limit: {limit})")
    
    # Get pending articles
    pending = get_news_outcomes_pending(ticker=ticker, limit=limit)
    logger.info(f"Found {len(pending)} articles needing evaluation")
    
    if not pending:
        logger.info("No pending articles to evaluate")
        return {'evaluated': 0, 'skipped': 0, 'accurate': 0, 'total': 0}
    
    evaluated = 0
    skipped = 0
    accurate = 0
    
    for article in pending:
        try:
            outcome = evaluate_single_article(article)
            
            if outcome is None:
                skipped += 1
                continue
            
            save_news_outcome(outcome)
            evaluated += 1
            
            if outcome['prediction_was_accurate_7day']:
                accurate += 1
                
        except Exception as e:
            logger.error(f"Failed to evaluate article {article.get('id', '?')}: {e}")
            skipped += 1
            continue
    
    # Summary
    accuracy_pct = (accurate / evaluated * 100) if evaluated > 0 else 0
    logger.info(f"Evaluation complete:")
    logger.info(f"  Evaluated: {evaluated}")
    logger.info(f"  Skipped: {skipped} (missing price data)")
    logger.info(f"  Accurate predictions: {accurate}/{evaluated} ({accuracy_pct:.1f}%)")
    
    return {
        'evaluated': evaluated,
        'skipped': skipped,
        'accurate': accurate,
        'total': len(pending),
        'accuracy_pct': accuracy_pct
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate news prediction outcomes')
    parser.add_argument('--ticker', type=str, default=None,
                        help='Specific ticker to evaluate (default: all)')
    parser.add_argument('--limit', type=int, default=500,
                        help='Max articles to process (default: 500)')
    
    args = parser.parse_args()
    
    results = run_evaluation(ticker=args.ticker, limit=args.limit)
    
    print(f"\nResults: {results['evaluated']} evaluated, "
          f"{results['skipped']} skipped, "
          f"{results['accuracy_pct']:.1f}% accuracy")
```

---

## db_manager Functions to Add First

Add ALL FOUR of these to `src/database/db_manager.py` before building the script:

### 1. get_news_outcomes_pending

```python
def get_news_outcomes_pending(ticker=None, limit=500, db_path=DEFAULT_DB_PATH):
    """Get articles needing outcome evaluation (7+ days old, no outcome yet)."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            
            if ticker:
                cursor.execute("""
                    SELECT n.id, n.ticker, n.published_at, 
                           n.sentiment_label, n.sentiment_score
                    FROM news_articles n
                    LEFT JOIN news_outcomes no ON n.id = no.news_id
                    WHERE n.ticker = ?
                    AND no.id IS NULL
                    AND n.published_at <= date('now', '-7 days')
                    ORDER BY n.published_at ASC
                    LIMIT ?
                """, (ticker, limit))
            else:
                cursor.execute("""
                    SELECT n.id, n.ticker, n.published_at,
                           n.sentiment_label, n.sentiment_score
                    FROM news_articles n
                    LEFT JOIN news_outcomes no ON n.id = no.news_id
                    WHERE no.id IS NULL
                    AND n.published_at <= date('now', '-7 days')
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
            return results
    except Exception as e:
        logger.error(f"Failed to get pending news outcomes: {e}")
        return []
```

### 2. get_price_on_date

```python
def get_price_on_date(ticker, date_str, db_path=DEFAULT_DB_PATH):
    """Get closing price on date (or nearest previous trading day)."""
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT close FROM price_data
                WHERE ticker = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            """, (ticker, date_str))
            row = cursor.fetchone()
            return float(row['close']) if row else None
    except Exception as e:
        logger.error(f"Failed to get price for {ticker} on {date_str}: {e}")
        return None
```

### 3. get_price_after_days

```python
def get_price_after_days(ticker, date_str, days, db_path=DEFAULT_DB_PATH):
    """Get closing price N calendar days after date (nearest trading day)."""
    try:
        from datetime import datetime, timedelta
        base_date = datetime.strptime(date_str[:10], '%Y-%m-%d').date()
        target_date = base_date + timedelta(days=days)
        
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT close, date FROM price_data
                WHERE ticker = ? AND date >= ?
                ORDER BY date ASC
                LIMIT 1
            """, (ticker, str(target_date)))
            row = cursor.fetchone()
            if row:
                return (float(row['close']), row['date'])
            return (None, None)
    except Exception as e:
        logger.error(f"Failed to get price for {ticker} after {days} days from {date_str}: {e}")
        return (None, None)
```

### 4. save_news_outcome

```python
def save_news_outcome(outcome_data, db_path=DEFAULT_DB_PATH):
    """Save a single news outcome evaluation."""
    try:
        with get_connection(db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO news_outcomes
                (news_id, ticker, price_at_news, price_1day_later,
                 price_3day_later, price_7day_later, price_change_1day,
                 price_change_3day, price_change_7day, predicted_direction,
                 actual_direction, prediction_was_accurate_7day)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    except Exception as e:
        logger.error(f"Failed to save news outcome for article {outcome_data.get('news_id')}: {e}")
```

---

## Database Table Required

Verify this exists in `src/database/schema.sql`:

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
    predicted_direction TEXT,
    actual_direction TEXT,
    prediction_was_accurate_7day BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (news_id) REFERENCES news_articles(id),
    UNIQUE(news_id)
);
```

Also verify the VIEW exists:

```sql
CREATE VIEW IF NOT EXISTS news_with_outcomes AS
SELECT 
    n.id, n.ticker, n.news_type, n.title, n.source,
    n.published_at, n.sentiment_label, n.sentiment_score,
    no.price_at_news, no.price_1day_later, no.price_3day_later,
    no.price_7day_later, no.price_change_1day, no.price_change_3day,
    no.price_change_7day, no.predicted_direction, no.actual_direction,
    no.prediction_was_accurate_7day
FROM news_articles n
JOIN news_outcomes no ON n.id = no.news_id;
```

---

## Verify news_articles Table Has Required Columns

The script reads from `news_articles`. Verify these columns exist:
- `id` (INTEGER PRIMARY KEY)
- `ticker` (TEXT)
- `published_at` (TEXT — date/datetime string)
- `sentiment_label` (TEXT — 'positive'/'negative'/'neutral' OR Alpha Vantage labels)
- `sentiment_score` (REAL — -1.0 to +1.0)

If `sentiment_label` isn't stored when Node 2 caches articles, that's a problem. Check what Node 2 actually saves. If it only saves title/url/source, then sentiment_label won't exist yet because Node 5 hasn't run. In that case, Node 5 needs to write sentiment back to the database, OR the background script needs to handle missing sentiment_label gracefully.

---

## Edge Cases to Handle

1. **Article published on weekend:** `get_price_on_date` should find the previous Friday's price
2. **Article published on holiday:** Same — find nearest previous trading day
3. **Price data gap:** If price_data doesn't cover the article's date range, skip the article
4. **Duplicate evaluation:** `UNIQUE(news_id)` constraint + `INSERT OR IGNORE` prevents re-evaluation
5. **Alpha Vantage vs FinBERT labels:** Handle both formats in `determine_predicted_direction`
6. **Very recent articles:** The 7-day filter ensures we only evaluate articles old enough to have outcomes
7. **No sentiment stored:** If `sentiment_label` is NULL, default predicted_direction to 'FLAT'

---

## How To Run

```bash
# First time — backfill all tickers
python -m scripts.update_news_outcomes

# Single ticker
python -m scripts.update_news_outcomes --ticker NVDA

# Larger batch
python -m scripts.update_news_outcomes --ticker NVDA --limit 1000
```

---

## Expected Output

```
INFO: Starting news outcomes evaluation for NVDA (limit: 500)
INFO: Found 347 articles needing evaluation
INFO: Evaluation complete:
INFO:   Evaluated: 312
INFO:   Skipped: 35 (missing price data)
INFO:   Accurate predictions: 194/312 (62.2%)

Results: 312 evaluated, 35 skipped, 62.2% accuracy
```

The ~62% accuracy is expected for raw sentiment without Node 8 learning. After Node 8 adjusts confidence based on source reliability, the effective accuracy should improve to ~73%.

---

## Build Order

1. Add 4 functions to `src/database/db_manager.py`
2. Verify `news_outcomes` table and `news_with_outcomes` VIEW exist in schema
3. Build the script `scripts/update_news_outcomes.py`
4. Test: run Nodes 1-2 for NVDA, then run the script, then check the database

---

## Files to Create
- `scripts/update_news_outcomes.py`

## Files to Modify
- `src/database/db_manager.py` — add 4 functions

## Files to Verify (not modify)
- `src/database/schema.sql` — news_outcomes table + news_with_outcomes VIEW must exist
