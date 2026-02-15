# Background Script - Quick Start Guide

## What Was Just Built ✅

### 1. Database Functions (4 new functions in `db_manager.py`)
- ✅ `get_news_outcomes_pending()` - Finds articles needing evaluation
- ✅ `get_price_on_date()` - Gets price on specific date
- ✅ `get_price_after_days()` - Gets price N days later
- ✅ `save_news_outcome()` - Saves evaluation results

### 2. Sentiment Caching Fixed
- ✅ Node 2 now saves Alpha Vantage sentiment data (`sentiment_label`, `sentiment_score`)
- ✅ Articles have sentiment available immediately after fetch
- ✅ Normalized labels: Bullish/Bearish → positive/negative/neutral

### 3. Background Script (`scripts/update_news_outcomes.py`)
- ✅ Evaluates historical news predictions vs actual price movements
- ✅ Populates `news_outcomes` table for Node 8 learning
- ✅ CLI interface with `--ticker`, `--limit`, `--quiet` options
- ✅ 23 tests, all passing

---

## How To Use It

### Step 1: Fetch Historical Data (if not done yet)

```bash
# Activate virtual environment
source venv/bin/activate

# Fetch 6 months of price data
python -m src.langgraph_nodes.node_01_price_data --ticker NVDA

# Fetch news with sentiment (Node 2 now saves sentiment_label)
python -m src.langgraph_nodes.node_02_news_fetching --ticker NVDA
```

### Step 2: Run Background Script (First Time)

```bash
# Backfill outcomes for all articles 7+ days old
python -m scripts.update_news_outcomes --ticker NVDA --limit 500
```

**Expected Output:**
```
INFO: Starting news outcomes evaluation for NVDA (limit: 500)
INFO: Found 347 articles needing evaluation
INFO: Evaluation complete in 2.45s:
INFO:   Evaluated: 312
INFO:   Skipped: 35 (missing price data)
INFO:   Accurate predictions: 194/312 (62.2%)
```

### Step 3: Verify Data in Database

```bash
sqlite3 data/stock_prices.db
```

```sql
-- Check how many outcomes were saved
SELECT COUNT(*) FROM news_outcomes WHERE ticker='NVDA';

-- Check accuracy by source
SELECT 
    n.source, 
    COUNT(*) as total,
    SUM(CASE WHEN no.prediction_was_accurate_7day THEN 1 ELSE 0 END) as correct,
    ROUND(AVG(CASE WHEN no.prediction_was_accurate_7day THEN 100.0 ELSE 0.0 END), 1) as accuracy_pct
FROM news_articles n
JOIN news_outcomes no ON n.id = no.news_id
WHERE n.ticker = 'NVDA'
GROUP BY n.source
ORDER BY accuracy_pct DESC;

-- Check the news_with_outcomes VIEW (used by Node 8)
SELECT COUNT(*) FROM news_with_outcomes WHERE ticker='NVDA';
```

### Step 4: Test Node 8 (if applicable)

```bash
# Node 8 should now have historical data to learn from
python -m scripts.validate_node_08
```

---

## Understanding the Output

### Accuracy Metrics:
- **~60-65%** - Raw sentiment accuracy (expected without learning)
- **~70-75%** - Expected accuracy after Node 8 adjustments
- **Skipped articles** - Missing price data (holidays, weekends, data gaps)

### What Gets Saved:
For each article, the script saves:
- Price at publication time
- Prices 1, 3, 7 days later
- % price changes
- Predicted direction (from sentiment: UP/DOWN/FLAT)
- Actual direction (from price movement: UP/DOWN/FLAT)
- Was prediction accurate? (TRUE/FALSE)

### Direction Thresholds:
- **UP**: price change > +0.5%
- **DOWN**: price change < -0.5%
- **FLAT**: price change within ±0.5%

---

## Daily Usage (Ongoing)

Once historical data is backfilled, run daily to evaluate newly-aged articles:

```bash
# Run daily (manual or via cron)
python -m scripts.update_news_outcomes --ticker NVDA
```

Or for all tickers:

```bash
python -m scripts.update_news_outcomes --limit 1000
```

---

## CLI Options

```bash
# All tickers, default limit (500)
python -m scripts.update_news_outcomes

# Single ticker
python -m scripts.update_news_outcomes --ticker NVDA

# Larger batch
python -m scripts.update_news_outcomes --limit 1000

# Quiet mode (minimal output)
python -m scripts.update_news_outcomes --quiet

# Help
python -m scripts.update_news_outcomes --help
```

---

## Key Files Modified

### Modified:
1. **`src/database/db_manager.py`**
   - Added 4 functions for outcomes evaluation
   - Fixed `cache_news()` to save sentiment data
   - Added `Tuple` import

### Created:
1. **`scripts/update_news_outcomes.py`** - Main background script
2. **`tests/test_scripts/test_update_news_outcomes.py`** - 23 comprehensive tests
3. **`BACKGROUND_SCRIPT_IMPLEMENTATION_SUMMARY.md`** - Full documentation

---

## Integration with Node 8

Node 8 uses the `news_with_outcomes` VIEW to:

1. **Calculate Source Reliability**
   - Groups outcomes by source
   - Calculates accuracy rate per source
   - High-accuracy sources (>80%) boost confidence
   - Low-accuracy sources (<60%) reduce confidence

2. **Calculate News Type Effectiveness**
   - Groups outcomes by news_type ('stock', 'market', 'related')
   - Measures which types predict best
   - Weights current articles by type performance

3. **Calculate Historical Correlation**
   - Pearson correlation: sentiment_score vs price_change_7day
   - Overall effectiveness metric
   - Ranges from 0.0 (no correlation) to 1.0 (perfect)

4. **Produce Learning Adjustment**
   - Combines all 3 metrics
   - Outputs `learning_adjustment` factor (0.5 - 2.0)
   - Adjusts current sentiment confidence scores

---

## Troubleshooting

### "No pending articles to evaluate"
- Articles must be 7+ days old
- Check if you have recent news in `news_articles` table
- Run Nodes 1-2 to fetch data if needed

### "Skipped: X articles (missing price data)"
- Normal for holidays, weekends, data gaps
- Price data from Node 1 may not cover all dates
- Articles will be skipped but script continues

### "NameError: name 'Tuple' is not defined"
- Already fixed! `Tuple` added to imports in `db_manager.py`

### High Skip Rate (>50%)
- Check if Node 1 price data covers the date range
- Verify `price_data` table has sufficient history
- Run: `SELECT MIN(date), MAX(date) FROM price_data WHERE ticker='NVDA';`

---

## What's Next

1. ✅ **Background script is complete and tested**
2. ✅ **Database functions are implemented**
3. ✅ **Sentiment caching is fixed**
4. 🔄 **Ready to test Node 8 with real historical data**
5. 🔄 **Ready to run full pipeline validation**

---

## Need Help?

- **Full Documentation:** `BACKGROUND_SCRIPT_IMPLEMENTATION_SUMMARY.md`
- **Test Coverage:** Run `pytest tests/test_scripts/test_update_news_outcomes.py -v`
- **Database Schema:** Check `src/database/schema.sql` for table definitions
- **Node 8 Integration:** See `NODE_08_IMPLEMENTATION_SUMMARY.md`

---

**Status: ✅ COMPLETE - All components built, tested, and ready to use!**
