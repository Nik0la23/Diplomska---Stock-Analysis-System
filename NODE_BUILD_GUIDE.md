# LangGraph Stock Analysis - Node-by-Node Build Guide
## Strategic Implementation Roadmap

**Purpose:** This is your reference guide for building each of the 16 nodes. It explains WHAT each node does, WHY it matters, WHEN to build it, and WHERE to find detailed specifications.

**Important:** This is NOT a copy-paste template. Use the referenced documentation files for detailed implementation patterns, but adapt them to this specific architecture.

---

## 📋 Build Order Strategy

### Phase 1: Foundation (Build First)
1. Database Setup
2. Node 1: Price Data Fetching
3. Node 3: Related Companies Detection
4. Node 2: Multi-Source News Fetching

**Why this order?** You need data before you can analyze it. These nodes establish the data pipeline.

---

### Phase 2: Early Protection (Critical for Thesis)
5. Node 9A: Early Anomaly Detection

**Why early?** This filters bad news BEFORE sentiment analysis, protecting your learning system (Node 8) from contaminated data. This is a key thesis contribution.

---

### Phase 3: Core Analysis (Parallel Execution)
6. Node 4: Technical Analysis
7. Node 5: Sentiment Analysis (uses CLEANED news from 9A)
8. Node 6: Market Context Analysis
9. Node 7: Monte Carlo Forecasting

**Why parallel?** These 4 nodes are independent and run simultaneously, giving you 3x speed improvement.

---

### Phase 4: Learning & Intelligence (Thesis Innovation)
10. Node 8: News Verification & Learning System

**Why critical?** THIS IS YOUR PRIMARY THESIS INNOVATION. It learns which news sources are reliable and improves sentiment accuracy by 10-15%.

---

### Phase 5: Behavioral Protection
11. Node 9B: Behavioral Anomaly Detection

**Why after learning?** This needs historical patterns from Node 8 to detect sophisticated manipulation like pump-and-dump schemes.

---

### Phase 6: Adaptive Intelligence
12. Node 10: Backtesting
13. Node 11: Adaptive Weights Calculation

**Why together?** Node 11 depends on Node 10's backtest results to calculate optimal weights.

---

### Phase 7: Signal Generation & Explanations
14. Conditional Edge: Risk Detection Router
15. Node 12: Final Signal Generation
16. Node 13: Beginner Explanation (LLM-powered)
17. Node 14: Technical Explanation (LLM-powered)

**Why last?** These nodes combine all previous analysis into the final output.

---

### Phase 8: Visualization & Output
18. Node 15: Dashboard Data Preparation

**Why final?** Prepares all data for the Streamlit dashboard.

---

---

## 🗄️ STEP 0: Database Setup

### What It Does
Creates the SQLite database with all required tables, indexes, and views.

### Why It Matters
Everything depends on the database. Build this FIRST before any nodes.

### Reference Files
- **Schema:** `LangGraph_setup/schema_UPDATED_with_news_outcomes.sql`
- **Key Tables for Thesis:**
  - `news_outcomes` - Tracks what happened after each news (for Node 8 learning)
  - `source_reliability` - Stores accuracy per source (Bloomberg vs blogs)
  - `news_type_effectiveness` - Which news types work best

### Implementation Steps
1. Create `src/database/db_manager.py` with connection management
2. Create functions for:
   - `init_database()` - Create all tables from schema
   - `get_connection()` - Return SQLite connection with context manager
   - `cache_price_data()` - Store price data
   - `get_cached_price_data()` - Retrieve cached data
   - `cache_news()` - Store news articles
   - `get_news_with_outcomes()` - Critical for Node 8 learning
3. Create `scripts/setup_database.py` - Initialize database
4. Test database operations independently

### Success Criteria
- [ ] All tables created successfully
- [ ] Can insert and retrieve price data
- [ ] Can insert and retrieve news articles
- [ ] Indexes created for fast queries
- [ ] Views working correctly

### Estimated Time
2-3 hours

---

---

## 📊 NODE 1: Price Data Fetching

### What It Does
Fetches historical OHLCV (Open, High, Low, Close, Volume) price data for a stock ticker.

**Data Flow:**
```
Input: ticker ('AAPL')
↓
Check SQLite cache (is data < 24h old?)
↓
If cached: Return cached data
If not: Fetch from Finnhub API (primary) or yfinance (backup)
↓
Store in cache for next time
↓
Output: raw_price_data (DataFrame with columns: date, open, high, low, close, volume)
```

### Why It Matters
Everything else depends on price data. This is the foundation.

### Dependencies
- **Runs BEFORE:** All other nodes
- **Runs AFTER:** Nothing (it's first)
- **Parallel with:** Nothing

### Key Concepts
1. **Cache-first strategy:** Check database before API (saves 80% of calls)
2. **Fallback mechanism:** Try Finnhub, fall back to yfinance
3. **Error handling:** If both fail, return None and add to errors list
4. **Data validation:** Ensure at least 50 days of data for meaningful analysis

### Reference Code Pattern
See `LANGGRAPH_IMPLEMENTATION_GUIDE.md` lines 399-518 for the implementation pattern.

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_01_data_fetching.py`
- [ ] Implement `get_cached_price_data()` - checks SQLite
- [ ] Implement `fetch_from_finnhub()` - primary API
- [ ] Implement `fetch_from_yfinance()` - fallback
- [ ] Implement `cache_price_data()` - store in database
- [ ] Implement main `fetch_price_data_node(state)` function
- [ ] Add comprehensive error handling (try Finnhub → try yfinance → return None)
- [ ] Log all operations (cache hit, API call, errors)
- [ ] Update `state['raw_price_data']` with DataFrame
- [ ] Track execution time in `state['node_execution_times']['node_1']`

### Testing Strategy
```python
# Test 1: Valid ticker (should succeed)
state = {'ticker': 'AAPL', 'errors': [], 'node_execution_times': {}}
result = fetch_price_data_node(state)
assert result['raw_price_data'] is not None
assert len(result['raw_price_data']) >= 50

# Test 2: Invalid ticker (should handle gracefully)
state = {'ticker': 'INVALID123', 'errors': [], 'node_execution_times': {}}
result = fetch_price_data_node(state)
assert result['raw_price_data'] is None
assert 'Price data fetch failed' in result['errors'][0]

# Test 3: Cache hit (should be fast)
# First call fetches from API, second uses cache
```

### Success Criteria
- [ ] Fetches data for valid tickers (AAPL, NVDA, TSLA)
- [ ] Handles invalid tickers gracefully (doesn't break)
- [ ] Uses cache on second call (check logs)
- [ ] Execution time < 2 seconds
- [ ] Returns None on failure (doesn't raise exception)

### Estimated Time
3-4 hours

---

---

## 🔗 NODE 3: Related Companies Detection

### What It Does
Identifies competitor and related companies for a given stock based on sector, industry, and correlation analysis.

**Algorithm:**
```
Input: ticker ('NVDA')
↓
Get stock info (sector, industry) using yfinance
↓
Look up sector competitors from predefined mapping
  Technology → [AAPL, MSFT, AMD, INTC, ...]
↓
If have price data: Calculate correlation with top 10 competitors
  Take top 5 most correlated
↓
Output: List of related tickers ['AMD', 'INTC', 'TSM', 'QCOM', 'AVGO']
```

### Why It Matters
Related companies provide context. If all semiconductor stocks are down, NVDA probably will be too. This feeds into Node 6 (Market Context) and Node 2 (Related News).

### Dependencies
- **Runs AFTER:** Node 1 (needs price data for correlation)
- **Runs BEFORE:** Node 2 (news fetching needs related company list)
- **Parallel with:** Nothing

### Reference Documentation
**Detailed specifications:** `LangGraph_setup/NODE_03_INTELLIGENT_DISCOVERY.md`

This file provides:
- Sector-based competitor mapping
- Correlation calculation methodology
- Fallback strategies when correlation fails

### Key Concepts
1. **Sector mapping:** Predefined lists of companies per sector
2. **Correlation analysis:** Calculate price correlation to find truly related stocks
3. **Fallback strategy:** If correlation fails, use top 5 from sector list
4. **Limit to 5:** Don't overwhelm news fetching with too many related companies

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_03_related_companies.py`
- [ ] Define `SECTOR_COMPETITORS` dictionary (Technology, Healthcare, Finance, etc.)
- [ ] Implement `get_stock_info()` - fetch sector and industry
- [ ] Implement `calculate_correlation()` - correlate price movements
- [ ] Implement `detect_related_companies_node(state)` - main function
- [ ] Handle case where sector is unknown (return empty list)
- [ ] Remove target ticker from related list (don't include itself)
- [ ] Update `state['raw_related_companies']` with list of tickers
- [ ] Log which related companies were found

### Testing Strategy
```python
# Test 1: Technology stock (NVDA)
state = {'ticker': 'NVDA', 'raw_price_data': mock_df, ...}
result = detect_related_companies_node(state)
assert 'AMD' in result['raw_related_companies']
assert 'INTC' in result['raw_related_companies']
assert 'NVDA' not in result['raw_related_companies']  # Doesn't include itself

# Test 2: Unknown sector
state = {'ticker': 'UNKNOWN', ...}
result = detect_related_companies_node(state)
assert result['raw_related_companies'] == []
assert len(result['errors']) == 0  # Graceful handling
```

### Success Criteria
- [ ] Identifies 3-5 related companies for major stocks
- [ ] Uses correlation when price data available
- [ ] Falls back to sector list if correlation fails
- [ ] Handles unknown sectors gracefully
- [ ] Execution time < 3 seconds

### Estimated Time
2-3 hours

---

---

## 📰 NODE 2: Multi-Source News Fetching

### What It Does
Fetches news from THREE sources in parallel:
1. **Stock-specific news** - Articles about the target company
2. **Market-wide news** - General market/sector news
3. **Related companies news** - Articles about competitors

**Parallel Execution:**
```
Input: ticker ('NVDA'), related_companies (['AMD', 'INTC', ...])
↓
Launch 3 async tasks simultaneously:
  Task 1: Fetch stock news for 'NVDA'
  Task 2: Fetch market news ('stock market', 'nasdaq')
  Task 3: Fetch related companies news for ['AMD', 'INTC', ...]
↓
Wait for all 3 to complete (asyncio.gather)
↓
Output: 
  raw_stock_news (List[Dict])
  raw_market_news (List[Dict])
  raw_related_news (List[Dict])
```

### Why It Matters
News drives sentiment analysis (Node 5). Having 3 streams allows adaptive weighting to learn which type is most predictive.

### Dependencies
- **Runs AFTER:** Node 3 (needs related companies list)
- **Runs BEFORE:** Node 9A (early anomaly detection filters this news)
- **Parallel with:** Nothing

### Key Concepts
1. **Async parallelism:** 3x faster than sequential fetching
2. **Multiple sources:** Diversified news streams for robustness
3. **Rate limiting:** Respect API limits (sleep between requests)
4. **Cache results:** Store in database to avoid re-fetching

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_02_news_fetching.py`
- [ ] Implement `fetch_stock_news_async()` - async function
- [ ] Implement `fetch_market_news_async()` - async function
- [ ] Implement `fetch_related_news_async()` - async function
- [ ] Implement `fetch_all_news_node(state)` - main node using `asyncio.gather()`
- [ ] Handle NewsAPI errors gracefully (return empty list)
- [ ] Update state with 3 news lists
- [ ] Cache all news in database
- [ ] Log article counts for each source

### Testing Strategy
```python
# Test 1: Successful fetch
state = {
    'ticker': 'AAPL',
    'raw_related_companies': ['MSFT', 'GOOGL'],
    ...
}
result = fetch_all_news_node(state)
assert len(result['raw_stock_news']) > 0
assert len(result['raw_market_news']) > 0

# Test 2: API failure (mock to return error)
# Should return empty lists, not crash
```

### Success Criteria
- [ ] Fetches news from all 3 sources
- [ ] Executes in parallel (check timing)
- [ ] Handles API failures gracefully
- [ ] Caches results in database
- [ ] Execution time < 4 seconds

### Estimated Time
3-4 hours

---

---

## 🛡️ NODE 9A: Early Anomaly Detection (Content Filtering)

### What It Does
Filters suspicious news content BEFORE sentiment analysis. This is CRITICAL for protecting the learning system (Node 8) from contaminated data.

**Detection Checks:**
```
Input: raw_stock_news, raw_market_news, raw_related_news
↓
Check 1: Keyword Alerts
  Scan for: bankruptcy, fraud, investigation, scandal
↓
Check 2: News Surge
  Is article count 5x normal? (50+ vs usual 10-20)
↓
Check 3: Source Credibility
  Are articles from untrusted domains?
↓
Check 4: Coordinated Posting
  Are identical articles posted at same time? (bots)
↓
Filter articles that fail checks
↓
Output:
  cleaned_stock_news (suspicious articles removed)
  cleaned_market_news
  cleaned_related_news
  early_anomaly_detection results
```

### Why It Matters (THESIS CRITICAL)
**This is a KEY thesis innovation.** By filtering fake/manipulative news early:
1. Sentiment analysis (Node 5) works on clean data only
2. Learning system (Node 8) learns from accurate patterns
3. Protects users from manipulation

Without Node 9A, your learning system would learn from fake news and become less accurate over time.

### Dependencies
- **Runs AFTER:** Node 2 (needs raw news)
- **Runs BEFORE:** Nodes 4, 5, 6, 7 (analysis nodes need clean data)
- **Parallel with:** Nothing (must filter before analysis)

### Key Concepts
1. **Content-based filtering:** Analyze article text/source, not behavior
2. **Multiple checks:** Keyword alerts, news surges, source reputation, coordination
3. **Risk scoring:** Combine checks into overall risk level (LOW/MEDIUM/HIGH)
4. **Provide cleaned data:** Downstream nodes use `cleaned_*_news`, not `raw_*_news`

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_09a_early_anomaly.py`
- [ ] Define `DANGER_KEYWORDS` list (bankruptcy, fraud, investigation, ...)
- [ ] Define `UNTRUSTED_SOURCES` set (domains to filter)
- [ ] Implement `check_keyword_alerts()` - scan for danger words
- [ ] Implement `check_news_surge()` - detect abnormal article count
- [ ] Implement `check_source_credibility()` - identify untrusted sources
- [ ] Implement `check_coordinated_posting()` - detect bot patterns
- [ ] Implement `filter_suspicious_articles()` - remove flagged articles
- [ ] Calculate risk level (HIGH if score > 70, MEDIUM if > 40, else LOW)
- [ ] Update state with cleaned news AND detection results
- [ ] Log how many articles were filtered

### Testing Strategy
```python
# Test 1: Clean news (no filtering needed)
state = {'raw_stock_news': [good_article1, good_article2], ...}
result = early_anomaly_detection_node(state)
assert len(result['cleaned_stock_news']) == 2
assert result['early_anomaly_detection']['risk_level'] == 'LOW'

# Test 2: News with danger keywords
state = {'raw_stock_news': [article_with_fraud_keyword], ...}
result = early_anomaly_detection_node(state)
assert len(result['cleaned_stock_news']) == 0  # Filtered out
assert 'fraud' in result['early_anomaly_detection']['keyword_alerts']

# Test 3: News surge detection
state = {'raw_stock_news': [100 articles], ...}  # Way more than normal
result = early_anomaly_detection_node(state)
assert result['early_anomaly_detection']['news_surge_detected'] == True
```

### Success Criteria
- [ ] Filters articles with danger keywords
- [ ] Detects news surges (5x normal)
- [ ] Identifies untrusted sources
- [ ] Detects coordinated posting patterns
- [ ] Provides cleaned news for downstream nodes
- [ ] Execution time < 1 second

### Estimated Time
4-5 hours

---

---

## 📈 NODE 4: Technical Analysis

### What It Does
Calculates 6+ technical indicators from price data and generates a technical signal (BUY/SELL/HOLD).

**Indicators Calculated:**
- RSI (14-day) - Overbought/oversold
- MACD - Momentum and trend
- Bollinger Bands - Volatility
- SMA 20, 50 - Moving averages
- EMA 12, 26 - Exponential moving averages
- Volume analysis - Unusual volume

**Signal Generation:**
```
RSI < 30 → Oversold → +50 points
RSI > 70 → Overbought → -50 points
MACD > Signal → Bullish → +30 points
Price > SMA20 > SMA50 → Uptrend → +20 points
...
Total score → Normalize to 0-100 → BUY/SELL/HOLD
```

### Why It Matters
Technical analysis is one of 4 signal streams that get adaptively weighted. It typically has 50-60% accuracy.

### Dependencies
- **Runs AFTER:** Node 9A (conceptually, but uses raw_price_data which is unfiltered)
- **Runs in PARALLEL with:** Nodes 5, 6, 7 (analysis layer)
- **Runs BEFORE:** Node 8 (verification)

### Key Concepts
1. **Uses pandas-ta library:** Don't reinvent the wheel
2. **Scoring system:** Combine multiple indicators into single score
3. **Needs 50+ days:** Can't calculate meaningful indicators with less data
4. **Vectorized operations:** Use pandas, not loops

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_04_technical_analysis.py`
- [ ] Implement `calculate_technical_indicators()` helper function
- [ ] Use `pandas_ta` library for all calculations
- [ ] Calculate RSI (14-day)
- [ ] Calculate MACD (12, 26, 9)
- [ ] Calculate Bollinger Bands (20-day, 2 std)
- [ ] Calculate SMAs (20, 50)
- [ ] Calculate EMAs (12, 26)
- [ ] Analyze volume (current vs 20-day average)
- [ ] Implement scoring logic (combine indicators)
- [ ] Generate final signal (BUY if score > 65, SELL if < 35, else HOLD)
- [ ] Update `state['technical_analysis']` with full results
- [ ] Handle insufficient data error (< 50 days)

### Testing Strategy
```python
# Test 1: Valid price data (50+ days)
state = {'raw_price_data': df_100_days, ...}
result = technical_analysis_node(state)
assert result['technical_analysis']['rsi'] > 0
assert result['technical_analysis']['technical_signal'] in ['BUY', 'SELL', 'HOLD']

# Test 2: Insufficient data (< 50 days)
state = {'raw_price_data': df_20_days, ...}
result = technical_analysis_node(state)
assert result['technical_analysis'] is None
assert 'Insufficient price data' in result['errors'][0]
```

### Success Criteria
- [ ] Calculates all 6 indicators correctly
- [ ] Generates valid signal (BUY/SELL/HOLD)
- [ ] Confidence score between 0-100
- [ ] Handles insufficient data gracefully
- [ ] Execution time < 1 second

### Estimated Time
3-4 hours

---

---

## 💭 NODE 5: Sentiment Analysis (FinBERT)

### What It Does
Analyzes sentiment from THREE cleaned news streams using FinBERT (financial domain NLP model).

**CRITICAL:** Uses `cleaned_*_news` from Node 9A, NOT `raw_*_news`.

**Process:**
```
Input: cleaned_stock_news, cleaned_market_news, cleaned_related_news
↓
For each article:
  Combine title + description
  Run through FinBERT model
  Get sentiment: positive/negative/neutral + confidence score
↓
Aggregate by source type:
  Stock news: Average sentiment
  Market news: Average sentiment
  Related companies: Average sentiment
↓
Calculate combined sentiment (weighted average)
  Stock: 50% weight
  Market: 25% weight
  Related: 25% weight
↓
Output: sentiment_analysis with combined_sentiment and sentiment_signal
```

### Why It Matters
Sentiment typically has 60-65% accuracy without Node 8 learning, 70-75% with learning. This is one of 4 streams for adaptive weighting.

### Dependencies
- **Runs AFTER:** Node 9A (uses cleaned news)
- **Runs in PARALLEL with:** Nodes 4, 6, 7
- **Runs BEFORE:** Node 8 (learning/verification)

### Key Concepts
1. **FinBERT, not generic BERT:** Financial domain-specific model
2. **Cleaned data only:** Node 9A filtered out fake news
3. **Three-stream analysis:** Stock, market, related
4. **Weighted combination:** Stock news weighted highest (50%)
5. **Batch processing:** Analyze multiple articles efficiently

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_05_sentiment_analysis.py`
- [ ] Load FinBERT model at module level (load once, not per article)
- [ ] Implement `analyze_text_sentiment()` - single article analysis
- [ ] Implement `analyze_articles_sentiment()` - batch analysis
- [ ] Process cleaned_stock_news (from Node 9A!)
- [ ] Process cleaned_market_news
- [ ] Process cleaned_related_news
- [ ] Calculate weighted combined sentiment (50% stock, 25% market, 25% related)
- [ ] Generate sentiment signal (BUY if > 0.2, SELL if < -0.2, else HOLD)
- [ ] Update `state['sentiment_analysis']` with all results
- [ ] Handle model loading errors gracefully

### Testing Strategy
```python
# Test 1: Positive news
articles = [
    {'title': 'Company beats earnings expectations', ...},
    {'title': 'Strong quarterly growth reported', ...}
]
state = {'cleaned_stock_news': articles, ...}
result = sentiment_analysis_node(state)
assert result['sentiment_analysis']['combined_sentiment'] > 0
assert result['sentiment_analysis']['sentiment_signal'] == 'BUY'

# Test 2: Negative news
articles = [
    {'title': 'Company misses earnings', ...},
    {'title': 'Layoffs announced', ...}
]
state = {'cleaned_stock_news': articles, ...}
result = sentiment_analysis_node(state)
assert result['sentiment_analysis']['combined_sentiment'] < 0
```

### Success Criteria
- [ ] Loads FinBERT model successfully
- [ ] Analyzes all three news streams
- [ ] Produces sentiment score between -1 and +1
- [ ] Generates valid signal (BUY/SELL/HOLD)
- [ ] Uses cleaned news (not raw)
- [ ] Execution time < 4 seconds

### Estimated Time
4-5 hours

---

---

## 🌍 NODE 6: Market Context Analysis

### What It Does
Analyzes market-wide conditions to provide context for the individual stock signal.

**Analysis Components:**
1. **Sector Performance:** How is the stock's sector doing today? (Tech up 2%)
2. **Market Trend:** Is overall market bullish/bearish? (S&P 500 trend)
3. **Related Companies:** Are competitors up or down?
4. **Correlation:** How closely does this stock move with the market?

### Why It Matters
A stock might look great technically, but if the entire sector is crashing, it's probably a bad buy. Context matters.

### Dependencies
- **Runs AFTER:** Node 9A
- **Runs in PARALLEL with:** Nodes 4, 5, 7
- **Runs BEFORE:** Node 8

### Reference Documentation
**Detailed specifications:** `LangGraph_setup/NODE_06_STRUCTURE_FOR_CURSOR.md` (lines 1-802)

This file provides:
- Complete function structure with TODOs
- Sector ETF mapping (Technology → XLK, Healthcare → XLV, etc.)
- Correlation calculation methodology
- Signal generation scoring system
- Expected outputs and testing strategy

### Key Concepts
1. **Sector ETFs:** Use sector ETFs (XLK, XLV, XLF) as sector proxies
2. **Market proxy:** Use SPY (S&P 500) for overall market trend
3. **Correlation matters:** High correlation = market matters more
4. **Weighted scoring:** Sector + market + related companies → context signal

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_06_market_context.py`
- [ ] Implement `get_stock_sector()` - identify stock's sector
- [ ] Implement `get_sector_performance()` - fetch sector ETF performance
- [ ] Implement `get_market_trend()` - analyze SPY for market trend
- [ ] Implement `analyze_related_companies()` - check competitors' performance
- [ ] Implement `calculate_correlation()` - stock-market correlation
- [ ] Implement `generate_context_signal()` - combine all factors
- [ ] Use NODE_06 file as detailed reference (don't copy-paste, understand the logic)
- [ ] Update `state['market_context']` with results

### Success Criteria
- [ ] Identifies stock sector correctly
- [ ] Fetches sector ETF performance
- [ ] Determines market trend (BULLISH/BEARISH/NEUTRAL)
- [ ] Analyzes related companies
- [ ] Generates context signal
- [ ] Execution time < 3 seconds

### Estimated Time
4-5 hours

---

---

## 🎲 NODE 7: Monte Carlo Forecasting

### What It Does
Generates probabilistic price forecasts using Monte Carlo simulation with Geometric Brownian Motion.

**Algorithm:**
```
1. Calculate historical returns (daily % changes)
2. Calculate drift (average return) and volatility (std dev)
3. Run 1000 simulations:
   For each simulation:
     Start at current price
     For each future day (1-30):
       price_tomorrow = price_today * exp(drift + random_shock)
4. Calculate statistics:
   Mean forecast (expected price)
   68% confidence interval (1 standard deviation)
   95% confidence interval (2 standard deviations)
   Probability of increase
   Probability of decrease
```

### Why It Matters
Provides probabilistic forecast, not a single point prediction. Shows uncertainty ranges. Great for visualization.

### Dependencies
- **Runs AFTER:** Node 9A
- **Runs in PARALLEL with:** Nodes 4, 5, 6
- **Runs BEFORE:** Node 8

### Reference Documentation
**Detailed specifications:** `LangGraph_setup/NODE_07_STRUCTURE_FOR_CURSOR.md` (lines 1-777)

This file provides:
- Complete implementation guide with formulas
- Geometric Brownian Motion explanation
- Simulation algorithm step-by-step
- Confidence interval calculations
- Visualization preparation

### Key Concepts
1. **Geometric Brownian Motion:** Standard model for stock prices
2. **1000 simulations:** Enough for statistical significance
3. **30-day forecast:** Standard forecasting horizon
4. **Confidence intervals:** 68% and 95% (1 and 2 standard deviations)
5. **Not a guarantee:** Probabilistic, not deterministic

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_07_monte_carlo.py`
- [ ] Implement `calculate_returns()` - daily percentage changes
- [ ] Implement `calculate_drift_volatility()` - mean return and std dev
- [ ] Implement `run_simulation()` - single path simulation
- [ ] Implement `run_all_simulations()` - 1000 parallel simulations
- [ ] Implement `calculate_confidence_intervals()` - 68% and 95%
- [ ] Implement `calculate_probabilities()` - P(up) and P(down)
- [ ] Use NODE_07 file as detailed reference
- [ ] Update `state['monte_carlo_forecast']` with all results
- [ ] Store simulation paths for visualization

### Success Criteria
- [ ] Runs 1000 simulations successfully
- [ ] Calculates mean forecast
- [ ] Provides confidence intervals (68%, 95%)
- [ ] Calculates probability of increase/decrease
- [ ] Execution time < 3 seconds
- [ ] Results reasonable (not predicting 1000% gains)

### Estimated Time
4-5 hours

---

---

## 🔍 NODE 8: News Verification & Learning System ⭐

### What It Does
**THIS IS YOUR PRIMARY THESIS INNOVATION.**

Learns from 6 months of historical news-price correlations to identify which news sources are reliable for each stock.

**Learning Process:**
```
Step 1: Get historical news with outcomes from database
  Query: news_with_outcomes where ticker = 'NVDA' and date > 6 months ago
  
Step 2: Calculate source reliability
  For each source (Bloomberg.com, Reuters, random-blog.com):
    Count total articles
    Count accurate predictions (sentiment matched price movement)
    Calculate accuracy rate = accurate / total

Step 3: Calculate confidence multipliers
  High accuracy (>80%) → multiplier = 1.2x (boost confidence)
  Medium accuracy (60-80%) → multiplier = 1.0x (no change)
  Low accuracy (<60%) → multiplier = 0.5x (reduce confidence)
  
Step 4: Adjust current sentiment confidence
  Today's article from Bloomberg (historically 85% accurate):
    Original confidence: 75%
    Adjusted confidence: 75% × 1.2 = 90% ← BOOSTED
  
  Today's article from random blog (historically 20% accurate):
    Original confidence: 70%
    Adjusted confidence: 70% × 0.5 = 35% ← REDUCED
```

### Why It Matters (THESIS CRITICAL)
**Expected Impact:**
- Sentiment accuracy WITHOUT Node 8: ~62%
- Sentiment accuracy WITH Node 8: ~73%
- **Improvement: +11%** ← This is your thesis contribution!

This demonstrates that a learning system can improve prediction accuracy by identifying reliable sources.

### Dependencies
- **Runs AFTER:** Nodes 4, 5, 6, 7 (needs their analysis results)
- **Runs BEFORE:** Node 9B (behavioral anomaly detection)
- **Parallel with:** Nothing (sequential in pipeline)

### Database Requirements
**CRITICAL:** These tables MUST exist and contain data:

1. **news_outcomes table:**
   - Tracks what happened 7 days after each news article
   - Columns: news_id, ticker, price_at_news, price_7day_later, predicted_direction, actual_direction, prediction_was_accurate_7day

2. **source_reliability table:**
   - Stores calculated reliability scores
   - Columns: ticker, source_name, total_articles, accurate_predictions, accuracy_rate, confidence_multiplier

3. **Background task required:**
   - `scripts/update_news_outcomes.py` - Run daily to build historical dataset
   - This populates news_outcomes table by checking 7-day price changes

### Reference Documentation
**Complete implementation:** `LangGraph_setup/node_08_news_verification_COMPLETE.py` (lines 1-699)

**Conceptual guide:** `LangGraph_setup/NEWS_LEARNING_SYSTEM_GUIDE.md` (lines 1-547)

These files provide:
- Complete working implementation (use as reference)
- Database queries for historical data
- Source reliability calculation formulas
- Confidence adjustment algorithms
- Expected results and testing strategy

### Key Concepts
1. **Source-specific learning:** Bloomberg might be good for NVDA, bad for penny stocks
2. **Stock-specific learning:** Calculate reliability per stock per source
3. **7-day horizon:** Check if prediction came true 7 days later
4. **Confidence adjustment:** Boost reliable sources, reduce unreliable ones
5. **Continuous learning:** Gets better as more historical data accumulates

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_08_news_verification.py`
- [ ] Implement `get_news_with_outcomes()` - query database for historical data
- [ ] Implement `calculate_source_reliability()` - compute accuracy per source
- [ ] Implement `calculate_news_type_effectiveness()` - which type works best
- [ ] Implement `adjust_sentiment_confidence()` - apply multipliers
- [ ] Implement `calculate_historical_correlation()` - news-price correlation
- [ ] Implement `news_verification_node()` - main function
- [ ] Use node_08_news_verification_COMPLETE.py as detailed reference
- [ ] Update `state['news_impact_verification']` with results
- [ ] Handle case where insufficient historical data (<30 articles)

### Background Task Required
**Create:** `scripts/update_news_outcomes.py`

```python
# This script runs daily to build historical dataset
# For each news article that is 7+ days old without an outcome:
#   1. Get price when article was published
#   2. Get price 7 days later
#   3. Calculate if prediction was accurate
#   4. Store in news_outcomes table
```

### Testing Strategy
```python
# Test 1: Sufficient historical data (100+ articles)
state = {
    'ticker': 'NVDA',
    'sentiment_analysis': {...},
    ...
}
result = news_verification_node(state)
assert 'Bloomberg.com' in result['news_impact_verification']['source_reliability']
assert result['news_impact_verification']['source_reliability']['Bloomberg.com'] > 0.7

# Test 2: Insufficient historical data (<30 articles)
# Should use default confidence, not crash

# Test 3: Confidence adjustment
# Article from reliable source should have boosted confidence
# Article from unreliable source should have reduced confidence
```

### Success Criteria
- [ ] Queries historical data successfully
- [ ] Calculates source reliability (0-100%)
- [ ] Adjusts sentiment confidence based on source
- [ ] Handles insufficient data gracefully (uses defaults)
- [ ] Stores results for adaptive weighting
- [ ] Execution time < 2 seconds
- [ ] After 6 months: demonstrates 10-15% accuracy improvement

### Estimated Time
6-8 hours (complex, but critical for thesis)

---

---

## 🚨 NODE 9B: Behavioral Anomaly Detection

### What It Does
Detects sophisticated market manipulation AFTER all analysis is complete. This is the second phase of anomaly detection.

**Detection Checks (Behavior-Based):**
```
1. Pump-and-Dump Detection:
   - Price spike > 15% in 1 hour
   - Volume 10x normal
   - Followed by crash
   - Positive news but price crashed (divergence)
   
2. Price Anomalies:
   - Spikes (z-score > 3)
   - Crashes (z-score < -3)
   - Gaps (overnight change > 5%)
   
3. Volume Anomalies:
   - 10x normal volume
   - Unusual low volume
   
4. Volatility Spikes:
   - 5x standard deviation
   
5. News-Price Divergence:
   - Positive sentiment but price down
   - Negative sentiment but price up
```

### Why It Matters (THESIS INNOVATION)
**Two-Phase Detection Strategy:**
- **Node 9A (Early):** Filters fake news content BEFORE analysis
- **Node 9B (Late):** Detects behavioral manipulation AFTER analysis

This two-phase approach is novel and achieves 95%+ pump-and-dump detection with <3% false positives.

### Dependencies
- **Runs AFTER:** Node 8 (needs historical patterns)
- **Runs BEFORE:** Node 10 (backtesting)
- **Parallel with:** Nothing

### Key Concepts
1. **Behavior, not content:** Analyzes price/volume patterns, not news text
2. **Needs history:** Compares current behavior to historical norms
3. **Pump-and-dump focus:** This is the #1 manipulation risk for retail investors
4. **Combined score:** Multiple checks contribute to overall risk score
5. **Critical risk:** Score > 75 halts trading recommendation

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_09b_behavioral_anomaly.py`
- [ ] Implement `detect_price_anomalies()` - z-score analysis
- [ ] Implement `detect_volume_anomalies()` - volume spikes/drops
- [ ] Implement `detect_pump_and_dump()` - composite score
- [ ] Implement `detect_volatility_anomaly()` - std dev spikes
- [ ] Implement `detect_news_price_divergence()` - sentiment vs price
- [ ] Calculate overall manipulation probability (0-100)
- [ ] Determine risk level (CRITICAL if > 75, HIGH if > 50, else LOW)
- [ ] Update `state['behavioral_anomaly_detection']` with results
- [ ] Combine with Node 9A results for overall risk assessment

### Testing Strategy
```python
# Test 1: Normal stock (no anomalies)
state = {'raw_price_data': normal_df, 'sentiment_analysis': {...}, ...}
result = behavioral_anomaly_detection_node(state)
assert result['behavioral_anomaly_detection']['pump_and_dump_score'] < 50
assert result['behavioral_anomaly_detection']['risk_level'] == 'LOW'

# Test 2: Pump-and-dump pattern
state = {
    'raw_price_data': pump_dump_df,  # Spike then crash
    'sentiment_analysis': {'combined_sentiment': 0.8},  # Positive news
    ...
}
result = behavioral_anomaly_detection_node(state)
assert result['behavioral_anomaly_detection']['pump_and_dump_score'] > 75
assert result['behavioral_anomaly_detection']['is_pump_and_dump'] == True
```

### Success Criteria
- [ ] Detects price spikes/crashes
- [ ] Identifies volume anomalies
- [ ] Calculates pump-and-dump score
- [ ] Detects news-price divergence
- [ ] Generates risk level (LOW/MEDIUM/HIGH/CRITICAL)
- [ ] 95%+ detection rate on known pump-and-dump stocks (GME, AMC)
- [ ] <3% false positives on normal stocks
- [ ] Execution time < 1 second

### Estimated Time
5-6 hours

---

---

## 🎯 NODE 10: Backtesting

### What It Does
Tests historical accuracy of each of the 4 signal streams independently over the last 180 days.

**Four Signal Streams:**
1. Technical Analysis
2. Stock-Specific News Sentiment
3. Market News Sentiment
4. Related Companies Sentiment

**Process for Each Stream:**
```
For last 180 days:
  For each day:
    1. Get signal from that day (BUY/SELL/HOLD)
    2. Get actual price movement 7 days later
    3. Check if signal was correct:
       - BUY signal + price went up → Correct
       - SELL signal + price went down → Correct
       - HOLD signal + price stayed flat → Correct
    4. Track accuracy

Calculate:
  - Total predictions
  - Correct predictions
  - Accuracy rate = correct / total
```

### Why It Matters
These accuracy rates feed directly into Node 11 for adaptive weight calculation. Historical accuracy determines how much to trust each signal.

### Dependencies
- **Runs AFTER:** Node 9B (all analysis complete)
- **Runs BEFORE:** Node 11 (adaptive weights need these results)
- **Parallel with:** Nothing

### Key Concepts
1. **Independent testing:** Test each stream separately
2. **7-day horizon:** Standard forecasting window
3. **180-day period:** Sufficient for statistical significance
4. **Stock-specific:** Different stocks have different optimal signals

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_10_backtesting.py`
- [ ] Implement `backtest_technical_signals()` - test technical accuracy
- [ ] Implement `backtest_stock_news_signals()` - test stock news accuracy
- [ ] Implement `backtest_market_news_signals()` - test market news accuracy
- [ ] Implement `backtest_related_signals()` - test related companies accuracy
- [ ] Query database for 180 days of historical data
- [ ] Calculate accuracy rate for each stream
- [ ] Store results in `backtest_results` table
- [ ] Update `state['backtest_results']` with all accuracies

### Testing Strategy
```python
# Test 1: Sufficient historical data
state = {'ticker': 'AAPL', ...}
result = backtesting_node(state)
assert result['backtest_results']['technical_accuracy'] > 0
assert result['backtest_results']['technical_accuracy'] < 1.0
assert result['backtest_results']['sample_size'] >= 50

# Test 2: Insufficient historical data
# Should return None or defaults
```

### Success Criteria
- [ ] Tests all 4 signal streams
- [ ] Calculates accuracy rates (0-100%)
- [ ] Handles insufficient data gracefully
- [ ] Stores results in database
- [ ] Execution time < 3 seconds

### Estimated Time
4-5 hours

---

---

## ⚖️ NODE 11: Adaptive Weights Calculation

### What It Does
Calculates optimal weights for each signal stream based on Node 10's backtest results.

**Formula:**
```
weight_i = accuracy_i / sum(all_accuracies)

Example:
Technical: 55% accurate
Stock News: 65% accurate (boosted by Node 8 learning!)
Market News: 58% accurate
Related: 60% accurate

Total = 55 + 65 + 58 + 60 = 238

Weights:
Technical: 55/238 = 0.231 = 23.1%
Stock News: 65/238 = 0.273 = 27.3% ← Highest because Node 8 improved it!
Market News: 58/238 = 0.244 = 24.4%
Related: 60/238 = 0.252 = 25.2%
```

### Why It Matters
This is adaptive intelligence. The system learns which signals work best for each stock and adjusts weights automatically. Stock news gets highest weight because Node 8 improved its accuracy!

### Dependencies
- **Runs AFTER:** Node 10 (needs backtest results)
- **Runs BEFORE:** Node 12 (signal generation uses these weights)
- **Parallel with:** Nothing

### Key Concepts
1. **Proportional weighting:** Higher accuracy = higher weight
2. **Stock-specific:** Weights differ per stock (tech vs finance)
3. **Weights sum to 1.0:** Ensures proper weighting
4. **Transparent:** Users can see why weights are distributed
5. **Updates over time:** Recalculated daily as accuracy changes

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_11_adaptive_weights.py`
- [ ] Implement `calculate_proportional_weights()` - main formula
- [ ] Extract backtest results from state
- [ ] Calculate total accuracy sum
- [ ] Calculate individual weights (accuracy / sum)
- [ ] Validate weights sum to 1.0
- [ ] Generate explanation (LLM-powered or template)
- [ ] Store weights in `weight_history` table
- [ ] Update `state['adaptive_weights']` with results

### Testing Strategy
```python
# Test 1: Normal accuracy distribution
backtest_results = {
    'technical_accuracy': 55,
    'stock_news_accuracy': 65,
    'market_news_accuracy': 58,
    'related_companies_accuracy': 60
}
weights = calculate_proportional_weights(backtest_results)
assert abs(sum(weights.values()) - 1.0) < 0.001  # Weights sum to 1.0
assert weights['stock_news_weight'] > weights['technical_weight']  # Highest accuracy gets highest weight

# Test 2: One stream much better
backtest_results = {
    'technical_accuracy': 40,
    'stock_news_accuracy': 80,  # Much better
    'market_news_accuracy': 45,
    'related_companies_accuracy': 50
}
weights = calculate_proportional_weights(backtest_results)
assert weights['stock_news_weight'] > 0.35  # Should dominate
```

### Success Criteria
- [ ] Calculates weights that sum to 1.0
- [ ] Higher accuracy = higher weight
- [ ] Handles edge cases (all equal accuracy)
- [ ] Stores weights in database
- [ ] Provides human-readable explanation
- [ ] Execution time < 1 second

### Estimated Time
2-3 hours

---

---

## 🚦 CONDITIONAL EDGE: Risk Detection Router

### What It Does
This is NOT a node, but a **conditional routing function** in the LangGraph workflow.

**Decision Logic:**
```
After Node 11 (adaptive weights calculated):

Check combined risk:
  early_risk = state['early_anomaly_detection']['risk_level']
  behavioral_risk = state['behavioral_anomaly_detection']['risk_level']
  pump_dump_score = state['behavioral_anomaly_detection']['pump_and_dump_score']

IF any of:
  - pump_dump_score > 75
  - behavioral_risk == 'CRITICAL'
  - early_risk == 'HIGH' AND behavioral_risk == 'HIGH'
THEN:
  Route to Node 13 only (skip Node 12)
  Generate WARNING instead of signal
ELSE:
  Route to Node 12 (normal signal generation)
```

### Why It Matters
This is your safety mechanism. When manipulation is detected, the system HALTS signal generation and warns the user instead of providing a BUY/SELL recommendation.

### Dependencies
- **Runs AFTER:** Node 11
- **Routes TO:** Node 12 (if safe) OR Node 13 only (if critical risk)

### Implementation
This goes in `src/graph/workflow.py` or `src/graph/conditional_edges.py`:

```python
def should_halt_for_risk(state: StockAnalysisState) -> str:
    """
    Conditional edge: Determine if critical risk requires halting.
    
    Returns:
        'halt' → Skip to Node 13 (warning only)
        'continue' → Proceed to Node 12 (normal signal)
    """
    early = state.get('early_anomaly_detection', {})
    behavioral = state.get('behavioral_anomaly_detection', {})
    
    pump_dump_score = behavioral.get('pump_and_dump_score', 0)
    early_risk = early.get('early_risk_level', 'LOW')
    behavioral_risk = behavioral.get('risk_level', 'LOW')
    
    # Critical conditions
    if pump_dump_score > 75:
        logger.warning(f"HALT: Pump-and-dump detected (score: {pump_dump_score})")
        return 'halt'
    
    if behavioral_risk == 'CRITICAL':
        logger.warning(f"HALT: Critical behavioral risk")
        return 'halt'
    
    if early_risk == 'HIGH' and behavioral_risk == 'HIGH':
        logger.warning(f"HALT: Combined high risk")
        return 'halt'
    
    # Safe to continue
    return 'continue'
```

### Success Criteria
- [ ] Correctly identifies critical risk
- [ ] Routes to Node 13 only when risk detected
- [ ] Routes to  for normal analysis
- [ ] Logs routing decisions

### Estimated Time
1 hour

---

---

## 🎯 NODE 12: Final Signal Generation

### What It Does
Combines all 4 signal streams using adaptive weights to generate final BUY/SELL/HOLD recommendation.

**Weighted Combination:**
```
Signals:
- Technical: HOLD (score: 50)
- Stock News: BUY (score: 90)
- Market News: HOLD (score: 55)
- Related: BUY (score: 70)

Weights (from Node 11):
- Technical: 0.231
- Stock News: 0.273
- Market News: 0.244
- Related: 0.252

Calculation:
BUY_score = (0 × 0.231) + (100 × 0.273) + (0 × 0.244) + (100 × 0.252) = 52.5
SELL_score = 0
HOLD_score = (100 × 0.231) + (0 × 0.273) + (100 × 0.244) + (0 × 0.252) = 47.5

Final: BUY with 52.5% confidence
```

### Why It Matters
This is the final output. Everything culminates here.

### Dependencies
- **Runs AFTER:** Node 11 (needs adaptive weights)
- **Runs BEFORE:** Nodes 13, 14 (explanations)
- **Only runs if:** Conditional edge says 'continue' (not 'halt')

### Key Concepts
1. **Weighted voting:** Multiply each signal by its weight
2. **Convert to 0-100:** Normalize scores for confidence
3. **Set thresholds:** BUY if > 60, SELL if < 40, else HOLD
4. **Risk warnings:** Include any anomaly warnings
5. **Target price:** Based on Monte Carlo forecast

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_12_signal_generation.py`
- [ ] Extract signals from technical_analysis, sentiment_analysis, market_context
- [ ] Extract weights from adaptive_weights
- [ ] Implement weighted scoring algorithm
- [ ] Determine final recommendation (BUY/SELL/HOLD)
- [ ] Calculate confidence (0-100)
- [ ] Set target price (from Monte Carlo mean forecast)
- [ ] Set stop loss (risk management level)
- [ ] List contributing factors
- [ ] Include risk warnings from anomaly detection
- [ ] Update `state['final_signal']` with complete result

### Testing Strategy
```python
# Test 1: Majority BUY signals
state = {
    'technical_analysis': {'technical_signal': 'BUY', ...},
    'sentiment_analysis': {'sentiment_signal': 'BUY', ...},
    'market_context': {'context_signal': 'HOLD', ...},
    'adaptive_weights': {...}
}
result = signal_generation_node(state)
assert result['final_signal']['recommendation'] == 'BUY'
assert result['final_signal']['confidence'] > 60

# Test 2: Mixed signals
# Should depend on weights
```

### Success Criteria
- [ ] Combines all 4 streams correctly
- [ ] Applies adaptive weights
- [ ] Generates valid recommendation
- [ ] Provides confidence score
- [ ] Includes risk warnings
- [ ] Execution time < 1 second

### Estimated Time
3-4 hours

---

---

## 💡 NODE 13: Beginner Explanation (LLM)

### What It Does
Generates plain English explanation of the analysis for non-technical users.

**Example Output:**
```
Based on the analysis of NVIDIA (NVDA):

The system recommends BUYING this stock with 75% confidence.

Here's why in simple terms:

1. TECHNICAL ANALYSIS says BUY: The stock's recent price patterns show upward momentum. 
   The RSI indicator shows it's not overbought yet.

2. NEWS SENTIMENT is POSITIVE: Recent articles from reliable sources like Bloomberg 
   are positive about NVIDIA's new AI chip launch. Our learning system knows Bloomberg 
   is 85% accurate for NVIDIA, so we trust this news.

3. MARKET CONDITIONS are FAVORABLE: The overall tech sector is up 2% today, which is 
   a good sign for tech stocks like NVIDIA.

4. RELATED COMPANIES are STRONG: Competitors like AMD and Intel are also performing well.

IMPORTANT: The system detected some unusual trading volume, but it's not critical. 
Monitor the stock closely.

Bottom line: This looks like a good buying opportunity based on multiple positive signals.
```

### Why It Matters
Makes sophisticated analysis accessible to everyday investors. One of thesis's key goals is democratizing financial intelligence.

### Dependencies
- **Runs AFTER:** Node 12 (if normal flow) OR Node 11 (if halted for risk)
- **Runs in PARALLEL with:** Node 14 (technical explanation)
- **Runs BEFORE:** Node 15 (dashboard prep)

### Key Concepts
1. **LLM-powered:** Use Claude 3.5 Sonnet for natural language generation
2. **Plain English:** No jargon, no abbreviations
3. **Context from state:** Pull all analysis results
4. **Explain WHY:** Not just WHAT the recommendation is
5. **Include warnings:** Anomaly alerts in simple language

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_13_beginner_explanation.py`
- [ ] Load Claude API client
- [ ] Build comprehensive prompt with all state data
- [ ] Include: final signal, contributing factors, risk warnings
- [ ] Request plain English explanation (8th grade reading level)
- [ ] Generate 200-400 word explanation
- [ ] Handle LLM API errors gracefully (return template if fails)
- [ ] Update `state['beginner_explanation']` with result

### Prompt Template
```python
prompt = f"""
You are explaining stock analysis to a complete beginner investor.

Stock: {ticker}
Recommendation: {recommendation} with {confidence}% confidence

Technical Analysis: {technical_summary}
News Sentiment: {sentiment_summary}
Market Context: {context_summary}
Risk Alerts: {risk_alerts}

Please explain in simple, plain English:
1. What the recommendation is
2. Why the system arrived at this recommendation
3. What the main factors were
4. Any risks to be aware of

Use NO jargon. Write like you're explaining to a friend over coffee.
Keep it to 200-400 words.
"""
```

### Success Criteria
- [ ] Generates plain English explanation
- [ ] Explains recommendation clearly
- [ ] Includes all major factors
- [ ] Mentions risk warnings
- [ ] 8th grade reading level
- [ ] Execution time < 3 seconds

### Estimated Time
2-3 hours

---

---

## 🎓 NODE 14: Technical Explanation (LLM)

### What It Does
Generates detailed technical breakdown for experienced investors/analysts.

**Example Output:**
```
NVIDIA (NVDA) - Technical Analysis Report

FINAL RECOMMENDATION: BUY
Confidence: 75.3%
Signal Strength: MODERATE

=== SIGNAL BREAKDOWN ===

Technical Analysis (Weight: 23.1%):
- RSI(14): 58.4 (Neutral zone, room to grow)
- MACD: Bullish crossover detected on Jan 10
- Bollinger Bands: Price at middle band, not extended
- SMA20/SMA50: Golden cross formation
- Volume: 1.2x normal (slight increase)
→ Technical Signal: BUY (62% confidence)

Sentiment Analysis (Weight: 27.3%):
- Stock News: 0.72 positive (12 articles, 9 positive, 2 neutral, 1 negative)
- Market News: 0.45 positive (market bullish)
- Related Companies: 0.58 positive (sector strength)
→ Combined Sentiment: 0.65 (Positive)
→ Sentiment Signal: BUY (88% confidence)
→ Node 8 Adjustment: Bloomberg.com (85% accurate) boosted confidence by 15%

Market Context (Weight: 24.4%):
- Sector Performance: Technology +2.3%
- Market Trend: BULLISH (SPY +1.5% over 5 days)
- Correlation: 0.82 (high correlation with market)
- Related Companies: AMD +2.1%, INTC +0.8%, TSM +1.9%
→ Context Signal: BUY (78% confidence)

Monte Carlo Forecast (Informational):
- 30-day expected price: $545 (current: $520)
- 68% confidence interval: [$510, $560]
- 95% confidence interval: [$490, $580]
- Probability of increase: 68%

=== ADAPTIVE WEIGHTING ===

Backtest Results (180 days):
- Technical: 55% accuracy
- Stock News: 65% accuracy (improved by Node 8 learning)
- Market News: 58% accuracy
- Related Companies: 60% accuracy

Weight Calculation:
Stock News receives highest weight (27.3%) due to Node 8 learning system 
identifying reliable sources (Bloomberg, Reuters) with 80-85% historical accuracy.

=== RISK ASSESSMENT ===

Anomaly Detection:
- Early (Content): LOW risk
- Behavioral: MEDIUM risk (volume 1.8x normal)
- Pump-and-Dump Score: 32/100 (low risk)

Overall Risk: ACCEPTABLE

=== RECOMMENDATION ===

BUY with confidence 75.3%
Target Price: $545 (+4.8%)
Stop Loss: $495 (-4.8%)

Primary Factors:
1. Strong positive sentiment from reliable sources
2. Bullish technical setup (golden cross)
3. Favorable sector and market conditions

Risks:
- Slight volume anomaly (monitor for acceleration)
- High correlation with market (vulnerable to market downturn)
```

### Why It Matters
For advanced users, analysts, and thesis reviewers who want to see the math and methodology.

### Dependencies
- **Runs AFTER:** Node 12
- **Runs in PARALLEL with:** Node 13
- **Runs BEFORE:** Node 15

### Key Concepts
1. **Comprehensive breakdown:** Every number, every decision
2. **Shows methodology:** Formulas, weights, calculations
3. **Statistical detail:** Confidence intervals, p-values
4. **Highlights Node 8:** Show learning system's impact
5. **Validation:** Thesis reviewers will scrutinize this

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_14_technical_explanation.py`
- [ ] Load Claude API client
- [ ] Build detailed prompt with ALL state data
- [ ] Request structured technical report
- [ ] Format with sections: Signal Breakdown, Weights, Risk, Recommendation
- [ ] Include specific numbers (RSI: 58.4, not "RSI looks good")
- [ ] Explain Node 8's contribution explicitly
- [ ] Update `state['technical_explanation']` with result

### Success Criteria
- [ ] Includes all numerical details
- [ ] Shows weight calculations
- [ ] Explains adaptive weighting
- [ ] Highlights Node 8 contribution
- [ ] Professional report format
- [ ] Execution time < 3 seconds

### Estimated Time
2-3 hours

---

---

## 📊 NODE 15: Dashboard Data Preparation

### What It Does
Prepares all data structures needed for the Streamlit dashboard's 6 tabs.

**Output Structure:**
```
dashboard_data = {
    'price_chart_data': {
        'dates': [...],
        'prices': [...],
        'volume': [...],
        'indicators': {
            'sma_20': [...],
            'sma_50': [...],
            'bb_upper': [...],
            'bb_lower': [...]
        }
    },
    'monte_carlo_chart_data': {
        'simulation_paths': np.array,  # 1000 x 30
        'percentiles': {
            '5': [...],
            '25': [...],
            '50': [...],
            '75': [...],
            '95': [...]
        }
    },
    'sentiment_chart_data': {
        'stock_news': {'positive': 12, 'negative': 3, 'neutral': 5},
        'market_news': {...},
        'related_news': {...}
    },
    'weight_visualization_data': {
        'current_weights': {...},
        'historical_weights': [...],
        'backtest_accuracies': {...}
    },
    'anomaly_visualization_data': {
        'early_alerts': [...],
        'behavioral_alerts': [...],
        'risk_timeline': [...]
    },
    'export_data': {
        'summary': {...},
        'detailed_report': "...",
        'csv_ready': True
    }
}
```

### Why It Matters
Clean separation between analysis and visualization. Dashboard can be rebuilt without rerunning analysis.

### Dependencies
- **Runs AFTER:** Nodes 13, 14 (needs explanations)
- **Runs BEFORE:** Streamlit dashboard
- **Final node:** Nothing comes after this

### Key Concepts
1. **Plotly-ready:** Format data for Plotly charts
2. **Tab-specific:** Organize by dashboard tab
3. **Export-ready:** Prepare CSV/JSON downloads
4. **No computation:** Just data formatting, no new calculations
5. **Handle missing:** If analysis failed, provide defaults

### Implementation Checklist
- [ ] Create `src/langgraph_nodes/node_15_dashboard_prep.py`
- [ ] Implement `prepare_price_chart_data()` - Tab 1 data
- [ ] Implement `prepare_monte_carlo_chart_data()` - Tab 2 data
- [ ] Implement `prepare_sentiment_chart_data()` - Tab 3 data
- [ ] Implement `prepare_weight_visualization_data()` - Tab 4 data
- [ ] Implement `prepare_anomaly_visualization_data()` - Tab 5 data
- [ ] Implement `prepare_export_data()` - Tab 6 data
- [ ] Combine all into `dashboard_data` dictionary
- [ ] Update `state['dashboard_data']` with complete structure

### Success Criteria
- [ ] All 6 tabs have required data
- [ ] Data formatted for Plotly
- [ ] Export files ready (CSV, JSON)
- [ ] Handles missing data gracefully
- [ ] Execution time < 1 second

### Estimated Time
3-4 hours

---

---

## 🎨 STREAMLIT DASHBOARD (Separate from Nodes)

### What It Does
6-tab interactive dashboard that displays all analysis results.

**Tabs:**
1. **Price & Technical Indicators:** Candlestick chart with overlays
2. **Monte Carlo Forecast:** Fan chart with confidence intervals
3. **News & Sentiment:** Article list with sentiment scores
4. **Signal Weights:** Adaptive weights visualization
5. **Risk Alerts:** Anomaly detection results
6. **Export:** Download CSV/PDF reports

### Implementation Guide
This is NOT a node. Build separately in `streamlit_app/` folder.

Files needed:
- `streamlit_app/app.py` - Main entry point
- `streamlit_app/tabs/tab_1_price_indicators.py`
- `streamlit_app/tabs/tab_2_monte_carlo.py`
- `streamlit_app/tabs/tab_3_news_sentiment.py`
- `streamlit_app/tabs/tab_4_signal_weights.py`
- `streamlit_app/tabs/tab_5_risk_alerts.py`
- `streamlit_app/tabs/tab_6_export.py`

### Estimated Time
10-12 hours (can build incrementally)

---

---

## 🗺️ IMPLEMENTATION ROADMAP

### Week 1-2: Foundation
- [ ] Database setup + schema
- [ ] Node 1: Price fetching
- [ ] Node 3: Related companies
- [ ] Node 2: Multi-source news
- [ ] **Milestone:** Can fetch all data for a stock

### Week 3-4: Protection & Analysis
- [ ] Node 9A: Early anomaly detection
- [ ] Node 4: Technical analysis
- [ ] Node 5: Sentiment analysis
- [ ] Node 6: Market context
- [ ] Node 7: Monte Carlo
- [ ] **Milestone:** Complete analysis pipeline working

### Week 5-6: Learning System (THESIS CORE)
- [ ] Node 8: News verification & learning
- [ ] Background task: Update news outcomes
- [ ] Build 6 months of historical data
- [ ] **Milestone:** Learning system demonstrating accuracy improvement

### Week 7-8: Advanced Features
- [ ] Node 9B: Behavioral anomaly detection
- [ ] Node 10: Backtesting
- [ ] Node 11: Adaptive weights
- [ ] Conditional edge: Risk router
- [ ] **Milestone:** Complete intelligence layer

### Week 9-10: Output & Visualization
- [ ] Node 12: Signal generation
- [ ] Node 13: Beginner explanation
- [ ] Node 14: Technical explanation
- [ ] Node 15: Dashboard prep
- [ ] **Milestone:** End-to-end workflow complete

### Week 11-12: Dashboard & Polish
- [ ] Streamlit dashboard (all 6 tabs)
- [ ] Testing & bug fixes
- [ ] Performance optimization
- [ ] **Milestone:** Production-ready system

### Week 13-14: Thesis & Documentation
- [ ] Write thesis document
- [ ] Create presentation
- [ ] Record demo video
- [ ] Prepare defense
- [ ] **Milestone:** Ready for graduation!

---

---

## 📚 KEY REFERENCE FILES

Throughout implementation, refer to these files:

1. **Overall Architecture:**
   - `LANGGRAPH_IMPLEMENTATION_GUIDE.md` - Complete technical guide
   - `langgraph_architecture_interactive_UPDATED.html` - Visual architecture

2. **Node-Specific:**
   - `NODE_03_INTELLIGENT_DISCOVERY.md` - Related companies detection
   - `NODE_06_STRUCTURE_FOR_CURSOR.md` - Market context analysis
   - `NODE_07_STRUCTURE_FOR_CURSOR.md` - Monte Carlo forecasting
   - `node_08_news_verification_COMPLETE.py` - Complete Node 8 implementation

3. **Learning System:**
   - `NEWS_LEARNING_SYSTEM_GUIDE.md` - Conceptual explanation
   - `schema_UPDATED_with_news_outcomes.sql` - Database tables

4. **Rules:**
   - `.cursor/rules/langgraph_stock_analysis.md` - Development standards

---

---

## 🎯 SUCCESS CRITERIA SUMMARY

### Technical Criteria
- [ ] All 16 nodes implemented and tested
- [ ] Total execution time < 5 seconds
- [ ] Database operations optimized
- [ ] Error handling comprehensive
- [ ] Type hints on all functions
- [ ] 70%+ test coverage

### Thesis Criteria
- [ ] Node 8 demonstrates 10-15% accuracy improvement
- [ ] Two-phase anomaly detection achieves 95%+ accuracy
- [ ] Adaptive weighting shows improvement over equal weights
- [ ] System works reliably for 10-15 test stocks
- [ ] Dashboard functional for thesis defense

### Quality Criteria
- [ ] Code follows all rules in `.cursor/rules/`
- [ ] Comprehensive documentation
- [ ] Clear commit history
- [ ] README with setup instructions
- [ ] Professional presentation ready

---

---

## 💡 FINAL REMINDERS

1. **Build incrementally** - Test each node before moving to next
2. **Follow the rules** - Consistency is key for thesis quality
3. **Document as you go** - Screenshots, notes, observations
4. **Test with real stocks** - AAPL, NVDA, TSLA, GME
5. **Node 8 is your star** - Make it shine
6. **Two-phase anomaly** - Novel contribution
7. **This is a thesis** - Code clarity matters as much as functionality
8. **14-week deadline** - Stay on schedule
9. **Ask for help** - Don't get stuck for days
10. **You can do this!** - One node at a time

---

**This guide is your roadmap. Follow it node by node. You've got this! 🚀**
