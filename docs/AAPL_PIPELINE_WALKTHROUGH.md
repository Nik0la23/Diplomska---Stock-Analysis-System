# AAPL Pipeline Walkthrough — Step by Step

This document explains **exactly what happened** when we ran **Apple (AAPL)** through the full LangGraph pipeline. Each section corresponds to one node or phase.

---

## High-Level Flow

```
START
  ↓
Node 1: Price Data        → 127 days of AAPL OHLCV
  ↓
Node 3: Related Companies → 5 peers (WDC, SNDK, DELL, HPE, PSTG)
  ↓
Node 2: News Fetching     → 50 stock + 100 market articles
  ↓
Node 9A: Content Analysis → Cleaned news + anomaly scores
  ↓
┌─────────┬─────────┬─────────┬─────────┐
│ Node 4  │ Node 5  │ Node 6  │ Node 7  │  ← PARALLEL (all 4 at once)
│ Tech    │ Sentiment│ Market  │ Monte   │
│ 0.17s   │ 5.88s   │ 2.16s   │ 0.02s   │
└─────────┴─────────┴─────────┴─────────┘
  ↓
END (Node 8 not yet implemented)
```

**Total time:** ~19.2 seconds (sequential part ~11s, parallel part ~6s max).

---

## Step 1: Node 1 — Price Data Fetching

**What it does:** Fetches historical OHLCV (Open, High, Low, Close, Volume) for AAPL.

**What happened:**
- Requested **180 days** of data from **yfinance**.
- Received **127 rows** (trading days only; weekends/holidays excluded).
- Data was **validated** (no missing critical fields).
- Results were **cached** in the database for future runs.
- **Time:** 0.59s.

**Output added to state:**
- `raw_price_data` — DataFrame with 127 rows (date, open, high, low, close, volume).

**Why it matters:** Every later node that needs price (technical analysis, Monte Carlo, correlation) uses this single source.

---

## Step 2: Node 3 — Related Companies Detection

**What it does:** Finds “peer” tickers (competitors/sector mates) for AAPL to use in news and market context.

**What happened:**
- Called **Finnhub** peer API for AAPL.
- Got **11 potential peers**.
- Peers were **ranked** (by correlation when possible); here no correlation was computed, so the **first 5** were kept: **WDC, SNDK, DELL, HPE, PSTG**.
- **Time:** 9.55s (Finnhub API is the slowest step in the pipeline).

**Output added to state:**
- `related_companies` — `['WDC', 'SNDK', 'DELL', 'HPE', 'PSTG']`.

**Why it matters:** Node 2 can fetch “related” news (if configured), and **Node 6** uses this list to compare AAPL’s performance to peers (e.g. “3 up, 2 down → BULLISH”).

---

## Step 3: Node 2 — Multi-Source News Fetching

**What it does:** Fetches recent news for the **ticker** and for the **market**, from multiple APIs.

**What happened:**
- **Date range** was taken from Node 1’s price data (e.g. 2025-08-14 to 2026-02-14).
- **Cache** was checked first; **miss** → fetch from APIs.
- **Finnhub:** 100 **market** articles (general category).
- **Alpha Vantage:** 50 **stock** articles specifically about AAPL (with built-in sentiment scores).
- Stock and market articles were **cached** in the DB.
- **Time:** 0.84s.

**Output added to state:**
- `stock_news` — 50 articles (Alpha Vantage, with sentiment).
- `market_news` — 100 articles (Finnhub).
- After Node 9A: `cleaned_stock_news`, `cleaned_market_news`, `cleaned_related_company_news`.

**Why it matters:** Node 9A cleans this news; Node 5 uses the **cleaned** lists for sentiment (Alpha Vantage + FinBERT for items without sentiment).

---

## Step 4: Node 9A — Content Analysis & Feature Extraction

**What it does:** Cleans and enriches the raw news (no filtering), and adds anomaly/relevance-style scores for downstream use.

**What happened:**
- Processed **150 articles** (50 stock + 100 market; 0 related in this run).
- Each article got **composite anomaly score** and **source reliability** (High/Medium/Low).
- **Average composite anomaly score:** 0.138.
- **High-risk articles:** 0.
- **Source distribution:** High 23, Medium 90, Low 37.
- **Time:** 0.01s.

**Output added to state:**
- `cleaned_stock_news`, `cleaned_market_news`, `cleaned_related_company_news` (with extra fields).
- `content_analysis_summary`, `behavioral_anomalies`, `source_reliability_scores`, etc.

**Why it matters:** Node 5 only sees **cleaned_*_news**; it does not use raw `stock_news` / `market_news`. So 9A is the bridge between “raw fetch” and “sentiment + rest of pipeline”.

---

## Step 5: Parallel Block — Four Nodes at Once

After Node 9A, the graph **fans out**: the same state is passed to **Nodes 4, 5, 6, and 7** at the same time. Each node only **writes its own** keys (partial state update), so they do not overwrite each other.

---

### Step 5a: Node 4 — Technical Analysis

**What it does:** Computes technical indicators (RSI, MACD, Bollinger Bands, etc.) from `raw_price_data` and emits a **technical signal** (BUY/SELL/HOLD) and confidence.

**What happened:**
- Used **127 days** of price data.
- **RSI:** 39.58 (below 50 → bearish momentum, but not oversold).
- **MACD:** 1.18 vs signal -0.01 → positive momentum.
- **Trend:** classified as **strong_downtrend**.
- Logic combined indicators → **Signal: HOLD**, **Confidence: 80%**.
- **Time:** 0.17s.

**Output added to state:**
- `technical_indicators` (RSI, MACD, Bollinger Bands, etc.)
- `technical_signal` = `"HOLD"`
- `technical_confidence` = 0.80

**Why it matters:** Gives a pure price-based view. For AAPL here: trend is down, but not extreme → HOLD.

---

### Step 5b: Node 5 — Sentiment Analysis

**What it does:** Aggregates sentiment from **cleaned** stock, market, and related news (Alpha Vantage + FinBERT for items without sentiment), then outputs a single **sentiment signal** and confidence.

**What happened:**
- **Input:** 50 stock, 100 market, 0 related articles (from Node 9A).
- **Stock (50):** All had **Alpha Vantage** sentiment → no FinBERT needed for these.
- **Market (100):** None had sentiment → **FinBERT** was loaded and run on all 100 (took ~2s after model load).
- **Aggregation:**
  - Stock: 17 positive, 10 negative, 23 neutral → weighted sentiment **0.119** → HOLD.
  - Market: 21 positive, 41 negative, 38 neutral → weighted sentiment **-0.167** → HOLD.
  - Related: 0 articles → 0.0.
- **Combined (50% stock, 25% market, 25% related):** **0.018** → still in HOLD range.
- **Confidence:** 0.46 → below 0.5 → **forced to HOLD**.
- **Time:** 5.88s (most of it: loading FinBERT once + analyzing 100 market articles).

**Output added to state:**
- `raw_sentiment_scores` (per-article list).
- `aggregated_sentiment` = 0.018
- `sentiment_signal` = `"HOLD"`
- `sentiment_confidence` = 0.46

**Why it matters:** Captures “mood” of news. Here: stock news slightly positive, market news slightly negative, combined near zero and low confidence → HOLD.

---

### Step 5c: Node 6 — Market Context

**What it does:** Looks at **sector** (via sector ETF), **broad market** (SPY), **related companies’** performance, and **correlation** with the market, then outputs a **context signal** (BUY/SELL/HOLD) and confidence.

**What happened:**
- **Sector:** yfinance → AAPL = **Technology**, **Consumer Electronics**.
- **Sector ETF:** Technology → **XLK**. Fetched 1-day performance → **+0.25%** → **FLAT** (within ±0.5%).
- **Market:** SPY 5-day performance **-1.28%**, volatility 0.77% → **NEUTRAL** (not clearly bullish/bearish).
- **Related companies:** WDC, SNDK, DELL, HPE, PSTG → **3 up, 2 down** → **BULLISH**.
- **Correlation:** With current price data, correlation/beta came out **nan** (e.g. date alignment or insufficient data) → treated as **LOW**.
- **Scoring:** Sector FLAT (0) + Market NEUTRAL (0) + Related BULLISH (+20) = **20** → below BUY (30) and above SELL (-30) → **HOLD**, confidence 50%.
- **Time:** 2.16s.

**Output added to state:**
- `market_context` (dict with sector, industry, sector_performance, market_trend, related_companies_*, correlation, beta, context_signal, confidence).
- Here: **context_signal** = `"HOLD"`, **confidence** = 50%.

**Why it matters:** Stops you from acting on a “good stock” when the sector or market is weak. Here: sector flat, market neutral, peers slightly bullish → context says “no strong edge” → HOLD.

---

### Step 5d: Node 7 — Monte Carlo Forecasting

**What it does:** Uses **raw_price_data** to estimate drift and volatility, runs **1000** simulated paths for the next **7 days**, and outputs a **forecasted price**, **expected return**, **probability of gain**, and **confidence intervals**.

**What happened:**
- **Input:** 127 days of price data.
- **Last 30 days** used for drift and volatility: drift **-0.18% per day**, volatility **1.76% per day**.
- **Current price:** $255.78.
- **1000 simulations × 7 days** → mean forecast **$255.75**, 68% CI [$255.03, $256.46], 95% CI [$254.27, $257.17].
- **Probability price goes up:** 47.8%.
- **Expected return (7d):** -0.01%.
- **Time:** 0.02s.

**Output added to state:**
- `monte_carlo_results` (current_price, mean_forecast, confidence intervals, probability_up, expected_return, etc.)
- `forecasted_price` = 255.75
- `price_range` (e.g. 95% interval)

**Why it matters:** Quantifies short-term price uncertainty. Here: almost no expected move, ~50% up/down → no strong forecast edge.

---

## Step 6: After the Parallel Block

- Each of Nodes 4, 5, 6, 7 **merged** its partial update into the shared state.
- **No conflicts:** each node writes only its own keys.
- The **conditional edge** after the parallel block currently sends the flow to **END** (Node 8 is not implemented yet).
- So the run **ends** with a state that contains:
  - Price data, related companies, news (raw + cleaned).
  - **Technical:** HOLD, 80%.
  - **Sentiment:** HOLD, 45.8%.
  - **Market context:** HOLD, 50%.
  - **Monte Carlo:** $255.75, -0.01% expected return, 47.8% probability up.

---

## Summary Table (AAPL Run)

| Node | Role | Main output | Time |
|------|------|-------------|------|
| 1 | Price data | 127 days OHLCV | 0.59s |
| 3 | Related companies | WDC, SNDK, DELL, HPE, PSTG | 9.55s |
| 2 | News | 50 stock + 100 market | 0.84s |
| 9A | Content analysis | Cleaned news + scores | 0.01s |
| 4 | Technical | HOLD, 80%, RSI 39.58 | 0.17s |
| 5 | Sentiment | HOLD, 45.8%, combined 0.018 | 5.88s |
| 6 | Market context | HOLD, 50%, Tech FLAT, peers BULLISH | 2.16s |
| 7 | Monte Carlo | $255.75, -0.01%, P(up)=47.8% | 0.02s |

**Interpretation for this run:** All four analysis nodes say **HOLD**: no strong buy or sell from technicals, sentiment, context, or short-term forecast. That’s a consistent “wait and see” for AAPL at this snapshot.

---

## How to Run It Yourself

From project root with venv activated:

```bash
python -c "
from src.graph.workflow import run_stock_analysis, print_analysis_summary
result = run_stock_analysis('AAPL')
print_analysis_summary(result)
"
```

Or use the integration script (e.g. for NVDA):

```bash
python scripts/test_nodes_05_06_integration.py
```

(Change the ticker inside the script if you want AAPL instead.)

---

---

## Data freshness: how far back each step goes

| Step | What we request | Effective “days back” |
|------|--------------------------------|-------------------------|
| **Node 1** | 180 days of price data (`days=180` → yfinance `period='6mo'`) | **6 months** (~127 trading days) |
| **Node 2 – Alpha Vantage** | `time_from` / `time_to` = **6 months back → today**; `limit=1000` | **6 months** (same window as Node 1) |
| **Node 2 – Finnhub company** | `from` / `to` = **6 months back → today** (company-news endpoint) | **6 months** |
| **Node 2 – Finnhub market** | General market news (no date range in API); then filter by 6-month window | Recent articles only, filtered to 6-month window |

**Node 2 date range:** The node always uses a **6-month (180-day) window**: `from_date = today - 180 days`, `to_date = today`. This is passed to Alpha Vantage (`time_from`, `time_to`) and to Finnhub company-news (`from`, `to`). Stock news is merged from both sources; market news is from Finnhub general and filtered to the same window.

**Doc generated from a single AAPL pipeline run; your numbers may vary slightly with date and data updates.**
