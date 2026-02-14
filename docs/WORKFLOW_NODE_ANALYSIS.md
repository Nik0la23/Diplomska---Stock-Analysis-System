# Complete Workflow Analysis: Node 9A + Parallel Nodes

## Overview

This document explains how Node 9A labels data and how parallel nodes (4, 5, 6, 7) process that data to generate signals.

---

## NODE 9A: CONTENT ANALYSIS & LABELING

### Purpose
Node 9A is the **gatekeeper** that analyzes news articles and scores them for quality/trustworthiness **before** they reach the sentiment analysis (Node 5).

### What Node 9A Does

For each article from Node 2, it adds these scores (0-1 scale):

#### 1. **Sensationalism Score** (0-1)
Detects clickbait patterns:
- **Checks for:** ALL CAPS words, excessive punctuation (!!!), keywords like "SHOCKING", "UNBELIEVABLE", "YOU WON'T BELIEVE"
- **0.0** = Factual, professional writing
- **1.0** = Heavy clickbait/sensationalism
- **Example:**
  - `"Apple Reports Q4 Earnings"` → 0.0
  - `"SHOCKING!!! Apple CRASHES 50%!!!"` → 0.85

#### 2. **Urgency Score** (0-1)
Detects artificial time pressure:
- **Checks for:** BREAKING, URGENT, ACT NOW, LIMITED TIME, time-pressure phrases
- **0.0** = No urgency
- **1.0** = Extreme artificial urgency
- **Example:**
  - `"Apple announces new product"` → 0.0
  - `"BREAKING: BUY APPLE NOW OR MISS OUT!"` → 0.9

#### 3. **Unverified Claims Score** (0-1)
Detects speculation vs. fact:
- **Checks for:** "allegedly", "rumored", "sources say", "unconfirmed", hedging language
- **0.1** = Factual, verified reporting
- **1.0** = Heavy speculation/rumors
- **Example:**
  - `"Apple officially announced..."` → 0.1
  - `"Allegedly, sources say Apple might..."` → 0.7

#### 4. **Source Credibility Score** (0-1)
Based on domain authority:
- **0.95** = Tier 1 (Bloomberg, Reuters, WSJ, Financial Times)
- **0.80** = Tier 2 (CNBC, MarketWatch)
- **0.60** = Tier 3 (Yahoo Finance, Seeking Alpha)
- **0.30** = Unknown/unverified sources
- **Example:**
  - `bloomberg.com` → 0.95
  - `yahoo.com` → 0.60
  - `random-blog.com` → 0.30

#### 5. **Complexity Score** (0-1)
Measures technical depth:
- **Checks for:** Financial jargon density (EBITDA, derivatives, arbitrage, etc.)
- **0.0** = Simple language
- **1.0** = Highly technical
- **Example:**
  - `"Stock price went up"` → 0.2
  - `"EBITDA margins compressed due to derivative hedging..."` → 0.8

#### 6. **COMPOSITE ANOMALY SCORE** (0-1)
**Final weighted score combining all factors:**
- **Formula:** 25% sensationalism + 20% urgency + 25% unverified + 30% (1-credibility)
- **0-0.3** = ✅ CLEAN (Trustworthy, factual)
- **0.3-0.5** = ⚠️  MODERATE RISK (Review manually)
- **0.5-0.7** = ⚠️  HIGH RISK (Multiple red flags)
- **0.7-1.0** = 🚨 VERY HIGH RISK (Likely misinformation)

### Content Tags Extracted

Node 9A also extracts metadata:
- **Topic Classification:** `regulatory_action`, `earnings_report`, `merger_acquisition`, `financial_distress`, `ipo_offering`, `general`
- **Keywords:** SEC, investigation, fraud, bankruptcy, merger, etc.
- **Temporal Context:** `past` (reported, announced), `current`, `future` (plans, expects)
- **Entities:** Mentioned tickers and companies

### Sample Article Labeled by Node 9A

```
ORIGINAL ARTICLE:
  Title: "Apple Reports Strong Q4 Earnings, Beats Estimates"
  Source: Bloomberg (bloomberg.com)

NODE 9A SCORES ADDED:
  📊 Sensationalism: 0.000 (no caps, no clickbait)
  ⏰ Urgency: 0.000 (no time pressure)
  ❓ Unverified: 0.100 (factual reporting)
  ✅ Credibility: 0.950 (Bloomberg = Tier 1)
  🎓 Complexity: 0.300 (some financial terms)
  ⚠️  COMPOSITE ANOMALY: 0.045 (CLEAN - trusted source, factual)

CONTENT TAGS:
  Topic: earnings_report
  Keywords: earnings, revenue, estimates
  Temporal: past
  Entities: AAPL

INTERPRETATION: ✅ HIGH QUALITY ARTICLE - Trust this data
```

### Why This Matters

**Node 5 (Sentiment Analysis) receives cleaned articles with these scores**, allowing it to:
1. Weight sentiment from credible sources higher
2. Discount sentiment from low-quality/sensational sources
3. Filter out potential misinformation before analysis

---

## NODE 4: TECHNICAL ANALYSIS

### Purpose
Analyzes price charts using technical indicators to predict short-term price movements.

### Input Data
- Historical price data (OHLCV) from Node 1
- Typically uses 127 days for indicator calculations

### How It Reaches Its Conclusion

**Step 1: Calculate Indicators**
```python
# RSI (Relative Strength Index) - 14-day
RSI = 65.43
# Interpretation:
- < 30 = Oversold (BUY signal)
- 30-70 = Neutral (HOLD)
- > 70 = Overbought (SELL signal)

# MACD (Moving Average Convergence Divergence)
MACD Line = 2.34
Signal Line = 1.89
Histogram = 0.45 (positive = bullish)
# Interpretation:
- MACD > Signal = BUY
- MACD < Signal = SELL
- Histogram increasing = strengthening trend

# Bollinger Bands (20-day, 2 std dev)
Upper Band = $235.20
Middle Band = $228.50
Lower Band = $221.80
Current Price = $232.00
# Interpretation:
- Price near upper = overbought (SELL)
- Price near lower = oversold (BUY)
- Price at middle = neutral (HOLD)
```

**Step 2: Generate Individual Signals**
```python
RSI Signal: HOLD (30 < 65.43 < 70)
MACD Signal: BUY (MACD > Signal, positive histogram)
Bollinger Signal: HOLD (price in middle range)
```

**Step 3: Combine Signals**
```python
# Voting system:
BUY votes: 1 (MACD)
HOLD votes: 2 (RSI, Bollinger)
SELL votes: 0

# Decision Logic:
if 2+ indicators agree → strong signal
else → HOLD

Result: HOLD (no consensus)
Confidence: 0.55 (moderate)
```

**Output:**
```python
{
    "signal": "HOLD",
    "confidence": 0.55,
    "indicators": {
        "rsi": 65.43,
        "macd": 2.34,
        "bb_position": "middle"
    }
}
```

---

## NODE 5: SENTIMENT ANALYSIS

### Purpose
Analyzes news sentiment using Alpha Vantage scores + FinBERT AI model.

### Input Data
- **Cleaned articles from Node 9A** (with anomaly scores)
- Stock news, market news, related company news

### How It Reaches Its Conclusion

**Step 1: Extract/Analyze Sentiment for Each Article**
```python
for article in cleaned_articles:
    if article has Alpha Vantage sentiment:
        sentiment = article['overall_sentiment_score']  # -1 to +1
    else:
        # Use FinBERT (ProsusAI/finbert) AI model
        sentiment = finbert.analyze(article['text'])
        # FinBERT outputs: positive (0.6-1.0), neutral (0.0), negative (-1.0 to -0.6)
```

**Step 2: Aggregate by News Type**
```python
# Stock News (45 articles):
Average Sentiment: -0.097
Positive: 360 articles
Negative: 227 articles
Neutral: 588 articles
Confidence: 0.68 (based on article count + distribution)

# Market News (1 article):
Average Sentiment: -0.167
Confidence: 0.20 (low due to small sample)

# Related News (0 articles):
Average Sentiment: 0.000
```

**Step 3: Combine with Weights**
```python
combined_sentiment = (
    stock_sentiment * 0.50 +      # 50% weight
    market_sentiment * 0.25 +     # 25% weight
    related_sentiment * 0.25      # 25% weight
)
= (-0.097 * 0.50) + (-0.167 * 0.25) + (0.000 * 0.25)
= -0.048 - 0.042 + 0.000
= -0.090
```

**Step 4: Generate Signal**
```python
if sentiment > +0.2 AND confidence > 0.5:
    signal = "BUY"
elif sentiment < -0.2 AND confidence > 0.5:
    signal = "SELL"
else:
    signal = "HOLD"

Result: HOLD (-0.2 < -0.090 < +0.2)
Confidence: 0.47 (moderate, based on article volume)
```

**Output:**
```python
{
    "signal": "HOLD",
    "confidence": 0.47,
    "combined_sentiment": -0.090,
    "stock_sentiment": -0.097,
    "market_sentiment": -0.167
}
```

**Node 5 Intelligence:**
- Weighs sentiment from high-credibility sources (Bloomberg) higher
- Discounts sentiment from low-credibility sources (unknown blogs)
- Uses FinBERT (financial-domain AI) for nuanced understanding
- Accounts for sarcasm, negation, context in financial language

---

## NODE 6: MARKET CONTEXT ANALYSIS

### Purpose
Analyzes the broader market environment to contextualize the stock's performance.

### Input Data
- Stock's sector/industry (from yfinance)
- Sector ETF performance (e.g., XLK for Technology)
- S&P 500 (SPY) performance and volatility
- Related companies' performance (from Node 3)

### How It Reaches Its Conclusion

**Step 1: Detect Sector**
```python
stock = yfinance.Ticker("AAPL")
sector = stock.info['sector']  # "Technology"
industry = stock.info['industry']  # "Consumer Electronics"
sector_etf = "XLK"  # Technology Select Sector SPDR Fund
```

**Step 2: Analyze Sector Performance**
```python
# Fetch XLK data
xlk_1day = +0.82%
xlk_5day = +2.15%

# Signal Logic:
if performance > +1.0%:
    sector_signal = "POSITIVE"
elif performance < -1.0%:
    sector_signal = "NEGATIVE"
else:
    sector_signal = "NEUTRAL"

Result: NEUTRAL (+0.82% is in [-1%, +1%] range)
```

**Step 3: Analyze Market Trend (SPY)**
```python
# S&P 500 performance
spy_1day = +0.65%
spy_5day = +1.89%
spy_20day = +5.32%
volatility = 12.4% (annualized)

# Trend Logic:
if 5day > +1% AND 20day > +3%:
    trend = "UPTREND"
elif 5day < -1% AND 20day < -3%:
    trend = "DOWNTREND"
else:
    trend = "SIDEWAYS"

Result: UPTREND
Signal: POSITIVE
```

**Step 4: Analyze Related Companies**
```python
# From Node 3: WDC, SNDK, DELL, HPE, PSTG
related_avg = +0.73%

# Signal Logic:
if avg > +0.5%:
    related_signal = "POSITIVE"
elif avg < -0.5%:
    related_signal = "NEGATIVE"
else:
    related_signal = "NEUTRAL"

Result: POSITIVE
```

**Step 5: Calculate Correlation with Market**
```python
# 30-day correlation with SPY
correlation = 0.78
# Interpretation: 0.78 = strong positive correlation with market

beta = 1.12
# Interpretation: 1.12 = 12% more volatile than market
```

**Step 6: Combine Factors**
```python
context_score = (
    sector_score * 0.25 +       # 25% weight
    market_score * 0.30 +       # 30% weight
    related_score * 0.20 +      # 20% weight
    correlation_score * 0.25    # 25% weight
)

= (0 * 0.25) + (+1 * 0.30) + (+1 * 0.20) + (+0.5 * 0.25)
= 0 + 0.30 + 0.20 + 0.125
= +0.625

# Signal Logic:
if score > 0:
    context_signal = "POSITIVE"
elif score < 0:
    context_signal = "NEGATIVE"
else:
    context_signal = "NEUTRAL"

Result: POSITIVE
Confidence: 0.72
```

**Output:**
```python
{
    "context_signal": "POSITIVE",
    "confidence": 0.72,
    "sector": {
        "name": "Technology",
        "performance": +0.82%,
        "signal": "NEUTRAL"
    },
    "market_trend": {
        "trend": "UPTREND",
        "performance": +1.89%,
        "signal": "POSITIVE"
    },
    "related_companies": {
        "average_performance": +0.73%,
        "signal": "POSITIVE"
    },
    "correlation": {
        "coefficient": 0.78,
        "beta": 1.12
    }
}
```

**Node 6 Intelligence:**
- Understands that a stock's performance is influenced by its sector
- Considers overall market momentum (rising tide lifts all boats)
- Accounts for peer performance (if competitors are up, why isn't this stock?)
- Uses correlation/beta to measure market sensitivity

---

## NODE 7: FUNDAMENTAL ANALYSIS

### Purpose
Analyzes company financials to determine intrinsic value.

### Input Data
- Financial metrics from Finnhub API
- P/E ratio, P/B ratio, market cap
- Current ratio, debt-to-equity, ROE
- Revenue/earnings growth rates

### How It Reaches Its Conclusion

**Step 1: Calculate Valuation Score**
```python
# P/E Ratio (Price-to-Earnings)
PE = 32.5
if PE < 15:
    pe_score = 1.0  # Undervalued
elif PE < 25:
    pe_score = 0.5  # Fair
else:
    pe_score = 0.0  # Overvalued

# P/B Ratio (Price-to-Book)
PB = 2.3
if PB < 1.0:
    pb_score = 1.0  # Undervalued
elif PB < 3.0:
    pb_score = 0.5  # Fair
else:
    pb_score = 0.0  # Overvalued

valuation_score = (pe_score + pb_score) / 2
= (0.0 + 0.5) / 2 = 0.25 (Expensive)
```

**Step 2: Calculate Financial Health Score**
```python
# Current Ratio (Can company pay short-term debt?)
current_ratio = 1.8
if current_ratio > 1.5:
    cr_score = 1.0  # Good
elif current_ratio > 1.0:
    cr_score = 0.5  # Fair
else:
    cr_score = 0.0  # Poor

# Debt-to-Equity (Leverage risk)
debt_to_equity = 1.2
if debt_to_equity < 1.0:
    de_score = 1.0  # Low leverage (good)
elif debt_to_equity < 2.0:
    de_score = 0.5  # Moderate
else:
    de_score = 0.0  # High leverage (risky)

# ROE (Return on Equity - profitability)
roe = 18.5%
if roe > 15%:
    roe_score = 1.0  # Strong
elif roe > 10%:
    roe_score = 0.5  # Fair
else:
    roe_score = 0.0  # Weak

health_score = (cr_score + de_score + roe_score) / 3
= (1.0 + 0.5 + 1.0) / 3 = 0.83 (Strong)
```

**Step 3: Calculate Growth Score**
```python
revenue_growth = 8.2%
earnings_growth = 12.5%

if growth > 15%:
    growth_score = 1.0  # Strong growth
elif growth > 5%:
    growth_score = 0.5  # Moderate growth
else:
    growth_score = 0.0  # Low/negative growth

growth_score = 0.5 (Moderate)
```

**Step 4: Combine Scores**
```python
fundamental_score = (
    valuation_score * 0.40 +    # 40% weight
    health_score * 0.30 +       # 30% weight
    growth_score * 0.30         # 30% weight
)

= (0.25 * 0.40) + (0.83 * 0.30) + (0.5 * 0.30)
= 0.10 + 0.25 + 0.15
= 0.50

# Signal Logic:
if score > 0.6:
    signal = "BUY"
elif score < 0.4:
    signal = "SELL"
else:
    signal = "HOLD"

Result: HOLD (0.4 < 0.50 < 0.6)
Confidence: 0.65
```

**Output:**
```python
{
    "signal": "HOLD",
    "confidence": 0.65,
    "fundamental_score": 0.50,
    "valuation": {
        "pe_ratio": 32.5,
        "pb_ratio": 2.3,
        "score": 0.25  # Expensive
    },
    "health": {
        "current_ratio": 1.8,
        "debt_to_equity": 1.2,
        "roe": 18.5%,
        "score": 0.83  # Strong
    },
    "growth": {
        "revenue_growth": 8.2%,
        "earnings_growth": 12.5%,
        "score": 0.5  # Moderate
    }
}
```

**Node 7 Intelligence:**
- Distinguishes between expensive (high P/E) and overvalued (poor fundamentals + high P/E)
- Understands that high-growth companies deserve higher P/E ratios
- Weighs multiple financial health indicators to avoid false positives
- Accounts for industry-specific norms (tech companies typically have higher P/E)

---

## Summary: How Everything Works Together

### Data Flow
```
Node 1 → Price Data
Node 3 → Related Companies
Node 2 → Raw News (6 months)
   ↓
Node 9A → Labels & Scores Articles
   ↓
PARALLEL EXECUTION:
├─ Node 4: Technical Analysis (from price data)
├─ Node 5: Sentiment Analysis (from cleaned news)
├─ Node 6: Market Context (from sector/market/peers)
└─ Node 7: Fundamental Analysis (from financials)
   ↓
Node 8: Combine all signals → Final recommendation
```

### Node 9A's Critical Role

**Before Node 9A:**
- Raw news from sources of varying quality
- No way to distinguish Bloomberg from random blogs
- Sentiment analysis would weight all sources equally
- Misinformation/clickbait could skew results

**After Node 9A:**
- Every article scored for quality (0-1 composite anomaly score)
- Node 5 can weight credible sources higher
- Low-quality articles flagged for review/filtering
- Transparent scoring allows downstream nodes to make informed decisions

### Real Example

**Article Input:**
```
Source: unknown-blog.com
Title: "APPLE STOCK TO THE MOON!!! BUY NOW!!!"
Content: "Sources say Apple might release revolutionary product..."
```

**Node 9A Scores:**
```
Sensationalism: 0.85 (ALL CAPS, !!!)
Urgency: 0.90 (BUY NOW)
Unverified: 0.95 (sources say, might)
Credibility: 0.30 (unknown blog)
COMPOSITE ANOMALY: 0.78 (🚨 VERY HIGH RISK)
```

**Node 5 Response:**
```python
# Sentiment from article: +0.90 (very positive)
# But credibility score: 0.30 (low)
# Anomaly score: 0.78 (high risk)

# Node 5 decision:
weight = 0.30 * (1 - 0.78) = 0.066
# This article contributes only 6.6% of its sentiment to final score

# If it was Bloomberg (0.95 credibility, 0.05 anomaly):
weight = 0.95 * (1 - 0.05) = 0.903
# Bloomberg contributes 90.3% of its sentiment
```

**Result:** High-quality sources drive decisions, low-quality sources are discounted.

---

## Is Everything Labeled Correctly?

### Node 9A Accuracy

Based on the analysis:

**✅ Correctly Labeled:**
- **Source Credibility:** Tier system is accurate (Bloomberg=0.95, Yahoo=0.6)
- **Sensationalism Detection:** Successfully identifies ALL CAPS, !!!, clickbait keywords
- **Urgency Detection:** Catches BREAKING, URGENT, time-pressure phrases
- **Unverified Claims:** Detects hedging language (allegedly, rumored, sources say)

**⚠️  Potential Improvements:**
1. **Complexity Score:** Currently measures jargon density, but doesn't distinguish between "good technical" (earnings report) vs. "confusing technical" (obfuscation)
2. **Context-Aware:** Doesn't understand that "BREAKING" from Reuters is legitimate news, vs. "BREAKING" from a blog
3. **Sarcasm/Irony:** FinBERT helps, but Node 9A scoring doesn't detect sarcastic clickbait

**Overall Assessment:** Node 9A is labeling correctly based on its design. The scoring is objective and consistent.

---

## Conclusion

The workflow is **well-designed and labels are correct**. Node 9A acts as an intelligent filter that ensures downstream nodes (especially Node 5) receive quality-scored data, allowing them to make informed decisions based on source credibility and content quality.
