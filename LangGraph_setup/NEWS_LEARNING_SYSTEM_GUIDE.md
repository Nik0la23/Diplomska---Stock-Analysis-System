# News Impact Learning System - Complete Guide
## How Your System Learns Which News Sources Are Reliable

**Last Updated:** February 2026  
**For:** Understanding Node 8 and the News Learning Innovation

---

## 🎯 The Big Picture

### What Problem Are We Solving?

**Problem:** Not all news is equal!
- Bloomberg.com articles are usually accurate
- Random blogs are often wrong or manipulative
- The same source might be good for NVIDIA but bad for penny stocks

**Solution:** Learn from history which sources we should trust

---

## 📖 How It Works - Step by Step

### Step 1: Collect Historical News (Past 6 Months)

```python
# Example: What we stored 6 months ago for NVIDIA

Article 1:
- Source: Bloomberg.com
- Published: January 5, 2024
- Title: "NVIDIA announces new AI chip"
- Sentiment: POSITIVE (from FinBERT)
- Price at time: $500

Article 2:
- Source: random-penny-stocks-blog.com
- Published: January 8, 2024
- Title: "NVIDIA will crash tomorrow!"
- Sentiment: NEGATIVE (from FinBERT)
- Price at time: $505

Article 3:
- Source: Bloomberg.com
- Published: February 10, 2024
- Title: "NVIDIA beats quarterly earnings"
- Sentiment: POSITIVE (from FinBERT)
- Price at time: $520
```

---

### Step 2: Wait 7 Days and Check What Actually Happened

This is done by the **background task** (runs daily):

```python
# For Article 1 (Bloomberg - Positive news):
Published: January 5
Price on Jan 5: $500
Price on Jan 12 (7 days later): $525

Price Change: +5% (went UP)
Sentiment was: POSITIVE
Actual movement: UP
Result: ✅ ACCURATE!

---

# For Article 2 (Random blog - Negative news):
Published: January 8
Price on Jan 8: $505
Price on Jan 15 (7 days later): $530

Price Change: +4.9% (went UP)
Sentiment was: NEGATIVE (said it would crash)
Actual movement: UP
Result: ❌ WRONG!

---

# For Article 3 (Bloomberg - Positive news):
Published: February 10
Price on Feb 10: $520
Price on Feb 17 (7 days later): $545

Price Change: +4.8% (went UP)
Sentiment was: POSITIVE
Actual movement: UP
Result: ✅ ACCURATE!
```

---

### Step 3: Store Results in Database

All of this goes into the **news_outcomes** table:

```sql
-- For Article 1
INSERT INTO news_outcomes (
    news_id: 1,
    ticker: 'NVDA',
    price_at_news: 500,
    price_7day_later: 525,
    price_change_7day: 5.0,
    predicted_direction: 'UP',
    actual_direction: 'UP',
    prediction_was_accurate_7day: TRUE
)

-- For Article 2
INSERT INTO news_outcomes (
    news_id: 2,
    ticker: 'NVDA',
    price_at_news: 505,
    price_7day_later: 530,
    price_change_7day: 4.9,
    predicted_direction: 'DOWN',
    actual_direction: 'UP',
    prediction_was_accurate_7day: FALSE
)

-- For Article 3
INSERT INTO news_outcomes (
    news_id: 3,
    ticker: 'NVDA',
    price_at_news: 520,
    price_7day_later: 545,
    price_change_7day: 4.8,
    predicted_direction: 'UP',
    actual_direction: 'UP',
    prediction_was_accurate_7day: TRUE
)
```

---

### Step 4: Calculate Source Reliability (Node 8 Does This)

After 6 months, we have lots of data. Node 8 analyzes it:

```python
Bloomberg.com for NVDA:
- Total articles: 20
- Accurate predictions: 17
- Accuracy Rate: 17/20 = 85%
- Average price impact: +2.3%
- Confidence Multiplier: 1.2x (boost confidence by 20%)

random-penny-stocks-blog.com for NVDA:
- Total articles: 10
- Accurate predictions: 2
- Accuracy Rate: 2/10 = 20%
- Average price impact: -0.5%
- Confidence Multiplier: 0.5x (reduce confidence by 50%)
```

This gets saved to **source_reliability** table.

---

### Step 5: Use This Knowledge Today!

When NEW news arrives today:

```python
# Today's News Article:
Source: Bloomberg.com
Title: "NVIDIA partners with Microsoft for cloud AI"
Sentiment: POSITIVE (from FinBERT)
FinBERT Confidence: 75%

# Node 8 Adjusts:
System checks: "Bloomberg.com is 85% accurate for NVDA"
System checks: "Their confidence multiplier is 1.2x"

Adjusted Confidence = 75% × 1.2 = 90%

Final Result:
- Sentiment: POSITIVE
- Original Confidence: 75%
- Adjusted Confidence: 90% ← Because Bloomberg is reliable!
- Weight for news signal: INCREASED

---

# Today's Another Article:
Source: random-penny-stocks-blog.com
Title: "NVIDIA will crash this week"
Sentiment: NEGATIVE (from FinBERT)
FinBERT Confidence: 70%

# Node 8 Adjusts:
System checks: "random-penny-stocks-blog.com is 20% accurate for NVDA"
System checks: "Their confidence multiplier is 0.5x"

Adjusted Confidence = 70% × 0.5 = 35%

Final Result:
- Sentiment: NEGATIVE
- Original Confidence: 70%
- Adjusted Confidence: 35% ← Because this source is unreliable!
- Weight for news signal: DECREASED
```

---

## 🗄️ Database Tables Explained

### Table 1: news_articles
**What it stores:** Every news article we fetch

```sql
id | ticker | source              | title                    | sentiment | published_at
1  | NVDA   | Bloomberg.com       | "New AI chip announced"  | positive  | 2024-01-05
2  | NVDA   | random-blog.com     | "Stock will crash"       | negative  | 2024-01-08
```

### Table 2: news_outcomes
**What it stores:** What happened AFTER each news article

```sql
news_id | price_at_news | price_7day_later | price_change_7day | prediction_accurate
1       | 500           | 525              | +5.0%             | TRUE
2       | 505           | 530              | +4.9%             | FALSE
```

### Table 3: source_reliability
**What it stores:** Calculated reliability scores per source per stock

```sql
ticker | source          | total_articles | accurate_predictions | accuracy_rate
NVDA   | Bloomberg.com   | 20             | 17                   | 0.85
NVDA   | random-blog.com | 10             | 2                    | 0.20
```

---

## 🔄 The Complete Learning Flow

```
┌─────────────────────────────────────────────────┐
│  6 MONTHS AGO                                   │
│  News Article Published                         │
│  "NVIDIA announces new chip"                    │
│  Source: Bloomberg.com                          │
│  Sentiment: POSITIVE                            │
│  Price: $500                                    │
└────────────────┬────────────────────────────────┘
                 │
                 │ Store in news_articles table
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  7 DAYS LATER (Background Task Runs Daily)      │
│  Check: What was the price 7 days later?        │
│  Answer: $525 (went UP by 5%)                   │
│  Prediction was: POSITIVE (up)                  │
│  Actual was: UP                                 │
│  Result: ACCURATE ✅                            │
└────────────────┬────────────────────────────────┘
                 │
                 │ Store in news_outcomes table
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  TODAY (Node 8 Runs)                            │
│  Analyze 6 months of outcomes                   │
│  Calculate: Bloomberg.com = 85% accurate        │
│  Store in source_reliability table              │
└────────────────┬────────────────────────────────┘
                 │
                 │ Use this knowledge
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  NEW NEWS ARRIVES TODAY                         │
│  Source: Bloomberg.com                          │
│  System thinks: "This source is 85% reliable"   │
│  Action: BOOST confidence in this news          │
│  Give higher weight to news signal              │
└─────────────────────────────────────────────────┘
```

---

## 📊 What Data We Compare

### We Need TWO Sources of Data:

**1. News Data (from NewsAPI):**
```python
{
    'source': 'Bloomberg.com',
    'title': 'NVIDIA announces partnership',
    'sentiment': 'POSITIVE',
    'published_at': '2024-01-05',
    'confidence': 85%
}
```

**2. Price Data (from Finnhub/yfinance):**
```python
{
    'date': '2024-01-05',
    'close': 500.00
},
{
    'date': '2024-01-12',  # 7 days later
    'close': 525.00
}
```

### We Match Them:

```python
IF (News Sentiment = POSITIVE) AND (Price went UP):
    → Source was ACCURATE ✅

IF (News Sentiment = POSITIVE) AND (Price went DOWN):
    → Source was WRONG ❌

IF (News Sentiment = NEGATIVE) AND (Price went DOWN):
    → Source was ACCURATE ✅

IF (News Sentiment = NEGATIVE) AND (Price went UP):
    → Source was WRONG ❌
```

---

## 🎯 Example: Full Learning Cycle for NVIDIA

### Month 1 (January)
```
Week 1: 
- Bloomberg says "New chip" (positive)
- Price: $500 → $525 (UP)
- Result: Bloomberg = 1/1 = 100% ✅

Week 2:
- Random blog says "Will crash" (negative)  
- Price: $505 → $530 (UP)
- Result: Random blog = 0/1 = 0% ❌

Week 3:
- Bloomberg says "Strong demand" (positive)
- Price: $510 → $535 (UP)
- Result: Bloomberg = 2/2 = 100% ✅

Week 4:
- Random blog says "Buy now!" (positive)
- Price: $520 → $515 (DOWN)
- Result: Random blog = 0/2 = 0% ❌
```

### Month 6 (June) - After 6 Months

**Bloomberg.com:**
- Articles: 20
- Correct: 17
- **Accuracy: 85%**
- **Confidence Multiplier: 1.2x**

**random-blog.com:**
- Articles: 10
- Correct: 2
- **Accuracy: 20%**
- **Confidence Multiplier: 0.5x**

### Today (Using Learned Knowledge)

**New Article from Bloomberg:**
```python
Original sentiment confidence: 75%
Adjusted confidence: 75% × 1.2 = 90%
Signal weight: INCREASED
Trust level: HIGH
```

**New Article from Random Blog:**
```python
Original sentiment confidence: 70%
Adjusted confidence: 70% × 0.5 = 35%
Signal weight: DECREASED
Trust level: LOW
```

---

## 🔧 Technical Implementation

### Background Task (Runs Daily)
```python
# File: scripts/update_news_outcomes.py

def update_news_outcomes_daily():
    """
    Run this every day to build historical dataset
    """
    
    # For all stocks we're tracking
    for ticker in ['AAPL', 'NVDA', 'TSLA', 'MSFT']:
        
        # Find news articles that are 7+ days old without outcomes
        old_news = get_news_without_outcomes(ticker, days_old=7)
        
        for article in old_news:
            # Get price 7 days after article
            published_date = article['published_at']
            future_price = get_price_on_date(ticker, published_date + 7 days)
            
            # Calculate if prediction was accurate
            predicted = 'UP' if article['sentiment'] == 'positive' else 'DOWN'
            actual = 'UP' if future_price > article_price else 'DOWN'
            accurate = (predicted == actual)
            
            # Save to database
            save_outcome(article_id, prices, accurate)
```

### Node 8 (Runs During Analysis)
```python
# File: src/langgraph_nodes/node_08_news_verification.py

def news_verification_node(state):
    """
    Runs when user asks to analyze a stock
    """
    
    # Get 6 months of historical outcomes
    history = get_news_with_outcomes(ticker, days=180)
    
    # Calculate source reliability
    bloomberg_accuracy = calculate_accuracy('Bloomberg.com', history)
    # Result: 85%
    
    random_blog_accuracy = calculate_accuracy('random-blog.com', history)
    # Result: 20%
    
    # Adjust today's sentiment confidence
    for today_article in current_news:
        if today_article['source'] == 'Bloomberg.com':
            # Boost confidence because Bloomberg is reliable
            today_article['confidence'] *= 1.2
        elif today_article['source'] == 'random-blog.com':
            # Reduce confidence because blog is unreliable
            today_article['confidence'] *= 0.5
    
    return updated_state
```

---

## 🎓 Why This Is Your Thesis Innovation

### What Makes This Special:

1. **Stock-Specific Learning**
   - Bloomberg might be great for NVIDIA but terrible for penny stocks
   - We learn separately for each stock

2. **Source-Specific Learning**
   - Not all news sources are equal
   - We track which websites are trustworthy

3. **Type-Specific Learning**
   - Stock news might be more predictive than market news
   - We track effectiveness by news type

4. **Continuous Improvement**
   - System gets smarter over time
   - More history = better predictions

5. **Transparent & Explainable**
   - Users can see which sources are reliable
   - Dashboard shows reliability scores

---

## 📈 Expected Results

After 6 months of learning:

**Accuracy Improvements:**
- Sentiment signal without learning: ~60%
- Sentiment signal with learning: ~70-75%
- **Improvement: +10-15%** ← This is your thesis contribution!

**Source Reliability Distribution:**
- High-quality sources (Bloomberg, Reuters): 80-90% accuracy
- Medium sources (CNBC, MarketWatch): 60-70% accuracy
- Low-quality sources (blogs, forums): 20-40% accuracy

---

## 🚀 User Experience

### In the Dashboard:

**Tab: "News & Sentiment"**
```
📰 Current News Analysis for NVIDIA

Recent Articles:
1. Bloomberg.com: "NVIDIA partners with Microsoft"
   Sentiment: POSITIVE
   Reliability: 🟢 HIGH (85% accurate historically)
   Confidence: 90% ←adjusted from 75%
   
2. random-blog.com: "Stock will crash"
   Sentiment: NEGATIVE
   Reliability: 🔴 LOW (20% accurate historically)
   Confidence: 35% ←adjusted from 70%
   ⚠️ This source is often wrong - ignoring

Final Sentiment Signal: BUY (confidence: 88%)
Primary sources: Bloomberg, Reuters (both highly reliable)
```

---

## ✅ Summary

### The Learning System in One Sentence:

**We look at 6 months of past news, check if each prediction came true by comparing to actual price movement, calculate which sources were accurate, and use that knowledge to trust good sources and ignore bad ones when analyzing today's news.**

### Key Tables:
1. **news_articles** - All news we collect
2. **news_outcomes** - What happened after each news
3. **source_reliability** - Which sources are trustworthy

### Process:
1. Collect historical news (6 months)
2. Wait 7 days, check if prediction came true
3. Calculate accuracy per source per stock
4. Use this to adjust confidence in new news
5. Give higher weight to reliable sources

**This is what makes your thesis innovative!** 🎯

---

*This learning system is Node 8 - the core intelligence that makes your system better than static analysis tools.*
