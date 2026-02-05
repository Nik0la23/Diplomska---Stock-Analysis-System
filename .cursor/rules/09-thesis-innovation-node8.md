---
description: "Node 8 news learning system - PRIMARY THESIS INNOVATION with 10-15% accuracy improvement"
alwaysApply: false
globs: ["**/node_08*.py", "**/news_verification*.py", "**/learning*.py"]
---

# Node 8: News Learning System (THESIS INNOVATION)

**THIS IS YOUR PRIMARY THESIS CONTRIBUTION.**

## What Node 8 Does

Learns from 6 months of historical news-price correlations to identify which news sources are reliable for each stock.

## Core Algorithm

```
1. Get historical news with outcomes from database
   Query: news_with_outcomes where ticker = 'NVDA' and date > 6 months ago
   
2. Calculate source reliability
   For each source (Bloomberg.com, Reuters, random-blog.com):
     - Count total articles
     - Count accurate predictions (sentiment matched price movement)
     - Calculate accuracy rate = accurate / total
   
3. Calculate confidence multipliers
   High accuracy (>80%) → multiplier = 1.2x (boost confidence)
   Low accuracy (<60%) → multiplier = 0.5x (reduce confidence)
   
4. Adjust current sentiment confidence
   Bloomberg article (85% accurate historically):
     Original: 75% → Adjusted: 75% × 1.2 = 90% (BOOSTED)
   
   Random blog (20% accurate historically):
     Original: 70% → Adjusted: 70% × 0.5 = 35% (REDUCED)
```

## Expected Impact

- Sentiment accuracy WITHOUT Node 8: ~62%
- Sentiment accuracy WITH Node 8: ~73%
- **Improvement: +11%** ← This is your thesis contribution!

## Critical Database Tables

1. **news_outcomes** - Tracks what happened 7 days after each news
2. **source_reliability** - Stores accuracy scores per source per stock
3. **news_type_effectiveness** - Which news types work best

## Reference Implementation

See `@LangGraph_setup/node_08_news_verification_COMPLETE.py` for complete working implementation.

See `@LangGraph_setup/NEWS_LEARNING_SYSTEM_GUIDE.md` for conceptual explanation.

## Dependencies

- **Runs AFTER:** Nodes 4, 5, 6, 7 (needs their analysis)
- **Runs BEFORE:** Node 9B (behavioral anomaly detection)

## Background Task Required

Create `scripts/update_news_outcomes.py` - Runs daily to populate `news_outcomes` table by checking 7-day price changes for historical news.

## Key Implementation Points

- Handle insufficient historical data (< 30 articles) - use defaults
- Calculate source-specific AND stock-specific reliability
- Adjust confidence for ALL current news based on source reliability
- Store results in `news_impact_verification` state field
- This enables adaptive weighting in Node 11 to give higher weight to improved sentiment signal
