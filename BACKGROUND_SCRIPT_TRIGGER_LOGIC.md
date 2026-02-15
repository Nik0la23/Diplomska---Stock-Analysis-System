# Background Script: When & Why It Runs

## What It Does
Evaluates historical news articles to check if their sentiment predictions were correct. Fills the `news_outcomes` table that Node 8 learns from.

## When It Runs

| Scenario | Background Script? | Why |
|----------|-------------------|-----|
| First time ticker (NVDA never seen) | ✅ YES | 6 months of articles need evaluation |
| Same ticker, 2 days later | ❌ NO | No new articles are 7+ days old yet |
| Same ticker, 10 days later | ✅ YES | Earlier articles have aged past 7 days |

## Trigger Logic

Don't check dates manually. Use `get_news_outcomes_pending()`:

```python
pending = get_news_outcomes_pending(ticker=ticker)
if len(pending) > 0:
    run_evaluation(ticker=ticker)
```

If it returns articles → run. If zero → skip. The query already filters for 7+ days old with no outcome.

## Where The Trigger Lives

In graph initialization or a wrapper script. NOT inside any node.

## Flow Diagram

```
FIRST RUN (new ticker):
  Nodes 1-2 → fetch 6 months → store in DB
  Background script → evaluate all articles → fill news_outcomes
  Nodes 3-15 → Node 8 learns from outcomes

SUBSEQUENT RUN (< 7 days later):
  Nodes 1-2 → fetch only new days
  Background script → skipped (nothing pending)
  Nodes 3-15 → Node 8 uses existing outcomes

SUBSEQUENT RUN (≥ 7 days later):
  Nodes 1-2 → fetch only new days
  Background script → evaluate newly aged articles
  Nodes 3-15 → Node 8 has bigger learning dataset
```

## Key Points
- The script is NOT part of the LangGraph pipeline
- Node 8 queries `news_outcomes` fresh every run (no caching)
- The learning dataset grows over time as more articles age and get evaluated
- Nodes 1-2 only fetch data that's not already in the database
