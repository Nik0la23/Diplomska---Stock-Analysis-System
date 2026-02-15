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

**Inside the workflow**, as the node `run_background_outcomes` in `src/graph/workflow.py`. It runs **after** the parallel layer (Nodes 4–7) and **before** Node 8 (News Verification). So the background script always finishes before Node 8 reads `news_outcomes`.

## Flow Diagram

```
FIRST RUN (new ticker):
  Nodes 1-2 → fetch 6 months → store in DB
  … → parallel (4,5,6,7) → run_background_outcomes (pending > 0 → evaluate) → Node 8 learns from outcomes

SUBSEQUENT RUN (< 7 days later):
  Nodes 1-2 → fetch only new days (or use cache)
  … → parallel → run_background_outcomes (pending = 0 → skip) → Node 8 uses existing outcomes

SUBSEQUENT RUN (≥ 7 days later):
  Nodes 1-2 → fetch only new days
  … → parallel → run_background_outcomes (pending > 0 → evaluate newly aged articles) → Node 8 has bigger learning dataset
```

## How “Pending” Is Determined

Nothing is inferred from what Node 1 or Node 2 did in this run. The workflow asks the **database**:

- **Source of articles:** Node 2 (and earlier runs) writes to `news_articles` when it caches news. So by the time we reach the background-outcomes node, the DB already has the latest news for this ticker.
- **Query:** `get_news_outcomes_pending(ticker)` returns rows from `news_articles` that:
  - belong to this ticker,
  - have `published_at <= today - 7 days`,
  - and have **no** row in `news_outcomes` (LEFT JOIN … WHERE no.id IS NULL).

If that list is non-empty, the background script runs and fills `news_outcomes` for those articles; then Node 8 runs and sees the updated outcomes.

## Key Points
- The background script runs as a workflow node **before** Node 8 (so Node 8 always sees up-to-date outcomes when it runs).
- Node 8 queries `news_outcomes` fresh every run (no caching).
- The learning dataset grows over time as more articles age and get evaluated.
- Nodes 1–2 only fetch data that's not already in the database.
