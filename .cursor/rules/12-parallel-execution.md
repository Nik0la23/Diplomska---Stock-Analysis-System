---
description: "Async patterns for parallel API calls and node execution"
alwaysApply: false
globs: ["**/node_02*.py", "**/news*.py", "**/async*.py"]
---

# Parallel Execution Patterns

## Async for Multiple API Calls

Use `asyncio.gather()` for parallel API calls in Node 2 (news fetching).

```python
import asyncio

async def fetch_stock_news_async(ticker: str) -> List[Dict]:
    """Fetch stock news asynchronously"""
    # API call here
    pass

async def fetch_market_news_async() -> List[Dict]:
    """Fetch market news asynchronously"""
    # API call here
    pass

async def fetch_related_news_async(tickers: List[str]) -> List[Dict]:
    """Fetch related company news asynchronously"""
    # API call here
    pass

def fetch_all_news_node(state: StockAnalysisState) -> StockAnalysisState:
    """Fetch all news sources IN PARALLEL - 3x faster"""
    ticker = state['ticker']
    related = state.get('raw_related_companies', [])
    
    # Run all 3 fetches in parallel
    loop = asyncio.get_event_loop()
    stock_news, market_news, related_news = loop.run_until_complete(
        asyncio.gather(
            fetch_stock_news_async(ticker),
            fetch_market_news_async(),
            fetch_related_news_async(related)
        )
    )
    
    state['raw_stock_news'] = stock_news
    state['raw_market_news'] = market_news
    state['raw_related_news'] = related_news
    
    return state
```

## LangGraph Parallel Nodes

Nodes 4, 5, 6, 7 run in parallel (defined in workflow):

```python
# In workflow definition
graph.add_node("technical_analysis", technical_analysis_node)
graph.add_node("sentiment_analysis", sentiment_analysis_node)
graph.add_node("market_context", market_context_node)
graph.add_node("monte_carlo", monte_carlo_node)

# These run simultaneously after Node 9A
graph.add_edge("early_anomaly", "technical_analysis")
graph.add_edge("early_anomaly", "sentiment_analysis")
graph.add_edge("early_anomaly", "market_context")
graph.add_edge("early_anomaly", "monte_carlo")
```

## Performance Benefits

- **Sequential:** 2s + 4s + 2s + 3s = 11 seconds
- **Parallel:** max(2s, 4s, 2s, 3s) = 4 seconds
- **Speed Gain:** 175% faster (2.75x improvement)

## When to Use

- Multiple independent API calls (Node 2)
- Multiple independent calculations (Nodes 4-7)
- I/O-bound operations (database reads)

## When NOT to Use

- Operations with dependencies (Node 8 needs results from Nodes 4-7)
- Sequential pipeline (Node 10 → Node 11 → Node 12)
- Single API call
