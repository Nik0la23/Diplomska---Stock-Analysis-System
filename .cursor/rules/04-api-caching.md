---
description: "API rate limiting and caching strategy - check cache before API calls"
alwaysApply: true
---

# API Rate Limiting & Caching

**Strategy:** Check SQLite cache BEFORE making API calls. Saves 80% of API requests.

## Cache-First Pattern

```python
def fetch_data_node(state: StockAnalysisState) -> StockAnalysisState:
    ticker = state['ticker']
    
    # 1. Check cache FIRST
    cached_data = get_cached_data(ticker, max_age_hours=24)
    if cached_data is not None:
        logger.info(f"Using cached data for {ticker}")
        state['data'] = cached_data
        return state
    
    # 2. Make API call only if needed
    try:
        data = fetch_from_api(ticker)
        cache_data(ticker, data)  # Cache for next time
        state['data'] = data
        return state
    except Exception as e:
        logger.error(f"API failed: {str(e)}")
        state['errors'].append(f"API fetch failed: {str(e)}")
        state['data'] = None
        return state
```

## Rate Limiting Decorator

```python
from functools import wraps
import time

def rate_limit(calls_per_minute: int = 60):
    """Decorator for rate limiting API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(60 / calls_per_minute)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

## Rules

- Check cache before every API call
- Cache results for 24 hours minimum
- Use exponential backoff for retries
- Log all API calls with timestamps
- Use `time.sleep()` between bulk requests
