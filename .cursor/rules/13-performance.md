---
description: "Performance optimization for <5 second total execution time"
alwaysApply: false
---

# Performance Optimization

**Target:** Total execution time < 5 seconds for entire workflow.

## Use Pandas Vectorized Operations

```python
# ✅ GOOD - Vectorized (fast)
def calculate_returns(df: pd.DataFrame) -> pd.Series:
    return df['close'].pct_change()

# ❌ BAD - Loop (slow)
def calculate_returns(df: pd.DataFrame) -> List[float]:
    returns = []
    for i in range(1, len(df)):
        ret = (df['close'][i] - df['close'][i-1]) / df['close'][i-1]
        returns.append(ret)
    return returns
```

## Cache Expensive Calculations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_sector_etf(sector: str) -> str:
    """Cache sector to ETF mapping"""
    return SECTOR_ETFS.get(sector, 'SPY')
```

## Limit API Result Sizes

```python
# Fetch only what you need
articles = newsapi.get_everything(
    q=ticker,
    page_size=50,  # Don't fetch 100+ if you only use 20
    sort_by='relevancy'
)
return articles[:20]  # Use only top 20
```

## Use ThreadPoolExecutor for I/O

```python
from concurrent.futures import ThreadPoolExecutor

def analyze_multiple_stocks(tickers: List[str]) -> Dict:
    """Process multiple stocks in parallel"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(analyze_stock, tickers)
    return dict(zip(tickers, results))
```

## Profile if Any Node > 2 Seconds

```python
# Add timing to slow nodes
import time
start = time.time()
result = expensive_operation()
elapsed = time.time() - start
logger.warning(f"Slow operation: {elapsed:.2f}s")
```

## Per-Node Time Targets

- Node 1 (Price fetching): < 2s
- Node 2 (News fetching): < 4s
- Node 4 (Technical): < 1s
- Node 5 (Sentiment): < 4s (FinBERT is slow)
- Node 6 (Market context): < 3s
- Node 7 (Monte Carlo): < 3s
- Node 8 (Learning): < 2s
- All others: < 1s each

Total: ~4-5 seconds with parallel execution.
