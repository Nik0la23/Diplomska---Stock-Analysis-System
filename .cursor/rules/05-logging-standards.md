---
description: "Structured logging with consistent format throughout the project"
alwaysApply: true
---

# Logging Standards

## Setup (Once per File)

```python
import logging

logger = logging.getLogger(__name__)
```

## Required Logging Points

```python
def node_name(state: StockAnalysisState) -> StockAnalysisState:
    ticker = state['ticker']
    start_time = datetime.now()
    
    # 1. Log at entry
    logger.info(f"Node X started for ticker {ticker}")
    
    try:
        result = do_work()
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # 2. Log at success with timing
        logger.info(f"Node X completed in {elapsed:.2f}s: {result['signal']}")
        return state
        
    except Exception as e:
        # 3. Log errors with context
        logger.error(f"Node X failed for {ticker}: {str(e)}")
        return state
```

## Log Levels

- **INFO:** Node entry/exit, important decisions
- **WARNING:** Non-critical issues, fallbacks used
- **ERROR:** Exceptions, failures
- **DEBUG:** Detailed diagnostic information (use sparingly)

## Format Examples

```python
# Node flow
logger.info(f"Node 5: Analyzing sentiment for {ticker}")
logger.info(f"Node 5: Completed in 3.2s")

# Important decisions
logger.info(f"Signal: BUY (confidence: 85%)")
logger.info(f"Using cached data for {ticker}")

# Errors
logger.error(f"Node 8 failed: Insufficient historical data (<30 articles)")
logger.warning(f"Finnhub API failed, falling back to yfinance")
```

## What NOT to Log

- ❌ Raw API responses (too verbose)
- ❌ Large DataFrames (use shape instead: `logger.info(f"Fetched {len(df)} rows")`)
- ❌ Sensitive data (API keys, user data)
