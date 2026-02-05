---
description: "Type hints required on all functions for thesis code quality"
alwaysApply: true
---

# Type Hints Everywhere

**Requirement:** All functions must have complete type hints for thesis code quality.

## Examples

```python
from typing import TypedDict, Optional, List, Dict, Tuple
import pandas as pd

# Function signatures
def calculate_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate technical indicators"""
    pass

def fetch_news(ticker: str, days: int = 7) -> List[Dict[str, str]]:
    """Fetch news articles"""
    pass

def analyze_sentiment(articles: List[Dict]) -> Tuple[float, str]:
    """Analyze sentiment, return (score, signal)"""
    pass

# Optional types for nullable returns
def get_cached_data(ticker: str) -> Optional[pd.DataFrame]:
    """Return DataFrame or None if not cached"""
    pass
```

## Rules

- All function parameters must have type hints
- All return values must have type hints
- Use `Optional[T]` for nullable values
- Use `List[T]`, `Dict[K, V]` for collections
- Use `TypedDict` for structured dictionaries (like state)

## Common Types

```python
from typing import Optional, List, Dict, Tuple, Union, Any

# For price data
price_df: pd.DataFrame

# For news articles
articles: List[Dict[str, Any]]

# For signals
signal: Tuple[str, float]  # ('BUY', 0.85)

# For optional values
cached_data: Optional[pd.DataFrame]

# For union types
result: Union[str, None]
```
