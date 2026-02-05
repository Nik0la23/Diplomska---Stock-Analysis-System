---
description: "Documentation standards - comprehensive docstrings for thesis quality"
alwaysApply: true
---

# Documentation Standards

**Requirement:** Every function must have comprehensive docstring for thesis quality code.

## Complete Docstring Template

```python
def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate technical indicators from price data.
    
    Calculates RSI, MACD, Bollinger Bands, SMAs, EMAs, and volume analysis
    using the pandas-ta library.
    
    Args:
        df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume'].
            Must have at least 50 rows for meaningful calculations.
    
    Returns:
        Dictionary with:
        {
            'rsi': float (0-100),
            'macd': {'value': float, 'signal': float, 'histogram': float},
            'bollinger_bands': {'upper': float, 'middle': float, 'lower': float},
            'sma_20': float,
            'sma_50': float,
            'ema_12': float,
            'ema_26': float,
            'technical_signal': str ('BUY' | 'SELL' | 'HOLD'),
            'confidence': float (0-100)
        }
    
    Raises:
        ValueError: If df has less than 50 rows
    
    Example:
        >>> df = pd.DataFrame({'close': [100, 101, 102], ...})
        >>> indicators = calculate_technical_indicators(df)
        >>> print(indicators['rsi'])
        65.4
    """
    pass
```

## Required Sections

1. **Brief description** (one line)
2. **Detailed description** (what it does, how it works)
3. **Args:** All parameters with types and descriptions
4. **Returns:** Complete structure with types
5. **Raises:** Exceptions that can be raised (if any)
6. **Example:** Usage example with expected output

## Node Docstrings

Include execution order information:

```python
def sentiment_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
    """
    Node 5: Analyze sentiment using FinBERT on CLEANED news.
    
    CRITICAL: Uses cleaned_*_news from Node 9A, NOT raw_*_news.
    This ensures sentiment analysis works on verified data only.
    
    Execution context:
    - Runs in PARALLEL with Nodes 4, 6, 7
    - Runs AFTER: Node 9A (needs filtered news)
    - Runs BEFORE: Node 8 (verification)
    
    Args:
        state: LangGraph state
        
    Returns:
        Updated state with sentiment_analysis results
    """
    pass
```

## What to Document

- ✅ Purpose and high-level algorithm
- ✅ All parameters and return values
- ✅ Data structure details
- ✅ Dependencies and execution order (for nodes)
- ✅ Usage examples
- ❌ Implementation details (let code speak)
- ❌ Obvious things (don't document `return None`)
