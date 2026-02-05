---
description: "Never break the graph - comprehensive error handling for all nodes"
alwaysApply: true
---

# Error Handling: Never Break the Graph

**Critical:** Nodes must NEVER raise exceptions that break graph execution.

## Required Pattern

```python
def node_name(state: StockAnalysisState) -> StockAnalysisState:
    try:
        # Main logic
        result = do_something()
        state['result'] = result
        return state
        
    except Exception as e:
        # Log the error
        logger.error(f"Node X failed: {str(e)}")
        
        # Add to errors list (don't break flow)
        state['errors'].append(f"Node X failed: {str(e)}")
        
        # Provide safe fallback
        state['result'] = get_safe_default()
        
        # ALWAYS return state
        return state
```

## Rules

- Wrap ALL node logic in try-except
- Log errors with context: `logger.error(f"Node X failed for {ticker}: {str(e)}")`
- Append errors to `state['errors']` list
- Provide safe default values (None, empty list, etc.)
- **ALWAYS return state (never raise)**

## Safe Defaults by Type

- Price data: `None`
- News lists: `[]` (empty list)
- Analysis results: `None` or neutral values
- Signals: `'HOLD'` with low confidence
