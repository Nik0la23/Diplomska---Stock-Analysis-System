---
description: "LangGraph state-first development - all nodes communicate via StockAnalysisState TypedDict"
alwaysApply: true
---

# State-First Development

**Core Principle:** All inter-node communication happens ONLY through the `StockAnalysisState` TypedDict.

## Node Function Signature

Every node MUST follow this exact pattern:

```python
def node_name(state: StockAnalysisState) -> StockAnalysisState:
    """Node description"""
    start_time = datetime.now()
    ticker = state['ticker']
    
    try:
        # Node logic here
        state['result_field'] = result
        state['node_execution_times']['node_X'] = (datetime.now() - start_time).total_seconds()
        return state
    except Exception as e:
        logger.error(f"Node X failed: {str(e)}")
        state['errors'].append(f"Node X failed: {str(e)}")
        state['node_execution_times']['node_X'] = (datetime.now() - start_time).total_seconds()
        return state
```

## Critical Rules

- **Every node receives `state` and returns updated `state`**
- **NEVER pass data between nodes except through state**
- **Update `StockAnalysisState` TypedDict BEFORE implementing nodes**
- **State is the single source of truth**

## Reference

See the complete state definition in `@src/graph/state.py` (TypedDict with all fields).
