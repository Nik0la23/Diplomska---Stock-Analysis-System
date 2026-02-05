---
description: "Standard file structure for all LangGraph nodes"
alwaysApply: true
---

# Node File Structure

All nodes must follow this consistent structure.

## Template

```python
"""
Node X: [Name]
[Brief description of what this node does]

Runs AFTER: Node Y, Node Z
Runs BEFORE: Node A, Node B
Can run in PARALLEL with: Node C, Node D
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTION 1
# ============================================================================

def helper_function_1(param: str) -> Dict:
    """
    Description of helper function.
    
    Args:
        param: Description
    
    Returns:
        Dictionary with results
    
    Example:
        >>> result = helper_function_1('AAPL')
        >>> print(result['value'])
    """
    pass

# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def node_name(state: 'StockAnalysisState') -> 'StockAnalysisState':
    """
    Node X: [Name]
    
    Execution flow:
    1. Extract data from state
    2. Process data
    3. Update state with results
    
    Args:
        state: LangGraph state
        
    Returns:
        Updated state with [field_name] populated
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    try:
        logger.info(f"Node X: Starting for {ticker}")
        
        # Extract dependencies
        dependency = state.get('required_data')
        if dependency is None:
            raise ValueError("Required data not available")
        
        # Process
        result = process_data(dependency)
        
        # Update state
        state['result_field'] = result
        state['node_execution_times']['node_x'] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Node X: Completed in {state['node_execution_times']['node_x']:.2f}s")
        return state
        
    except Exception as e:
        logger.error(f"Node X failed: {str(e)}")
        state['errors'].append(f"Node X failed: {str(e)}")
        state['result_field'] = None
        state['node_execution_times']['node_x'] = (datetime.now() - start_time).total_seconds()
        return state
```

## Required Sections

1. **File docstring** - Describe node purpose and dependencies
2. **Imports** - All necessary libraries
3. **Logger setup** - `logger = logging.getLogger(__name__)`
4. **Helper functions** - With clear section separators
5. **Main node function** - Named `[descriptor]_node(state)`

## File Naming

- File: `node_XX_description.py`
- Function: `description_node(state)`

Example:
- File: `node_04_technical_analysis.py`
- Function: `technical_analysis_node(state)`
