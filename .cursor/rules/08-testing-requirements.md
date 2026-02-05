---
description: "Test requirements for all nodes - mock external APIs"
alwaysApply: true
---

# Testing Requirements

**Requirement:** Write tests for EVERY node. Mock all external APIs.

## Test Structure

```python
import pytest
from unittest.mock import Mock, patch
import pandas as pd

@pytest.fixture
def mock_state():
    """Fixture for basic state"""
    return {
        'ticker': 'AAPL',
        'analysis_date': '2024-01-15',
        'errors': [],
        'warnings': [],
        'node_execution_times': {}
    }

@pytest.fixture
def mock_price_data():
    """Fixture for price data"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'close': [150.0 + i * 0.5 for i in range(100)],
        'volume': [1000000] * 100
    })

def test_node_success(mock_state, mock_price_data):
    """Test node with valid data"""
    mock_state['raw_price_data'] = mock_price_data
    
    result = node_function(mock_state)
    
    assert 'result_field' in result
    assert result['result_field']['signal'] in ['BUY', 'SELL', 'HOLD']
    assert len(result['errors']) == 0

def test_node_missing_data(mock_state):
    """Test node handles missing data gracefully"""
    mock_state['raw_price_data'] = None
    
    result = node_function(mock_state)
    
    assert result['result_field'] is None
    assert len(result['errors']) == 1
    assert 'failed' in result['errors'][0].lower()

@patch('finnhub.Client')
def test_node_with_mock_api(mock_finnhub, mock_state):
    """Test with mocked API"""
    mock_client = Mock()
    mock_client.stock_candles.return_value = {'s': 'ok', 'c': [150, 151]}
    mock_finnhub.return_value = mock_client
    
    result = node_function(mock_state)
    
    assert result['result_field'] is not None
    assert len(result['errors']) == 0
```

## Required Tests Per Node

1. **Success case** - Valid inputs, expected outputs
2. **Missing data** - Graceful handling of None/empty inputs
3. **API failure** - Mocked API errors
4. **Edge cases** - Boundary conditions specific to node

## Rules

- Mock ALL external API calls (Finnhub, NewsAPI, etc.)
- Use pytest fixtures for common test data
- Test both success AND failure cases
- Aim for 70%+ code coverage
- Never make real API calls in tests

## Reference

See `@tests/` directory for examples.
