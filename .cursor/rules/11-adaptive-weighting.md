---
description: "Adaptive weighting formula - weights based on historical accuracy from backtesting"
alwaysApply: false
globs: ["**/node_10*.py", "**/node_11*.py", "**/backtest*.py", "**/weight*.py"]
---

# Adaptive Weighting System

**Principle:** Signal weights are proportional to historical accuracy.

## Formula

```
weight_i = accuracy_i / sum(all_accuracies)
```

## Example Calculation

```
Backtest Results (Node 10):
- Technical: 55% accurate
- Stock News: 65% accurate (improved by Node 8 learning!)
- Market News: 58% accurate
- Related: 60% accurate

Total = 55 + 65 + 58 + 60 = 238

Weights (Node 11):
- Technical: 55/238 = 0.231 = 23.1%
- Stock News: 65/238 = 0.273 = 27.3% ← Highest (Node 8 improved it!)
- Market News: 58/238 = 0.244 = 24.4%
- Related: 60/238 = 0.252 = 25.2%

Total: 100% (weights sum to 1.0)
```

## Node 10: Backtesting

Test each of 4 signal streams independently over last 180 days:

```python
For each day in last 180 days:
    1. Get signal from that day (BUY/SELL/HOLD)
    2. Get actual price movement 7 days later
    3. Check if signal was correct:
       - BUY + price up → Correct
       - SELL + price down → Correct
       - HOLD + price flat → Correct
    4. Track accuracy
```

Output: Accuracy rates for each stream.

## Node 11: Adaptive Weights

Use Node 10's results to calculate optimal weights:

```python
def calculate_weights(backtest_results: Dict) -> Dict[str, float]:
    total = sum(backtest_results.values())
    
    return {
        'technical_weight': backtest_results['technical_accuracy'] / total,
        'stock_news_weight': backtest_results['stock_news_accuracy'] / total,
        'market_news_weight': backtest_results['market_news_accuracy'] / total,
        'related_companies_weight': backtest_results['related_companies_accuracy'] / total
    }
    
    # Validate: weights should sum to 1.0
    assert abs(sum(weights.values()) - 1.0) < 0.001
```

## Why Stock News Gets Highest Weight

**Because Node 8 improved its accuracy!**

Without learning: Stock news ~62% accurate → Weight: ~24%
With Node 8 learning: Stock news ~73% accurate → Weight: ~27-28%

The learning system identifies reliable sources (Bloomberg, Reuters) and boosts their confidence, improving overall sentiment accuracy, which increases the weight.

## Node 12: Using Weights

```python
# Get signals
technical_signal = state['technical_analysis']['technical_signal']
sentiment_signal = state['sentiment_analysis']['sentiment_signal']
context_signal = state['market_context']['context_signal']

# Get weights
weights = state['adaptive_weights']

# Calculate weighted score
buy_score = (
    (100 if technical_signal == 'BUY' else 0) * weights['technical_weight'] +
    (100 if sentiment_signal == 'BUY' else 0) * weights['stock_news_weight'] +
    # ... etc
)

final_signal = 'BUY' if buy_score > 60 else 'SELL' if buy_score < 40 else 'HOLD'
```

## Key Points

- Weights are stock-specific (different for AAPL vs NVDA)
- Weights update as accuracy changes
- Higher accuracy = higher influence on final signal
- Transparent to users (they see why weights are distributed)
