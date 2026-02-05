# NODE 6: MARKET CONTEXT ANALYSIS
## Implementation Structure for Cursor AI

---

## 📍 Overview

**File:** `src/langgraph_nodes/node_06_market_context.py`

**Purpose:** Analyze market-wide conditions and sector performance to provide context for individual stock signals

**Why Important:** A stock might look great technically, but if the entire sector is crashing, you need to know!

---

## 🎯 What Node 6 Does

```
Input from state:
- ticker (e.g., 'NVDA')
- raw_price_data (from Node 1)
- raw_related_companies (from Node 3)
- sentiment_analysis (from Node 5)

What it does:
1. Determine the stock's sector (Technology, Healthcare, etc.)
2. Fetch sector performance (how is tech sector doing today?)
3. Analyze related companies (are competitors up or down?)
4. Determine overall market trend (bull/bear/neutral)
5. Calculate correlation with market/sector
6. Generate context signal

Output to state:
- market_context {
    sector: str,
    sector_performance: float,  # % change
    related_companies_signals: List[Dict],
    market_trend: str,  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    correlation_strength: float,  # 0-1
    context_signal: str,  # 'BUY', 'SELL', 'HOLD'
    confidence: float
  }
```

---

## 🏗️ Node 6 Structure (For Cursor to Build)

```python
# File: src/langgraph_nodes/node_06_market_context.py

"""
Node 6: Market Context Analysis

Provides market-wide context to inform individual stock decisions.
Prevents buying a stock just because it looks good technically,
when the entire sector or market is collapsing.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SECTOR MAPPING (Stock to Sector)
# ============================================================================

SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Energy': 'XLE',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Utilities': 'XLU',
    'Communication Services': 'XLC'
}

MARKET_INDICES = {
    'S&P 500': 'SPY',
    'NASDAQ': 'QQQ',
    'Dow Jones': 'DIA'
}


# ============================================================================
# HELPER FUNCTION 1: Get Stock Sector
# ============================================================================

def get_stock_sector(ticker: str) -> Tuple[str, str]:
    """
    Get the sector and industry for a stock.
    
    Algorithm:
    1. Use yfinance to get stock info
    2. Extract 'sector' and 'industry' fields
    3. If not available, return 'Unknown'
    
    Args:
        ticker: Stock symbol (e.g., 'NVDA')
    
    Returns:
        (sector, industry) tuple
        e.g., ('Technology', 'Semiconductors')
    
    Example:
        sector, industry = get_stock_sector('NVDA')
        # Returns: ('Technology', 'Semiconductors')
    """
    try:
        # TODO: Use yfinance to fetch stock info
        # stock = yf.Ticker(ticker)
        # info = stock.info
        # sector = info.get('sector', 'Unknown')
        # industry = info.get('industry', 'Unknown')
        
        pass
        
    except Exception as e:
        logger.error(f"Failed to get sector for {ticker}: {str(e)}")
        return ('Unknown', 'Unknown')


# ============================================================================
# HELPER FUNCTION 2: Get Sector Performance
# ============================================================================

def get_sector_performance(sector: str, days: int = 1) -> Dict:
    """
    Get the performance of a sector over the last N days.
    
    Uses sector ETFs (e.g., XLK for Technology) as proxy.
    
    Algorithm:
    1. Map sector name to ETF ticker (e.g., Technology → XLK)
    2. Fetch ETF price data for last N days
    3. Calculate percentage change
    4. Determine trend direction
    
    Args:
        sector: Sector name (e.g., 'Technology')
        days: Number of days to analyze (default: 1)
    
    Returns:
        {
            'sector': 'Technology',
            'etf_ticker': 'XLK',
            'performance': 2.3,  # % change
            'trend': 'UP',  # 'UP', 'DOWN', 'FLAT'
            'price_current': 180.5,
            'price_start': 176.4
        }
    
    Example:
        performance = get_sector_performance('Technology', days=1)
        # Returns: {'sector': 'Technology', 'performance': 2.3, ...}
    """
    try:
        # TODO: Get sector ETF ticker
        etf_ticker = SECTOR_ETFS.get(sector)
        
        if not etf_ticker:
            # TODO: Return neutral defaults
            pass
        
        # TODO: Fetch ETF price data using yfinance
        # etf = yf.Ticker(etf_ticker)
        # history = etf.history(period=f'{days+1}d')
        
        # TODO: Calculate performance
        # price_start = history['Close'].iloc[0]
        # price_current = history['Close'].iloc[-1]
        # performance = ((price_current - price_start) / price_start) * 100
        
        # TODO: Determine trend
        # trend = 'UP' if performance > 0.5 else 'DOWN' if performance < -0.5 else 'FLAT'
        
        pass
        
    except Exception as e:
        logger.error(f"Failed to get sector performance: {str(e)}")
        return {
            'sector': sector,
            'etf_ticker': None,
            'performance': 0.0,
            'trend': 'FLAT',
            'error': str(e)
        }


# ============================================================================
# HELPER FUNCTION 3: Get Market Trend
# ============================================================================

def get_market_trend(days: int = 5) -> Dict:
    """
    Determine overall market trend.
    
    Uses S&P 500 (SPY) as market proxy.
    
    Algorithm:
    1. Fetch SPY price data for last N days
    2. Calculate performance over period
    3. Calculate volatility (standard deviation)
    4. Determine trend:
       - BULLISH: performance > 2% and low volatility
       - BEARISH: performance < -2% or high volatility
       - NEUTRAL: otherwise
    
    Args:
        days: Number of days to analyze (default: 5)
    
    Returns:
        {
            'market_trend': 'BULLISH',  # 'BULLISH', 'BEARISH', 'NEUTRAL'
            'spy_performance': 3.2,  # % change
            'volatility': 1.1,  # Standard deviation
            'confidence': 0.85  # How confident in this trend
        }
    
    Example:
        trend = get_market_trend(days=5)
        # Returns: {'market_trend': 'BULLISH', 'spy_performance': 3.2, ...}
    """
    try:
        # TODO: Fetch SPY data
        # spy = yf.Ticker('SPY')
        # history = spy.history(period=f'{days+1}d')
        
        # TODO: Calculate performance
        # price_start = history['Close'].iloc[0]
        # price_current = history['Close'].iloc[-1]
        # performance = ((price_current - price_start) / price_start) * 100
        
        # TODO: Calculate volatility
        # returns = history['Close'].pct_change()
        # volatility = returns.std() * 100
        
        # TODO: Determine trend
        # if performance > 2.0 and volatility < 2.0:
        #     trend = 'BULLISH'
        #     confidence = 0.8
        # elif performance < -2.0 or volatility > 3.0:
        #     trend = 'BEARISH'
        #     confidence = 0.8
        # else:
        #     trend = 'NEUTRAL'
        #     confidence = 0.6
        
        pass
        
    except Exception as e:
        logger.error(f"Failed to get market trend: {str(e)}")
        return {
            'market_trend': 'NEUTRAL',
            'spy_performance': 0.0,
            'volatility': 0.0,
            'confidence': 0.5
        }


# ============================================================================
# HELPER FUNCTION 4: Analyze Related Companies
# ============================================================================

def analyze_related_companies(related_tickers: List[str]) -> Dict:
    """
    Analyze performance of related companies.
    
    Algorithm:
    1. For each related company, fetch 1-day price change
    2. Calculate average performance
    3. Count how many are up vs down
    4. Determine if related companies are bullish or bearish
    
    Args:
        related_tickers: List of ticker symbols (e.g., ['AMD', 'INTC', 'TSM'])
    
    Returns:
        {
            'related_companies': [
                {'ticker': 'AMD', 'performance': 1.5, 'trend': 'UP'},
                {'ticker': 'INTC', 'performance': -0.8, 'trend': 'DOWN'},
                {'ticker': 'TSM', 'performance': 2.1, 'trend': 'UP'}
            ],
            'avg_performance': 0.93,
            'up_count': 2,
            'down_count': 1,
            'overall_signal': 'BULLISH'  # 'BULLISH', 'BEARISH', 'NEUTRAL'
        }
    
    Example:
        analysis = analyze_related_companies(['AMD', 'INTC'])
        # Returns: {'avg_performance': 0.35, 'overall_signal': 'NEUTRAL', ...}
    """
    if not related_tickers:
        return {
            'related_companies': [],
            'avg_performance': 0.0,
            'up_count': 0,
            'down_count': 0,
            'overall_signal': 'NEUTRAL'
        }
    
    try:
        # TODO: Fetch performance for each related company
        # related_data = []
        # for ticker in related_tickers:
        #     stock = yf.Ticker(ticker)
        #     history = stock.history(period='2d')
        #     if len(history) >= 2:
        #         price_prev = history['Close'].iloc[-2]
        #         price_current = history['Close'].iloc[-1]
        #         performance = ((price_current - price_prev) / price_prev) * 100
        #         trend = 'UP' if performance > 0.5 else 'DOWN' if performance < -0.5 else 'FLAT'
        #         related_data.append({
        #             'ticker': ticker,
        #             'performance': performance,
        #             'trend': trend
        #         })
        
        # TODO: Calculate statistics
        # performances = [r['performance'] for r in related_data]
        # avg_performance = sum(performances) / len(performances)
        # up_count = sum(1 for r in related_data if r['trend'] == 'UP')
        # down_count = sum(1 for r in related_data if r['trend'] == 'DOWN')
        
        # TODO: Determine overall signal
        # if up_count > down_count and avg_performance > 1.0:
        #     overall_signal = 'BULLISH'
        # elif down_count > up_count and avg_performance < -1.0:
        #     overall_signal = 'BEARISH'
        # else:
        #     overall_signal = 'NEUTRAL'
        
        pass
        
    except Exception as e:
        logger.error(f"Failed to analyze related companies: {str(e)}")
        return {
            'related_companies': [],
            'avg_performance': 0.0,
            'up_count': 0,
            'down_count': 0,
            'overall_signal': 'NEUTRAL',
            'error': str(e)
        }


# ============================================================================
# HELPER FUNCTION 5: Calculate Correlation
# ============================================================================

def calculate_correlation(ticker: str, price_data: pd.DataFrame) -> Dict:
    """
    Calculate correlation between stock and market/sector.
    
    Algorithm:
    1. Get 30 days of price data for stock
    2. Get 30 days of SPY (market) data
    3. Calculate Pearson correlation
    4. High correlation (>0.7) means stock moves with market
    5. Low correlation (<0.3) means stock is independent
    
    Args:
        ticker: Stock symbol
        price_data: DataFrame with stock price history
    
    Returns:
        {
            'market_correlation': 0.85,  # Correlation with SPY
            'correlation_strength': 'HIGH',  # 'HIGH', 'MEDIUM', 'LOW'
            'beta': 1.2  # Stock's beta (volatility vs market)
        }
    
    Example:
        corr = calculate_correlation('NVDA', price_data)
        # Returns: {'market_correlation': 0.85, 'correlation_strength': 'HIGH', ...}
    """
    try:
        # TODO: Get 30 days of stock returns
        # stock_returns = price_data['close'].pct_change().dropna()
        
        # TODO: Get 30 days of SPY returns
        # spy = yf.Ticker('SPY')
        # spy_history = spy.history(period='30d')
        # spy_returns = spy_history['Close'].pct_change().dropna()
        
        # TODO: Align dates and calculate correlation
        # Ensure both series have same dates
        # correlation = stock_returns.corr(spy_returns)
        
        # TODO: Calculate beta
        # beta = stock_returns.std() / spy_returns.std() * correlation
        
        # TODO: Determine strength
        # if abs(correlation) > 0.7:
        #     strength = 'HIGH'
        # elif abs(correlation) > 0.4:
        #     strength = 'MEDIUM'
        # else:
        #     strength = 'LOW'
        
        pass
        
    except Exception as e:
        logger.error(f"Failed to calculate correlation: {str(e)}")
        return {
            'market_correlation': 0.5,
            'correlation_strength': 'MEDIUM',
            'beta': 1.0
        }


# ============================================================================
# HELPER FUNCTION 6: Generate Context Signal
# ============================================================================

def generate_context_signal(
    sector_performance: Dict,
    market_trend: Dict,
    related_analysis: Dict,
    correlation: Dict
) -> Tuple[str, float]:
    """
    Generate final context signal based on all market factors.
    
    Algorithm:
    1. Start with neutral signal
    2. Adjust based on sector performance
    3. Adjust based on market trend
    4. Adjust based on related companies
    5. Weight by correlation (high correlation = market matters more)
    
    Scoring:
    - Sector up +20, down -20
    - Market bullish +30, bearish -30
    - Related companies bullish +20, bearish -20
    - High correlation: multiply by 1.5
    
    Args:
        sector_performance: From get_sector_performance()
        market_trend: From get_market_trend()
        related_analysis: From analyze_related_companies()
        correlation: From calculate_correlation()
    
    Returns:
        (signal, confidence) tuple
        signal: 'BUY', 'SELL', 'HOLD'
        confidence: 0-100
    
    Example:
        signal, confidence = generate_context_signal(sector, market, related, corr)
        # Returns: ('BUY', 75)
    """
    # TODO: Initialize score
    score = 0
    
    # TODO: Add sector score
    # sector_perf = sector_performance.get('performance', 0)
    # if sector_perf > 1.0:
    #     score += 20
    # elif sector_perf < -1.0:
    #     score -= 20
    
    # TODO: Add market trend score
    # market = market_trend.get('market_trend', 'NEUTRAL')
    # if market == 'BULLISH':
    #     score += 30
    # elif market == 'BEARISH':
    #     score -= 30
    
    # TODO: Add related companies score
    # related_signal = related_analysis.get('overall_signal', 'NEUTRAL')
    # if related_signal == 'BULLISH':
    #     score += 20
    # elif related_signal == 'BEARISH':
    #     score -= 20
    
    # TODO: Apply correlation multiplier
    # corr_strength = correlation.get('market_correlation', 0.5)
    # if corr_strength > 0.7:
    #     score = score * 1.5  # High correlation: market matters more
    
    # TODO: Convert score to signal
    # if score > 30:
    #     signal = 'BUY'
    #     confidence = min(100, 50 + score)
    # elif score < -30:
    #     signal = 'SELL'
    #     confidence = min(100, 50 + abs(score))
    # else:
    #     signal = 'HOLD'
    #     confidence = 50
    
    pass


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def market_context_node(state: 'StockAnalysisState') -> 'StockAnalysisState':
    """
    Node 6: Market Context Analysis
    
    Execution flow:
    1. Get stock's sector
    2. Get sector performance
    3. Get overall market trend
    4. Analyze related companies
    5. Calculate correlation with market
    6. Generate context signal
    7. Update state
    
    Runs in PARALLEL with Nodes 4, 5, 7
    Runs AFTER: Nodes 1, 2, 3 (need price + news data)
    Runs BEFORE: Node 8 (verification)
    
    Args:
        state: LangGraph state
        
    Returns:
        Updated state with market_context results
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    try:
        logger.info(f"Node 6: Analyzing market context for {ticker}")
        
        # STEP 1: Get stock's sector
        sector, industry = get_stock_sector(ticker)
        logger.info(f"Sector: {sector}, Industry: {industry}")
        
        # STEP 2: Get sector performance
        sector_performance = get_sector_performance(sector, days=1)
        
        # STEP 3: Get overall market trend
        market_trend = get_market_trend(days=5)
        
        # STEP 4: Analyze related companies
        related_tickers = state.get('raw_related_companies', [])
        related_analysis = analyze_related_companies(related_tickers)
        
        # STEP 5: Calculate correlation with market
        price_data = state.get('raw_price_data')
        if price_data is not None and not price_data.empty:
            correlation = calculate_correlation(ticker, price_data)
        else:
            correlation = {
                'market_correlation': 0.5,
                'correlation_strength': 'MEDIUM',
                'beta': 1.0
            }
        
        # STEP 6: Generate context signal
        context_signal, confidence = generate_context_signal(
            sector_performance,
            market_trend,
            related_analysis,
            correlation
        )
        
        # STEP 7: Build results
        results = {
            'sector': sector,
            'industry': industry,
            'sector_performance': sector_performance.get('performance', 0.0),
            'sector_trend': sector_performance.get('trend', 'FLAT'),
            'market_trend': market_trend.get('market_trend', 'NEUTRAL'),
            'market_performance': market_trend.get('spy_performance', 0.0),
            'related_companies_signals': related_analysis.get('related_companies', []),
            'related_companies_avg': related_analysis.get('avg_performance', 0.0),
            'related_companies_signal': related_analysis.get('overall_signal', 'NEUTRAL'),
            'market_correlation': correlation.get('market_correlation', 0.5),
            'correlation_strength': correlation.get('correlation_strength', 'MEDIUM'),
            'beta': correlation.get('beta', 1.0),
            'context_signal': context_signal,
            'confidence': confidence
        }
        
        logger.info(f"Context signal: {context_signal} (confidence: {confidence:.1f}%)")
        logger.info(f"Sector: {sector_performance.get('trend', 'FLAT')}, " +
                   f"Market: {market_trend.get('market_trend', 'NEUTRAL')}, " +
                   f"Related: {related_analysis.get('overall_signal', 'NEUTRAL')}")
        
        # Update state
        state['market_context'] = results
        state['node_execution_times']['node_6'] = (datetime.now() - start_time).total_seconds()
        
        return state
        
    except Exception as e:
        logger.error(f"Node 6 failed: {str(e)}")
        state['errors'].append(f"Market context analysis failed: {str(e)}")
        
        # Failsafe: neutral context
        state['market_context'] = {
            'sector': 'Unknown',
            'industry': 'Unknown',
            'sector_performance': 0.0,
            'market_trend': 'NEUTRAL',
            'context_signal': 'HOLD',
            'confidence': 50.0
        }
        state['node_execution_times']['node_6'] = (datetime.now() - start_time).total_seconds()
        
        return state
```

---

## 📊 Example Output

```python
# For NVIDIA (NVDA)

market_context = {
    'sector': 'Technology',
    'industry': 'Semiconductors',
    'sector_performance': 2.3,  # Tech sector up 2.3% today
    'sector_trend': 'UP',
    'market_trend': 'BULLISH',  # Overall market bullish
    'market_performance': 1.5,  # SPY up 1.5%
    'related_companies_signals': [
        {'ticker': 'AMD', 'performance': 2.1, 'trend': 'UP'},
        {'ticker': 'INTC', 'performance': -0.5, 'trend': 'DOWN'},
        {'ticker': 'TSM', 'performance': 1.8, 'trend': 'UP'}
    ],
    'related_companies_avg': 1.13,  # Average: +1.13%
    'related_companies_signal': 'BULLISH',
    'market_correlation': 0.85,  # Highly correlated with market
    'correlation_strength': 'HIGH',
    'beta': 1.3,  # More volatile than market
    'context_signal': 'BUY',  # Everything looks good!
    'confidence': 82.0
}

# Interpretation:
# - Tech sector is UP ✅
# - Market is BULLISH ✅
# - Related companies mostly UP ✅
# - High correlation means NVDA moves with market ✅
# - CONTEXT SIGNAL: BUY (safe to buy) ✅
```

---

## 🔄 Integration with Other Nodes

**Node 1 (Price Data):**
- Provides price_data for correlation calculation

**Node 3 (Related Companies):**
- Provides list of related tickers to analyze

**Node 11 (Adaptive Weights):**
- Uses context_signal confidence in final weighting

**Node 12 (Signal Generation):**
- Context signal is one of 4 signals combined for final decision

---

## 🧪 Testing Strategy

```python
# File: tests/test_node_06.py

def test_sector_detection():
    """Test that NVDA is correctly identified as Technology"""
    sector, industry = get_stock_sector('NVDA')
    assert sector == 'Technology'
    assert 'Semiconductor' in industry

def test_sector_performance():
    """Test sector ETF performance calculation"""
    perf = get_sector_performance('Technology', days=1)
    assert 'performance' in perf
    assert 'trend' in perf
    assert perf['etf_ticker'] == 'XLK'

def test_market_trend():
    """Test market trend determination"""
    trend = get_market_trend(days=5)
    assert trend['market_trend'] in ['BULLISH', 'BEARISH', 'NEUTRAL']
    assert 0 <= trend['confidence'] <= 1

def test_related_companies():
    """Test related companies analysis"""
    analysis = analyze_related_companies(['AMD', 'INTC'])
    assert 'avg_performance' in analysis
    assert 'overall_signal' in analysis

def test_correlation():
    """Test correlation calculation"""
    # TODO: Create mock price data
    # corr = calculate_correlation('NVDA', mock_data)
    # assert -1 <= corr['market_correlation'] <= 1
    pass
```

---

## 📈 Expected Behavior

### Scenario 1: Market Crash
```
Sector: DOWN -5%
Market: BEARISH
Related: All DOWN
Context Signal: SELL ⚠️

Even if technical looks good, context says AVOID!
```

### Scenario 2: Sector Boom
```
Sector: UP +3%
Market: BULLISH
Related: Mostly UP
Context Signal: BUY ✅

Good environment for buying!
```

### Scenario 3: Mixed Signals
```
Sector: FLAT
Market: NEUTRAL
Related: Mixed
Context Signal: HOLD 

Wait for clearer conditions.
```

---

## 🎯 Key Formulas

**Sector Performance:**
```python
performance = ((price_current - price_start) / price_start) * 100
```

**Correlation:**
```python
correlation = stock_returns.corr(spy_returns)
```

**Beta:**
```python
beta = (stock_std / market_std) * correlation
```

**Context Score:**
```python
score = sector_score + market_score + related_score
score = score * (1 + correlation/2)  # Amplify if high correlation
```

---

## 💡 Cursor Instructions Summary

**For Cursor to implement Node 6:**

1. Create file: `src/langgraph_nodes/node_06_market_context.py`

2. Implement functions in order:
   - `get_stock_sector()` - Use yfinance to get sector
   - `get_sector_performance()` - Fetch sector ETF data
   - `get_market_trend()` - Analyze SPY for market trend
   - `analyze_related_companies()` - Check related company performance
   - `calculate_correlation()` - Compute stock-market correlation
   - `generate_context_signal()` - Combine all factors
   - `market_context_node()` - Main function

3. Key libraries needed:
   - `yfinance` - For fetching market/sector data
   - `pandas` - For correlation calculations
   - `datetime` - For date handling

4. Error handling:
   - If API fails, return neutral defaults
   - If correlation fails, assume medium correlation (0.5)
   - Always return valid signal even if some data missing

**Difficulty: MEDIUM** (lots of API calls but straightforward logic)

---

**This provides essential market awareness!** 🌍
