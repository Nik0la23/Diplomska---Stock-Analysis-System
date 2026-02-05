# NODE 7: MONTE CARLO PRICE FORECASTING
## Implementation Structure for Cursor AI

---

## 📍 Overview

**File:** `src/langgraph_nodes/node_07_monte_carlo.py`

**Purpose:** Forecast future stock prices using Monte Carlo simulation with Geometric Brownian Motion (GBM)

**Why Important:** Provides probabilistic price forecasts with confidence intervals instead of just "BUY/SELL"

---

## 🎯 What Node 7 Does

```
Input from state:
- ticker (e.g., 'NVDA')
- raw_price_data (from Node 1)

What it does:
1. Calculate historical volatility (standard deviation of returns)
2. Calculate historical drift (average daily return)
3. Run 1000 Monte Carlo simulations using GBM
4. Calculate confidence intervals (68% and 95%)
5. Calculate probability of price going up vs down
6. Generate forecast visualization data

Output to state:
- monte_carlo_forecast {
    simulations: np.ndarray,  # Shape: (1000, forecast_days)
    mean_forecast: float,  # Expected price
    median_forecast: float,
    confidence_68: {'lower': float, 'upper': float},
    confidence_95: {'lower': float, 'upper': float},
    probability_up: float,  # 0-1
    probability_down: float,  # 0-1
    expected_return: float,  # % return
    forecast_days: int
  }
```

---

## 📚 Theory: Geometric Brownian Motion (GBM)

**The Formula:**
```
S(t+1) = S(t) * exp((μ - 0.5*σ²)*Δt + σ*√Δt*Z)

Where:
- S(t) = Stock price at time t
- μ (mu) = Drift (average return)
- σ (sigma) = Volatility (standard deviation)
- Δt = Time step (1 day = 1/252 for trading days)
- Z = Random normal variable (mean=0, std=1)
```

**In Plain English:**
- Tomorrow's price = Today's price × (growth factor + random shock)
- Growth factor = historical average return
- Random shock = random number × historical volatility

---

## 🏗️ Node 7 Structure (For Cursor to Build)

```python
# File: src/langgraph_nodes/node_07_monte_carlo.py

"""
Node 7: Monte Carlo Price Forecasting

Uses Geometric Brownian Motion (GBM) to simulate 1000 possible
future price paths and calculate probability distributions.

Standard technique in quantitative finance.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTION 1: Calculate Historical Statistics
# ============================================================================

def calculate_historical_statistics(price_data: pd.DataFrame, days: int = 30) -> Tuple[float, float, float]:
    """
    Calculate historical drift and volatility from price data.
    
    Algorithm:
    1. Calculate daily returns: (price[t] - price[t-1]) / price[t-1]
    2. Calculate drift (μ): mean of daily returns
    3. Calculate volatility (σ): standard deviation of daily returns
    4. Get current price
    
    Args:
        price_data: DataFrame with 'close' prices
        days: Number of days to use for calculation (default: 30)
    
    Returns:
        (drift, volatility, current_price) tuple
        - drift: Average daily return (e.g., 0.002 = 0.2% per day)
        - volatility: Daily volatility (e.g., 0.025 = 2.5% per day)
        - current_price: Most recent close price
    
    Example:
        drift, vol, price = calculate_historical_statistics(price_data, days=30)
        # Returns: (0.0015, 0.023, 525.50)
    """
    try:
        # TODO: Get last N days of data
        # recent_data = price_data.tail(days)
        # prices = recent_data['close'].values
        
        # TODO: Calculate daily returns
        # returns = np.diff(prices) / prices[:-1]
        # Or using pandas:
        # returns = recent_data['close'].pct_change().dropna()
        
        # TODO: Calculate drift (mean return)
        # drift = returns.mean()
        
        # TODO: Calculate volatility (std of returns)
        # volatility = returns.std()
        
        # TODO: Get current price
        # current_price = prices[-1]
        
        # TODO: Annualize if needed
        # drift_annual = drift * 252  # 252 trading days per year
        # volatility_annual = volatility * np.sqrt(252)
        
        pass
        
    except Exception as e:
        logger.error(f"Failed to calculate statistics: {str(e)}")
        # Return neutral defaults
        return (0.0, 0.02, 100.0)


# ============================================================================
# HELPER FUNCTION 2: Run Single GBM Simulation
# ============================================================================

def simulate_gbm_path(
    current_price: float,
    drift: float,
    volatility: float,
    days: int,
    dt: float = 1/252
) -> np.ndarray:
    """
    Simulate a single price path using Geometric Brownian Motion.
    
    GBM Formula:
    S(t+1) = S(t) * exp((μ - 0.5*σ²)*Δt + σ*√Δt*Z)
    
    Algorithm:
    1. Initialize array with current price
    2. For each day:
       a. Generate random normal variable Z ~ N(0,1)
       b. Calculate drift component: (μ - 0.5*σ²)*Δt
       c. Calculate diffusion component: σ*√Δt*Z
       d. Calculate next price: S(t) * exp(drift + diffusion)
    3. Return array of prices
    
    Args:
        current_price: Starting price (e.g., 525.50)
        drift: Daily drift (e.g., 0.002)
        volatility: Daily volatility (e.g., 0.025)
        days: Number of days to forecast (e.g., 7)
        dt: Time step (default: 1/252 for daily)
    
    Returns:
        Array of simulated prices, shape: (days+1,)
        [525.50, 528.30, 532.10, 529.80, ...]
    
    Example:
        path = simulate_gbm_path(525.50, 0.002, 0.025, 7)
        # Returns: array([525.5, 528.3, 532.1, 529.8, 533.2, 535.1, 537.8, 540.2])
    """
    # TODO: Initialize price array
    # prices = np.zeros(days + 1)
    # prices[0] = current_price
    
    # TODO: Generate all random shocks at once (more efficient)
    # Z = np.random.standard_normal(days)
    
    # TODO: Calculate drift and diffusion components
    # drift_component = (drift - 0.5 * volatility**2) * dt
    # diffusion_component = volatility * np.sqrt(dt)
    
    # TODO: Simulate price path
    # for i in range(days):
    #     shock = drift_component + diffusion_component * Z[i]
    #     prices[i+1] = prices[i] * np.exp(shock)
    
    # Alternative vectorized approach (faster):
    # shocks = drift_component + diffusion_component * Z
    # prices[1:] = current_price * np.exp(np.cumsum(shocks))
    
    pass


# ============================================================================
# HELPER FUNCTION 3: Run Monte Carlo Simulations
# ============================================================================

def run_monte_carlo_simulations(
    current_price: float,
    drift: float,
    volatility: float,
    forecast_days: int = 7,
    num_simulations: int = 1000
) -> np.ndarray:
    """
    Run multiple Monte Carlo simulations.
    
    Algorithm:
    1. Create array to store all simulations
    2. For each simulation (1 to 1000):
       a. Run simulate_gbm_path()
       b. Store result
    3. Return 2D array of all paths
    
    Args:
        current_price: Starting price
        drift: Daily drift
        volatility: Daily volatility
        forecast_days: How many days to forecast (default: 7)
        num_simulations: How many paths to simulate (default: 1000)
    
    Returns:
        2D array of simulated prices, shape: (num_simulations, forecast_days+1)
        [
            [525.5, 528.3, 532.1, ...],  # Simulation 1
            [525.5, 523.2, 520.8, ...],  # Simulation 2
            [525.5, 527.1, 530.5, ...],  # Simulation 3
            ...
        ]
    
    Example:
        sims = run_monte_carlo_simulations(525.50, 0.002, 0.025, 7, 1000)
        # Returns: (1000, 8) array
    """
    # TODO: Initialize simulations array
    # simulations = np.zeros((num_simulations, forecast_days + 1))
    
    # TODO: Run simulations
    # for i in range(num_simulations):
    #     path = simulate_gbm_path(current_price, drift, volatility, forecast_days)
    #     simulations[i, :] = path
    
    # Optional: Show progress for large simulations
    # if i % 100 == 0:
    #     logger.info(f"Completed {i}/{num_simulations} simulations")
    
    pass


# ============================================================================
# HELPER FUNCTION 4: Calculate Statistics from Simulations
# ============================================================================

def calculate_forecast_statistics(simulations: np.ndarray, current_price: float) -> Dict:
    """
    Calculate statistics from Monte Carlo simulations.
    
    Algorithm:
    1. Get final prices (last column of simulations)
    2. Calculate mean and median
    3. Calculate percentiles for confidence intervals:
       - 68% confidence: 16th and 84th percentiles (±1 std dev)
       - 95% confidence: 2.5th and 97.5th percentiles (±2 std dev)
    4. Calculate probability of gain vs loss
    5. Calculate expected return
    
    Args:
        simulations: 2D array from run_monte_carlo_simulations()
        current_price: Starting price for comparison
    
    Returns:
        {
            'mean_forecast': 540.2,
            'median_forecast': 538.5,
            'std_dev': 15.3,
            'confidence_68': {'lower': 525.2, 'upper': 555.2},
            'confidence_95': {'lower': 510.1, 'upper': 570.3},
            'probability_up': 0.68,
            'probability_down': 0.32,
            'expected_return': 2.8,  # % return
            'min_price': 480.5,
            'max_price': 610.3
        }
    
    Example:
        stats = calculate_forecast_statistics(simulations, 525.50)
        # Returns: {'mean_forecast': 540.2, 'probability_up': 0.68, ...}
    """
    # TODO: Get final prices from all simulations
    # final_prices = simulations[:, -1]
    
    # TODO: Calculate basic statistics
    # mean_forecast = final_prices.mean()
    # median_forecast = np.median(final_prices)
    # std_dev = final_prices.std()
    
    # TODO: Calculate confidence intervals
    # Percentiles for 68% confidence (±1 std dev)
    # lower_68 = np.percentile(final_prices, 16)
    # upper_68 = np.percentile(final_prices, 84)
    
    # Percentiles for 95% confidence (±2 std dev)
    # lower_95 = np.percentile(final_prices, 2.5)
    # upper_95 = np.percentile(final_prices, 97.5)
    
    # TODO: Calculate probability of gain
    # prices_above_current = final_prices > current_price
    # probability_up = prices_above_current.sum() / len(final_prices)
    # probability_down = 1 - probability_up
    
    # TODO: Calculate expected return
    # expected_return = ((mean_forecast - current_price) / current_price) * 100
    
    # TODO: Calculate min/max
    # min_price = final_prices.min()
    # max_price = final_prices.max()
    
    pass


# ============================================================================
# HELPER FUNCTION 5: Generate Visualization Data
# ============================================================================

def prepare_visualization_data(simulations: np.ndarray, statistics: Dict) -> Dict:
    """
    Prepare data for dashboard visualization.
    
    Algorithm:
    1. Select subset of paths to display (e.g., 100 out of 1000)
    2. Calculate mean path
    3. Calculate percentile bands
    4. Format for Plotly charting
    
    Args:
        simulations: Full simulation array
        statistics: Statistics from calculate_forecast_statistics()
    
    Returns:
        {
            'sample_paths': array,  # Subset of paths to show (100 x days)
            'mean_path': array,     # Average of all paths
            'upper_68': array,      # Upper 68% confidence band
            'lower_68': array,      # Lower 68% confidence band
            'upper_95': array,      # Upper 95% confidence band
            'lower_95': array,      # Lower 95% confidence band
            'days': [0, 1, 2, 3, 4, 5, 6, 7]  # X-axis
        }
    
    Example:
        viz_data = prepare_visualization_data(simulations, stats)
        # Use in Plotly to create fan chart
    """
    # TODO: Select random subset of paths for display
    # num_display = min(100, simulations.shape[0])
    # indices = np.random.choice(simulations.shape[0], num_display, replace=False)
    # sample_paths = simulations[indices, :]
    
    # TODO: Calculate mean path across all simulations
    # mean_path = simulations.mean(axis=0)
    
    # TODO: Calculate percentile bands for each day
    # upper_68 = np.percentile(simulations, 84, axis=0)
    # lower_68 = np.percentile(simulations, 16, axis=0)
    # upper_95 = np.percentile(simulations, 97.5, axis=0)
    # lower_95 = np.percentile(simulations, 2.5, axis=0)
    
    # TODO: Create day labels
    # days = list(range(simulations.shape[1]))
    
    pass


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def monte_carlo_node(state: 'StockAnalysisState') -> 'StockAnalysisState':
    """
    Node 7: Monte Carlo Price Forecasting
    
    Execution flow:
    1. Get price data from state
    2. Calculate historical drift and volatility
    3. Run 1000 Monte Carlo simulations
    4. Calculate forecast statistics
    5. Prepare visualization data
    6. Update state
    
    Runs in PARALLEL with Nodes 4, 5, 6
    Runs AFTER: Node 1 (need price data)
    Runs BEFORE: Node 8 (verification)
    
    Args:
        state: LangGraph state
        
    Returns:
        Updated state with monte_carlo_forecast results
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    # Configuration
    FORECAST_DAYS = 7  # Forecast 1 week ahead
    NUM_SIMULATIONS = 1000  # Standard for Monte Carlo
    LOOKBACK_DAYS = 30  # Use 30 days of history
    
    try:
        logger.info(f"Node 7: Running Monte Carlo simulation for {ticker}")
        
        # STEP 1: Get price data
        price_data = state.get('raw_price_data')
        
        if price_data is None or price_data.empty:
            raise ValueError("No price data available for Monte Carlo")
        
        if len(price_data) < LOOKBACK_DAYS:
            raise ValueError(f"Insufficient price history (need {LOOKBACK_DAYS} days)")
        
        # STEP 2: Calculate historical statistics
        logger.info(f"Calculating historical statistics from {LOOKBACK_DAYS} days...")
        drift, volatility, current_price = calculate_historical_statistics(
            price_data, 
            days=LOOKBACK_DAYS
        )
        
        logger.info(f"Current price: ${current_price:.2f}")
        logger.info(f"Historical drift: {drift:.6f} ({drift*100:.3f}% per day)")
        logger.info(f"Historical volatility: {volatility:.6f} ({volatility*100:.2f}% per day)")
        
        # STEP 3: Run Monte Carlo simulations
        logger.info(f"Running {NUM_SIMULATIONS} Monte Carlo simulations...")
        simulations = run_monte_carlo_simulations(
            current_price=current_price,
            drift=drift,
            volatility=volatility,
            forecast_days=FORECAST_DAYS,
            num_simulations=NUM_SIMULATIONS
        )
        
        logger.info(f"Simulations shape: {simulations.shape}")
        
        # STEP 4: Calculate statistics
        logger.info("Calculating forecast statistics...")
        statistics = calculate_forecast_statistics(simulations, current_price)
        
        # STEP 5: Prepare visualization data
        logger.info("Preparing visualization data...")
        viz_data = prepare_visualization_data(simulations, statistics)
        
        # STEP 6: Build results
        results = {
            'current_price': float(current_price),
            'drift': float(drift),
            'volatility': float(volatility),
            'forecast_days': FORECAST_DAYS,
            'num_simulations': NUM_SIMULATIONS,
            'simulations': simulations,  # Full array (may be large)
            'mean_forecast': statistics['mean_forecast'],
            'median_forecast': statistics['median_forecast'],
            'std_dev': statistics['std_dev'],
            'confidence_68': statistics['confidence_68'],
            'confidence_95': statistics['confidence_95'],
            'probability_up': statistics['probability_up'],
            'probability_down': statistics['probability_down'],
            'expected_return': statistics['expected_return'],
            'min_price': statistics['min_price'],
            'max_price': statistics['max_price'],
            'visualization_data': viz_data
        }
        
        logger.info(f"Monte Carlo forecast complete:")
        logger.info(f"  Expected price in {FORECAST_DAYS} days: ${statistics['mean_forecast']:.2f}")
        logger.info(f"  68% confidence: ${statistics['confidence_68']['lower']:.2f} - ${statistics['confidence_68']['upper']:.2f}")
        logger.info(f"  95% confidence: ${statistics['confidence_95']['lower']:.2f} - ${statistics['confidence_95']['upper']:.2f}")
        logger.info(f"  Probability of gain: {statistics['probability_up']*100:.1f}%")
        logger.info(f"  Expected return: {statistics['expected_return']:.2f}%")
        
        # Update state
        state['monte_carlo_forecast'] = results
        state['node_execution_times']['node_7'] = (datetime.now() - start_time).total_seconds()
        
        return state
        
    except Exception as e:
        logger.error(f"Node 7 failed: {str(e)}")
        state['errors'].append(f"Monte Carlo simulation failed: {str(e)}")
        
        # Failsafe: return null forecast
        state['monte_carlo_forecast'] = None
        state['node_execution_times']['node_7'] = (datetime.now() - start_time).total_seconds()
        
        return state
```

---

## 📊 Example Output

```python
# For NVIDIA at $525.50, forecasting 7 days ahead

monte_carlo_forecast = {
    'current_price': 525.50,
    'drift': 0.0015,  # 0.15% average daily return
    'volatility': 0.025,  # 2.5% daily volatility
    'forecast_days': 7,
    'num_simulations': 1000,
    
    # Forecast results
    'mean_forecast': 540.20,  # Expected price in 7 days
    'median_forecast': 538.50,
    'std_dev': 15.30,
    
    # Confidence intervals
    'confidence_68': {
        'lower': 525.00,  # 68% chance price is above this
        'upper': 555.00   # 68% chance price is below this
    },
    'confidence_95': {
        'lower': 510.00,  # 95% chance price is above this
        'upper': 570.00   # 95% chance price is below this
    },
    
    # Probabilities
    'probability_up': 0.68,  # 68% chance of gain
    'probability_down': 0.32,  # 32% chance of loss
    'expected_return': 2.8,  # Expected 2.8% return in 7 days
    
    # Extremes
    'min_price': 480.50,  # Worst simulation
    'max_price': 610.30   # Best simulation
}

# Interpretation:
# Current: $525.50
# Expected in 7 days: $540.20 (+2.8%)
# 68% confident: $525-$555 range
# 68% probability of gain
# Very likely to be above $510 (95% confidence)
```

---

## 📈 Visualization Data Structure

The `visualization_data` is used to create a **fan chart** in the dashboard:

```python
visualization_data = {
    'sample_paths': array([[525.5, 528.3, ...], ...]),  # 100 example paths
    'mean_path': array([525.5, 527.2, 529.8, ...]),     # Average path
    'upper_68': array([525.5, 530.0, 535.0, ...]),      # Upper bound
    'lower_68': array([525.5, 524.5, 523.0, ...]),      # Lower bound
    'upper_95': array([525.5, 535.0, 545.0, ...]),      # Wide upper bound
    'lower_95': array([525.5, 520.0, 515.0, ...]),      # Wide lower bound
    'days': [0, 1, 2, 3, 4, 5, 6, 7]
}

# Dashboard uses this to plot:
# - Gray lines: sample_paths (show variability)
# - Blue line: mean_path (expected trajectory)
# - Light blue band: 68% confidence area
# - Very light blue band: 95% confidence area
```

---

## 🧪 Testing Strategy

```python
# File: tests/test_node_07.py

def test_historical_statistics():
    """Test drift and volatility calculation"""
    # Create mock price data with known returns
    prices = pd.DataFrame({
        'close': [100, 101, 102, 101.5, 103]
    })
    drift, vol, current = calculate_historical_statistics(prices)
    assert drift > 0  # Prices trending up
    assert vol > 0  # Has volatility
    assert current == 103

def test_single_gbm_path():
    """Test single simulation path"""
    path = simulate_gbm_path(100, 0.001, 0.02, 7)
    assert len(path) == 8  # 7 days + starting price
    assert path[0] == 100  # Starts at current price
    assert all(p > 0 for p in path)  # Prices stay positive

def test_monte_carlo_shape():
    """Test simulation array shape"""
    sims = run_monte_carlo_simulations(100, 0.001, 0.02, 7, 1000)
    assert sims.shape == (1000, 8)  # 1000 sims, 7 days + start

def test_statistics_calculation():
    """Test statistics from simulations"""
    # Create mock simulations
    sims = np.random.normal(105, 5, (1000, 8))
    sims[:, 0] = 100  # Set start price
    stats = calculate_forecast_statistics(sims, 100)
    assert 'mean_forecast' in stats
    assert 'probability_up' in stats
    assert 0 <= stats['probability_up'] <= 1

def test_confidence_intervals():
    """Test that confidence intervals make sense"""
    # TODO: Verify that 95% CI is wider than 68% CI
    pass
```

---

## 🎯 Key Formulas

**Daily Return:**
```python
return = (price[t] - price[t-1]) / price[t-1]
```

**Drift (μ):**
```python
drift = mean(daily_returns)
```

**Volatility (σ):**
```python
volatility = std(daily_returns)
```

**GBM Next Price:**
```python
S(t+1) = S(t) * exp((μ - 0.5*σ²)*Δt + σ*√Δt*Z)
where Z ~ N(0,1)
```

**Confidence Intervals:**
```python
68% CI = [16th percentile, 84th percentile]  # ±1 std dev
95% CI = [2.5th percentile, 97.5th percentile]  # ±2 std dev
```

**Expected Return:**
```python
expected_return = ((mean_forecast - current_price) / current_price) * 100
```

---

## 🔄 Integration with Other Nodes

**Node 1 (Price Data):**
- Provides historical price data for calculating drift and volatility

**Node 12 (Signal Generation):**
- Uses probability_up and expected_return in final decision
- Higher probability_up = stronger BUY signal

**Node 15 (Dashboard Prep):**
- Uses visualization_data to create fan chart
- Shows all confidence bands

---

## ⚠️ Important Notes

**Performance:**
- 1000 simulations × 7 days = 7000 calculations
- Should complete in < 2 seconds with NumPy
- Use vectorized operations (avoid loops where possible)

**Accuracy:**
- GBM assumes log-normal distribution (standard in finance)
- Assumes constant volatility (not always true)
- Past performance doesn't guarantee future results
- BUT: Standard technique used by quants everywhere

**Limitations:**
- Doesn't account for news events
- Doesn't account for market crashes
- Assumes random walk (efficient market hypothesis)
- Best used as ONE input to final decision, not sole factor

---

## 💡 Cursor Instructions Summary

**For Cursor to implement Node 7:**

1. Create file: `src/langgraph_nodes/node_07_monte_carlo.py`

2. Implement functions in order:
   - `calculate_historical_statistics()` - Get drift, volatility from prices
   - `simulate_gbm_path()` - Single price path simulation
   - `run_monte_carlo_simulations()` - Run 1000 paths
   - `calculate_forecast_statistics()` - Compute mean, CI, probabilities
   - `prepare_visualization_data()` - Format for Plotly
   - `monte_carlo_node()` - Main function

3. Key libraries needed:
   - `numpy` - For simulations and statistics
   - `pandas` - For price data handling
   - `scipy.stats` - For statistical functions

4. Key parameters:
   - `FORECAST_DAYS = 7` (1 week ahead)
   - `NUM_SIMULATIONS = 1000` (standard)
   - `LOOKBACK_DAYS = 30` (historical window)

5. Performance tips:
   - Use NumPy vectorization (avoid Python loops)
   - Generate all random numbers at once
   - Use `np.exp(np.cumsum())` for efficient path calculation

**Difficulty: MEDIUM-HIGH** (math-heavy but well-defined formulas)

---

## 📚 Additional Resources

**Geometric Brownian Motion:**
- https://en.wikipedia.org/wiki/Geometric_Brownian_Motion
- Used in Black-Scholes option pricing
- Standard model for stock price movement

**Monte Carlo in Finance:**
- https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance
- Used by all major financial institutions
- Standard quant technique

**Implementation Reference:**
```python
# Quick example of GBM formula
import numpy as np

S0 = 100  # Starting price
mu = 0.001  # Drift
sigma = 0.02  # Volatility
dt = 1/252  # Daily step
T = 7  # Days
n = 1000  # Simulations

# Generate random shocks
Z = np.random.standard_normal((n, T))

# Calculate price paths
drift_term = (mu - 0.5 * sigma**2) * dt
diffusion_term = sigma * np.sqrt(dt)
shocks = drift_term + diffusion_term * Z
price_paths = S0 * np.exp(np.cumsum(shocks, axis=1))
```

---

**This provides essential price forecasting capability!** 📈
