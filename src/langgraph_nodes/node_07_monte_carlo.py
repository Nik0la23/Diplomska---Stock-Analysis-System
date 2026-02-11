"""
Node 7: Monte Carlo Price Forecasting

Uses Geometric Brownian Motion (GBM) to simulate 1000 possible
future price paths and calculate probability distributions.

Standard technique in quantitative finance.

Formula: S(t+1) = S(t) * exp((μ - 0.5*σ²)*Δt + σ*√Δt*Z)
where:
- μ = drift (historical average return)
- σ = volatility (historical standard deviation)
- Z = random normal variable

Runs in PARALLEL with: Nodes 4, 5, 6
Runs AFTER: Node 1 (needs price data)
Runs BEFORE: Node 8 (verification)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTION 1: Calculate Historical Statistics
# ============================================================================

def calculate_historical_statistics(
    price_data: pd.DataFrame, 
    days: int = 30
) -> Tuple[float, float, float]:
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
        >>> drift, vol, price = calculate_historical_statistics(price_data, days=30)
        >>> print(f"Drift: {drift*100:.3f}%, Volatility: {vol*100:.2f}%")
        Drift: 0.150%, Volatility: 2.50%
    """
    try:
        if len(price_data) < days:
            logger.warning(f"Using all {len(price_data)} days (requested {days})")
            days = len(price_data)
        
        # Get last N days of data
        recent_data = price_data.tail(days).copy()
        
        # Calculate daily returns (percentage change)
        returns = recent_data['close'].pct_change().dropna()
        
        if len(returns) == 0:
            logger.warning("No returns calculated, using defaults")
            return (0.0, 0.02, float(price_data['close'].iloc[-1]))
        
        # Calculate drift (mean return)
        drift = returns.mean()
        
        # Calculate volatility (std of returns)
        volatility = returns.std()
        
        # Get current price
        current_price = float(price_data['close'].iloc[-1])
        
        # Validate results
        if pd.isna(drift) or pd.isna(volatility):
            logger.warning("NaN values in drift/volatility, using defaults")
            return (0.0, 0.02, current_price)
        
        logger.info(f"Historical statistics calculated:")
        logger.info(f"  Drift: {drift:.6f} ({drift*100:.3f}% per day)")
        logger.info(f"  Volatility: {volatility:.6f} ({volatility*100:.2f}% per day)")
        logger.info(f"  Current price: ${current_price:.2f}")
        
        return (float(drift), float(volatility), current_price)
        
    except Exception as e:
        logger.error(f"Failed to calculate statistics: {str(e)}")
        # Return neutral defaults
        current_price = float(price_data['close'].iloc[-1]) if len(price_data) > 0 else 100.0
        return (0.0, 0.02, current_price)


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
        >>> path = simulate_gbm_path(525.50, 0.002, 0.025, 7)
        >>> print(f"Day 7 price: ${path[-1]:.2f}")
        Day 7 price: $540.20
    """
    try:
        # Initialize price array
        prices = np.zeros(days + 1)
        prices[0] = current_price
        
        # Generate all random shocks at once (more efficient)
        Z = np.random.standard_normal(days)
        
        # Calculate drift and diffusion components
        drift_component = (drift - 0.5 * volatility**2) * dt
        diffusion_component = volatility * np.sqrt(dt)
        
        # Vectorized simulation (faster than loop)
        shocks = drift_component + diffusion_component * Z
        prices[1:] = current_price * np.exp(np.cumsum(shocks))
        
        return prices
        
    except Exception as e:
        logger.error(f"GBM simulation failed: {str(e)}")
        # Return flat forecast
        return np.full(days + 1, current_price)


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
        >>> sims = run_monte_carlo_simulations(525.50, 0.002, 0.025, 7, 1000)
        >>> print(f"Shape: {sims.shape}")
        Shape: (1000, 8)
    """
    try:
        logger.info(f"Running {num_simulations} Monte Carlo simulations...")
        
        # Initialize simulations array
        simulations = np.zeros((num_simulations, forecast_days + 1))
        
        # Run simulations
        for i in range(num_simulations):
            path = simulate_gbm_path(current_price, drift, volatility, forecast_days)
            simulations[i, :] = path
            
            # Optional: Show progress for large simulations
            if (i + 1) % 200 == 0:
                logger.info(f"  Completed {i + 1}/{num_simulations} simulations")
        
        logger.info(f"Simulations complete! Shape: {simulations.shape}")
        return simulations
        
    except Exception as e:
        logger.error(f"Monte Carlo simulations failed: {str(e)}")
        # Return default simulations
        simulations = np.full((num_simulations, forecast_days + 1), current_price)
        return simulations


# ============================================================================
# HELPER FUNCTION 4: Calculate Statistics from Simulations
# ============================================================================

def calculate_forecast_statistics(
    simulations: np.ndarray, 
    current_price: float
) -> Dict[str, Any]:
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
        >>> stats = calculate_forecast_statistics(simulations, 525.50)
        >>> print(f"Expected price: ${stats['mean_forecast']:.2f}")
        Expected price: $540.20
    """
    try:
        # Get final prices from all simulations (last day)
        final_prices = simulations[:, -1]
        
        # Calculate basic statistics
        mean_forecast = float(final_prices.mean())
        median_forecast = float(np.median(final_prices))
        std_dev = float(final_prices.std())
        
        # Calculate confidence intervals
        # 68% confidence (±1 standard deviation)
        lower_68 = float(np.percentile(final_prices, 16))
        upper_68 = float(np.percentile(final_prices, 84))
        
        # 95% confidence (±2 standard deviations)
        lower_95 = float(np.percentile(final_prices, 2.5))
        upper_95 = float(np.percentile(final_prices, 97.5))
        
        # Calculate probability of gain
        prices_above_current = final_prices > current_price
        probability_up = float(prices_above_current.sum() / len(final_prices))
        probability_down = 1.0 - probability_up
        
        # Calculate expected return (percentage)
        expected_return = ((mean_forecast - current_price) / current_price) * 100
        
        # Calculate extremes
        min_price = float(final_prices.min())
        max_price = float(final_prices.max())
        
        results = {
            'mean_forecast': mean_forecast,
            'median_forecast': median_forecast,
            'std_dev': std_dev,
            'confidence_68': {
                'lower': lower_68,
                'upper': upper_68
            },
            'confidence_95': {
                'lower': lower_95,
                'upper': upper_95
            },
            'probability_up': probability_up,
            'probability_down': probability_down,
            'expected_return': expected_return,
            'min_price': min_price,
            'max_price': max_price
        }
        
        logger.info(f"Forecast statistics calculated:")
        logger.info(f"  Mean: ${mean_forecast:.2f}")
        logger.info(f"  68% CI: [${lower_68:.2f}, ${upper_68:.2f}]")
        logger.info(f"  95% CI: [${lower_95:.2f}, ${upper_95:.2f}]")
        logger.info(f"  Probability up: {probability_up*100:.1f}%")
        logger.info(f"  Expected return: {expected_return:.2f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Statistics calculation failed: {str(e)}")
        # Return defaults
        return {
            'mean_forecast': current_price,
            'median_forecast': current_price,
            'std_dev': 0.0,
            'confidence_68': {'lower': current_price, 'upper': current_price},
            'confidence_95': {'lower': current_price, 'upper': current_price},
            'probability_up': 0.5,
            'probability_down': 0.5,
            'expected_return': 0.0,
            'min_price': current_price,
            'max_price': current_price
        }


# ============================================================================
# HELPER FUNCTION 5: Generate Visualization Data
# ============================================================================

def prepare_visualization_data(
    simulations: np.ndarray, 
    statistics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare data for dashboard visualization.
    
    Algorithm:
    1. Select subset of paths to display (e.g., 100 out of 1000)
    2. Calculate mean path
    3. Calculate percentile bands for each day
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
        >>> viz_data = prepare_visualization_data(simulations, stats)
        >>> print(f"Sample paths shape: {viz_data['sample_paths'].shape}")
        Sample paths shape: (100, 8)
    """
    try:
        # Select random subset of paths for display (avoid cluttering chart)
        num_display = min(100, simulations.shape[0])
        indices = np.random.choice(simulations.shape[0], num_display, replace=False)
        sample_paths = simulations[indices, :]
        
        # Calculate mean path across all simulations
        mean_path = simulations.mean(axis=0)
        
        # Calculate percentile bands for each day
        upper_68 = np.percentile(simulations, 84, axis=0)
        lower_68 = np.percentile(simulations, 16, axis=0)
        upper_95 = np.percentile(simulations, 97.5, axis=0)
        lower_95 = np.percentile(simulations, 2.5, axis=0)
        
        # Create day labels
        days = list(range(simulations.shape[1]))
        
        viz_data = {
            'sample_paths': sample_paths,
            'mean_path': mean_path,
            'upper_68': upper_68,
            'lower_68': lower_68,
            'upper_95': upper_95,
            'lower_95': lower_95,
            'days': days
        }
        
        logger.info(f"Visualization data prepared: {num_display} sample paths")
        return viz_data
        
    except Exception as e:
        logger.error(f"Visualization data preparation failed: {str(e)}")
        # Return minimal data
        return {
            'sample_paths': simulations[:10],
            'mean_path': simulations.mean(axis=0),
            'upper_68': simulations.mean(axis=0),
            'lower_68': simulations.mean(axis=0),
            'upper_95': simulations.mean(axis=0),
            'lower_95': simulations.mean(axis=0),
            'days': list(range(simulations.shape[1]))
        }


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def monte_carlo_forecasting_node(state: Dict[str, Any]) -> Dict[str, Any]:
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
        Updated state with monte_carlo_results populated
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    # Configuration
    FORECAST_DAYS = 7  # Forecast 1 week ahead
    NUM_SIMULATIONS = 1000  # Standard for Monte Carlo
    LOOKBACK_DAYS = 30  # Use 30 days of history
    
    try:
        logger.info(f"Node 7: Running Monte Carlo simulation for {ticker}")
        
        # ====================================================================
        # STEP 1: Get Price Data
        # ====================================================================
        price_data = state.get('raw_price_data')
        
        if price_data is None or price_data.empty:
            raise ValueError("No price data available for Monte Carlo")
        
        if len(price_data) < LOOKBACK_DAYS:
            raise ValueError(f"Insufficient price history (need {LOOKBACK_DAYS} days, have {len(price_data)})")
        
        logger.info(f"Using {len(price_data)} days of price data")
        
        # ====================================================================
        # STEP 2: Calculate Historical Statistics
        # ====================================================================
        logger.info(f"Calculating historical statistics from {LOOKBACK_DAYS} days...")
        drift, volatility, current_price = calculate_historical_statistics(
            price_data, 
            days=LOOKBACK_DAYS
        )
        
        # ====================================================================
        # STEP 3: Run Monte Carlo Simulations
        # ====================================================================
        logger.info(f"Running {NUM_SIMULATIONS} Monte Carlo simulations for {FORECAST_DAYS} days...")
        simulations = run_monte_carlo_simulations(
            current_price=current_price,
            drift=drift,
            volatility=volatility,
            forecast_days=FORECAST_DAYS,
            num_simulations=NUM_SIMULATIONS
        )
        
        # ====================================================================
        # STEP 4: Calculate Statistics
        # ====================================================================
        logger.info("Calculating forecast statistics...")
        statistics = calculate_forecast_statistics(simulations, current_price)
        
        # ====================================================================
        # STEP 5: Prepare Visualization Data
        # ====================================================================
        logger.info("Preparing visualization data...")
        viz_data = prepare_visualization_data(simulations, statistics)
        
        # ====================================================================
        # STEP 6: Build Results
        # ====================================================================
        results = {
            'current_price': current_price,
            'drift': drift,
            'volatility': volatility,
            'forecast_days': FORECAST_DAYS,
            'num_simulations': NUM_SIMULATIONS,
            'simulations': simulations.tolist(),  # Convert to list for JSON serialization
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
            'visualization_data': {
                'sample_paths': viz_data['sample_paths'].tolist(),
                'mean_path': viz_data['mean_path'].tolist(),
                'upper_68': viz_data['upper_68'].tolist(),
                'lower_68': viz_data['lower_68'].tolist(),
                'upper_95': viz_data['upper_95'].tolist(),
                'lower_95': viz_data['lower_95'].tolist(),
                'days': viz_data['days']
            }
        }
        
        # ====================================================================
        # STEP 7: Log Results
        # ====================================================================
        logger.info(f"Monte Carlo forecast complete for {ticker}:")
        logger.info(f"  Expected price in {FORECAST_DAYS} days: ${statistics['mean_forecast']:.2f}")
        logger.info(f"  68% confidence: ${statistics['confidence_68']['lower']:.2f} - ${statistics['confidence_68']['upper']:.2f}")
        logger.info(f"  95% confidence: ${statistics['confidence_95']['lower']:.2f} - ${statistics['confidence_95']['upper']:.2f}")
        logger.info(f"  Probability of gain: {statistics['probability_up']*100:.1f}%")
        logger.info(f"  Expected return: {statistics['expected_return']:.2f}%")
        
        # ====================================================================
        # STEP 8: Update State
        # ====================================================================
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Node 7: Monte Carlo completed in {elapsed:.2f}s")
        
        # Return only the fields this node updates (for parallel execution)
        return {
            'monte_carlo_results': results,
            'forecasted_price': statistics['mean_forecast'],
            'price_range': (
                statistics['confidence_95']['lower'],
                statistics['confidence_95']['upper']
            ),
            'node_execution_times': {'node_7': elapsed}
        }
        
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Node 7: Monte Carlo failed for {ticker}: {str(e)}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Return only the fields this node updates (for parallel execution)
        return {
            'errors': [f"Node 7: Monte Carlo simulation failed - {str(e)}"],
            'monte_carlo_results': None,
            'forecasted_price': None,
            'price_range': None,
            'node_execution_times': {'node_7': elapsed}
        }
