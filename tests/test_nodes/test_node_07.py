"""
Tests for Node 7: Monte Carlo Price Forecasting

Tests cover:
1. Historical statistics calculation (drift, volatility)
2. Single GBM path simulation
3. Monte Carlo simulations (1000 paths)
4. Forecast statistics calculation
5. Visualization data preparation
6. Main node function execution
7. Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.langgraph_nodes.node_07_monte_carlo import (
    calculate_historical_statistics,
    simulate_gbm_path,
    run_monte_carlo_simulations,
    calculate_forecast_statistics,
    prepare_visualization_data,
    monte_carlo_forecasting_node
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_price_data_100_days():
    """Generate 100 days of realistic price data"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Generate realistic price movement (small uptrend with volatility)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)  # 0.1% avg return, 2% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    return df


@pytest.fixture
def uptrend_price_data():
    """Generate strong uptrend data"""
    np.random.seed(123)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Strong uptrend with positive drift
    base_price = 100
    returns = np.random.normal(0.01, 0.015, 100)  # 1% avg return, 1.5% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': [1000000] * 100
    })
    
    return df


@pytest.fixture
def insufficient_price_data():
    """Generate insufficient data (< 30 days)"""
    dates = pd.date_range(end=datetime.now(), periods=20, freq='D')
    prices = np.linspace(100, 105, 20)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': [1000000] * 20
    })
    
    return df


# ============================================================================
# TEST HISTORICAL STATISTICS
# ============================================================================

def test_calculate_historical_statistics_valid_data(sample_price_data_100_days):
    """Test historical statistics calculation with valid data"""
    drift, volatility, current_price = calculate_historical_statistics(
        sample_price_data_100_days, 
        days=30
    )
    
    # Check types
    assert isinstance(drift, float)
    assert isinstance(volatility, float)
    assert isinstance(current_price, float)
    
    # Check reasonable values
    assert -0.1 < drift < 0.1  # Daily drift typically small
    assert 0 < volatility < 0.5  # Volatility should be positive and reasonable
    assert current_price > 0  # Price must be positive


def test_calculate_historical_statistics_uptrend(uptrend_price_data):
    """Test that uptrend data produces positive drift"""
    drift, volatility, current_price = calculate_historical_statistics(
        uptrend_price_data,
        days=30
    )
    
    # Uptrend should have positive drift
    assert drift > 0
    assert volatility > 0


def test_calculate_historical_statistics_insufficient_data(insufficient_price_data):
    """Test statistics calculation with insufficient data"""
    # Should handle gracefully
    drift, volatility, current_price = calculate_historical_statistics(
        insufficient_price_data,
        days=30
    )
    
    # Should return defaults or use available data
    assert isinstance(drift, float)
    assert isinstance(volatility, float)
    assert current_price > 0


def test_calculate_historical_statistics_values(sample_price_data_100_days):
    """Test that calculated values are reasonable"""
    drift, volatility, current_price = calculate_historical_statistics(
        sample_price_data_100_days,
        days=30
    )
    
    # Drift should be small (daily returns are typically < 1%)
    assert abs(drift) < 0.05
    
    # Volatility should be reasonable (typically 1-5% daily)
    assert 0.005 < volatility < 0.1
    
    # Current price should match last price
    expected_price = sample_price_data_100_days['close'].iloc[-1]
    assert abs(current_price - expected_price) < 0.01


# ============================================================================
# TEST GBM SIMULATION
# ============================================================================

def test_simulate_gbm_path_shape():
    """Test that GBM path has correct shape"""
    path = simulate_gbm_path(
        current_price=100.0,
        drift=0.001,
        volatility=0.02,
        days=7
    )
    
    # Should have days+1 elements (including start price)
    assert len(path) == 8
    assert isinstance(path, np.ndarray)


def test_simulate_gbm_path_starts_at_current_price():
    """Test that simulation starts at current price"""
    current_price = 525.50
    path = simulate_gbm_path(
        current_price=current_price,
        drift=0.001,
        volatility=0.02,
        days=7
    )
    
    # First price should be current price
    assert path[0] == current_price


def test_simulate_gbm_path_all_positive():
    """Test that all simulated prices stay positive"""
    path = simulate_gbm_path(
        current_price=100.0,
        drift=0.001,
        volatility=0.02,
        days=30
    )
    
    # All prices should be positive
    assert np.all(path > 0)


def test_simulate_gbm_path_randomness():
    """Test that multiple paths are different (randomness works)"""
    np.random.seed(42)
    path1 = simulate_gbm_path(100.0, 0.001, 0.02, 7)
    
    np.random.seed(123)
    path2 = simulate_gbm_path(100.0, 0.001, 0.02, 7)
    
    # Paths should be different (except first element)
    assert not np.array_equal(path1[1:], path2[1:])
    assert path1[0] == path2[0]  # Both start at same price


# ============================================================================
# TEST MONTE CARLO SIMULATIONS
# ============================================================================

def test_run_monte_carlo_simulations_shape():
    """Test that Monte Carlo produces correct shape"""
    simulations = run_monte_carlo_simulations(
        current_price=100.0,
        drift=0.001,
        volatility=0.02,
        forecast_days=7,
        num_simulations=1000
    )
    
    # Should be (1000, 8) - 1000 simulations, 7 days + start
    assert simulations.shape == (1000, 8)
    assert isinstance(simulations, np.ndarray)


def test_run_monte_carlo_simulations_all_start_same():
    """Test that all simulations start at current price"""
    current_price = 525.50
    simulations = run_monte_carlo_simulations(
        current_price=current_price,
        drift=0.001,
        volatility=0.02,
        forecast_days=7,
        num_simulations=100
    )
    
    # All first prices should be current price
    assert np.all(simulations[:, 0] == current_price)


def test_run_monte_carlo_simulations_variability():
    """Test that simulations show variability"""
    simulations = run_monte_carlo_simulations(
        current_price=100.0,
        drift=0.001,
        volatility=0.02,
        forecast_days=7,
        num_simulations=1000
    )
    
    # Final prices should have variance
    final_prices = simulations[:, -1]
    std_dev = np.std(final_prices)
    
    assert std_dev > 0  # Should have variability
    assert std_dev < 50  # But not unreasonably high


def test_run_monte_carlo_simulations_positive_prices():
    """Test that all simulated prices stay positive"""
    simulations = run_monte_carlo_simulations(
        current_price=100.0,
        drift=0.001,
        volatility=0.02,
        forecast_days=30,
        num_simulations=500
    )
    
    # All prices should be positive
    assert np.all(simulations > 0)


# ============================================================================
# TEST FORECAST STATISTICS
# ============================================================================

def test_calculate_forecast_statistics_structure():
    """Test that forecast statistics have correct structure"""
    simulations = run_monte_carlo_simulations(100.0, 0.001, 0.02, 7, 1000)
    stats = calculate_forecast_statistics(simulations, 100.0)
    
    # Check all required fields exist
    required_fields = [
        'mean_forecast', 'median_forecast', 'std_dev',
        'confidence_68', 'confidence_95',
        'probability_up', 'probability_down',
        'expected_return', 'min_price', 'max_price'
    ]
    
    for field in required_fields:
        assert field in stats


def test_calculate_forecast_statistics_confidence_intervals():
    """Test that confidence intervals are ordered correctly"""
    simulations = run_monte_carlo_simulations(100.0, 0.001, 0.02, 7, 1000)
    stats = calculate_forecast_statistics(simulations, 100.0)
    
    # 95% CI should be wider than 68% CI
    assert stats['confidence_95']['lower'] <= stats['confidence_68']['lower']
    assert stats['confidence_95']['upper'] >= stats['confidence_68']['upper']
    
    # Lower bounds should be below upper bounds
    assert stats['confidence_68']['lower'] < stats['confidence_68']['upper']
    assert stats['confidence_95']['lower'] < stats['confidence_95']['upper']


def test_calculate_forecast_statistics_probabilities():
    """Test that probabilities sum to 1.0"""
    simulations = run_monte_carlo_simulations(100.0, 0.001, 0.02, 7, 1000)
    stats = calculate_forecast_statistics(simulations, 100.0)
    
    # Probabilities should sum to 1.0
    total_prob = stats['probability_up'] + stats['probability_down']
    assert abs(total_prob - 1.0) < 0.001
    
    # Each probability should be between 0 and 1
    assert 0 <= stats['probability_up'] <= 1
    assert 0 <= stats['probability_down'] <= 1


def test_calculate_forecast_statistics_mean_median():
    """Test that mean and median are reasonable"""
    simulations = run_monte_carlo_simulations(100.0, 0.001, 0.02, 7, 1000)
    stats = calculate_forecast_statistics(simulations, 100.0)
    
    # Mean and median should be close to each other (normal distribution)
    assert abs(stats['mean_forecast'] - stats['median_forecast']) < 10
    
    # Should be within min/max range
    assert stats['min_price'] <= stats['mean_forecast'] <= stats['max_price']
    assert stats['min_price'] <= stats['median_forecast'] <= stats['max_price']


def test_calculate_forecast_statistics_uptrend():
    """Test that uptrend data produces positive expected return"""
    # Create simulations with positive drift
    simulations = run_monte_carlo_simulations(
        current_price=100.0,
        drift=0.01,  # Strong positive drift
        volatility=0.015,
        forecast_days=7,
        num_simulations=1000
    )
    stats = calculate_forecast_statistics(simulations, 100.0)
    
    # Expected return should be positive
    assert stats['expected_return'] > 0
    assert stats['mean_forecast'] > 100.0
    assert stats['probability_up'] > 0.5


# ============================================================================
# TEST VISUALIZATION DATA
# ============================================================================

def test_prepare_visualization_data_structure():
    """Test that visualization data has correct structure"""
    simulations = run_monte_carlo_simulations(100.0, 0.001, 0.02, 7, 1000)
    stats = calculate_forecast_statistics(simulations, 100.0)
    viz_data = prepare_visualization_data(simulations, stats)
    
    # Check all required fields
    required_fields = [
        'sample_paths', 'mean_path',
        'upper_68', 'lower_68',
        'upper_95', 'lower_95',
        'days'
    ]
    
    for field in required_fields:
        assert field in viz_data


def test_prepare_visualization_data_sample_paths():
    """Test that sample paths are subset of simulations"""
    simulations = run_monte_carlo_simulations(100.0, 0.001, 0.02, 7, 1000)
    stats = calculate_forecast_statistics(simulations, 100.0)
    viz_data = prepare_visualization_data(simulations, stats)
    
    sample_paths = viz_data['sample_paths']
    
    # Should be at most 100 paths
    assert len(sample_paths) <= 100
    
    # Each path should have 8 days (7 forecast + start)
    assert sample_paths.shape[1] == 8


def test_prepare_visualization_data_bands():
    """Test that confidence bands are ordered correctly"""
    simulations = run_monte_carlo_simulations(100.0, 0.001, 0.02, 7, 1000)
    stats = calculate_forecast_statistics(simulations, 100.0)
    viz_data = prepare_visualization_data(simulations, stats)
    
    # At each day, 95% bands should be wider than 68% bands
    for day in range(8):
        assert viz_data['lower_95'][day] <= viz_data['lower_68'][day]
        assert viz_data['upper_95'][day] >= viz_data['upper_68'][day]
        
        # Mean should be within bands
        assert viz_data['lower_95'][day] <= viz_data['mean_path'][day] <= viz_data['upper_95'][day]


# ============================================================================
# TEST MAIN NODE FUNCTION
# ============================================================================

def test_monte_carlo_forecasting_node_success(sample_price_data_100_days):
    """Test main node function with valid data"""
    state = {
        'ticker': 'AAPL',
        'raw_price_data': sample_price_data_100_days,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = monte_carlo_forecasting_node(state)
    
    # Check that state was updated
    assert 'monte_carlo_results' in result
    assert 'forecasted_price' in result
    assert 'price_range' in result
    assert 'node_7' in result['node_execution_times']
    
    # Check monte_carlo_results structure
    assert result['monte_carlo_results'] is not None
    mc_results = result['monte_carlo_results']
    
    assert 'mean_forecast' in mc_results
    assert 'confidence_68' in mc_results
    assert 'confidence_95' in mc_results
    assert 'probability_up' in mc_results
    assert 'simulations' in mc_results
    
    # Check forecasted_price
    assert result['forecasted_price'] == mc_results['mean_forecast']
    
    # Check price_range (95% CI)
    assert result['price_range'][0] == mc_results['confidence_95']['lower']
    assert result['price_range'][1] == mc_results['confidence_95']['upper']


def test_monte_carlo_forecasting_node_no_price_data():
    """Test node handles missing price data"""
    state = {
        'ticker': 'AAPL',
        'raw_price_data': None,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = monte_carlo_forecasting_node(state)
    
    # Should add error
    assert len(result['errors']) > 0
    assert 'Node 7' in result['errors'][0]
    
    # Should set defaults
    assert result['monte_carlo_results'] is None
    assert result['forecasted_price'] is None
    assert result['price_range'] is None


def test_monte_carlo_forecasting_node_insufficient_data(insufficient_price_data):
    """Test node handles insufficient price data"""
    state = {
        'ticker': 'AAPL',
        'raw_price_data': insufficient_price_data,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = monte_carlo_forecasting_node(state)
    
    # Should add error about insufficient data
    assert len(result['errors']) > 0
    assert 'insufficient' in result['errors'][0].lower() or 'failed' in result['errors'][0].lower()


def test_monte_carlo_forecasting_node_execution_time(sample_price_data_100_days):
    """Test node executes within acceptable time"""
    state = {
        'ticker': 'AAPL',
        'raw_price_data': sample_price_data_100_days,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = monte_carlo_forecasting_node(state)
    
    # Should complete in < 5 seconds
    assert result['node_execution_times']['node_7'] < 5.0


def test_monte_carlo_forecasting_node_results_reasonable(sample_price_data_100_days):
    """Test that forecast results are reasonable"""
    state = {
        'ticker': 'NVDA',
        'raw_price_data': sample_price_data_100_days,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = monte_carlo_forecasting_node(state)
    
    mc_results = result['monte_carlo_results']
    current_price = mc_results['current_price']
    
    # Forecast should not be wildly different from current price
    # (e.g., not predicting 10x gains or 90% losses in 7 days)
    mean_forecast = mc_results['mean_forecast']
    assert 0.5 * current_price < mean_forecast < 2.0 * current_price
    
    # Confidence intervals should be reasonable
    assert mc_results['confidence_95']['lower'] > 0
    assert mc_results['confidence_95']['upper'] < 10 * current_price


# ============================================================================
# TEST INTEGRATION
# ============================================================================

def test_full_monte_carlo_flow(sample_price_data_100_days):
    """Test complete Monte Carlo forecasting flow"""
    # Simulate complete flow
    state = {
        'ticker': 'NVDA',
        'raw_price_data': sample_price_data_100_days,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = monte_carlo_forecasting_node(state)
    
    # Verify all components
    assert result['monte_carlo_results'] is not None
    assert result['forecasted_price'] is not None
    assert result['price_range'] is not None
    
    # Verify forecast quality
    mc_results = result['monte_carlo_results']
    assert mc_results['num_simulations'] == 1000
    assert mc_results['forecast_days'] == 7
    
    # Verify statistics
    assert 'mean_forecast' in mc_results
    assert 'probability_up' in mc_results
    assert 'expected_return' in mc_results
    
    # Verify visualization data
    assert 'visualization_data' in mc_results
    viz_data = mc_results['visualization_data']
    assert 'sample_paths' in viz_data
    assert 'mean_path' in viz_data
    
    # Verify no errors
    assert len(result['errors']) == 0
    
    print(f"\n✅ Monte Carlo Forecasting Complete:")
    print(f"   Current Price: ${mc_results['current_price']:.2f}")
    print(f"   Mean Forecast (7 days): ${mc_results['mean_forecast']:.2f}")
    print(f"   Expected Return: {mc_results['expected_return']:.2f}%")
    print(f"   Probability Up: {mc_results['probability_up']*100:.1f}%")
    print(f"   95% CI: [${mc_results['confidence_95']['lower']:.2f}, ${mc_results['confidence_95']['upper']:.2f}]")


def test_monte_carlo_serialization(sample_price_data_100_days):
    """Test that Monte Carlo results can be serialized"""
    state = {
        'ticker': 'AAPL',
        'raw_price_data': sample_price_data_100_days,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = monte_carlo_forecasting_node(state)
    mc_results = result['monte_carlo_results']
    
    # Results should be JSON-serializable (no numpy arrays)
    import json
    try:
        json_str = json.dumps(mc_results)
        assert len(json_str) > 0
    except TypeError:
        pytest.fail("Monte Carlo results not JSON-serializable")
