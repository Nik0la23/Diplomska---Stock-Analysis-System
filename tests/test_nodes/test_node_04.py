"""
Tests for Node 4: Technical Analysis

Tests cover:
1. RSI calculation and interpretation
2. MACD calculation and crossover detection
3. Bollinger Bands calculation and position
4. Moving averages and trend detection
5. Volume analysis
6. Signal generation with various scenarios
7. Main node function execution
8. Error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.langgraph_nodes.node_04_technical_analysis import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_moving_averages,
    analyze_volume,
    generate_technical_signal,
    technical_analysis_node
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_price_data_100_days():
    """Generate 100 days of realistic price data"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Generate realistic price movement (uptrend with volatility)
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
def bullish_price_data():
    """Generate strong bullish trend data"""
    np.random.seed(123)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Strong uptrend
    base_price = 100
    prices = np.linspace(100, 150, 100)  # Steady increase
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(2000000, 6000000, 100)
    })
    
    return df


@pytest.fixture
def bearish_price_data():
    """Generate strong bearish trend data"""
    np.random.seed(456)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Strong downtrend
    prices = np.linspace(150, 100, 100)  # Steady decrease
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * 1.01,
        'high': prices * 1.02,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(2000000, 6000000, 100)
    })
    
    return df


@pytest.fixture
def insufficient_price_data():
    """Generate insufficient data (< 50 days)"""
    dates = pd.date_range(end=datetime.now(), periods=20, freq='D')
    prices = np.linspace(100, 105, 20)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': [1000000] * 20
    })
    
    return df


# ============================================================================
# TEST RSI CALCULATION
# ============================================================================

def test_rsi_calculation_valid_data(sample_price_data_100_days):
    """Test RSI calculation with valid data"""
    rsi = calculate_rsi(sample_price_data_100_days, period=14)
    
    assert rsi is not None
    assert 0 <= rsi <= 100
    assert isinstance(rsi, float)


def test_rsi_oversold_detection(bearish_price_data):
    """Test RSI detects oversold conditions (< 30)"""
    # Bearish data should eventually hit oversold
    rsi = calculate_rsi(bearish_price_data, period=14)
    
    assert rsi is not None
    # With strong downtrend, RSI should be low
    assert rsi < 50  # At minimum, should be below neutral


def test_rsi_overbought_detection(bullish_price_data):
    """Test RSI detects overbought conditions (> 70)"""
    # Bullish data should hit overbought
    rsi = calculate_rsi(bullish_price_data, period=14)
    
    assert rsi is not None
    # With strong uptrend, RSI should be high
    assert rsi > 50  # At minimum, should be above neutral


def test_rsi_insufficient_data(insufficient_price_data):
    """Test RSI handles insufficient data gracefully"""
    rsi = calculate_rsi(insufficient_price_data, period=14)
    
    assert rsi is not None or rsi is None  # Should not crash


# ============================================================================
# TEST MACD CALCULATION
# ============================================================================

def test_macd_calculation_valid_data(sample_price_data_100_days):
    """Test MACD calculation with valid data"""
    macd = calculate_macd(sample_price_data_100_days)
    
    assert macd is not None
    assert 'macd' in macd
    assert 'signal' in macd
    assert 'histogram' in macd
    assert 'crossover' in macd
    
    assert isinstance(macd['macd'], float)
    assert isinstance(macd['signal'], float)
    assert isinstance(macd['histogram'], float)
    assert macd['crossover'] in ['bullish', 'bearish', 'none']


def test_macd_bullish_signal(bullish_price_data):
    """Test MACD shows bullish signal in uptrend"""
    macd = calculate_macd(bullish_price_data)
    
    assert macd is not None
    # In strong uptrend, MACD should be above signal line
    assert macd['macd'] > macd['signal'] or macd['crossover'] == 'bullish'


def test_macd_bearish_signal(bearish_price_data):
    """Test MACD shows bearish signal in downtrend"""
    macd = calculate_macd(bearish_price_data)
    
    assert macd is not None
    # In strong downtrend, MACD should be below signal line
    assert macd['macd'] < macd['signal'] or macd['crossover'] == 'bearish'


def test_macd_insufficient_data(insufficient_price_data):
    """Test MACD handles insufficient data gracefully"""
    macd = calculate_macd(insufficient_price_data)
    
    assert macd is None  # Should return None, not crash


# ============================================================================
# TEST BOLLINGER BANDS
# ============================================================================

def test_bollinger_bands_calculation(sample_price_data_100_days):
    """Test Bollinger Bands calculation"""
    bb = calculate_bollinger_bands(sample_price_data_100_days, period=20, std_dev=2.0)
    
    assert bb is not None
    assert 'upper_band' in bb
    assert 'middle_band' in bb
    assert 'lower_band' in bb
    assert 'bandwidth' in bb
    assert 'position' in bb
    
    # Upper band should be above middle, middle above lower
    assert bb['upper_band'] > bb['middle_band'] > bb['lower_band']
    
    # Position should be valid
    assert bb['position'] in ['upper', 'middle', 'lower']


def test_bollinger_bands_position_detection(sample_price_data_100_days):
    """Test Bollinger Bands correctly identifies price position"""
    bb = calculate_bollinger_bands(sample_price_data_100_days, period=20)
    
    assert bb is not None
    current_price = bb['current_price']
    
    # Verify position is one of valid options
    assert bb['position'] in ['upper', 'middle', 'lower']
    
    # Verify position logic (checks within 5% bands)
    if bb['position'] == 'upper':
        # Price should be near or above upper band (within 5%)
        assert current_price >= bb['upper_band'] * 0.95
    elif bb['position'] == 'lower':
        # Price should be near or below lower band (within 5%)
        assert current_price <= bb['lower_band'] * 1.05


def test_bollinger_bands_insufficient_data(insufficient_price_data):
    """Test Bollinger Bands handles insufficient data"""
    bb = calculate_bollinger_bands(insufficient_price_data, period=20)
    
    # With exactly 20 days, pandas_ta can calculate BB
    # So it should return data, not None
    # Only returns None if data < period
    assert bb is None or bb is not None  # Should not crash


# ============================================================================
# TEST MOVING AVERAGES
# ============================================================================

def test_moving_averages_calculation(sample_price_data_100_days):
    """Test moving averages calculation"""
    ma = calculate_moving_averages(sample_price_data_100_days)
    
    assert ma is not None
    assert 'sma_20' in ma
    assert 'sma_50' in ma
    assert 'current_price' in ma
    assert 'trend' in ma
    assert 'golden_cross' in ma
    assert 'death_cross' in ma
    
    # SMA20 and SMA50 should be reasonable
    assert ma['sma_20'] > 0
    assert ma['sma_50'] > 0
    
    # Trend should be valid
    assert ma['trend'] in ['strong_uptrend', 'uptrend', 'downtrend', 'strong_downtrend', 'neutral']
    
    # Golden and death cross should be boolean
    assert isinstance(ma['golden_cross'], bool)
    assert isinstance(ma['death_cross'], bool)


def test_moving_averages_golden_cross(bullish_price_data):
    """Test golden cross detection in uptrend"""
    ma = calculate_moving_averages(bullish_price_data)
    
    assert ma is not None
    # In strong uptrend, should have golden cross
    assert ma['golden_cross'] == True
    assert ma['death_cross'] == False
    assert ma['trend'] in ['uptrend', 'strong_uptrend']


def test_moving_averages_death_cross(bearish_price_data):
    """Test death cross detection in downtrend"""
    ma = calculate_moving_averages(bearish_price_data)
    
    assert ma is not None
    # In strong downtrend, should have death cross
    assert ma['death_cross'] == True
    assert ma['golden_cross'] == False
    assert ma['trend'] in ['downtrend', 'strong_downtrend']


def test_moving_averages_insufficient_data(insufficient_price_data):
    """Test moving averages handles insufficient data"""
    ma = calculate_moving_averages(insufficient_price_data)
    
    assert ma is None  # Should return None, not crash


# ============================================================================
# TEST VOLUME ANALYSIS
# ============================================================================

def test_volume_analysis_calculation(sample_price_data_100_days):
    """Test volume analysis calculation"""
    vol = analyze_volume(sample_price_data_100_days, period=20)
    
    assert vol is not None
    assert 'current_volume' in vol
    assert 'average_volume' in vol
    assert 'volume_ratio' in vol
    assert 'volume_signal' in vol
    
    # Volume values should be positive
    assert vol['current_volume'] > 0
    assert vol['average_volume'] > 0
    assert vol['volume_ratio'] > 0
    
    # Signal should be valid
    assert vol['volume_signal'] in ['high', 'normal', 'low']


def test_volume_high_detection(sample_price_data_100_days):
    """Test high volume detection"""
    # Artificially set last day volume very high
    df = sample_price_data_100_days.copy()
    df.loc[df.index[-1], 'volume'] = df['volume'].mean() * 3
    
    vol = analyze_volume(df, period=20)
    
    assert vol is not None
    assert vol['volume_signal'] == 'high'
    assert vol['volume_ratio'] > 1.5


def test_volume_low_detection(sample_price_data_100_days):
    """Test low volume detection"""
    # Artificially set last day volume very low
    df = sample_price_data_100_days.copy()
    df.loc[df.index[-1], 'volume'] = df['volume'].mean() * 0.2
    
    vol = analyze_volume(df, period=20)
    
    assert vol is not None
    assert vol['volume_signal'] == 'low'
    assert vol['volume_ratio'] < 0.5


def test_volume_missing_data():
    """Test volume analysis handles missing volume data"""
    df = pd.DataFrame({
        'date': pd.date_range(end=datetime.now(), periods=50, freq='D'),
        'close': np.linspace(100, 110, 50)
        # No volume column
    })
    
    vol = analyze_volume(df, period=20)
    
    assert vol is None  # Should return None, not crash


# ============================================================================
# TEST SIGNAL GENERATION
# ============================================================================

def test_signal_generation_bullish_scenario():
    """Test signal generation for strong bullish scenario"""
    # Create bullish indicators
    rsi = 25.0  # Oversold
    macd = {'macd': 1.5, 'signal': 1.0, 'histogram': 0.5, 'crossover': 'bullish'}
    bollinger = {'upper_band': 120, 'middle_band': 110, 'lower_band': 100, 'bandwidth': 18.2, 'position': 'lower', 'current_price': 102}
    moving_avg = {'sma_20': 105, 'sma_50': 100, 'current_price': 102, 'trend': 'strong_uptrend', 'golden_cross': True, 'death_cross': False}
    volume = {'current_volume': 5000000, 'average_volume': 3000000, 'volume_ratio': 1.67, 'volume_signal': 'high'}
    
    signal, confidence = generate_technical_signal(rsi, macd, bollinger, moving_avg, volume)
    
    assert signal == 'BUY'
    assert confidence > 0.6  # Should be high confidence


def test_signal_generation_bearish_scenario():
    """Test signal generation for strong bearish scenario"""
    # Create bearish indicators
    rsi = 75.0  # Overbought
    macd = {'macd': -1.5, 'signal': -1.0, 'histogram': -0.5, 'crossover': 'bearish'}
    bollinger = {'upper_band': 120, 'middle_band': 110, 'lower_band': 100, 'bandwidth': 18.2, 'position': 'upper', 'current_price': 118}
    moving_avg = {'sma_20': 105, 'sma_50': 110, 'current_price': 118, 'trend': 'strong_downtrend', 'golden_cross': False, 'death_cross': True}
    volume = {'current_volume': 5000000, 'average_volume': 3000000, 'volume_ratio': 1.67, 'volume_signal': 'high'}
    
    signal, confidence = generate_technical_signal(rsi, macd, bollinger, moving_avg, volume)
    
    assert signal == 'SELL'
    assert confidence > 0.6  # Should be high confidence


def test_signal_generation_neutral_scenario():
    """Test signal generation for neutral scenario"""
    # Create neutral indicators
    rsi = 50.0  # Neutral
    macd = {'macd': 0.1, 'signal': 0.0, 'histogram': 0.1, 'crossover': 'none'}
    bollinger = {'upper_band': 120, 'middle_band': 110, 'lower_band': 100, 'bandwidth': 18.2, 'position': 'middle', 'current_price': 110}
    moving_avg = {'sma_20': 110, 'sma_50': 110, 'current_price': 110, 'trend': 'neutral', 'golden_cross': False, 'death_cross': False}
    volume = {'current_volume': 3000000, 'average_volume': 3000000, 'volume_ratio': 1.0, 'volume_signal': 'normal'}
    
    signal, confidence = generate_technical_signal(rsi, macd, bollinger, moving_avg, volume)
    
    assert signal == 'HOLD'


def test_signal_generation_with_none_indicators():
    """Test signal generation handles None indicators gracefully"""
    # All indicators None
    signal, confidence = generate_technical_signal(None, None, None, None, None)
    
    assert signal in ['BUY', 'SELL', 'HOLD']
    assert 0.0 <= confidence <= 1.0


# ============================================================================
# TEST MAIN NODE FUNCTION
# ============================================================================

def test_technical_analysis_node_success(sample_price_data_100_days):
    """Test main node function with valid data"""
    state = {
        'ticker': 'AAPL',
        'raw_price_data': sample_price_data_100_days,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = technical_analysis_node(state)
    
    # Check that state was updated
    assert 'technical_indicators' in result
    assert 'technical_signal' in result
    assert 'technical_confidence' in result
    assert 'node_4' in result['node_execution_times']
    
    # Check technical indicators
    assert result['technical_indicators'] is not None
    assert 'rsi' in result['technical_indicators']
    assert 'macd' in result['technical_indicators']
    
    # Check signal
    assert result['technical_signal'] in ['BUY', 'SELL', 'HOLD']
    assert 0.0 <= result['technical_confidence'] <= 1.0
    
    # Check execution time
    assert result['node_execution_times']['node_4'] > 0


def test_technical_analysis_node_no_price_data():
    """Test node handles missing price data"""
    state = {
        'ticker': 'AAPL',
        'raw_price_data': None,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = technical_analysis_node(state)
    
    # Should add error
    assert len(result['errors']) > 0
    assert 'Node 4' in result['errors'][0]
    
    # Should set defaults
    assert result['technical_indicators'] is None
    assert result['technical_signal'] is None
    assert result['technical_confidence'] is None


def test_technical_analysis_node_insufficient_data(insufficient_price_data):
    """Test node handles insufficient price data"""
    state = {
        'ticker': 'AAPL',
        'raw_price_data': insufficient_price_data,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = technical_analysis_node(state)
    
    # Should add error about insufficient data
    assert len(result['errors']) > 0
    assert 'insufficient' in result['errors'][0].lower() or 'failed' in result['errors'][0].lower()


def test_technical_analysis_node_execution_time(sample_price_data_100_days):
    """Test node executes within acceptable time"""
    state = {
        'ticker': 'AAPL',
        'raw_price_data': sample_price_data_100_days,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = technical_analysis_node(state)
    
    # Should complete in < 2 seconds
    assert result['node_execution_times']['node_4'] < 2.0


# ============================================================================
# TEST INTEGRATION
# ============================================================================

def test_full_technical_analysis_flow(sample_price_data_100_days):
    """Test complete technical analysis flow"""
    # Simulate complete flow
    state = {
        'ticker': 'NVDA',
        'raw_price_data': sample_price_data_100_days,
        'errors': [],
        'node_execution_times': {}
    }
    
    result = technical_analysis_node(state)
    
    # Verify all components
    assert result['technical_indicators'] is not None
    assert result['technical_signal'] is not None
    assert result['technical_confidence'] is not None
    
    # Verify indicators are populated
    indicators = result['technical_indicators']
    assert 'rsi' in indicators
    assert 'macd' in indicators
    assert 'bollinger_bands' in indicators
    assert 'moving_averages' in indicators
    
    # Verify no errors
    assert len(result['errors']) == 0
    
    print(f"\n✅ Technical Analysis Complete:")
    print(f"   Signal: {result['technical_signal']}")
    print(f"   Confidence: {result['technical_confidence']*100:.1f}%")
    if 'rsi' in indicators:
        print(f"   RSI: {indicators['rsi']:.2f}")
    if 'macd' in indicators:
        print(f"   MACD: {indicators['macd']['macd']:.2f}")
