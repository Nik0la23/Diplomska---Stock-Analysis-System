"""
Node 4: Technical Analysis

Calculates technical indicators from price data and generates trading signals.

Indicators calculated:
- RSI (14-day) - Relative Strength Index
- MACD (12, 26, 9) - Moving Average Convergence Divergence
- Bollinger Bands (20-day, 2σ)
- SMA (20, 50) - Simple Moving Averages
- Volume Analysis

Runs in PARALLEL with: Nodes 5, 6, 7
Runs AFTER: Node 9A (content analysis)
Runs BEFORE: Node 8 (verification)
"""

import pandas as pd
import pandas_ta as ta
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTION 1: Calculate RSI
# ============================================================================

def calculate_rsi(price_data: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures overbought/oversold conditions:
    - RSI > 70: Overbought (potential SELL signal)
    - RSI < 30: Oversold (potential BUY signal)
    - RSI 30-70: Neutral zone
    
    Args:
        price_data: DataFrame with 'close' prices
        period: RSI period (default: 14 days)
        
    Returns:
        Current RSI value (0-100) or None if calculation fails
        
    Example:
        >>> rsi = calculate_rsi(price_df, period=14)
        >>> if rsi < 30:
        ...     print("Oversold - potential BUY")
    """
    try:
        if len(price_data) < period + 1:
            logger.warning(f"Insufficient data for RSI calculation (need {period + 1} days)")
            return None
        
        # Use pandas_ta for RSI calculation
        rsi_series = ta.rsi(price_data['close'], length=period)
        
        if rsi_series is None or rsi_series.empty:
            logger.warning("RSI calculation returned empty result")
            return None
        
        # Get the most recent RSI value
        current_rsi = rsi_series.iloc[-1]
        
        # Validate RSI is in valid range
        if pd.isna(current_rsi) or current_rsi < 0 or current_rsi > 100:
            logger.warning(f"Invalid RSI value: {current_rsi}")
            return None
        
        return float(current_rsi)
        
    except Exception as e:
        logger.error(f"RSI calculation failed: {str(e)}")
        return None


# ============================================================================
# HELPER FUNCTION 2: Calculate MACD
# ============================================================================

def calculate_macd(price_data: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD shows momentum and trend:
    - MACD > Signal: Bullish (potential BUY)
    - MACD < Signal: Bearish (potential SELL)
    - Crossovers indicate trend changes
    
    Args:
        price_data: DataFrame with 'close' prices
        
    Returns:
        {
            'macd': float,        # MACD line value
            'signal': float,      # Signal line value
            'histogram': float,   # MACD - Signal
            'crossover': str      # 'bullish', 'bearish', or 'none'
        }
        None if calculation fails
        
    Example:
        >>> macd_data = calculate_macd(price_df)
        >>> if macd_data['macd'] > macd_data['signal']:
        ...     print("Bullish momentum")
    """
    try:
        if len(price_data) < 35:  # Need 26 + 9 days minimum
            logger.warning("Insufficient data for MACD calculation (need 35+ days)")
            return None
        
        # Calculate MACD using pandas_ta (fast=12, slow=26, signal=9)
        macd = ta.macd(price_data['close'], fast=12, slow=26, signal=9)
        
        if macd is None or macd.empty:
            logger.warning("MACD calculation returned empty result")
            return None
        
        # Extract components
        macd_line = macd.iloc[-1, 0]  # MACD_12_26_9
        signal_line = macd.iloc[-1, 1]  # MACDs_12_26_9
        histogram = macd.iloc[-1, 2]  # MACDh_12_26_9
        
        # Detect crossover
        if len(macd) >= 2:
            prev_histogram = macd.iloc[-2, 2]
            if prev_histogram < 0 and histogram > 0:
                crossover = 'bullish'
            elif prev_histogram > 0 and histogram < 0:
                crossover = 'bearish'
            else:
                crossover = 'none'
        else:
            crossover = 'none'
        
        return {
            'macd': float(macd_line),
            'signal': float(signal_line),
            'histogram': float(histogram),
            'crossover': crossover
        }
        
    except Exception as e:
        logger.error(f"MACD calculation failed: {str(e)}")
        return None


# ============================================================================
# HELPER FUNCTION 3: Calculate Bollinger Bands
# ============================================================================

def calculate_bollinger_bands(
    price_data: pd.DataFrame, 
    period: int = 20, 
    std_dev: float = 2.0
) -> Optional[Dict[str, float]]:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands measure volatility:
    - Price near upper band: Overbought
    - Price near lower band: Oversold
    - Band width: Volatility indicator
    
    Args:
        price_data: DataFrame with 'close' prices
        period: Moving average period (default: 20)
        std_dev: Standard deviations (default: 2.0)
        
    Returns:
        {
            'upper_band': float,
            'middle_band': float,  # SMA
            'lower_band': float,
            'bandwidth': float,    # (upper - lower) / middle
            'position': str        # 'upper', 'middle', 'lower'
        }
        None if calculation fails
        
    Example:
        >>> bb = calculate_bollinger_bands(price_df)
        >>> if bb['position'] == 'lower':
        ...     print("Price near lower band - potential BUY")
    """
    try:
        if len(price_data) < period:
            logger.warning(f"Insufficient data for Bollinger Bands (need {period}+ days)")
            return None
        
        # Calculate Bollinger Bands using pandas_ta
        bbands = ta.bbands(price_data['close'], length=period, std=std_dev)
        
        if bbands is None or bbands.empty:
            logger.warning("Bollinger Bands calculation returned empty result")
            return None
        
        # Extract bands (pandas_ta returns: BBL, BBM, BBU, BBB, BBP)
        lower_band = bbands.iloc[-1, 0]  # BBL_20_2.0
        middle_band = bbands.iloc[-1, 1]  # BBM_20_2.0
        upper_band = bbands.iloc[-1, 2]  # BBU_20_2.0
        
        current_price = price_data['close'].iloc[-1]
        
        # Calculate bandwidth (volatility measure)
        bandwidth = ((upper_band - lower_band) / middle_band) * 100
        
        # Determine price position relative to bands
        if current_price >= upper_band * 0.95:  # Within 5% of upper band
            position = 'upper'
        elif current_price <= lower_band * 1.05:  # Within 5% of lower band
            position = 'lower'
        else:
            position = 'middle'
        
        return {
            'upper_band': float(upper_band),
            'middle_band': float(middle_band),
            'lower_band': float(lower_band),
            'bandwidth': float(bandwidth),
            'position': position,
            'current_price': float(current_price)
        }
        
    except Exception as e:
        logger.error(f"Bollinger Bands calculation failed: {str(e)}")
        return None


# ============================================================================
# HELPER FUNCTION 4: Calculate Moving Averages
# ============================================================================

def calculate_moving_averages(price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Calculate Simple Moving Averages (SMA).
    
    Moving averages show trend:
    - Price > SMA20 > SMA50: Strong uptrend (golden cross)
    - Price < SMA20 < SMA50: Strong downtrend (death cross)
    
    Args:
        price_data: DataFrame with 'close' prices
        
    Returns:
        {
            'sma_20': float,
            'sma_50': float,
            'current_price': float,
            'trend': str,  # 'strong_uptrend', 'uptrend', 'downtrend', 'strong_downtrend', 'neutral'
            'golden_cross': bool,  # SMA20 > SMA50
            'death_cross': bool    # SMA20 < SMA50
        }
        None if calculation fails
    """
    try:
        if len(price_data) < 50:
            logger.warning("Insufficient data for moving averages (need 50+ days)")
            return None
        
        # Calculate SMAs
        sma_20 = ta.sma(price_data['close'], length=20)
        sma_50 = ta.sma(price_data['close'], length=50)
        
        if sma_20 is None or sma_50 is None:
            logger.warning("SMA calculation returned empty result")
            return None
        
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        current_price = price_data['close'].iloc[-1]
        
        # Determine trend
        if current_price > current_sma_20 > current_sma_50:
            trend = 'strong_uptrend'
        elif current_price > current_sma_20:
            trend = 'uptrend'
        elif current_price < current_sma_20 < current_sma_50:
            trend = 'strong_downtrend'
        elif current_price < current_sma_20:
            trend = 'downtrend'
        else:
            trend = 'neutral'
        
        # Detect golden/death cross (convert numpy bool to Python bool)
        golden_cross = bool(current_sma_20 > current_sma_50)
        death_cross = bool(current_sma_20 < current_sma_50)
        
        return {
            'sma_20': float(current_sma_20),
            'sma_50': float(current_sma_50),
            'current_price': float(current_price),
            'trend': trend,
            'golden_cross': golden_cross,
            'death_cross': death_cross
        }
        
    except Exception as e:
        logger.error(f"Moving averages calculation failed: {str(e)}")
        return None


# ============================================================================
# HELPER FUNCTION 5: Analyze Volume
# ============================================================================

def analyze_volume(price_data: pd.DataFrame, period: int = 20) -> Optional[Dict[str, Any]]:
    """
    Analyze trading volume patterns.
    
    Volume confirms trends:
    - High volume + price increase: Strong buying
    - High volume + price decrease: Strong selling
    - Low volume: Weak conviction
    
    Args:
        price_data: DataFrame with 'volume' column
        period: Period for average volume (default: 20)
        
    Returns:
        {
            'current_volume': float,
            'average_volume': float,
            'volume_ratio': float,  # current / average
            'volume_signal': str    # 'high', 'normal', 'low'
        }
        None if calculation fails
    """
    try:
        if 'volume' not in price_data.columns:
            logger.warning("No volume data available")
            return None
        
        if len(price_data) < period:
            logger.warning(f"Insufficient data for volume analysis (need {period}+ days)")
            return None
        
        current_volume = price_data['volume'].iloc[-1]
        average_volume = price_data['volume'].tail(period).mean()
        
        volume_ratio = current_volume / average_volume if average_volume > 0 else 1.0
        
        # Classify volume
        if volume_ratio > 1.5:
            volume_signal = 'high'
        elif volume_ratio < 0.5:
            volume_signal = 'low'
        else:
            volume_signal = 'normal'
        
        return {
            'current_volume': float(current_volume),
            'average_volume': float(average_volume),
            'volume_ratio': float(volume_ratio),
            'volume_signal': volume_signal
        }
        
    except Exception as e:
        logger.error(f"Volume analysis failed: {str(e)}")
        return None


# ============================================================================
# HELPER FUNCTION 6: Generate Technical Signal
# ============================================================================

def generate_technical_signal(
    rsi: Optional[float],
    macd: Optional[Dict],
    bollinger: Optional[Dict],
    moving_avg: Optional[Dict],
    volume: Optional[Dict]
) -> Tuple[str, float]:
    """
    Generate final technical signal by combining all indicators.
    
    Scoring system:
    - RSI < 30: +50 points (oversold, bullish)
    - RSI > 70: -50 points (overbought, bearish)
    - MACD > Signal: +30 points (bullish momentum)
    - MACD < Signal: -30 points (bearish momentum)
    - Golden Cross: +20 points (strong uptrend)
    - Death Cross: -20 points (strong downtrend)
    - Price near lower BB: +15 points (oversold)
    - Price near upper BB: -15 points (overbought)
    - High volume with uptrend: +10 points
    - High volume with downtrend: -10 points
    
    Args:
        rsi: RSI value
        macd: MACD data dict
        bollinger: Bollinger Bands data dict
        moving_avg: Moving averages data dict
        volume: Volume analysis data dict
        
    Returns:
        (signal, confidence) tuple
        - signal: 'BUY', 'SELL', or 'HOLD'
        - confidence: 0.0 to 1.0
        
    Example:
        >>> signal, confidence = generate_technical_signal(rsi, macd, bb, ma, vol)
        >>> print(f"{signal} with {confidence*100:.1f}% confidence")
    """
    score = 0.0
    max_possible_score = 0.0
    
    # RSI scoring
    if rsi is not None:
        max_possible_score += 50
        if rsi < 30:
            score += 50  # Oversold - bullish
        elif rsi > 70:
            score -= 50  # Overbought - bearish
        elif rsi < 40:
            score += 25  # Moderately oversold
        elif rsi > 60:
            score -= 25  # Moderately overbought
    
    # MACD scoring
    if macd is not None:
        max_possible_score += 30
        if macd['macd'] > macd['signal']:
            score += 30  # Bullish momentum
        else:
            score -= 30  # Bearish momentum
        
        # Bonus for crossover
        if macd['crossover'] == 'bullish':
            score += 10
        elif macd['crossover'] == 'bearish':
            score -= 10
    
    # Moving averages scoring
    if moving_avg is not None:
        max_possible_score += 20
        if moving_avg['golden_cross']:
            score += 20  # Bullish trend
        elif moving_avg['death_cross']:
            score -= 20  # Bearish trend
        
        # Bonus for strong trend
        if moving_avg['trend'] == 'strong_uptrend':
            score += 10
        elif moving_avg['trend'] == 'strong_downtrend':
            score -= 10
    
    # Bollinger Bands scoring
    if bollinger is not None:
        max_possible_score += 15
        if bollinger['position'] == 'lower':
            score += 15  # Near lower band - oversold
        elif bollinger['position'] == 'upper':
            score -= 15  # Near upper band - overbought
    
    # Volume scoring
    if volume is not None and moving_avg is not None:
        max_possible_score += 10
        if volume['volume_signal'] == 'high':
            if moving_avg['trend'] in ['uptrend', 'strong_uptrend']:
                score += 10  # Strong buying
            elif moving_avg['trend'] in ['downtrend', 'strong_downtrend']:
                score -= 10  # Strong selling
    
    # Normalize score to 0-100 scale
    if max_possible_score > 0:
        normalized_score = ((score + max_possible_score) / (2 * max_possible_score)) * 100
    else:
        normalized_score = 50  # Neutral if no indicators
    
    # Determine signal
    if normalized_score >= 65:
        signal = 'BUY'
        confidence = min(normalized_score / 100, 1.0)
    elif normalized_score <= 35:
        signal = 'SELL'
        confidence = min((100 - normalized_score) / 100, 1.0)
    else:
        signal = 'HOLD'
        confidence = 1.0 - abs(normalized_score - 50) / 50
    
    return signal, confidence


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def technical_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 4: Technical Analysis
    
    Execution flow:
    1. Get price data from state
    2. Calculate RSI (14-day)
    3. Calculate MACD (12, 26, 9)
    4. Calculate Bollinger Bands (20, 2σ)
    5. Calculate Moving Averages (20, 50)
    6. Analyze Volume
    7. Generate technical signal
    8. Update state
    
    Runs in PARALLEL with: Nodes 5, 6, 7
    Runs AFTER: Node 9A
    Runs BEFORE: Node 8
    
    Args:
        state: LangGraph state containing 'raw_price_data'
        
    Returns:
        Updated state with technical analysis results
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    try:
        logger.info(f"Node 4: Starting technical analysis for {ticker}")
        
        # ====================================================================
        # STEP 1: Get Price Data
        # ====================================================================
        price_data = state.get('raw_price_data')
        
        if price_data is None or price_data.empty:
            raise ValueError("No price data available for technical analysis")
        
        if len(price_data) < 50:
            raise ValueError(f"Insufficient price data (need 50+ days, have {len(price_data)})")
        
        logger.info(f"Analyzing {len(price_data)} days of price data")
        
        # ====================================================================
        # STEP 2: Calculate All Indicators
        # ====================================================================
        logger.info("Calculating RSI...")
        rsi = calculate_rsi(price_data, period=14)
        
        logger.info("Calculating MACD...")
        macd = calculate_macd(price_data)
        
        logger.info("Calculating Bollinger Bands...")
        bollinger = calculate_bollinger_bands(price_data, period=20, std_dev=2.0)
        
        logger.info("Calculating Moving Averages...")
        moving_avg = calculate_moving_averages(price_data)
        
        logger.info("Analyzing Volume...")
        volume = analyze_volume(price_data, period=20)
        
        # ====================================================================
        # STEP 3: Generate Signal
        # ====================================================================
        logger.info("Generating technical signal...")
        signal, confidence = generate_technical_signal(rsi, macd, bollinger, moving_avg, volume)
        
        # ====================================================================
        # STEP 4: Build Results
        # ====================================================================
        technical_indicators = {}
        
        if rsi is not None:
            technical_indicators['rsi'] = rsi
        
        if macd is not None:
            technical_indicators['macd'] = macd
        
        if bollinger is not None:
            technical_indicators['bollinger_bands'] = bollinger
        
        if moving_avg is not None:
            technical_indicators['moving_averages'] = moving_avg
        
        if volume is not None:
            technical_indicators['volume'] = volume
        
        # ====================================================================
        # STEP 5: Log Results
        # ====================================================================
        logger.info(f"Technical Analysis Results:")
        logger.info(f"  Signal: {signal}")
        logger.info(f"  Confidence: {confidence*100:.1f}%")
        if rsi is not None:
            logger.info(f"  RSI: {rsi:.2f}")
        if macd is not None:
            logger.info(f"  MACD: {macd['macd']:.2f} vs Signal: {macd['signal']:.2f}")
        if moving_avg is not None:
            logger.info(f"  Trend: {moving_avg['trend']}")
        
        # ====================================================================
        # STEP 6: Update State
        # ====================================================================
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Node 4: Technical analysis completed in {elapsed:.2f}s")
        
        # Return only the fields this node updates (for parallel execution)
        return {
            'technical_indicators': technical_indicators,
            'technical_signal': signal,
            'technical_confidence': confidence,
            'node_execution_times': {'node_4': elapsed}
        }
        
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Node 4: Technical analysis failed for {ticker}: {str(e)}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Return only the fields this node updates (for parallel execution)
        return {
            'errors': [f"Node 4: Technical analysis failed - {str(e)}"],
            'technical_indicators': None,
            'technical_signal': None,
            'technical_confidence': None,
            'node_execution_times': {'node_4': elapsed}
        }
