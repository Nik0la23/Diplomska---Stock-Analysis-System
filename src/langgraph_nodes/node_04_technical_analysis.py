"""
Node 4: Technical Analysis

Calculates technical indicators and a continuous normalised score [0, 100].
Does NOT produce a discrete BUY/SELL/HOLD signal — that decision belongs to
Node 12 after combining all four analytical streams.

Indicators calculated:
- RSI (14-day) - Relative Strength Index
- MACD (12, 26, 9) - Moving Average Convergence Divergence
- Bollinger Bands (20-day, 2σ)
- SMA (20, 50) - Simple Moving Averages
- Volume Analysis
- ADX (14-day) - Average Directional Index (trend strength)
- Persistent Pressure detector (multi-day directional bias)

Output (all inside technical_indicators dict):
- normalized_score: float [0, 100] — core metric consumed by Node 12
- hold_low / hold_high: dynamic neutral band thresholds (ADX-adjusted)
- market_regime: 'trending_up' | 'trending_down' | 'choppy'
- persistent_pressure: dict with directional pressure sub-signals
- technical_summary: human-readable narrative for Node 14

Runs in PARALLEL with: Nodes 5, 6, 7
Runs AFTER: Node 9A (content analysis)
Runs BEFORE: Node 8 (verification)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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
# HELPER FUNCTION 6 (NEW): Calculate ADX
# ============================================================================

def calculate_adx(price_data: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Calculate the Average Directional Index (ADX).

    ADX measures trend strength regardless of direction:
    - ADX > 25: strong trend (directional signals are reliable)
    - ADX 18–25: moderate trend
    - ADX < 18: weak / choppy market (signals less reliable)

    Args:
        price_data: DataFrame with 'high', 'low', 'close' columns.
        period:     ADX period (default: 14).

    Returns:
        Current ADX value (float ≥ 0) or None if calculation fails.

    Example:
        >>> adx = calculate_adx(price_df)
        >>> if adx and adx > 25:
        ...     print("Strong trend — directional signals reliable")
    """
    try:
        required = period * 2 + 1
        if len(price_data) < required:
            logger.warning(f"Insufficient data for ADX (need {required} days)")
            return None

        adx_result = ta.adx(
            price_data['high'],
            price_data['low'],
            price_data['close'],
            length=period,
        )

        if adx_result is None or adx_result.empty:
            logger.warning("ADX calculation returned empty result")
            return None

        adx_col = [c for c in adx_result.columns if c.startswith('ADX_')]
        if not adx_col:
            logger.warning("ADX column not found in pandas-ta output")
            return None

        adx_value = adx_result[adx_col[0]].iloc[-1]

        if pd.isna(adx_value) or adx_value < 0:
            logger.warning(f"Invalid ADX value: {adx_value}")
            return None

        return float(adx_value)

    except Exception as e:
        logger.error(f"ADX calculation failed: {str(e)}")
        return None


# ============================================================================
# HELPER FUNCTION 7 (NEW): Detect Persistent Pressure
# ============================================================================

def detect_persistent_pressure(
    price_data: pd.DataFrame,
    lookback: int = 10,
) -> Dict[str, Any]:
    """
    Detect slow, consistent directional pressure over recent N trading days.

    Identifies slow grinding trends that individual indicators might classify
    as "neutral" (e.g. RSI=45, MACD slightly negative) but that represent a
    real, tradeable directional bias when viewed over multiple sessions.

    Four sub-signals are evaluated:
    1. Bearish-day ratio — what fraction of the last N days closed below open.
    2. SMA20 slope — is the 20-day average itself trending up or down?
    3. Consecutive days below SMA20 — persistent position pressure.
    4. MACD histogram slope — is momentum accelerating bearish/bullish?

    Args:
        price_data: DataFrame with 'open', 'close' columns and at least
                    ``lookback + 20`` rows (for SMA20 computation).
        lookback:   Number of recent sessions to evaluate (default: 10).

    Returns:
        Dict with keys:
        {
            'pressure_direction':    str,    # 'bearish' | 'bullish' | 'neutral'
            'pressure_strength':     float,  # 0.0 to 1.0
            'consecutive_below_sma': int,    # sessions in a row below SMA20
            'macd_histogram_slope':  float,  # positive = improving, negative = worsening
            'bearish_day_ratio':     float,  # fraction of bearish sessions
            'sma20_slope':           float,  # % change of SMA20 over 5 days
        }

    Example:
        >>> p = detect_persistent_pressure(price_df, lookback=10)
        >>> if p['pressure_direction'] == 'bearish' and p['pressure_strength'] >= 0.75:
        ...     print(f"Grinding bear — {p['consecutive_below_sma']} days below SMA20")
    """
    try:
        closes = price_data['close']
        opens = price_data['open']

        lookback_closes = closes.tail(lookback)
        lookback_opens = opens.tail(lookback)

        # 1. Bearish-day ratio
        bearish_days = sum(
            1 for c, o in zip(lookback_closes, lookback_opens) if c < o
        )
        bearish_ratio = bearish_days / lookback if lookback > 0 else 0.5

        # 2. SMA20 slope (% change over last 5 sessions)
        sma20 = ta.sma(closes, length=20)
        sma_slope = 0.0
        if sma20 is not None and len(sma20) >= 6:
            sma_now = sma20.iloc[-1]
            sma_5d_ago = sma20.iloc[-6]
            if sma_5d_ago and sma_5d_ago != 0 and not pd.isna(sma_5d_ago):
                sma_slope = float((sma_now - sma_5d_ago) / abs(sma_5d_ago) * 100)

        # 3. Consecutive sessions below SMA20
        consecutive = 0
        if sma20 is not None and len(sma20) >= 1:
            for i in range(1, min(lookback + 1, len(price_data) + 1)):
                idx = -i
                try:
                    price_val = closes.iloc[idx]
                    sma_val = sma20.iloc[idx]
                    if pd.isna(sma_val):
                        break
                    if price_val < sma_val:
                        consecutive += 1
                    else:
                        break
                except IndexError:
                    break

        # 4. MACD histogram slope
        macd_hist_slope = 0.0
        macd_result = ta.macd(closes, fast=12, slow=26, signal=9)
        if macd_result is not None and len(macd_result) >= 6:
            hist_col = [c for c in macd_result.columns if 'h' in c.lower()]
            if hist_col:
                hist_now = macd_result[hist_col[0]].iloc[-1]
                hist_5d_ago = macd_result[hist_col[0]].iloc[-6]
                if not pd.isna(hist_now) and not pd.isna(hist_5d_ago):
                    macd_hist_slope = float(hist_now - hist_5d_ago)

        # Tally directional signals
        bearish_signals = 0
        bullish_signals = 0

        if bearish_ratio > 0.6:
            bearish_signals += 1
        elif bearish_ratio < 0.4:
            bullish_signals += 1

        if sma_slope < -0.3:
            bearish_signals += 1
        elif sma_slope > 0.3:
            bullish_signals += 1

        if consecutive >= 5:
            bearish_signals += 1
        elif consecutive == 0:
            bullish_signals += 1

        if macd_hist_slope < -0.05:
            bearish_signals += 1
        elif macd_hist_slope > 0.05:
            bullish_signals += 1

        if bearish_signals >= 3:
            direction = 'bearish'
            strength = min(bearish_signals / 4.0, 1.0)
        elif bullish_signals >= 3:
            direction = 'bullish'
            strength = min(bullish_signals / 4.0, 1.0)
        else:
            direction = 'neutral'
            strength = 0.0

        return {
            'pressure_direction':    direction,
            'pressure_strength':     float(strength),
            'consecutive_below_sma': int(consecutive),
            'macd_histogram_slope':  float(macd_hist_slope),
            'bearish_day_ratio':     float(bearish_ratio),
            'sma20_slope':           float(sma_slope),
        }

    except Exception as e:
        logger.error(f"Persistent pressure detection failed: {str(e)}")
        return {
            'pressure_direction':    'neutral',
            'pressure_strength':     0.0,
            'consecutive_below_sma': 0,
            'macd_histogram_slope':  0.0,
            'bearish_day_ratio':     0.5,
            'sma20_slope':           0.0,
        }


# ============================================================================
# HELPER FUNCTION 8 (NEW): Build Feature Matrix for IC Regression
# ============================================================================

def build_feature_matrix(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build an 8-column feature matrix from OHLCV price data for Ridge regression.

    Features (one per column):
    - rsi           : RSI-14
    - macd_hist     : MACD histogram (12, 26, 9)
    - bb_pct_b      : Bollinger %B = (close - lower) / (upper - lower), clipped [0, 1]
    - sma20_slope_5d: 5-day % change of SMA-20
    - volume_ratio  : current volume / 20-day average volume
    - adx           : ADX-14
    - mom_5d        : 5-day price return %
    - mom_20d       : 20-day price return %

    Args:
        price_data: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.

    Returns:
        DataFrame with 8 feature columns and same index as price_data.
        Rows with NaN are kept — caller is responsible for alignment/dropping.
    """
    close  = price_data['close']
    high   = price_data['high']
    low    = price_data['low']
    volume = price_data['volume'] if 'volume' in price_data.columns else pd.Series(np.nan, index=price_data.index)

    features: Dict[str, pd.Series] = {}

    # 1. RSI-14
    rsi_s = ta.rsi(close, length=14)
    features['rsi'] = rsi_s if rsi_s is not None else pd.Series(np.nan, index=price_data.index)

    # 2. MACD histogram
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        hist_col = [c for c in macd_df.columns if 'h' in c.lower()]
        features['macd_hist'] = macd_df[hist_col[0]] if hist_col else pd.Series(np.nan, index=price_data.index)
    else:
        features['macd_hist'] = pd.Series(np.nan, index=price_data.index)

    # 3. Bollinger %B
    bb_df = ta.bbands(close, length=20, std=2.0)
    if bb_df is not None and not bb_df.empty:
        lower_col = [c for c in bb_df.columns if c.startswith('BBL')]
        upper_col = [c for c in bb_df.columns if c.startswith('BBU')]
        if lower_col and upper_col:
            lower = bb_df[lower_col[0]]
            upper = bb_df[upper_col[0]]
            band_width = upper - lower
            pct_b = (close - lower) / band_width.replace(0, np.nan)
            features['bb_pct_b'] = pct_b.clip(0.0, 1.0)
        else:
            features['bb_pct_b'] = pd.Series(np.nan, index=price_data.index)
    else:
        features['bb_pct_b'] = pd.Series(np.nan, index=price_data.index)

    # 4. SMA-20 slope over 5 days (%)
    sma20 = ta.sma(close, length=20)
    if sma20 is not None:
        features['sma20_slope_5d'] = sma20.pct_change(5) * 100
    else:
        features['sma20_slope_5d'] = pd.Series(np.nan, index=price_data.index)

    # 5. Volume ratio (current / 20-day average)
    vol_mean = volume.rolling(20).mean()
    features['volume_ratio'] = (volume / vol_mean.replace(0, np.nan)).clip(0, 10)

    # 6. ADX-14
    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None and not adx_df.empty:
        adx_col = [c for c in adx_df.columns if c.startswith('ADX_')]
        features['adx'] = adx_df[adx_col[0]] if adx_col else pd.Series(np.nan, index=price_data.index)
    else:
        features['adx'] = pd.Series(np.nan, index=price_data.index)

    # 7. 5-day momentum (%)
    features['mom_5d'] = close.pct_change(5) * 100

    # 8. 20-day momentum (%)
    features['mom_20d'] = close.pct_change(20) * 100

    return pd.DataFrame(features, index=price_data.index)


# ============================================================================
# HELPER FUNCTION 9 (NEW): Fit IC/Ridge Regression
# ============================================================================

def fit_ic_regression(price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Fit a Ridge regression model and compute Information Coefficient (IC).

    The IC (Pearson correlation between predicted and actual forward returns)
    is the standard professional quant metric for factor quality.

    Steps:
    1. Build 8-feature matrix from price_data
    2. Build target: 7-day forward return % (shifted -7)
    3. Align features and target; drop NaN rows from training set
    4. Fit Ridge(alpha=1.0) on StandardScaler-normalized features
    5. Compute IC = pearsonr(in-sample predictions, actual returns)
    6. Predict today's expected return from most recent feature row

    Args:
        price_data: Full OHLCV DataFrame. Minimum 60 rows required after NaN drop.

    Returns:
        Dict with keys:
            ic_score             : float, Pearson IC in [-1, +1]
            predicted_return_pct : float, today's predicted 7-day return %
            return_std           : float, std dev of training returns
            n_training_samples   : int
        Returns None if insufficient data or any exception occurs.
    """
    MIN_TRAINING_ROWS = 60

    try:
        feature_df = build_feature_matrix(price_data)
        close = price_data['close']

        # Target: 7-day forward return % — shift(-7) so each row has its future return
        y_full = close.pct_change(7).shift(-7) * 100

        # Align on the same index
        combined = feature_df.copy()
        combined['__target__'] = y_full

        # Training rows: where ALL features AND target are valid
        train = combined.dropna()
        if len(train) < MIN_TRAINING_ROWS:
            logger.debug(
                f"fit_ic_regression: only {len(train)} clean training rows "
                f"(need {MIN_TRAINING_ROWS}) — returning None"
            )
            return None

        X_train = train[feature_df.columns].values
        y_train = train['__target__'].values

        # Today's features: last row of feature_df that has no NaN
        feature_only = feature_df.dropna()
        if feature_only.empty:
            return None
        X_today = feature_only.iloc[[-1]].values  # shape (1, 8)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_today_scaled = scaler.transform(X_today)

        # Fit
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # IC: in-sample correlation
        pred_train = model.predict(X_train_scaled)
        if np.std(pred_train) == 0 or np.std(y_train) == 0:
            ic = 0.0
        else:
            ic = float(pearsonr(pred_train, y_train)[0])
            if np.isnan(ic):
                ic = 0.0

        # Today's prediction
        predicted_return_today = float(model.predict(X_today_scaled)[0])
        return_std = float(np.std(y_train))

        return {
            'ic_score':             ic,
            'predicted_return_pct': predicted_return_today,
            'return_std':           return_std if return_std > 0 else 1.0,
            'n_training_samples':   len(y_train),
        }

    except Exception as e:
        logger.warning(f"fit_ic_regression failed: {e}")
        return None


# ============================================================================
# HELPER FUNCTION 10 (NEW): Get Dynamic Hold Band
# ============================================================================

def get_hold_band(adx: float, market_regime: str) -> Tuple[float, float]:
    """
    Return a dynamic (hold_low, hold_high) band for the normalised score.

    In strong trends (high ADX or clear regime), the neutral band is narrowed
    so that a consistent-but-mild directional bias still produces a signal.
    In choppy / range-bound markets the band is widened — genuinely uncertain.

    Thresholds:
    - Strong trend (ADX > 25 or trending regime): (45, 55)  — was static (35, 65)
    - Moderate trend (ADX 18–25):                  (42, 58)
    - Weak / choppy (ADX < 18):                    (38, 62)

    Args:
        adx:           Current ADX value.
        market_regime: One of 'trending_up', 'trending_down', 'choppy'.

    Returns:
        (hold_low, hold_high) tuple of floats.

    Example:
        >>> low, high = get_hold_band(adx=28.0, market_regime='trending_down')
        >>> print(f"Band: ({low}, {high})")  # Band: (45, 55)
    """
    if market_regime in ('trending_up', 'trending_down') or adx > 25:
        return (45.0, 55.0)
    if adx > 18:
        return (42.0, 58.0)
    return (38.0, 62.0)


# ============================================================================
# HELPER FUNCTION 9 (NEW): Derive Market Regime
# ============================================================================

def _derive_market_regime(moving_avg: Optional[Dict[str, Any]]) -> str:
    """
    Map the moving-averages trend string to a coarse market regime label.

    Translates Node 4's internal trend classification into the three-bucket
    regime used by ``get_hold_band`` and stored in ``technical_indicators``
    for downstream consumers (Node 12, Node 14).

    Mapping:
    - 'strong_uptrend' | 'uptrend'   → 'trending_up'
    - 'strong_downtrend' | 'downtrend' → 'trending_down'
    - 'neutral' or missing             → 'choppy'

    Args:
        moving_avg: Moving averages dict from ``calculate_moving_averages()``
                    or None when that calculation failed.

    Returns:
        One of 'trending_up', 'trending_down', 'choppy'.

    Example:
        >>> regime = _derive_market_regime({'trend': 'strong_downtrend', ...})
        >>> print(regime)  # 'trending_down'
    """
    if moving_avg is None:
        return 'choppy'
    trend = moving_avg.get('trend', 'neutral')
    if trend in ('strong_uptrend', 'uptrend'):
        return 'trending_up'
    if trend in ('strong_downtrend', 'downtrend'):
        return 'trending_down'
    return 'choppy'


# ============================================================================
# HELPER FUNCTION 10 (RENAMED): Calculate Technical Score
# ============================================================================

def calculate_technical_score(
    rsi: Optional[float],
    macd: Optional[Dict[str, Any]],
    bollinger: Optional[Dict[str, Any]],
    moving_avg: Optional[Dict[str, Any]],
    volume: Optional[Dict[str, Any]],
    price_data: Optional[pd.DataFrame] = None,
    adx: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute a continuous technical score on [0, 100] by combining all indicators.

    Replaces the old generate_technical_signal() — instead of returning a
    discrete BUY/SELL/HOLD string, this function returns a numeric score dict
    that Node 12 can use directly for weighted signal combination.

    Scoring system (raw points before normalisation):
    - RSI < 30:  +50  (oversold, bullish)
    - RSI > 70:  -50  (overbought, bearish)
    - RSI < 40:  +25  (moderately oversold)
    - RSI > 60:  -25  (moderately overbought)
    - MACD > Signal: +30  (bullish momentum)
    - MACD < Signal: -30  (bearish momentum)
    - MACD bullish crossover: +10 bonus
    - MACD bearish crossover: -10 bonus
    - Golden cross: +20 | Death cross: -20
    - strong_uptrend: +10 bonus | strong_downtrend: -10 bonus
    - Price near lower BB: +15 | upper BB: -15
    - High volume + uptrend: +10 | downtrend: -10

    After normalisation to [0, 100]:
    - Persistent pressure (if price_data provided) adjusts score by ±8 × strength
    - Dynamic hold band is computed from ADX + market_regime

    Args:
        rsi:        RSI value (0–100) or None.
        macd:       MACD result dict (keys: macd, signal, histogram, crossover).
        bollinger:  Bollinger Bands dict (key: position).
        moving_avg: Moving averages dict (keys: trend, golden_cross, death_cross).
        volume:     Volume analysis dict (keys: volume_signal).
        price_data: Raw OHLCV DataFrame — enables persistent pressure detection.
        adx:        ADX value — used to set dynamic hold band width.

    Returns:
        Dict with keys:
        {
            'normalized_score':    float,  # 0–100, pressure-adjusted
            'hold_low':            float,  # dynamic band lower bound
            'hold_high':           float,  # dynamic band upper bound
            'market_regime':       str,    # 'trending_up' | 'trending_down' | 'choppy'
            'pressure_applied':    bool,
            'pressure_adjustment': float,  # points added (+) or subtracted (-)
        }

    Example:
        >>> result = calculate_technical_score(rsi, macd, bb, ma, vol, price_data, adx)
        >>> print(f"score={result['normalized_score']:.1f}  band=({result['hold_low']}, {result['hold_high']})")
    """
    # -----------------------------------------------------------------------
    # PRIMARY PATH: Ridge regression when sufficient price data is available
    # -----------------------------------------------------------------------
    technical_alpha: float = 0.0
    predicted_return_pct: float = 0.0
    ic_score: float = 0.0
    n_training_samples: int = 0
    regression_valid: bool = False

    if price_data is not None and len(price_data) >= 60:
        reg_result = fit_ic_regression(price_data)
        if reg_result is not None:
            pred_ret = reg_result['predicted_return_pct']
            ret_std  = reg_result['return_std']
            # Z-score: ±3σ prediction maps to ±1.0 alpha
            raw_alpha = pred_ret / (3.0 * ret_std)
            technical_alpha      = float(np.clip(raw_alpha, -1.0, 1.0))
            predicted_return_pct = float(pred_ret)
            ic_score             = reg_result['ic_score']
            n_training_samples   = reg_result['n_training_samples']
            regression_valid     = True

    # -----------------------------------------------------------------------
    # FALLBACK PATH: hand-crafted heuristic when regression is unavailable
    # -----------------------------------------------------------------------
    if not regression_valid:
        score = 0.0
        max_possible_score = 0.0

        # RSI scoring
        if rsi is not None:
            max_possible_score += 50
            if rsi < 30:
                score += 50
            elif rsi > 70:
                score -= 50
            elif rsi < 40:
                score += 25
            elif rsi > 60:
                score -= 25

        # MACD scoring
        if macd is not None:
            max_possible_score += 30
            if macd['macd'] > macd['signal']:
                score += 30
            else:
                score -= 30
            if macd['crossover'] == 'bullish':
                score += 10
            elif macd['crossover'] == 'bearish':
                score -= 10

        # Moving averages scoring (cross confirmed by price position)
        if moving_avg is not None:
            max_possible_score += 20
            trend = moving_avg['trend']
            if moving_avg['golden_cross'] and trend in ('uptrend', 'strong_uptrend'):
                score += 20
            elif moving_avg['death_cross'] and trend in ('downtrend', 'strong_downtrend'):
                score -= 20
            if trend == 'strong_uptrend':
                score += 10
            elif trend == 'strong_downtrend':
                score -= 10

        # Bollinger Bands scoring (lower only in non-downtrend)
        if bollinger is not None:
            max_possible_score += 15
            _ma_trend = moving_avg['trend'] if moving_avg is not None else 'neutral'
            if bollinger['position'] == 'lower' and _ma_trend not in ('downtrend', 'strong_downtrend'):
                score += 15
            elif bollinger['position'] == 'upper':
                score -= 15

        # Volume scoring
        if volume is not None and moving_avg is not None:
            max_possible_score += 10
            if volume['volume_signal'] == 'high':
                if moving_avg['trend'] in ['uptrend', 'strong_uptrend']:
                    score += 10
                elif moving_avg['trend'] in ['downtrend', 'strong_downtrend']:
                    score -= 10

        # Normalise raw score to [0, 100]
        if max_possible_score > 0:
            heuristic_score = ((score + max_possible_score) / (2 * max_possible_score)) * 100
        else:
            heuristic_score = 50.0

        # Persistent pressure adjustment (heuristic path only)
        if price_data is not None and len(price_data) >= 30:
            try:
                pressure = detect_persistent_pressure(price_data, lookback=10)
                if pressure['pressure_direction'] == 'bearish' and pressure['pressure_strength'] >= 0.5:
                    heuristic_score += -15.0 * pressure['pressure_strength']
                elif pressure['pressure_direction'] == 'bullish' and pressure['pressure_strength'] >= 0.5:
                    heuristic_score += 15.0 * pressure['pressure_strength']
            except Exception as e:
                logger.warning(f"Persistent pressure adjustment skipped: {e}")

        heuristic_score = float(np.clip(heuristic_score, 0.0, 100.0))
        # Map [0, 100] → [-1, +1] for consistency with regression path
        technical_alpha = float(np.clip((heuristic_score - 50.0) / 50.0, -1.0, 1.0))

    # -----------------------------------------------------------------------
    # SHARED: market regime, hold band, normalized_score (legacy)
    # -----------------------------------------------------------------------
    market_regime = _derive_market_regime(moving_avg)
    hold_low, hold_high = get_hold_band(adx if adx is not None else 15.0, market_regime)

    # Legacy [0, 100] score derived from alpha (kept for Node 10 backward compat)
    normalized_score = float(np.clip((technical_alpha + 1.0) * 50.0, 0.0, 100.0))

    return {
        'technical_alpha':      technical_alpha,       # PRIMARY: z-scored alpha [-1, +1]
        'predicted_return_pct': predicted_return_pct,
        'ic_score':             ic_score,
        'regression_valid':     regression_valid,
        'n_training_samples':   n_training_samples,
        'normalized_score':     normalized_score,      # LEGACY [0, 100] for Node 10 compat
        'market_regime':        market_regime,
        'hold_low':             float(hold_low),
        'hold_high':            float(hold_high),
    }


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def technical_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 4: Technical Analysis

    Calculates technical indicators and a continuous normalised score on
    [0, 100]. Does NOT produce a discrete BUY/SELL/HOLD signal — Node 12
    converts the numeric score to a directional stream contribution after
    weighting it against sentiment, market context, and Monte Carlo.

    Execution flow:
    1. Get price data from state
    2. Calculate RSI, MACD, Bollinger Bands, Moving Averages, Volume, ADX
    3. Compute normalised score via calculate_technical_score()
       (includes persistent-pressure adjustment and dynamic hold band)
    4. Detect persistent multi-day pressure pattern
    5. Build technical_summary narrative
    6. Return enriched technical_indicators dict

    Runs in PARALLEL with: Nodes 5, 6, 7
    Runs AFTER:  Node 9A (content analysis)
    Runs BEFORE: Node 8 (news verification)

    Args:
        state: LangGraph state containing 'raw_price_data'.

    Returns:
        Partial state update with 'technical_indicators' and
        'node_execution_times'. Does NOT write 'technical_signal' or
        'technical_confidence' — those state fields remain None.
    """
    start_time = datetime.now()
    ticker: str = state.get('ticker', 'UNKNOWN')

    try:
        logger.info(f"Node 4: Starting technical analysis for {ticker}")

        # ====================================================================
        # STEP 1: Get Price Data
        # ====================================================================
        price_data: Optional[pd.DataFrame] = state.get('raw_price_data')

        if price_data is None or price_data.empty:
            raise ValueError("No price data available for technical analysis")

        if len(price_data) < 50:
            raise ValueError(
                f"Insufficient price data (need 50+ days, have {len(price_data)})"
            )

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

        logger.info("Calculating ADX...")
        adx = calculate_adx(price_data, period=14)

        # ====================================================================
        # STEP 3: Compute Continuous Technical Score
        # ====================================================================
        logger.info("Computing technical score...")
        score_result = calculate_technical_score(
            rsi=rsi,
            macd=macd,
            bollinger=bollinger,
            moving_avg=moving_avg,
            volume=volume,
            price_data=price_data,
            adx=adx,
        )
        normalized_score  = score_result['normalized_score']
        hold_low          = score_result['hold_low']
        hold_high         = score_result['hold_high']
        market_regime     = score_result['market_regime']
        technical_alpha   = score_result['technical_alpha']
        ic_score          = score_result['ic_score']
        regression_valid  = score_result['regression_valid']
        n_training_samples = score_result['n_training_samples']

        # ====================================================================
        # STEP 4: Persistent Pressure (standalone — also stored in indicators)
        # ====================================================================
        logger.info("Detecting persistent pressure...")
        persistent_pressure = detect_persistent_pressure(price_data, lookback=10)

        # ====================================================================
        # STEP 5: Build Technical Summary Narrative
        # ====================================================================
        rsi_str = f"RSI {rsi:.1f}" if rsi is not None else "RSI N/A"
        macd_str = (
            f"MACD {'bullish' if macd['macd'] > macd['signal'] else 'bearish'} "
            f"(hist {macd['histogram']:+.3f})"
            if macd is not None else "MACD N/A"
        )
        adx_str = f"ADX {adx:.1f}" if adx is not None else "ADX N/A"
        trend_str = moving_avg['trend'].replace('_', ' ') if moving_avg else "trend N/A"
        pressure_dir = persistent_pressure['pressure_direction']
        pressure_str_count = int(round(persistent_pressure['pressure_strength'] * 4))
        consec = persistent_pressure['consecutive_below_sma']

        if consec > 0:
            sma_context = f"price below SMA20 for {consec} consecutive day{'s' if consec != 1 else ''}"
        else:
            sma_context = "price above SMA20"

        technical_summary = (
            f"{ticker} — {trend_str.title()} | {rsi_str}, {macd_str}, "
            f"{adx_str} ({market_regime.replace('_', ' ')}). "
            f"Technical alpha: {technical_alpha:+.3f} "
            f"({'regression' if regression_valid else 'heuristic'}, "
            f"IC={ic_score:.3f}, n={n_training_samples}). "
            f"Persistent pressure: {pressure_dir} ({pressure_str_count}/4 signals). "
            f"{sma_context.capitalize()}."
        )

        # ====================================================================
        # STEP 6: Build technical_indicators dict
        # ====================================================================
        technical_indicators: Dict[str, Any] = {}

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

        if adx is not None:
            technical_indicators['adx'] = adx

        # Primary regression output (Node 12 prefers these)
        technical_indicators['technical_alpha']      = technical_alpha
        technical_indicators['predicted_return_pct'] = score_result['predicted_return_pct']
        technical_indicators['ic_score']             = ic_score
        technical_indicators['regression_valid']     = regression_valid
        technical_indicators['n_training_samples']   = n_training_samples

        # Legacy scoring parameters kept for Node 10 backward compat
        technical_indicators['normalized_score'] = normalized_score
        technical_indicators['hold_low']         = hold_low
        technical_indicators['hold_high']        = hold_high
        technical_indicators['market_regime']    = market_regime

        # Persistent pressure dict — used by Node 12 similarity + Node 14 narrative
        technical_indicators['persistent_pressure'] = persistent_pressure

        # Human-readable summary for Node 14 LLM prompt
        technical_indicators['technical_summary'] = technical_summary

        # ====================================================================
        # STEP 7: Log Results
        # ====================================================================
        logger.info(
            f"Node 4 results — alpha={technical_alpha:+.3f} "
            f"ic={ic_score:.3f} "
            f"regression={regression_valid} "
            f"regime={market_regime}"
        )
        if rsi is not None:
            logger.info(f"  RSI: {rsi:.2f}")
        if macd is not None:
            logger.info(f"  MACD: {macd['macd']:.4f} vs Signal: {macd['signal']:.4f}")
        if moving_avg is not None:
            logger.info(f"  Trend: {moving_avg['trend']}")
        if adx is not None:
            logger.info(f"  ADX: {adx:.2f}")
        logger.info(
            f"  Pressure: {pressure_dir} "
            f"(strength={persistent_pressure['pressure_strength']:.2f}, "
            f"consecutive_below_sma={consec})"
        )
        logger.info(f"  normalized_score (legacy): {normalized_score:.1f}")

        # ====================================================================
        # STEP 8: Return State Update
        # ====================================================================
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Node 4: Technical analysis completed in {elapsed:.2f}s")

        # technical_signal and technical_confidence are intentionally NOT set —
        # Node 12 derives the technical stream contribution from normalized_score.
        return {
            'technical_indicators': technical_indicators,
            'node_execution_times': {'node_4': elapsed},
        }

    except Exception as e:
        logger.error(f"Node 4: Technical analysis failed for {ticker}: {str(e)}")
        elapsed = (datetime.now() - start_time).total_seconds()
        return {
            'errors': [f"Node 4: Technical analysis failed — {str(e)}"],
            'technical_indicators': None,
            'node_execution_times': {'node_4': elapsed},
        }
