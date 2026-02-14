"""
Node 6: Market Context Analysis

Analyzes market-wide conditions to provide context for individual stock signals.

Components analyzed:
1. Sector Performance - How is the stock's sector performing?
2. Market Trend - Overall market direction (bullish/bearish/neutral)
3. Related Companies - Competitor performance
4. Correlation - Stock-market correlation strength

Prevents buying a stock just because it looks good technically,
when the entire sector or market is collapsing.

Runs in PARALLEL with: Nodes 4, 5, 7
Runs AFTER: Node 9A (content analysis)
Runs BEFORE: Node 8 (verification)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# SECTOR AND MARKET MAPPINGS
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
    Get the sector and industry for a stock using yfinance.
    
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
        >>> sector, industry = get_stock_sector('NVDA')
        >>> print(sector)
        'Technology'
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        logger.info(f"{ticker} sector: {sector}, industry: {industry}")
        return (sector, industry)
        
    except Exception as e:
        logger.error(f"Failed to get sector for {ticker}: {str(e)}")
        return ('Unknown', 'Unknown')


# ============================================================================
# HELPER FUNCTION 2: Get Sector Performance
# ============================================================================

def get_sector_performance(sector: str, days: int = 1) -> Dict[str, Any]:
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
        >>> perf = get_sector_performance('Technology', days=1)
        >>> print(perf['performance'])
        2.3
    """
    try:
        # Get sector ETF ticker
        etf_ticker = SECTOR_ETFS.get(sector)
        
        if not etf_ticker:
            logger.warning(f"No ETF mapping for sector: {sector}")
            return {
                'sector': sector,
                'etf_ticker': None,
                'performance': 0.0,
                'trend': 'FLAT',
                'price_current': 0.0,
                'price_start': 0.0
            }
        
        # Fetch ETF price data
        etf = yf.Ticker(etf_ticker)
        history = etf.history(period=f'{days+1}d')
        
        if history.empty or len(history) < 2:
            logger.warning(f"Insufficient data for {etf_ticker}")
            return {
                'sector': sector,
                'etf_ticker': etf_ticker,
                'performance': 0.0,
                'trend': 'FLAT',
                'price_current': 0.0,
                'price_start': 0.0
            }
        
        # Calculate performance
        price_start = float(history['Close'].iloc[0])
        price_current = float(history['Close'].iloc[-1])
        performance = ((price_current - price_start) / price_start) * 100
        
        # Determine trend
        if performance > 0.5:
            trend = 'UP'
        elif performance < -0.5:
            trend = 'DOWN'
        else:
            trend = 'FLAT'
        
        logger.info(f"Sector {sector} ({etf_ticker}): {performance:.2f}% ({trend})")
        
        return {
            'sector': sector,
            'etf_ticker': etf_ticker,
            'performance': performance,
            'trend': trend,
            'price_current': price_current,
            'price_start': price_start
        }
        
    except Exception as e:
        logger.error(f"Failed to get sector performance: {str(e)}")
        return {
            'sector': sector,
            'etf_ticker': None,
            'performance': 0.0,
            'trend': 'FLAT',
            'price_current': 0.0,
            'price_start': 0.0,
            'error': str(e)
        }


# ============================================================================
# HELPER FUNCTION 3: Get Market Trend
# ============================================================================

def get_market_trend(days: int = 5) -> Dict[str, Any]:
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
        >>> trend = get_market_trend(days=5)
        >>> print(trend['market_trend'])
        'BULLISH'
    """
    try:
        # Fetch SPY data
        spy = yf.Ticker('SPY')
        history = spy.history(period=f'{days+1}d')
        
        if history.empty or len(history) < 2:
            logger.warning("Insufficient SPY data for market trend")
            return {
                'market_trend': 'NEUTRAL',
                'spy_performance': 0.0,
                'volatility': 0.0,
                'confidence': 0.5
            }
        
        # Calculate performance
        price_start = float(history['Close'].iloc[0])
        price_current = float(history['Close'].iloc[-1])
        performance = ((price_current - price_start) / price_start) * 100
        
        # Calculate volatility (std dev of daily returns)
        returns = history['Close'].pct_change().dropna()
        volatility = float(returns.std() * 100) if len(returns) > 0 else 0.0
        
        # Determine trend
        if performance > 2.0 and volatility < 2.0:
            trend = 'BULLISH'
            confidence = 0.8
        elif performance < -2.0 or volatility > 3.0:
            trend = 'BEARISH'
            confidence = 0.8
        else:
            trend = 'NEUTRAL'
            confidence = 0.6
        
        logger.info(f"Market trend: {trend} (SPY: {performance:.2f}%, volatility: {volatility:.2f}%)")
        
        return {
            'market_trend': trend,
            'spy_performance': performance,
            'volatility': volatility,
            'confidence': confidence
        }
        
    except Exception as e:
        logger.error(f"Failed to get market trend: {str(e)}")
        return {
            'market_trend': 'NEUTRAL',
            'spy_performance': 0.0,
            'volatility': 0.0,
            'confidence': 0.5,
            'error': str(e)
        }


# ============================================================================
# HELPER FUNCTION 4: Analyze Related Companies
# ============================================================================

def analyze_related_companies(related_tickers: List[str]) -> Dict[str, Any]:
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
        >>> analysis = analyze_related_companies(['AMD', 'INTC'])
        >>> print(analysis['overall_signal'])
        'BULLISH'
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
        related_data = []
        
        for ticker in related_tickers:
            try:
                stock = yf.Ticker(ticker)
                history = stock.history(period='2d')
                
                if len(history) >= 2:
                    price_prev = float(history['Close'].iloc[-2])
                    price_current = float(history['Close'].iloc[-1])
                    performance = ((price_current - price_prev) / price_prev) * 100
                    
                    trend = 'UP' if performance > 0.5 else 'DOWN' if performance < -0.5 else 'FLAT'
                    
                    related_data.append({
                        'ticker': ticker,
                        'performance': performance,
                        'trend': trend
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {str(e)}")
                continue
        
        if not related_data:
            return {
                'related_companies': [],
                'avg_performance': 0.0,
                'up_count': 0,
                'down_count': 0,
                'overall_signal': 'NEUTRAL'
            }
        
        # Calculate statistics
        performances = [r['performance'] for r in related_data]
        avg_performance = sum(performances) / len(performances)
        up_count = sum(1 for r in related_data if r['trend'] == 'UP')
        down_count = sum(1 for r in related_data if r['trend'] == 'DOWN')
        
        # Determine overall signal
        if up_count > down_count and avg_performance > 1.0:
            overall_signal = 'BULLISH'
        elif down_count > up_count and avg_performance < -1.0:
            overall_signal = 'BEARISH'
        else:
            overall_signal = 'NEUTRAL'
        
        logger.info(f"Related companies: {up_count} up, {down_count} down → {overall_signal}")
        
        return {
            'related_companies': related_data,
            'avg_performance': avg_performance,
            'up_count': up_count,
            'down_count': down_count,
            'overall_signal': overall_signal
        }
        
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

def calculate_correlation(ticker: str, price_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate correlation between stock and market.
    
    Algorithm:
    1. Get 30 days of stock returns
    2. Get 30 days of SPY returns
    3. Calculate Pearson correlation
    4. Calculate beta (volatility ratio)
    5. High correlation (>0.7) means stock moves with market
    6. Low correlation (<0.3) means stock is independent
    
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
        >>> corr = calculate_correlation('NVDA', price_data)
        >>> print(corr['correlation_strength'])
        'HIGH'
    """
    try:
        # Get 30 days of stock returns
        if len(price_data) < 30:
            logger.warning(f"Insufficient price data for correlation ({len(price_data)} days)")
            return {
                'market_correlation': 0.5,
                'correlation_strength': 'MEDIUM',
                'beta': 1.0
            }
        
        stock_returns = price_data['close'].pct_change().dropna().tail(30)
        
        # Get 30 days of SPY returns
        spy = yf.Ticker('SPY')
        spy_history = spy.history(period='35d')  # Get extra to align dates
        
        if len(spy_history) < 30:
            logger.warning("Insufficient SPY data for correlation")
            return {
                'market_correlation': 0.5,
                'correlation_strength': 'MEDIUM',
                'beta': 1.0
            }
        
        spy_returns = spy_history['Close'].pct_change().dropna().tail(30)
        
        # Align to same length (in case of mismatched trading days)
        min_len = min(len(stock_returns), len(spy_returns))
        stock_returns = stock_returns.tail(min_len)
        spy_returns = spy_returns.tail(min_len)
        
        if len(stock_returns) < 10:
            logger.warning("Too few returns for reliable correlation")
            return {
                'market_correlation': 0.5,
                'correlation_strength': 'MEDIUM',
                'beta': 1.0
            }
        
        # Calculate correlation
        correlation = float(stock_returns.corr(spy_returns))
        
        # Calculate beta
        if spy_returns.std() != 0:
            beta = float((stock_returns.std() / spy_returns.std()) * correlation)
        else:
            beta = 1.0
        
        # Determine strength
        if abs(correlation) > 0.7:
            strength = 'HIGH'
        elif abs(correlation) > 0.4:
            strength = 'MEDIUM'
        else:
            strength = 'LOW'
        
        logger.info(f"Market correlation: {correlation:.2f} ({strength}), beta: {beta:.2f}")
        
        return {
            'market_correlation': correlation,
            'correlation_strength': strength,
            'beta': beta
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate correlation: {str(e)}")
        return {
            'market_correlation': 0.5,
            'correlation_strength': 'MEDIUM',
            'beta': 1.0,
            'error': str(e)
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
    - Sector up: +20, down: -20
    - Market bullish: +30, bearish: -30
    - Related companies bullish: +20, bearish: -20
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
        >>> signal, conf = generate_context_signal(sector, market, related, corr)
        >>> print(signal)
        'BUY'
    """
    score = 0
    
    # Add sector score
    sector_perf = sector_performance.get('performance', 0)
    if sector_perf > 1.0:
        score += 20
    elif sector_perf < -1.0:
        score -= 20
    
    # Add market trend score
    market = market_trend.get('market_trend', 'NEUTRAL')
    if market == 'BULLISH':
        score += 30
    elif market == 'BEARISH':
        score -= 30
    
    # Add related companies score
    related_signal = related_analysis.get('overall_signal', 'NEUTRAL')
    if related_signal == 'BULLISH':
        score += 20
    elif related_signal == 'BEARISH':
        score -= 20
    
    # Apply correlation multiplier
    corr_strength = correlation.get('market_correlation', 0.5)
    if corr_strength > 0.7:
        score = score * 1.5  # High correlation: market matters more
    
    # Convert score to signal
    if score > 30:
        signal = 'BUY'
        confidence = min(100, 50 + score)
    elif score < -30:
        signal = 'SELL'
        confidence = min(100, 50 + abs(score))
    else:
        signal = 'HOLD'
        confidence = 50
    
    logger.info(f"Context signal: {signal} (score: {score:.1f}, confidence: {confidence:.0f}%)")
    
    return signal, confidence


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def market_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 6: Market Context Analysis
    
    Execution flow:
    1. Get stock's sector
    2. Get sector performance
    3. Get overall market trend
    4. Analyze related companies
    5. Calculate correlation with market
    6. Generate context signal
    7. Return partial state update (for parallel execution)
    
    Runs in PARALLEL with: Nodes 4, 5, 7
    Runs AFTER: Node 9A
    Runs BEFORE: Node 8
    
    Args:
        state: LangGraph state
        
    Returns:
        Partial state update with market_context results
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    try:
        logger.info(f"Node 6: Analyzing market context for {ticker}")
        
        # ====================================================================
        # STEP 1: Get Stock's Sector
        # ====================================================================
        sector, industry = get_stock_sector(ticker)
        
        # ====================================================================
        # STEP 2: Get Sector Performance
        # ====================================================================
        sector_performance = get_sector_performance(sector, days=1)
        
        # ====================================================================
        # STEP 3: Get Overall Market Trend
        # ====================================================================
        market_trend = get_market_trend(days=5)
        
        # ====================================================================
        # STEP 4: Analyze Related Companies
        # ====================================================================
        related_tickers = state.get('related_companies', [])
        related_analysis = analyze_related_companies(related_tickers)
        
        # ====================================================================
        # STEP 5: Calculate Correlation with Market
        # ====================================================================
        price_data = state.get('raw_price_data')
        if price_data is not None and not price_data.empty:
            correlation = calculate_correlation(ticker, price_data)
        else:
            correlation = {
                'market_correlation': 0.5,
                'correlation_strength': 'MEDIUM',
                'beta': 1.0
            }
        
        # ====================================================================
        # STEP 6: Generate Context Signal
        # ====================================================================
        context_signal, confidence = generate_context_signal(
            sector_performance,
            market_trend,
            related_analysis,
            correlation
        )
        
        # ====================================================================
        # STEP 7: Build Results
        # ====================================================================
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
        
        logger.info(f"Market Context Results:")
        logger.info(f"  Sector: {sector} ({sector_performance.get('trend', 'FLAT')})")
        logger.info(f"  Market: {market_trend.get('market_trend', 'NEUTRAL')}")
        logger.info(f"  Related: {related_analysis.get('overall_signal', 'NEUTRAL')}")
        logger.info(f"  Signal: {context_signal} (confidence: {confidence:.0f}%)")
        
        # ====================================================================
        # STEP 8: Return Partial State Update (for parallel execution)
        # ====================================================================
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Node 6: Market context analysis completed in {elapsed:.2f}s")
        
        # Return only the fields this node updates
        return {
            'market_context': results,
            'node_execution_times': {'node_6': elapsed}
        }
        
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Node 6: Market context analysis failed for {ticker}: {str(e)}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Return only the fields this node updates (for parallel execution)
        return {
            'errors': [f"Node 6: Market context analysis failed - {str(e)}"],
            'market_context': {
                'sector': 'Unknown',
                'industry': 'Unknown',
                'sector_performance': 0.0,
                'market_trend': 'NEUTRAL',
                'context_signal': 'HOLD',
                'confidence': 50.0
            },
            'node_execution_times': {'node_6': elapsed}
        }
