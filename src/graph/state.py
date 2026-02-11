"""
LangGraph State Definition
Complete state for 16-node stock analysis system.

This is the single source of truth for all inter-node communication.
All nodes receive and return this state.
"""

from typing import TypedDict, List, Dict, Optional, Any, Annotated
import pandas as pd
from datetime import datetime
import operator


class StockAnalysisState(TypedDict):
    """
    Complete state for LangGraph stock analysis pipeline.
    
    All 16 nodes communicate through this shared state.
    Use Annotated with operator.add for lists that accumulate across nodes.
    """
    
    # ========================================================================
    # INPUT (Set by user/dashboard)
    # ========================================================================
    ticker: str                                    # Stock ticker (e.g., 'AAPL')
    
    
    # ========================================================================
    # NODE 1: Price Data Fetching
    # ========================================================================
    raw_price_data: Optional[pd.DataFrame]         # OHLCV data (50+ days)
    
    
    # ========================================================================
    # NODE 2: Multi-Source News Fetching
    # ========================================================================
    stock_news: List[Dict[str, Any]]               # Stock-specific news
    market_news: List[Dict[str, Any]]              # Broad market news
    related_company_news: List[Dict[str, Any]]     # Related company news
    
    
    # ========================================================================
    # NODE 3: Related Companies Detection
    # ========================================================================
    related_companies: List[str]                   # Related tickers (up to 5)
    
    
    # ========================================================================
    # NODE 4: Technical Analysis
    # ========================================================================
    technical_indicators: Optional[Dict[str, float]]  # RSI, MACD, etc.
    technical_signal: Optional[str]                   # 'BUY', 'SELL', 'HOLD'
    technical_confidence: Optional[float]             # 0.0 to 1.0
    
    
    # ========================================================================
    # NODE 5: Sentiment Analysis (FinBERT)
    # ========================================================================
    raw_sentiment_scores: List[Dict[str, float]]      # Per-article scores
    aggregated_sentiment: Optional[float]             # Overall sentiment
    sentiment_signal: Optional[str]                   # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
    sentiment_confidence: Optional[float]             # 0.0 to 1.0
    
    
    # ========================================================================
    # NODE 6: Market Context Analysis
    # ========================================================================
    market_context: Optional[Dict[str, Any]]          # Sector trends, market mood
    
    
    # ========================================================================
    # NODE 7: Monte Carlo Forecasting
    # ========================================================================
    monte_carlo_results: Optional[Dict[str, Any]]     # Simulations, confidence intervals
    forecasted_price: Optional[float]                 # Expected price
    price_range: Optional[tuple]                      # (lower_bound, upper_bound)
    
    
    # ========================================================================
    # NODE 8: News Verification & Learning (THESIS INNOVATION)
    # ========================================================================
    verified_sentiment: Optional[float]               # Sentiment after reliability adjustment
    verified_confidence: Optional[float]              # Adjusted confidence
    source_reliability_scores: Dict[str, float]       # Per-source accuracy
    
    
    # ========================================================================
    # NODE 9A: Early Anomaly Detection (Content-based)
    # ========================================================================
    cleaned_stock_news: List[Dict[str, Any]]          # Stock news with embedded scores
    cleaned_market_news: List[Dict[str, Any]]         # Market news with embedded scores
    cleaned_related_company_news: List[Dict[str, Any]] # Related news with embedded scores
    content_analysis_summary: Optional[Dict[str, Any]] # Overall content analysis results
    
    
    # ========================================================================
    # NODE 9B: Behavioral Anomaly Detection
    # ========================================================================
    behavioral_anomalies: List[Dict[str, Any]]        # Pump-and-dump patterns
    manipulation_risk: Optional[str]                  # 'LOW', 'MEDIUM', 'HIGH'
    
    
    # ========================================================================
    # NODE 10: Backtesting
    # ========================================================================
    backtest_results: Optional[Dict[str, float]]      # Historical performance
    
    
    # ========================================================================
    # NODE 11: Adaptive Weights Calculation
    # ========================================================================
    adaptive_weights: Optional[Dict[str, float]]      # Optimized signal weights
    
    
    # ========================================================================
    # NODE 12: Final Signal Generation
    # ========================================================================
    final_signal: Optional[str]                       # 'BUY', 'SELL', 'HOLD'
    final_confidence: Optional[float]                 # 0.0 to 1.0
    signal_components: Optional[Dict[str, Any]]       # Breakdown of signal
    
    
    # ========================================================================
    # NODE 13: Beginner Explanation (LLM)
    # ========================================================================
    beginner_explanation: Optional[str]               # Plain English explanation
    
    
    # ========================================================================
    # NODE 14: Technical Explanation (LLM)
    # ========================================================================
    technical_explanation: Optional[str]              # Detailed technical report
    
    
    # ========================================================================
    # NODE 15: Dashboard Data Preparation
    # ========================================================================
    dashboard_data: Optional[Dict[str, Any]]          # All viz-ready data
    
    
    # ========================================================================
    # SYSTEM TRACKING
    # ========================================================================
    errors: Annotated[List[str], operator.add]        # Accumulated errors
    node_execution_times: Annotated[Dict[str, float], operator.or_]  # Per-node timing (mergeable)
    total_execution_time: Optional[float]             # End-to-end time
    timestamp: Optional[datetime]                     # When analysis started


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_initial_state(ticker: str) -> StockAnalysisState:
    """
    Create initial state for a new stock analysis.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        Initialized state with required fields
        
    Example:
        >>> state = create_initial_state('AAPL')
        >>> state['ticker']
        'AAPL'
    """
    return StockAnalysisState(
        ticker=ticker.upper(),
        
        # Initialize all fields
        raw_price_data=None,
        stock_news=[],
        market_news=[],
        related_company_news=[],
        related_companies=[],
        technical_indicators=None,
        technical_signal=None,
        technical_confidence=None,
        raw_sentiment_scores=[],
        aggregated_sentiment=None,
        sentiment_signal=None,
        sentiment_confidence=None,
        market_context=None,
        monte_carlo_results=None,
        forecasted_price=None,
        price_range=None,
        verified_sentiment=None,
        verified_confidence=None,
        source_reliability_scores={},
        cleaned_stock_news=[],
        cleaned_market_news=[],
        cleaned_related_company_news=[],
        content_analysis_summary=None,
        behavioral_anomalies=[],
        manipulation_risk=None,
        backtest_results=None,
        adaptive_weights=None,
        final_signal=None,
        final_confidence=None,
        signal_components=None,
        beginner_explanation=None,
        technical_explanation=None,
        dashboard_data=None,
        
        # System tracking
        errors=[],
        node_execution_times={},
        total_execution_time=None,
        timestamp=datetime.now()
    )


def validate_state(state: StockAnalysisState) -> bool:
    """
    Validate that state has minimum required fields.
    
    Args:
        state: State to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['ticker', 'errors', 'node_execution_times']
    return all(field in state for field in required_fields)
