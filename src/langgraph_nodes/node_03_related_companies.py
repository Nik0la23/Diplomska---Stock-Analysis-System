"""
Node 3: Related Companies Detection
Identifies competitor and related companies using Finnhub Peers API and correlation analysis.

Runs AFTER: Node 1 (uses price data for correlation)
Runs BEFORE: Node 2 (news fetching needs related company list)
Can run in PARALLEL with: Nothing
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import finnhub

from src.utils.config import FINNHUB_API_KEY
from src.database.db_manager import get_cached_price_data

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTION: Fetch Peers from Finnhub
# ============================================================================

def fetch_peers_from_finnhub(ticker: str) -> List[str]:
    """
    Fetch peer companies from Finnhub API.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'NVDA')
        
    Returns:
        List of peer ticker symbols
        Empty list if fetch fails
        
    Example:
        >>> peers = fetch_peers_from_finnhub('NVDA')
        >>> print(peers)
        ['AMD', 'INTC', 'TSM', 'QCOM', 'AVGO', ...]
    """
    try:
        if not FINNHUB_API_KEY:
            logger.warning("Finnhub API key not configured")
            return []
        
        logger.info(f"Fetching peers from Finnhub for {ticker}")
        
        # Initialize Finnhub client
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        
        # Fetch peers
        peers = finnhub_client.company_peers(ticker)
        
        if not peers:
            logger.warning(f"No peers returned from Finnhub for {ticker}")
            return []
        
        # Remove the target ticker from peers list (don't include itself)
        peers = [p for p in peers if p.upper() != ticker.upper()]
        
        logger.info(f"Finnhub: Found {len(peers)} peers for {ticker}")
        return peers
        
    except Exception as e:
        logger.error(f"Finnhub peers fetch failed for {ticker}: {str(e)}")
        return []


# ============================================================================
# HELPER FUNCTION: Calculate Price Correlation
# ============================================================================

def calculate_price_correlation(
    target_ticker: str,
    peer_ticker: str,
    target_df: pd.DataFrame
) -> Optional[float]:
    """
    Calculate price correlation between target and peer stock.
    
    Args:
        target_ticker: Target stock ticker
        peer_ticker: Peer stock ticker
        target_df: Price data for target stock
        
    Returns:
        Correlation coefficient (0.0 to 1.0)
        None if calculation fails
        
    Example:
        >>> corr = calculate_price_correlation('NVDA', 'AMD', nvda_df)
        >>> print(f"Correlation: {corr:.2f}")
        Correlation: 0.85
    """
    try:
        # Get peer price data from cache
        peer_df = get_cached_price_data(peer_ticker, max_age_hours=24)
        
        if peer_df is None or peer_df.empty:
            logger.debug(f"No cached price data for {peer_ticker}")
            return None
        
        # Align dataframes by date
        target_df = target_df[['date', 'close']].copy()
        peer_df = peer_df[['date', 'close']].copy()
        
        # Merge on date
        merged = pd.merge(
            target_df,
            peer_df,
            on='date',
            suffixes=('_target', '_peer')
        )
        
        if len(merged) < 30:
            logger.debug(f"Insufficient overlapping data for correlation ({len(merged)} days)")
            return None
        
        # Calculate correlation
        correlation = merged['close_target'].corr(merged['close_peer'])
        
        # Return absolute correlation (we care about relationship strength)
        return abs(correlation)
        
    except Exception as e:
        logger.debug(f"Correlation calculation failed for {peer_ticker}: {str(e)}")
        return None


# ============================================================================
# HELPER FUNCTION: Rank and Filter Peers
# ============================================================================

def rank_peers_by_correlation(
    target_ticker: str,
    peers: List[str],
    target_df: Optional[pd.DataFrame],
    max_peers: int = 5
) -> List[str]:
    """
    Rank peers by correlation and return top N.
    
    Args:
        target_ticker: Target stock ticker
        peers: List of peer tickers
        target_df: Price data for target stock (for correlation)
        max_peers: Maximum number of peers to return
        
    Returns:
        List of top N peer tickers sorted by correlation
        
    Example:
        >>> ranked = rank_peers_by_correlation('NVDA', all_peers, price_df, max_peers=5)
        >>> print(ranked)
        ['AMD', 'INTC', 'TSM', 'QCOM', 'AVGO']
    """
    if target_df is None or target_df.empty or len(peers) == 0:
        # No price data for correlation, just return first N peers
        return peers[:max_peers]
    
    logger.info(f"Ranking {len(peers)} peers by correlation with {target_ticker}")
    
    # Calculate correlation for each peer
    peer_correlations = []
    for peer in peers:
        corr = calculate_price_correlation(target_ticker, peer, target_df)
        if corr is not None:
            peer_correlations.append((peer, corr))
    
    if not peer_correlations:
        # No correlations calculated, return first N peers
        logger.info(f"No correlations calculated, returning first {max_peers} peers")
        return peers[:max_peers]
    
    # Sort by correlation (highest first)
    peer_correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N peers
    top_peers = [peer for peer, corr in peer_correlations[:max_peers]]
    
    # Log correlations
    logger.info(f"Top {len(top_peers)} peers by correlation:")
    for peer, corr in peer_correlations[:max_peers]:
        logger.info(f"  {peer}: {corr:.3f}")
    
    return top_peers


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

def detect_related_companies_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 3: Related Companies Detection
    
    Execution flow:
    1. Fetch peer companies from Finnhub API
    2. Optionally calculate price correlations
    3. Rank peers by correlation strength
    4. Return top 3-5 related companies
    5. Update state with related companies list
    
    Args:
        state: LangGraph state containing 'ticker' and optionally 'raw_price_data'
        
    Returns:
        Updated state with 'related_companies' populated
    """
    start_time = datetime.now()
    ticker = state['ticker']
    
    try:
        logger.info(f"Node 3: Starting related companies detection for {ticker}")
        
        # ====================================================================
        # STEP 1: Fetch Peers from Finnhub
        # ====================================================================
        peers = fetch_peers_from_finnhub(ticker)
        
        if not peers:
            logger.warning(f"Node 3: No peers found for {ticker}")
            state['related_companies'] = []
            elapsed = (datetime.now() - start_time).total_seconds()
            state['node_execution_times']['node_3'] = elapsed
            logger.info(f"Node 3: Completed with 0 peers in {elapsed:.2f}s")
            return state
        
        logger.info(f"Node 3: Found {len(peers)} potential peers for {ticker}")
        
        # ====================================================================
        # STEP 2: Get Target Stock Price Data (for correlation)
        # ====================================================================
        target_df = state.get('raw_price_data')
        
        if target_df is None:
            logger.info(f"Node 3: No price data available for correlation analysis")
        
        # ====================================================================
        # STEP 3: Rank Peers by Correlation
        # ====================================================================
        max_peers = 5  # Limit to 5 to avoid overwhelming news fetching
        ranked_peers = rank_peers_by_correlation(ticker, peers, target_df, max_peers)
        
        # ====================================================================
        # STEP 4: Validate Results
        # ====================================================================
        # Ensure target ticker is not in the list
        ranked_peers = [p for p in ranked_peers if p.upper() != ticker.upper()]
        
        # Ensure we have at least some peers (but not too many)
        if len(ranked_peers) > max_peers:
            ranked_peers = ranked_peers[:max_peers]
        
        # ====================================================================
        # STEP 5: Update State
        # ====================================================================
        state['related_companies'] = ranked_peers
        elapsed = (datetime.now() - start_time).total_seconds()
        state['node_execution_times']['node_3'] = elapsed
        
        logger.info(f"Node 3: Successfully found {len(ranked_peers)} related companies for {ticker} in {elapsed:.2f}s")
        logger.info(f"Node 3: Related companies: {', '.join(ranked_peers)}")
        
        return state
        
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Node 3: Unexpected error for {ticker}: {str(e)}")
        state['errors'].append(f"Node 3: Unexpected error - {str(e)}")
        state['related_companies'] = []
        elapsed = (datetime.now() - start_time).total_seconds()
        state['node_execution_times']['node_3'] = elapsed
        return state
