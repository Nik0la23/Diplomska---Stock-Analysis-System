"""
Domain Authority and Content Analysis Configuration
Centralized mappings for source credibility assessment and content scoring.

Used by Node 9A for content-based anomaly detection.
"""

from typing import Dict, List

# ============================================================================
# SOURCE CREDIBILITY MAPPING
# ============================================================================

TRUSTED_DOMAINS: Dict[str, float] = {
    # Tier 1: Premium Financial News (0.9-1.0)
    'bloomberg.com': 0.95,
    'reuters.com': 0.95,
    'wsj.com': 0.92,
    'ft.com': 0.90,
    'apnews.com': 0.90,
    
    # Tier 2: Major Financial Media (0.7-0.8)
    'cnbc.com': 0.80,
    'marketwatch.com': 0.78,
    'forbes.com': 0.75,
    'businessinsider.com': 0.72,
    'barrons.com': 0.78,
    'investing.com': 0.70,
    
    # Tier 3: General Finance Sites (0.5-0.6)
    'finance.yahoo.com': 0.60,
    'seekingalpha.com': 0.55,
    'medium.com': 0.50,
    'fool.com': 0.55,
    'benzinga.com': 0.58,
    
    # Tier 4: News Aggregators (0.4-0.5)
    'google.com': 0.45,
    'news.google.com': 0.45,
    'msn.com': 0.40,
}

# Default score for unknown domains
DEFAULT_CREDIBILITY_SCORE: float = 0.30


# ============================================================================
# FINANCIAL KEYWORDS (for content tagging)
# ============================================================================

FINANCIAL_KEYWORDS: List[str] = [
    # Negative Events
    'fraud', 'investigation', 'lawsuit', 'scandal', 'controversy',
    'bankruptcy', 'insolvency', 'default', 'violation', 'penalty',
    'fine', 'sanctions', 'probe', 'criminal', 'charges',
    
    # Regulatory
    'SEC', 'FINRA', 'DOJ', 'FTC', 'regulatory', 'compliance',
    'subpoena', 'inquiry', 'enforcement', 'settlement',
    
    # Corporate Events
    'merger', 'acquisition', 'takeover', 'buyout', 'IPO',
    'earnings', 'revenue', 'profit', 'loss', 'guidance',
    'dividend', 'split', 'buyback', 'restructuring',
    
    # Market Terms
    'volatility', 'correction', 'crash', 'rally', 'surge',
    'plunge', 'soar', 'tumble', 'spike', 'slump',
]


# ============================================================================
# URGENCY PHRASES (for urgency scoring)
# ============================================================================

URGENCY_PHRASES: List[str] = [
    'BREAKING',
    'URGENT',
    'ALERT',
    'ACT NOW',
    'IMMEDIATE',
    'HURRY',
    'LIMITED TIME',
    'LAST CHANCE',
    'EXPIRES SOON',
    'DON\'T MISS',
    'TIME SENSITIVE',
    'DEADLINE',
    'RIGHT NOW',
    'THIS INSTANT',
    'ASAP',
]


# ============================================================================
# SENSATIONALISM INDICATORS (for sensationalism scoring)
# ============================================================================

SENSATIONALISM_KEYWORDS: List[str] = [
    'SHOCKING',
    'UNBELIEVABLE',
    'INCREDIBLE',
    'EXPOSED',
    'REVEALED',
    'SECRET',
    'HIDDEN',
    'TRUTH',
    'INSIDER',
    'EXCLUSIVE',
    'BOMBSHELL',
    'STUNNING',
    'REVOLUTIONARY',
    'BREAKTHROUGH',
    'GAME-CHANGER',
    'MIRACLE',
    'ULTIMATE',
    'PERFECT',
    'GUARANTEED',
    'EXPLOSIVE',
]


# ============================================================================
# UNVERIFIED CLAIMS INDICATORS (for hedging language detection)
# ============================================================================

HEDGING_PHRASES: List[str] = [
    'allegedly',
    'reportedly',
    'rumored',
    'rumoured',
    'sources say',
    'sources claim',
    'unconfirmed',
    'speculation',
    'could be',
    'might be',
    'may be',
    'possibly',
    'potentially',
    'appears to',
    'seems to',
    'according to rumors',
    'word on the street',
    'insiders suggest',
    'anonymous sources',
    'unnamed sources',
]


# ============================================================================
# FINANCIAL JARGON (for complexity scoring)
# ============================================================================

TECHNICAL_FINANCIAL_TERMS: List[str] = [
    # Advanced Financial Concepts
    'derivatives', 'arbitrage', 'hedging', 'leverage', 'liquidity',
    'volatility', 'beta', 'alpha', 'sharpe ratio', 'correlation',
    'covariance', 'diversification', 'portfolio', 'asset allocation',
    
    # Accounting Terms
    'EBITDA', 'GAAP', 'depreciation', 'amortization', 'accrual',
    'balance sheet', 'income statement', 'cash flow', 'equity',
    'liability', 'assets', 'retained earnings',
    
    # Trading Terms
    'call option', 'put option', 'strike price', 'futures', 'swaps',
    'short selling', 'margin', 'stop loss', 'limit order', 'market order',
    
    # Corporate Finance
    'cap table', 'dilution', 'convertible note', 'preferred stock',
    'common stock', 'warrants', 'vesting', 'lockup period',
    
    # Valuation
    'DCF', 'P/E ratio', 'EPS', 'book value', 'market cap',
    'enterprise value', 'multiples', 'comparable analysis',
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_domain_credibility(url: str, source_name: str = '') -> float:
    """
    Get credibility score for a domain/source.
    
    Args:
        url: Article URL
        source_name: Source name (fallback if URL parsing fails)
        
    Returns:
        Credibility score (0.0 to 1.0)
        
    Example:
        >>> get_domain_credibility('https://bloomberg.com/article')
        0.95
        >>> get_domain_credibility('https://unknown-blog.com/news')
        0.30
    """
    # Extract domain from URL
    if url:
        # Simple domain extraction
        url_lower = url.lower()
        for domain, score in TRUSTED_DOMAINS.items():
            if domain in url_lower:
                return score
    
    # Try source name as fallback
    if source_name:
        source_lower = source_name.lower()
        for domain, score in TRUSTED_DOMAINS.items():
            # Check if domain name appears in source name
            domain_base = domain.replace('.com', '').replace('.', '')
            if domain_base in source_lower:
                return score
    
    # Unknown source - return default low score
    return DEFAULT_CREDIBILITY_SCORE


def is_trusted_source(url: str, threshold: float = 0.7) -> bool:
    """
    Check if a source is considered trusted (above threshold).
    
    Args:
        url: Article URL
        threshold: Minimum credibility score to be considered trusted
        
    Returns:
        True if trusted, False otherwise
    """
    return get_domain_credibility(url) >= threshold
