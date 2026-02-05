"""
Helper Utilities
Common utility functions used across the application.
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date string to datetime object.
    
    Handles multiple date formats commonly used in financial APIs.
    
    Args:
        date_str: Date string in various formats
    
    Returns:
        datetime object or None if parsing fails
    
    Example:
        >>> dt = parse_date('2024-01-15')
        >>> dt = parse_date('2024-01-15T10:30:00Z')
    """
    if not date_str:
        return None
    
    # Try ISO format first
    formats = [
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%d %H:%M:%S'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
    
    Returns:
        Percentage change (e.g., 5.0 for 5% increase)
    
    Example:
        >>> change = calculate_percentage_change(100, 105)
        >>> print(change)  # 5.0
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Top number
        denominator: Bottom number
        default: Value to return if division by zero (default: 0.0)
    
    Returns:
        Result of division or default
    
    Example:
        >>> result = safe_divide(10, 2)  # 5.0
        >>> result = safe_divide(10, 0)  # 0.0
    """
    if denominator == 0:
        return default
    return numerator / denominator


def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol to uppercase.
    
    Args:
        ticker: Stock ticker (may be lowercase or mixed case)
    
    Returns:
        Uppercase ticker symbol
    
    Example:
        >>> normalize_ticker('aapl')  # 'AAPL'
        >>> normalize_ticker('NvDa')  # 'NVDA'
    """
    return ticker.strip().upper()


def get_date_range(days: int = 7) -> tuple[str, str]:
    """
    Get date range for fetching historical data.
    
    Args:
        days: Number of days to look back
    
    Returns:
        Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
    
    Example:
        >>> start, end = get_date_range(7)
        >>> # Returns dates for last 7 days
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return (
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (default: 100)
    
    Returns:
        Truncated text with '...' if longer than max_length
    
    Example:
        >>> truncate_text("Very long article title here...", 20)
        'Very long article...'
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length-3] + '...'
