"""
Configuration Management
Loads environment variables and validates required API keys.
"""

from dotenv import load_dotenv
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'

if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment variables from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}. Using environment variables only.")
    load_dotenv()  # Try to load from system environment


# ============================================================================
# REQUIRED API KEYS
# ============================================================================

# Data sources
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')      # Primary price data (Node 1)
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')      # News, peers, market news (Nodes 2, 3)

# LLM for explanations
GROQ_API_KEY = os.getenv('GROQ_API_KEY')            # Explanations (Nodes 13, 14)


# ============================================================================
# OPTIONAL SETTINGS
# ============================================================================

DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/stock_prices.db')
CACHE_HOURS = int(os.getenv('CACHE_HOURS', '24'))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config() -> None:
    """
    Validate that all required configuration is present.
    
    Raises:
        ValueError: If required API keys are missing
    
    Example:
        >>> from src.utils.config import validate_config
        >>> validate_config()
        >>> # Raises ValueError if keys missing
    """
    missing_keys = []
    
    if not POLYGON_API_KEY:
        missing_keys.append('POLYGON_API_KEY')
    
    if not FINNHUB_API_KEY:
        missing_keys.append('FINNHUB_API_KEY')
    
    if not GROQ_API_KEY:
        missing_keys.append('GROQ_API_KEY')
    
    if missing_keys:
        raise ValueError(
            f"Missing required API keys in .env file: {', '.join(missing_keys)}\n"
            f"Please copy .env.example to .env and add your API keys."
        )
    
    logger.info("Configuration validated successfully")


def get_config_summary() -> dict:
    """
    Get configuration summary for debugging.
    
    Returns:
        Dict with configuration status (keys masked for security)
    """
    return {
        'polygon_key_present': bool(POLYGON_API_KEY),
        'finnhub_key_present': bool(FINNHUB_API_KEY),
        'groq_key_present': bool(GROQ_API_KEY),
        'database_path': DATABASE_PATH,
        'cache_hours': CACHE_HOURS,
        'log_level': LOG_LEVEL
    }


# ============================================================================
# AUTO-VALIDATION ON IMPORT (Optional - comment out if not desired)
# ============================================================================

# Uncomment to validate on import:
# try:
#     validate_config()
# except ValueError as e:
#     logger.warning(f"Configuration incomplete: {str(e)}")
