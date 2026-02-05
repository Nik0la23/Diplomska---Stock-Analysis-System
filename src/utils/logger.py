"""
Logging Configuration
Provides structured logging setup for the entire application.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = __name__, 
    level: str = "INFO",
    log_to_file: bool = False,
    log_file_path: str = "logs/application.log"
) -> logging.Logger:
    """
    Setup structured logging with consistent format.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_to_file: Whether to also log to file
        log_file_path: Path to log file if log_to_file is True
    
    Returns:
        Configured logger instance
    
    Example:
        >>> from src.utils.logger import setup_logger
        >>> logger = setup_logger(__name__)
        >>> logger.info("This is a test message")
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Formatter with timestamp, name, level, message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        log_file = Path(log_file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_node_logger(node_name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a logger specifically for a LangGraph node.
    
    Args:
        node_name: Name of the node (e.g., 'node_1', 'node_8')
        level: Logging level
    
    Returns:
        Configured logger with node-specific name
    
    Example:
        >>> logger = get_node_logger('node_1')
        >>> logger.info("Fetching price data for AAPL")
    """
    logger_name = f"langgraph.{node_name}"
    return setup_logger(logger_name, level)


def setup_application_logging(level: str = "INFO") -> None:
    """
    Setup logging for the entire application.
    
    Call this once at application startup.
    
    Args:
        level: Global logging level
    
    Example:
        >>> from src.utils.logger import setup_application_logging
        >>> setup_application_logging('INFO')
    """
    # Setup root logger
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Suppress overly verbose third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('newsapi').setLevel(logging.WARNING)
    logging.getLogger('finnhub').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Application logging configured at {level} level")
