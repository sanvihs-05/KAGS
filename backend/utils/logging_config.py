# utils/logging_config.py
"""
Logging configuration utility.
Provides a setup_logger function for consistent logging across modules.
"""

import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up and return a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
    
    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger
