"""
Logger Utility.

Sets up a project-wide standardized logger with file and console output.
"""

import os
import logging
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "poker_bot",
                 level: str = "INFO",
                 log_dir: str = "logs",
                 log_to_file: bool = True,
                 log_to_console: bool = True) -> logging.Logger:
    """
    Set up a standardized logger for the project.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "poker_bot") -> logging.Logger:
    """
    Get existing logger or create new one with defaults.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If no handlers, set up with defaults
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                self.setup_logging()
                self.log.info("Initialized")
    """
    
    def setup_logging(self, name: Optional[str] = None):
        """Set up logger for this instance."""
        logger_name = name or self.__class__.__name__
        self.log = get_logger(logger_name)


# Default logger
_default_logger: Optional[logging.Logger] = None


def log(message: str, level: str = "INFO"):
    """
    Quick logging function using default logger.
    
    Args:
        message: Message to log
        level: Log level
    """
    global _default_logger
    
    if _default_logger is None:
        _default_logger = setup_logger()
    
    log_method = getattr(_default_logger, level.lower(), _default_logger.info)
    log_method(message)


__all__ = ['setup_logger', 'get_logger', 'LoggerMixin', 'log']
