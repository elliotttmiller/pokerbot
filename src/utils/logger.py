"""Logging utilities."""

import os
from datetime import datetime
from typing import Optional


class Logger:
    """Simple logger for poker bot."""
    
    def __init__(self, log_dir: str = "logs", verbose: bool = True):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files
            verbose: Whether to print to console
        """
        self.log_dir = log_dir
        self.verbose = verbose
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"pokerbot_{timestamp}.log")
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message.
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        # Print to console if verbose
        if self.verbose:
            print(log_message)
        
        # Write to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def info(self, message: str):
        """Log info message."""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """Log warning message."""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")
    
    def debug(self, message: str):
        """Log debug message."""
        if self.verbose:
            self.log(message, "DEBUG")
