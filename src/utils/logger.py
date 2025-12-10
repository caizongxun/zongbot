"""Logging configuration and utilities."""

import logging
import logging.handlers
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Get or create a logger with file and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory for log files
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / "zongbot.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
