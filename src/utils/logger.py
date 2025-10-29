"""
Logging utilities for the Quantum Optimal Transport project.

This module provides a centralized logging configuration with support for
both file and console handlers, colored output, and timing utilities.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import contextmanager
import time


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Parameters
    ----------
    name : str
        Logger name (usually __name__)
    log_dir : Optional[Path], optional
        Directory for log files, by default None (uses logs/)
    level : str, optional
        Logging level, by default "INFO"
    log_to_file : bool, optional
        Enable file logging, by default True
    log_to_console : bool, optional
        Enable console logging, by default True
    
    Returns
    -------
    logging.Logger
        Configured logger instance
        
    Examples
    --------
    >>> from src.utils.logger import setup_logger
    >>> logger = setup_logger(__name__)
    >>> logger.info("Starting computation")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path("logs")
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name.replace('.', '_')}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    return logger


@contextmanager
def log_timing(logger: logging.Logger, operation: str):
    """
    Context manager for timing operations.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    operation : str
        Description of the operation being timed
        
    Yields
    ------
    None
    
    Examples
    --------
    >>> from src.utils.logger import setup_logger, log_timing
    >>> logger = setup_logger(__name__)
    >>> with log_timing(logger, "data loading"):
    ...     data = load_large_dataset()
    INFO - Starting data loading
    INFO - Completed data loading in 2.35s
    """
    start = time.time()
    logger.info(f"Starting {operation}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed {operation} in {elapsed:.2f}s")


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger by name.
    
    If logger doesn't exist, creates a basic one.
    
    Parameters
    ----------
    name : str
        Logger name
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name)
    return logger


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    
    Classes inheriting from this mixin will have a `logger` attribute
    automatically configured.
    
    Examples
    --------
    >>> class MyClass(LoggerMixin):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.logger.info("MyClass initialized")
    ...     
    ...     def process(self):
    ...         self.logger.debug("Processing...")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(name)
