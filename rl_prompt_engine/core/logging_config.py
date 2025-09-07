#!/usr/bin/env python3
"""
Logging configuration for RL Prompt Engine
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """
    Set up logging configuration for the RL Prompt Engine.
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory to store log files (default: "logs")
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler
            logging.FileHandler(log_file, mode='w'),
            # Console handler
            logging.StreamHandler()
        ]
    )
    
    # Create specific loggers
    training_logger = logging.getLogger('training')
    env_logger = logging.getLogger('environment')
    model_logger = logging.getLogger('model')
    
    # Set levels for specific loggers
    training_logger.setLevel(logging.INFO)
    env_logger.setLevel(logging.DEBUG)
    model_logger.setLevel(logging.INFO)
    
    # Log the setup
    training_logger.info(f"Logging initialized. Log file: {log_file}")
    training_logger.info(f"Log level: {logging.getLevelName(log_level)}")
    
    return {
        'training': training_logger,
        'environment': env_logger,
        'model': model_logger,
        'log_file': str(log_file)
    }

def get_logger(name: str):
    """Get a logger by name."""
    return logging.getLogger(name)
