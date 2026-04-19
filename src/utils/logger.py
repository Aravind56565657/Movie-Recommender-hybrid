"""Logging utility"""
import logging
import sys

def setup_logger(name, config):
    """Setup logger with consistent formatting"""
    
    logger = logging.getLogger(name)
    logger.setLevel(config['level'])
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(config['log_file'])
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(config['format'])
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
