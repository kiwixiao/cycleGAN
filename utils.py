import logging
import torch
import os
from datetime import datetime

def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    # Create a unique log filename based on the current date and time
    log_filename = os.path.join('logs', f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename),
                            logging.StreamHandler()
                        ])
    return logging.getLogger(__name__)

logger = setup_logger()

def check_tensor_size(tensor, expected_shape, tensor_name):
    if tensor.shape != expected_shape:
        error_msg = f"Size mismatch for {tensor_name}. Expected {expected_shape}, got {tensor.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info(f"{tensor_name} shape: {tensor.shape}")