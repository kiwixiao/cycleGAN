import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logger()

def check_tensor_size(tensor, expected_shape, tensor_name):
    if tensor.shape != expected_shape:
        error_msg = f"Size mismatch for {tensor_name}. Expected {expected_shape}, got {tensor.shape}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info(f"{tensor_name} shape: {tensor.shape}")