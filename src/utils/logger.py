import logging

def set_up_logger():
    """Sets up a basic logger
    
    Returns
        logger (logger): The logger object
    """
    # Set up logging
    logging.basicConfig(level=logging.DEBUG,  # Set the logging level
                        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
                        handlers=[logging.StreamHandler()])  # Output to console
    return logging.getLogger(__name__)
