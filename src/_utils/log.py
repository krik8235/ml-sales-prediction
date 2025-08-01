import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
main_logger = logging.getLogger('system')

# logging.debug("This is a debug message - won't show with INFO level by default")
# logging.warning("This is a warning message")
# logging.error("This is an error message")