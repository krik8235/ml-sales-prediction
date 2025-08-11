import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

main_logger = logging.getLogger()
main_logger.setLevel(logging.INFO)
