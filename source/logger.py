import os
import logging
from source.constant.constant import ARTIFACT_DIR


def setup_log_timestamp(global_timestamp):

    # Use the global timestamp variable
    LOG_FILE = f"{global_timestamp}.log"
    logs_path = os.path.join(os.getcwd(), ARTIFACT_DIR +'/'+global_timestamp+'/log')

    os.makedirs(logs_path, exist_ok=True)
    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

    # Configure the logger
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

