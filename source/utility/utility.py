import os
import logging
from datetime import datetime
from source.exception import ChurnException

global_timestamp = None


def generate_global_timestamp():
    global global_timestamp
    if global_timestamp is None:
        global_timestamp = str(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    return global_timestamp


def export_csv_file(df, filename, file_path):
    try:
        # Check directory existence
        if not os.path.exists(file_path):
            os.makedirs(file_path)  # Create directory if it doesn't exist

        df.to_csv(os.path.join(file_path, filename), index=False)

    except ChurnException as e:
        raise e

