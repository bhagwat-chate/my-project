import os
import pandas as pd
import os.path
from pandas import DataFrame
from pymongo.mongo_client import MongoClient
from source.logger import logging
from source.exception import ChurnException
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, train_config):
        self.train_config = train_config

    def export_data_into_feature_store(self):
        try:
            database = self.train_config.database_name
            collection = self.train_config.collection_name
            mongodb_url_key = self.train_config.mongodb_url_key

            client = MongoClient(mongodb_url_key)
            database = client[database]
            collection = database[collection]

            cursor = collection.find()
            data = pd.DataFrame(list(cursor))

            feature_store_file_path = self.train_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            data.to_csv(feature_store_file_path, index=False, header=True)

            return data

        except ChurnException as e:
            raise e

    def split_train_test_split(self, dataframe: DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.train_config.train_test_split_ratio)
            logging.info("performed train, test split on the dataframe")
            logging.info("Exited train_test_split of the data_ingestion class")

            dir_path = os.path.dirname(self.train_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path")

            train_set.to_csv(self.train_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.train_config.testing_file_path, index=False, header=True)

        except ChurnException as e:
            raise e

    def initiate_data_ingestion(self):
        data = self.export_data_into_feature_store()
        self.split_train_test_split(data)

