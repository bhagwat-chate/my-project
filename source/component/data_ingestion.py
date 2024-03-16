import os, sys
import pandas as pd
import os.path
from pandas import DataFrame
from pymongo.mongo_client import MongoClient
from source.utility.utility import export_csv_file
from source.logger import logging
from source.exception import ChurnException
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, utility_config):
        self.utility_config = utility_config

    def export_data_into_feature_store(self, key):
        try:
            # Determine collection name and feature store file path based on the key
            if key == 'train':
                collection_name = self.utility_config.train_di_collection_name
                feature_store_file_path = self.utility_config.train_di_feature_store_file_path
            else:
                collection_name = self.utility_config.predict_di_collection_name
                feature_store_file_path = self.utility_config.predict_di_feature_store_file_path

            # Connect to MongoDB and retrieve data from the specified collection
            client = MongoClient(self.utility_config.mongodb_url_key)
            database = client[self.utility_config.database_name]

            collection = database[collection_name]
            cursor = collection.find()
            data = pd.DataFrame(list(cursor))

            # Ensure directory for feature store file exists
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Export data to CSV file
            data.to_csv(feature_store_file_path, index=False, header=True)

            return data

        except ChurnException as e:
            raise e

    def split_train_test_split(self, dataframe: DataFrame):
        try:

            train_data, test_data = train_test_split(dataframe, test_size=self.utility_config.train_test_split_ratio)

            return train_data, test_data

        except ChurnException as e:
            raise e

    def clean_data(self, data, key):
        try:
            logging.info("start: data cleaning")

            if key == 'train':
                drop_column = []
                # Remove duplicates
                data = data.drop_duplicates()

                # Remove low-variance columns (with only one unique value)
                data = data.loc[:, data.nunique() > 1]

                # Remove high-cardinality categorical columns (more than 80% unique values)
                for col in data.select_dtypes(include=['object']).columns:
                    unique_count = data[col].nunique()
                    if unique_count / len(data) > 0.5:
                        data.drop(col, axis=1, inplace=True)
                        drop_column.append(col)

                logging.info(f"data clean: dropped columns: {drop_column}")

            elif key == 'predict':
                # Remove duplicates
                data = data.drop_duplicates()

            logging.info("complete: data cleaning")

            return data

        except ChurnException as e:
            raise e

    def process_data(self, data: DataFrame, key) -> DataFrame:
        try:
            logging.info("start: data process")

            if key == 'train':
                mandatory_cols = self.utility_config.dv_mandatory_col_list
            else:
                mandatory_cols = self.utility_config.dv_mandatory_col_list.copy()
                self.utility_config.col_drop_in_clean.append(self.utility_config.target_column)

                for col in self.utility_config.col_drop_in_clean:
                    if col in mandatory_cols:
                        mandatory_cols.remove(col)

            for col in mandatory_cols:
                if col not in data.columns:
                    raise ChurnException(f"Missing mandatory column: {col}")
                if data[col].dtype != self.utility_config.dv_mandatory_col_data_type[col]:
                    try:
                        data[col] = data[col].astype(self.utility_config.dv_mandatory_col_data_type[col])
                    except ValueError as e:
                        raise ChurnException(f"Error converting data type for column '{col}': {e}")

            # drop if any extra columns present
            data = data[mandatory_cols]

            logging.info("complete: data process")

            return data  # Return the final DataFrame

        except ChurnException as e:
            raise e  # Re-raise any exceptions to halt execution

    def initiate_data_ingestion(self, key):
        try:

            logging.info("start: data ingestion")

            data = self.export_data_into_feature_store(key)
            data = self.process_data(data, key)
            data = self.clean_data(data, key)

            if key == 'train':

                train_data, test_data = self.split_train_test_split(data)

                export_csv_file(train_data, self.utility_config.train_file_name, self.utility_config.di_train_file_path)
                export_csv_file(test_data, self.utility_config.test_file_name, self.utility_config.di_test_file_path)

            if key == 'predict':
                export_csv_file(data, self.utility_config.predict_file_name, self.utility_config.predict_di_file_path)

            logging.info("end: data ingestion")

        except ChurnException as e:
            raise e
