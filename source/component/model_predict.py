import pandas as pd
import pickle
import pymongo
from pymongo.mongo_client import MongoClient
from pymongo import MongoClient
from source.exception import ChurnException
from source.utility.utility import import_csv_file


class ModelPredict:
    def __init__(self, utility_config):
        self.utility_config = utility_config

    def load_model_pickle(self, model_path):
        try:

            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            return model

        except ChurnException as e:
            raise e

    def predict(self, model, data):
        try:
            return model.predict(data)

        except ChurnException as e:
            raise e

    def export_prediction_into_db(self, feature_data):
        try:
            # Connect to MongoDB
            with MongoClient(self.utility_config.mongodb_url_key) as client:
                database = client[self.utility_config.database_name]
                collection = database[self.utility_config.predict_di_collection_name]

                # Prepare bulk write operations
                bulk_operations = []
                for index, row in feature_data.iterrows():
                    cust_id = row['customerID']
                    churn_value = row['Churn']
                    # Update document or insert if it doesn't exist
                    bulk_operations.append(
                        pymongo.UpdateOne({"customerID": cust_id}, {"$set": {"Churn": churn_value}}, upsert=True)
                    )

                # Execute bulk write operations
                if bulk_operations:
                    collection.bulk_write(bulk_operations)

        except Exception as e:
            raise ChurnException(f"An error occurred while exporting predictions to the database: {e}")

    def initiate_model_prediction(self):
        try:
            predict_data = import_csv_file(self.utility_config.predict_file_name, self.utility_config.predict_dt_file_path)

            model = self.load_model_pickle('source/ml/final_model/Gradient_Boosting_Machines.pkl')

            feature_data = import_csv_file(self.utility_config.predict_di_feature_store_file_name, self.utility_config.predict_di_feature_store_file_path)
            feature_data['Churn'] = self.predict(model, predict_data)

            self.export_prediction_into_db(feature_data)


        except ChurnException as e:
            raise e
