import os
import pandas as pd
import os.path
from pandas import DataFrame
from pymongo.mongo_client import MongoClient
from source.logger import logging
from source.exception import ChurnException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataTransformation:
    def __init__(self, utility_config):
        self.utility_config = utility_config

    def min_max_scaling(self, data, type):
        """
        Perform Min-Max Scaling on the numerical features of the input data and save scaling details.

        Parameters:
        - data (DataFrame): Input data to be scaled.
        - type (str): Indicates whether the data is for training or testing.

        Returns:
        - DataFrame: Scaled data.
        """

        if type == 'train':
            # Extract numerical columns
            numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

            # Create MinMaxScaler object
            scaler = MinMaxScaler()

            # Fit the scaler to the training data
            scaler.fit(data[numeric_columns])

            # Prepare DataFrame with columns: feature, scalar_value
            scaling_details = pd.DataFrame({'Feature': numeric_columns,
                                            'Scalar_Min': scaler.data_min_,
                                            'Scalar_Max': scaler.data_max_})

            # Save scaling details to CSV
            scaling_details.to_csv('source/ml/scaling_details.csv', index=False)

            # Transform the numerical features
            scaled_data = scaler.transform(data[numeric_columns])

            # Assign scaled values back to the original DataFrame
            data.loc[:, numeric_columns] = scaled_data

            return data

        elif type == 'test':
            # Read scaling details from CSV
            scaling_details = pd.read_csv('source/ml/scaling_details.csv')

            # Initialize MinMaxScaler object
            scaler = MinMaxScaler()

            # Set the min and max values for scaling
            scaler.min_ = scaling_details.set_index('Feature')['Scalar_Min']
            scaler.scale_ = (scaling_details.set_index('Feature')['Scalar_Max'] - scaler.min_)

            # Extract numerical columns
            # numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

            # Transform the numerical features
            # scaled_data = scaler.transform(data[numeric_columns])

            counter=0
            for col in data.select_dtypes(include=['float64', 'int64']).columns:

                min: int = scaling_details.loc[]
                max = scaling_details.set_index('Feature')['Scalar_Max']

                for i in range(len(data)):
                    print(data.loc[i, col])
                    print((data.loc[i, col] - min) / (max - min))
                    data.loc[i, col] = (data.loc[i, col] - min) / (max - min)

            # for col in numeric_columns:
            #     data[data] = scaler.transform(data[col])




            return data

    def initiate_data_transformation(self):
        train_data = pd.read_csv(self.utility_config.training_file_path, dtype={'SeniorCitizen': 'object', 'TotalCharges': 'float64'})
        test_data = pd.read_csv(self.utility_config.testing_file_path, dtype={'SeniorCitizen': 'object', 'TotalCharges': 'float64'})

        train_data_scaled = self.min_max_scaling(train_data, type='train')
        test_data_scaled = self.min_max_scaling(test_data, type='test')

        print('done')