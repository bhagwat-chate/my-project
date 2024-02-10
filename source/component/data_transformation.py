import os
import pandas as pd
import os.path
from pandas import DataFrame
from pymongo.mongo_client import MongoClient
from source.logger import logging
from source.exception import ChurnException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


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

            for col in data.select_dtypes(include=['float64', 'int64']).columns:

                data[col] = data[col].astype('float64')

                temp = scaling_details[scaling_details['Feature'] == col]

                # Check if temp is empty before accessing its values

                if not temp.empty:

                    min = temp.loc[temp.index[0], 'Scalar_Min']

                    max = temp.loc[temp.index[0], 'Scalar_Max']

                    for i in range(len(data)):

                        if data.loc[i, col] - min != 0.0:
                            data.loc[i, col] = (data.loc[i, col] - min) / (max - min)
                else:
                    print(f"No scaling details found for column '{col}'")

            return data

    def initiate_data_transformation(self):
        train_data = pd.read_csv(self.utility_config.training_file_path, dtype={'SeniorCitizen': 'object', 'TotalCharges': 'float64'})
        test_data = pd.read_csv(self.utility_config.testing_file_path, dtype={'SeniorCitizen': 'object', 'TotalCharges': 'float64'})

        train_data_scaled = self.min_max_scaling(train_data, type='train')
        test_data_scaled = self.min_max_scaling(test_data, type='test')

        print('done')