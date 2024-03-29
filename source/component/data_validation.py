import os
import pandas as pd
import numpy as np
from pandas import DataFrame
from source.utility.utility import export_csv_file, import_csv_file
from source.exception import ChurnException
from source.logger import logging


class DataValidation:

    def __init__(self, utility_config):
        self.utility_config = utility_config
        self.outlier_params = {}

    def handle_missing_value(self, data, key):
        try:
            if key == 'train':
                # Impute numerical columns with median
                numerical_columns = data.select_dtypes(include=['number']).columns
                numerical_imputation_values = data[numerical_columns].median()
                data[numerical_columns] = data[numerical_columns].fillna(numerical_imputation_values)

                # Impute categorical columns with mode
                categorical_columns = data.select_dtypes(include=['object']).columns
                categorical_imputation_values = data[categorical_columns].mode().iloc[0]
                data[categorical_columns] = data[categorical_columns].fillna(categorical_imputation_values)

                # Save imputation values to CSV
                imputation_values = pd.concat([numerical_imputation_values, categorical_imputation_values])
                imputation_values.to_csv('source/ml/imputation_values.csv', header=['imputation_value'])

            else:
                # Read imputation values from CSV
                imputation_values = pd.read_csv('source/ml/imputation_values.csv', index_col=0)['imputation_value']

                # Impute numerical columns with the corresponding imputation values
                numerical_columns = data.select_dtypes(include=['number']).columns
                data[numerical_columns] = data[numerical_columns].fillna(imputation_values[numerical_columns])

                # Impute categorical columns with the corresponding imputation values
                categorical_columns = data.select_dtypes(include=['object']).columns
                data[categorical_columns] = data[categorical_columns].fillna(imputation_values[categorical_columns].iloc[0])

            return data

        except ChurnException as e:
            raise e

    def outlier_detection_and_handle(self, data, key):

        if key == 'train':
            for column_name in data.select_dtypes(include=['number']).columns:
                # Calculate the IQR (Interquartile Range)
                Q1 = data[column_name].quantile(0.25)
                Q3 = data[column_name].quantile(0.75)
                IQR = Q3 - Q1

                # Define the lower and upper bounds for outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Store the outlier parameters in the dictionary
                self.outlier_params[column_name] = {'Q1': Q1, 'Q3': Q3, 'IQR': IQR}

                # Detect outliers
                outliers_mask = (data[column_name] < lower_bound) | (data[column_name] > upper_bound)

                # Handle outliers using Logarithmic transformation
                data.loc[outliers_mask, column_name] = np.log1p(data.loc[outliers_mask, column_name])

            # Save outlier parameters to CSV during training
            outlier_params_df = pd.DataFrame(self.outlier_params)
            outlier_params_df.to_csv(self.utility_config.dv_outlier_params_file, index=False)

        elif key in ['predict', 'test']:

            # Load outlier parameters from CSV during test or prediction
            outlier_params_df = pd.read_csv(self.utility_config.dv_outlier_params_file)
            self.outlier_params = outlier_params_df.to_dict(orient='list')

            for column_name in data.select_dtypes(include=['number']).columns:
                if column_name in self.outlier_params:
                    # Use the stored Q1, Q3, IQR values for handling outliers
                    Q1 = self.outlier_params[column_name][0]
                    Q3 = self.outlier_params[column_name][1]
                    IQR = self.outlier_params[column_name][2]

                    # Define the lower and upper bounds for outliers
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # Detect outliers
                    outliers_mask = (data[column_name] < lower_bound) | (data[column_name] > upper_bound)

                    # Handle outliers using Logarithmic transformation
                    data.loc[outliers_mask, column_name] = np.log1p(data.loc[outliers_mask, column_name])

        return data

    def export_data_file(self, train_data, test_data):
        try:

            dir_path = os.path.dirname(self.utility_config.dv_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_data.to_csv(self.utility_config.dv_train_file_path, index=False, header=True)
            test_data.to_csv(self.utility_config.dv_test_file_path, index=False, header=True)

            logging.info("data validation files exported")

        except ChurnException as e:
            raise e

    def initiate_data_validation(self, key):

        if key == 'train':

            train_data = import_csv_file(self.utility_config.train_file_name, self.utility_config.di_train_file_path)
            test_data = import_csv_file(self.utility_config.test_file_name, self.utility_config.di_test_file_path)

            train_data = self.outlier_detection_and_handle(train_data, key)
            test_data = self.outlier_detection_and_handle(test_data, key)

            train_data = self.handle_missing_value(train_data, key)
            test_data = self.handle_missing_value(test_data, key)

            # self.export_data_file(train_data, test_data)
            export_csv_file(train_data, self.utility_config.train_file_name, self.utility_config.dv_train_file_path)
            export_csv_file(test_data, self.utility_config.test_file_name, self.utility_config.dv_test_file_path)

        else:
            predict_data = import_csv_file(self.utility_config.predict_file_name, self.utility_config.predict_di_file_path)
            predict_data = self.outlier_detection_and_handle(predict_data, key)
            predict_data = self.handle_missing_value(predict_data, key)

            export_csv_file(predict_data, self.utility_config.predict_file_name, self.utility_config.predict_dv_file_path)
            print('complete: predict data validation')
