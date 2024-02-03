import pandas as pd
from source.exception import ChurnException


class DataValidation:

    def __init__(self, utility_config):
        self.utility_config = utility_config

    def handle_missing_value1(self, data, type):
        try:
            if type == 'train':
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

        except ChurnException as e:
            raise e

    def initiate_data_validation(self):

        train_data = pd.read_csv(self.utility_config.training_file_path, dtype={'SeniorCitizen': 'object'})
        test_data = pd.read_csv(self.utility_config.testing_file_path, dtype={'SeniorCitizen': 'object'})

        self.handle_missing_value1(train_data, type='train')
        self.handle_missing_value1(test_data, type='test')

        print('done')
