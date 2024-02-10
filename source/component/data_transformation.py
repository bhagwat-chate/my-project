import os
import os.path
import pickle
import pandas as pd
import category_encoders as ce
from source.logger import logging
from source.exception import ChurnException
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

    def feature_encoding(self, data, target, save_encoder_path=None, load_encoder_path=None):
        try:
            # Map binary categorical variables to 0 and 1 directly
            for col in self.utility_config.binary_class_col:
                data[col] = data[col].map({'No': 0, 'Yes': 1})

            # Encode 'gender' column
            data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})

            if target != '':
                data[self.utility_config.target_column] = data[self.utility_config.target_column].map({'No': 0, 'Yes': 1})

            if save_encoder_path:
                # Target encoding for all categorical columns and save encoder
                encoder = ce.TargetEncoder(cols=self.utility_config.multi_class_col)
                data_encoded = encoder.fit_transform(data[self.utility_config.multi_class_col], data[self.utility_config.target_column])

                # Save encoder object using pickle
                with open(save_encoder_path, 'wb') as f:
                    pickle.dump(encoder, f)

            if load_encoder_path:
                # Load encoder object
                with open(load_encoder_path, 'rb') as f:
                    encoder = pickle.load(f)

                # Transform categorical columns using loaded encoder
                data_encoded = encoder.transform(data[self.utility_config.multi_class_col])

            # Merge encoded columns with original DataFrame
            data = pd.concat([data.drop(columns=self.utility_config.multi_class_col), data_encoded], axis=1)

            # Display DataFrame
            print(data)

            return data

        except ChurnException as e:
            raise e

    def export_data_file(self, train_data, test_data):
        try:

            dir_path = os.path.dirname(self.utility_config.dt_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_data.to_csv(self.utility_config.dt_train_file_path, index=False, header=True)
            test_data.to_csv(self.utility_config.dt_test_file_path, index=False, header=True)

            logging.info("data transformation files exported")

        except ChurnException as e:
            raise e

    def initiate_data_transformation(self):
        train_data = pd.read_csv(self.utility_config.dv_train_file_path, dtype={'SeniorCitizen': 'object', 'TotalCharges': 'float64'})
        test_data = pd.read_csv(self.utility_config.dv_test_file_path, dtype={'SeniorCitizen': 'object', 'TotalCharges': 'float64'})

        train_data = self.min_max_scaling(train_data, type='train')
        test_data = self.min_max_scaling(test_data, type='test')

        train_data = self.feature_encoding(train_data, target='Churn', save_encoder_path=self.utility_config.multi_class_encoder)
        test_data = self.feature_encoding(test_data, target='', load_encoder_path=self.utility_config.multi_class_encoder)

        self.export_data_file(train_data, test_data)

        print('done')
