import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from source.logger import logging
from source.exception import ChurnException
from sklearn.preprocessing import MinMaxScaler
from source.utility.utility import import_csv_file, export_csv_file
import warnings

warnings.filterwarnings('ignore')


class DataTransformation:
    def __init__(self, utility_config):
        self.utility_config = utility_config

    def min_max_scaling(self, data, key):
        """
        Perform Min-Max Scaling on the numerical features of the input data and save scaling details.

        Parameters:
        - data (DataFrame): Input data to be scaled.
        - type (str): Indicates whether the data is for training or testing.

        Returns:
        - DataFrame: Scaled data.
        """

        if key == 'train':
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

        elif key in ['predict', 'test']:

            # Read scaling details from CSV

            scaling_details = pd.read_csv('source/ml/scaling_details.csv')

            for col in data.select_dtypes(include=['float64', 'int64']).columns:

                data[col] = data[col].astype('float64')

                temp = scaling_details[scaling_details['Feature'] == col]

                # Check if temp is empty before accessing its values

                if not temp.empty:

                    min_value = temp.loc[temp.index[0], 'Scalar_Min']

                    max_value = temp.loc[temp.index[0], 'Scalar_Max']

                    for i in range(len(data)):

                        if data.loc[i, col] - min_value != 0.0:
                            data.loc[i, col] = (data.loc[i, col] - min_value) / (max_value - min_value)
                else:
                    print(f"No scaling details found for column '{col}'")

            return data

    def feature_encoding(self, data, target, save_encoder_path=None, load_encoder_path=None):
        try:
            # Map binary categorical variables to 0 and 1 directly
            for col in self.utility_config.dt_binary_class_col:
                data[col] = data[col].map({'No': 0, 'Yes': 1})

            # Encode 'gender' column
            data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})

            if target != '':
                data[self.utility_config.target_column] = data[self.utility_config.target_column].map({'No': 0, 'Yes': 1})

            if save_encoder_path:
                # Target encoding for all categorical columns and save encoder
                encoder = ce.TargetEncoder(cols=self.utility_config.dt_multi_class_col)
                data_encoded = encoder.fit_transform(data[self.utility_config.dt_multi_class_col], data[self.utility_config.target_column])

                # Save encoder object using pickle
                with open(save_encoder_path, 'wb') as f:
                    pickle.dump(encoder, f)

            if load_encoder_path:
                # Load encoder object
                with open(load_encoder_path, 'rb') as f:
                    encoder = pickle.load(f)

                # Transform categorical columns using loaded encoder
                data_encoded = encoder.transform(data[self.utility_config.dt_multi_class_col])

            # Merge encoded columns with original DataFrame
            data = pd.concat([data.drop(columns=self.utility_config.dt_multi_class_col), data_encoded], axis=1)

            return data

        except ChurnException as e:
            raise e

    def oversample_smote(self, data):
        try:
            # Separate features and target variable
            X = data.drop(columns=[self.utility_config.target_column])
            y = data[self.utility_config.target_column]

            # Apply SMOTE
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Combine resampled features and target variable into a new DataFrame
            resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                                        pd.DataFrame(y_resampled, columns=[self.utility_config.target_column])], axis=1)

            return resampled_data

        except ChurnException as e:
            raise e

    def initiate_data_transformation(self, key):

        if key == 'train':

            logging.info('start: train data transformation')

            train_data = import_csv_file(self.utility_config.train_file_name, self.utility_config.dv_train_file_path)
            test_data = import_csv_file(self.utility_config.test_file_name, self.utility_config.dv_test_file_path)

            train_data = self.min_max_scaling(train_data, key)
            test_data = self.min_max_scaling(test_data, key)

            train_data = self.feature_encoding(train_data, target='Churn', save_encoder_path=self.utility_config.dt_multi_class_encoder)
            test_data = self.feature_encoding(test_data, target='Churn', load_encoder_path=self.utility_config.dt_multi_class_encoder)

            train_data = self.oversample_smote(train_data)

            export_csv_file(train_data, self.utility_config.train_file_name, self.utility_config.dt_train_file_path)
            export_csv_file(test_data, self.utility_config.test_file_name, self.utility_config.dt_test_file_path)

            logging.info('complete: train data transformation')

        elif key == 'predict':

            logging.info('start: predict data transformation')

            predict_data = import_csv_file(self.utility_config.predict_file_name, self.utility_config.predict_dv_file_path)
            predict_data = self.min_max_scaling(predict_data, key)
            predict_data = self.feature_encoding(predict_data, target='', load_encoder_path=self.utility_config.dt_multi_class_encoder)

            export_csv_file(predict_data, self.utility_config.predict_file_name, self.utility_config.predict_dt_file_path)

            logging.info('complete: predict data transformation')

