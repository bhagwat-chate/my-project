import os
from datetime import datetime

# Common constant
TIME_STAMP = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
TARGET_COLUMN = "Churn"
TRAIN_PIPELINE_NAME: str = "train"
ARTIFACT_DIR: str = "artifact"
TRAINING_FILE_NAME: str = "training_data.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# App constant
APP_HOST = "0.0.0.0"
APP_PORT = 8080

# Database constant
MONGODB_URL_KEY = "MONGODB_KEY"
DATABASE_NAME = 'db-customer-churn'

# Data ingestion constant
DI_TRAIN_COLLECTION_NAME: str = "telco-customer-churn"
DI_DIR_NAME: str = "data_ingestion"
DI_FEATURE_STORE_DIR: str = "feature_store"
DI_INGESTED_DIR: str = "ingested"
DI_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data EDA constant

# Data validation
DV_DIR_NAME: str = "data_validtion"
MANDATORY_COLUMN_LIST = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                       'MonthlyCharges', 'TotalCharges', 'Churn']
MANDATORY_COLUMN_DATA_TYPE = {'gender': 'object', 'SeniorCitizen': 'object', 'Partner': 'object',
                              'Dependents': 'object', 'tenure': 'int64', 'PhoneService': 'object',
                              'MultipleLines': 'object', 'InternetService': 'object', 'OnlineSecurity': 'object',
                              'OnlineBackup': 'object', 'DeviceProtection': 'object', 'TechSupport': 'object',
                              'StreamingTV': 'object', 'StreamingMovies': 'object', 'Contract': 'object',
                              'PaperlessBilling': 'object', 'PaymentMethod': 'object', 'MonthlyCharges': 'float64',
                              'TotalCharges': 'float64', 'Churn': 'object'}
OUTLIER_PARAMS_FILE = "source/ml/outlier_artifact.csv"
OUTLIER_PARAMS = {}


# Data transformation constant
DT_DIR_NAME: str = "data_transformation"
MULTI_CLASS_ENCODER = "source/ml/multi_class_encoder.pkl"
MULTI_CLASS_COL = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
BINARY_CLASS_COL = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

# Model training constant
MODEL_PATH = "source/ml/artifact"
FINAL_MODEL = "source/ml/final_model"

# Model evaluation constant

# Model push constant

###############################
PREDICT_PIPELINE_NAME: str = "predict"
PREDICT_FILE_NAME: str = "prediction_data.csv"
PREDICT_DI_COLLECTION_NAME: str = "predict-telco-customer-churn"
