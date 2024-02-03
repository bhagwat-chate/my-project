import os
from datetime import datetime

# Common constant
TIME_STAMP = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
TARGET_COLUMN = "Churn"
PIPELINE_NAME: str = "training_pipeline"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "training_data.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# App constant
APP_HOST = "0.0.0.0"
APP_PORT = 8080

# Database constant
# MONGODB_URL_KEY = "mongodb+srv://datawave05:I9JhcMBUfP82kulr@datawave.aignamw.mongodb.net/?retryWrites=true&w=majority"
MONGODB_URL_KEY = "MONGODB_KEY"
DATABASE_NAME = 'db-customer-churn'

# Data ingestion constant
DATA_INGESTION_COLLECTION_NAME: str = "telco-customer-churn"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Data EDA constant

# Data validation
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

# Model training constant

# Model evaluation constant

# Model push constant

