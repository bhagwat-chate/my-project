import os
from source.constant import constant


class TrainingPipelineConfig:

    def __init__(self, global_timestamp):
        self.artifact_dir: str = os.path.join(constant.ARTIFACT_DIR, str(global_timestamp))
        self.global_timestamp: str = global_timestamp
        self.target_column: str = constant.TARGET_COLUMN
        self.training_pipeline: str = constant.PIPELINE_NAME

        self.data_ingestion_dir: str = os.path.join(self.artifact_dir, constant.DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, constant.DATA_INGESTION_FEATURE_STORE_DIR, constant.FILE_NAME)
        self.training_file_path: str = os.path.join(self.data_ingestion_dir, constant.DATA_INGESTION_INGESTED_DIR, constant.TRAIN_FILE_NAME)
        self.testing_file_path: str = os.path.join(self.data_ingestion_dir, constant.DATA_INGESTION_INGESTED_DIR, constant.TEST_FILE_NAME)
        self.train_test_split_ratio: float = constant.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.mongodb_url_key = os.environ[constant.MONGODB_URL_KEY]
        self.database_name: str = constant.DATABASE_NAME
        self.collection_name: str = constant.DATA_INGESTION_COLLECTION_NAME

        # data validation
        self.mandatory_col_list = constant.MANDATORY_COLUMN_LIST
        self.mandatory_col_data_type = constant.MANDATORY_COLUMN_DATA_TYPE
        self.outlier_params_file  = constant.OUTLIER_PARAMS_FILE
        self.outlier_params = constant.OUTLIER_PARAMS
        self.dv_train_file_path: str = os.path.join(self.artifact_dir, constant.DV_DIR_NAME, constant.TRAIN_FILE_NAME)
        self.dv_test_file_path: str = os.path.join(self.artifact_dir, constant.DV_DIR_NAME, constant.TEST_FILE_NAME)

        # data transformation
        self.multi_class_encoder = constant.MULTI_CLASS_ENCODER
        self.multi_class_col = constant.MULTI_CLASS_COL
        self.binary_class_col = constant.BINARY_CLASS_COL
        self.dt_train_file_path: str = os.path.join(self.artifact_dir, constant.DT_DIR_NAME, constant.TRAIN_FILE_NAME)
        self.dt_test_file_path: str = os.path.join(self.artifact_dir, constant.DT_DIR_NAME, constant.TEST_FILE_NAME)

        self.model_path = constant.MODEL_PATH



