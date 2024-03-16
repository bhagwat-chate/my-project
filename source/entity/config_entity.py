import os
from source.constant import constant


class PipelineUtilityConfig:

    def __init__(self, global_timestamp):

        # generic constant
        self.artifact_dir: str = os.path.join(constant.ARTIFACT_DIR, str(global_timestamp))
        self.global_timestamp: str = global_timestamp
        self.target_column: str = constant.TARGET_COLUMN

        self.train_file_name = constant.TRAIN_FILE_NAME
        self.test_file_name = constant.TEST_FILE_NAME

        # data ingestion
        self.train_di_dir_name: str = os.path.join(self.artifact_dir, constant.TRAIN_PIPELINE_NAME, constant.DI_DIR_NAME)
        self.train_di_feature_store_file_path: str = os.path.join(self.train_di_dir_name, constant.DI_FEATURE_STORE_DIR, constant.TRAINING_FILE_NAME)
        self.train_di_feature_store_file_path: str = os.path.join(self.train_di_dir_name, constant.DI_FEATURE_STORE_DIR)
        self.di_feature_store_file_name = constant.TRAINING_FILE_NAME
        self.di_train_file_path: str = os.path.join(self.train_di_dir_name, constant.DI_INGESTED_DIR)
        self.di_test_file_path: str = os.path.join(self.train_di_dir_name, constant.DI_INGESTED_DIR)
        self.train_test_split_ratio: float = constant.DI_TRAIN_TEST_SPLIT_RATIO
        self.mongodb_url_key = os.environ[constant.MONGODB_URL_KEY]
        self.database_name: str = constant.DATABASE_NAME
        self.train_di_collection_name: str = constant.DI_TRAIN_COLLECTION_NAME
        self.col_drop_in_clean = constant.COL_DROP_IN_CLEAN

        # data validation
        self.dv_mandatory_col_list = constant.MANDATORY_COLUMN_LIST
        self.dv_mandatory_col_data_type = constant.MANDATORY_COLUMN_DATA_TYPE
        self.dv_outlier_params_file = constant.OUTLIER_PARAMS_FILE
        self.dv_outlier_params = constant.OUTLIER_PARAMS
        self.dv_train_file_path: str = os.path.join(self.artifact_dir, constant.TRAIN_PIPELINE_NAME, constant.DV_DIR_NAME)
        self.dv_test_file_path: str = os.path.join(self.artifact_dir, constant.TRAIN_PIPELINE_NAME, constant.DV_DIR_NAME)

        # data transformation
        self.dt_multi_class_encoder = constant.MULTI_CLASS_ENCODER
        self.dt_multi_class_col = constant.MULTI_CLASS_COL
        self.dt_binary_class_col = constant.BINARY_CLASS_COL
        self.dt_train_file_path: str = os.path.join(self.artifact_dir, constant.TRAIN_PIPELINE_NAME, constant.DT_DIR_NAME)
        self.dt_test_file_path: str = os.path.join(self.artifact_dir, constant.TRAIN_PIPELINE_NAME, constant.DT_DIR_NAME)

        # model train and evaluate
        self.mt_model_path = constant.MODEL_PATH
        self.mt_final_model = constant.FINAL_MODEL

        # PREDICTION CONSTANT
        # data ingestion
        self.predict_di_dir_name: str = os.path.join(self.artifact_dir, constant.PREDICT_PIPELINE_NAME, constant.DI_DIR_NAME)
        # self.predict_di_feature_store_file_path: str = os.path.join(self.predict_di_dir_name, constant.DI_FEATURE_STORE_DIR, constant.PREDICT_FEA_STORE_FILE_NAME)
        self.predict_di_feature_store_file_path: str = os.path.join(self.predict_di_dir_name, constant.DI_FEATURE_STORE_DIR)
        self.predict_di_feature_store_file_name = constant.PREDICT_FEA_STORE_FILE_NAME
        self.predict_di_file_path: str = os.path.join(self.predict_di_dir_name, constant.DI_INGESTED_DIR)
        self.predict_di_collection_name = constant.PREDICT_DI_COLLECTION_NAME

        # data validation
        self.predict_dv_file_path: str = os.path.join(self.artifact_dir, constant.PREDICT_PIPELINE_NAME, constant.DV_DIR_NAME)
        self.predict_file_name = constant.PREDICT_FILE

        # data transformation
        self.predict_dt_file_path: str = os.path.join(self.artifact_dir, constant.PREDICT_PIPELINE_NAME, constant.DT_DIR_NAME)


