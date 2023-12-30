import os
from datetime import datetime
from source.constant import constant


class TrainingPipelineConfig:

    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_dir: str = os.path.join(constant.ARTIFACT_DIR, timestamp)
        self.timestamp: str = timestamp
        self.target_column: str = constant.TARGET_COLUMN
        self.training_pipeline: str = constant.PIPELINE_NAME
        self.artifact_dir: str = constant.ARTIFACT_DIR


        self.data_ingestion_dir: str = os.path.join(self.artifact_dir, constant.DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, constant.DATA_INGESTION_FEATURE_STORE_DIR, constant.FILE_NAME)
        self.training_file_path: str = os.path.join(self.data_ingestion_dir, constant.DATA_INGESTION_INGESTED_DIR, constant.TRAIN_FILE_NAME)
        self.testing_file_path: str = os.path.join(self.data_ingestion_dir, constant.DATA_INGESTION_INGESTED_DIR, constant.TEST_FILE_NAME)
        self.train_test_split_ratio: float = constant.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = constant.DATA_INGESTION_COLLECTION_NAME