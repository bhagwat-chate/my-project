from source.entity.config_entity import PipelineUtilityConfig
from source.component.data_ingestion import DataIngestion
from source.component.data_validation import DataValidation
from source.component.data_transformation import DataTransformation
from source.component.model_train import ModelTrain


class DataPipeline:

    def __init__(self, global_timestamp):
        self.utility_config = PipelineUtilityConfig(global_timestamp)

    def start_data_ingestion(self, key):
        data_ingestion_obj = DataIngestion(self.utility_config)
        data_ingestion_obj.initiate_data_ingestion(key)

    def start_data_validation(self):
        data_validation_obj = DataValidation(self.utility_config)
        data_validation_obj.initiate_data_validation()

    def start_data_transformation(self):
        data_transformation_obj = DataTransformation(self.utility_config)
        data_transformation_obj.initiate_data_transformation()

    def start_model_training(self):
        model_training_obj = ModelTrain(self.utility_config)
        model_training_obj.initiate_model_training()

    def run_training_pipeline(self):
        self.start_data_ingestion('train')
        self.start_data_validation()
        self.start_data_transformation()
        self.start_model_training()

    def run_prediction_pipeline(self):
        self.start_data_ingestion('predict')
        self.start_data_validation()
        self.start_data_transformation()