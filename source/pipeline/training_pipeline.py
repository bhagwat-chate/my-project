from source.entity.config_entity import TrainingPipelineConfig
from source.component.data_ingestion import DataIngestion
from source.component.data_validation import DataValidation

class TrainPipeline:

    def __init__(self, global_timestamp):
        self.train_config = TrainingPipelineConfig(global_timestamp)

    def start_data_ingestion(self):
        data_ingestion_obj = DataIngestion(self.train_config)
        data_ingestion_obj.initiate_data_ingestion()

    def start_data_validation(self):
        data_validation_obj = DataValidation(self.train_config)
        data_validation_obj.initiate_data_validation()

    def run_training_pipeline(self):
        self.start_data_ingestion()
        self.start_data_validation()
