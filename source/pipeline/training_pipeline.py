from source.entity.config_entity import TrainingPipelineConfig
from source.component.data_ingestion import DataIngestion

class TrainPipeline:

    def __init__(self):
        self.train_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        data_ingestion_obj = DataIngestion(self.train_config)
        data_ingestion_obj.initiate_data_ingestion()


    def run_training_pipeline(self):
        self.start_data_ingestion()