from source.entity.config_entity import TrainingPipelineConfig
from source.pipeline.training_pipeline import TrainPipeline
from source.utility.utility import generate_global_timestamp
from source.logger import setup_logger

if __name__ == '__main__':

    global_timestamp = generate_global_timestamp()

    # Call the setup_logger function to configure the logger
    setup_logger(global_timestamp)

    # train_config = TrainingPipelineConfig(global_timestamp)
    # print(train_config.__dict__)

    train_pipeline_obj = TrainPipeline(global_timestamp)
    train_pipeline_obj.run_training_pipeline()
