from source.entity.config_entity import TrainingPipelineConfig
from source.pipeline.training_pipeline import TrainPipeline

if __name__ == '__main__':

    # train_config = TrainingPipelineConfig()
    # print(train_config.__dict__)

    train_pipeline_obj = TrainPipeline()
    train_pipeline_obj.run_training_pipeline()
