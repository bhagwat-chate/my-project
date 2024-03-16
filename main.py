from source.entity.config_entity import PipelineUtilityConfig
from source.pipeline.pipeline import DataPipeline
from source.utility.utility import generate_global_timestamp
from source.logger import setup_log_timestamp
from source.logger import logging


if __name__ == '__main__':

    global_timestamp = generate_global_timestamp()

    # Call the setup_logger function to configure the logger
    setup_log_timestamp(global_timestamp)

    pipeline_obj = DataPipeline(global_timestamp)

    # logging.info("START: MODEL TRAINING")
    # train_config = PipelineUtilityConfig(global_timestamp)
    # print(train_config.__dict__)

    pipeline_obj.run_training_pipeline()
    print("model training complete")
    logging.info("END: MODEL TRAINING")


    logging.info("START: MODEL PREDICTION")

    pipeline_obj.run_prediction_pipeline()

    logging.info("END: MODEL PREDICTION")
