from wipernet import logger
from wipernet.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from wipernet.pipeline.stage_02_data_preprocessing import DataPreProcessingPipeline


STAGE_NAME = "Data ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# STAGE_NAME = "Data pre-processing stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
#    preprocessor = DataPreProcessingPipeline()
#    preprocessor.main()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e