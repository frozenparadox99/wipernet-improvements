from wipernet.components.data_preprocessing import DataPreProcessing
from wipernet.config.configuration import ConfigurationManager
from wipernet.components.data_ingestion import DataIngestion
from wipernet.components.GAN import GAN
from wipernet import logger


STAGE_NAME = "Data pre-processing stage"

class DataPreProcessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()

        preprocessor = DataPreProcessing(data_preprocessing_config)
        preprocessor.preprocess()
        # preprocessor.save_preprocessed_images()
        gan = GAN(epochs=3, path='./', mode='train', output_path='artifacts/output')
        gan.fit(preprocessor.train_dataset, 3)
        


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreProcessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e