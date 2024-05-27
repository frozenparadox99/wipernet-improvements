from wipernet.config.configuration import ConfigurationManager
from wipernet.components.data_ingestion import DataIngestion
from wipernet import logger


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        # data_ingestion.download_file(data_ingestion_config.train_rain_H_URL, data_ingestion_config.local_train_H_path)
        # data_ingestion.download_file(data_ingestion_config.train_rain_L_URL, data_ingestion_config.local_train_L_path)
        # data_ingestion.download_file(data_ingestion_config.test_rain_H_URL, data_ingestion_config.local_test_H_path)
        # data_ingestion.download_file(data_ingestion_config.test_rain_L_URL, data_ingestion_config.local_test_L_path)
        # data_ingestion.extract_zip_file(data_ingestion_config.unzip_dir_train_H, data_ingestion_config.local_train_H_path)
        # data_ingestion.extract_zip_file(data_ingestion_config.unzip_dir_train_L, data_ingestion_config.local_train_L_path)
        # data_ingestion.extract_zip_file(data_ingestion_config.unzip_dir_test_H, data_ingestion_config.local_test_H_path)
        # data_ingestion.extract_zip_file(data_ingestion_config.unzip_dir_test_L, data_ingestion_config.local_test_L_path)
        
        # data_ingestion.combine_directories(data_ingestion_config.train_dir, data_ingestion_config.root_dir)
        data_ingestion.combine_directories(data_ingestion_config.test_dir, data_ingestion_config.unzip_dir_test_H)




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e