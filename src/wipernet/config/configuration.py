from wipernet.constants import *
import os
from pathlib import Path
from wipernet.utils.common import read_yaml, create_directories
from wipernet.entity.config_entity import (DataIngestionConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            train_rain_H_URL=config.train_rain_H_URL,
            train_rain_L_URL=config.train_rain_L_URL,
            local_train_H_path=config.local_train_H_path,
            local_train_L_path=config.local_train_L_path,
            unzip_dir_train_H=config.unzip_dir_train_H, 
            unzip_dir_train_L=config.unzip_dir_train_L, 
        )

        return data_ingestion_config