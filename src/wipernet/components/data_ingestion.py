import os
import urllib.request as request
import zipfile
from wipernet import logger
from wipernet.utils.common import get_size
from wipernet.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_file(self, src_url, config_file):
        if not os.path.exists(config_file):
            filename, headers = request.urlretrieve(
                url = src_url,
                filename = config_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(config_file))}")  


    
    def extract_zip_file(self, unzip_dir, config_file):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(config_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)