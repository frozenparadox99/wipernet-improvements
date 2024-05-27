import os
import shutil
import urllib.request as request
import re
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

    def combine_directories(self, images_dir, main_dir):
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
        ground_truth_counter = 1
        degraded_counter = 1
        
        
        for root, dirs, files in os.walk(main_dir):
            for sub_dir in dirs:
                image_pairs = {}
                sub_dir_path = os.path.join(root, sub_dir)
                for file_name in os.listdir(sub_dir_path):
                    if file_name.endswith('.png'):
                        match = re.match(r'(norain|rain)-(\d+)\.png', file_name)
                        if match:
                            prefix, number = match.groups()
                            if number not in image_pairs:
                                image_pairs[number] = {}
                            image_pairs[number][prefix] = os.path.join(sub_dir_path, file_name)
                        
                for counter, paths in sorted(image_pairs.items()):
                    if 'norain' in paths:
                        dest_file_name = f"ground_truth_{ground_truth_counter}.png"
                        ground_truth_counter += 1
                        dest_file_path = os.path.join(images_dir, dest_file_name)
                        shutil.copy(paths['norain'], dest_file_path)

                    if 'rain' in paths:
                        dest_file_name = f"degraded_{degraded_counter}.png"
                        degraded_counter += 1
                        dest_file_path = os.path.join(images_dir, dest_file_name)
                        shutil.copy(paths['rain'], dest_file_path)


                        
                        
        logger.info(f"Processed {ground_truth_counter - 1} ground truth images and {degraded_counter - 1} degraded images.")