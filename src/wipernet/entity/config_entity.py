from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    train_rain_H_URL: str
    train_rain_L_URL: str
    test_rain_H_URL: str
    test_rain_L_URL: str
    local_train_H_path: Path
    local_train_L_path: Path
    local_test_H_path: Path
    local_test_L_path: Path
    unzip_dir_train_H: Path
    unzip_dir_train_L: Path
    unzip_dir_test_H: Path
    unzip_dir_test_L: Path
    train_dir: Path
    test_dir: Path

@dataclass(frozen=True)
class DataPreProcessingConfig:
    output_dir: Path
    train_dir: Path
    test_dir: Path
    params_batch_size: int
    params_image_width: int
    params_image_height: int