from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    train_rain_H_URL: str
    train_rain_L_URL: str
    local_train_H_path: Path
    local_train_L_path: Path
    unzip_dir_train_H: Path
    unzip_dir_train_L: Path
    train_dir: Path