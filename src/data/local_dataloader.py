from typing import Optional
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from src.data.components.LocalDataset import LocalDataset
import numpy as np


class LocalDataModule(LightningDataModule):
    def __init__(
        self,
        ids_path: str = None,
        data_dir: str = None,
        has_noise: bool = False,
        noise_level: float = 0.01,
        has_outliers: bool = False,
        outlier_ratio: float = 0.1,
        has_transform: bool = True,
        batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        train_ids_path = f"{ids_path}/train_ids.npy"
        val_ids_path = f"{ids_path}/val_ids.npy"
        test_ids_path = f"{ids_path}/test_ids.npy"
        self.train_id_list = np.load(train_ids_path)
        self.val_id_list = np.load(val_ids_path)
        self.test_id_list = np.load(test_ids_path)
        self.ids_path = ids_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.has_noise = has_noise
        self.noise_level = noise_level
        self.has_outliers = has_outliers
        self.outlier_ratio = outlier_ratio
        self.has_transform = has_transform

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = LocalDataset(
            id_list=self.train_id_list,
            data_dir=self.data_dir,
            has_noise=self.has_noise,
            noise_level=self.noise_level,
            has_outliers=self.has_outliers,
            outlier_ratio=self.outlier_ratio,
            has_transform=self.has_transform,
        )
        self.val_dataset = LocalDataset(
            id_list=self.val_id_list,
            data_dir=self.data_dir,
            has_noise=self.has_noise,
            noise_level=self.noise_level,
            has_outliers=False,
            outlier_ratio=self.outlier_ratio,
            has_transform=self.has_transform,
        )
        self.test_dataset = LocalDataset(
            id_list=self.test_id_list,
            data_dir=self.data_dir,
            has_noise=self.has_noise,
            noise_level=self.noise_level,
            has_outliers=self.has_outliers,
            outlier_ratio=self.outlier_ratio,
            has_transform=self.has_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    _ = LocalDataModule()
