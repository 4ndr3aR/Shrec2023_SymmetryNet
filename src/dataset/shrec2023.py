import copy
import os
from typing import Optional, Callable, Tuple, List

import lightning
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader

from src.dataset.preprocessing import Shrec2023Transform


def default_symmetry_dataset_collate_fn(batch):
    idxs = torch.tensor([item[0] for item in batch])
    points = torch.stack([item[1] for item in batch])
    sym_planes = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True)
    transforms = [item[3] for item in batch]
    return idxs, points, sym_planes, transforms


class SymmetryDataset(Dataset):
    def __init__(
            self,
            data_source_path: str = "path/to/dataset",
            transform: Optional[Shrec2023Transform] = None,
            has_ground_truth: bool = True
    ):
        """
        Dataset used for a track of SHREC2023. It contains a set of 3D points
        and planes that represent reflective symmetries.
        :param data_source_path: Path to folder that contains the points and symmetries.
        :param transform: Transform applied to dataset item.
        """
        self.data_source_path = data_source_path
        self.transform = transform
        self.length = len(os.listdir(self.data_source_path)) // 2
        self.has_ground_truth = has_ground_truth

    def read_points(self, idx: int) -> torch.Tensor:
        """
        Reads the points with index idx.
        :param idx: Index of points to be read.
        :return: A tensor of shape N x 3 where N is the amount of points.
        """
        points = torch.tensor(
            np.loadtxt(os.path.join(self.data_source_path, f"points{idx}.txt"))
        )
        return points

    def read_planes(self, idx: int) -> torch.Tensor:
        """
        Read symmetry planes from file with its first line being the number of symmetry planes
        and the rest being the symmetry planes.
        :param idx: The idx of the syms to reads
        :return: A tensor of planes represented by their normals and points. N x 6 where
        N is the amount of planes and 6 because the first 3 elements
        are the normal and the last 3 are the point.
        """
        with open(os.path.join(self.data_source_path, f"points{idx}_sym.txt")) as f:
            n_planes = int(f.readline().strip())
            sym_planes = torch.tensor(np.loadtxt(f))
        if n_planes == 1:
            sym_planes = sym_planes.unsqueeze(0)
        return sym_planes

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> (int, torch.Tensor, Optional[torch.Tensor], List[Shrec2023Transform]):
        points = self.read_points(idx)
        planes = None

        if self.has_ground_truth:
            planes = self.read_planes(idx)

        if self.transform is not None:
            idx, points, planes = self.transform(idx, points, planes)

        transform_used = copy.deepcopy(self.transform)
        return idx, points.float(), planes.float(), transform_used


class SymmetryDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            train_data_path: str = "/path/to/train_data",
            test_data_path: str = "/path/to/test_data",
            predict_data_path: str = "/path/to/predict_data",
            does_predict_has_ground_truths: bool = False,
            batch_size: int = 2,
            transform: Optional[Shrec2023Transform] = None,
            collate_function: Callable[
                [List[Tuple[int, torch.Tensor, torch.Tensor, Shrec2023Transform]]],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Shrec2023Transform]]
            ] = default_symmetry_dataset_collate_fn,
            validation_percentage: float = 0.1,
            shuffle: bool = True,
            n_workers: int = 1,
    ):
        """
        Data module designed to load Shrec2023 symmetry dataset.
        :param train_data_path: Path to train data it must contain points and planes.
        :param test_data_path:  Path to test data it must contain points and planes.
        :param predict_data_path: Path to predict it must contain points and can contain planes
        :param does_predict_has_ground_truths : Boolean flag to indicate if the predict data has
        ground truths.
        :param batch_size: Batch size used on all dataloaders.
        :param transform: Transform applied to all dataloaders.
        :param collate_function: Function used to batch the items from symmetry dataset.
        :param validation_percentage: Percentage used for validation data.
        :param shuffle: True if you want to shuffle the train dataloader every epoch.
        :param n_workers: Amount of workers used for loading data into RAM.
        """
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.predict_data_path = predict_data_path
        self.does_predict_has_ground_truths = does_predict_has_ground_truths
        self.batch_size = batch_size
        self.transform = transform
        self.collate_function = collate_function
        self.validation_percentage = validation_percentage
        self.shuffle = shuffle
        self.n_workers = n_workers

    def setup(self, stage: str):
        if stage == "fit":
            dataset_full = SymmetryDataset(
                data_source_path=self.train_data_path,
                transform=self.transform,
                has_ground_truth=True
            )

            proportions = [1 - self.validation_percentage, self.validation_percentage]
            lengths = [int(p * len(dataset_full)) for p in proportions]
            lengths[-1] = len(dataset_full) - sum(lengths[:-1])

            self.train_dataset, self.validation_dataset = random_split(
                dataset_full, lengths
            )

        if stage == "test":
            self.test_dataset = SymmetryDataset(
                data_source_path=self.test_data_path,
                transform=self.transform,
                has_ground_truth=True
            )

        if stage == "predict":
            self.predict_dataset = SymmetryDataset(
                data_source_path=self.predict_data_path,
                transform=self.transform,
                has_ground_truth=self.does_predict_has_ground_truths
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.n_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collate_function,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
        )


if __name__ == "__main__":
    from src.dataset.preprocessing import *

    DATA_PATH = "/data/shrec_2023/benchmark-train"

    scaler = UnitSphereNormalization()
    sampler = RandomSampler(sample_size=3, keep_copy=True)
    compose_transform = ComposeTransform([scaler, sampler])

    dataset = SymmetryDataset(DATA_PATH, compose_transform)

    example_idx, example_points, example_syms, example_tr = dataset[0]
    print("transformed", example_idx, example_points[0, :], example_syms[0, :], example_tr)
    ridx, rpoints, rsyms = example_tr.inverse_transform(example_idx, example_points, example_syms)
    print("inverse_transform", ridx, rpoints[0, :], rsyms[0, :])
    ridx, rpoints, rsyms = example_tr.transform(ridx, rpoints, rsyms)
    print("transformed 2", ridx, rpoints[0, :], rsyms[0, :])

    datamodule = SymmetryDataModule(
        train_data_path=DATA_PATH,
        test_data_path=DATA_PATH,
        predict_data_path=DATA_PATH,
        does_predict_has_ground_truths=True,
        batch_size=1,
        transform=compose_transform,
        collate_function=default_symmetry_dataset_collate_fn,
        validation_percentage=0.1,
        shuffle=True,
        n_workers=1,
    )
    datamodule.setup("fit")

    train_dataloader = datamodule.train_dataloader()
    print(len(train_dataloader))

    batch = next(iter(train_dataloader))

    for x in batch:
        if isinstance(x, torch.Tensor):
            print(x.shape)
        else:
            print(len(x))
