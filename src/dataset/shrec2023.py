import copy
import os

from pathlib import Path

from typing import Optional, Callable, Tuple, List

import lightning
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader

from src.dataset.preprocessing import Shrec2023Transform, UnitSphereNormalization, RandomSampler, ComposeTransform


def default_symmetry_dataset_collate_fn(batch):
    idxs = torch.tensor([item[0] for item in batch])
    points = torch.stack([item[1] for item in batch])
    sym_planes = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True)
    transforms = [item[3] for item in batch]
    return idxs, points, sym_planes, transforms


def default_symmetry_dataset_collate_fn_list_sym(batch):
    idxs = torch.tensor([item[0] for item in batch])
    points = torch.stack([item[1] for item in batch])
    sym_planes = [item[2] for item in batch]
    transforms = [item[3] for item in batch]
    return idxs, points, sym_planes, transforms


class SymmetryDataset(Dataset):
    def __init__(
            self,
            data_source_path: str = "path/to/dataset",
            transform: Optional[Shrec2023Transform] = None,
            has_ground_truth: bool = True,
            max_points: int = 14400,
            #max_points: int = 5000,
            dataset_type: str = 'txt',			# txt or npz or gz or xz (the content also changes with the format)
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
        self.max_points = max_points
        self.dataset_type = dataset_type
        #print(f'{self.data_source_path = }')
        #print(f'{self.dataset_type = }')
        self.files = []
        if self.dataset_type == 'npz':
            self.files = self.find_npz(self.data_source_path)
        print(f'Creating a dataset from: {self.data_source_path} (dataset type: {self.dataset_type}) - each shape has {self.max_points} points')

    def find_npz(self, path) -> list:
        print(f'Looking for npz files in {path}')
        files = Path(path).glob('*.npz')
        files = list(files)
        print(f'Found {len(files)} files')
        return files

    def downsample_or_upsample(self, points, debug=False) -> torch.Tensor:
        if self.max_points - points.shape[0] > 0:
            # Duplicate points
            sampling_indices = np.random.choice(points.shape[0], self.max_points - points.shape[0])
            if debug:
                    print(f'downsample_or_upsample(): {len(sampling_indices) = } - {sampling_indices = }')
            #print(f'{points.shape = } - {type(points) = } - {points = }')
            new_points = points[sampling_indices, : ] 
            points = np.concatenate((points, new_points), axis=0)
        else:
            # sample points
            sampling_indices = np.random.choice(points.shape[0], self.max_points)
            if debug:
                print(f'downsample_or_upsample(): {len(sampling_indices) = } - {sampling_indices = }')
            points = points[sampling_indices, :]
        if debug:
            print(f'downsample_or_upsample() returning: {len(points) = } - {points = }')

        return torch.tensor(points)

    def read_points(self, idx: int) -> torch.Tensor:
        """
        Reads the points with index idx.
        :param idx: Index of points to be read.
        :return: A tensor of shape N x 3 where N is the amount of points.
        """
        if self.dataset_type == 'txt':
            points = torch.tensor(np.loadtxt(os.path.join(self.data_source_path, f"points{idx}.txt")))
        elif self.dataset_type == 'npz':
            if self.files and idx < len(self.files) and self.files[idx].is_file():
                #print(f'{self.files[idx] = }')
                points = torch.tensor(np.load(self.files[idx])['points'])
            else:
                raise FileNotFoundError
            '''
            points = torch.tensor(
                np.load(os.path.join(self.data_source_path, f"points{idx}.npz"))['points']
            )
            '''
        elif self.dataset_type == 'gz':
            print(f'TODO: implement me')
            raise NotImplementedError
        elif self.dataset_type == 'xz':
            print(f'TODO: implement me')
            raise NotImplementedError
        else:
            print(f'Error: unknown dataset format...')
            raise NotImplementedError
        '''
        fd = np.load(os.path.join(self.data_source_path, f"points{idx}.npz"))
        print(f'{type(fd) = }')
        points = fd['points']
        print(f'{type(points) = }')
        print(f'{points.shape = }')
        '''

        if self.max_points != points.shape[0]:
            points = self.downsample_or_upsample(points)

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
        if self.dataset_type == 'txt':
            with open(os.path.join(self.data_source_path, f"points{idx}_sym.txt")) as f:
                n_planes = int(f.readline().strip())
                sym_planes = torch.tensor(np.loadtxt(f))
        elif self.dataset_type == 'npz':
            if self.files and idx < len(self.files) and self.files[idx].is_file():
                #points = torch.tensor(np.load(self.files[idx])['points'])
                gt_fn = str(self.files[idx]).replace('.npz', '-sym.txt')
                #print(f'{gt_fn = }')
                with open(gt_fn, 'r') as f:
                    n_planes = int(f.readline().strip())
                sym_planes = torch.tensor(np.loadtxt(gt_fn, usecols=[i for i in range(1,7)], skiprows=1))
                sym_planes = torch.cat((sym_planes[:, 3:6], sym_planes[:, 0:3]), axis=1)			# give the data the right (backward compatible) format
            else:
                raise FileNotFoundError
            '''
            sym_planes = torch.tensor(np.loadtxt(f, usecols=[i for i in range(1,7)]))
            sym_planes = torch.cat((sym_planes[:, 3:6], sym_planes[:, 0:3]), axis=1)
            '''
        elif self.dataset_type == 'gz':
            print(f'TODO: implement me')
            raise NotImplementedError
        elif self.dataset_type == 'xz':
            print(f'TODO: implement me')
            raise NotImplementedError
        else:
            print(f'Error: unknown dataset format...')
            raise NotImplementedError

        if n_planes == 1:
            sym_planes = sym_planes.unsqueeze(0)
        return sym_planes

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> (int, torch.Tensor, Optional[torch.Tensor], List[Shrec2023Transform]):
        points = self.read_points(idx)
        planes = None
        #print(f'[{idx}] Reading GT of: {self.data_source_path}')

        if self.has_ground_truth:
            planes = self.read_planes(idx)
            #print(f'[{idx}] Planes: {planes}')

        if self.transform is not None:
            idx, points, planes = self.transform(idx, points, planes)
            #print(f'[{idx}] {points = } - {planes = }')

        transform_used = copy.deepcopy(self.transform)
        #print(f'{transform_used = }')
        return idx, points.float(), planes.float(), transform_used


scaler = UnitSphereNormalization()
sampler = RandomSampler(sample_size=1024, keep_copy=False)
# default_transform = ComposeTransform([scaler, sampler])
default_transform = scaler


class SymmetryDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            train_data_path: str = "/path/to/train_data",
            test_data_path: str = "/path/to/test_data",
            predict_data_path: str = "/path/to/predict_data",
            does_predict_has_ground_truths: bool = False,
            batch_size: int = 2,
            transform: Optional[Shrec2023Transform] = None,
            collate_function= None,
            validation_percentage: float = 0.1,
            shuffle: bool = True,
            n_workers: int = 1,
            dataset_type: str = 'txt',
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
        :param dataset_type: Type of the dataset (can be 'txt', 'npz', 'gz', 'xz')
        """
        super().__init__()
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.predict_data_path = predict_data_path
        self.does_predict_has_ground_truths = does_predict_has_ground_truths
        self.batch_size = batch_size
        self.transform = default_transform if transform is None else transform
        self.collate_function = default_symmetry_dataset_collate_fn_list_sym if collate_function is None else collate_function
        self.validation_percentage = validation_percentage
        self.shuffle = shuffle
        self.n_workers = n_workers
        self.dataset_type = dataset_type

    def setup(self, stage: str):
        if stage == "fit":
            dataset_full = SymmetryDataset(
                data_source_path=self.train_data_path,
                transform=self.transform,
                has_ground_truth=True,
                dataset_type=self.dataset_type
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
                has_ground_truth=True,
                dataset_type=self.dataset_type
            )

        if stage == "predict":
            self.predict_dataset = SymmetryDataset(
                data_source_path=self.predict_data_path,
                transform=self.transform,
                has_ground_truth=self.does_predict_has_ground_truths,
                dataset_type=self.dataset_type
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
            self.predict_dataset,
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
    default_transform = ComposeTransform([scaler, sampler])

    dataset = SymmetryDataset(DATA_PATH, default_transform)

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
        transform=default_transform,
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
