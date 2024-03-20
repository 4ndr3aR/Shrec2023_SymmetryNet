#!/usr/bin/env python3

from lightning.pytorch.cli import LightningCLI
from src.dataset.shrec2023 import SymmetryDataModule
from src.model.center_n_normals_net import LightingCenterNNormalsNet
import torch


def cli_main():
    torch.set_float32_matmul_precision('high')
    cli = LightningCLI(LightingCenterNNormalsNet, SymmetryDataModule)


if __name__ == "__main__":
    cli_main()
