#!/usr/bin/env python3
import pathlib
from argparse import ArgumentParser

import lightning
import torch
from lightning import Trainer
import polyscope as ps

import sys
sys.path.insert(0, '..')

from src.dataset.preprocessing import ComposeTransform, RandomSampler, UnitSphereNormalization
from src.metrics.mAP import get_mean_average_precision, get_match_sequence
from src.model.center_n_normals_net import LightingCenterNNormalsNet
from src.dataset.shrec2023 import SymmetryDataModule, default_symmetry_dataset_collate_fn, \
    default_symmetry_dataset_collate_fn_list_sym
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader, Subset
from lightning.pytorch.callbacks import EarlyStopping

from src.utils.plane import SymmetryPlane


def visualize_prediction(pred_planes, input_points, real_planes):
    """
    :param pred_planes: N x 7
    :param input_points: S x 3
    :param real_planes: M x 6
    """
    # Create symmetryPlane Objs
    original_symmetries = [
        SymmetryPlane(
            normal=real_planes[idx, 0:3].detach().numpy(),
            point=real_planes[idx, 3::].detach().numpy(),
        )
        for idx in range(real_planes.shape[0])
    ]

    predicted_symmetries = [
        SymmetryPlane(
            normal=pred_planes[idx, 0:3].detach().numpy(),
            point=pred_planes[idx, 3::].detach().numpy(),
        )
        for idx in range(pred_planes.shape[0])
    ]

    # Visualize
    ps.init()
    ps.set_ground_plane_mode("none")  # set +Z as up direction
    ps.remove_all_structures()

    ps.register_point_cloud("original pcd", input_points.detach().numpy())

    for idx, sym_plane in enumerate(original_symmetries):
        ps.register_surface_mesh(
            f"original_sym_plane_{idx}",
            sym_plane.coords,
            sym_plane.trianglesBase,
            enabled=True,
            transparency=0.5
        )

    for idx, sym_plane in enumerate(predicted_symmetries):
        ps.register_surface_mesh(
            f"predicted_sym_plane_{idx}",
            sym_plane.coords,
            sym_plane.trianglesBase,
            enabled=False,
            transparency=0.5
        )

    ps.show()


parser = ArgumentParser()
parser.add_argument("--data_path", type=pathlib.Path, required=True)
parser.add_argument("--model_path", type=pathlib.Path, required=True)
parser.add_argument("--figure_idx", type=int, required=True)
parser.add_argument("--n_workers", type=int, required=False, default=1)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    DATA_PATH = args["data_path"]
    MODEL_PATH = args["model_path"]
    N_WORKERS = args["n_workers"]
    FIG_IDX = args["figure_idx"]

    COLLATE_FN = default_symmetry_dataset_collate_fn_list_sym

    model = LightingCenterNNormalsNet.load_from_checkpoint(MODEL_PATH)
    data_module = SymmetryDataModule(
        test_data_path=DATA_PATH,
        predict_data_path=DATA_PATH,
        does_predict_has_ground_truths=True,
        batch_size=1,
        n_workers=N_WORKERS
    )
    data_module.setup('predict')

    predict_dataset = Subset(data_module.predict_dataset, [i for i in range(FIG_IDX, FIG_IDX+1)])
    predict_dataloader = DataLoader(predict_dataset, batch_size=1, collate_fn=COLLATE_FN)

    trainer = Trainer()
    predictions = trainer.predict(model, predict_dataloader)

    mean_avg_precision = get_mean_average_precision(predictions)
    print(mean_avg_precision)

    torch.set_printoptions  (linewidth=200)
    torch.set_printoptions  (precision=3)
    torch.set_printoptions  (sci_mode=False)
    #np.set_printoptions     (precision=3)
    #np.set_printoptions     (suppress=True)

    (idxs, points, y_true, transforms), y_pred = predictions[0]
    print(f'GT        : {y_true[0].shape}  - {y_true[0]}')
    print(f'Prediction: {y_pred.shape} - {y_pred[0]}')
    pred_y = y_pred[0][y_pred[0][:, -1] > 0.5]
    print(f'Prediction: {pred_y.shape} - {pred_y}')

    # 548
    match_sequence = get_match_sequence(pred_y, y_true[0], points[0], 0.01, 0.0174533)
    print(match_sequence)

    visualize_prediction(
        pred_y,
        points[0],
        y_true[0]
    )

