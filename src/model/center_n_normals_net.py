#!/usr/bin/env python3

from typing import Callable

import lightning
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
from torch.utils.data import Subset, DataLoader

if __name__ == "__main__":
	import sys
	sys.path.insert(0, '../..')
	from src.metrics.mAP import get_mean_average_precision
	from src.metrics.phc import get_phc
	from src.model.decoders.center_prediction_head import CenterPredictionHead
	from src.model.decoders.normal_prediction_head import NormalPredictionHead
	from src.model.encoders.pointnet_encoder import PointNetEncoder
	from src.model.losses.discrete_prediction_loss import calculate_loss
	from src.model.losses.utils import calculate_cost_matrix_normals
	from src.model.postprocessing.utils import reverse_transformation
else:
	from src.metrics.mAP import get_mean_average_precision
	from src.metrics.phc import get_phc
	from src.model.decoders.center_prediction_head import CenterPredictionHead
	from src.model.decoders.normal_prediction_head import NormalPredictionHead
	from src.model.encoders.pointnet_encoder import PointNetEncoder
	from src.model.losses.discrete_prediction_loss import calculate_loss
	from src.model.losses.utils import calculate_cost_matrix_normals
	from src.model.postprocessing.utils import reverse_transformation


class CenterNNormalsNet(nn.Module):
    def __init__(
            self,
            amount_of_normals_predicted: int = 3,
            use_bn=False
    ):
        super().__init__()
        self.use_bn = use_bn
        self.h = amount_of_normals_predicted

        self.encoder = PointNetEncoder(use_bn=self.use_bn)
        self.normal_prediction_heads = nn.ModuleList(
            [NormalPredictionHead(use_bn=self.use_bn) for _ in range(self.h)]
        )

        self.center_prediction_head = CenterPredictionHead(use_bn=self.use_bn)

    def forward(self, x):
        batch_size = x.shape[0]
        normal_list = []

        x = self.encoder(x)

        for head in self.normal_prediction_heads:
            normal_list.append(head(x))

        center = self.center_prediction_head(x).unsqueeze(dim=1).repeat(1, self.h, 1)
        normals = torch.vstack(normal_list).view(batch_size, self.h, 4)  # Normal (3) + Confidence(1)

        predictions = torch.concat((normals, center), dim=2)
        reorder = torch.tensor([0, 1, 2, 4, 5, 6, 3], device=predictions.device).long()

        predictions = predictions[:, :, reorder]

        predictions[:, :, -1] = torch.sigmoid(predictions[:, :, -1])
        predictions[:, :, 0:3] = torch.nn.functional.normalize(predictions[:, :, 0:3].clone(), dim=2)

        return predictions


class LightingCenterNNormalsNet(lightning.LightningModule):
    def __init__(self,
                 amount_of_normals_predicted: int = 3,
                 confidence_loss_constant: float = 1.0,
                 sde_loss_constant: float = 1.0,
                 distance_loss_constant: float = 1.0,
                 angle_loss_constant: float = 1.0,
                 cost_matrix_method: Callable = calculate_cost_matrix_normals,
                 print_losses: bool = False,
                 use_bn: bool = False,
                 ):
        super().__init__()
        self.use_bn = use_bn
        self.print_losses = print_losses
        self.cost_matrix_method = cost_matrix_method
        self.losses_weights = torch.tensor([
            confidence_loss_constant,
            sde_loss_constant,
            distance_loss_constant,
            angle_loss_constant,
        ])

        self.net = CenterNNormalsNet(
            amount_of_normals_predicted=amount_of_normals_predicted,
            use_bn=self.use_bn
        )
        self.save_hyperparameters(ignore=["net"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = calculate_loss(
            batch, y_pred,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        prediction = [(batch, y_pred)]
        mean_avg_precision = get_mean_average_precision(prediction)
        phc = get_phc(prediction)

        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("train_MAP", mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("train_PHC", phc, on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=len(sym_planes))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = calculate_loss(
            batch, y_pred,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        prediction = [(batch, y_pred)]
        mean_avg_precision = get_mean_average_precision(prediction)
        phc = get_phc(prediction)

        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("val_MAP", mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("val_PHC", phc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))

        return loss

    def test_step(self, batch, batch_idx):
        idxs, points, sym_planes, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)
        loss = calculate_loss(
            batch, y_pred,
            self.cost_matrix_method, self.losses_weights,
            self.print_losses
        )

        prediction = [(batch, y_pred)]
        mean_avg_precision = get_mean_average_precision(prediction)
        phc = get_phc(prediction)

        unscaled_batch, unscaled_y_pred = reverse_transformation(batch, y_pred)

        unscaled_prediction = [(unscaled_batch, unscaled_y_pred)]
        unscaled_mean_avg_precision = get_mean_average_precision(unscaled_prediction)
        unscaled_phc = get_phc(unscaled_prediction)

        self.log("test_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("test_MAP", mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("test_PHC", phc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("unscaled_test_MAP", unscaled_mean_avg_precision, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        self.log("unscaled_test_PHC", unscaled_phc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=len(sym_planes))
        return batch, y_pred

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        idxs, points, y_true, transforms = batch
        points = torch.transpose(points, 1, 2).float()

        y_pred = self.net.forward(points)

        return batch, y_pred

    def on_after_backward(self):
        for name, param in self.net.named_parameters():
            if param.grad.isnan().any():
                print(f"{name} got nan!")


if __name__ == "__main__":
    from src.dataset.shrec2023 import (SymmetryDataset,
                                       SymmetryDataModule,
                                       default_symmetry_dataset_collate_fn_list_sym)
    from src.dataset.preprocessing import *

    #DATA_PATH = "/data/shrec_2023/benchmark-train"
    DATA_PATH = "/tmp/ramdrive/benchmark-train"
    TEST_PATH = "/tmp/ramdrive/benchmark-test"
    BATCH_SIZE = 128
    EXAMPLES_USED = 10
    SAMPLE_SIZE = 1024
    COLLATE_FN = default_symmetry_dataset_collate_fn_list_sym
    NUM_WORKERS = 15

    scaler = UnitSphereNormalization()
    sampler = RandomSampler(sample_size=SAMPLE_SIZE, keep_copy=True)
    #compose_transform = ComposeTransform([scaler])
    compose_transform = ComposeTransform([scaler, sampler])

    dataset = SymmetryDataset(DATA_PATH, compose_transform)
    datamodule = SymmetryDataModule(
        train_data_path=DATA_PATH,
        #val_data_path=DATA_PATH,		# correctly, this must not exist
        test_data_path=TEST_PATH,
        predict_data_path=TEST_PATH,
        does_predict_has_ground_truths=True,
        batch_size=BATCH_SIZE,
        transform=compose_transform,
        collate_function=COLLATE_FN,
        validation_percentage=0.2,
        shuffle=True,
        n_workers=NUM_WORKERS,
    )

    test_net = LightingCenterNNormalsNet(
        27, use_bn=False,
        print_losses=False)  #
    # mpath = "modelos_interesantes/simple_net/version_9/checkpoints/epoch_epoch=30_val_loss=0.47_train_loss=0.47.ckpt"
    # test_net = LightingMyNet.load_from_checkpoint(mpath)
    datamodule.setup("predict")
    datamodule.setup("fit")

    trainer = lightning.Trainer(
        fast_dev_run=False,
        limit_val_batches=0.0,
        enable_progress_bar=True,
        max_epochs=500,
        callbacks=[
            EarlyStopping("train_loss", patience=10, verbose=True)
        ]
    )


    #training_dataset = Subset(datamodule.training_dataset, [i for i in range(1, EXAMPLES_USED)])
    print(f'{len(datamodule.train_dataset) = }')
    #train_dataloader = DataLoader(datamodule.train_dataset, batch_size=BATCH_SIZE,
    #                                collate_fn=COLLATE_FN, num_workers=NUM_WORKERS)
    train_dataloader = datamodule.train_dataloader()
    print(f'{len(train_dataloader) = }')
    val_dataloader = datamodule.val_dataloader()
    print(f'{len(val_dataloader) = }')

    print(f'{len(datamodule.predict_dataset) = }')
    predict_dataset = Subset(datamodule.predict_dataset, [i for i in range(1, EXAMPLES_USED)])
    print(f'{len(predict_dataset) = }')
    predict_dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE,
                                    collate_fn=COLLATE_FN, num_workers=NUM_WORKERS)
    print(f'{len(predict_dataloader) = }')

    #trainer.fit(test_net, predict_dataloader)
    trainer.fit(test_net, train_dataloader, val_dataloader)
    predictions = trainer.predict(test_net, predict_dataloader)

    batch, y_pred = predictions[0]
    _, _, y_true, _ = batch

    print(y_true[0])
    print(y_pred[0])

    print("Metricas normales")
    print("PHC", get_phc(predictions).item())
    print("MAP", get_mean_average_precision(predictions).item())

    print("Match normales")
    print("PHC", get_phc(predictions, eps=1).item())
    print("MAP", get_mean_average_precision(predictions, eps=1).item())

    print("Match centros")
    print("PHC", get_phc(predictions, theta=10).item())
    print("MAP", get_mean_average_precision(predictions, theta=10).item())
