import os
import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
import numpy as np
from typing import Any


class Local2UDFModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module = None,
        pts_net: torch.nn.Module = None,
        vec_net: torch.nn.Module = None,
        pts_denoise_net: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler = None,
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.pts_net = pts_net
        self.vec_net = vec_net
        self.pts_denoise_net = pts_denoise_net
        # self.optimizer = optimizer
        self.loss_func = torch.nn.functional.l1_loss
        self.L1_regulization = torch.nn.L1Loss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MinMetric()  # take val_loss as the metric

    def forward(self, input_data: torch.Tensor, device: str):
        pts = input_data["verts"]
        vecs_q = input_data["vecs_q"]
        query = input_data["query"]
        query = query.unsqueeze(1)
        query = torch.tile(query, (1, pts.shape[1], 1))
        query = query.float().to(device)
        pts = pts.float().to(device)
        vecs_q = vecs_q.float().to(device)
        self.net.to(device)
        self.pts_net.to(device)
        self.vec_net.to(device)
        pts_feature = self.pts_net(pts)
        pts_denoise_feature = self.pts_denoise_net(pts)
        vecs_feature = self.vec_net(vecs_q)
        distance = torch.norm(vecs_q, dim=2)
        # pred_udf, displacement = self.net(pts_feature, vecs_feature, distance)
        pred_udf, displacement = self.net(
            pts_feature, vecs_feature, pts_denoise_feature, distance
        )
        
        return pred_udf, displacement

    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any, device: str):
        pred_udf, displacement = self.forward(batch, device)
        gt_udf = batch["gt_udf"].unsqueeze(1).float()
        loss_udf = self.loss_func(pred_udf, gt_udf)
        loss_disp = self.L1_regulization(displacement, torch.zeros_like(displacement))
        loss = loss_udf + 0.01 * loss_disp
        # loss = loss_udf
        return loss, loss_udf, loss_disp

    def training_step(self, batch: Any, batch_idx: int):
        loss, loss_udf, loss_disp = self.model_step(batch, self.device)
        self.train_loss(loss)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss_udf", loss_udf, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss_disp", loss_disp, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pred_udf, _ = self.forward(batch, self.device)
        gt_udf = batch["gt_udf"].unsqueeze(1).float()
        loss_udf = self.loss_func(pred_udf, gt_udf)
        self.log("val_loss_udf", loss_udf, on_step=True, on_epoch=True, prog_bar=True)
        self.val_loss(loss_udf)
        return loss_udf

    def on_validation_epoch_end(self):
        pass
        # acc = self.val_loss.compute()
        # self.val_acc_best(acc)
        # self.log(
        #     "val_acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        # )

    def test_step(self, batch: Any, batch_idx: int):
        pred_udf, _ = self.forward(batch, self.device)
        gt_udf = batch["gt_udf"].unsqueeze(1).float()
        loss_udf = self.loss_func(pred_udf, gt_udf)
        self.test_loss(loss_udf)
        self.log("test_loss", loss_udf, on_step=True, on_epoch=True, prog_bar=False)
        return loss_udf

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss",
                    "interval": "epoch",
                    "frequency": 20,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = Local2UDFModule(None, None, None, None, None, False)
