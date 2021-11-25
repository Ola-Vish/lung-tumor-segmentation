import torch
import torchmetrics
import pytorch_lightning as pl
from argparse import ArgumentParser


class LitLungTumorSegModel(pl.LightningModule):
    def __init__(self, model, loss_fn, num_classes=2, learning_rate=1e-4, lr_scheduler_patience=5,
                 lr_scheduler_threshold=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = loss_fn
        self.train_iou = torchmetrics.IoU(num_classes)
        self.validation_iou = torchmetrics.IoU(num_classes)

    def forward(self, data):
        logits = self.model(data)
        preds = torch.argmax(logits, dim=1)
        return preds

    def training_step(self, batch, batch_idx):
        ct_scans, ct_masks = batch
        labels = ct_masks.squeeze(1).long()
        pred = self.model(ct_scans)
        loss = self.loss_fn(pred, labels)

        self.log("Train Loss", loss)
        self.train_iou(pred, labels)
        self.log("Train IOU", self.train_iou)
        return loss

    def validation_step(self, batch, batch_idx):
        ct_scans, ct_masks = batch
        labels = ct_masks.squeeze(1).long()
        pred = self.model(ct_scans)
        loss = self.loss_fn(pred, labels)

        self.log("Val Loss", loss)
        self.validation_iou(pred, labels)
        self.log("Val IOU", self.validation_iou)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                        patience=self.hparams.lr_scheduler_patience,
                                                                        threshold=self.hparams.lr_scheduler_threshold),
                "monitor": "Val Loss",
                "frequency": 1
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--lr_scheduler_patience', type=int, default=5)
        parser.add_argument('--lr_scheduler_threshold', type=float, default=1e-5)
        return parser