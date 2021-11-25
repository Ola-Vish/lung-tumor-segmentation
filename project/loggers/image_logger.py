import wandb
import pytorch_lightning as pl


class ImagePredictionLogger(pl.Callback):
    def __init__(self, data_samples, num_samples=32):
        super().__init__()
        self.imgs, self.labels = data_samples
        self.imgs = self.imgs[:num_samples]
        self.labels = self.labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        imgs = self.imgs.to(device=pl_module.device)
        preds = pl_module(imgs)
        class_labels = {0: "no tumor", 1: "tumor"}

        for idx in range(len(imgs)):
            mask_img = {f"img_{idx}": wandb.Image(imgs[idx], masks={
                "ground_truth": {
                    "mask_data": self.labels[idx][0].numpy(),
                    "class_labels": class_labels
                },
                "predictions": {
                    "mask_data": preds[idx].cpu().numpy(),
                    "class_labels": class_labels
                }
            })}
            trainer.logger.experiment.log(mask_img)