from argparse import ArgumentParser
from collections import Counter
import os

import torch
import imgaug.augmenters as iaa
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

from .models.lit_segmentation_model import LitLungTumorSegModel
from .models.segnet import SegNet
from .datasets.lung_tumor_dataset import get_dataset
from .loggers.image_logger import ImagePredictionLogger


def get_class_balancing_weights(training_data):
    img_class_count = Counter([1 if mask.sum() > 0 else 0 for data, mask in tqdm(training_data)])
    pos_weight = img_class_count[0] / sum(img_class_count.values())
    neg_weight = img_class_count[1] / sum(img_class_count.values())
    return neg_weight, pos_weight


def get_weighted_random_sampler(training_data, neg_weight, pos_weight):
    weighted_list = [pos_weight if mask.sum() > 0 else neg_weight for (_, mask) in training_data]
    return torch.utils.data.sampler.WeightedRandomSampler(weighted_list, len(weighted_list))


def train_model(input_args):
    pl.seed_everything(1234)

    aug_pipeline = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.LinearContrast((0.75, 1.25)),
        iaa.Affine(translate_percent=0.15, scale=(0.85, 1.15), rotate=(-45, 45)),
        iaa.ElasticTransformation()
    ])

    training_data = get_dataset(input_args.preprocessed_input_dir, aug_pipeline, data_type='train')
    validation_data = get_dataset(input_args.preprocessed_input_dir, data_type='val')
    neg_weight, pos_weight = get_class_balancing_weights(training_data)
    sampler = get_weighted_random_sampler(training_data, neg_weight, pos_weight)

    num_workers = os.cpu_count()
    train_dataloader = DataLoader(training_data, batch_size=input_args.batch_size, num_workers=num_workers,
                                  sampler=sampler)
    validation_dataloader = DataLoader(validation_data, batch_size=input_args.batch_size, num_workers=num_workers,
                                       shuffle=False)
    visualizations_dataloader = DataLoader(validation_data, batch_size=input_args.batch_size, num_workers=num_workers,
                                           shuffle=True)
    print(f"There are {len(training_data)} train images and {len(validation_data)} val images")

    segnet = SegNet(input_args.encoder_channels, input_args.decoder_channels, input_args.num_classes,
                    input_args.warm_start)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([neg_weight, pos_weight]))
    model = LitLungTumorSegModel(segnet, loss_fn, input_args.learning_rate, input_args.lr_scheduler_patience,
                                 input_args.lr_scheduler_threshold)

    visualizations_samples = next(iter(visualizations_dataloader))
    images_saving_callback = ImagePredictionLogger(visualizations_samples)
    learning_rate_callback = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='Val IOU', save_top_k=input_args.top_k)
    early_stopping_callback = EarlyStopping(monitor='Val Loss', min_delta=1e-5,
                                            patience=input_args.early_stopping_patience)

    gpus = 1 if torch.cuda.is_available() else 0
    wandb_logger = WandbLogger(project=input_args.project_name)
    wandb_logger.watch(model)
    trainer = pl.Trainer(gpus=gpus, logger=wandb_logger, log_every_n_steps=1,
                         callbacks=[checkpoint_callback, early_stopping_callback, images_saving_callback,
                                    learning_rate_callback], max_epochs=input_args.max_epochs)

    trainer.fit(model, train_dataloader, validation_dataloader)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--top_k_checkpoints', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--project_name', type=str, default="lung-tumor-segmentation-segnet")
    parser.add_argument('--preprocessed_input_dir', type=str, default=os.getcwd())
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--warm_start', type=bool, default=True)
    parser.add_argument('--encoder_channels', type=tuple, default=(3, 64, 128, 256, 512, 512))
    parser.add_argument('--decoder_channels', type=tuple, default=(512, 512, 256, 128, 64, 64))

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitLungTumorSegModel.add_model_specific_args(parser)
    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    cli_main()
