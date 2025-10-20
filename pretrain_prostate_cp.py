import argparse
import os
from pathlib import Path
from typing import Dict
import yaml
import numpy as np
from tabulate import tabulate
from torchvision import transforms
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from albumentations.core.serialization import from_dict
import albumentations as A
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import segmentation_models_pytorch as smp
import torchmetrics
import wandb
from data.prostate_dataset import PretrainDataset
from models.pretrained_models import get_state_dict_form_pretrained_model_zoo, modify_state_dict, pretrained_model_zoo, get_state_dict_form_pretrained_model_label
import utils

class CardiacSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        if self.config["dataset"].lower() == "prostate":
            self.num_classes = 2
            self.class_labels = ["BG", "PR"]
            self.in_channels = 1
            self.out_channels = self.config['out_channel']
            self.ignore_index = 0

            # Instantiating encoder and loading pretrained weights
        encoder_weights = self.config["model"].get("encoder_weights", None)

        # Only supervised ImageNet weights can be loaded when instantiating an smp.Unet:
        if encoder_weights in ['imagenet', 'supervised-imagenet']:
            auto_loaded_encoder_weights = 'imagenet'
        else:
            auto_loaded_encoder_weights = None

        self.model = smp.Unet(
            encoder_name=self.config["model"]["encoder_name"],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=auto_loaded_encoder_weights,
            # for timm models, anything that is Not none will load imagenet weights...
            in_channels=self.in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            out_channels=self.out_channels,
            classes=self.num_classes,  # model output channels (number of classes in your dataset)
        )

        if self.config["model"]["weights"]:
            pretrained_weights = get_state_dict_form_pretrained_model_label("last.ckpt")
            self.model.encoder.load_state_dict(pretrained_weights['encoder'])
            self.model.decoder.load_state_dict(pretrained_weights['decoder'])
            self.model.segmentation_head.load_state_dict(pretrained_weights['segmentation'])
            self.model.regression_head.load_state_dict(pretrained_weights['regression'])

        else:
            pretrained_weights = get_state_dict_form_pretrained_model_zoo(self.config["model"]["encoder_weights"],
                                                                          in_chans=self.in_channels,
                                                                          prefix_to_add='model.')
            if pretrained_weights is not None:
                self.model.encoder.load_state_dict(pretrained_weights)
                print("Encoder pretrained weights loaded from {}".format(encoder_weights))

            if self.config["model"].get("pretrained_decoder", False):
                pretrained_decoder_with_head = torch.load("../models/supervised_decoder.pth")
                pretrained_decoder = modify_state_dict(pretrained_decoder_with_head, prefix_to_remove='model.decoder.')
                pretrained_segmentation_head = modify_state_dict(pretrained_decoder_with_head,
                                                                 prefix_to_remove='model.segmentation_head.')
                self.model.decoder.load_state_dict(pretrained_decoder)
                self.model.segmentation_head.load_state_dict(pretrained_segmentation_head)
                print('Pretained weights for decoder loaded.')

                if self.config['model']['regression']:
                    self.model.classification_head_define.load_state_dict(pretrained_segmentation_head)
                    print('Pretained weights for classification banch loaded.')

        self.loss = smp.losses.__dict__[self.config["loss"]](smp.losses.MULTICLASS_MODE)
        self.bce_loss = F.binary_cross_entropy_with_logits
        self.val_focal_loss = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)

        self.train_iou = torchmetrics.JaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index)
        self.val_iou = torchmetrics.JaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index)
        self.reg_label_iou = torchmetrics.JaccardIndex(num_classes=1)
        self.ce_loss = F.cross_entropy
        self.best_val_iou = 0.
        self.sub_bs = self.config['sub_bs']
        self.iou = self.config['iou']
        self.bce = self.config['bce']
        self.example_input_array = torch.zeros((1, 1, 224, 224))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        if len(batch) == 1:
            batch = (batch, torch.zeros(batch.shape[0]))
        mask, reg, _ = self.model(batch)

        return mask, reg, _

    def setup(self, stage):
        if self.config["dataset"].lower() == "prostate":
            self.train_dataset = PretrainDataset(self.config["dataset_root"], "train", self.config["split_file"], center_index=self.config['center'])
            self.val_dataset = PretrainDataset(self.config["dataset_root"], "val", self.config["split_file"], center_index=self.config['center'])
            self.test_dataset = PretrainDataset(self.config["dataset_root"], "test", self.config["split_file"], center_index=self.config['center'])
        print("Number of  train samples = ", len(self.train_dataset))
        print("Number of val samples = ", len(self.val_dataset))
        print("Number of test samples = ", len(self.test_dataset))

    def train_dataloader(self):
        train_aug = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # RandomRotFlip()
        ])
        self.train_dataset.transforms = train_aug
        self.train_dataset.cp_transforms = None

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
            drop_last=True,
        )

        print("Train dataloader = ", len(train_loader), "Train samples =", len(self.train_dataset))
        wandb.config.update({"num_samples": len(self.train_dataset)})
        return train_loader

    def val_dataloader(self):
        val_aug = A.Compose([
            A.RandomResizedCrop(224, 224),
        ])
        self.val_dataset.transforms = val_aug

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            drop_last=False,
        )
        print("Val dataloader = ", len(val_loader))
        return val_loader

    def configure_optimizers(self):
        optimizer = utils.object_from_dict(
            self.config["optimizer"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = utils.object_from_dict(self.config["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def normalize_feat(self, feat):
        n, fh, fw = feat.size()
        feat = feat.view(n, -1)
        min_val, _ = torch.min(feat, dim=-1, keepdim=True)
        max_val, _ = torch.max(feat, dim=-1, keepdim=True)
        norm_feat = (feat - min_val) / (max_val - min_val + 1e-15)
        norm_feat = norm_feat.view(n, fh, fw)

        return norm_feat

    def get_cls_loss(self, logits, label):
        return self.ce_loss(logits, label[:, 0].long())

    def training_step(self, batch, batch_idx):

        features, targets, gts = batch
        targets = targets.to(torch.int64)

        img_a, img_b = features[:self.sub_bs], features[self.sub_bs:]
        lab_a, lab_b = targets[:self.sub_bs], targets[self.sub_bs:]
        with torch.no_grad():
            img_mask, loss_mask = utils.context_mask(img_a, self.config['mask_ratio'])

        """Mix Input"""
        features = img_a * img_mask + img_b * (1 - img_mask)
        targets = lab_a * img_mask + lab_b * (1 - img_mask)


        logits, regs, _ = self.forward((features.float(), gts))

        iou_loss = self.iou * self.loss(logits, targets)
        total_loss = iou_loss
        self.log("train_loss/iou_loss", iou_loss)
        self.log("train_iou", self.train_iou(preds=logits, target=targets.to(torch.int64)), on_epoch=True)

        if self.out_channels == 1:
            bce_loss = self.bce * self.bce_loss(regs.squeeze(1), targets.float())
        elif self.out_channels == 2:
            bce_loss = self.bce * self.loss(regs, targets)

        total_loss += bce_loss
        self.log("train_loss/bce_loss", bce_loss)
        self.log("train_loss/total_loss", total_loss)
        return total_loss

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0]

    def validation_step(self, batch, batch_idx):
        features, targets, gts = batch
        targets = targets.to(torch.int64)
        logits, regs, _ = self.forward((features.float(), gts))

        iou_loss = self.iou * self.loss(logits, targets)
        val_iou = self.val_iou(preds=logits, target=targets)
        total_loss = iou_loss
        self.log("val_metrics/iou_loss", iou_loss)

        self.log("val_metrics/total_loss", total_loss)
        self.log("val_metrics/mean_iou", val_iou)
        if val_iou > self.best_val_iou:
            self.best_val_iou = val_iou
        self.log("val_metrics/best_mean_iou", self.best_val_iou)
        self.log("val_metrics/focal_loss", self.val_focal_loss(logits, targets))
        self.log("val_lr", self._get_current_lr())
        per_class_ious = torchmetrics.functional.jaccard_index(logits, targets, absent_score=np.NaN,
                                                               num_classes=self.num_classes, average='none')
        for i in range(self.num_classes):
            self.log(f"val_metrics/{self.class_labels[i]}_iou", per_class_ious[i])

        return total_loss

    def on_after_backward(self) -> None:
        # Cancel gradients for the encoder in the first few epochs
        if self.current_epoch < self.config['model'].get("freeze_encoder_weights_epochs", 0):
            for p in self.model.encoder.parameters():
                p.grad = None
        return super().on_after_backward()


def main():
    # import torch
    # torch.zeros(1).cuda()
    os.environ['WANDB_MODE'] = 'dryrun'
    default_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretrain_prostate.yaml')
    config = utils.get_config(default_config_path)

    pl.seed_everything(config["seed"], workers=True)

    pipeline = CardiacSegmentation(config)

    tb_logger = pl.loggers.TensorBoardLogger(config["artifacts_root"], name=config["experiment_name"], log_graph=False)
    wandb_logger = pl.loggers.WandbLogger(name=config["experiment_name"], project='prostate-seg',
                                          save_dir=config["artifacts_root"])
    # pl.LightningModule.hparams is set by pytorch lightning when calling save_hyperparameters
    tb_logger.log_hyperparams(pipeline.hparams)
    wandb_logger.log_hyperparams(pipeline.hparams)
    if config.get("wandb_tag", "") != "":
        wandb.run.tags = wandb.run.tags + (config["wandb_tag"],)

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, f"checkpoints"),
        save_last=True,  # False to reduce disk load from constant checkpointing
        save_top_k=10,
        monitor='val_metrics/mean_iou',
        mode='max',
        every_n_epochs=10,
    )

    trainer = pl.Trainer(
        devices=0 if config["trainer"]["gpus"] == '0' or not torch.cuda.is_available() else config["trainer"]["gpus"],
        max_epochs=config["trainer"]["max_epochs"],
        precision=config["trainer"]["precision"] if torch.cuda.is_available() else 32,
        logger=[tb_logger],
        callbacks=[checkpoint],
        log_every_n_steps=10,
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
        accelerator='gpu',
        )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
