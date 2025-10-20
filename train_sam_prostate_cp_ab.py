import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from albumentations.core.serialization import from_dict
import albumentations as A
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import transforms
import torchmetrics
import wandb
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from models.pretrained_models import get_state_dict_form_pretrained_model_zoo, modify_state_dict, pretrained_model_zoo, get_state_dict_form_pretrained_model_label_ab
import utils
sys.path.append('../segment_anything')
from segment_anything import sam_model_registry
sys.path.append('../data')
from data.prostate_dataset import TrainProstateDataset

sam_config = {'sam':{'image_size':256, 'model_type':'vit_b', 'sam_checkpoint': "../../SSL-MedSeg-main/SAM-Med2D-main/workdir/models/sam-med2d-ab-26/epoch20_sam.pth",
  'encoder_adapter':True}}
if not os.path.exists(sam_config['sam']['sam_checkpoint']):
    sam_config['sam']['sam_checkpoint'] = sam_config['sam']['sam_checkpoint'][3:]
if not os.path.exists(sam_config['sam']['sam_checkpoint']):
    sam_config['sam']['sam_checkpoint'] = sam_config['sam']['sam_checkpoint'][3:]
sam_model = sam_model_registry[sam_config['sam']['model_type']](sam_config['sam']).to('cuda')
sam_model = sam_model.eval()

class CardiacSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        if 'out_channel' not in self.config.keys():
            self.config['out_channel'] = 1

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
            pretrained_weights = get_state_dict_form_pretrained_model_label_ab("prostate_a_to_b.ckpt")
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
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.val_focal_loss = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)

        self.train_iou = torchmetrics.JaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index)
        self.val_iou = torchmetrics.JaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index)
        self.reg_label_iou = torchmetrics.JaccardIndex(num_classes=self.num_classes)
        self.ce_loss = F.cross_entropy
        self.best_val_iou = 0.
        self.mse = 0.01
        self.iou = self.config['iou']
        self.bce = self.config['bce']
        self.ram = self.config['ram']
        self.sam_bce = self.config['sam_bce']
        self.u_weight = self.config['u_weight']
        self.example_input_array = torch.zeros((1, 1, 224, 224))
        self.iter = 0

    def forward(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        batch = (batch, torch.zeros(batch.shape[0]))
        mask, reg, f = self.model(batch)

        return mask, reg, f

    def setup(self, stage=0):
        if self.config["dataset"].lower() == "prostate":
            self.train_dataset = TrainProstateDataset(self.config["dataset_root"], "train", self.config["split_file"], center_index=0)
            self.val_dataset = TrainProstateDataset(self.config["dataset_root"], "val", self.config["split_file"], center_index=0)
            self.test_dataset = TrainProstateDataset(self.config["dataset_root"], "test", self.config["split_file"], center_index=0)
        print("Number of  train samples = ", len(self.train_dataset))
        print("Number of val samples = ", len(self.val_dataset))
        print("Number of test samples = ", len(self.test_dataset))


    def train_dataloader(self):
        train_aug = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomRotate90(p=0.5)
        ])
        self.train_dataset.transforms = train_aug
        self.train_dataset.cp_transforms = None
        labelnum = self.config['label_num']
        self.labeled_bs = self.config['labeled_bs']
        max_sample = len(self.train_dataset.png_paths)
        labeled_idxs = list(range(labelnum))
        unlabeled_idxs = list(range(labelnum, max_sample))
        batch_sampler = utils.TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, self.config["batch_size"], self.config["batch_size"]-self.labeled_bs)
        self.sub_bs = int(self.labeled_bs / 2)
        train_loader = DataLoader(
            self.train_dataset,
            num_workers=self.config["num_workers"],
            batch_sampler=batch_sampler,
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


    def get_ra_loss(self, logits, label=1, th_bg=0.3, bg_fg_gap=0.0):
        n, _, _, _ = logits.size()
        cls_logits = F.softmax(logits, dim=1)
        var_logits = torch.var(cls_logits, dim=1)
        norm_var_logits = self.normalize_feat(var_logits)
        bg_mask = (norm_var_logits < th_bg).float()
        fg_mask = (norm_var_logits > (th_bg + bg_fg_gap)).float()

        cls_map = logits[torch.arange(n), label, ...]
        cls_map = torch.sigmoid(cls_map)

        ra_loss = torch.mean(cls_map * fg_mask + (1 - cls_map) * bg_mask)
        return ra_loss

    def normalize_feat(self, feat):
        n, fh, fw = feat.size()
        feat = feat.view(n, -1)
        min_val, _ = torch.min(feat, dim=-1, keepdim=True)
        max_val, _ = torch.max(feat, dim=-1, keepdim=True)
        norm_feat = (feat - min_val) / (max_val - min_val + 1e-15)
        norm_feat = norm_feat.view(n, fh, fw)

        return norm_feat

    def training_sam(self, unlabel_logits, imgs):
        with torch.no_grad():
            boxes = []
            b, _, ho, wo = unlabel_logits.shape
            imgs = F.interpolate(imgs.float(), (self.config['sam']['image_size'], self.config['sam']['image_size']),
                                           mode="bilinear", align_corners=False, )
            unlabel_logits = F.interpolate(unlabel_logits, (self.config['sam']['image_size'], self.config['sam']['image_size']),
                                           mode="bilinear", align_corners=False, )
            unlabel_preds = torch.argmax(unlabel_logits, dim=1)

            for i in range(b):
                if (unlabel_preds[i] == 1).any().item():
                    p = unlabel_preds[i].cpu().numpy()
                    boxes.append(utils.mask2box(p))
                else:
                    boxes.append([0, 0, self.config['sam']['image_size'], self.config['sam']['image_size']])

            boxes = np.array(boxes)
            boxes = torch.from_numpy(boxes).cuda()
            image_embeddings, _ = sam_model.image_encoder(imgs)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )
            low_res_masks, iou_predictions = sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            unlabel_refined_logits = F.interpolate(low_res_masks, (ho, wo), mode="bilinear", align_corners=False, )
            unlabel_refined = torch.sigmoid(unlabel_refined_logits)
            unlabel_refined[unlabel_refined > 0.5] = int(1)
            unlabel_refined[unlabel_refined <= 0.5] = int(0)
            unlabel_refined = unlabel_refined.to(torch.int64).squeeze()

        return unlabel_refined

    def training_step(self, batch, batch_idx):

        features, targets, gts, img, name = batch
        targets = targets.to(torch.int64)

        sam_unimg_a = img[self.labeled_bs:self.labeled_bs + self.sub_bs]
        sam_unimg_b = img[self.labeled_bs + self.sub_bs:]

        img_a, img_b = features[:self.sub_bs], features[self.sub_bs:self.labeled_bs]
        lab_a, lab_b = targets[:self.sub_bs], targets[self.sub_bs:self.labeled_bs]

        unimg_a = features[self.labeled_bs:self.labeled_bs + self.sub_bs]
        unimg_b = features[self.labeled_bs + self.sub_bs:]

        # label branch jaccard loss
        output_a, _, _ = self.forward((img_a.float()))
        output_b, _, _ = self.forward((img_b.float()))
        iou_loss_a = self.loss(output_a, lab_a)
        iou_loss_b = self.loss(output_b, lab_b)
        iou_loss = iou_loss_a + iou_loss_b
        self.log("train_loss/iou_loss", iou_loss)
        iou_a = self.train_iou(preds=output_a, target=lab_a)
        iou_b = self.train_iou(preds=output_b, target=lab_b)
        self.log("train_iou", iou_a + iou_b, on_epoch=True)

        # mix jaccard loss
        with torch.no_grad():
            unoutput_a, _, _ = self.forward((unimg_a.float()))
            unoutput_b, _, _ = self.forward((unimg_b.float()))
            plab_a = utils.get_cut_mask(unoutput_a, nms=1)
            plab_b = utils.get_cut_mask(unoutput_b, nms=1)
            img_mask, loss_mask = utils.context_mask(img_a, self.config['mask_ratio'])

        mixl_img = img_a * img_mask + unimg_a * (1 - img_mask)
        mixu_img = unimg_b * img_mask + img_b * (1 - img_mask)

        _, outputs_l, _ = self.forward((mixl_img.float()))
        _, outputs_u, _ = self.forward((mixu_img.float()))
        loss_l = utils.mix_loss(outputs_l, lab_a, plab_a, loss_mask, u_weight=self.u_weight)
        loss_u = utils.mix_loss(outputs_u, plab_b, lab_b, loss_mask, u_weight=self.u_weight, unlab=True)
        bce_loss = loss_l + loss_u
        self.log("train_loss/label_loss", loss_l)
        self.log("train_loss/unlabel_loss", loss_u)

        # ram loss
        ram_loss_a = self.get_ra_loss(outputs_l, 1)
        ram_loss_b = self.get_ra_loss(outputs_u, 1)
        ram_loss = ram_loss_a + ram_loss_b
        self.log("train_loss/ram_loss", ram_loss)

        # sam mix jaccard loss
        if self.config['sam']['used']:
            sam_plab_a = self.training_sam(unoutput_a, sam_unimg_a)
            sam_plab_b = self.training_sam(unoutput_b, sam_unimg_b)
            self.iter += 1
            sam_loss_l = utils.mix_loss(outputs_l, lab_a, sam_plab_a, loss_mask, u_weight=self.u_weight)
            sam_loss_u = utils.mix_loss(outputs_u, sam_plab_b, lab_b, loss_mask, u_weight=self.u_weight, unlab=True)
            sam_bce_loss = sam_loss_u + sam_loss_l
            self.log("train_loss/sam_label_loss", sam_loss_l)
            self.log("train_loss/sam_unlabel_loss", sam_loss_u)
        else:
            def l1_regularization(model):
                l1_loss = []
                for module in model.modules():
                    if type(module) is nn.BatchNorm2d:
                        l1_loss.append(torch.abs(module.weight).sum())
                return 0.5 * sum(l1_loss)

            def l2_regularization(model):
                l2_loss = []
                for module in model.modules():
                    if type(module) is nn.Conv2d:
                        l2_loss.append((module.weight ** 2).sum() / 2.0)
                return 0.5 * sum(l2_loss)

            sam_bce_loss = l1_regularization(self.model)
            self.log("train_loss/l1_loss", sam_bce_loss)
            # sam_bce_loss = l2_regularization(self.model)
            # self.log("train_loss/l2_loss", sam_bce_loss)

        total_loss = self.iou * iou_loss + self.ram * ram_loss + self.bce * bce_loss + self.sam_bce * sam_bce_loss
        self.log("train_loss/total_loss", total_loss)
        return total_loss

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore
        return torch.Tensor([lr])[0]

    def validation_step(self, batch, batch_id, total_loss=0):
        features, targets, _, _, _ = batch
        targets = targets.to(torch.int64)
        logits, regs, _ = self.forward((features.float()))

        iou_loss = self.loss(regs, targets)
        self.log("val_loss/val_iou_loss", iou_loss)
        ram_loss = self.get_ra_loss(regs, 1)
        self.log("val_loss/val_ram_loss", ram_loss)
        loss = ram_loss + iou_loss
        self.log("val_loss/val_total_loss", loss)

        val_iou = self.val_iou(preds=regs, target=targets)
        self.log("val_metrics/mean_iou", val_iou)

        if val_iou > self.best_val_iou:
            self.best_val_iou = val_iou
        self.log("val_metrics/best_mean_iou", self.best_val_iou)

        per_class_ious = torchmetrics.functional.jaccard_index(regs, targets, absent_score=np.NaN,
                                                               num_classes=self.num_classes, average='none')
        for i in range(self.num_classes):
            self.log(f"val_metrics/{self.class_labels[i]}_iou", per_class_ious[i])
        self.log("val_metrics/focal_loss", self.val_focal_loss(regs, targets))
        self.log("val_metrics/val_lr", self._get_current_lr())
        return loss

    def on_after_backward(self) -> None:
        # Cancel gradients for the encoder in the first few epochs
        if self.current_epoch < self.config['model'].get("freeze_encoder_weights_epochs", 0):
            for p in self.model.encoder.parameters():
                p.grad = None
        return super().on_after_backward()



def main():
    os.environ['WANDB_MODE'] = 'dryrun'
    default_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config_sam_prostate_cp_ab.yaml')
    config = utils.get_config(default_config_path)

    pl.seed_everything(config["seed"], workers=True)

    pipeline = CardiacSegmentation(config)

    tb_logger = pl.loggers.TensorBoardLogger(config["artifacts_root"], name=config["experiment_name"], log_graph=False)
    wandb_logger = pl.loggers.WandbLogger(name=config["experiment_name"], project='de-Segmentation',
                                          save_dir=config["artifacts_root"])
    # pl.LightningModule.hparams is set by pytorch lightning when calling save_hyperparameters
    tb_logger.log_hyperparams(pipeline.hparams)
    wandb_logger.log_hyperparams(pipeline.hparams)
    if config.get("wandb_tag", "") != "":
        wandb.run.tags = wandb.run.tags + (config["wandb_tag"],)

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, f"checkpoints"),
        save_last=True,  # False to reduce disk load from constant checkpointing
        save_top_k=20,
        monitor='val_metrics/mean_iou',
        mode='max',
        every_n_epochs=100,
    )
    # model_summary = pl.callbacks.ModelSummary(max_depth=5)

    trainer = pl.Trainer(
        devices=0 if config["trainer"]["gpus"] == '0' or not torch.cuda.is_available() else config["trainer"]["gpus"],
        max_epochs=config["trainer"]["max_epochs"],
        precision=config["trainer"]["precision"] if torch.cuda.is_available() else 32,
        logger=[tb_logger],
        callbacks=[checkpoint],
        log_every_n_steps=1,
        gradient_clip_val=config["trainer"]["gradient_clip_val"],
        accelerator='gpu',
        )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
