from albumentations.augmentations import transforms
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader

import torch
# import madgrad

import albumentations as A
import pytorch_lightning as pl
import torch.nn.functional as F

from module.cgd import CGD, set_bn_eval
from dataset import GLDataset

from thop import profile, clever_format
from module.utils import LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss


class LitCGD(pl.LightningModule):
    def __init__(self, config, learning_rate=None):
        super().__init__()
        self.config = config

        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = config.lr

        self.model = CGD(config)
        flops, params = profile(self.model, inputs=(torch.randn(1, 3, 224, 224),))
        flops, params = clever_format([flops, params])
        print(f"# Model Params: {params} FLOPs: {flops}")

        self.class_criterion = LabelSmoothingCrossEntropyLoss(smoothing=config.smoothing, temperature=config.temperature)
        self.feature_criterion = BatchHardTripletLoss(margin=config.margin)

        self.lambda_a = config.lambda_a

    def setup(self, stage):
        if self.config.aug:
            transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
                A.RandomResizedCrop(self.config.img_size, self.config.img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(p=1.0),
            ])
        else:
            transform = A.Compose([
                A.CenterCrop(self.config.img_size, self.config.img_size),
                A.Normalize(),
                ToTensorV2(p=1.0),
            ])
        
        self.train_set = GLDataset(root=self.config.data_root, fold_no=self.config.fold_no,
                                    split_df_root=self.config.split_df_root,
                                    split='train', transform=transform)
        self.test_set = GLDataset(root=self.config.data_root, fold_no=self.config.fold_no,
                                    split_df_root=self.config.split_df_root,
                                    split='val', transform=transform)
        print(len(self.train_set), len(self.test_set))

    def training_step(self, batch, _):
        stage = 'train'
        inputs, targets = batch

        # self.model.apply(set_bn_eval)
        
        features, classes = self.model(inputs)
        class_loss = self.class_criterion(classes, targets)
        ranking_loss = self.feature_criterion(features, targets)
        loss = class_loss * self.lambda_a + ranking_loss

        self.log(f"{stage}_class_loss", class_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_ranking_loss", ranking_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        pred = torch.argmax(classes, dim=-1)
        acc = torch.sum(pred==targets).item() / inputs.size(0) * 100

        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, _):
        stage = 'val'
        inputs, targets = batch
        
        features, classes = self.model(inputs)
        class_loss = self.class_criterion(classes, targets)
        ranking_loss = self.feature_criterion(features, targets)
        loss = class_loss * self.lambda_a + ranking_loss

        self.log(f"{stage}_class_loss", class_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_ranking_loss", ranking_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        pred = torch.argmax(classes, dim=-1)
        acc = torch.sum(pred==targets).item() / inputs.size(0) * 100

        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def test_step(self, batch, _):
        stage = 'test'
        inputs, targets = batch

        self.model.apply(set_bn_eval)
        
        features, classes = self.model(inputs)
        class_loss = self.class_criterion(classes, targets)
        ranking_loss = self.feature_criterion(features, targets)
        loss = class_loss * self.lambda_a + ranking_loss

        return loss

    def configure_optimizers(self):
        if self.config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
                nesterov=True,
            )
        elif self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        # elif self.config.optimizer == "madgrad":
        #     optimizer = madgrad.MADGRAD(
        #         self.model.parameters(),
        #         lr=self.learning_rate,
        #         weight_decay=1e-5,
        #     )

        if self.config.lr_scheduler:
            lr_scheduler = {
                # "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-8),
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[int(0.6 * self.config.max_epochs), int(0.8 * self.config.max_epochs)], 
                    gamma=0.1),
                "interval": "epoch",
            }
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.batch_size,
                          shuffle=True, pin_memory=True, num_workers=self.config.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size,
                          pin_memory=True, num_workers=self.config.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size,
                          pin_memory=True, num_workers=self.config.num_workers, drop_last=True)