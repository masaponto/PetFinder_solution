import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm import create_model
import pytorch_lightning as pl
from preprocessing import get_default_transforms


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    学習データの2つを選択し、新たな画像を作成するデータ拡張手法。
    """
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.__build_model()
        self._criterion = eval(self.cfg.loss)()
        self.transform = get_default_transforms()
        self.save_hyperparameters(cfg)

    def __build_model(self):
        self.backbone = create_model(
            self.cfg.model.name, pretrained=True, num_classes=0, in_chans=3
        )
        # file_name = '../input/petfinder-timm-swint/swin_tiny_patch4_window7_224.pth'
        # self.backbone.load_state_dict(torch.load(file_name))
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(num_features, self.cfg.model.output_dim)
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.fc(f)
        return out

    def training_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "train")
        return {"loss": loss, "pred": pred, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, pred, labels = self.__share_step(batch, "val")
        return {"pred": pred, "labels": labels}

    def __share_step(self, batch, mode):
        images, labels = batch
        labels = labels.float() / 100.0
        images = self.transform[mode](images)

        if torch.rand(1)[0] < 0.5 and mode == "train":
            mix_images, target_a, target_b, lam = mixup(images, labels, alpha=0.5)
            logits = self.forward(mix_images).squeeze(1)
            loss = self._criterion(logits, target_a) * lam + (
                1 - lam
            ) * self._criterion(logits, target_b)
        else:
            logits = self.forward(images).squeeze(1)
            loss = self._criterion(logits, labels)

        pred = logits.sigmoid().detach().cpu() * 100.0
        labels = labels.detach().cpu() * 100.0
        return loss, pred, labels

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, "val")

    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out["pred"], out["labels"]
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        metrics = torch.sqrt(((labels - preds) ** 2).mean())
        self.log(f"{mode}_loss", metrics)

    # def check_gradcam(self, dataloader, target_layer, target_category, reshape_transform=None):
    #    cam = GradCAMPlusPlus(
    #        model=self,
    #        target_layers=[target_layer],
    #        use_cuda=self.cfg.trainer.gpus,
    #        reshape_transform=reshape_transform)
    #
    #    org_images, labels = iter(dataloader).next()
    #    cam.batch_size = len(org_images)
    #    images = self.transform['val'](org_images)
    #    images = images.to(self.device)
    #    logits = self.forward(images).squeeze(1)
    #    pred = logits.sigmoid().detach().cpu().numpy() * 100
    #    labels = labels.cpu().numpy()
    #
    #    grayscale_cam = cam(input_tensor=images, target_category=target_category, eigen_smooth=True)
    #    org_images = org_images.detach().cpu().numpy().transpose(0, 2, 3, 1) / 255.
    #    return org_images, grayscale_cam, pred, labels

    def configure_optimizers(self):
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        scheduler = eval(self.cfg.scheduler.name)(
            optimizer, **self.cfg.scheduler.params
        )
        return [optimizer], [scheduler]
