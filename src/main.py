import os
import warnings
from pprint import pprint
from glob import glob
from tqdm import tqdm
from box import Box
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import StratifiedKFold

# pytorch
import torch
import pytorch_lightning as pl
import torchvision.transforms as T
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# SwinTransformer
from src.models import swint

# SVR
import cuml
from cuml.svm import SVR

from src.dataset import PetfinderDataModule

warnings.filterwarnings("ignore")


config = {
    "seed": 2021,
    "root": "/kaggle/input/petfinder-pawpularity-score/",
    #"n_splits": 5,
    "n_splits": 2,
    "epoch": 20,
    "trainer": {
        "gpus": 1,
        "accumulate_grad_batches": 1,
        "progress_bar_refresh_rate": 1,
        "fast_dev_run": False,
        "num_sanity_val_steps": 0,
        "resume_from_checkpoint": None,
    },
    "transform": {"name": "get_default_transforms", "image_size": 224},
    "train_loader": {
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": False,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 64,
        "shuffle": False,
        "num_workers": 2,
        "pin_memory": False,
        "drop_last": False,
    },
    "model": {"name": "swin_tiny_patch4_window7_224", "output_dim": 1},
    "optimizer": {"name": "optim.AdamW", "params": {"lr": 1e-5},},
    "scheduler": {
        "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params": {"T_0": 20, "eta_min": 1e-4,},
    },
    "loss": "nn.BCEWithLogitsLoss",
    "svr": {"C": 20.0},
}

config = Box(config)


def train_swint(df):
    # train test split
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df[["Id", "Pawpularity"]],
        test_size=0.25,
        random_state=config.seed,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    datamodule = PetfinderDataModule(train_df, val_df, config)
    model = swint.Model(config)
    earystopping = EarlyStopping(monitor="val_loss")
    lr_monitor = callbacks.LearningRateMonitor()
    loss_checkpoint = callbacks.ModelCheckpoint(
        filename="best_loss",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=False,
    )
    logger = TensorBoardLogger(config.model.name)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.epoch,
        callbacks=[lr_monitor, loss_checkpoint, earystopping],
        **config.trainer,
    )

    trainer.fit(model, datamodule=datamodule)

    return model

def train_swint_by_cv(df):
    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )

    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
        print(f"=====fold {fold}=======")

        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        print(train_df.shape)
        print(val_df.shape)
        datamodule = PetfinderDataModule(train_df, val_df, config)
        model = swint.Model(config)

        earystopping = EarlyStopping(monitor="val_loss")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
        )
        logger = TensorBoardLogger(config.model.name)

        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.epoch,
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **config.trainer,
        )
        trainer.fit(model, datamodule=datamodule)

    models.append(model)

    return models


def train_svr(df, embed):
    svr = SVR(C=config.svr.C)
    svr.fit(embed.astype('float32'), df["Pawpularity"].astype('int32'))

    return svr


def make_swint_embed(df, model):
    embed = []
    config.train_loader.batch_size = 10
    config.train_loader.drop_last = False
    datamodule = PetfinderDataModule(df, None, config)
    for org_train_image, label in tqdm(datamodule.train_dataloader()):
        images = model.transform["val"](org_train_image)
        images = images.to(model.device)
        emb = model.backbone.forward(images).squeeze(1)
        emb = emb.detach().cpu().numpy()
        embed.append(emb)

    config.train_loader.drop_last = True

    embed = np.concatenate(embed).astype('float32')
    return embed


def inference(df_test, model, svr, w=0.2):
    config.val_loader.batch_size = 10
    test_data_module = PetfinderDataModule(None, df_test, config)
    loader = test_data_module.val_dataloader()

    swint_preds = []
    svr_preds = []
    for org_test_image in loader:
        images = model.transform["val"](org_test_image)
        images = images.to(model.device)
        logits = model.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        swint_preds.extend(list(pred))

        # for svr
        emb = model.backbone.forward(images).squeeze(1)
        emb = emb.detach().cpu().numpy()
        pred = svr.predict(emb)
        svr_preds.extend(list(pred))

    w = 0.2
    final_preds = [
        ((1 - w) * swint_score) + (w * svr_score)
        for (swint_score, svr_score) in zip(swint_preds, svr_preds)
    ]

    return final_preds


def make_submission(test, final_preds, path):
    df_pred = pd.DataFrame()
    df_pred["Id"] = test["Id"]
    df_pred["Pawpularity"] = final_preds
    df_pred.to_csv(f"{path}/submission.csv", index=False)


def main():

    torch.autograd.set_detect_anomaly(True)
    seed_everything(config.seed)  # seed固定

    df = pd.read_csv(os.path.join(config.root, "train.csv"))
    df = df.head(500) # for debug
    df["Id"] = df["Id"].apply(lambda x: os.path.join(config.root, "train", x + ".jpg"))

    train_swint_by_cv(df)

    # swint = train_swint(df)

    # embed = make_swint_embed(df, swint)
    # svr = train_svr(df, embed)

    # df_test = pd.read_csv(os.path.join(config.root, "test.csv"))
    # df_test["Id"] = df_test["Id"].apply(
    #     lambda x: os.path.join(config.root, "test", x + ".jpg")
    # )

    # prediction = inference(df_test, swint, svr)

    # df_test = pd.read_csv(os.path.join(config.root, "test.csv"))
    # make_submission(df_test, prediction, ".")
    print("done")

if __name__ == "__main__":
    main()