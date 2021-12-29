import os
import warnings
from pprint import pprint
from glob import glob
from tqdm import tqdm
from box import Box
import numpy as np
import pandas as pd
import joblib
import gc

# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
    "n_splits": 10,
    "epoch": 20,
    "trainer": {
        "gpus": 1,
        "accumulate_grad_batches": 4,
        "progress_bar_refresh_rate": 1,
        "fast_dev_run": False,
        "num_sanity_val_steps": 0,
        "resume_from_checkpoint": None,
    },
    "transform": {"name": "get_default_transforms", "image_size": 224},
    "train_loader": {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": False,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 8,
        "shuffle": False,
        "num_workers": 2,
        "pin_memory": False,
        "drop_last": False,
    },
    # "model": {"name": "swin_tiny_patch4_window7_224", "output_dim": 1},
    "model": {"name": "swin_large_patch4_window7_224_in22k", "output_dim": 1},
    # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py#L84
    "optimizer": {
        "name": "optim.AdamW",
        "params": {"lr": 1e-5},
    },
    "scheduler": {
        "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params": {
            "T_0": 20,
            "eta_min": 1e-4,
        },
    },
    "loss": "nn.BCEWithLogitsLoss",
    "svr": {"C": 20.0},
}

config = Box(config)


def train_swint(df):
    # train test split

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


def train_swint_by_cv(df, path):
    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
        print(f"=====fold {fold}=======")

        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)

        datamodule = PetfinderDataModule(train_df, val_df, config)
        model = swint.Model(config)

        earystopping = EarlyStopping(monitor="val_loss")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            dirpath=path,
            filename=f"best_loss_fold_{fold}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
        )
        logger = TensorBoardLogger(path, name="lightning_logs", version=fold)

        trainer = pl.Trainer(
            logger=logger,
            max_epochs=config.epoch,
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **config.trainer,
        )
        trainer.fit(model, datamodule=datamodule)
        trainer.validate(model, dataloaders=datamodule.val_dataloader())

        del train_df, val_df, model
        gc.collect()


def train_svr(df, embed):
    svr = SVR(C=config.svr.C)
    svr.fit(embed.astype("float32"), df["Pawpularity"].astype("int32"))

    return svr


def make_swint_embed(df, model):
    embed = []
    # config.train_loader.batch_size = 8
    config.train_loader.drop_last = False
    datamodule = PetfinderDataModule(df, None, config)
    for org_train_image, label in tqdm(datamodule.train_dataloader()):
        images = model.transform["val"](org_train_image)
        images = images.to(model.device)
        emb = model.backbone.forward(images).squeeze(1)
        emb = emb.detach().cpu().numpy()
        embed.append(emb)

    config.train_loader.drop_last = True

    embed = np.concatenate(embed).astype("float32")
    return embed


def inference_swint_svr(df_test, model, svr, w=0.2):
    # config.val_loader.batch_size = 8
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


def train_ensemble(df, model_path):
    df["Id"] = df["Id"].apply(lambda x: os.path.join(config.root, "train", x + ".jpg"))

    print("===train swint===")
    train_swint_by_cv(df, model_path)

    print("===train svr===")
    for fold in range(config.config.n_splits):
        swint = swint.Model.load_from_checkpoint(
            f"{model_path}/best_loss_fold_{fold}.ckpt"
        )
        svr = train_svr(df, swint)
        joblib.dump(svr, f"{model_path}/svr_{fold}.joblib")


def inference_ensemble(df_test, model_path):

    print("===test===")
    df_test["Id"] = df_test["Id"].apply(
        lambda x: os.path.join(config.root, "test", x + ".jpg")
    )

    prediction = np.zeros(len(df_test))

    for fold in range(config.n_splits):
        swint = swint.Model.load_from_checkpoint(
            f"{model_path}/best_loss_fold_{fold}.ckpt"
        )
        svr = joblib.load(f"{model_path}/svr_{fold}.joblib")

        # calc mean
        prediction = (float(fold) / (fold + 1)) * prediction + np.array(
            inference_swint_svr(df_test, swint, svr)
        ) / (fold + 1)

    return prediction


def experiment(df, path):
    df["Id"] = df["Id"].apply(lambda x: os.path.join(config.root, "train", x + ".jpg"))
    print("===split vaild===")
    df, df_val = train_test_split(
        df[["Id", "Pawpularity"]],
        test_size=0.25,
        random_state=config.seed,
        shuffle=True,
    )
    df = df.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    train_ensemble(df, path)
    prediction = inference_ensemble(df_val, path)
    rmse = np.sqrt(mean_squared_error(df_val["Pawpularity"], prediction))
    print(f"RMSE: {rmse}")


def main():
    torch.autograd.set_detect_anomaly(True)
    seed_everything(config.seed)  # seed固定

    # df = pd.read_csv(os.path.join(config.root, "train.csv"))
    # experiment(df, "test")

    model_path = "model_submission"
    df = pd.read_csv(os.path.join(config.root, "train.csv"))
    train_ensemble(df, model_path)

    df_test = pd.read_csv(os.path.join(config.root, "test.csv"))
    prediction = inference_ensemble(df_test, model_path)
    make_submission(df_test, prediction, ".")


if __name__ == "__main__":
    main()
