import os
from re import I
import warnings
from pprint import pprint
from glob import glob
from tqdm import tqdm
from box import Box
import numpy as np
import pandas as pd
import joblib
import gc
import optuna
import time

# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from copy import deepcopy

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

# LightGBM
from lightgbm import LGBMRegressor

# datamodule
from src.dataset import PetfinderDataModule


warnings.filterwarnings("ignore")


config = {
    "seed": 2021,
    "root": "/kaggle/input/petfinder-pawpularity-score/",
    "crop_data_path": "/kaggle/input/crop/",
    "n_splits": 10,
    # "epoch": 20,
    "epoch": 5,
    "trainer": {
        "gpus": 1,
        # "accumulate_grad_batches": 4,
        "accumulate_grad_batches": 8,
        "progress_bar_refresh_rate": 1,
        "fast_dev_run": False,
        "num_sanity_val_steps": 0,
        "resume_from_checkpoint": None,
    },
    # "transform": {"name": "get_default_transforms", "image_size": 224},
    "transform": {"name": "get_default_transforms", "image_size": 384},
    "train_loader": {
        # "batch_size": 8,
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": False,
        "drop_last": True,
    },
    "val_loader": {
        # "batch_size": 8,
        "batch_size": 4,
        "shuffle": False,
        "num_workers": 2,
        "pin_memory": False,
        "drop_last": False,
    },
    # "model_tiny": {"name": "swin_tiny_patch4_window7_224", "output_dim": 1},
    # "model": {"name": "swin_large_patch4_window7_224_in22k", "output_dim": 1},
    "model": {"name": "swin_large_patch4_window12_384_in22k", "output_dim": 1},
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
    "svr": {},
    "lgbm": {},
}

config = Box(config)


def train_swint(
    train_df,
    val_df,
    fold,
    path,
):
    print(f"=====fold {fold}=======")
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

    model.eval()
    val_out = trainer.validate(model, dataloaders=datamodule.val_dataloader())

    print(val_out)


def train_svr(df, embed):
    # params = {
    #     "gamma": 0.006550458887347427,
    #     "C": 5.7273276958907715,
    #     "epsilon": 6.834062667406395,
    # }
    # ????????? 17.68122135877512
    # ????????????465.755384683609???

    params = {"C": 20}

    svr = SVR(**params)
    svr.fit(embed.astype("float32"), df["Pawpularity"].astype("int32"))

    return svr


def train_lightgbm(df, embed, df_val, embed_val):
    # tuned by oputuna
    # params = {
    #     "reg_alpha": 0.006013503575632864,
    #     "reg_lambda": 0.004075989478302265,
    #     "num_leaves": 16,
    #     "colsample_bytree": 0.5064800798900133,
    #     "subsample": 0.7871572722868186,
    #     "subsample_freq": 1,
    #     "min_child_samples": 93,
    # }

    params = {}

    lgbm = LGBMRegressor(random_state=config.seed, n_estimators=10000)
    lgbm.set_params(**params)
    lgbm.fit(
        embed.astype("float32"),
        df["Pawpularity"].astype("int32"),
        eval_metric="rmse",
        eval_set=[(embed_val.astype("float32"), df_val["Pawpularity"].astype("int32"))],
        early_stopping_rounds=1000,
    )

    return lgbm


def make_swint_embed(df, model, mode, fold, path=None):
    assert mode in ("train", "val")

    embed = []
    datamodule = PetfinderDataModule(None, df.drop("Pawpularity", axis=1), config)
    loader = datamodule.val_dataloader()

    if path:
        os.makedirs(path, exist_ok=True)

    for org_train_image in tqdm(loader):

        images = model.transform[mode](org_train_image)
        images = images.to(model.device)
        emb = model.backbone.forward(images).squeeze(1)
        emb = emb.detach().cpu().numpy()
        embed.append(emb)

    embed = np.concatenate(embed).astype("float32")

    if path:
        np.save(f"{path}/swint_embed_{mode}_{fold}.npy", embed)

    return embed


def inference(df_test, model):
    if "Pawpularity" in df_test.columns:
        df_test = df_test.drop("Pawpularity", axis=1)

    test_data_module = PetfinderDataModule(None, df_test, config)
    loader = test_data_module.val_dataloader()

    swint_preds = []
    for org_test_image in tqdm(loader):
        images = model.transform["val"](org_test_image)
        images = images.to(model.device)
        logits = model.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        swint_preds.extend(list(pred))

    final_preds = swint_preds

    return final_preds


def inference_swint_svr(df_test, model, svr, w=0.2):
    # config.val_loader.batch_size = 8
    test_data_module = PetfinderDataModule(None, df_test, config)
    loader = test_data_module.val_dataloader()

    swint_preds = []
    svr_preds = []
    for org_test_image in tqdm(loader):
        images = model.transform["val"](org_test_image)
        images = images.to(model.device)
        logits = model.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        swint_preds.extend(list(pred))

        # for svr
        emb = model.backbone.forward(images).squeeze(1)
        emb = emb.detach().cpu().numpy()
        pred = svr.predict(emb.astype("float32"))
        svr_preds.extend(list(pred))

    final_preds = [
        ((1 - w) * swint_score) + (w * svr_score)
        for (swint_score, svr_score) in zip(swint_preds, svr_preds)
    ]

    return final_preds


def inference_swint_svr_lgbm(df_test, model, svr, lgbm, w_svr=0.2, w_lgbm=0.2):

    assert 1 > (w_svr + w_lgbm) >= 0

    # config.val_loader.batch_size = 8
    test_data_module = PetfinderDataModule(None, df_test, config)
    loader = test_data_module.val_dataloader()

    swint_preds = []
    svr_preds = []
    lgbm_preds = []

    for org_test_image in tqdm(loader):
        images = model.transform["val"](org_test_image)
        images = images.to(model.device)
        logits = model.forward(images).squeeze(1)
        pred = logits.sigmoid().detach().cpu().numpy() * 100
        swint_preds.extend(list(pred))

        # for svr
        emb = model.backbone.forward(images).squeeze(1)
        emb = emb.detach().cpu().numpy()
        svr_pred = svr.predict(emb.astype("float32"))
        svr_preds.extend(list(svr_pred))

        # for lgbm
        lgbm_pred = lgbm.predict(emb.astype("float32"))

        lgbm_preds.extend(list(lgbm_pred))

    final_preds = [
        ((1 - w_svr - w_lgbm) * swint_score)
        + (w_svr * svr_score)
        + (w_lgbm * lgbm_score)
        for (swint_score, svr_score, lgbm_score) in zip(
            swint_preds, svr_preds, lgbm_preds
        )
    ]

    return final_preds


def make_submission(test, final_preds, path):
    df_pred = pd.DataFrame()
    df_pred["Id"] = test["Id"]
    df_pred["Pawpularity"] = final_preds
    df_pred.to_csv(f"{path}/submission.csv", index=False)


def train_ensemble(df, model_path):
    print("original data shape", df.shape)
    # df_crop = df.copy()
    df["Id"] = df["Id"].apply(lambda x: os.path.join(config.root, "train", x + ".jpg"))

    # df_crop["Id"] = df_crop["Id"].apply(
    #     lambda x: os.path.join(config.crop_data_path, x + ".jpg")
    # )

    # df = pd.concat([df, df_crop]).reset_index(drop=True)

    # print("Add data shape", df.shape)

    print("===train svr lgbm===")

    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    lgbm_rmse_list = []
    svr_rmse_list = []
    swint_rmse_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df["Id"], df["Pawpularity"])):
        if fold != 0:
            continue

        print(f"===fold: {fold}===")

        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)

        # train_swint(train_df, val_df, fold, "model_original3")
        # convert_ckpt_to_state_dict("model_original3", model_path, fold)

        # swint_model = swint.Model(config)
        # swint_model.load_state_dict(torch.load(f"{model_path}/best_loss_{fold}.pth"))
        # swint_model.eval()
        # swint_model.to("cuda:0")

        # valid swint
        # pred = inference(val_df, swint_model)
        # rmse = np.sqrt(mean_squared_error(val_df["Pawpularity"], pred))
        # print("swint", rmse)
        # swint_rmse_list.append(rmse)

        # embed = make_swint_embed(
        #    train_df, swint_model, "train", fold, path="swint_embed_3"
        # )
        # val_embed = make_swint_embed(
        #    val_df, swint_model, "val", fold, path="swint_embed_3"
        # )

        embed = np.load(f"swint_embed_3/swint_embed_train_{fold}.npy")
        val_embed = np.load(f"swint_embed_3/swint_embed_val_{fold}.npy")

        # train svr
        svr = train_svr(train_df, embed)
        pred = svr.predict(val_embed)
        rmse = np.sqrt(mean_squared_error(val_df["Pawpularity"], pred))
        print("svr", rmse)
        joblib.dump(svr, f"{model_path}/svr_{fold}.joblib")

        svr_rmse_list.append(rmse)

        # train LightGBM
        # lgbm = train_lightgbm(train_df, embed, val_df, val_embed)
        # pred = lgbm.predict(val_embed, num_iteration=lgbm.best_iteration_)
        # rmse = np.sqrt(mean_squared_error(val_df["Pawpularity"], pred))
        # print("lgbm", rmse)
        # joblib.dump(lgbm, f"{model_path}/lgbm_{fold}.joblib")

        # lgbm_rmse_list.append(rmse)

    print("=== svr rmse ===")
    for rmse in svr_rmse_list:
        print(rmse)
    print("mean:", np.mean(svr_rmse_list))

    # print("=== lightgbm rmse ===")
    # for rmse in lgbm_rmse_list:
    #    print(rmse)
    # print("mean:", np.mean(lgbm_rmse_list))


def inference_ensemble_state_dict(df_test, model_path, mode, w_svr=0.2, w_lgbm=0.2):

    print("===test===")
    df_test["Id"] = df_test["Id"].apply(
        lambda x: os.path.join(config.root, "test", x + ".jpg")
    )

    prediction = np.zeros(len(df_test))
    print(f"=mode: {mode}=")
    for fold in range(config.n_splits):
        swint_model = swint.Model(config)
        swint_model.load_state_dict(torch.load(f"{model_path}/best_loss_{fold}.pth"))
        swint_model.eval()
        swint_model.to("cuda:0")

        if mode == "swint":
            x = inference(df_test, swint_model)

        elif mode == "swint_svr":
            svr = joblib.load(f"{model_path}/svr_{fold}.joblib")
            x = inference_swint_svr(df_test, swint_model, svr)

        elif mode == "swint_svr_lgbm":
            svr = joblib.load(f"{model_path}/svr_{fold}.joblib")
            lgbm = joblib.load(f"{model_path}/lgbm_{fold}.joblib")
            x = inference_swint_svr_lgbm(
                df_test, swint_model, svr, lgbm, w_svr=w_svr, w_lgbm=w_lgbm
            )
        else:
            raise Exception("invalid mode")

        # calc mean
        prediction = (float(fold) / (fold + 1)) * prediction + np.array(x) / (fold + 1)

    return prediction


def convert_ckpt_to_state_dict(src_path, dst_path, fold):
    os.makedirs(dst_path, exist_ok=True)
    swint_model = swint.Model.load_from_checkpoint(
        f"{src_path}/best_loss_fold_{fold}.ckpt",
        cfg=config,
    )
    torch.save(swint_model.state_dict(), f"{dst_path}/best_loss_{fold}.pth")


def tune_lightgbm(df, emb_path="swint_embed"):
    print("lightgbm tune start")
    start = time.time()

    def bayes_objective(trial):
        params = {
            # "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.1, log=True),
            # "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.1, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0001, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 200),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 0, 100),
        }
        # ?????????????????????????????????
        model = LGBMRegressor(random_state=config.seed, n_estimators=10000)
        model.set_params(**params)

        # cross_val_score?????????????????????????????????
        skf = StratifiedKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.seed
        )

        rmse_list = []
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(df["Id"], df["Pawpularity"])
        ):
            lgbm = deepcopy(model)
            train_df = df.loc[train_idx].reset_index(drop=True)
            val_df = df.loc[val_idx].reset_index(drop=True)

            train_embed = np.load(f"{emb_path}/swint_embed_train_{fold}.npy")
            val_embed = np.load(f"{emb_path}/swint_embed_val_{fold}.npy")

            lgbm.fit(
                train_embed.astype("float32"),
                train_df["Pawpularity"].astype("int32"),
                eval_metric="rmse",
                eval_set=[
                    (val_embed.astype("float32"), val_df["Pawpularity"].astype("int32"))
                ],
                early_stopping_rounds=10,
            )

            pred = lgbm.predict(val_embed, num_iteration=lgbm.best_iteration_)
            rmse = np.sqrt(mean_squared_error(val_df["Pawpularity"], pred))
            rmse_list.append(rmse)

        rmse = np.mean(rmse_list)

        return rmse

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(bayes_objective, n_trials=100, n_jobs=-1)

    best_params = study.best_trial.params
    best_score = study.best_trial.value
    print(f"????????????????????? {best_params}\n????????? {best_score}")
    print(f"????????????{time.time() - start}???")

    # trial 8
    # {'reg_alpha': 0.08119805342335897, 'reg_lambda': 0.05173352164757917, 'num_leaves': 170, 'colsample_bytree': 0.6325844744041931, 'subsample': 0.5336184209382349, 'subsample_freq': 0, 'min_child_samples': 74}
    # ????????? 17.835622159304837
    # ????????????3706.5731043815613???

    # trial 100
    # ????????????????????? {'reg_alpha': 0.00021028343930250634, 'reg_lambda': 0.008156559010111118, 'num_leaves': 132, 'colsample_bytree': 0.3266158434999465, 'subsample': 0.8718341463395, 'subsample_freq': 4, 'min_child_samples': 87}
    # ????????? 17.75303574840021
    # ????????????2885.906862974167???

    # fig bug
    # {'reg_alpha': 0.006013503575632864, 'reg_lambda': 0.004075989478302265, 'num_leaves': 16, 'colsample_bytree': 0.5064800798900133, 'subsample': 0.7871572722868186, 'subsample_freq': 1, 'min_child_samples': 93}
    # ????????? 17.728024045874047
    # ????????????5492.3369576931???


def tune_svr(df, emb_path="swint_embed"):
    # TODO Add SVR param tune
    print("svr tune start")
    start = time.time()

    def bayes_objective(trial):

        params = {
            "gamma": trial.suggest_loguniform("gamma", 1e-5, 1e5),
            "C": trial.suggest_loguniform("C", 1e-5, 1e5),
            "epsilon": trial.suggest_loguniform("epsilon", 1e-5, 1e5),
        }
        # ?????????????????????????????????
        model = SVR(**params)

        # cross_val_score?????????????????????????????????
        skf = StratifiedKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.seed
        )

        rmse_list = []
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(df["Id"], df["Pawpularity"])
        ):

            svr = deepcopy(model)

            train_df = df.loc[train_idx].reset_index(drop=True)
            val_df = df.loc[val_idx].reset_index(drop=True)

            train_embed = np.load(f"{emb_path}/swint_embed_train_{fold}.npy")
            val_embed = np.load(f"{emb_path}/swint_embed_val_{fold}.npy")

            svr.fit(
                train_embed.astype("float32"),
                train_df["Pawpularity"].astype("int32"),
            )

            pred = svr.predict(val_embed)
            rmse = np.sqrt(mean_squared_error(val_df["Pawpularity"], pred))
            rmse_list.append(rmse)

        rmse = np.mean(rmse_list)

        return rmse

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(bayes_objective, n_trials=100, n_jobs=1)

    best_params = study.best_trial.params
    best_score = study.best_trial.value
    print(f"????????????????????? {best_params}\n????????? {best_score}")
    print(f"????????????{time.time() - start}???")

    # {"gamma": 0.006550458887347427,"C": 5.7273276958907715,"epsilon": 6.834062667406395,}
    # ????????? 17.68122135877512
    # ????????????465.755384683609???


def main():
    torch.autograd.set_detect_anomaly(True)
    seed_everything(config.seed)  # seed??????

    # df = pd.read_csv(os.path.join(config.root, "train.csv"))
    # experiment(df, "test")

    model_path = "model_submission_7"
    df = pd.read_csv(os.path.join(config.root, "train.csv"))
    train_ensemble(df, model_path)
    # tune_lightgbm(df)
    # tune_svr(df)

    # df_test = pd.read_csv(os.path.join(config.root, "test.csv"))
    # prediction = inference_ensemble(df_test, model_path)
    # print(prediction)

    # df_test = pd.read_csv(os.path.join(config.root, "test.csv"))
    # prediction = inference_ensemble_state_dict(
    #    df_test.copy(),
    #    "model_submission_6",
    #    mode="swint_svr",
    # )
    # print(prediction)
    # make_submission(df_test, prediction, ".")


if __name__ == "__main__":
    main()
