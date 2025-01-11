import os
import hydra
import pyrootutils
import torch
import prody as pr
import pytorch_lightning as pl

from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
#from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers import Logger
LightningLoggerBase=Logger
from typing import List, Optional, Tuple

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src import LR_SCHEDULER_MANUAL_INTERPOLATION_HELPER_CONFIG_ITEMS, LR_SCHEDULER_MANUAL_INTERPOLATION_PRIMARY_CONFIG_ITEMS, MODEL_WATCHING_LOGGERS, unwatch_model, utils, watch_model
from src.utils.pylogger import get_pylogger

pr.confProDy(verbosity="none")

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is an optional line at the top of each entry file
# that helps to make the environment more robust and convenient
#
# the main advantages are:
# - allows you to keep all entry files in "src/" without installing project as a package
# - makes paths and scripts always work no matter where is your current work dir
# - automatically loads environment variables from ".env" file if exists
#
# how it works:
# - the line above recursively searches for either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
#   to make all paths always relative to the project root
# - loads environment variables from ".env" file in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. simply remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
# 3. always run entry files from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #


log = get_pylogger(__name__)




@hydra.main(version_base="1.3.2", config_path="../configs", config_name="train_torch.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # work around Hydra's (current) lack of support for arithmetic expressions with interpolated config variables
    # reference: https://github.com/facebookresearch/hydra/issues/1286
    """
    if cfg.model.get("scheduler") is not None:
        for key in cfg.model.scheduler.keys():
            if key in LR_SCHEDULER_MANUAL_INTERPOLATION_PRIMARY_CONFIG_ITEMS:
                setattr(cfg.model.scheduler, key, eval(cfg.model.scheduler.get(key)))
        # ensure that all requested arithmetic expressions have been performed using interpolated config variables
        lr_scheduler_key_names = [name for name in cfg.model.scheduler.keys()]
        for key in lr_scheduler_key_names:
            if key in LR_SCHEDULER_MANUAL_INTERPOLATION_HELPER_CONFIG_ITEMS:
                delattr(cfg.model.scheduler, key)
    """
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    dataloaders = [
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader()
    ]
    splits = ["train", "valid", "test"]
    dataloaders = {split: dataloader for (split, dataloader) in zip(splits, dataloaders)}
    data1 = next(iter(dataloaders["train"]))
    data2=data1
    model = hydra.utils.instantiate(
        cfg.model,
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
        layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
        diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
        dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
        path_cfg=cfg.paths
    )




    end=model.compute_loss(data1) #计算出一轮正确的损失


    splits = ["train", "valid", "test"]
    splits = ["train", "valid", "test"]
    splits = ["train", "valid", "test"]





if __name__ == "__main__":
    main()
