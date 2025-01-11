# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import hydra
import os
import pyrootutils
import torch

import numpy as np
import prody as pr

from omegaconf import DictConfig
from pytorch_lightning.loggers import Logger
LightningLoggerBase=Logger
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from typing import List, Tuple
import matplotlib.pyplot as plt
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src.datamodules.components.edm import get_bond_length_arrays
from src.utils.pylogger import get_pylogger

from src import LR_SCHEDULER_MANUAL_INTERPOLATION_HELPER_CONFIG_ITEMS, LR_SCHEDULER_MANUAL_INTERPOLATION_PRIMARY_CONFIG_ITEMS, utils

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

def compute_mean_mad_from_dataloader(
    dataloader,
    properties
) :
    property_norms = {}
    property_key = properties
    values = dataloader.dataset.data[property_key]

    mean = torch.mean(values)
    ma = torch.abs(values - mean)
    mad = torch.mean(ma)
    var= torch.var(values)
    return mean, mad,var
@utils.task_wrapper
def evaluate(cfg: DictConfig) :
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # assert cfg.get("ckpt_path") is not None and os.path.exists(cfg.ckpt_path)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()
    splits = ["train", "valid", "test"]
    dataloaders = [
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader()
    ]
    dataloaders = {split: dataloader for (split, dataloader) in zip(splits, dataloaders)}
    mean,mad,var=compute_mean_mad_from_dataloader(dataloaders['train'],'alpha')
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(
        cfg.model,
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
        layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
        diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
        dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
        path_cfg=cfg.paths
    )

    log.info("Loading checkpoint!")
    device = (
        ("cuda:1" if isinstance(cfg.trainer.devices, int) else f"cuda:{cfg.trainer.devices[1]}")
        if torch.cuda.is_available()
        else "cpu"
    )
    """
    model = model.load_from_checkpoint(
        # allow one to evaluate with an older model using custom hyperparameters
        checkpoint_path=cfg.ckpt_path,
        map_location=device,
        strict=False,
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
        layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
        diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
        dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
        path_cfg=cfg.paths
    )
    """
    model = model.to(device)
    file_name = "/tmp/pycharm_project_80/checkpoint_latent_without_trained_vae/" + "best_loss_mini_batch.pth"
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    # ensure valid bond lengths have been added to each dataset's metadata collection (i.e., `model.dataset_info`)
    if any([
        not getattr(model.dataset_info, "bonds1", None),
        not getattr(model.dataset_info, "bonds2", None),
        not getattr(model.dataset_info, "bonds3", None)
    ]):
        bonds = get_bond_length_arrays(model.dataset_info["atom_encoder"])
        model.dataset_info["bonds1"], model.dataset_info["bonds2"], model.dataset_info["bonds3"] = (
            bonds[0], bonds[1], bonds[2]
        )

    log.info("Starting sampling!")

    p=model.sample_and_compute_alpha(mean,mad,num_samples=1000)
    torch.save(p,'/tmp/pycharm_project_80/data_for_distribution/sample_alpha.pth')
    return {},{}


@hydra.main(version_base="1.2", config_path= "../configs", config_name="mol_gen_eval_torch_latent.yaml")
def main(cfg: DictConfig) -> None:
    # work around Hydra's (current) lack of support for arithmetic expressions with interpolated config variables
    # reference: https://github.com/facebookresearch/hydra/issues/1286
    if cfg.model.get("scheduler") is not None:
        for key in cfg.model.scheduler.keys():
            if key in LR_SCHEDULER_MANUAL_INTERPOLATION_PRIMARY_CONFIG_ITEMS:
                setattr(cfg.model.scheduler, key, eval(cfg.model.scheduler.get(key)))
        # ensure that all requested arithmetic expressions have been performed using interpolated config variables
        lr_scheduler_key_names = [name for name in cfg.model.scheduler.keys()]
        for key in lr_scheduler_key_names:
            if key in LR_SCHEDULER_MANUAL_INTERPOLATION_HELPER_CONFIG_ITEMS:
                delattr(cfg.model.scheduler, key)

    evaluate(cfg)


if __name__ == "__main__":
    main()
