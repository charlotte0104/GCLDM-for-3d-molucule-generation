# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import hydra
import pyrootutils
import torch
import prody as pr
import pytorch_lightning as pl

from tqdm import tqdm
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
#from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers import Logger
LightningLoggerBase=Logger
from typing import List, Optional, Tuple
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

def save_checkpoint(model,optimizer,epoch,best_flag=False,mini_batch_flag=False):
    checkpoint={
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if best_flag==False:
        file_name="/tmp/pycharm_project_80/checkpoint_latent_without_trained_vae/mini_2000/"+"{}_epoch.pth".format(epoch)
    else:
        if mini_batch_flag == True :
            file_name="/tmp/pycharm_project_442/checkpoint_latent_without_trained_vae/mini_2000/"+"best_loss_mini_batch.pth"
        else :
            file_name = "/tmp/pycharm_project_442/checkpoint_latent_without_trained_vae/mini_2000/" + "best_loss.pth"
    #/tmp/pycharm_project_763/checkpoint/
    torch.save(checkpoint, file_name)
@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    max_epoch=1000
    max_epoch_mini_batch=1000
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)


    log.info(f"Instantiating model <{cfg.model._target_}>")
    context_info={}
    model = hydra.utils.instantiate(
        cfg.model,
        context_info=context_info,
        model_cfg=hydra.utils.instantiate(cfg.model.model_cfg),
        module_cfg=hydra.utils.instantiate(cfg.model.module_cfg),
        layer_cfg=hydra.utils.instantiate(cfg.model.layer_cfg),
        diffusion_cfg=hydra.utils.instantiate(cfg.model.diffusion_cfg),
        dataloader_cfg=getattr(cfg.datamodule, "dataloader_cfg", None),
        path_cfg=cfg.paths
    )
    model.to(device)

    # model_para_path="/tmp/pycharm_project80/checkpoint_latent_without_trained_vae/mini_2000/save/"+"best_loss_mini_batch.pth"
    # model_trained_params=torch.load(model_para_path,map_location='cuda:0')
    # epoch_now=model_trained_params['epoch']
    # model.load_state_dict(model_trained_params['model_state_dict'])
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, amsgrad=True,weight_decay=1e-12)
    # optim.load_state_dict(model_trained_params['optimizer_state_dict'])
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epoch)
    datamodule.setup()
    dataloaders = [
            datamodule.train_dataloader(),
            datamodule.val_dataloader(),
            datamodule.test_dataloader()
        ]
    splits = ["train", "valid", "test"]
    dataloaders = {split: dataloader for (split, dataloader) in zip(splits, dataloaders)}
    # save_checkpoint(model, optim, 0)
    for epoch in range(2000):
        batch_idx=0
        print('> epoch %s:' % str(epoch).zfill(3))
        model.train()
        loss_total=0
        loss_diff_total=0
        loss_recon_total=0
        loop= tqdm(dataloaders["train"])
        loop_val = tqdm(dataloaders['valid'])

        loss_best_val = 10000
        for data in loop:
            data.to(device)
            loss_diff,loss_recon,loss_all = model.compute_loss(data)
            loss=loss_all
            optim.zero_grad()
            loss.backward()
            optim.step()
            # print(f"loss_recon of batch_{batch_idx} is {loss_recon}")
            loss_total=loss_total+loss
            loss_diff_total=loss_diff_total+loss_diff
            loss_recon_total=loss_recon_total+loss_recon
            batch_idx+=1
        # if (epoch+1) % 100 == 0:
        #     save_checkpoint(model, optim, epoch)
        if epoch+1 >= 1500 and epoch+1 /50==0:
        # if True:

            loss_diff_total_val = 0

            for data in loop_val:
                with torch.no_grad():
                    loss_diff_val, loss_recon_val, loss_all_val = model.compute_loss(data)

                    loss_diff_total_val=loss_diff_total_val+loss_diff_val

            if epoch+1==1500:
                save_checkpoint(model, optim, epoch, True, True)
            if loss_diff_total_val<loss_best_val:
                loss_best_val=loss_diff_val
                save_checkpoint(model, optim, epoch, True, True)
        print(f"mean_loss of epoch_{epoch} : {loss_total/batch_idx}")
        print(f"mean_loss_recon of epoch_{epoch} : {loss_recon_total / batch_idx}")
        print(f"mean_loss_diff of epoch_{epoch} : {loss_diff_total / batch_idx}")






@hydra.main(version_base="1.3.2", config_path="../configs", config_name="train_torch_latent.yaml")
def main(cfg: DictConfig) -> Optional[float]:
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

    # train the model
    train(cfg)




if __name__ == "__main__":
    main()
