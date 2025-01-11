# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import math
import os
import torch
import torchmetrics

import torch.nn.functional as F

from time import time
from pathlib import Path
from rdkit import Chem
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch_geometric.data import Batch
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from omegaconf import DictConfig
from torch_scatter import scatter

import src.datamodules.components.edm.utils as qm9utils

from src.models.components import centralize, num_nodes_to_batch_index, save_xyz_file, visualize_mol, visualize_mol_chain
from src.datamodules.components.edm.rdkit_functions import BasicMolecularMetrics, build_molecule, process_molecule
from src.datamodules.components.edm.datasets_config import GEOM_NO_H, GEOM_WITH_H
from src.datamodules.components.edm import check_molecular_stability, get_bond_length_arrays
from src.models.components.egnn import EGNNDynamics
from src.models.components.variational_diffusion import EquivariantVariationalDiffusion

from src.models.components.gcpnet import GCPNetDynamics, GCPNetEncoder, GCPNetDecoder
from src.models import HALT_FILE_EXTENSION, CategoricalDistribution, PropertiesDistribution, Queue, batch_tensor_to_list, compute_mean_mad, get_grad_norm, log_grad_flow_lite, reverse_tensor

from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard

from src.utils.pylogger import get_pylogger
from src.models.components.EnLatentDiffusion import EnLatentDiffusion
patch_typeguard()  # use before @typechecked


log = get_pylogger(__name__)

class GEOMMoleculeGenerationDDPM(torch.nn.Module):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            model_cfg: DictConfig,
            module_cfg: DictConfig,
            layer_cfg: DictConfig,
            diffusion_cfg: DictConfig,
            dataloader_cfg: DictConfig,
            path_cfg: DictConfig = None,
            **kwargs
    ):
        super().__init__()

        # hyperparameters #
        self.model_cfg=model_cfg
        self.module_cfg=module_cfg
        self.layer_cfg=layer_cfg
        self.diffusion_cfg=diffusion_cfg
        self.dataloader_cfg=dataloader_cfg
        # prior to saving hyperparameters, adjust number of evaluation samples based on one's conditioning argument(s)
        diffusion_cfg.num_eval_samples = (
            diffusion_cfg.num_eval_samples // 2 if len(module_cfg.conditioning) > 0 else diffusion_cfg.num_eval_samples
        )

        # also prior to saving hyperparameters, adjust the effective number of atom types used
        dataloader_cfg.num_atom_types = (
            dataloader_cfg.num_atom_types - 1
            if dataloader_cfg.remove_h
            else dataloader_cfg.num_atom_types
        )

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)

        # DDPM
        ddpm_modes = {
            "unconditional": EquivariantVariationalDiffusion,
            "inpainting": EquivariantVariationalDiffusion,
            "latent": EnLatentDiffusion
        }
        self.ddpm_mode = diffusion_cfg.ddpm_mode
        assert self.ddpm_mode in ddpm_modes, f"Selected DDPM mode {self.ddpm_mode} is currently not supported."

        dynamics_networks = {
            "gcpnet": GCPNetDynamics,
            "egnn": EGNNDynamics,
            "encoder": GCPNetEncoder,
            "decoder": GCPNetDecoder
        }
        assert diffusion_cfg.dynamics_network in dynamics_networks, f"Selected dynamics network {diffusion_cfg.dynamics_network} is currently not supported."

        self.T = diffusion_cfg.num_timesteps
        self.loss_type = diffusion_cfg.loss_type
        self.num_atom_types = dataloader_cfg.num_atom_types
        self.num_x_dims = dataloader_cfg.num_x_dims
        self.include_charges = dataloader_cfg.include_charges
        self.condition_on_context = len(module_cfg.conditioning) > 0

        # dataset metadata
        dataset_info_mapping = {"GEOM": GEOM_NO_H if dataloader_cfg.remove_h else GEOM_WITH_H}
        self.dataset_info = dataset_info_mapping[dataloader_cfg.dataset]

        # PyTorch modules #
        dynamics_network = dynamics_networks[diffusion_cfg.dynamics_network](
            model_cfg=model_cfg,
            module_cfg=module_cfg,
            layer_cfg=layer_cfg,
            diffusion_cfg=diffusion_cfg,
            dataloader_cfg=dataloader_cfg
        )
        encoder=dynamics_networks['encoder'](
            model_cfg=model_cfg,
            module_cfg=module_cfg,
            layer_cfg=layer_cfg,
            diffusion_cfg=diffusion_cfg,
            dataloader_cfg=dataloader_cfg
        )
        decoder=dynamics_networks['decoder'](
            model_cfg=model_cfg,
            module_cfg=module_cfg,
            layer_cfg=layer_cfg,
            diffusion_cfg=diffusion_cfg,
            dataloader_cfg=dataloader_cfg
        )

        # self.ddpm = ddpm_modes[self.ddpm_mode](
        #     dynamics_network=dynamics_network,
        #     diffusion_cfg=diffusion_cfg,
        #     dataloader_cfg=dataloader_cfg,
        #     dataset_info=self.dataset_info
        # )
        self.ddpm = EnLatentDiffusion(
            model_cfg,
            module_cfg,
            layer_cfg,
            diffusion_cfg,
            dataloader_cfg,
            dynamics_network=dynamics_network,
            dataset_info=self.dataset_info
        )
        # distributions #
        self.node_type_distribution = CategoricalDistribution(
            self.dataset_info["atom_types"],
            self.dataset_info["atom_encoder"]
        )

        # training #
        if module_cfg.clip_gradients:
            self.gradnorm_queue = Queue()
            self.gradnorm_queue.add(3000)  # add large value that will be flushed

        # metrics #
        self.train_phase, self.val_phase, self.test_phase = "train", "val", "test"
        self.phases = [self.train_phase, self.val_phase, self.test_phase]
        self.metrics_to_monitor = [
            "loss", "loss_t", "SNR_weight", "loss_0",
            "kl_prior", "delta_log_px", "neg_log_const_0", "log_pN",
            "eps_hat_x", "eps_hat_h","error_recon"
        ]
        self.eval_metrics_to_monitor = self.metrics_to_monitor + ["log_SNR_max", "log_SNR_min"]
        for phase in self.phases:
            metrics_to_monitor = (
                self.metrics_to_monitor
                if phase == self.train_phase
                else self.eval_metrics_to_monitor
            )
            for metric in metrics_to_monitor:
                # note: individual metrics e.g., for averaging loss across batches
                setattr(self, f"{phase}_{metric}", torchmetrics.MeanMetric())

        # sample metrics
        if (dataloader_cfg.smiles_filepath is None) or not os.path.exists(dataloader_cfg.smiles_filepath):
            smiles_list = None
        else:
            with open(dataloader_cfg.smiles_filepath, "r") as f:
                smiles_list = f.read().split("\n")
                smiles_list.remove("")  # remove last line (i.e., an empty entry)
        self.molecular_metrics = BasicMolecularMetrics(
            self.dataset_info,
            data_dir=dataloader_cfg.data_dir,
            dataset_smiles_list=smiles_list
        )
    @typechecked
    def forward(
        self,
        batch: Batch,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[
        torch.Tensor,
        Dict[str, Any]
    ]:
        """
        Compute the loss (type L2 or negative log-likelihood (NLL)) if `training`.
        If `eval`, then always compute NLL.
        """
        # centralize node positions to make them translation-invariant
        _, batch.x = centralize(
            batch,
            key="x",
            batch_index=batch.batch,
            node_mask=batch.mask,
            edm=True
        )

        # construct invariant node features
        batch.h = {"categorical": batch.one_hot, "integer": batch.charges}

        # derive property contexts (i.e., conditionals)
        if self.condition_on_context:
            batch.props_context = qm9utils.prepare_context(
                list(self.module_cfg.conditioning),
                batch,
                self.props_norms
            ).type(dtype)
        else:
            batch.props_context = None

        # derive node counts per batch
        num_nodes = scatter(batch.mask.int(), batch.batch, dim=0, reduce="sum")
        batch.num_nodes_present = num_nodes

        # note: `L` terms in e.g., the GCDM paper represent log-likelihoods,
        # while our loss terms are negative (!) log-likelihoods
        (
            delta_log_px, error_t, SNR_weight,
            loss_0_x, loss_0_h, neg_log_const_0,
            kl_prior, log_pN, t_int, error_recon,loss_info
        ) = self.ddpm(batch, return_loss_info=True)

        # support L2 loss training step
        if self.training and self.loss_type == "l2":
            # normalize `loss_t`
            effective_num_nodes = (
                num_nodes.max()
                if self.diffusion_cfg.norm_training_by_max_nodes
                else num_nodes
            )
            denom = (self.num_x_dims + self.ddpm.num_node_scalar_features) * effective_num_nodes
            error_t = error_t / denom
            loss_t = 0.5 * error_t

            # normalize `loss_0` via `loss_0_x` normalization
            loss_0_x = loss_0_x / denom
            loss_0 = loss_0_x + loss_0_h

        # support VLB objective or evaluation step
        else:
            loss_t = self.T * 0.5 * SNR_weight * error_t
            loss_0 = loss_0_x + loss_0_h
            loss_0 = loss_0 + neg_log_const_0

        # combine loss terms
        nll = loss_t + loss_0 + kl_prior

        # correct for normalization on `x`
        nll = nll - delta_log_px

        # transform conditional `nll` into joint `nll`
        # note: loss = -log p(x,h|N) and log p(x,h,N) = log p(x,h|N) + log p(N);
        # therefore, log p(x,h,N) = -loss + log p(N)
        # => loss_new = -log p(x,h,N) = loss - log p(N)
        nll = nll - log_pN

        # collect all metrics' batch-averaged values
        local_variables = locals()
        for metric in self.metrics_to_monitor:
            if metric in ["eps_hat_x", "eps_hat_h"]:
                continue
            if metric != "loss":
                loss_info[metric] = local_variables[metric].mean(0)

        return nll, loss_info
    def compute_loss(self, batch: Batch) :
        nll, metrics_dict = self(batch)
        # ensure all intermediate losses to be logged as metrics have their gradients ignored
        metrics_dict2 = {key: value.detach() for key, value in metrics_dict.items()}

        # calculate standard NLL from forward KL-divergence while preserving its gradients
        metrics_dict2["loss"] = nll.mean(0)+metrics_dict["error_recon"].mean(0)

        # update metrics
        for metric in self.metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.train_phase}_{metric}")
            torchmetric(metrics_dict2[metric])
        return nll.mean(0).detach(),metrics_dict2["error_recon"].mean(0),metrics_dict2['loss']