from src.models.components.variational_diffusion import EquivariantVariationalDiffusion, GCPNetAutoEncoder, H_INPUT_TYPE
import torch.nn as nn
import math
import torch
import numpy as np

import torch.nn.functional as F

from omegaconf import DictConfig
from random import random
from torch_geometric.data import Batch
from torch_scatter import scatter
from typing import Any, Dict, List, Optional, Tuple, Union

from src.models.components import centralize, num_nodes_to_batch_index
from src.models import NumNodesDistribution, inflate_batch_array

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from src.utils import make_and_save_network_graphviz_plot
from src.utils.pylogger import get_pylogger
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

"""
需要重写  mol_gen_sample和训练的函数
"""
class EnLatentDiffusion(EquivariantVariationalDiffusion):
    def __init__(
        self,
        model_cfg: DictConfig,
        module_cfg: DictConfig,
        layer_cfg: DictConfig,
        diffusion_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        dynamics_network,
        dataset_info: Dict[str, Any]):

        super().__init__(dynamics_network=dynamics_network,diffusion_cfg=diffusion_cfg,dataloader_cfg=dataloader_cfg,dataset_info=dataset_info)
        self.vae=GCPNetAutoEncoder(
        model_cfg,
        module_cfg,
        layer_cfg,
        diffusion_cfg,
        dataloader_cfg) #.requires_grad_(False)

    def sample_combined_position_feature_noise(
        self,
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        generate_x_only: bool = False
    ) -> TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]:
        z_x = self.sample_center_gravity_zero_gaussian_with_mask(
            size=(len(batch_index), self.num_x_dims),
            batch_index=batch_index,
            node_mask=node_mask,
            device=batch_index.device
        )
        if generate_x_only:
            # bypass calculations for `h`
            return z_x
        z_h = self.sample_gaussian_with_mask(
            size=(len(batch_index), self.vae.encoded_h_dim),
            node_mask=node_mask,
            device=batch_index.device
        )
        z = torch.cat([z_x, z_h], dim=-1)
        return z
    def log_pxh_given_z0_without_constants(
        self,
        h: Union[Dict[str, H_INPUT_TYPE], H_INPUT_TYPE],
        z_0: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        eps: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        net_out: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        gamma_0: TensorType["batch_size", 1],
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        device: Union[torch.device, str],
        generate_x_only: bool = False,
        epsilon: float = 1e-10
    ) -> Tuple[
        TensorType["batch_size"],
        Optional[TensorType["batch_size"]]
    ]:
        eps_x = eps[:, :self.num_x_dims]
        net_x = net_out[:, :self.num_x_dims]

        eps_h=eps[:,self.num_x_dims:]
        net_h=eps[:,self.num_x_dims:]

        log_p_x_given_z0_without_constants = (
                -0.5 * self.sum_node_features_except_batch(
            (eps_x - net_x) ** 2,
            batch_index
        )
        )

        log_p_h_given_z0_without_constants= (
                -0.5 * self.sum_node_features_except_batch(
            (eps_h - net_h) ** 2,
            batch_index
        )
        )

        return log_p_x_given_z0_without_constants, log_p_h_given_z0_without_constants
    def forward(self, batch: Batch, return_loss_info: bool = False) -> Tuple[Any, ...]:
        batch.h = {"categorical": batch.one_hot, "integer": batch.charges}
        h_saved=batch.h
        x_saved=batch.x
        self.vae.encode(batch)
        batch.x=batch.x.detach()
        batch.h=batch.h.detach()

        #compute reconstruction error
        x_recon, h_recon = self.vae.decode_without_roundint_and_onehot(batch)
        error_recon=self.vae.compute_reconstruction_error(x_saved,x_recon,h_saved,h_recon,batch.batch,batch.mask,num_nodes=batch.num_nodes,batch_size=batch.batch_size)

        batch_index, batch_size, num_nodes, node_mask = (
            batch.batch, batch.num_graphs, batch.num_nodes_present, batch.mask
        )
        delta_log_px = self.delta_log_px(num_nodes)

        if self.training and self.diffusion_cfg.loss_type == "l2":
            delta_log_px = torch.zeros_like(delta_log_px)

        lowest_t = 0 if self.training else 1
        t_int = torch.randint(
            lowest_t,
            self.T + 1,
            size=(batch.num_graphs, 1),
            device=batch.x.device
        )

        s_int = t_int - 1  # previous timestep
        # note: these are important for computing log p(x | z0)
        t_is_zero = (t_int == 0).float()
        t_is_not_zero = 1 - t_is_zero
        s = s_int / self.T
        t = t_int / self.T
        gamma_s = inflate_batch_array(self.gamma(s), batch.x)
        gamma_t = inflate_batch_array(self.gamma(t), batch.x)

        batch.h={'categorical': torch.zeros(0).float().to(device=device), 'integer': batch.h}
        #需要修改dynamic接受的的h长度

        xh = torch.cat([batch.x, batch.h["categorical"], batch.h["integer"].reshape(-1, 1)], dim=-1)

        z_t, eps_t = self.compute_noised_representation(xh, batch_index, batch.mask, gamma_t)
        self_cond = None
        """
        self_condition: todo
        self_cond = None
        self_conditioning = (
            self.training and self.diffusion_cfg.self_condition and not (t_int == self.T).any() and random() < self_conditioning_prob
        )
        if self_conditioning:
            with torch.no_grad():
                s_array_self_cond = torch.full((batch_size, 1), fill_value=0, device=t_int.device) / self.T
                t_array_self_cond = (t_int + 1) / self.T
                gamma_t_self_cond = inflate_batch_array(self.gamma(t_array_self_cond), batch.x)
                z_t_self_cond, _ = self.compute_noised_representation(xh, batch_index, batch.mask, gamma_t_self_cond)

                self_cond = self.sample_p_zs_given_zt(
                    s=s_array_self_cond,
                    t=t_array_self_cond,
                    z=z_t_self_cond,
                    batch_index=batch_index,
                    node_mask=node_mask,
                    context=getattr(batch, "props_context", None),
                    fix_noise=fix_self_conditioning_noise,
                    generate_x_only=False,
                    self_condition=True
                ).detach_()

        """
        _, net_out = self.dynamics_network(batch, z_t, t[batch_index], xh_self_cond=self_cond)
        error_t = self.sum_node_features_except_batch((eps_t - net_out) ** 2, batch_index)
        if self.training and self.diffusion_cfg.loss_type == "l2":
            SNR_weight = torch.ones_like(error_t)
        else:
            # compute weighting with SNR: (SNR(s - t) - 1) for epsilon parametrization
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(-1)
        assert error_t.shape == SNR_weight.shape
        neg_log_constants = -self.log_constants_p_x_given_z0(
            num_nodes=num_nodes,
            device=batch.x.device
        )
        if self.training and self.diffusion_cfg.loss_type == "l2":
            neg_log_constants = torch.zeros_like(neg_log_constants)
        kl_prior = self.compute_kl_prior(
            xh,
            batch_index=batch_index,
            node_mask=node_mask,
            num_nodes=num_nodes,
            device=batch.x.device
        )
        if self.training:
            # compute the `L_0` term (even if `gamma_t` is not actually `gamma_0`),
            # as this will later be selected via masking
            log_p_x_given_z0_without_constants, log_ph_given_z0 = (
                self.log_pxh_given_z0_without_constants(
                    h=batch.h,
                    z_0=z_t,
                    eps=eps_t,
                    net_out=net_out,
                    gamma_0=gamma_t,
                    batch_index=batch_index,
                    node_mask=node_mask,
                    device=batch.x.device
                )
            )

            loss_0_x = (
                    -log_p_x_given_z0_without_constants * t_is_zero.squeeze()
            )
            loss_0_h = (
                    -log_ph_given_z0 * t_is_zero.squeeze()
            )

            # apply `t_is_zero` mask
            error_t = error_t * t_is_not_zero.squeeze()

        else:
            # compute noise values for `t = 0`
            t_zeros = torch.zeros_like(s)
            gamma_0 = inflate_batch_array(self.gamma(t_zeros), batch.x)

            # sample `z_0` given `x`, `h` for timestep `t`, from q(`z_t` | `x`, `h`)
            z_0, eps_0 = self.compute_noised_representation(xh, batch_index, batch.mask, gamma_0)

            _, net_out_0 = self.dynamics_network(batch, z_0, t_zeros[batch_index])

            log_p_x_given_z0_without_constants, log_ph_given_z0 = (
                self.log_pxh_given_z0_without_constants(
                    h=batch.h,
                    z_0=z_0,
                    eps=eps_0,
                    net_out=net_out_0,
                    gamma_0=gamma_0,
                    batch_index=batch_index,
                    node_mask=node_mask,
                    device=batch.x.device
                )
            )
            loss_0_x = -log_p_x_given_z0_without_constants
            loss_0_h = -log_ph_given_z0
        log_pN = self.log_pN(num_nodes)
        loss_terms = (
            delta_log_px, error_t, SNR_weight,
            loss_0_x, loss_0_h, neg_log_constants,
            kl_prior, log_pN, t_int.squeeze(), error_recon
        )

        if return_loss_info:
            loss_info = {
                "eps_hat_x": scatter(
                    net_out[:, :self.num_x_dims].abs().mean(-1),
                    batch_index,
                    dim=0,
                    reduce="mean"
                ).mean(),
                "eps_hat_h": scatter(
                    net_out[:, self.num_x_dims:].abs().mean(-1),
                    batch_index,
                    dim=0,
                    reduce="mean"
                ).mean()
            }
            return (*loss_terms, loss_info)

        return loss_terms

    def sample_p_xh_given_z0(
        self,
        z_0: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        batch_size: int,
        batch: Optional[Batch] = None,
        context: Optional[TensorType["batch_num_nodes", "num_context_features"]] = None,
        fix_noise: bool = False,
        generate_x_only: bool = False,
        xh_self_cond: Optional[TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]] = None
    ) -> Union[
        Tuple[
            TensorType["batch_num_nodes", 3],
            Dict[
                str,
                torch.Tensor  # note: for when `include_charges=False`
            ]
        ],
        TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]
    ]:
        """Sample `x` ~ p(x | z0)."""
        t_zeros = torch.zeros(size=(batch_size, 1), device=batch_index.device)
        gamma_0 = self.gamma(t_zeros)

        # compute sqrt(`sigma_0` ^ 2 / `alpha_0` ^ 2)
        sigma_x = self.SNR(-0.5 * gamma_0)

        # construct batch input object (e.g., when using a molecule generation DDPM)
        if batch is None:
            batch = Batch(batch=batch_index, mask=node_mask, props_context=context)

        # make network prediction
        _, net_out = self.dynamics_network(
            batch,
            z_0,
            t_zeros[batch_index],
            x_self_cond=xh_self_cond,
            xh_self_cond=xh_self_cond
        )

        # compute `mu` for p(zs | zt)
        mu_x = self.compute_x_pred(z_0, net_out, gamma_0, batch_index)
        xh = self.sample_normal(
            mu=mu_x,
            sigma=sigma_x,
            batch_index=batch_index,
            node_mask=node_mask,
            fix_noise=fix_noise,
            generate_x_only=generate_x_only
        )

        x = xh[:, :self.num_x_dims]

        # bypass scalar predictions for nodes
        if generate_x_only:
            return x, {}
        h={'integer':xh[:,self.num_x_dims:].to(x.device),'categorical':torch.zeros(0).to(float).to(x.device)}
        return x,h
    def mol_gen_sample(
        self,
        num_samples: int,
        num_nodes: TensorType["batch_size"],
        device: Union[torch.device, str],
        return_frames: int = 1,
        num_timesteps: Optional[int] = None,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        fix_noise: bool = False,
        generate_x_only: bool = False,
        fix_self_conditioning_noise: bool = False
    ) -> Tuple[
        Union[
            TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
            TensorType["num_timesteps", "batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]
        ],
        TensorType["batch_num_nodes"],
        TensorType["batch_num_nodes"]
    ]:
        xh,batch_idx,node_mask=super().mol_gen_sample(num_samples=num_samples,num_nodes=num_nodes,device=device,context=context)
        x=xh[:, :self.num_x_dims]
        h=xh[:,self.num_x_dims:]
        unique_idx = torch.unique(batch_idx)
        graph_x= [x[batch_idx == i] for i in unique_idx]
        data_list = [Data(x=x) for x in graph_x]
        batch = Batch.from_data_list(data_list)
        batch.mask=node_mask
        batch.h=h
        if context is not None:
            context = context[batch.batch]
            context = context * node_mask.float().unsqueeze(-1)
            batch.props_context=context
        x_recon, h_recon=self.vae.decode_without_roundint_and_onehot(batch)
        if self.include_charges:
            xh_recon=torch.cat([x, h_recon["categorical"], h_recon["integer"].unsqueeze(-1)], dim=-1)
        else:
            xh_recon = torch.cat([x, h_recon["categorical"]], dim=-1)
        return xh_recon,batch_idx,node_mask