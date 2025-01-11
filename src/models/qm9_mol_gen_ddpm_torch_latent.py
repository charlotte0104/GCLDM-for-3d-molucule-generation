import math
import os
import torch
import torchmetrics
import pickle
import numpy as np
from src import get_classifier_model, get_classifier_adj_matrix
import torch.nn.functional as F

from time import time, strftime
from pathlib import Path
from rdkit import Chem
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch_geometric.data import Batch
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import omegaconf
from omegaconf import DictConfig
from torch_scatter import scatter

import src.datamodules.components.edm.utils as qm9utils

from src.models.components import centralize, num_nodes_to_batch_index, save_xyz_file, visualize_mol, visualize_mol_chain
from src.datamodules.components.edm.rdkit_functions import BasicMolecularMetrics, build_molecule, process_molecule
from src.datamodules.components.edm.datasets_config import QM9_SECOND_HALF, QM9_WITH_H, QM9_WITHOUT_H
from src.datamodules.components.edm import check_molecular_stability, get_bond_length_arrays
from src.models.components.egnn import EGNNDynamics
from src.models.components.variational_diffusion import EquivariantVariationalDiffusion, GCPNetAutoEncoder

from src.models.components.gcpnet import GCPNetDynamics, GCPNetEncoder, GCPNetDecoder
from src.models import HALT_FILE_EXTENSION, CategoricalDistribution, PropertiesDistribution, Queue, batch_tensor_to_list, compute_mean_mad, get_grad_norm, log_grad_flow_lite, reverse_tensor
from src.utils.pylogger import get_pylogger

from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from src.models.components.EnLatentDiffusion import EnLatentDiffusion


patch_typeguard()  # use before @typechecked


log = get_pylogger(__name__)


class QM9MoleculeGenerationDDPM(torch.nn.Module):
    """LightningModule for QM9 small molecule generation using a DDPM.

    This LightningModule organizes the PyTorch code into 9 sections:
        - Computations (init)
        - Forward (forward)
        - Step (step)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers and LR schedulers (configure_optimizers)
        - Gradient clipping (configure_gradient_clipping)
        - End of model training (on_fit_end)
    """

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
        context_info=None,
        **kwargs
    ):
        super().__init__()

        # hyperparameters #
        # if self.condition_on_context:
        #     context_info=omegaconf.OmegaConf.to_container(context_info)
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
        #self.save_hyperparameters(logger=False)

        # DDPM
        ddpm_modes = {
            "unconditional": EquivariantVariationalDiffusion,
            "inpainting": EquivariantVariationalDiffusion,
            "latent" : EnLatentDiffusion
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
        if self.condition_on_context and context_info:
            context_info=omegaconf.OmegaConf.to_container(context_info)
            self.props_norms=context_info['props_norms']
            self.props_distr = context_info['props_distr']
            self.num_context_node_feats = context_info['num_context_node_feats']
        """
         context_info = {
            'props_norms': props_norms,
            'props_distr': props_distr,
            'num_context_node_feats': num_context_node_feats
        }
        """
        # dataset metadata
        dataset_info_mapping = {
            "QM9": QM9_WITHOUT_H if dataloader_cfg.remove_h else QM9_WITH_H,
            "QM9_second_half": QM9_SECOND_HALF
        }
        self.dataset_info = dataset_info_mapping[dataloader_cfg.dataset]

        if dataloader_cfg.dataset == "QM9_second_half" and dataloader_cfg.remove_h:
            raise NotImplementedError(f"Missing config for dataset {dataloader_cfg.dataset} without hydrogen atoms")

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
        #     encoder=encoder,
        #     decoder=decoder,
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
        smiles_list = (
            None
            if (dataloader_cfg.smiles_filepath is None)
            or not os.path.exists(dataloader_cfg.smiles_filepath)
            else np.load(dataloader_cfg.smiles_filepath, allow_pickle=True)
        )
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
        #loss_diff,loss_recon,loss_total
    """
            metrics_dict2 = {key: value.detach() for key, value in metrics_dict.items()}

        # calculate standard NLL from forward KL-divergence while preserving its gradients
        metrics_dict2["loss"] = nll.mean(0)

        # update metrics
        for metric in self.metrics_to_monitor:
            # e.g., averaging loss across batches
            torchmetric = getattr(self, f"{self.train_phase}_{metric}")
            torchmetric(metrics_dict2[metric])
        self.training_step_outputs.append(metrics_dict2)
        self.log('train_loss', metrics_dict2["loss"],prog_bar=True)
        return metrics_dict2
    """


    def sample_and_analyze(
        self,
        num_samples: int,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        batch_size: Optional[int] = None,
        max_num_nodes: Optional[int] = 100
    ) -> Dict[str, Any]:


        max_num_nodes = (
            self.dataset_info["max_n_nodes"]
            if "max_n_nodes" in self.dataset_info
            else max_num_nodes
        )

        batch_size = self.dataloader_cfg.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, num_samples)

        # note: each item in `molecules` is a tuple of (`position`, `atom_type_encoded`)
        molecules, atom_types, charges = [], [], []
        for _ in range(math.ceil(num_samples / batch_size)):
            # node count-conditioning
            num_samples_batch = min(batch_size, num_samples - len(molecules))
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples_batch)

            assert int(num_nodes.max()) <= max_num_nodes

            # context-conditioning
            if self.condition_on_context:
                if context is None:
                    context = self.props_distr.sample_batch(num_nodes)
            else:
                context = None

            xh, batch_index, _ = self.ddpm.mol_gen_sample(
                num_samples=num_samples_batch,
                num_nodes=num_nodes,
                node_mask=node_mask,
                context=context,
                device=next(self.parameters()).device
            )

            x_ = xh[:, :self.num_x_dims].detach().cpu()
            atom_types_ = (
                xh[:, self.num_x_dims:-1].argmax(-1).detach().cpu()
                if self.include_charges
                else xh[:, self.num_x_dims:].argmax(-1).detach().cpu()
            )
            charges_ = (
                xh[:, -1]
                if self.include_charges
                else torch.zeros(0, device=self.device)
            )

            molecules.extend(
                list(
                    zip(
                        batch_tensor_to_list(x_, batch_index),
                        batch_tensor_to_list(atom_types_, batch_index)
                    )
                )
            )

            atom_types.extend(atom_types_.tolist())
            charges.extend(charges_.tolist())

        return self.analyze_samples(molecules, atom_types, charges)

    @typechecked
    def analyze_samples(
        self,
        molecules: List[Tuple[torch.Tensor, ...]],
        atom_types: List[int],
        charges: List[float]
    ) -> Dict[str, Any]:
        # assess distribution of node types
        kl_div_atom = (
            self.node_type_distribution.kl_divergence(atom_types)
            if self.node_type_distribution is not None
            else -1
        )

        # measure molecular stability
        molecule_stable, nr_stable_bonds, num_atoms = 0, 0, 0
        for pos, atom_type in molecules:
            validity_results = check_molecular_stability(
                positions=pos,
                atom_types=atom_type,
                dataset_info=self.dataset_info
            )
            molecule_stable += int(validity_results[0])
            nr_stable_bonds += int(validity_results[1])
            num_atoms += int(validity_results[2])

        fraction_mol_stable = molecule_stable / float(len(molecules))
        fraction_atm_stable = nr_stable_bonds / float(num_atoms)

        # collect other basic molecular metrics
        metrics = self.molecular_metrics.evaluate(molecules)
        validity, uniqueness, novelty = metrics[0], metrics[1], metrics[2]

        return {
            "kl_div_atom_types": kl_div_atom,
            "mol_stable": fraction_mol_stable,
            "atm_stable": fraction_atm_stable,
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty
        }

    @torch.inference_mode()
    @typechecked
    def sample(
        self,
        num_samples: int,
        num_nodes: Optional[TensorType["batch_size"]] = None,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        fix_noise: bool = False,
        num_timesteps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # node count-conditioning
        if num_nodes is None:
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
            max_num_nodes = (
                self.dataset_info["max_n_nodes"]
                if "max_n_nodes" in self.dataset_info
                else num_nodes.max().item()
            )
            assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        if self.condition_on_context:
            if context is None:
                context = self.props_distr.sample_batch(num_nodes)
        else:
            context = None

        # sampling
        xh, batch_index, _ = self.ddpm.mol_gen_sample(
            num_samples=num_samples,
            num_nodes=num_nodes,
            node_mask=node_mask,
            context=context,
            fix_noise=fix_noise,
            fix_self_conditioning_noise=fix_noise,
            device=next(self.parameters()).device,
            num_timesteps=num_timesteps
        )
        molecules, atom_types, charges = [], [], []

        # x_ = xh[:, :self.num_x_dims].detach().cpu()
        # atom_types_ = (
        #     xh[:, self.num_x_dims:-1].argmax(-1).detach().cpu()
        #     if self.include_charges
        #     else xh[:, self.num_x_dims:].argmax(-1).detach().cpu()
        # )
        # charges_ = (
        #     xh[:, -1]
        #     if self.include_charges
        #     else torch.zeros(0, device=next(self.parameters()).device)
        # )
        #
        # molecules.extend(
        #     list(
        #         zip(
        #             batch_tensor_to_list(x_, batch_index),
        #             batch_tensor_to_list(atom_types_, batch_index)
        #         )
        #     )
        # )
        #
        # atom_types.extend(atom_types_.tolist())
        # charges.extend(charges_.tolist())
        # info= self.analyze_samples(molecules, atom_types, charges)

        x = xh[:, :self.num_x_dims]
        one_hot = xh[:, self.num_x_dims:-1] if self.include_charges else xh[:, self.num_x_dims:]
        max_indices=torch.argmax(one_hot,dim=1)
        zero_tensor=torch.zeros_like(one_hot)
        zero_tensor[torch.arange(one_hot.size(0)),max_indices]=1
        one_hot=zero_tensor
        charges = xh[:, -1:] if self.include_charges else torch.zeros(0, device=next(self.parameters()).device)

        return x, one_hot, charges, batch_index

    def sample_and_compute_alpha(
        self,
        mean,
        mad,
        num_samples: int,
        num_nodes: Optional[TensorType["batch_size"]] = None,
        node_mask: Optional[TensorType["batch_num_nodes"]] = None,
        context: Optional[TensorType["batch_size", "num_context_features"]] = None,
        fix_noise: bool = False,
        num_timesteps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # node count-conditioning
        if num_nodes is None:
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
            max_num_nodes = (
                self.dataset_info["max_n_nodes"]
                if "max_n_nodes" in self.dataset_info
                else num_nodes.max().item()
            )
            assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        if self.condition_on_context:
            if context is None:
                context = self.props_distr.sample_batch(num_nodes)
        else:
            context = None

        # sampling
        alpha_list=[]
        classifier_dir = '/tmp/pycharm_project_80/checkpoint_latent_condition_qm9/Property_Classifiers/exp_class_alpha'
        with open(os.path.join(classifier_dir, "args.pickle"), "rb") as f:
            args_classifier = pickle.load(f)
        args_classifier.device = next(self.parameters()).device
        args_classifier.model_name = "egnn"
        classifier = get_classifier_model(args_classifier)
        # file_name = "/tmp/pycharm_project_80/checkpoint_latent_condition_qm9/Property_Classifiers/exp_class_alpha/" + "best_checkpoint.npy"
        file_name = "/tmp/pycharm_project_80/checkpoint_latent_condition_qm9/Property_Classifiers/exp_class_alpha/" + "best_checkpoint.npy"
        checkpoint = torch.load(file_name, map_location=torch.device("cpu"))
        # classifier_state_dict = torch.load(
        #     os.path.join(model_dir, "best_checkpoint.npy"),
        #     map_location=torch.device("cpu")
        # )
        device_l = next(self.parameters()).device
        classifier.load_state_dict(checkpoint)
        for _ in range(10):
            xh, batch_index, _ = self.ddpm.mol_gen_sample(
                num_samples=num_samples,
                num_nodes=num_nodes,
                node_mask=node_mask,
                context=context,
                fix_noise=fix_noise,
                fix_self_conditioning_noise=fix_noise,
                device=next(self.parameters()).device,
                num_timesteps=num_timesteps
            )

            x = xh[:, :self.num_x_dims]
            one_hot = xh[:, self.num_x_dims:-1] if self.include_charges else xh[:, self.num_x_dims:]
            max_indices=torch.argmax(one_hot,dim=1)
            zero_tensor=torch.zeros_like(one_hot)
            zero_tensor[torch.arange(one_hot.size(0)),max_indices]=1
            one_hot=zero_tensor
            charges = xh[:, -1:] if self.include_charges else torch.zeros(0, device=next(self.parameters()).device)
            max_num_nodes = num_nodes.max().item()
            node_mask_range_tensor = torch.arange(max_num_nodes, device=next(self.parameters()).device).unsqueeze(0)
            node_mask = node_mask_range_tensor < num_nodes.unsqueeze(-1)
            dense_x = torch.zeros((num_samples, max_num_nodes, x.size(-1)), device=next(self.parameters()).device)
            dense_x[node_mask] = x
            dense_one_hot = torch.zeros((num_samples, max_num_nodes, one_hot.size(-1)), device=next(self.parameters()).device)
            dense_one_hot[node_mask] = one_hot
            bs, n_nodes = num_samples, max_num_nodes
            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
            diag_mask = diag_mask.to(next(self.parameters()).device)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
            data = {
                "positions": dense_x.detach(),
                "atom_mask": node_mask.detach(),
                "edge_mask": edge_mask.detach(),
                "one_hot": dense_one_hot.detach(),
            }


            batch_size, n_nodes, _ = data["positions"].size()
            atom_positions = data["positions"].view(batch_size * n_nodes, -1).to(device_l, torch.float32)
            atom_mask = data["atom_mask"].view(batch_size * n_nodes, -1).to(device_l, torch.float32)
            edge_mask = data["edge_mask"].to(device_l, torch.float32)
            nodes = data["one_hot"].to(device_l, torch.float32)

            nodes = nodes.view(batch_size * n_nodes, -1)
            edges = get_classifier_adj_matrix(n_nodes, batch_size, device_l, edges_dic={})
            pred = classifier(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask,
                         edge_mask=edge_mask,
                         n_nodes=n_nodes)
            pred=mad * pred + mean  #100,
            pred=pred.to('cpu')
            pred=pred.detach()
            alpha_list.append(pred)
            torch.cuda.empty_cache()
        final_tensor = torch.cat(alpha_list)
        return final_tensor

    def sample_and_compute_mu(
            self,
            mean,
            mad,
            num_samples: int,
            num_nodes: Optional[TensorType["batch_size"]] = None,
            node_mask: Optional[TensorType["batch_num_nodes"]] = None,
            context: Optional[TensorType["batch_size", "num_context_features"]] = None,
            fix_noise: bool = False,
            num_timesteps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # node count-conditioning
        if num_nodes is None:
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
            max_num_nodes = (
                self.dataset_info["max_n_nodes"]
                if "max_n_nodes" in self.dataset_info
                else num_nodes.max().item()
            )
            assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        if self.condition_on_context:
            if context is None:
                context = self.props_distr.sample_batch(num_nodes)
        else:
            context = None

        # sampling
        alpha_list = []
        classifier_dir = '/tmp/pycharm_project_80/checkpoint_latent_condition_qm9/Property_Classifiers/exp_class_mu'
        with open(os.path.join(classifier_dir, "args.pickle"), "rb") as f:
            args_classifier = pickle.load(f)
        args_classifier.device = next(self.parameters()).device
        args_classifier.model_name = "egnn"
        classifier = get_classifier_model(args_classifier)
        # file_name = "/tmp/pycharm_project_80/checkpoint_latent_condition_qm9/Property_Classifiers/exp_class_alpha/" + "best_checkpoint.npy"
        file_name = "/tmp/pycharm_project_80/checkpoint_latent_condition_qm9/Property_Classifiers/exp_class_mu/" + "best_checkpoint.npy"
        checkpoint = torch.load(file_name, map_location=torch.device("cpu"))
        # classifier_state_dict = torch.load(
        #     os.path.join(model_dir, "best_checkpoint.npy"),
        #     map_location=torch.device("cpu")
        # )
        device_l = next(self.parameters()).device
        classifier.load_state_dict(checkpoint)
        for _ in range(10):
            xh, batch_index, _ = self.ddpm.mol_gen_sample(
                num_samples=num_samples,
                num_nodes=num_nodes,
                node_mask=node_mask,
                context=context,
                fix_noise=fix_noise,
                fix_self_conditioning_noise=fix_noise,
                device=next(self.parameters()).device,
                num_timesteps=num_timesteps
            )

            x = xh[:, :self.num_x_dims]
            one_hot = xh[:, self.num_x_dims:-1] if self.include_charges else xh[:, self.num_x_dims:]
            max_indices = torch.argmax(one_hot, dim=1)
            zero_tensor = torch.zeros_like(one_hot)
            zero_tensor[torch.arange(one_hot.size(0)), max_indices] = 1
            one_hot = zero_tensor
            charges = xh[:, -1:] if self.include_charges else torch.zeros(0, device=next(self.parameters()).device)
            max_num_nodes = num_nodes.max().item()
            node_mask_range_tensor = torch.arange(max_num_nodes, device=next(self.parameters()).device).unsqueeze(0)
            node_mask = node_mask_range_tensor < num_nodes.unsqueeze(-1)
            dense_x = torch.zeros((num_samples, max_num_nodes, x.size(-1)), device=next(self.parameters()).device)
            dense_x[node_mask] = x
            dense_one_hot = torch.zeros((num_samples, max_num_nodes, one_hot.size(-1)),
                                        device=next(self.parameters()).device)
            dense_one_hot[node_mask] = one_hot
            bs, n_nodes = num_samples, max_num_nodes
            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
            diag_mask = diag_mask.to(next(self.parameters()).device)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
            data = {
                "positions": dense_x.detach(),
                "atom_mask": node_mask.detach(),
                "edge_mask": edge_mask.detach(),
                "one_hot": dense_one_hot.detach(),
            }

            batch_size, n_nodes, _ = data["positions"].size()
            atom_positions = data["positions"].view(batch_size * n_nodes, -1).to(device_l, torch.float32)
            atom_mask = data["atom_mask"].view(batch_size * n_nodes, -1).to(device_l, torch.float32)
            edge_mask = data["edge_mask"].to(device_l, torch.float32)
            nodes = data["one_hot"].to(device_l, torch.float32)

            nodes = nodes.view(batch_size * n_nodes, -1)
            edges = get_classifier_adj_matrix(n_nodes, batch_size, device_l, edges_dic={})
            pred = classifier(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask,
                              edge_mask=edge_mask,
                              n_nodes=n_nodes)
            pred = mad * pred + mean  # 100,
            pred = pred.to('cpu')
            pred = pred.detach()
            alpha_list.append(pred)
            torch.cuda.empty_cache()
        final_tensor = torch.cat(alpha_list)
        return final_tensor
    def sample_and_compute_Cv(
            self,
            mean,
            mad,
            num_samples: int,
            num_nodes: Optional[TensorType["batch_size"]] = None,
            node_mask: Optional[TensorType["batch_num_nodes"]] = None,
            context: Optional[TensorType["batch_size", "num_context_features"]] = None,
            fix_noise: bool = False,
            num_timesteps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # node count-conditioning
        if num_nodes is None:
            num_nodes = self.ddpm.num_nodes_distribution.sample(num_samples)
            max_num_nodes = (
                self.dataset_info["max_n_nodes"]
                if "max_n_nodes" in self.dataset_info
                else num_nodes.max().item()
            )
            assert int(num_nodes.max()) <= max_num_nodes

        # context-conditioning
        if self.condition_on_context:
            if context is None:
                context = self.props_distr.sample_batch(num_nodes)
        else:
            context = None

        # sampling
        alpha_list = []
        classifier_dir = '/tmp/pycharm_project_80/checkpoint_latent_condition_qm9/Property_Classifiers/exp_class_Cv'
        with open(os.path.join(classifier_dir, "args.pickle"), "rb") as f:
            args_classifier = pickle.load(f)
        args_classifier.device = next(self.parameters()).device
        args_classifier.model_name = "egnn"
        classifier = get_classifier_model(args_classifier)
        # file_name = "/tmp/pycharm_project_80/checkpoint_latent_condition_qm9/Property_Classifiers/exp_class_alpha/" + "best_checkpoint.npy"
        file_name = "/tmp/pycharm_project_80/checkpoint_latent_condition_qm9/Property_Classifiers/exp_class_Cv/" + "best_checkpoint.npy"
        checkpoint = torch.load(file_name, map_location=torch.device("cpu"))
        # classifier_state_dict = torch.load(
        #     os.path.join(model_dir, "best_checkpoint.npy"),
        #     map_location=torch.device("cpu")
        # )
        device_l = next(self.parameters()).device
        classifier.load_state_dict(checkpoint)
        for _ in range(10):
            xh, batch_index, _ = self.ddpm.mol_gen_sample(
                num_samples=num_samples,
                num_nodes=num_nodes,
                node_mask=node_mask,
                context=context,
                fix_noise=fix_noise,
                fix_self_conditioning_noise=fix_noise,
                device=next(self.parameters()).device,
                num_timesteps=num_timesteps
            )

            x = xh[:, :self.num_x_dims]
            one_hot = xh[:, self.num_x_dims:-1] if self.include_charges else xh[:, self.num_x_dims:]
            max_indices = torch.argmax(one_hot, dim=1)
            zero_tensor = torch.zeros_like(one_hot)
            zero_tensor[torch.arange(one_hot.size(0)), max_indices] = 1
            one_hot = zero_tensor
            charges = xh[:, -1:] if self.include_charges else torch.zeros(0, device=next(self.parameters()).device)
            max_num_nodes = num_nodes.max().item()
            node_mask_range_tensor = torch.arange(max_num_nodes, device=next(self.parameters()).device).unsqueeze(0)
            node_mask = node_mask_range_tensor < num_nodes.unsqueeze(-1)
            dense_x = torch.zeros((num_samples, max_num_nodes, x.size(-1)), device=next(self.parameters()).device)
            dense_x[node_mask] = x
            dense_one_hot = torch.zeros((num_samples, max_num_nodes, one_hot.size(-1)),
                                        device=next(self.parameters()).device)
            dense_one_hot[node_mask] = one_hot
            bs, n_nodes = num_samples, max_num_nodes
            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
            diag_mask = diag_mask.to(next(self.parameters()).device)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
            data = {
                "positions": dense_x.detach(),
                "atom_mask": node_mask.detach(),
                "edge_mask": edge_mask.detach(),
                "one_hot": dense_one_hot.detach(),
            }

            batch_size, n_nodes, _ = data["positions"].size()
            atom_positions = data["positions"].view(batch_size * n_nodes, -1).to(device_l, torch.float32)
            atom_mask = data["atom_mask"].view(batch_size * n_nodes, -1).to(device_l, torch.float32)
            edge_mask = data["edge_mask"].to(device_l, torch.float32)
            nodes = data["one_hot"].to(device_l, torch.float32)

            nodes = nodes.view(batch_size * n_nodes, -1)
            edges = get_classifier_adj_matrix(n_nodes, batch_size, device_l, edges_dic={})
            pred = classifier(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask,
                              edge_mask=edge_mask,
                              n_nodes=n_nodes)
            pred = mad * pred + mean  # 100,
            pred = pred.to('cpu')
            pred = pred.detach()
            alpha_list.append(pred)
            torch.cuda.empty_cache()
        final_tensor = torch.cat(alpha_list)
        return final_tensor
