# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for Bio-Diffusion (https://github.com/BioinfoMachineLearning/bio-diffusion):
# -------------------------------------------------------------------------------------------------------------------------------------

import math
import torch
import numpy as np
import torch.nn as nn
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
from src.models.components.gcpnet import  GCPNetEncoder , GCPNetDecoder
patch_typeguard()  # use before @typechecked


log = get_pylogger(__name__)


H_INPUT_TYPE = Union[
    TensorType["batch_num_nodes", "num_atom_types"],
    torch.Tensor  # note: for when `include_charges=False`
]
NODE_FEATURE_DIFFUSION_TARGETS = ["atom_types_and_coords"]


@typechecked
def cosine_beta_schedule(
    num_timesteps: int,
    s: float = 0.008,
    raise_to_power: float = 1
) -> np.ndarray:
    """
    A cosine variance schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """
    steps = num_timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


@typechecked
def clip_noise_schedule(
    alphas2: np.ndarray,
    clip_value: float = 0.001
) -> np.ndarray:
    """
    For a noise schedule given by (alpha ^ 2), this clips alpha_t / (alpha_t - 1).
    This may help improve stability during sampling.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


@typechecked
def polynomial_schedule(
    num_timesteps: int,
    s: float = 1e-4,
    power: float = 3.0
) -> np.ndarray:
    """
    A noise schedule based on a simple polynomial equation: 1 - (x ^ power).
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """
    steps = num_timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


class PositiveLinear(nn.Module):
    """
    A linear layer with weights forced to be positive.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Union[torch.device, str],
        bias: bool = True,
        weight_init_offset: int = -2
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)
        self.weight_init_offset = weight_init_offset

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @typechecked
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class GammaNetwork(nn.Module):
    """
    The gamma network models a monotonically-increasing function. Constructed as in the VDM paper.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """

    def __init__(self, verbose: bool = True):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = nn.Parameter(torch.tensor([-5.0]))
        self.gamma_1 = nn.Parameter(torch.tensor([10.0]))

        if verbose:
            self.display_schedule()

    @typechecked
    def display_schedule(self, num_steps: int = 50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        log.info(f"Gamma schedule: {gamma.detach().cpu().numpy().reshape(num_steps)}")

    @typechecked
    def gamma_tilde(self, t: TensorType["batch_size", 1]) -> TensorType["batch_size", 1]:
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    @typechecked
    def forward(self, t: TensorType["batch_size", 1]) -> TensorType["batch_size", 1]:
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)

        # note: not very efficient
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # normalize to [0, 1]
        normalized_gamma = (
            (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)
        )

        # rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


class PredefinedNoiseSchedule(nn.Module):
    """
    A predefined noise schedule. Essentially, creates a lookup array for predefined (non-learned) noise schedules.
    From: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py
    """

    def __init__(
        self,
        noise_schedule: str,
        num_timesteps: int,
        noise_precision: float,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__()

        self.timesteps = num_timesteps

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(num_timesteps)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(num_timesteps, s=noise_precision, power=power)
        else:
            raise ValueError(noise_schedule)

        if verbose:
            log.info(f"alphas2: {alphas2}")

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        if verbose:
            log.info(f"gamma: {-log_alphas2_to_sigmas2}")

        self.gamma = nn.Parameter(
            torch.tensor(-log_alphas2_to_sigmas2).float(),
            requires_grad=False
        )

    @typechecked
    def forward(self, t: TensorType["batch_size", 1]) -> TensorType["batch_size", 1]:
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GCPNetAutoEncoder(nn.Module):
    """
    The Equivariant Variational Diffusion (EVD) Module.
    """

    def __init__(
        self,
        encoder: GCPNetEncoder,
        decoder: GCPNetDecoder,
        dynamics_network: nn.Module,
        diffusion_cfg: DictConfig,
        dataloader_cfg: DictConfig,
        dataset_info: Dict[str, Any]
    ):
        super().__init__()

        # ensure config arguments are of valid values and structures
        assert diffusion_cfg.parametrization in ["eps"], "Epsilon is currently the only supported parametrization."
        assert diffusion_cfg.loss_type in [
            "vlb", "l2"
        ], "Variational lower-bound and L2 losses are currently the only supported diffusion loss functions."

        if diffusion_cfg.noise_schedule == "learned":
            assert diffusion_cfg.loss_type == "vlb", "A noise schedule can only be learned with a variational lower-bound objective."

        # hyperparameters #
        self.diffusion_cfg = diffusion_cfg
        self.diffusion_target = diffusion_cfg.diffusion_target
        self.num_atom_types = dataloader_cfg.num_atom_types
        self.num_x_dims = dataloader_cfg.num_x_dims
        self.include_charges = dataloader_cfg.include_charges
        self.num_node_scalar_features = dataloader_cfg.num_atom_types + dataloader_cfg.include_charges
        self.T = diffusion_cfg.num_timesteps
        self.encoder=encoder
        self.decoder=decoder
        # Forward pass #

        forward_prefix_mapping = {
            "atom_types_and_coords": "atom_types_and_coords_"
        }
        forward_prefix = forward_prefix_mapping[self.diffusion_target]
        self.forward_fn = getattr(self, forward_prefix + "forward")

        # PyTorch modules #

        # network that will predict the noise
        self.dynamics_network = dynamics_network

        # distribution of node counts
        histogram = {int(k): int(v) for k, v in dataset_info["n_nodes"].items()}
        self.num_nodes_distribution = NumNodesDistribution(histogram)

        # noise schedule
        if diffusion_cfg.noise_schedule == "learned":
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(**diffusion_cfg)

        if diffusion_cfg.noise_schedule != "learned":
            self.detect_issues_with_norm_values()

    @staticmethod
    @typechecked
    def sigma(
        gamma: TensorType["batch_size", 1],
        target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Compute `sigma` given `gamma`."""
        return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    @staticmethod
    @typechecked
    def alpha(
        gamma: TensorType["batch_size", 1],
        target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Compute `alpha` given `gamma`."""
        return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    @staticmethod
    @typechecked
    def SNR(gamma: TensorType["batch_size", 1]) -> TensorType["batch_size", 1]:
        """Compute signal to noise ratio (SNR) (i.e., alpha ^ 2 / sigma ^ 2) given `gamma`."""
        return torch.exp(-gamma)




    @staticmethod
    @typechecked
    def sample_center_gravity_zero_gaussian_with_mask(
        size: torch.Size,
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        device: Union[torch.device, str]
    ) -> torch.Tensor:
        assert len(size) == 2
        x = torch.randn(size, device=device)

        x_masked = x * node_mask.float().unsqueeze(-1)

        # note: this projection only works because Gaussians are
        # rotation-invariant around zero and their samples are independent!
        _, x_projected = centralize(
            Batch(x=x_masked),
            "x",
            batch_index=batch_index,
            node_mask=node_mask,
            edm=True
        )
        return x_projected

    @staticmethod
    @typechecked
    def sample_gaussian(
        size: torch.Size,
        device: Union[torch.device, str]
    ) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x

    @staticmethod
    @typechecked
    def sample_gaussian_with_mask(
        size: torch.Size,
        node_mask: TensorType["batch_num_nodes"],
        device: Union[torch.device, str]
    ) -> torch.Tensor:
        x = torch.randn(size, device=device)
        x_masked = x * node_mask.float().unsqueeze(-1)
        return x_masked

    @staticmethod
    @typechecked
    def assert_correctly_masked(variable: torch.Tensor, node_mask: torch.Tensor):
        assert (node_mask.all()) or (variable[~node_mask].abs().max().item() < 1e-4), "Variables not masked properly."

    @staticmethod
    @typechecked
    def sum_node_features_except_batch(
        values: TensorType["batch_num_nodes", "num_node_features"],
        batch_index: TensorType["batch_num_nodes"]
    ):
        return scatter(values.sum(-1), batch_index, dim=0, reduce="sum")

    @staticmethod
    @typechecked
    def check_mask_correct(variables: torch.Tensor, node_mask: torch.Tensor):
        for variable in variables:
            if len(variable) > 0:
                assert (node_mask.all()) or (variable[~node_mask].abs(
                ).max().item() < 1e-4), "Variables not masked properly."

    @staticmethod
    @typechecked
    def assert_mean_zero_with_mask(
        x: TensorType["batch_num_nodes", 3],
        node_mask: TensorType["batch_num_nodes"],
        eps: float = 1e-10
    ):
        assert (node_mask.all()) or (x[~node_mask].abs().max().item() < 1e-4), "Variables not masked properly."
        largest_value = x.abs().max().item()
        error = torch.sum(x, dim=0, keepdim=True).abs().max().item()
        rel_error = error / (largest_value + eps)
        assert rel_error < 1e-2, f"Mean is not zero, as relative_error {rel_error}"

    @typechecked
    def detect_issues_with_norm_values(self, num_std_dev: int = 8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # detect if (1 / `norm_value`) is still larger than (10 * standard deviation)
        if len(self.diffusion_cfg.norm_values) > 1:
            norm_value = self.diffusion_cfg.norm_values[1]
            if (sigma_0 * num_std_dev) > (1.0 / norm_value):
                raise ValueError(
                    f"Value for normalization value {norm_value} is probably"
                    f" too large with sigma_0={sigma_0:.5f}"
                    f" and (1 / norm_value = {1.0 / norm_value})"
                )

    @typechecked
    def subspace_dimensionality(
        self,
        num_nodes: TensorType["batch_size"]
    ) -> TensorType["batch_size"]:
        """Compute the dimensionality on translation-invariant linear subspace where distributions on `x` are defined."""
        return (num_nodes - 1) * self.num_x_dims



    @typechecked
    def normalize(
        self,
        x: TensorType["batch_num_nodes", 3],
        h: Union[Dict[str, H_INPUT_TYPE], H_INPUT_TYPE],
        node_mask: TensorType["batch_num_nodes"],
        generate_x_only: bool = False
    ) -> Tuple[
        TensorType["batch_num_nodes", 3],
        Union[Dict[str, H_INPUT_TYPE], H_INPUT_TYPE]
    ]:
        x = x / self.diffusion_cfg.norm_values[0]

        # bypass individual normalizations for components of `h`
        if generate_x_only:
            h = (h.float() - self.diffusion_cfg.norm_biases[1]) / self.diffusion_cfg.norm_values[1]
            return x, h

        # cast to float in case `h` still has `long` or `int` type
        h_cat = (h["categorical"].float() - self.diffusion_cfg.norm_biases[1]) / self.diffusion_cfg.norm_values[1]
        h_cat = h_cat * node_mask.float().unsqueeze(-1)
        h_int = (
            (h["integer"].float() - self.diffusion_cfg.norm_biases[2]) / self.diffusion_cfg.norm_values[2]
        )

        if self.include_charges:
            h_int = h_int * node_mask.float()

        # create new `h` dictionary
        h = {"categorical": h_cat, "integer": h_int}

        return x, h

    @typechecked
    def unnormalize(
        self,
        x: TensorType["batch_num_nodes", 3],
        node_mask: TensorType["batch_num_nodes"],
        h_cat: Optional[TensorType["batch_num_nodes", "num_node_categories"]] = None,
        h_int: Optional[H_INPUT_TYPE] = None,
        generate_x_only: bool = False
    ) -> Tuple[
        TensorType["batch_num_nodes", 3],
        Optional[TensorType["batch_num_nodes", "num_node_categories"]],
        Optional[H_INPUT_TYPE]
    ]:
        x = x * self.diffusion_cfg.norm_values[0]

        if generate_x_only:
            return x, None, None

        h_cat = h_cat * self.diffusion_cfg.norm_values[1] + self.diffusion_cfg.norm_biases[1]
        h_cat = h_cat * node_mask.float().unsqueeze(-1)
        h_int = h_int * self.diffusion_cfg.norm_values[2] + self.diffusion_cfg.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask.float().unsqueeze(-1)

        return x, h_cat, h_int

    @typechecked
    def unnormalize_z(
        self,
        z: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        node_mask: TensorType["batch_num_nodes"],
        generate_x_only: bool = False
    ) -> TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]:
        # parse from `z`
        x = z[:, 0:self.num_x_dims]
        h_cat = (
            None
            if generate_x_only
            else z[:, self.num_x_dims:self.num_x_dims + self.num_atom_types]
        )
        h_int = (
            None
            if generate_x_only
            else z[:, self.num_x_dims + self.num_atom_types:]
        )

        # unnormalize
        if generate_x_only:
            x, _, _ = self.unnormalize(x, node_mask, generate_x_only=True)
            output = x
        else:
            x, h_cat, h_int = self.unnormalize(x, node_mask, h_cat=h_cat, h_int=h_int)
            output = (
                torch.cat([x, h_cat, h_int], dim=-1)
                if self.include_charges
                else torch.cat([x, h_cat], dim=-1)
            )
        return output

    @typechecked
    def sample_combined_position_feature_noise(
        self,
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        generate_x_only: bool = False
    ) -> TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]:
        """
        Sample mean-centered normal noise for `z_x`, and standard normal noise for `z_h`.
        """
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
            size=(len(batch_index), self.num_node_scalar_features),
            node_mask=node_mask,
            device=batch_index.device
        )
        z = torch.cat([z_x, z_h], dim=-1)
        return z

    @typechecked
    def sample_normal(
        self,
        mu: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        sigma: TensorType["batch_size", 1],
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        fix_noise: bool = False,
        generate_x_only: bool = False
    ) -> TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]:
        """Sample from a Normal distribution."""
        if fix_noise:
            batch_index_ = torch.zeros_like(batch_index)  # broadcast same noise across batch
            eps = self.sample_combined_position_feature_noise(batch_index_, node_mask, generate_x_only=generate_x_only)
        else:
            eps = self.sample_combined_position_feature_noise(batch_index, node_mask, generate_x_only=generate_x_only)
        return mu + sigma[batch_index] * eps


    @typechecked
    def compute_noised_representation(
        self,
        xh: TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        batch_index: TensorType["batch_num_nodes"],
        node_mask: TensorType["batch_num_nodes"],
        gamma_t: TensorType["batch_size", 1],
        generate_x_only: bool = False
    ) -> Tuple[
        TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"],
        TensorType["batch_num_nodes", "num_x_dims_plus_num_node_scalar_features"]
    ]:
        # compute `alpha_t` and `sigma_t` from gamma
        alpha_t = self.alpha(gamma_t, xh)
        sigma_t = self.sigma(gamma_t, xh)

        # sample `zt` ~ Normal(`alpha_t` `x`, `sigma_t`)
        eps = self.sample_combined_position_feature_noise(batch_index, node_mask, generate_x_only=generate_x_only)

        # sample `z_t` given `x`, `h` for timestep `t`, from q(`z_t` | `x`, `h`)
        z_t = alpha_t[batch_index] * xh + sigma_t[batch_index] * eps

        return z_t, eps

    @typechecked
    def log_pN(self, num_nodes: TensorType["batch_size"]) -> TensorType["batch_size"]:
        """
        Prior on the sample size for computing log p(x, h, N) = log(x, h | N) + log p(N),
        where log p(x, h | N) is a model's output.
        """
        log_pN = self.num_nodes_distribution.log_prob(num_nodes)
        return log_pN

    @typechecked
    def delta_log_px(self, num_nodes: TensorType["batch_size"]) -> TensorType["batch_size"]:
        d = self.subspace_dimensionality(num_nodes)
        return -d * np.log(self.diffusion_cfg.norm_values[0])


    def encode(self,batch,context=None,need_sample=True):
        batch.h = {"categorical": batch.one_hot, "integer": batch.charges}
        xh = torch.cat([batch.x, batch.h["categorical"], batch.h["integer"].reshape(-1, 1)], dim=-1)
        batch,xh_mu=self.encoder(batch,xh)
        x_mu=xh_mu[:, :self.num_x_dims]
        h_mu=xh_mu[:, self.num_x_dims:]
        bs,_=xh_mu.size()
        t_zeros = torch.zeros(size=(bs, 1), device=batch.device)
        gamma_0 = self.gamma(t_zeros)
        sigma_0 = self.sigma(gamma_0, xh_mu)
        xh_new=self.sample_normal(xh_mu,sigma_0,batch.batch,batch.mask)
        x_init = xh_new[:, :self.num_x_dims]
        h_init = xh_new[:, self.num_x_dims:]
        if need_sample:
            batch.x=x_init
            batch.h=h_init
        return batch

    def decode(self,batch):
        xh = torch.cat([batch.x, batch.h], dim=-1)
        batch,xh_recon=self.decoder(batch,xh)
        x_recon=xh_recon[:, :self.num_x_dims]
        h_recon=xh_recon[:, self.num_x_dims:]
        h_int=h_recon[:,-1:]
        h_cat=h_recon[:,:-1]
        h_int=torch.round(h_int)
        h_recon={"categorical": h_cat, "integer": h_int}
        batch.h=h_recon

        return batch
    def compute_reconstruction_error(self,batch,batch_recon):
        x=batch.x
        x_recon=batch_recon.x
        error_x=self.sum_node_features_except_batch((x-x_recon)**2,batch.batch)

        cat=batch.h["categorical"]
        cat_recon=batch_recon.h["categorical"]
        error_cat=self.sum_node_features_except_batch((cat-cat_recon)**2,batch.batch)
        if self.include_charges:
            int=batch.h["integer"]
            int_recon=batch_recon.h["integer"]
            error_int=self.sum_node_features_except_batch((int-int_recon)**2,batch.batch)
        else:
            error_int=0
        error=error_x+error_int+error_cat

        error=error*batch.mask
        return error
    def forward(self,batch):
        batch_saved=batch
        batch_encoded=self.encode(batch)
        batch_recon=self.decode(batch_encoded)
        error_recon=self.compute_reconstruction_error(batch_saved,batch_recon)
        return error_recon




    
