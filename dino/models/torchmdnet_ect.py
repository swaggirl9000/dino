import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torchmdnet.models.tensornet import TensorNet

from .ect_module import ECTPlugIn
from qm9_ect import build_icosphere_directions


def _make_tensornet(hidden_channels, num_layers, cutoff_upper):
    return TensorNet(
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        cutoff_lower=0.0,
        cutoff_upper=cutoff_upper,
        max_num_neighbors=64,
        equivariance_invariance_group="O(3)",
        dtype=torch.float32,
    )


class TorchMDNetBaseline(nn.Module):
    """TensorNet + pooling + MLP head. No ECT. Identical settings to the ECT
    variant so results are directly comparable."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 2,
        cutoff_upper: float = 5.0,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.representation = _make_tensornet(hidden_channels, num_layers, cutoff_upper)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, z, pos, batch):
        x, _, _, pos_out, batch_out = self.representation(z, pos, batch)
        graph_feat = global_mean_pool(x, batch_out)   # [B, hidden_channels]
        return self.head(graph_feat).squeeze(-1)       # [B]

    def forward_with_forces(self, z, pos, batch):
        """Energy + forces via autograd. Caller must set pos.requires_grad_(True)."""
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class TorchMDNetWithECT(nn.Module):
    """TensorNet backbone + ECT geometric token fused at the last layer.

    The ECT is computed from atomic positions inside forward() so that
    autograd forces (F = -dE/dpos) flow through both the backbone and ECT.
    Only the ECT branch and fusion head need to be trained when the backbone
    is frozen.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        hidden_channels: int = 128,
        num_layers: int = 2,
        cutoff_upper: float = 5.0,
        ect_out_dim: int = 64,
        mlp_hidden: int = 128,
        icosphere_nu: int = 3,
        resolution: int = 64,
        radius: float = 8.0,
    ):
        super().__init__()
        self.representation = _make_tensornet(hidden_channels, num_layers, cutoff_upper)

        directions, _ = build_icosphere_directions(nu=icosphere_nu)
        self.register_buffer("directions", directions)

        self.ect_plugin = ECTPlugIn(
            graph_dim=hidden_channels,
            ect_edge_indices=ect_edge_indices,
            ect_out_dim=ect_out_dim,
            mlp_hidden=mlp_hidden,
            resolution=resolution,
            radius=radius,
        )

    def forward(self, z, pos, batch):
        x, _, _, pos_out, batch_out = self.representation(z, pos, batch)
        graph_feat = global_mean_pool(x, batch_out)   # [B, hidden_channels]
        # pos_out is the same tensor as pos (TensorNet passes it through),
        # so autograd gradients for forces flow correctly.
        return self.ect_plugin(graph_feat, pos_out, batch_out, self.directions)

    def forward_with_forces(self, z, pos, batch):
        """Energy + forces via autograd. Caller must set pos.requires_grad_(True)."""
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces

    def freeze_backbone(self):
        """Freeze TensorNet weights. Train only ECT branch + fusion head."""
        for p in self.representation.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.representation.parameters():
            p.requires_grad_(True)
