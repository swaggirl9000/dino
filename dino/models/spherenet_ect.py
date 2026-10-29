import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from types import SimpleNamespace

import torch
import torch.nn as nn
from dig.threedgraph.method import SphereNet

from .ect_module import ECTPlugIn
from qm9_ect import build_icosphere_directions


def _make_spherenet(hidden_channels, out_emb_channels, num_layers, cutoff):
    """
    out_channels is set to out_emb_channels so SphereNet returns a full-dim
    graph embedding rather than a scalar. Both baseline and ECT variant share
    this backbone — only the output head differs, keeping the comparison fair.
    """
    return SphereNet(
        energy_and_force=False,   # we handle forces via autograd
        cutoff=cutoff,
        num_layers=num_layers,
        hidden_channels=hidden_channels,
        out_channels=out_emb_channels,   # embedding dim, not 1
        int_emb_size=64,
        basis_emb_size_dist=8,
        basis_emb_size_angle=8,
        basis_emb_size_torsion=8,
        out_emb_channels=out_emb_channels,
        num_spherical=7,
        num_radial=6,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
    )


class SphereNetBaseline(nn.Module):
    """SphereNet backbone + linear head. No ECT. Identical backbone to the ECT
    variant so results are directly comparable."""

    def __init__(
        self,
        hidden_channels: int = 128,
        out_emb_channels: int = 256,
        num_layers: int = 4,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.backbone = _make_spherenet(hidden_channels, out_emb_channels, num_layers, cutoff)
        self.head = nn.Linear(out_emb_channels, 1)

    def forward(self, z, pos, batch):
        # SphereNet expects an object with .z, .pos, .batch attributes
        data = SimpleNamespace(z=z, pos=pos, batch=batch)
        u = self.backbone(data)                    # [B, out_emb_channels]
        return self.head(u).squeeze(-1)            # [B]

    def forward_with_forces(self, z, pos, batch):
        """Energy + forces via autograd. Caller must set pos.requires_grad_(True)."""
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class SphereNetWithECT(nn.Module):
    """SphereNet backbone + ECT geometric token fused at the last layer.

    ECT is computed inside forward() from pos so that autograd forces
    (F = -dE/dpos) flow through both SphereNet and the ECT branch.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        hidden_channels: int = 128,
        out_emb_channels: int = 256,
        num_layers: int = 4,
        cutoff: float = 5.0,
        ect_out_dim: int = 64,
        mlp_hidden: int = 128,
        icosphere_nu: int = 3,
        resolution: int = 64,
        radius: float = 8.0,
    ):
        super().__init__()
        self.backbone = _make_spherenet(hidden_channels, out_emb_channels, num_layers, cutoff)

        directions, _ = build_icosphere_directions(nu=icosphere_nu)
        self.register_buffer("directions", directions)

        self.ect_plugin = ECTPlugIn(
            graph_dim=out_emb_channels,
            ect_edge_indices=ect_edge_indices,
            ect_out_dim=ect_out_dim,
            mlp_hidden=mlp_hidden,
            resolution=resolution,
            radius=radius,
        )

    def forward(self, z, pos, batch):
        data = SimpleNamespace(z=z, pos=pos, batch=batch)
        u = self.backbone(data)                    # [B, out_emb_channels]
        return self.ect_plugin(u, pos, batch, self.directions)

    def forward_with_forces(self, z, pos, batch):
        """Energy + forces via autograd. Caller must set pos.requires_grad_(True)."""
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces

    def freeze_backbone(self):
        """Freeze SphereNet weights. Train only ECT branch + fusion head."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)
