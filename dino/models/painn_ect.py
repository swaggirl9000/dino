import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, radius_graph
from schnetpack.representation import PaiNN
from schnetpack.nn import GaussianRBF, CosineCutoff
import schnetpack.properties as props

from .ect_module import ECTPlugIn
from qm9_ect import build_icosphere_directions


def _make_painn(n_atom_basis, n_interactions, cutoff):
    return PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=GaussianRBF(n_rbf=20, cutoff=cutoff),
        cutoff_fn=CosineCutoff(cutoff=cutoff),
    )


def _build_inputs(z, pos, batch, cutoff):
    """Convert PyG-format (z, pos, batch) to SchNetPack input dict.

    SchNetPack expects pairwise displacement vectors and neighbor indices
    rather than positions directly. We compute them here using PyG's
    radius_graph so the rest of the code stays in PyG conventions.
    """
    edge_index = radius_graph(pos, r=cutoff, batch=batch, max_num_neighbors=64)
    idx_i, idx_j = edge_index[0], edge_index[1]
    Rij = pos[idx_j] - pos[idx_i]          # [n_pairs, 3]
    return {
        props.Z:     z,                     # [N]
        props.Rij:   Rij,                   # [n_pairs, 3]
        props.idx_i: idx_i,                 # [n_pairs]
        props.idx_j: idx_j,                 # [n_pairs]
        props.idx_m: batch,                 # [N]  molecule index per atom
    }


class PaiNNBaseline(nn.Module):
    """PaiNN backbone + pooling + MLP head. No ECT. Identical settings to the
    ECT variant so results are directly comparable."""

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        cutoff: float = 5.0,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.representation = _make_painn(n_atom_basis, n_interactions, cutoff)
        self.head = nn.Sequential(
            nn.Linear(n_atom_basis, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def _node_features(self, z, pos, batch):
        inputs = _build_inputs(z, pos, batch, self.cutoff)
        inputs = self.representation(inputs)
        return inputs["scalar_representation"]  # [N, n_atom_basis]

    def forward(self, z, pos, batch):
        x = self._node_features(z, pos, batch)
        graph_feat = global_mean_pool(x, batch)    # [B, n_atom_basis]
        return self.head(graph_feat).squeeze(-1)   # [B]

    def forward_with_forces(self, z, pos, batch):
        """Energy + forces via autograd. Caller must set pos.requires_grad_(True)."""
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class PaiNNWithECT(nn.Module):
    """PaiNN backbone + ECT geometric token fused at the last layer.

    ECT is computed inside forward() from pos so that autograd forces
    (F = -dE/dpos) flow through both PaiNN and the ECT branch.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        cutoff: float = 5.0,
        ect_out_dim: int = 64,
        mlp_hidden: int = 128,
        icosphere_nu: int = 3,
        resolution: int = 64,
        radius: float = 8.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.representation = _make_painn(n_atom_basis, n_interactions, cutoff)

        directions, _ = build_icosphere_directions(nu=icosphere_nu)
        self.register_buffer("directions", directions)

        self.ect_plugin = ECTPlugIn(
            graph_dim=n_atom_basis,
            ect_edge_indices=ect_edge_indices,
            ect_out_dim=ect_out_dim,
            mlp_hidden=mlp_hidden,
            resolution=resolution,
            radius=radius,
        )

    def forward(self, z, pos, batch):
        inputs = _build_inputs(z, pos, batch, self.cutoff)
        inputs = self.representation(inputs)
        x = inputs["scalar_representation"]        # [N, n_atom_basis]
        graph_feat = global_mean_pool(x, batch)    # [B, n_atom_basis]
        return self.ect_plugin(graph_feat, pos, batch, self.directions)

    def forward_with_forces(self, z, pos, batch):
        """Energy + forces via autograd. Caller must set pos.requires_grad_(True)."""
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces

    def freeze_backbone(self):
        """Freeze PaiNN weights. Train only ECT branch + fusion head."""
        for p in self.representation.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.representation.parameters():
            p.requires_grad_(True)
