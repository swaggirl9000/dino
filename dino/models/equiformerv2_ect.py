import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

from .ect_module import ECTPlugIn
from qm9_ect import build_icosphere_directions

try:
    from fairchem.core.models.equiformer_v2.equiformer_v2_oc20 import EquiformerV2OC20
    _EV2_CLS = EquiformerV2OC20
except ImportError:
    _EV2_CLS = None


# Compact config suited for QM9 (< 30 atoms, no periodic boundary conditions)
_DEFAULT_CFG = dict(
    sphere_channels=128,
    attn_hidden_channels=64,
    num_heads=8,
    attn_alpha_channels=64,
    attn_value_channels=16,
    ffn_hidden_channels=128,
    norm_type="layer_norm_sh",
    lmax_list=[4],
    mmax_list=[2],
    max_num_neighbors=20,
    distance_function="gaussian",
    num_distance_basis=512,
    use_e2_updates=True,
    use_attn_renorm=True,
    use_gate_act=True,
    use_grid_mlp=True,
    use_sep_s2_act=True,
    alpha_drop=0.0,
    path_drop=0.0,
    proj_drop=0.0,
    otf_graph=True,
    use_pbc=False,
    regress_forces=False,
    direct_forces=False,
)


def _check_import():
    if _EV2_CLS is None:
        raise ImportError(
            "fairchem-core is required for EquiformerV2. "
            "Install with: pip install fairchem-core"
        )


def _make_equiformer(sphere_channels: int, num_layers: int, cutoff: float):
    _check_import()
    cfg = dict(_DEFAULT_CFG)
    cfg["sphere_channels"] = sphere_channels
    cfg["num_layers"] = num_layers
    model = _EV2_CLS(
        num_atoms=128,   # max atoms upper bound for embedding tables
        bond_feat_dim=0,
        num_targets=1,
        **cfg,
    )
    model.cutoff = cutoff
    return model


def _to_data(z, pos, batch):
    """Wrap PyG tensors in a Data object for FAIRChem's EquiformerV2."""
    return Data(
        pos=pos,
        atomic_numbers=z.long(),
        batch=batch,
        natoms=torch.bincount(batch),
        # OC20 bookkeeping fields — unused for QM9 but required by the model
        tags=torch.zeros(len(z), dtype=torch.long, device=z.device),
        fixed=torch.zeros(len(z), dtype=torch.bool, device=z.device),
    )


class _CapturePreHook:
    """Forward pre-hook that saves the input to EquiformerV2's energy_block.

    The energy_block receives the per-atom invariant (l=0) features with
    shape [N, sphere_channels], which we use as the graph-level embedding
    after pooling. NOT detached so gradients flow through for autograd forces.
    """

    def __init__(self):
        self.embedding = None

    def __call__(self, module, args):
        inp = args[0]
        if inp.dim() == 3:
            # Full SO3 tensor [N, sphere_channels, num_coefficients]; take l=0
            self.embedding = inp[:, :, 0]
        else:
            # Already invariant [N, sphere_channels]
            self.embedding = inp


def _attach_hook(backbone):
    capture = _CapturePreHook()
    if not hasattr(backbone, "energy_block"):
        raise AttributeError(
            "Expected backbone.energy_block; your fairchem-core version may differ. "
            "Check the EquiformerV2OC20 source and update the hook target."
        )
    backbone.energy_block.register_forward_pre_hook(capture)
    return capture


class EquiformerV2Baseline(nn.Module):
    """EquiformerV2 backbone + pooling + MLP head. No ECT.

    Identical backbone settings to EquiformerV2WithECT so results are
    directly comparable.
    """

    def __init__(
        self,
        sphere_channels: int = 128,
        num_layers: int = 6,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.backbone = _make_equiformer(sphere_channels, num_layers, cutoff)
        self._capture = _attach_hook(self.backbone)
        self.head = nn.Sequential(
            nn.Linear(sphere_channels, sphere_channels),
            nn.SiLU(),
            nn.Linear(sphere_channels, 1),
        )

    def forward(self, z, pos, batch):
        data = _to_data(z, pos, batch)
        self.backbone(data)                           # hook fires, sets _capture.embedding
        x = self._capture.embedding                  # [N, sphere_channels]
        graph_feat = global_mean_pool(x, batch)      # [B, sphere_channels]
        return self.head(graph_feat).squeeze(-1)     # [B]

    def forward_with_forces(self, z, pos, batch):
        """Energy + forces via autograd. Caller must set pos.requires_grad_(True)."""
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class EquiformerV2WithECT(nn.Module):
    """EquiformerV2 backbone + ECT geometric token fused at the last layer.

    ECT is computed from atomic positions inside forward() so that autograd
    forces (F = -dE/dpos) flow through both the backbone and the ECT branch.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        sphere_channels: int = 128,
        num_layers: int = 6,
        cutoff: float = 5.0,
        ect_out_dim: int = 64,
        mlp_hidden: int = 128,
        icosphere_nu: int = 3,
        resolution: int = 64,
        radius: float = 8.0,
    ):
        super().__init__()
        self.backbone = _make_equiformer(sphere_channels, num_layers, cutoff)
        self._capture = _attach_hook(self.backbone)

        directions, _ = build_icosphere_directions(nu=icosphere_nu)
        self.register_buffer("directions", directions)

        self.ect_plugin = ECTPlugIn(
            graph_dim=sphere_channels,
            ect_edge_indices=ect_edge_indices,
            ect_out_dim=ect_out_dim,
            mlp_hidden=mlp_hidden,
            resolution=resolution,
            radius=radius,
        )

    def forward(self, z, pos, batch):
        data = _to_data(z, pos, batch)
        self.backbone(data)
        x = self._capture.embedding                  # [N, sphere_channels]
        graph_feat = global_mean_pool(x, batch)      # [B, sphere_channels]
        return self.ect_plugin(graph_feat, pos, batch, self.directions)

    def forward_with_forces(self, z, pos, batch):
        """Energy + forces via autograd. Caller must set pos.requires_grad_(True)."""
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces

    def freeze_backbone(self):
        """Freeze EquiformerV2 weights. Train only ECT branch + fusion head."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)
