import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, radius_graph
from torch_scatter import scatter

from EGNN import EGNN
from ECT import ECTBranch
from .ect_module import ECTPlugIn
from qm9_ect import build_icosphere_directions, batched_point_ect
from qm9_ect import pca_canonical as _pca_canonical


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

def _make_egnn(hidden_features, num_layers):
    return EGNN(
        node_features=hidden_features,
        hidden_features=hidden_features,
        out_features=hidden_features,
        num_layers=num_layers,
        dim=3,
        radius=None,
        aggr="mean",
        act=nn.SiLU,
        pool=global_add_pool,
        use_head=False,
    )


def _edge_dist(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances for EGNN edge attributes. Returns [E, 1]."""
    return ((pos[edge_index[1]] - pos[edge_index[0]]).pow(2).sum(-1, keepdim=True) + 1e-8).sqrt()


# ---------------------------------------------------------------------------
# Shared ECT encoder (positions → graph-level ECT feature)
# ---------------------------------------------------------------------------

class _ECTEncoder(nn.Module):
    """Positions → ECT graph-level feature [B, ect_out_dim].
    Used by all conditioning variants that need direct access to the ECT feature
    before the final fusion head.

    encoder="branch": ECTBranch (conv over thresholds + SGConv over directions,
                      mean over directions, so invariant to icosphere symmetries)
    encoder="mlp":    flatten -> Linear -> SiLU -> Linear; invariant only
                      through the canonical frame of the ECT
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
        encoder: str = "branch",
    ):
        super().__init__()
        directions, _ = build_icosphere_directions(nu=icosphere_nu)
        self.register_buffer("directions", directions)
        # Separate attribute names, so a checkpoint of one encoder can't load into the other.
        if encoder == "branch":
            self.ect_branch = ECTBranch(ect_edge_indices, out_dim=ect_out_dim)
        elif encoder == "mlp":
            self.ect_mlp = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(directions.shape[1] * resolution, ect_out_dim),
                nn.SiLU(),
                nn.Linear(ect_out_dim, ect_out_dim),
            )
        else:
            raise ValueError(f"bad encoder={encoder!r}")
        self.encoder = encoder
        self.resolution = resolution
        self.ect_radius = ect_radius

    def forward(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        ect_raw = batched_point_ect(
            pos, batch, self.directions,
            resolution=self.resolution,
            radius=self.ect_radius,
        )
        encode = self.ect_branch if self.encoder == "branch" else self.ect_mlp
        return encode(ect_raw)   # [B, ect_out_dim]


# ---------------------------------------------------------------------------
# Baseline (no ECT)
# ---------------------------------------------------------------------------

class EGNNBaseline(nn.Module):
    """EGNN backbone + pooling + MLP head. No ECT."""

    def __init__(
        self,
        hidden_features: int = 128,
        num_layers: int = 7,
        cutoff: float = 10.0,
        max_z: int = 100,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.atom_embed = nn.Embedding(max_z, hidden_features)
        self.backbone = _make_egnn(hidden_features, num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, z, pos, batch):
        x = self.atom_embed(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        graph_feat = self.backbone(x, pos, edge_index, batch, update_pos=False)
        return self.head(graph_feat).squeeze(-1)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


# ---------------------------------------------------------------------------
# ECT variant 1 — concat fusion (original)
# ---------------------------------------------------------------------------

class EGNNWithECT(nn.Module):
    """EGNN + ECT fused at the final layer via concatenation + MLP head.

    graph_feat [B, H] || ect_feat [B, D] → MLP → [B]
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        hidden_features: int = 128,
        num_layers: int = 7,
        cutoff: float = 10.0,
        max_z: int = 100,
        ect_out_dim: int = 64,
        mlp_hidden: int = 128,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.atom_embed = nn.Embedding(max_z, hidden_features)
        self.backbone = _make_egnn(hidden_features, num_layers)

        directions, _ = build_icosphere_directions(nu=icosphere_nu)
        self.register_buffer("directions", directions)

        self.ect_plugin = ECTPlugIn(
            graph_dim=hidden_features,
            ect_edge_indices=ect_edge_indices,
            ect_out_dim=ect_out_dim,
            mlp_hidden=mlp_hidden,
            resolution=resolution,
            radius=ect_radius,
        )

    def forward(self, z, pos, batch):
        x = self.atom_embed(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        graph_feat = self.backbone(x, pos, edge_index, batch, update_pos=False)
        return self.ect_plugin(graph_feat, pos, batch, self.directions)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.atom_embed.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        for p in self.atom_embed.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# ECT variant 2 — FiLM at the final (graph-level) layer
# ---------------------------------------------------------------------------

class EGNNWithECT_FiLMFinal(nn.Module):
    """FiLM conditioning applied once to the final pooled graph embedding.

    γ, β = Linear(ect_feat)             [B, H] each
    out  = (1 + γ) ⊙ graph_feat + β    [B, H]
         → Linear head                  [B]

    Zero-initialized projection ensures identity conditioning at init.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        hidden_features: int = 128,
        num_layers: int = 7,
        cutoff: float = 10.0,
        max_z: int = 100,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.atom_embed = nn.Embedding(max_z, hidden_features)
        self.backbone = _make_egnn(hidden_features, num_layers)
        self.ect_encoder = _ECTEncoder(
            ect_edge_indices, ect_out_dim, icosphere_nu, resolution, ect_radius
        )

        self.film = nn.Linear(ect_out_dim, 2 * hidden_features)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.head = nn.Linear(hidden_features, 1)

    def forward(self, z, pos, batch):
        x = self.atom_embed(z)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        graph_feat = self.backbone(x, pos, edge_index, batch, update_pos=False)   # [B, H]

        ect_feat = self.ect_encoder(pos, batch)                                    # [B, D]
        gamma, beta = self.film(ect_feat).chunk(2, dim=-1)                         # [B, H] each
        return self.head((1 + gamma) * graph_feat + beta).squeeze(-1)             # [B]

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.atom_embed.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        for p in self.atom_embed.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# ECT variant 3 — FiLM after each EGNN layer
# ---------------------------------------------------------------------------

class EGNNWithECT_FiLMLayer(nn.Module):
    """FiLM conditioning injected after every EGNN message-passing layer.

    At layer l (node-level):
        γ_l, β_l = Linear_l(ect_node)    [N, H] each
        h         = (1 + γ_l) ⊙ h + β_l

    ECT is computed once from the original positions (before EGNN updates them).
    Zero-initialized projections ensure identity conditioning at init.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        hidden_features: int = 128,
        num_layers: int = 7,
        cutoff: float = 10.0,
        max_z: int = 100,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.atom_embed = nn.Embedding(max_z, hidden_features)
        self.backbone = _make_egnn(hidden_features, num_layers)
        self.ect_encoder = _ECTEncoder(
            ect_edge_indices, ect_out_dim, icosphere_nu, resolution, ect_radius
        )

        self.film_layers = nn.ModuleList()
        for _ in range(num_layers):
            proj = nn.Sequential(nn.SiLU(), nn.Linear(ect_out_dim, 2 * hidden_features))
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)
            self.film_layers.append(proj)

        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, z, pos, batch):
        ect_feat = self.ect_encoder(pos, batch)   # [B, D]
        ect_node  = ect_feat[batch]                # [N, D]

        x = self.atom_embed(z)
        x = self.backbone.embedder(x)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        dist = _edge_dist(pos, edge_index)   # fixed positions — compute once

        for l, layer in enumerate(self.backbone.layers):
            x, _ = layer(x, pos, edge_index, dist)   # ignore returned pos
            gamma, beta = self.film_layers[l](ect_node).chunk(2, dim=-1)
            x = (1 + 0.5 * torch.tanh(gamma)) * x + beta   # scale bounded to [0.5, 1.5], never zeros features

        graph_feat = self.backbone.pooler(x, batch)   # [B, H]
        return self.head(graph_feat).squeeze(-1)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.atom_embed.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        for p in self.atom_embed.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# ECT variant 4 — AdaLN after each EGNN layer  (DiT-style)
# ---------------------------------------------------------------------------

class EGNNWithECT_AdaLNLayer(nn.Module):
    """FiLM conditioning after every EGNN layer, with the ECT signal layer-normed first.

    At each layer l:
        s          = LN_no_bias(ect_node)           stabilize conditioning magnitude
        γ_l, β_l  = SiLU → Linear_l(s)             zero-initialized
        x          = (1 + 0.5·tanh(γ_l)) ⊙ x + β_l  scale ∈ [0.5, 1.5]

    Normalizing x (AdaLN-style) is deliberately avoided: LN on node features kills the
    feature scale that autograd needs to propagate forces (dE/dpos) through the network.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        hidden_features: int = 128,
        num_layers: int = 7,
        cutoff: float = 10.0,
        max_z: int = 100,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.atom_embed = nn.Embedding(max_z, hidden_features)
        self.backbone = _make_egnn(hidden_features, num_layers)
        self.ect_encoder = _ECTEncoder(
            ect_edge_indices, ect_out_dim, icosphere_nu, resolution, ect_radius
        )

        self.cond_norms  = nn.ModuleList()   # LN(D, bias=False) — normalize ECT signal only
        self.film_layers = nn.ModuleList()   # SiLU → Linear(D, 2H) — zero-initialized
        for _ in range(num_layers):
            self.cond_norms.append(nn.LayerNorm(ect_out_dim, bias=False))
            proj = nn.Sequential(nn.SiLU(), nn.Linear(ect_out_dim, 2 * hidden_features))
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)
            self.film_layers.append(proj)

        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, z, pos, batch):
        ect_feat = self.ect_encoder(pos, batch)   # [B, D]
        ect_node  = ect_feat[batch]                # [N, D]

        x = self.atom_embed(z)
        x = self.backbone.embedder(x)
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        dist = _edge_dist(pos, edge_index)

        for l, layer in enumerate(self.backbone.layers):
            x, _ = layer(x, pos, edge_index, dist)
            s = self.cond_norms[l](ect_node)
            gamma, beta = self.film_layers[l](s).chunk(2, dim=-1)
            x = (1 + 0.5 * torch.tanh(gamma)) * x + beta   # scale ∈ [0.5, 1.5]

        graph_feat = self.backbone.pooler(x, batch)   # [B, H]
        return self.head(graph_feat).squeeze(-1)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.atom_embed.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        for p in self.atom_embed.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# LEAP — Local ECT as per-atom positional encoding (Amboage et al., ICLR 2026)
# ---------------------------------------------------------------------------

class _LocalECTEncoder(nn.Module):
    """Per-atom local ECT: soft-ECT of each atom's radius-graph neighborhood.

    For atom i with neighbors N(i) from the molecular radius graph:
        r_ij = pos_j - pos_i           relative positions, centered at i
        local_ect[i] = soft-ECT({r_ij : j in N(i) union {i}})

    Computed via a single scatter-add over edges — no Python loop over atoms.
    Output: [N, ect_out_dim].
    """

    def __init__(
        self,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
        scale: float = 64.0,
    ):
        super().__init__()
        directions, _ = build_icosphere_directions(nu=icosphere_nu)
        self.register_buffer("directions", directions)   # [3, num_dirs]
        self.resolution = resolution
        self.ect_radius = ect_radius
        self.scale = scale

        num_dirs = directions.shape[1]
        self.proj = nn.Sequential(
            nn.Linear(num_dirs * resolution, ect_out_dim),
            nn.SiLU(),
            nn.Linear(ect_out_dim, ect_out_dim),
        )

    def forward(
        self, pos: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        return self.proj(self.raw(pos, edge_index, batch).flatten(1))   # [N, ect_out_dim]

    def raw(
        self, pos: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Unencoded per-atom soft ECT, [N, num_dirs, resolution]."""
        N = pos.shape[0]
        num_dirs = self.directions.shape[1]
        device = pos.device

        # Rotate into PCA canonical frame so local ECT is rotation-invariant
        pos_canon = _pca_canonical(pos, batch)          # [N, 3]

        # Add self-loops so each atom contributes to its own local ECT
        self_idx = torch.arange(N, device=device)
        ei = torch.cat([edge_index, torch.stack([self_idx, self_idx])], dim=1)
        dst, src = ei[0], ei[1]

        # Relative positions in canonical frame (centered at destination atom i)
        rel_pos = pos_canon[src] - pos_canon[dst]       # [E+N, 3]

        # Project onto icosphere directions
        ips = rel_pos @ self.directions                  # [E+N, num_dirs]

        # Differentiable soft-ECT sublevel set
        thresholds = torch.linspace(
            -self.ect_radius, self.ect_radius, self.resolution, device=device
        )
        diff = thresholds.view(1, 1, -1) - ips.unsqueeze(-1)   # [E+N, num_dirs, res]
        soft_below = torch.sigmoid(diff * self.scale)

        # Scatter-add per atom: [N, num_dirs, resolution]
        local_ect = soft_below.new_zeros(N, num_dirs, self.resolution)
        local_ect.scatter_add_(0, dst.view(-1, 1, 1).expand_as(soft_below), soft_below)

        return local_ect


class EGNNWithLocalECT(nn.Module):
    """LEAP-style local ECT injected as a per-atom positional encoding.

    For each atom i:
        local_ect_i = soft-ECT of i's radius-graph neighborhood (relative positions)
        x_i = Linear([atom_embed(z_i) || local_ect_i])

    Then standard EGNN message passing -> global pool -> energy head.
    Forces via autograd: F = -dE/dpos.

    Unlike global-ECT conditioning variants, local ECT sits entirely in the
    input embedding, with no per-layer interference with the force gradient path.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,   # unused; kept for API compatibility
        hidden_features: int = 128,
        num_layers: int = 7,
        cutoff: float = 10.0,
        max_z: int = 100,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.atom_embed = nn.Embedding(max_z, hidden_features)
        self.local_ect = _LocalECTEncoder(ect_out_dim, icosphere_nu, resolution, ect_radius)
        self.input_proj = nn.Linear(hidden_features + ect_out_dim, hidden_features)
        self.backbone = _make_egnn(hidden_features, num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, z, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        local_ect = self.local_ect(pos, edge_index, batch)                       # [N, D]
        x = self.input_proj(torch.cat([self.atom_embed(z), local_ect], dim=-1))  # [N, H]
        graph_feat = self.backbone(x, pos, edge_index, batch, update_pos=False)
        return self.head(graph_feat).squeeze(-1)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.atom_embed.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        for p in self.atom_embed.parameters():
            p.requires_grad_(True)
