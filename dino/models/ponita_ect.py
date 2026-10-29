import sys
import os
import copy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph, global_add_pool, global_mean_pool

from .ect_module import ECTPlugIn
from .egnn_ect import _LocalECTEncoder, _pca_canonical, _ECTEncoder
from qm9_ect import build_icosphere_directions, batched_point_ect
from ECT import ECTBranch

try:
    from ponita.models.ponita import Ponita
    _PONITA_AVAILABLE = True
except ImportError:
    _PONITA_AVAILABLE = False

try:
    from ponita.transforms.position_orientation_graph import PositionOrientationGraph as _PositionOrientationGraph
    from ponita.transforms.invariants import SEnInvariantAttributes as _SEnInvariantAttributesBase

    class PositionOrientationGraph(_PositionOrientationGraph):
        def forward(self, data):
            return data  # __call__ is overridden; forward never invoked

    class SEnInvariantAttributes(_SEnInvariantAttributesBase):
        def forward(self, data):
            return data  # same pattern — __call__ overridden in ponita

    # Ponita.__init__ also instantiates these transforms internally.
    # Patch both source modules so any import in ponita's own code gets the
    # concrete subclasses instead of the abstract base classes.
    import ponita.transforms.position_orientation_graph as _pog_mod
    import ponita.transforms.invariants as _inv_mod
    import ponita.models.ponita as _ponita_model_mod
    _pog_mod.PositionOrientationGraph = PositionOrientationGraph
    _inv_mod.SEnInvariantAttributes = SEnInvariantAttributes
    _ponita_model_mod.PositionOrientationGraph = PositionOrientationGraph
    _ponita_model_mod.SEnInvariantAttributes = SEnInvariantAttributes

    _TRANSFORMS_AVAILABLE = True
except ImportError:
    _TRANSFORMS_AVAILABLE = False


def _check_import():
    if not _PONITA_AVAILABLE:
        raise ImportError(
            "ponita is required. Install with:\n"
            "  pip install git+https://github.com/ebekkers/ponita.git"
        )


def get_ponita_pre_transform(num_ori: int = 20, radius: float = 1000.0):
    """Return a PyG Compose transform that lifts QM9 graphs to position-orientation space.

    Apply this as the `pre_transform` argument when loading the QM9 dataset so that
    the graph lifting is done once and cached on disk:

        dataset = QM9(root="./data/QM9_PONITA",
                      pre_transform=get_ponita_pre_transform(num_ori=20))

    The transform adds orientation fiber structure to each graph. After lifting:
      data.pos  — still [N, 3] atom positions (used by ECT)
      data.x    — lifted to [N, num_ori, feature_dim] (consumed by PONITA backbone)
      data.batch — still [N] (used by ECT)
    """
    if not _TRANSFORMS_AVAILABLE:
        raise ImportError(
            "ponita transforms not found. Install with:\n"
            "  pip install git+https://github.com/ebekkers/ponita.git"
        )
    from torch_geometric.transforms import Compose
    return Compose([
        PositionOrientationGraph(num_ori=num_ori, radius=radius),
        SEnInvariantAttributes(separable=True, point_cloud=(num_ori <= 0)),
    ])


def _make_backbone(input_dim, hidden_dim, num_layers, num_ori, radius, basis_dim,
                   multiple_readouts=True, readout="mlp"):
    # readout="mlp": PONITA sums atom features into one molecule embedding [B, H].
    # readout="sum": PONITA returns per-atom features [N, H]; the energy head sums
    #                per-atom energies, as in PONITA's own MD17 setup.
    _check_import()
    return Ponita(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_layers=num_layers,
        output_dim_vec=0,
        num_ori=num_ori,
        radius=radius,
        basis_dim=basis_dim,
        degree=3,
        widening_factor=4,
        task_level="node" if readout == "sum" else "graph",
        multiple_readouts=multiple_readouts,
    )


def _make_head(hidden_dim, readout, extra_dim=0):
    """Energy head. mlp: 2-layer MLP on the molecule embedding (parameter names
    unchanged from earlier runs). sum: one linear map per atom, summed."""
    if readout not in ("mlp", "sum"):
        raise ValueError(f"bad readout={readout!r}")
    d = hidden_dim + extra_dim
    if readout == "sum":
        return nn.Sequential(nn.Linear(d, 1))
    return nn.Sequential(nn.Linear(d, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))


def _energy(head, out, batch, readout, extra=None):
    """out: [B, H] (mlp) or [N, H] (sum). extra: optional molecule features [B, D]."""
    if readout == "sum" and out.shape[0] != batch.shape[0]:
        raise RuntimeError(f"expected per-atom backbone output, got {tuple(out.shape)}")
    if extra is not None:
        out = torch.cat([out, extra[batch] if readout == "sum" else extra], dim=-1)
    e = head(out)                                              # [B, 1] or [N, 1]
    if readout == "sum":
        e = global_add_pool(e, batch)                          # [B, 1]
    return e.squeeze(-1)


def _per_readout(t, batch, readout):
    """Broadcast a molecule-level tensor [B, H] to atoms when the readout is per atom."""
    return t[batch] if readout == "sum" else t


class PONITABaseline(nn.Module):
    """PONITA FiberBundle backbone + MLP head. No ECT.

    Expects data pre-transformed with get_ponita_pre_transform() so that
    orientation fiber structure is already attached.

    Forward input: PyG Data with
        data.x    — [N, num_ori, feature_dim]  lifted node features
        data.pos  — [N, 3]                      atom positions
        data.batch— [N]                          graph index per atom
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_ori: int = 20,
        radius: float = 1000.0,
        basis_dim: int = 256,
    ):
        super().__init__()
        self.backbone = _make_backbone(input_dim, hidden_dim, num_layers, num_ori, radius, basis_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        out, _ = self.backbone(data)          # [B, hidden_dim]
        return self.head(out).squeeze(-1)     # [B]

    def forward_with_forces(self, data):
        """Energy + forces via autograd. Caller must set data.pos.requires_grad_(True)."""
        energy = self.forward(data)
        forces = -torch.autograd.grad(
            energy.sum(), data.pos, create_graph=self.training
        )[0]
        return energy, forces


class PONITAWithECT(nn.Module):
    """PONITA FiberBundle backbone + ECT geometric token fused at the last layer.

    After the backbone produces a [B, hidden_dim] graph embedding, ECT is
    computed from the original 3D atom positions (data.pos, unchanged by the
    orientation-lifting transform) and fused via ECTPlugIn.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_ori: int = 20,
        radius: float = 1000.0,
        basis_dim: int = 256,
        ect_out_dim: int = 64,
        mlp_hidden: int = 128,
        icosphere_nu: int = 3,
        ect_resolution: int = 64,
        ect_radius: float = 8.0,
    ):
        super().__init__()
        self.backbone = _make_backbone(input_dim, hidden_dim, num_layers, num_ori, radius, basis_dim)

        directions, _ = build_icosphere_directions(nu=icosphere_nu)
        self.register_buffer("directions", directions)

        self.ect_plugin = ECTPlugIn(
            graph_dim=hidden_dim,
            ect_edge_indices=ect_edge_indices,
            ect_out_dim=ect_out_dim,
            mlp_hidden=mlp_hidden,
            resolution=ect_resolution,
            radius=ect_radius,
        )

    def forward(self, data):
        # Save pos/batch before backbone in case transforms modified data in-place
        pos, batch = data.pos, data.batch
        out, _ = self.backbone(data)     # [B, hidden_dim]
        # ECT uses original 3D atom positions [N, 3]; pos is unchanged by the
        # fiber-bundle transform (only data.x gains the orientation dimension).
        return self.ect_plugin(out, pos, batch, self.directions)

    def forward_with_forces(self, data):
        """Energy + forces via autograd. Caller must set data.pos.requires_grad_(True)."""
        energy = self.forward(data)
        forces = -torch.autograd.grad(
            energy.sum(), data.pos, create_graph=self.training
        )[0]
        return energy, forces

    def freeze_backbone(self):
        """Freeze PONITA weights. Train only ECT branch + fusion head."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)

# ---------------------------------------------------------------------------
# MD17-specific PONITA variants (energy + force prediction)
# ---------------------------------------------------------------------------

# One-hot atom type mapping used by PONITA for MD17 molecules
# Covers: H, C, N, O, F, S, Cl, P, Br
_Z_TO_IDX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 16: 5, 17: 6, 15: 7, 35: 8}
_INPUT_DIM = 9   # one-hot dimension


def z_to_one_hot(z: torch.Tensor) -> torch.Tensor:
    """Convert atomic numbers [N] to one-hot [N, 9]."""
    idx = torch.tensor([_Z_TO_IDX.get(int(zi), 0) for zi in z],
                       device=z.device, dtype=torch.long)
    return torch.zeros(len(z), _INPUT_DIM, device=z.device).scatter_(1, idx.unsqueeze(1), 1.0)


class PONITABaseline_MD17(nn.Module):
    """PONITA backbone for MD17 energy + force prediction. No ECT.

    Hyperparameters match the PONITA paper (Bekkers et al., ICLR 2024):
    hidden_dim=128, num_layers=5, num_ori=20, basis_dim=256, multiple_readouts=True.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_ori: int = 20,
        basis_dim: int = 256,
        radius: float = 1000.0,
        readout: str = "mlp",
    ):
        super().__init__()
        self.readout = readout
        self.backbone = _make_backbone(
            input_dim=_INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_ori=num_ori,
            radius=radius,
            basis_dim=basis_dim,
            multiple_readouts=True,
            readout=readout,
        )
        self.head = _make_head(hidden_dim, readout)

    def forward(self, z, pos, batch):
        data = _make_data(z, pos, batch)
        out, _ = self.backbone(data)
        return _energy(self.head, out, batch, self.readout)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class _TokenRegularizer(nn.Module):
    """Training-time regularisers on a conditioning token: one per molecule [B, D],
    or one per atom [N, D] when `batch` is given. Draws are always per molecule.

    noise: Gaussian noise at a random level sigma ~ U(0, noise) per molecule, in
           units of each feature's std over the batch, i.e. of the variation that
           tells conformations apart (all MD17 frames are one molecule). Relative,
           so the encoder can't outgrow it by scaling the token up. For per-atom
           tokens the std is over all atoms, so it also includes the differences
           between atoms and the same level is stronger.
    drop:  probability per molecule of replacing the whole token (all its atoms'
           tokens) with a learned null token (condition dropout), so the backbone
           also has to work alone.
    Both are off in eval mode. use_null=True feeds the null token to every
    molecule, to measure how much the trained model relies on the condition.
    """

    def __init__(self, dim: int, drop: float = 0.0, noise: float = 0.0):
        super().__init__()
        if not 0.0 <= drop < 1.0:
            raise ValueError(f"cond drop must be in [0, 1), got {drop}")
        if noise < 0.0:
            raise ValueError(f"cond noise must be >= 0, got {noise}")
        self.drop, self.noise = drop, noise
        # Only created when used, so checkpoints of runs without it still load.
        self.null = nn.Parameter(torch.zeros(dim)) if drop > 0 else None
        self.use_null = False

    def forward(self, cond, batch=None):
        if self.use_null:
            return self.null.expand_as(cond)
        if not self.training:
            return cond
        B = cond.shape[0] if batch is None else int(batch.max()) + 1

        def rows(t):                                   # [B, 1] -> one row per token
            return t if batch is None else t[batch]

        if self.noise > 0 and cond.shape[0] > 1:
            spread = cond.detach().std(dim=0, keepdim=True)                  # [1, D]
            level = rows(self.noise * torch.rand(B, 1, device=cond.device))  # [B|N, 1]
            cond = cond + level * spread * torch.randn_like(cond)
        if self.drop > 0:
            keep = rows((torch.rand(B, 1, device=cond.device) >= self.drop).to(cond.dtype))
            cond = keep * cond + (1 - keep) * self.null
        return cond


class PONITAWithECT_Concat_MD17(nn.Module):
    """PONITA + global ECT concatenated to atom input features.

    ECT is computed per-graph (PCA-aligned) and broadcast to every atom,
    then concatenated with the one-hot atom type before PONITA.
    cond_drop / cond_noise: see _TokenRegularizer. ect_encoder: see _ECTEncoder.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_ori: int = 20,
        basis_dim: int = 256,
        radius: float = 1000.0,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
        readout: str = "mlp",
        ect_dropout: float = 0.0,
        cond_drop: float = 0.0,
        cond_noise: float = 0.0,
        ect_encoder: str = "branch",
    ):
        super().__init__()
        self.readout = readout
        self.ect_encoder = _ECTEncoder(
            ect_edge_indices, ect_out_dim, icosphere_nu, resolution, ect_radius,
            encoder=ect_encoder,
        )
        # Dropout on the token, so it can't serve as an exact per-geometry fingerprint.
        self.ect_dropout = nn.Dropout(ect_dropout)
        self.cond_reg = _TokenRegularizer(ect_out_dim, drop=cond_drop, noise=cond_noise)
        self.backbone = _make_backbone(
            input_dim=_INPUT_DIM + ect_out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_ori=num_ori,
            radius=radius,
            basis_dim=basis_dim,
            multiple_readouts=True,
            readout=readout,
        )
        self.head = _make_head(hidden_dim, readout)

    def forward(self, z, pos, batch):
        ect_graph = self.ect_dropout(self.ect_encoder(pos, batch))  # [B, D]
        ect_graph = self.cond_reg(ect_graph)                      # [B, D]
        ect_node  = ect_graph[batch]                              # [N, D]
        x = torch.cat([z_to_one_hot(z), ect_node], dim=-1)       # [N, 9+D]
        data = _make_data(z, pos, batch, x_override=x)
        out, _ = self.backbone(data)
        return _energy(self.head, out, batch, self.readout)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class PONITAWithECT_FiLMFinal_MD17(nn.Module):
    """PONITA + global ECT FiLM conditioning at the graph level.

    After PONITA pooling:
        gamma, beta = SiLU -> Linear(ect_feat)
        out = (1 + 0.5*tanh(gamma)) * out + beta  -> head
    cond_drop / cond_noise: see _TokenRegularizer. ect_encoder: see _ECTEncoder.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_ori: int = 20,
        basis_dim: int = 256,
        radius: float = 1000.0,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
        readout: str = "mlp",
        cond_drop: float = 0.0,
        cond_noise: float = 0.0,
        ect_encoder: str = "branch",
    ):
        super().__init__()
        self.readout = readout
        self.backbone = _make_backbone(
            input_dim=_INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_ori=num_ori,
            radius=radius,
            basis_dim=basis_dim,
            multiple_readouts=True,
            readout=readout,
        )
        self.ect_encoder = _ECTEncoder(
            ect_edge_indices, ect_out_dim, icosphere_nu, resolution, ect_radius,
            encoder=ect_encoder,
        )
        film = nn.Sequential(nn.SiLU(), nn.Linear(ect_out_dim, 2 * hidden_dim))
        nn.init.zeros_(film[-1].weight)
        nn.init.zeros_(film[-1].bias)
        self.film = film
        self.cond_reg = _TokenRegularizer(ect_out_dim, drop=cond_drop, noise=cond_noise)
        self.head = _make_head(hidden_dim, readout)

    def forward(self, z, pos, batch):
        data = _make_data(z, pos, batch)
        out, _ = self.backbone(data)                          # [B, H] or [N, H]
        ect_feat = self.cond_reg(self.ect_encoder(pos, batch))  # [B, D]
        gamma, beta = self.film(ect_feat).chunk(2, dim=-1)
        # With the per-atom readout, every atom's features get its molecule's FiLM.
        gamma, beta = _per_readout(gamma, batch, self.readout), _per_readout(beta, batch, self.readout)
        out = (1 + 0.5 * torch.tanh(gamma)) * out + beta
        return _energy(self.head, out, batch, self.readout)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class PONITAWithLocalECT_MD17(nn.Module):
    """PONITA + LEAP-style local ECT as per-atom positional encoding.

    local_ect_i = PCA-invariant soft-ECT of atom i's neighborhood
    data.x = [one_hot(z_i) || local_ect_i]  -> PONITA -> head
    cond_drop / cond_noise: see _TokenRegularizer (per-atom tokens).
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,   # unused; kept for API compat
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_ori: int = 20,
        basis_dim: int = 256,
        radius: float = 1000.0,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
        local_cutoff: float = 5.0,
        readout: str = "mlp",
        cond_drop: float = 0.0,
        cond_noise: float = 0.0,
    ):
        super().__init__()
        self.readout = readout
        self.local_cutoff = local_cutoff
        self.local_ect = _LocalECTEncoder(ect_out_dim, icosphere_nu, resolution, ect_radius)
        self.local_reg = _TokenRegularizer(ect_out_dim, drop=cond_drop, noise=cond_noise)
        self.backbone = _make_backbone(
            input_dim=_INPUT_DIM + ect_out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_ori=num_ori,
            radius=radius,
            basis_dim=basis_dim,
            multiple_readouts=True,
            readout=readout,
        )
        self.head = _make_head(hidden_dim, readout)

    def forward(self, z, pos, batch):
        edge_index = radius_graph(pos, r=self.local_cutoff, batch=batch)
        local_ect = self.local_ect(pos, edge_index, batch)          # [N, D]
        local_ect = self.local_reg(local_ect, batch)                # [N, D]
        x = torch.cat([z_to_one_hot(z), local_ect], dim=-1)         # [N, 9+D]
        data = _make_data(z, pos, batch, x_override=x)
        out, _ = self.backbone(data)
        return _energy(self.head, out, batch, self.readout)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class PONITAWithLocalPooledECT_Concat_MD17(nn.Module):
    """PONITA + mean-pooled local ECT concatenated to atom input features.

    Per-atom local ECTs are computed from each atom's neighborhood (like leap),
    then mean-pooled to a single graph-level token [B, D]. That token is
    broadcast to every atom and concatenated with one-hot, same injection as
    the global-ECT concat variant — avoiding the per-atom gradient coupling
    that makes leap's forces degrade.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,   # unused; kept for API compat
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_ori: int = 20,
        basis_dim: int = 256,
        radius: float = 1000.0,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
        local_cutoff: float = 5.0,
    ):
        super().__init__()
        self.local_cutoff = local_cutoff
        self.local_ect = _LocalECTEncoder(ect_out_dim, icosphere_nu, resolution, ect_radius)
        self.backbone = _make_backbone(
            input_dim=_INPUT_DIM + ect_out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_ori=num_ori,
            radius=radius,
            basis_dim=basis_dim,
            multiple_readouts=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, pos, batch):
        edge_index = radius_graph(pos, r=self.local_cutoff, batch=batch)
        local_ect = self.local_ect(pos, edge_index, batch)   # [N, D]
        ect_graph = global_mean_pool(local_ect, batch)        # [B, D]
        ect_node  = ect_graph[batch]                          # [N, D]
        x = torch.cat([z_to_one_hot(z), ect_node], dim=-1)   # [N, 9+D]
        data = _make_data(z, pos, batch, x_override=x)
        out, _ = self.backbone(data)
        return self.head(out).squeeze(-1)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class PONITAWithLocalPooledECT_FiLMFinal_MD17(nn.Module):
    """PONITA + mean-pooled local ECT FiLM conditioning at the graph level.

    Per-atom local ECTs are mean-pooled to a graph-level token, which
    conditions the backbone's graph embedding via bounded FiLM — same
    injection as the global-ECT film-final variant.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,   # unused; kept for API compat
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_ori: int = 20,
        basis_dim: int = 256,
        radius: float = 1000.0,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
        local_cutoff: float = 5.0,
    ):
        super().__init__()
        self.local_cutoff = local_cutoff
        self.local_ect = _LocalECTEncoder(ect_out_dim, icosphere_nu, resolution, ect_radius)
        self.backbone = _make_backbone(
            input_dim=_INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_ori=num_ori,
            radius=radius,
            basis_dim=basis_dim,
            multiple_readouts=True,
        )
        film = nn.Sequential(nn.SiLU(), nn.Linear(ect_out_dim, 2 * hidden_dim))
        nn.init.zeros_(film[-1].weight)
        nn.init.zeros_(film[-1].bias)
        self.film = film
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, pos, batch):
        data = _make_data(z, pos, batch)
        out, _ = self.backbone(data)                                  # [B, H]
        edge_index = radius_graph(pos, r=self.local_cutoff, batch=batch)
        local_ect = self.local_ect(pos, edge_index, batch)            # [N, D]
        ect_feat = global_mean_pool(local_ect, batch)                 # [B, D]
        gamma, beta = self.film(ect_feat).chunk(2, dim=-1)
        out = (1 + 0.5 * torch.tanh(gamma)) * out + beta
        return self.head(out).squeeze(-1)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class PONITAWithECT_ConcatAblation_MD17(nn.Module):
    """Concat injection with the ECT point set and encoder chosen independently.

    point_set="global": soft ECT of the whole molecule -> [B, dirs, res]
    point_set="local":  soft ECT of each atom's neighborhood -> [N, dirs, res],
                        encoded per atom, then mean-pooled to [B, D]
    encoder="branch":   ECTBranch (conv over thresholds + SGConv over directions)
    encoder="mlp":      flatten -> Linear -> SiLU -> Linear
    The token is broadcast to atoms and concatenated with one-hot, as in concat.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        point_set: str,
        encoder: str,
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_ori: int = 20,
        basis_dim: int = 256,
        radius: float = 1000.0,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
        local_cutoff: float = 5.0,
    ):
        super().__init__()
        if point_set not in ("global", "local") or encoder not in ("branch", "mlp"):
            raise ValueError(f"bad point_set={point_set!r} / encoder={encoder!r}")
        self.point_set = point_set
        self.local_cutoff = local_cutoff
        self.resolution = resolution
        self.ect_radius = ect_radius

        directions, _ = build_icosphere_directions(nu=icosphere_nu)
        self.register_buffer("directions", directions)
        # Only raw() is used; drop its MLP so parameter counts stay comparable.
        self.local_ect = _LocalECTEncoder(ect_out_dim, icosphere_nu, resolution, ect_radius)
        self.local_ect.proj = nn.Identity()

        num_dirs = directions.shape[1]
        if encoder == "branch":
            self.encode = ECTBranch(ect_edge_indices, out_dim=ect_out_dim)
        else:
            self.encode = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(num_dirs * resolution, ect_out_dim),
                nn.SiLU(),
                nn.Linear(ect_out_dim, ect_out_dim),
            )

        self.backbone = _make_backbone(
            input_dim=_INPUT_DIM + ect_out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_ori=num_ori,
            radius=radius,
            basis_dim=basis_dim,
            multiple_readouts=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _token(self, pos, batch):
        if self.point_set == "global":
            grid = batched_point_ect(
                pos, batch, self.directions,
                resolution=self.resolution, radius=self.ect_radius,
            )                                                     # [B, dirs, res]
            return self.encode(grid)                              # [B, D]
        edge_index = radius_graph(pos, r=self.local_cutoff, batch=batch)
        grid = self.local_ect.raw(pos, edge_index, batch)         # [N, dirs, res]
        return global_mean_pool(self.encode(grid), batch)         # [B, D]

    def forward(self, z, pos, batch):
        ect_node = self._token(pos, batch)[batch]                 # [N, D]
        x = torch.cat([z_to_one_hot(z), ect_node], dim=-1)        # [N, 9+D]
        data = _make_data(z, pos, batch, x_override=x)
        out, _ = self.backbone(data)
        return self.head(out).squeeze(-1)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class _AdaptiveConvNext(nn.Module):
    """Wraps one PONITA ConvNext block with ECT conditioning (Proteina-style).

    Original block:  conv -> LN -> MLP -> layer_scale -> + residual
    Wrapped block:   AdaAffine -> conv -> LN -> AdaAffine -> MLP -> AdaScale
                     -> layer_scale -> + residual (residual on the raw input)

    AdaAffine: x * (1 + 0.5 tanh g) + b.   AdaScale: x * 2 sigmoid(s).
    ada_scale=False drops the AdaScale gate (AdaLN only).
    hidden_gate=True also gates the MLP hidden layer (adaptive GLU):
        h = act(linear_1(x)) * 2 sigmoid(W cond).
    All projections are zero-initialised, so the block starts out identical to
    the unwrapped one. The conditioning is per atom and per channel, shared
    across PONITA's orientation grid, which keeps the block equivariant.
    """

    def __init__(self, block: nn.Module, cond_dim: int, hidden_gate: bool = False,
                 ada_scale: bool = True):
        super().__init__()
        self.block = block
        channels = block.norm.normalized_shape[0]
        # Sizes of [g_in, b_in, g_ln, b_ln, scale gate, hidden gate]
        self.sizes = [channels] * 4 + [channels if ada_scale else 0,
                                       block.linear_1.out_features if hidden_gate else 0]
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, sum(self.sizes)))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)
        self.cond = None   # [N, cond_dim], set by the parent model before each forward

    def forward(self, x, edge_index, edge_attr, **kwargs):
        b = self.block
        if self.cond is None or self.cond.shape[0] != x.shape[0]:
            raise RuntimeError("ECT conditioning not set, or not one row per atom")
        parts = torch.split(self.mod(self.cond), self.sizes, dim=-1)       # each [N, *]
        if x.dim() == 3:   # fiber bundle features [N, num_ori, C]
            parts = [t.unsqueeze(1) for t in parts]
        g_in, b_in, g_ln, b_ln, s, hid = parts

        residual = x
        x = (1 + 0.5 * torch.tanh(g_in)) * x + b_in
        x = b.conv(x, edge_index, edge_attr, **kwargs)
        x = b.norm(x)
        x = (1 + 0.5 * torch.tanh(g_ln)) * x + b_ln
        h = b.act_fn(b.linear_1(x))
        if hid.shape[-1]:
            h = h * (2 * torch.sigmoid(hid))
        x = b.linear_2(h)
        if s.shape[-1]:
            x = x * (2 * torch.sigmoid(s))
        if b.layer_scale is not None:
            x = b.layer_scale * x
        if residual.shape == x.shape:
            x = x + residual
        return x


class PONITAWithECT_Adaptive_MD17(nn.Module):
    """PONITA + global ECT token conditioning every block (AdaLN + adaptive scale).

    Uses the global ECT with ECTBranch, the only combination that improved forces.
    input_concat=True also keeps the concat variant's input injection.
    ect_encoder: see _ECTEncoder.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        input_concat: bool,
        hidden_gate: bool = False,
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_ori: int = 20,
        basis_dim: int = 256,
        radius: float = 1000.0,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
        ect_encoder: str = "branch",
    ):
        super().__init__()
        self.input_concat = input_concat
        self.ect_encoder = _ECTEncoder(
            ect_edge_indices, ect_out_dim, icosphere_nu, resolution, ect_radius,
            encoder=ect_encoder,
        )
        self.backbone = _make_backbone(
            input_dim=_INPUT_DIM + (ect_out_dim if input_concat else 0),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_ori=num_ori,
            radius=radius,
            basis_dim=basis_dim,
            multiple_readouts=True,
        )
        layers = self.backbone.interaction_layers
        for i in range(len(layers)):
            layers[i] = _AdaptiveConvNext(layers[i], ect_out_dim, hidden_gate)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, pos, batch):
        ect_node = self.ect_encoder(pos, batch)[batch]                # [N, D]
        for layer in self.backbone.interaction_layers:
            layer.cond = ect_node
        x = torch.cat([z_to_one_hot(z), ect_node], dim=-1) if self.input_concat else None
        data = _make_data(z, pos, batch, x_override=x)
        out, _ = self.backbone(data)
        return self.head(out).squeeze(-1)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


class PONITAHybrid_MD17(nn.Module):
    """Something in the middle of PONITA + global ECT concat at the readout.

    middle="global": AdaLN in every block, driven by the global ECT token
                     (whole molecule, ECTBranch), same for every atom.
    middle="local":  LEAP, i.e. each atom's own local ECT (5 Å, MLP encoder)
                     concatenated to its input features. No AdaLN.
    In both, the global token is concatenated to PONITA's pooled molecule
    embedding just before the energy head.
    cond_drop / cond_noise: see _TokenRegularizer; applied to the global token
    and, for middle="local", independently to the local ECTs.
    ect_encoder: encoder of the global token, see _ECTEncoder. The local ECTs
    always use their MLP.
    """

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        middle: str,
        hidden_dim: int = 128,
        num_layers: int = 5,
        num_ori: int = 20,
        basis_dim: int = 256,
        radius: float = 1000.0,
        ect_out_dim: int = 64,
        icosphere_nu: int = 3,
        resolution: int = 64,
        ect_radius: float = 8.0,
        local_cutoff: float = 5.0,
        readout: str = "mlp",
        cond_drop: float = 0.0,
        cond_noise: float = 0.0,
        ect_encoder: str = "branch",
    ):
        super().__init__()
        if middle not in ("global", "local"):
            raise ValueError(f"bad middle={middle!r}")
        self.middle = middle
        self.readout = readout
        self.local_cutoff = local_cutoff
        self.ect_encoder = _ECTEncoder(
            ect_edge_indices, ect_out_dim, icosphere_nu, resolution, ect_radius,
            encoder=ect_encoder,
        )
        self.cond_reg = _TokenRegularizer(ect_out_dim, drop=cond_drop, noise=cond_noise)
        if middle == "local":
            self.local_ect = _LocalECTEncoder(ect_out_dim, icosphere_nu, resolution, ect_radius)
            self.local_reg = _TokenRegularizer(ect_out_dim, drop=cond_drop, noise=cond_noise)
        self.backbone = _make_backbone(
            input_dim=_INPUT_DIM + (ect_out_dim if middle == "local" else 0),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_ori=num_ori,
            radius=radius,
            basis_dim=basis_dim,
            multiple_readouts=True,
            readout=readout,
        )
        if middle == "global":
            layers = self.backbone.interaction_layers
            for i in range(len(layers)):
                layers[i] = _AdaptiveConvNext(layers[i], ect_out_dim, ada_scale=False)
        self.head = _make_head(hidden_dim, readout, extra_dim=ect_out_dim)

    def forward(self, z, pos, batch):
        ect_graph = self.cond_reg(self.ect_encoder(pos, batch))         # [B, D]
        if self.middle == "global":
            for layer in self.backbone.interaction_layers:
                layer.cond = ect_graph[batch]                           # [N, D]
            data = _make_data(z, pos, batch)
        else:
            edge_index = radius_graph(pos, r=self.local_cutoff, batch=batch)
            local_ect = self.local_reg(self.local_ect(pos, edge_index, batch), batch)  # [N, D]
            x = torch.cat([z_to_one_hot(z), local_ect], dim=-1)         # [N, 9+D]
            data = _make_data(z, pos, batch, x_override=x)
        out, _ = self.backbone(data)                                    # [B, H] or [N, H]
        return _energy(self.head, out, batch, self.readout, extra=ect_graph)

    def forward_with_forces(self, z, pos, batch):
        energy = self.forward(z, pos, batch)
        forces = -torch.autograd.grad(
            energy.sum(), pos, create_graph=self.training
        )[0]
        return energy, forces


# ---------------------------------------------------------------------------
# Helper: build a minimal PyG Data object from (z, pos, batch)
# ---------------------------------------------------------------------------

# PONITA's own MD17 setup adds self-loops to the graph. Off by default so earlier
# runs resume unchanged; the training script turns it on with --self-loops.
GRAPH_SELF_LOOPS = False


def set_graph_self_loops(on: bool):
    global GRAPH_SELF_LOOPS
    GRAPH_SELF_LOOPS = bool(on)


def _make_data(z, pos, batch, x_override=None):
    """Wrap raw tensors in a PyG Data object for PONITA's forward."""
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import radius_graph
    x = x_override if x_override is not None else z_to_one_hot(z)
    # Build per-graph Data objects, then batch them
    n_total = pos.shape[0]
    num_graphs = int(batch.max().item()) + 1
    data_list = []
    for g in range(num_graphs):
        mask = (batch == g)
        n_g = int(mask.sum())
        data_list.append(Data(
            x=x[mask],
            pos=pos[mask].detach(),
            num_nodes=n_g,
        ))
    batched = Batch.from_data_list(data_list)
    # Preserve original pos tensor (with grad) and full-batch metadata
    batched.pos = pos
    batched.x = x
    batched.batch = batch
    batched.num_nodes = n_total
    # PONITA's transform expands input edge_index into the SE(3) orientation graph.
    # Use r=1000 (effectively fully connected) with an explicit cap so radius_graph
    # doesn't silently truncate at its default max_num_neighbors=32.
    batched.edge_index = radius_graph(
        pos.detach(), r=1000.0, batch=batch, loop=GRAPH_SELF_LOOPS, max_num_neighbors=n_total
    )
    return batched
