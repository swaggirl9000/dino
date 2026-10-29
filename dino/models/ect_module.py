import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from ECT import ECTBranch
from qm9_ect import batched_point_ect


class ECTPlugIn(nn.Module):
    """
    Shared ECT plug-in module. Accepts any backbone's graph-level embedding,
    fuses it with an ECT geometric token, and outputs a scalar prediction.

    Plug-in pattern:
        backbone → graph_feat [B, graph_dim]
                                    ↓
        pos → ECT → ect_feat [B, ect_out_dim]
                                    ↓
                  cat → [B, graph_dim + ect_out_dim] → head → [B]

    For forces: compute energy via forward(), then
        forces = -autograd.grad(energy.sum(), pos, create_graph=True)[0]
    This works because batched_point_ect is differentiable w.r.t. pos.
    """

    def __init__(
        self,
        graph_dim: int,
        ect_edge_indices: torch.Tensor,
        ect_out_dim: int = 64,
        mlp_hidden: int = 128,
        resolution: int = 64,
        radius: float = 8.0,
    ):
        super().__init__()
        self.ect_branch = ECTBranch(ect_edge_indices, out_dim=ect_out_dim)
        self.head = nn.Sequential(
            nn.Linear(graph_dim + ect_out_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.resolution = resolution
        self.radius = radius

    def forward(
        self,
        graph_feat: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """
        graph_feat : [B, graph_dim]
        pos        : [N, 3]  — atom positions; keep in autograd graph for forces
        batch      : [N]     — graph index per atom
        directions : [3, num_directions]
        Returns    : [B]     — scalar prediction per graph
        """
        ect = batched_point_ect(
            pos, batch, directions,
            resolution=self.resolution,
            radius=self.radius,
        )
        ect_feat = self.ect_branch(ect)
        combined = torch.cat([graph_feat, ect_feat], dim=-1)
        return self.head(combined).squeeze(-1)
