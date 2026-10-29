import torch
from torch_geometric.utils import scatter

from mesh_lib import icosphere, triangles2edges


def build_icosphere_directions(nu: int = 3):
    """
    Returns:
      directions: (3, num_directions) float tensor, unit vectors
      edge_index: (2, num_edges) long tensor -- graph over the directions,
                  for ECTBranch's SGConv layers
    """
    v, f = icosphere(nu)
    directions = torch.from_numpy(v).float().T  # (3, num_directions)
    edge_index = triangles2edges(torch.from_numpy(f).int())
    return directions, edge_index


def random_so3(pos: torch.Tensor) -> torch.Tensor:
    """Random SO(3) rotation of positions. Apply during training only."""
    Q, _ = torch.linalg.qr(torch.randn(3, 3, device=pos.device, dtype=pos.dtype))
    Q = Q * torch.linalg.det(Q).sign()   # enforce det = +1
    return pos @ Q.T



def pca_canonical(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Rotate positions into a per-graph PCA canonical frame. Returns [N, 3].

    The same molecule in any orientation maps to the same canonical positions,
    so any descriptor computed from them is rotation-invariant.
    """
    num_graphs = int(batch.max().item()) + 1

    centroid = scatter(pos, batch, dim=0, dim_size=num_graphs, reduce="mean")
    pos_c = pos - centroid[batch]

    outer = torch.einsum("ni,nj->nij", pos_c, pos_c)          # [N, 3, 3]
    cov = scatter(
        outer.reshape(-1, 9), batch, dim=0,
        dim_size=num_graphs, reduce="mean",
    ).reshape(num_graphs, 3, 3)

    # eigh, not svd: forces backprop through this frame, and svd gradients
    # blow up / NaN when singular values are close (near-symmetric molecules).
    # eps*I shifts eigenvalues without changing eigenvectors.
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    cov = cov + 1e-6 * torch.eye(3, device=cov.device, dtype=cov.dtype)

    _, eigvecs = torch.linalg.eigh(cov)                        # ascending eigenvalues
    Vt = eigvecs.transpose(-1, -2).flip(-2)                    # rows=eigvecs, descending
    pos_canon = torch.einsum("bij,nj->ni", Vt[batch], pos_c)

    # Fix per-axis sign ambiguity via third moment
    skew = scatter(pos_canon.pow(3), batch, dim=0,
                   dim_size=num_graphs, reduce="sum")
    signs = torch.where(skew >= 0, skew.new_ones(skew.shape),
                        skew.new_full(skew.shape, -1.0))
    return pos_canon * signs[batch]


def batched_point_ect(
    pos: torch.Tensor,
    batch: torch.Tensor,
    directions: torch.Tensor,
    resolution: int = 64,
    radius: float = 8.0,
    scale: float = 64.0,
) -> torch.Tensor:
    """
    Parameters
    ----------
    pos : (N_total, 3) -- atom positions, concatenated across the whole
          batch (standard torch_geometric convention: data.pos after
          Batch.from_data_list).
    batch : (N_total,) -- graph index for each atom (data.batch).
    directions : (3, num_directions) -- fixed unit vectors, from
          build_icosphere_directions.
    resolution : number of threshold bins along each direction.
    radius : sublevel-set threshold range, [-radius, radius].
    scale : sigmoid steepness for the differentiable approximation.

    Returns
    -------
    (B, num_directions, resolution) tensor.
    """
    num_graphs = int(batch.max().item()) + 1
    pos_canonical = pca_canonical(pos, batch)                       # [N, 3]

    # Soft ECT
    ips = pos_canonical @ directions                                 # [N, num_dirs]
    thresholds = torch.linspace(-radius, radius, resolution, device=pos.device)
    diff = thresholds.view(1, 1, -1) - ips.unsqueeze(-1)           # [N, num_dirs, res]
    soft_below = torch.sigmoid(diff * scale)

    ect = pos.new_zeros(num_graphs, directions.shape[1], resolution)
    ect.index_add_(0, batch, soft_below)
    return ect