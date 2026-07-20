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
    radius : sublevel-set threshold range, [-radius, radius]. QM9
          molecules are small (bond lengths ~1-2 Angstrom, a few dozen
          atoms) -- 8 is a generous default, tighten if your molecules are
          consistently smaller after centering.
    scale : sigmoid steepness for the differentiable "point <= threshold"
          approximation. Higher = closer to a hard step function, but with
          smaller/noisier gradients.

    Returns
    -------
    (B, num_directions, resolution) tensor.
    """
    num_graphs = int(batch.max().item()) + 1

    centroid = scatter(pos, batch, dim=0, dim_size=num_graphs, reduce="mean")
    pos_centered = pos - centroid[batch]

    ips = pos_centered @ directions 
    thresholds = torch.linspace(-radius, radius, resolution, device=pos.device)
    diff = thresholds.view(1, 1, -1) - ips.unsqueeze(-1)  
    soft_below = torch.sigmoid(diff * scale)

    ect = pos.new_zeros(num_graphs, directions.shape[1], resolution)
    ect.index_add_(0, batch, soft_below)
    return ect