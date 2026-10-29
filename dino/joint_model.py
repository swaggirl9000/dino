import torch
import torch_geometric as tg
from torch import nn

from ECT import ECTBranch
from EGNN import EGNN


class JointRegressionModel(nn.Module):
    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        ect_out_dim: int,
        egnn_node_features: int,
        egnn_hidden_features: int,
        egnn_out_dim: int,
        egnn_num_layers: int,
        egnn_radius=None,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.ect_branch = ECTBranch(ect_edge_indices, out_dim=ect_out_dim)
        self.egnn_branch = EGNN(
            node_features=egnn_node_features,
            hidden_features=egnn_hidden_features,
            out_features=egnn_hidden_features,  # unused — use_head=False
            num_layers=egnn_num_layers,
            dim=3,
            radius=egnn_radius,
            use_head=False,
        )
        # concat dim: ECT features + raw pooled EGNN features (no internal head)
        self.head = nn.Sequential(
            nn.Linear(ect_out_dim + egnn_hidden_features, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, ect, egnn_x, pos, edge_index, batch):
        ect_feat = self.ect_branch(ect)                          
        egnn_feat = self.egnn_branch(egnn_x, pos, edge_index, batch)  
        combined = torch.cat([ect_feat, egnn_feat], dim=-1)
        return self.head(combined).squeeze(-1)                   


class FiLMJointModel(nn.Module):
    """ECT branch generates FiLM scale/shift (γ, β) to modulate EGNN features
    before the regression head, instead of concatenating the two feature vectors."""

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        ect_out_dim: int,
        egnn_node_features: int,
        egnn_hidden_features: int,
        egnn_num_layers: int,
        egnn_radius=None,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.ect_branch = ECTBranch(ect_edge_indices, out_dim=ect_out_dim)
        self.egnn_branch = EGNN(
            node_features=egnn_node_features,
            hidden_features=egnn_hidden_features,
            out_features=egnn_hidden_features,
            num_layers=egnn_num_layers,
            dim=3,
            radius=egnn_radius,
            use_head=False,
        )
        self.film_gamma = nn.Linear(ect_out_dim, egnn_hidden_features)
        self.film_beta = nn.Linear(ect_out_dim, egnn_hidden_features)
        self.head = nn.Sequential(
            nn.Linear(egnn_hidden_features, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, ect, egnn_x, pos, edge_index, batch):
        ect_feat = self.ect_branch(ect)
        egnn_feat = self.egnn_branch(egnn_x, pos, edge_index, batch)
        modulated = self.film_gamma(ect_feat) * egnn_feat + self.film_beta(ect_feat)
        return self.head(modulated).squeeze(-1)


class LayerwiseFiLMJointModel(nn.Module):
    """FiLM conditioning injected after every EGNN message-passing layer.
    ECT features are expanded from per-graph to per-node (via batch index)
    so each node's update is scaled and shifted by the molecule's global shape."""

    def __init__(
        self,
        ect_edge_indices: torch.Tensor,
        ect_out_dim: int,
        egnn_node_features: int,
        egnn_hidden_features: int,
        egnn_num_layers: int,
        egnn_radius=None,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        self.ect_branch = ECTBranch(ect_edge_indices, out_dim=ect_out_dim)
        self.egnn_branch = EGNN(
            node_features=egnn_node_features,
            hidden_features=egnn_hidden_features,
            out_features=egnn_hidden_features,
            num_layers=egnn_num_layers,
            dim=3,
            radius=egnn_radius,
            use_head=False,
        )
        # one (gamma, beta) pair per EGNN layer
        self.film_gammas = nn.ModuleList([
            nn.Linear(ect_out_dim, egnn_hidden_features) for _ in range(egnn_num_layers)
        ])
        self.film_betas = nn.ModuleList([
            nn.Linear(ect_out_dim, egnn_hidden_features) for _ in range(egnn_num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(egnn_hidden_features, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, ect, egnn_x, pos, edge_index, batch):
        ect_feat = self.ect_branch(ect)  # [B, ect_out_dim]

        # manually step through the EGNN so we can inject FiLM after each layer
        x = self.egnn_branch.embedder(egnn_x)
        dist = (pos[edge_index[1]] - pos[edge_index[0]]).pow(2).sum(-1, keepdim=True).sqrt()

        for i, layer in enumerate(self.egnn_branch.layers):
            x, pos = layer(x, pos, edge_index, dist)

            # ect_feat is [B, dim]; expand to [N_total, dim] via batch index
            gamma = self.film_gammas[i](ect_feat)[batch]  # [N_total, hidden]
            beta  = self.film_betas[i](ect_feat)[batch]   # [N_total, hidden]
            x = gamma * x + beta

            dist = (pos[edge_index[1]] - pos[edge_index[0]]).pow(2).sum(-1, keepdim=True).sqrt()
            if self.egnn_branch.radius:
                edge_index = tg.nn.radius_graph(pos, self.egnn_branch.radius, batch)
                dist = (pos[edge_index[1]] - pos[edge_index[0]]).pow(2).sum(-1, keepdim=True).sqrt()

        x = self.egnn_branch.pooler(x, batch)
        return self.head(x).squeeze(-1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/inner-product-transforms-main/src")
    sys.path.insert(0, "/home/claude")
    import subprocess

    from torch_geometric.data import Data, Batch

    torch.manual_seed(0)
    num_directions = 42  
    n_bins = 64

    src = torch.arange(num_directions)
    dst = (src + 1) % num_directions
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

    ect = torch.randn(4, num_directions, n_bins)

    graphs = []
    for _ in range(4):
        n_atoms = 5
        x = torch.randn(n_atoms, 11) 
        pos = torch.randn(n_atoms, 3)
        src2 = torch.randint(0, n_atoms, (10,))
        dst2 = torch.randint(0, n_atoms, (10,))
        ei = torch.stack([src2, dst2])
        graphs.append(Data(x=x, pos=pos, edge_index=ei))
    batch = Batch.from_data_list(graphs)

    model = JointRegressionModel(
        ect_edge_indices=edge_index,
        ect_out_dim=64,
        egnn_node_features=11,
        egnn_hidden_features=32,
        egnn_out_dim=64,
        egnn_num_layers=2,
        egnn_radius=None,
    )

    out = model(ect, batch.x, batch.pos, batch.edge_index, batch.batch)
    print("output shape:", out.shape, "expected: (4,)")
    assert out.shape == (4,)

    target = torch.randn(4)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    optimizer = torch.optim.Adam([
        {"params": model.ect_branch.parameters(), "lr": 1e-4},
        {"params": model.egnn_branch.parameters(), "lr": 1e-3},
        {"params": model.head.parameters(), "lr": 1e-3},
    ])
    print("OK: forward, backward, and dual-LR optimizer all check out")

    ect_grads = [p.grad.norm().item() for p in model.ect_branch.parameters() if p.grad is not None]
    egnn_grads = [p.grad.norm().item() for p in model.egnn_branch.parameters() if p.grad is not None]
    print(f"ECT branch: {len(ect_grads)} params with grad, mean norm {sum(ect_grads)/len(ect_grads):.4f}")
    print(f"EGNN branch: {len(egnn_grads)} params with grad, mean norm {sum(egnn_grads)/len(egnn_grads):.4f}")
    assert len(ect_grads) > 0 and len(egnn_grads) > 0
    print("OK: gradients confirmed flowing into both branches")