"""
QM9 training script for the joint ECT + EGNN regression model, matching the
meeting notes: baseline (EGNN only) vs. treatment (EGNN + ECT), same target,
timed, trained jointly with two learning rates.

Requires network access to download QM9 (~130k molecules, requires rdkit
for full processing, falls back to a pre-processed version without it) --
this hasn't been run end-to-end against the real download in this
environment; the data-shape/column-index logic below is checked directly
against the installed torch_geometric.datasets.qm9 source, and the model
wiring (ECT computation -> JointRegressionModel -> loss -> backward) is
verified separately against QM9-shaped synthetic batches.
"""

import time

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from ECT import ECTBranch
from EGNN import EGNN
from joint_model import JointRegressionModel
from qm9_ect import build_icosphere_directions, batched_point_ect

# QM9 target column indices (confirmed against torch_geometric's QM9 class
# docstring / processing code): 0 = mu (dipole moment), 1 = alpha
# (isotropic polarizability).
TARGET_COLUMNS = {"mu": 0, "alpha": 1}


class EGNNOnlyModel(torch.nn.Module):
    """Baseline: EGNN branch alone, no ECT branch, for the with/without comparison."""

    def __init__(self, node_features, hidden_features, out_features, num_layers, radius=None):
        super().__init__()
        self.egnn = EGNN(
            node_features=node_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_layers=num_layers,
            dim=3,
            radius=radius,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(out_features, out_features),
            torch.nn.ReLU(),
            torch.nn.Linear(out_features, 1),
        )

    def forward(self, x, pos, edge_index, batch):
        feat = self.egnn(x, pos, edge_index, batch)
        return self.head(feat).squeeze(-1)


def run_epoch(model, loader, optimizer, directions, target_idx, use_ect, device, train=True):
    model.train(train)
    total_loss, n_batches = 0.0, 0

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for data in loader:
            data = data.to(device)
            target = data.y[:, target_idx]

            if use_ect:
                ect = batched_point_ect(data.pos, data.batch, directions.to(device))
                pred = model(ect, data.x, data.pos, data.edge_index, data.batch)
            else:
                pred = model(data.x, data.pos, data.edge_index, data.batch)

            loss = torch.nn.functional.l1_loss(pred, target)  # MAE, standard for QM9

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return total_loss / n_batches, elapsed


def main(target: str = "alpha", epochs: int = 10, batch_size: int = 64, icosphere_nu: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_idx = TARGET_COLUMNS[target]

    dataset = QM9(root="./data/QM9")
    dataset = dataset.shuffle()
    n_val = len(dataset) // 10
    train_set, val_set = dataset[n_val:], dataset[:n_val]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    node_features = dataset.num_node_features  # 11, confirmed against qm9.py's x construction
    directions, ect_edges = build_icosphere_directions(nu=icosphere_nu)

    for use_ect in [False, True]:
        print(f"\n=== {'EGNN + ECT' if use_ect else 'EGNN baseline'} on target={target} ===")

        if use_ect:
            model = JointRegressionModel(
                ect_edge_indices=ect_edges,
                ect_out_dim=64,
                egnn_node_features=node_features,
                egnn_hidden_features=64,
                egnn_out_dim=64,
                egnn_num_layers=3,
            ).to(device)
            optimizer = torch.optim.Adam([
                {"params": model.ect_branch.parameters(), "lr": 1e-4},
                {"params": model.egnn_branch.parameters(), "lr": 1e-3},
                {"params": model.head.parameters(), "lr": 1e-3},
            ])
        else:
            model = EGNNOnlyModel(
                node_features=node_features,
                hidden_features=64,
                out_features=64,
                num_layers=3,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            train_loss, train_time = run_epoch(
                model, train_loader, optimizer, directions, target_idx, use_ect, device, train=True
            )
            val_loss, val_time = run_epoch(
                model, val_loader, optimizer, directions, target_idx, use_ect, device, train=False
            )
            print(
                f"epoch {epoch:02d} | train MAE {train_loss:.4f} ({train_time:.1f}s) "
                f"| val MAE {val_loss:.4f} ({val_time:.1f}s)"
            )


if __name__ == "__main__":
    main()