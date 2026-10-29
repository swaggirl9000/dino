"""
QM9 training script: separate pretraining of each branch, then joint fine-tuning.

Phase 1 — EGNN backbone pretrained alone on the target.
Phase 2 — ECT branch pretrained alone on the target.
Phase 3 — weights transferred into JointRegressionModel, fine-tuned end-to-end.
"""

import time

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from ECT import ECTBranch
from EGNN import EGNN
from joint_model import JointRegressionModel, FiLMJointModel, LayerwiseFiLMJointModel
from qm9_ect import build_icosphere_directions, batched_point_ect

# QM9 target column indices (confirmed against torch_geometric's QM9 class
# docstring / processing code): 0 = mu (dipole moment), 1 = alpha
# (isotropic polarizability).
TARGET_COLUMNS = {"mu": 0, "alpha": 1}


class EGNNOnlyModel(torch.nn.Module):
    """EGNN backbone + regression head. use_head=False in EGNN so the backbone
    weights transfer directly into JointRegressionModel.egnn_branch."""

    def __init__(self, node_features, hidden_features, num_layers, radius=None):
        super().__init__()
        self.egnn = EGNN(
            node_features=node_features,
            hidden_features=hidden_features,
            out_features=hidden_features,
            num_layers=num_layers,
            dim=3,
            radius=radius,
            use_head=False,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, 1),
        )

    def forward(self, x, pos, edge_index, batch):
        return self.head(self.egnn(x, pos, edge_index, batch)).squeeze(-1)


class ECTOnlyModel(torch.nn.Module):
    """ECT branch + regression head for standalone pretraining."""

    def __init__(self, ect_edge_indices, out_dim=64):
        super().__init__()
        self.ect_branch = ECTBranch(ect_edge_indices, out_dim=out_dim)
        self.head = torch.nn.Linear(out_dim, 1)

    def forward(self, ect):
        return self.head(self.ect_branch(ect)).squeeze(-1)


def run_epoch(forward_fn, loader, optimizer, directions, target_idx, device, train=True):
    """Generic epoch loop. forward_fn(data, ect) -> (B,) predictions."""
    total_loss, n_batches = 0.0, 0

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for data in loader:
            data = data.to(device)
            target = data.y[:, target_idx]
            ect = batched_point_ect(data.pos, data.batch, directions.to(device))
            pred = forward_fn(data, ect)
            loss = torch.nn.functional.l1_loss(pred, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    if device == "cuda":
        torch.cuda.synchronize()
    return total_loss / n_batches, time.time() - start


def _log(epoch, tr_loss, tr_time, val_loss, val_time):
    print(
        f"  epoch {epoch:02d} | train MAE {tr_loss:.4f} ({tr_time:.1f}s)"
        f" | val MAE {val_loss:.4f} ({val_time:.1f}s)"
    )


def main(
    target: str = "alpha",
    pretrain_epochs: int = 500,
    finetune_epochs: int = 500,
    batch_size: int = 64,
    icosphere_nu: int = 3,
    phases=None,  # set of ints; None means run all
):
    run = lambda n: phases is None or n in phases
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_idx = TARGET_COLUMNS[target]
    hidden = 64

    dataset = QM9(root="./data/QM9")
    dataset = dataset.shuffle()
    n_val = len(dataset) // 10
    train_set, val_set = dataset[n_val:], dataset[:n_val]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    node_features = dataset.num_node_features
    directions, ect_edges = build_icosphere_directions(nu=icosphere_nu)

    egnn_model = ect_model = None
    results = {}

    # ------------------------------------------------------------------ #
    # Phase 1: Pretrain EGNN backbone alone                               #
    # ------------------------------------------------------------------ #
    if run(1):
        print(f"\n=== Phase 1: EGNN pretraining ({pretrain_epochs} epochs) ===")
        egnn_model = EGNNOnlyModel(node_features, hidden, num_layers=3).to(device)
        egnn_opt = torch.optim.Adam(egnn_model.parameters(), lr=1e-3)
        egnn_fn = lambda data, _: egnn_model(data.x, data.pos, data.edge_index, data.batch)
        for epoch in range(pretrain_epochs):
            tr = run_epoch(egnn_fn, train_loader, egnn_opt, directions, target_idx, device, train=True)
            val = run_epoch(egnn_fn, val_loader, egnn_opt, directions, target_idx, device, train=False)
            _log(epoch, *tr, *val)

    # ------------------------------------------------------------------ #
    # Phase 2: Pretrain ECT branch alone                                  #
    # ------------------------------------------------------------------ #
    if run(2):
        print(f"\n=== Phase 2: ECT pretraining ({pretrain_epochs} epochs) ===")
        ect_model = ECTOnlyModel(ect_edges, out_dim=hidden).to(device)
        ect_opt = torch.optim.Adam(ect_model.parameters(), lr=1e-4)
        ect_fn = lambda _, ect: ect_model(ect)
        for epoch in range(pretrain_epochs):
            tr = run_epoch(ect_fn, train_loader, ect_opt, directions, target_idx, device, train=True)
            val = run_epoch(ect_fn, val_loader, ect_opt, directions, target_idx, device, train=False)
            _log(epoch, *tr, *val)

    # ------------------------------------------------------------------ #
    # Phase 3: Concat — pretrained branches, fine-tuned jointly           #
    # ------------------------------------------------------------------ #
    if run(3):
        print(f"\n=== Phase 3: Concat from pretrained weights ({finetune_epochs} epochs) ===")
        assert egnn_model and ect_model, "Phase 3 requires phases 1 and 2 to have run."
        m = JointRegressionModel(ect_edge_indices=ect_edges, ect_out_dim=hidden,
                                 egnn_node_features=node_features, egnn_hidden_features=hidden,
                                 egnn_out_dim=hidden, egnn_num_layers=3).to(device)
        m.egnn_branch.load_state_dict(egnn_model.egnn.state_dict())
        m.ect_branch.load_state_dict(ect_model.ect_branch.state_dict())
        opt = torch.optim.Adam([{"params": m.ect_branch.parameters(), "lr": 1e-4},
                                 {"params": m.egnn_branch.parameters(), "lr": 1e-3},
                                 {"params": m.head.parameters(), "lr": 1e-3}])
        fn = lambda data, ect: m(ect, data.x, data.pos, data.edge_index, data.batch)
        for epoch in range(finetune_epochs):
            tr = run_epoch(fn, train_loader, opt, directions, target_idx, device, train=True)
            val = run_epoch(fn, val_loader, opt, directions, target_idx, device, train=False)
            _log(epoch, *tr, *val)
        results["Concat        | pretrained → fine-tuned"] = val[0]

    # ------------------------------------------------------------------ #
    # Phase 4: Concat — from scratch                                      #
    # ------------------------------------------------------------------ #
    if run(4):
        print(f"\n=== Phase 4: Concat from scratch ({finetune_epochs} epochs) ===")
        m = JointRegressionModel(ect_edge_indices=ect_edges, ect_out_dim=hidden,
                                 egnn_node_features=node_features, egnn_hidden_features=hidden,
                                 egnn_out_dim=hidden, egnn_num_layers=3).to(device)
        opt = torch.optim.Adam([{"params": m.ect_branch.parameters(), "lr": 1e-4},
                                 {"params": m.egnn_branch.parameters(), "lr": 1e-3},
                                 {"params": m.head.parameters(), "lr": 1e-3}])
        fn = lambda data, ect: m(ect, data.x, data.pos, data.edge_index, data.batch)
        for epoch in range(finetune_epochs):
            tr = run_epoch(fn, train_loader, opt, directions, target_idx, device, train=True)
            val = run_epoch(fn, val_loader, opt, directions, target_idx, device, train=False)
            _log(epoch, *tr, *val)
        results["Concat        | from scratch           "] = val[0]

    # ------------------------------------------------------------------ #
    # Phase 5: FiLM (output) — from scratch                               #
    # ------------------------------------------------------------------ #
    if run(5):
        print(f"\n=== Phase 5: FiLM (output) from scratch ({finetune_epochs} epochs) ===")
        m = FiLMJointModel(ect_edge_indices=ect_edges, ect_out_dim=hidden,
                           egnn_node_features=node_features, egnn_hidden_features=hidden,
                           egnn_num_layers=3).to(device)
        opt = torch.optim.Adam([{"params": m.ect_branch.parameters(), "lr": 1e-4},
                                 {"params": m.egnn_branch.parameters(), "lr": 1e-3},
                                 {"params": m.film_gamma.parameters(), "lr": 1e-3},
                                 {"params": m.film_beta.parameters(), "lr": 1e-3},
                                 {"params": m.head.parameters(), "lr": 1e-3}])
        fn = lambda data, ect: m(ect, data.x, data.pos, data.edge_index, data.batch)
        for epoch in range(finetune_epochs):
            tr = run_epoch(fn, train_loader, opt, directions, target_idx, device, train=True)
            val = run_epoch(fn, val_loader, opt, directions, target_idx, device, train=False)
            _log(epoch, *tr, *val)
        results["FiLM (output) | from scratch           "] = val[0]

    # ------------------------------------------------------------------ #
    # Phase 6: FiLM (output) — pretrained branches                        #
    # ------------------------------------------------------------------ #
    if run(6):
        print(f"\n=== Phase 6: FiLM (output) from pretrained weights ({finetune_epochs} epochs) ===")
        assert egnn_model and ect_model, "Phase 6 requires phases 1 and 2 to have run."
        m = FiLMJointModel(ect_edge_indices=ect_edges, ect_out_dim=hidden,
                           egnn_node_features=node_features, egnn_hidden_features=hidden,
                           egnn_num_layers=3).to(device)
        m.egnn_branch.load_state_dict(egnn_model.egnn.state_dict())
        m.ect_branch.load_state_dict(ect_model.ect_branch.state_dict())
        opt = torch.optim.Adam([{"params": m.ect_branch.parameters(), "lr": 1e-4},
                                 {"params": m.egnn_branch.parameters(), "lr": 1e-3},
                                 {"params": m.film_gamma.parameters(), "lr": 1e-3},
                                 {"params": m.film_beta.parameters(), "lr": 1e-3},
                                 {"params": m.head.parameters(), "lr": 1e-3}])
        fn = lambda data, ect: m(ect, data.x, data.pos, data.edge_index, data.batch)
        for epoch in range(finetune_epochs):
            tr = run_epoch(fn, train_loader, opt, directions, target_idx, device, train=True)
            val = run_epoch(fn, val_loader, opt, directions, target_idx, device, train=False)
            _log(epoch, *tr, *val)
        results["FiLM (output) | pretrained → fine-tuned"] = val[0]

    # ------------------------------------------------------------------ #
    # Phase 7: FiLM (layers) — from scratch                               #
    # ------------------------------------------------------------------ #
    if run(7):
        print(f"\n=== Phase 7: FiLM (layers) from scratch ({finetune_epochs} epochs) ===")
        m = LayerwiseFiLMJointModel(ect_edge_indices=ect_edges, ect_out_dim=hidden,
                                    egnn_node_features=node_features, egnn_hidden_features=hidden,
                                    egnn_num_layers=3).to(device)
        opt = torch.optim.Adam([{"params": m.ect_branch.parameters(), "lr": 1e-4},
                                 {"params": m.egnn_branch.parameters(), "lr": 1e-3},
                                 {"params": m.film_gammas.parameters(), "lr": 1e-3},
                                 {"params": m.film_betas.parameters(), "lr": 1e-3},
                                 {"params": m.head.parameters(), "lr": 1e-3}])
        fn = lambda data, ect: m(ect, data.x, data.pos, data.edge_index, data.batch)
        for epoch in range(finetune_epochs):
            tr = run_epoch(fn, train_loader, opt, directions, target_idx, device, train=True)
            val = run_epoch(fn, val_loader, opt, directions, target_idx, device, train=False)
            _log(epoch, *tr, *val)
        results["FiLM (layers) | from scratch           "] = val[0]

    # ------------------------------------------------------------------ #
    # Phase 8: FiLM (layers) — pretrained branches                        #
    # ------------------------------------------------------------------ #
    if run(8):
        print(f"\n=== Phase 8: FiLM (layers) from pretrained weights ({finetune_epochs} epochs) ===")
        assert egnn_model and ect_model, "Phase 8 requires phases 1 and 2 to have run."
        m = LayerwiseFiLMJointModel(ect_edge_indices=ect_edges, ect_out_dim=hidden,
                                    egnn_node_features=node_features, egnn_hidden_features=hidden,
                                    egnn_num_layers=3).to(device)
        m.egnn_branch.load_state_dict(egnn_model.egnn.state_dict())
        m.ect_branch.load_state_dict(ect_model.ect_branch.state_dict())
        opt = torch.optim.Adam([{"params": m.ect_branch.parameters(), "lr": 1e-4},
                                 {"params": m.egnn_branch.parameters(), "lr": 1e-3},
                                 {"params": m.film_gammas.parameters(), "lr": 1e-3},
                                 {"params": m.film_betas.parameters(), "lr": 1e-3},
                                 {"params": m.head.parameters(), "lr": 1e-3}])
        fn = lambda data, ect: m(ect, data.x, data.pos, data.edge_index, data.batch)
        for epoch in range(finetune_epochs):
            tr = run_epoch(fn, train_loader, opt, directions, target_idx, device, train=True)
            val = run_epoch(fn, val_loader, opt, directions, target_idx, device, train=False)
            _log(epoch, *tr, *val)
        results["FiLM (layers) | pretrained → fine-tuned"] = val[0]

    if results:
        print("\n=== Summary (final val MAE) ===")
        print(f"  {'Model':<36} {'Val MAE':>10}")
        print(f"  {'-'*48}")
        for label, mae in results.items():
            print(f"  {label:<36} {mae:>10.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="alpha", choices=list(TARGET_COLUMNS.keys()))
    parser.add_argument("--pretrain-epochs", type=int, default=500)
    parser.add_argument("--finetune-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--phases", type=int, nargs="+", default=None,
        help="Which phases to run, e.g. --phases 1 2 7. "
             "Phases 3/6/8 require 1&2 to have run first (pretrained weights).",
    )
    args = parser.parse_args()
    main(
        target=args.target,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        phases=set(args.phases) if args.phases else None,
    )
