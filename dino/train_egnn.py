"""
EGNN with and without ECT geometric token on QM9.

Conditioning types (--cond-type):
  concat      — concatenate ECT feature with graph embedding, MLP head  [default]
  film-final  — FiLM modulate the final pooled graph embedding
  film-layer  — FiLM modulate node features after each EGNN layer
  adaln-layer — AdaLN (DiT-style) after each EGNN layer

Usage:
    python train_egnn.py --model all                        # baseline + concat ECT
    python train_egnn.py --model ect --cond-type film-final
    python train_egnn.py --model ect --cond-type film-layer
    python train_egnn.py --model ect --cond-type adaln-layer
    python train_egnn.py --model baseline                   # no ECT
"""

import argparse
import time

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models.egnn_ect import (
    EGNNBaseline,
    EGNNWithECT,
    EGNNWithECT_FiLMFinal,
    EGNNWithECT_FiLMLayer,
    EGNNWithECT_AdaLNLayer,
)
from qm9_ect import build_icosphere_directions

TARGET_COLUMNS = {"mu": 0, "alpha": 1, "homo": 2, "lumo": 3, "gap": 4}

_COND_CLASSES = {
    "concat":      EGNNWithECT,
    "film-final":  EGNNWithECT_FiLMFinal,
    "film-layer":  EGNNWithECT_FiLMLayer,
    "adaln-layer": EGNNWithECT_AdaLNLayer,
}


def run_epoch(model, loader, optimizer, target_idx, device, train=True):
    model.train(train)
    total_loss, n = 0.0, 0
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for data in loader:
            data = data.to(device)
            pred = model(data.z, data.pos, data.batch)
            loss = torch.nn.functional.l1_loss(pred, data.y[:, target_idx])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n += 1
    if device == "cuda":
        torch.cuda.synchronize()
    return total_loss / n, time.time() - t0


def run_experiment(label, model, train_loader, val_loader, target_idx, device, epochs, lr):
    print(f"\n=== {label} ===")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-10,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)

    val_mae = None
    for epoch in range(epochs):
        tr_loss, tr_time = run_epoch(model, train_loader, optimizer, target_idx, device, train=True)
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            val_mae, val_time = run_epoch(model, val_loader, optimizer, target_idx, device, train=False)
            print(f"  epoch {epoch:03d} | train MAE {tr_loss:.4f} ({tr_time:.1f}s)"
                  f" | val MAE {val_mae:.4f} ({val_time:.1f}s)")

    return val_mae


def _build_ect_model(cond_type, ect_edges, backbone_kwargs, ect_out_dim, icosphere_nu, device):
    cls = _COND_CLASSES[cond_type]
    kwargs = dict(backbone_kwargs, ect_edge_indices=ect_edges, ect_out_dim=ect_out_dim,
                  icosphere_nu=icosphere_nu)
    # concat variant also takes mlp_hidden; others don't, so pop it safely
    if cond_type != "concat":
        kwargs.pop("mlp_hidden", None)
    return cls(**kwargs).to(device)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_idx = TARGET_COLUMNS[args.target]

    dataset = QM9(root="./data/QM9").shuffle()
    n_val = len(dataset) // 10
    train_loader = DataLoader(dataset[n_val:], batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(dataset[:n_val],  batch_size=args.batch_size)

    _, ect_edges = build_icosphere_directions(nu=args.icosphere_nu)

    backbone_kwargs = dict(
        hidden_features=args.hidden_features,
        num_layers=args.num_layers,
        cutoff=args.cutoff,
        mlp_hidden=args.mlp_hidden,
    )

    results = {}

    # Baseline
    if args.model in ("baseline", "all"):
        model = EGNNBaseline(
            hidden_features=args.hidden_features,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
        ).to(device)
        results["EGNN baseline"] = run_experiment(
            "EGNN baseline", model,
            train_loader, val_loader, target_idx, device,
            args.epochs, args.lr,
        )

    # --model all: train every conditioning variant in sequence
    # --model ect: train only the variant selected by --cond-type
    cond_types_to_run = (
        list(_COND_CLASSES.keys()) if args.model in ("all", "ect-all") else
        [args.cond_type]           if args.model == "ect"               else []
    )

    for cond_type in cond_types_to_run:
        label = f"EGNN + ECT ({cond_type})"
        model = _build_ect_model(
            cond_type, ect_edges, backbone_kwargs,
            args.ect_out_dim, args.icosphere_nu, device,
        )
        if args.freeze_backbone:
            model.freeze_backbone()
            print("  [backbone frozen — training ECT branch + head only]")
        results[label] = run_experiment(
            label, model,
            train_loader, val_loader, target_idx, device,
            args.epochs, args.lr,
        )

    # Summary table (all runs)
    if results:
        print(f"\n=== Summary — target: {args.target} (val MAE) ===")
        print(f"  {'Model':<30} {'Val MAE':>10}")
        print(f"  {'-' * 42}")
        for label, mae in results.items():
            print(f"  {label:<30} {mae:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",           default="alpha", choices=list(TARGET_COLUMNS.keys()))
    parser.add_argument("--epochs",           type=int,   default=1000)
    parser.add_argument("--batch-size",       type=int,   default=96)
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--hidden-features",  type=int,   default=128)
    parser.add_argument("--num-layers",       type=int,   default=7)
    parser.add_argument("--cutoff",           type=float, default=10.0)
    parser.add_argument("--ect-out-dim",      type=int,   default=64)
    parser.add_argument("--mlp-hidden",       type=int,   default=128,
                        help="MLP hidden dim for concat fusion head (ignored by FiLM/AdaLN).")
    parser.add_argument("--icosphere-nu",     type=int,   default=3)
    parser.add_argument("--cond-type",        default="concat",
                        choices=["concat", "film-final", "film-layer", "adaln-layer"],
                        help="How the ECT token conditions the backbone.")
    parser.add_argument("--model",            default="all",
                        choices=["all", "baseline", "ect", "ect-all"])
    parser.add_argument("--freeze-backbone",  action="store_true")
    args = parser.parse_args()
    main(args)
