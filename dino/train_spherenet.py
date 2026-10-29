"""
SphereNet (DIG) with and without ECT geometric token on QM9.

Runs two experiments back-to-back under identical settings and prints a
comparison table at the end.

Usage:
    python train_spherenet.py                         # default: alpha, 300 epochs
    python train_spherenet.py --target mu --epochs 500
    python train_spherenet.py --model ect             # only ECT variant
    python train_spherenet.py --model baseline        # only baseline
    python train_spherenet.py --freeze-backbone       # ECT only; SphereNet frozen

Requires:
    pip install dive-into-graphs
"""

import argparse
import time

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from models.spherenet_ect import SphereNetBaseline, SphereNetWithECT
from qm9_ect import build_icosphere_directions

TARGET_COLUMNS = {"mu": 0, "alpha": 1, "homo": 2, "lumo": 3, "gap": 4}


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


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_idx = TARGET_COLUMNS[args.target]

    dataset = QM9(root="./data/QM9").shuffle()
    n_val = len(dataset) // 10
    train_loader = DataLoader(dataset[n_val:], batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(dataset[:n_val],  batch_size=args.batch_size)

    _, ect_edges = build_icosphere_directions(nu=args.icosphere_nu)

    shared_backbone_kwargs = dict(
        hidden_channels=args.hidden_channels,
        out_emb_channels=args.out_emb_channels,
        num_layers=args.num_layers,
        cutoff=args.cutoff,
    )

    results = {}

    if args.model in ("baseline", "all"):
        model = SphereNetBaseline(**shared_backbone_kwargs).to(device)
        results["SphereNet baseline"] = run_experiment(
            "SphereNet baseline", model,
            train_loader, val_loader, target_idx, device,
            args.epochs, args.lr,
        )

    if args.model in ("ect", "all"):
        model = SphereNetWithECT(
            ect_edge_indices=ect_edges,
            ect_out_dim=args.ect_out_dim,
            icosphere_nu=args.icosphere_nu,
            **shared_backbone_kwargs,
        ).to(device)
        if args.freeze_backbone:
            model.freeze_backbone()
            print("  [backbone frozen — training ECT branch + head only]")
        results["SphereNet + ECT"] = run_experiment(
            "SphereNet + ECT", model,
            train_loader, val_loader, target_idx, device,
            args.epochs, args.lr,
        )

    if len(results) > 1:
        print(f"\n=== Summary — target: {args.target} (val MAE) ===")
        print(f"  {'Model':<28} {'Val MAE':>10}")
        print(f"  {'-' * 40}")
        for label, mae in results.items():
            print(f"  {label:<28} {mae:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",           default="alpha", choices=list(TARGET_COLUMNS.keys()))
    parser.add_argument("--epochs",           type=int,   default=300)
    parser.add_argument("--batch-size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--hidden-channels",  type=int,   default=128)
    parser.add_argument("--out-emb-channels", type=int,   default=256)
    parser.add_argument("--num-layers",       type=int,   default=4)
    parser.add_argument("--cutoff",           type=float, default=5.0)
    parser.add_argument("--ect-out-dim",      type=int,   default=64)
    parser.add_argument("--icosphere-nu",     type=int,   default=3)
    parser.add_argument("--model",            default="all", choices=["all", "baseline", "ect"])
    parser.add_argument("--freeze-backbone",  action="store_true",
                        help="Freeze SphereNet weights; train only ECT branch + head.")
    args = parser.parse_args()
    main(args)
