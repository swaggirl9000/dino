"""
EGNN with and without ECT geometric token on rMD17.

Predicts energy (scalar) and forces (per-atom vectors) from atomic positions.
Forces are computed as F = -dE/dpos via autograd, matching PONITA's setup.

Loss: MAE(energy) + lambda_F * MAE(forces)

Molecules: aspirin (default), benzene, ethanol, malonaldehyde, naphthalene,
           salicylate, toluene, uracil, paracetamol, azobenzene

Split (following PONITA): 950 train, 50 val, rest test.

Usage:
    python train_egnn_md17.py                              # aspirin, all variants
    python train_egnn_md17.py --molecule malonaldehyde
    python train_egnn_md17.py --model ect --cond-type film-layer
    python train_egnn_md17.py --model baseline
"""

import argparse
import os
import random
import time

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader

from models.egnn_ect import (
    EGNNBaseline,
    EGNNWithECT,
    EGNNWithECT_FiLMFinal,
    EGNNWithECT_FiLMLayer,
    EGNNWithECT_AdaLNLayer,
    EGNNWithLocalECT,
)
from qm9_ect import build_icosphere_directions

_COND_CLASSES = {
    "concat":      EGNNWithECT,
    "film-final":  EGNNWithECT_FiLMFinal,
    "film-layer":  EGNNWithECT_FiLMLayer,
    "adaln-layer": EGNNWithECT_AdaLNLayer,
    "leap":        EGNNWithLocalECT,
}

# MD17 molecule names as expected by PyG (original MD17, stable download URL)
MOLECULE_MAP = {
    "aspirin":       "aspirin",
    "benzene":       "benzene",
    "ethanol":       "ethanol",
    "malonaldehyde": "malonaldehyde",
    "naphthalene":   "naphthalene",
    "salicylate":    "salicylate",
    "toluene":       "toluene",
    "uracil":        "uracil",
}

# kcal/mol → meV (matches PONITA's Kcal2meV transform)
KCAL_TO_MEV = 43.3634


def load_md17(root, molecule_name, train_size=950, val_size=50):
    """Load MD17; return (train_ds, val_ds, test_ds, energy_mean_kcal).

    energy_mean_kcal is the mean energy (kcal/mol) of the training set.
    Subtract it from targets before converting to meV so the model predicts
    deviations, not absolute molecular energies (~−300 000 kcal/mol for aspirin).
    """
    full = MD17(root=root, name=molecule_name)

    perm = torch.randperm(len(full), generator=torch.Generator().manual_seed(42))
    train_idx = perm[:train_size]
    val_idx   = perm[train_size:train_size + val_size]
    test_idx  = perm[train_size + val_size:]

    train_ds = full[train_idx]
    val_ds   = full[val_idx]
    test_ds  = full[test_idx]

    energies = torch.stack([train_ds[i].energy.squeeze() for i in range(len(train_ds))])
    energy_mean = float(energies.mean())
    energy_std = float(energies.std())
    return train_ds, val_ds, test_ds, energy_mean, energy_std


def run_epoch(model, loader, optimizer, device, lambda_f, energy_mean, scale=1.0, train=True):
    model.train(train)
    e_loss_sum, f_loss_sum, n = 0.0, 0.0, 0
    t0 = time.time()

    # Force computation requires autograd even at eval time — no torch.no_grad()
    for data in loader:
        data = data.to(device)

        # Leaf tensor with requires_grad so autograd.grad can compute forces
        data.pos = data.pos.requires_grad_(True)

        e_pred, f_pred = model.forward_with_forces(data.z, data.pos, data.batch)

        # Targets in meV, divided by `scale` (energy std) so the loss is O(1).
        # Forces share the energy scale (F = -dE/dpos), so both use the same factor.
        e_true = (data.energy.squeeze(-1) - energy_mean) * KCAL_TO_MEV / scale   # [B]
        f_true = data.force * KCAL_TO_MEV / scale                                 # [N, 3]

        e_loss = F.l1_loss(e_pred, e_true)
        f_loss = F.l1_loss(f_pred, f_true)
        loss   = e_loss + lambda_f * f_loss

        if train:
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        # Accumulate MAE back in physical meV units regardless of scaling.
        e_loss_sum += e_loss.item() * scale
        f_loss_sum += f_loss.item() * scale
        n += 1

    elapsed = time.time() - t0
    return e_loss_sum / max(n, 1), f_loss_sum / max(n, 1), elapsed


def run_experiment(label, model, train_loader, val_loader, device, args, energy_mean, scale, ckpt_path):
    print(f"\n=== {label} ===")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )

    best_val_e, best_val_f = float("inf"), float("inf")
    for epoch in range(args.epochs):
        tr_e, tr_f, tr_t = run_epoch(
            model, train_loader, optimizer, device, args.lambda_f, energy_mean, scale, train=True
        )

        if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
            val_e, val_f, val_t = run_epoch(
                model, val_loader, optimizer, device, args.lambda_f, energy_mean, scale, train=False
            )
            scheduler.step(val_f)
            if val_f < best_val_f:
                best_val_f = val_f
                best_val_e = val_e
                torch.save(model.state_dict(), ckpt_path)
            print(
                f"  epoch {epoch:04d} | "
                f"train E {tr_e:.3f} F {tr_f:.3f} ({tr_t:.1f}s) | "
                f"val E {val_e:.3f} F {val_f:.3f} ({val_t:.1f}s)"
                + (" *" if val_f == best_val_f else "")
            )

    # reload best checkpoint so final reported numbers match the saved model
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return best_val_e, best_val_f


def _build_ect_model(cond_type, ect_edges, backbone_kw, ect_out_dim, icosphere_nu, device):
    cls = _COND_CLASSES[cond_type]
    kw = dict(backbone_kw, ect_edge_indices=ect_edges,
              ect_out_dim=ect_out_dim, icosphere_nu=icosphere_nu)
    if cond_type != "concat":
        kw.pop("mlp_hidden", None)
    return cls(**kw).to(device)


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    _set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mol_name = MOLECULE_MAP.get(args.molecule, args.molecule)

    print(f"Molecule : {mol_name}")
    print(f"Seed     : {args.seed}")
    print(f"Device   : {device}")

    train_ds, val_ds, test_ds, energy_mean, energy_std = load_md17(
        args.data_root, mol_name, args.train_size, args.val_size
    )
    scale = (energy_std * KCAL_TO_MEV) if args.normalize else 1.0
    print(f"Energy mean (train): {energy_mean:.2f} kcal/mol  "
          f"({energy_mean * KCAL_TO_MEV:.1f} meV)")
    print(f"Normalize targets  : {args.normalize} (scale = {scale:.2f} meV)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    _, ect_edges = build_icosphere_directions(nu=args.icosphere_nu)

    backbone_kw = dict(
        hidden_features=args.hidden_features,
        num_layers=args.num_layers,
        cutoff=args.cutoff,
        mlp_hidden=args.mlp_hidden,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    results = {}       # (val_e, val_f) at best val epoch
    test_results = {}  # (test_e, test_f) of the saved best checkpoint

    def _ckpt(label):
        safe = label.replace(" ", "_").replace("(", "").replace(")", "")
        return os.path.join(args.ckpt_dir, f"{safe}_{mol_name}_seed{args.seed}.pt")

    if args.model in ("baseline", "all"):
        model = EGNNBaseline(
            hidden_features=args.hidden_features,
            num_layers=args.num_layers,
            cutoff=args.cutoff,
        ).to(device)
        results["EGNN baseline"] = run_experiment(
            "EGNN baseline", model, train_loader, val_loader, device, args, energy_mean, scale,
            _ckpt("EGNN_baseline"),
        )
        te, tf, _ = run_epoch(model, test_loader, None, device, args.lambda_f, energy_mean, scale, train=False)
        test_results["EGNN baseline"] = (te, tf)
        print(f"  → test  E {te:.3f} F {tf:.3f}")

    cond_types = (
        list(_COND_CLASSES.keys()) if args.model in ("all", "ect-all") else
        [args.cond_type]           if args.model == "ect"               else []
    )
    for cond_type in cond_types:
        label = f"EGNN + ECT ({cond_type})"
        model = _build_ect_model(
            cond_type, ect_edges, backbone_kw, args.ect_out_dim, args.icosphere_nu, device
        )
        results[label] = run_experiment(
            label, model, train_loader, val_loader, device, args, energy_mean, scale,
            _ckpt(label),
        )
        te, tf, _ = run_epoch(model, test_loader, None, device, args.lambda_f, energy_mean, scale, train=False)
        test_results[label] = (te, tf)
        print(f"  → test  E {te:.3f} F {tf:.3f}")

    if results:
        print(f"\n=== Summary — {mol_name} seed {args.seed} | val MAE at best checkpoint ===")
        print(f"  {'Model':<30} {'Val E':>10} {'Val F':>10} {'Test E':>10} {'Test F':>10}")
        print(f"  {'-' * 72}")
        for label, (ve, vf) in results.items():
            te, tf = test_results.get(label, (float("nan"), float("nan")))
            print(f"  {label:<30} {ve:>10.3f} {vf:>10.3f} {te:>10.3f} {tf:>10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule",       default="aspirin",
                        choices=list(MOLECULE_MAP.keys()))
    parser.add_argument("--data-root",      default="./data/md17")
    parser.add_argument("--train-size",     type=int,   default=950)
    parser.add_argument("--val-size",       type=int,   default=50)
    parser.add_argument("--epochs",         type=int,   default=600)
    parser.add_argument("--log-interval",   type=int,   default=10,
                        help="Print train/val stats every N epochs.")
    parser.add_argument("--batch-size",     type=int,   default=5)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-16)
    parser.add_argument("--lambda-f",       type=float, default=100.0,
                        help="Force loss coefficient (meV/Å units after conversion).")
    parser.add_argument("--lr-patience",   type=int,   default=30,
                        help="ReduceLROnPlateau patience in log-interval units.")
    parser.add_argument("--hidden-features",type=int,   default=128)
    parser.add_argument("--num-layers",     type=int,   default=7)
    parser.add_argument("--cutoff",         type=float, default=5.0)
    parser.add_argument("--ect-out-dim",    type=int,   default=64)
    parser.add_argument("--mlp-hidden",     type=int,   default=128)
    parser.add_argument("--icosphere-nu",   type=int,   default=3)
    parser.add_argument("--cond-type",      default="concat",
                        choices=["concat", "film-final", "film-layer", "adaln-layer", "leap"])
    parser.add_argument("--model",          default="all",
                        choices=["all", "baseline", "ect", "ect-all"])
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--ckpt-dir",       default="./checkpoints")
    parser.add_argument("--normalize",      action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Divide energy/force targets by the train energy std "
                             "for a stable, well-scaled loss (on by default).")
    args = parser.parse_args()
    main(args)
