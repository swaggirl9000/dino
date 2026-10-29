"""
PONITA with and without ECT geometric token on MD17.

Predicts energy (scalar) and forces (per-atom vectors) from atomic positions.
Forces computed as F = -dE/dpos via autograd.

Hyperparameters match PONITA paper (Bekkers et al., ICLR 2024):
  hidden_dim=128, num_layers=5, num_ori=20, basis_dim=256,
  epochs=5000, lr=5e-4, lambda_f=500, warmup=100, grad_clip=0.5

Models:
  baseline    -- PONITA, no ECT
  film-final  -- PONITA + global ECT FiLM at graph level
  leap        -- PONITA + local ECT per-atom positional encoding

Usage:
    python train_ponita_md17.py                     # all models, aspirin
    python train_ponita_md17.py --model baseline
    python train_ponita_md17.py --model ect --cond-type leap
    python train_ponita_md17.py --molecule ethanol
"""

import argparse
import contextlib
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader

from models.ponita_ect import (
    set_graph_self_loops,
    _TokenRegularizer,
    PONITABaseline_MD17,
    PONITAWithECT_Concat_MD17,
    PONITAWithECT_FiLMFinal_MD17,
    PONITAWithLocalECT_MD17,
    PONITAWithLocalPooledECT_Concat_MD17,
    PONITAWithLocalPooledECT_FiLMFinal_MD17,
    PONITAWithECT_ConcatAblation_MD17,
    PONITAWithECT_Adaptive_MD17,
    PONITAHybrid_MD17,
)
from functools import partial
from qm9_ect import build_icosphere_directions

_COND_CLASSES = {
    "concat":        PONITAWithECT_Concat_MD17,
    "film-final":    PONITAWithECT_FiLMFinal_MD17,
    "leap":          PONITAWithLocalECT_MD17,
    "concat-local":  PONITAWithLocalPooledECT_Concat_MD17,
    "film-local":    PONITAWithLocalPooledECT_FiLMFinal_MD17,
    # Encoder-swap ablation: global ECT + flat MLP, local ECT + ECTBranch.
    "concat-mlp":          partial(PONITAWithECT_ConcatAblation_MD17,
                                   point_set="global", encoder="mlp"),
    "concat-local-branch": partial(PONITAWithECT_ConcatAblation_MD17,
                                   point_set="local", encoder="branch"),
    # Global ECT conditioning every PONITA block (AdaLN + adaptive scale).
    "concat-adaptive":     partial(PONITAWithECT_Adaptive_MD17, input_concat=True),
    "adaptive":            partial(PONITAWithECT_Adaptive_MD17, input_concat=False),
    "concat-adaptive-glu": partial(PONITAWithECT_Adaptive_MD17, input_concat=True,
                                   hidden_gate=True),
    # Hybrids, both with global ECT concat before the energy head:
    # hybrid-adaln = AdaLN (global ECT) in every block; hybrid-leap = LEAP input concat.
    "hybrid-adaln":        partial(PONITAHybrid_MD17, middle="global"),
    "hybrid-leap":         partial(PONITAHybrid_MD17, middle="local"),
}
# Variants that support the per-atom summed readout (--readout sum).
_SUM_READOUT_TYPES = {"concat", "film-final", "leap", "hybrid-adaln", "hybrid-leap"}
# Variants that support --cond-drop / --cond-noise on their ECT tokens.
_COND_REG_TYPES = {"concat", "film-final", "leap", "hybrid-adaln", "hybrid-leap"}
# Variants whose global ECT token can use either encoder (--ect-encoder).
_ENCODER_TYPES = {"concat", "film-final", "concat-adaptive", "adaptive",
                  "concat-adaptive-glu", "hybrid-adaln", "hybrid-leap"}
# Original variants run by --model all / ect-all; the ablations are opt-in.
_DEFAULT_COND_TYPES = ["concat", "film-final", "leap"]

MOLECULE_MAP = {
    "aspirin":       "aspirin",
    "benzene":       "benzene",
    "ethanol":       "ethanol",
    "malonaldehyde": "malonaldehyde",
    "naphthalene":   "naphthalene",
    "salicylate":    "salicylate",
    "toluene":       "toluene",
    "uracil":        "uracil",
    # Revised MD17 (rMD17, Christensen & von Lilienfeld 2020), the version
    # PONITA reports on. If PyG's download fails, extract rmd17.tar.bz2 so that
    # <data-root>/raw/rmd17/npz_data/rmd17_<name>.npz exists.
    "revised-aspirin":       "revised aspirin",
    "revised-benzene":       "revised benzene",
    "revised-ethanol":       "revised ethanol",
    "revised-malonaldehyde": "revised malonaldehyde",
    "revised-naphthalene":   "revised naphthalene",
    "revised-salicylic":     "revised salicylic acid",
    "revised-toluene":       "revised toluene",
    "revised-uracil":        "revised uracil",
    "revised-paracetamol":   "revised paracetamol",
    "revised-azobenzene":    "revised azobenzene",
}

KCAL_TO_MEV = 43.3634


def load_md17(root, molecule_name, train_size=950, val_size=50):
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
    forces = torch.cat([train_ds[i].force for i in range(len(train_ds))])
    force_rms = float(forces.pow(2).mean().sqrt())
    return train_ds, val_ds, test_ds, energy_mean, energy_std, force_rms


def run_epoch(model, loader, optimizer, device, lambda_f, energy_mean, scale=1.0,
              train=True, warmup_scheduler=None, loss_type="mae", rot_augment=False, ema=None):
    model.train(train)
    e_sum, f_sum, n = 0.0, 0.0, 0
    t0 = time.time()

    for data in loader:
        data = data.to(device)
        if train and rot_augment:
            # PONITA's orientation grid makes it only approximately equivariant;
            # PONITA's own training rotates every batch at random.
            R = _random_rotation(data.pos.dtype, device)
            data.pos = data.pos @ R.T
            data.force = data.force @ R.T
        data.pos = data.pos.requires_grad_(True)

        e_pred, f_pred = model.forward_with_forces(data.z, data.pos, data.batch)

        # Targets in meV, divided by `scale` (energy std or force RMS) so the loss is O(1).
        # Forces share the energy scale because F = -dE/dpos, so dividing both by
        # the same factor keeps the autograd relationship exact.
        e_true = (data.energy.squeeze(-1) - energy_mean) * KCAL_TO_MEV / scale
        f_true = data.force * KCAL_TO_MEV / scale

        e_mae = F.l1_loss(e_pred, e_true)
        f_mae = F.l1_loss(f_pred, f_true)
        if loss_type == "mse":   # PONITA's objective; MAE is still what gets reported
            loss = F.mse_loss(e_pred, e_true) + lambda_f * F.mse_loss(f_pred, f_true)
        else:
            loss = e_mae + lambda_f * f_mae

        if train:
            if not torch.isfinite(loss):
                # Skip a poisoned batch (e.g. a degenerate-geometry NaN) instead
                # of letting one bad step wipe out the run.
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            if warmup_scheduler is not None:
                warmup_scheduler.step()

        # Accumulate MAE back in physical meV units regardless of scaling.
        e_sum += e_mae.item() * scale
        f_sum += f_mae.item() * scale
        n += 1

    return e_sum / max(n, 1), f_sum / max(n, 1), time.time() - t0


def run_experiment(label, model, train_loader, val_loader, device, args, energy_mean, scale, ckpt_path):
    print(f"\n=== {label} ===")

    # Lazy init for PONITA (needed before parameter count is meaningful)
    with torch.no_grad():
        dummy = next(iter(train_loader)).to(device)
        model.forward(dummy.z, dummy.pos, dummy.batch)

    from torch.nn.parameter import UninitializedParameter
    _init_params = [p for p in model.parameters() if p.requires_grad and not isinstance(p, UninitializedParameter)]
    print(f"  Parameters: {sum(p.numel() for p in _init_params):,}")

    optimizer = torch.optim.Adam(
        _init_params,
        lr=args.lr, weight_decay=args.weight_decay,
    )
    ema = _EMA(model, args.ema) if args.ema else None
    config = _run_config(args)

    # Linear warmup for first --warmup epochs
    warmup_steps = args.warmup * len(train_loader)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    ) if args.warmup > 0 else None

    # Cosine decay after warmup
    main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup, eta_min=1e-6
    )

    # Two "best" checkpoints: lowest val force MAE (ckpt_path) and lowest val
    # energy MAE (*_bestE.pt). Each tracks the other metric at that point too.
    best_val_f = float("inf")      # best val F ...
    best_val_e = float("inf")      # ... and val E at that checkpoint
    best_e_val_e = float("inf")    # best val E ...
    best_e_val_f = float("inf")    # ... and val F at that checkpoint
    history = _empty_history()
    start_epoch = 0

    stem = ckpt_path[:-3] if ckpt_path.endswith(".pt") else ckpt_path
    best_e_path = stem + "_bestE.pt"
    hist_path = stem + "_history.json"

    # Resume: a "<ckpt>_last.pt" holds full training state so a killed run
    # (e.g. a Colab disconnect or a SLURM time limit) continues where it stopped.
    last_path = stem + "_last.pt"
    if os.path.exists(last_path):
        try:
            state = torch.load(last_path, map_location=device, weights_only=False)
            # The cosine schedule depends on --epochs, so a checkpoint from a run
            # with a different epoch budget can't be continued meaningfully.
            if state.get("epochs") != args.epochs:
                raise ValueError(
                    f"checkpoint was for --epochs {state.get('epochs')}, now {args.epochs}"
                )
            if state.get("loss", "mae") != args.loss:
                raise ValueError(
                    f"checkpoint was trained with --loss {state.get('loss', 'mae')}, now {args.loss}"
                )
            saved = {**_DEFAULT_CONFIG, **state.get("config", {})}
            if saved != config:
                raise ValueError(f"checkpoint was trained with {saved}, now {config}")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if ema is not None and state.get("ema") is not None:
                ema.shadow = state["ema"]
            if warmup_sched is not None and state.get("warmup_sched") is not None:
                warmup_sched.load_state_dict(state["warmup_sched"])
            main_sched.load_state_dict(state["main_sched"])
            best_val_f = state["best_val_f"]
            best_val_e = state["best_val_e"]
            # Older checkpoints predate the energy checkpoint and the history.
            best_e_val_e = state.get("best_e_val_e", float("inf"))
            best_e_val_f = state.get("best_e_val_f", float("inf"))
            history = state.get("history", _empty_history())
            start_epoch = state["epoch"] + 1
            print(f"  Resumed from {last_path} at epoch {start_epoch} (best val F {best_val_f:.3f})")
        except Exception as e:
            print(f"  Could not resume from {last_path} ({e}); starting fresh.")

    def _save_last(epoch):
        _atomic_save({
            "epoch": epoch,
            "epochs": args.epochs,
            "loss": args.loss,
            "config": config,
            "model": model.state_dict(),
            "ema": ema.shadow if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "warmup_sched": warmup_sched.state_dict() if warmup_sched is not None else None,
            "main_sched": main_sched.state_dict(),
            "best_val_f": best_val_f,
            "best_val_e": best_val_e,
            "best_e_val_e": best_e_val_e,
            "best_e_val_f": best_e_val_f,
            "history": history,
        }, last_path)

    for epoch in range(start_epoch, args.epochs):
        active_warmup = warmup_sched if epoch < args.warmup else None
        lr = optimizer.param_groups[0]["lr"]
        tr_e, tr_f, tr_t = run_epoch(
            model, train_loader, optimizer, device, args.lambda_f, energy_mean, scale,
            train=True, warmup_scheduler=active_warmup, loss_type=args.loss,
            rot_augment=args.rot_augment, ema=ema,
        )
        if epoch >= args.warmup:
            main_sched.step()
        history["epoch"].append(epoch)
        history["train_e"].append(tr_e)
        history["train_f"].append(tr_f)
        history["lr"].append(lr)

        if epoch % args.log_interval == 0 or epoch == args.epochs - 1:
            # With EMA, validate and save the averaged weights, as PONITA does.
            with (ema.applied(model) if ema is not None else contextlib.nullcontext()):
                val_e, val_f, val_t = run_epoch(
                    model, val_loader, None, device, args.lambda_f, energy_mean, scale, train=False
                )
                marks = ""
                if val_f < best_val_f:
                    best_val_f, best_val_e = val_f, val_e
                    _atomic_save(model.state_dict(), ckpt_path)
                    marks += " *F"
                if val_e < best_e_val_e:
                    best_e_val_e, best_e_val_f = val_e, val_f
                    _atomic_save(model.state_dict(), best_e_path)
                    marks += " *E"
            history["val_epoch"].append(epoch)
            history["val_e"].append(val_e)
            history["val_f"].append(val_f)
            print(
                f"  epoch {epoch:05d} | "
                f"train E {tr_e:.3f} F {tr_f:.3f} ({tr_t:.1f}s) | "
                f"val E {val_e:.3f} F {val_f:.3f} ({val_t:.1f}s)" + marks
            )
            _save_last(epoch)
            _atomic_json(history, hist_path)
            if args.deadline is not None and time.time() > args.deadline and epoch < args.epochs - 1:
                print(f"  --max-hours reached; stopped after epoch {epoch}. "
                      f"Run the same command again to resume.")
                return {"stopped": True}

    _atomic_json(history, hist_path)
    return {
        "best_f": (best_val_e, best_val_f), "best_f_path": ckpt_path,
        "best_e": (best_e_val_e, best_e_val_f), "best_e_path": best_e_path,
    }


def _random_rotation(dtype, device):
    """Uniformly random proper rotation (det = +1)."""
    q, r = torch.linalg.qr(torch.randn(3, 3, dtype=dtype, device=device))
    q = q * torch.sign(torch.diagonal(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class _EMA:
    """Exponential moving average of the trainable weights, updated every step."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].lerp_(p.detach(), 1.0 - self.decay)

    @contextlib.contextmanager
    def applied(self, model):
        """Temporarily swap the averaged weights into the model."""
        backup = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    backup[n] = p.detach().clone()
                    p.copy_(self.shadow[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])


# Training settings that change the model or its training; runs with different
# values get different checkpoint files and never resume each other.
_DEFAULT_CONFIG = {"readout": "mlp", "rot_augment": False, "ema": 0.0,
                   "self_loops": False, "target_scale": "energy_std",
                   "ect_out_dim": 64, "ect_dropout": 0.0,
                   "cond_drop": 0.0, "cond_noise": 0.0, "ect_encoder": "branch",
                   "ponita_radius": 1000.0}


def _run_config(args):
    return {"readout": args.readout, "rot_augment": bool(args.rot_augment),
            "ema": float(args.ema or 0.0), "self_loops": bool(args.self_loops),
            "target_scale": args.target_scale,
            "ect_out_dim": int(args.ect_out_dim), "ect_dropout": float(args.ect_dropout),
            "cond_drop": float(args.cond_drop), "cond_noise": float(args.cond_noise),
            "ect_encoder": args.ect_encoder, "ponita_radius": float(args.ponita_radius)}


def _config_tag(args):
    """Filename suffix; empty for the original settings so earlier runs resume."""
    c = _run_config(args)
    tags = ["mse"] if args.loss == "mse" else []
    if c["readout"] == "sum":
        tags.append("sum")
    if c["rot_augment"]:
        tags.append("aug")
    if c["ema"]:
        tags.append(f"ema{c['ema']:g}".replace(".", ""))
    if c["self_loops"]:
        tags.append("loop")
    if c["target_scale"] == "force_rms":
        tags.append("frms")
    if c["ect_out_dim"] != 64:
        tags.append(f"d{c['ect_out_dim']}")
    if c["ect_dropout"]:
        tags.append(f"drop{c['ect_dropout']:g}".replace(".", ""))
    if c["cond_drop"]:
        tags.append(f"cdrop{c['cond_drop']:g}".replace(".", ""))
    if c["cond_noise"]:
        tags.append(f"cnoise{c['cond_noise']:g}".replace(".", ""))
    if c["ect_encoder"] != "branch":
        tags.append(c["ect_encoder"])
    if c["ponita_radius"] != 1000.0:
        tags.append(f"pr{c['ponita_radius']:g}".replace(".", "p"))
    return ("_" + "_".join(tags)) if tags else ""


def _atomic_save(obj, path):
    # Temp file + rename, so a crash mid-save can't corrupt the file.
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _empty_history():
    return {"epoch": [], "train_e": [], "train_f": [], "lr": [],
            "val_epoch": [], "val_e": [], "val_f": []}


def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    args.deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    _set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mol_name = MOLECULE_MAP.get(args.molecule, args.molecule)

    print(f"Molecule : {mol_name}")
    print(f"Seed     : {args.seed}")
    print(f"Device   : {device}")
    print(f"lambda_f : {args.lambda_f}")

    set_graph_self_loops(args.self_loops)
    train_ds, val_ds, test_ds, energy_mean, energy_std, force_rms = load_md17(
        args.data_root, mol_name, args.train_size, args.val_size
    )
    base = force_rms if args.target_scale == "force_rms" else energy_std
    scale = (base * KCAL_TO_MEV) if args.normalize else 1.0
    print(f"Energy mean (train): {energy_mean:.2f} kcal/mol")
    print(f"Normalize targets  : {args.normalize} ({args.target_scale}, scale = {scale:.2f} meV)")
    print(f"Settings           : {_run_config(args)}, loss {args.loss}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    _, ect_edges = build_icosphere_directions(nu=args.icosphere_nu)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    results = {}

    def _ckpt(label):
        safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p")
        mol = mol_name.replace(" ", "_")
        # Runs with different loss / settings get their own files and never resume each other.
        return os.path.join(args.ckpt_dir, f"ponita_{safe}_{mol}_seed{args.seed}{_config_tag(args)}.pt")

    def _finish(label, model, info):
        """Test both best checkpoints; save val/test MAEs to *_results.json."""
        out = {}
        for which in ("best_f", "best_e"):
            path = info[which + "_path"]
            if not os.path.exists(path):
                continue
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            te, tf, _ = run_epoch(model, test_loader, None, device, args.lambda_f,
                                  energy_mean, scale, train=False)
            ve, vf = info[which]
            out[which] = {"val_e": ve, "val_f": vf, "test_e": te, "test_f": tf}
            print(f"  -> test ({which}) E {te:.3f} F {tf:.3f}")
            regs = [m for m in model.modules()
                    if isinstance(m, _TokenRegularizer) and m.null is not None]
            if regs:
                # Same weights, learned null tokens for every molecule: how much
                # of the result comes from the ECT condition.
                for m in regs:
                    m.use_null = True
                try:
                    te0, tf0, _ = run_epoch(model, test_loader, None, device, args.lambda_f,
                                            energy_mean, scale, train=False)
                finally:
                    for m in regs:
                        m.use_null = False
                out[which].update(test_e_null=te0, test_f_null=tf0)
                print(f"  -> test ({which}, null ECT token) E {te0:.3f} F {tf0:.3f}")
        results[label] = out
        _atomic_json(
            {"label": label, "molecule": mol_name, "seed": args.seed,
             "epochs": args.epochs, "loss": args.loss, "config": _run_config(args), **out},
            info["best_f_path"][:-3] + "_results.json",
        )

    backbone_kw = dict(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_ori=args.num_ori,
        basis_dim=args.basis_dim,
        radius=args.ponita_radius,
    )

    if args.model in ("baseline", "all"):
        model = PONITABaseline_MD17(**backbone_kw, readout=args.readout).to(device)
        info = run_experiment(
            "PONITA baseline", model, train_loader, val_loader,
            device, args, energy_mean, scale, _ckpt("PONITA_baseline"),
        )
        if info.get("stopped"):
            return
        _finish("PONITA baseline", model, info)

    cond_types = (
        _DEFAULT_COND_TYPES        if args.model in ("all", "ect-all") else
        [args.cond_type]           if args.model == "ect"               else []
    )
    for cond_type in cond_types:
        label = f"PONITA + ECT ({cond_type})"
        cls = _COND_CLASSES[cond_type]
        extra_kw = {}
        if cond_type in _SUM_READOUT_TYPES:
            extra_kw["readout"] = args.readout
        elif args.readout != "mlp":
            raise SystemExit(f"--readout {args.readout} is not implemented for {cond_type}; "
                             f"supported: {sorted(_SUM_READOUT_TYPES)}")
        if cond_type == "concat":
            extra_kw["ect_dropout"] = args.ect_dropout
        elif args.ect_dropout:
            raise SystemExit("--ect-dropout is only implemented for --cond-type concat")
        if cond_type in _COND_REG_TYPES:
            extra_kw["cond_drop"] = args.cond_drop
            extra_kw["cond_noise"] = args.cond_noise
        elif args.cond_drop or args.cond_noise:
            raise SystemExit(f"--cond-drop / --cond-noise are not implemented for {cond_type}; "
                             f"supported: {sorted(_COND_REG_TYPES)}")
        if cond_type in _ENCODER_TYPES:
            extra_kw["ect_encoder"] = args.ect_encoder
        elif args.ect_encoder != "branch":
            raise SystemExit(f"--ect-encoder {args.ect_encoder} is not implemented for {cond_type}; "
                             f"supported: {sorted(_ENCODER_TYPES)}")
        if cond_type in ("leap", "concat-local", "film-local", "concat-local-branch",
                         "hybrid-leap"):
            extra_kw["local_cutoff"] = args.local_cutoff
            if args.local_cutoff != 5.0:
                label += f" rc{args.local_cutoff:g}"   # separate checkpoint per cutoff
        model = cls(
            ect_edge_indices=ect_edges,
            ect_out_dim=args.ect_out_dim,
            icosphere_nu=args.icosphere_nu,
            **backbone_kw,
            **extra_kw,
        ).to(device)
        info = run_experiment(
            label, model, train_loader, val_loader,
            device, args, energy_mean, scale, _ckpt(label),
        )
        if info.get("stopped"):
            return
        _finish(label, model, info)

    if results:
        print(f"\n=== Summary — {mol_name} seed {args.seed} | best-F and best-E checkpoints ===")
        print(f"  {'Model':<35} {'Ckpt':<7} {'Val E':>9} {'Val F':>9} {'Test E':>9} {'Test F':>9}")
        print(f"  {'-' * 82}")
        for label, out in results.items():
            for which, r in out.items():
                # also print in kcal/mol for comparison with PONITA paper table
                print(
                    f"  {label:<35} {which:<7} {r['val_e']:>9.3f} {r['val_f']:>9.3f} "
                    f"{r['test_e']:>9.3f} {r['test_f']:>9.3f}"
                    f"  ({r['test_e']/KCAL_TO_MEV:.3f} / {r['test_f']/KCAL_TO_MEV:.3f} kcal/mol)"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule",       default="aspirin",
                        choices=list(MOLECULE_MAP.keys()))
    parser.add_argument("--data-root",      default="./data/md17")
    parser.add_argument("--train-size",     type=int,   default=950)
    parser.add_argument("--val-size",       type=int,   default=50)
    # PONITA paper hyperparameters
    parser.add_argument("--epochs",         type=int,   default=5000)
    parser.add_argument("--warmup",         type=int,   default=100)
    parser.add_argument("--log-interval",   type=int,   default=50)
    parser.add_argument("--batch-size",     type=int,   default=5)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-16)
    parser.add_argument("--lambda-f",       type=float, default=500.0)
    parser.add_argument("--readout",        default="mlp", choices=["mlp", "sum"],
                        help="mlp: pooled molecule embedding -> MLP. sum: per-atom energies "
                             "summed over atoms, as in PONITA's MD17 code.")
    parser.add_argument("--rot-augment",    action="store_true",
                        help="Randomly rotate positions and forces of every training batch.")
    parser.add_argument("--ema",            type=float, default=0.0,
                        help="Decay of an exponential moving average of the weights "
                             "(PONITA uses 0.99). 0 = off. Validation/test use the average.")
    parser.add_argument("--self-loops",     action="store_true",
                        help="Include self-loops in the PONITA graph, as PONITA's MD17 code does.")
    parser.add_argument("--target-scale",   default="energy_std", choices=["energy_std", "force_rms"],
                        help="Divide energy and force targets by the train energy std, or by "
                             "the train force RMS (PONITA).")
    parser.add_argument("--official",       action="store_true",
                        help="PONITA's own MD17 training setup: --loss mse --readout sum "
                             "--rot-augment --ema 0.99 --self-loops --target-scale force_rms.")
    parser.add_argument("--max-hours",      type=float, default=None,
                        help="Stop cleanly (checkpoint saved) after this many hours, e.g. to "
                             "fit a Kaggle session. Re-running the same command resumes.")
    parser.add_argument("--loss",           default="mae", choices=["mae", "mse"],
                        help="Training objective. mse = squared error, as in the PONITA "
                             "paper. Reported metrics are MAE either way.")
    parser.add_argument("--hidden-dim",     type=int,   default=128)
    parser.add_argument("--num-layers",     type=int,   default=5)
    parser.add_argument("--num-ori",        type=int,   default=20)
    parser.add_argument("--basis-dim",      type=int,   default=256)
    parser.add_argument("--ponita-radius",  type=float, default=1000.0,
                        help="Cutoff (Å) of PONITA's smooth polynomial window on every "
                             "message; atoms this far apart do not interact. "
                             "1000 = all atoms (aspirin is ~9 Å across).")
    # ECT hyperparameters
    parser.add_argument("--ect-out-dim",    type=int,   default=64)
    parser.add_argument("--icosphere-nu",   type=int,   default=3)
    parser.add_argument("--ect-dropout",    type=float, default=0.0,
                        help="Dropout on the global ECT token (concat only), against memorisation.")
    parser.add_argument("--cond-drop",      type=float, default=0.0,
                        help="Condition dropout: per molecule, replace the whole ECT token with "
                             "a learned null token with this probability.")
    parser.add_argument("--cond-noise",     type=float, default=0.0,
                        help="Noise on the ECT token: Gaussian, level drawn per molecule from "
                             "U(0, this), relative to each feature's batch std.")
    parser.add_argument("--ect-encoder",    default="branch", choices=["branch", "mlp"],
                        help="Encoder of the global ECT token. branch: ECTBranch (invariant "
                             "encoder, mean over directions). mlp: flatten -> 2-layer MLP "
                             "(invariant ECT + MLP).")
    parser.add_argument("--local-cutoff",   type=float, default=5.0,
                        help="Neighbor radius (Å) for the leap local ECT. "
                             "1000 = all atoms, matching the PONITA backbone graph.")
    parser.add_argument("--model",          default="all",
                        choices=["all", "baseline", "ect", "ect-all"])
    parser.add_argument("--cond-type",      default="film-final",
                        choices=list(_COND_CLASSES.keys()))
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--ckpt-dir",       default="./checkpoints")
    parser.add_argument("--normalize",      action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Divide energy/force targets by the train energy std "
                             "for a stable, well-scaled loss (on by default).")
    args = parser.parse_args()
    if args.official:
        args.loss, args.readout, args.rot_augment = "mse", "sum", True
        args.ema, args.self_loops, args.target_scale = 0.99, True, "force_rms"
    main(args)
