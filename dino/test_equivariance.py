"""Equivariance / invariance sanity checks for the ECT geometric encoders.

Run on the cluster (needs torch, torch_geometric, torch_scatter):

    source venv/bin/activate
    python test_equivariance.py

Checks, in order of importance:
  1. PCA canonical frame is rotation-invariant (same molecule, any orientation
     -> same canonical coordinates). This is what makes the local ECT invariant.
  2. Local ECT descriptor is rotation-invariant.
  3. Forces (F = -dE/dpos) are finite for a near-symmetric geometry — the case
     where the old svd-based PCA produced NaN gradients.
  4. End-to-end: EGNN-leap energy is rotation-invariant and forces are
     rotation-equivariant (F(Rx) = R F(x)).

Exit code is non-zero if any check fails, so it can gate a job script.
"""
import sys
import torch

from models.egnn_ect import _pca_canonical, _LocalECTEncoder, EGNNWithLocalECT
from torch_geometric.nn import radius_graph
from qm9_ect import batched_point_ect, build_icosphere_directions


def _random_rotation(dtype, device):
    a = torch.randn(3, 3, dtype=dtype, device=device)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))          # make it deterministic-ish
    if torch.det(q) < 0:                            # ensure a proper rotation
        q[:, 0] = -q[:, 0]
    return q


def _molecule(n=15, seed=0, dtype=torch.float64, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=dtype, device=device) * 2.0
    z = torch.randint(1, 9, (n,), generator=g, device=device)
    batch = torch.zeros(n, dtype=torch.long, device=device)
    return z, pos, batch


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def main():
    torch.manual_seed(0)
    dtype, device = torch.float64, "cpu"   # float64 for tight tolerances
    results = []

    # ---- 1. PCA canonical frame rotation invariance ----------------------
    z, pos, batch = _molecule(dtype=dtype, device=device)
    R = _random_rotation(dtype, device)
    canon_a = _pca_canonical(pos, batch)
    canon_b = _pca_canonical(pos @ R.T, batch)
    err = (canon_a - canon_b).abs().max().item()
    results.append(check("PCA canonical is rotation-invariant", err < 1e-6, f"max|Δ|={err:.2e}"))

    # ---- 2. Local ECT descriptor rotation invariance ---------------------
    enc = _LocalECTEncoder(ect_out_dim=32, icosphere_nu=2, resolution=32).to(device).double()
    ei_a = radius_graph(pos, r=5.0, batch=batch)
    ei_b = radius_graph(pos @ R.T, r=5.0, batch=batch)
    with torch.no_grad():
        ect_a = enc(pos, ei_a, batch)
        ect_b = enc(pos @ R.T, ei_b, batch)
    err = (ect_a - ect_b).abs().max().item()
    results.append(check("Local ECT is rotation-invariant", err < 1e-5, f"max|Δ|={err:.2e}"))

    # ---- 2b. Global ECT (concat / FiLM variants) rotation invariance ------
    dirs, _ = build_icosphere_directions(nu=2)
    dirs = dirs.to(dtype)
    g_a = batched_point_ect(pos, batch, dirs, resolution=32)
    g_b = batched_point_ect(pos @ R.T, batch, dirs, resolution=32)
    err = (g_a - g_b).abs().max().item()
    results.append(check("Global ECT is rotation-invariant", err < 1e-5, f"max|Δ|={err:.2e}"))

    # ---- 3. Forces finite for a near-symmetric geometry ------------------
    # A symmetric ring gives a covariance with degenerate eigenvalues — the
    # case that made the old svd path emit NaN gradients.
    ang = torch.linspace(0, 2 * torch.pi, 7, dtype=dtype, device=device)[:-1]
    ring = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], dim=1)
    ring = ring + 1e-4 * torch.randn_like(ring)      # tiny break, as in real MD frames
    rb = torch.zeros(ring.shape[0], dtype=torch.long, device=device)
    rp = ring.clone().requires_grad_(True)
    eir = radius_graph(rp, r=5.0, batch=rb)
    out = enc(rp, eir, rb).sum()
    grad = torch.autograd.grad(out, rp)[0]
    results.append(check("Gradients finite on near-symmetric geometry",
                         bool(torch.isfinite(grad).all()),
                         f"max|grad|={grad.abs().max().item():.2e}"))

    # ---- 4. End-to-end EGNN-leap invariance + equivariance ---------------
    model = EGNNWithLocalECT(
        ect_edge_indices=None, hidden_features=32, num_layers=2,
        cutoff=10.0, ect_out_dim=32, icosphere_nu=2, resolution=32,
    ).to(device).double().eval()

    p1 = pos.clone().requires_grad_(True)
    e1, f1 = model.forward_with_forces(z, p1, batch)

    p2 = (pos @ R.T).clone().requires_grad_(True)
    e2, f2 = model.forward_with_forces(z, p2, batch)

    e_err = (e1 - e2).abs().max().item()
    results.append(check("EGNN-leap energy is rotation-invariant", e_err < 1e-5, f"|ΔE|={e_err:.2e}"))

    # Forces must rotate with the molecule: f2 == f1 @ R.T
    f_err = (f2 - f1 @ R.T).abs().max().item()
    results.append(check("EGNN-leap forces are rotation-equivariant", f_err < 1e-4, f"max|Δ|={f_err:.2e}"))

    print()
    if all(results):
        print("All equivariance checks passed.")
        return 0
    print("Some checks FAILED — see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
