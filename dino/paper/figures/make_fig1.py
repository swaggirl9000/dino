"""Figure 1: pipeline of ECT conditioning for PONITA.

Top row (ECT branch) is drawn from a real rMD17 aspirin geometry with the same
canonical frame and soft ECT settings as training. Bottom row is a schematic of
PONITA with the four injection sites marked.

    python paper/figures/make_fig1.py [path/to/rmd17_aspirin.npz]

Writes fig1_pipeline.pdf and fig1_pipeline.png next to this script.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_DATA = Path.home() / "Desktop/rmd17/npz_data/rmd17_aspirin.npz"

# Same settings as training (models/ponita_ect.py, qm9_ect.py).
LAM, RES, H_MAX, R_CUT, FRAME = 64.0, 64, 8.0, 5.0, 0

W, H = 5.5, 3.25                     # NeurIPS text width, inches
INK, MUTED, RULE = "#1f2328", "#6b6f76", "#c9ccd1"
FILL, TRAIN_FILL = "#f3f4f6", "#fff7e0"
SITE = {1: "#0072B2", 2: "#009E73", 3: "#E69F00", 4: "#CC79A7"}   # Okabe-Ito
ELEM = {1: ("#ffffff", 9), 6: ("#55585e", 20), 8: ("#e0453a", 22)}

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42,
})


def load_icosphere():
    # Load the module file directly: mesh_lib/__init__ imports torch.
    spec = importlib.util.spec_from_file_location("ico", REPO / "mesh_lib/icosphere.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    v, _ = mod.icosphere(3)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def canonical(x):
    """NumPy copy of qm9_ect.pca_canonical for one molecule."""
    x = x - x.mean(0)
    c = x.T @ x / len(x) + 1e-6 * np.eye(3)
    _, vec = np.linalg.eigh(c)
    y = x @ vec[:, ::-1]
    return y * np.where((y ** 3).sum(0) >= 0, 1.0, -1.0)


def soft_ect(p, dirs, h):
    t = p @ dirs.T
    z = np.clip(LAM * (h[None, None, :] - t[:, :, None]), -60, 60)
    return (1 / (1 + np.exp(-z))).sum(0)            # [dirs, res]


def bonds(x, z):
    cut = {frozenset((6, 6)): 1.7, frozenset((6, 8)): 1.6, frozenset((8, 8)): 1.6}
    out = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            pair = frozenset((z[i], z[j]))
            lim = 1.2 if 1 in pair else cut.get(pair, 0)
            if np.linalg.norm(x[i] - x[j]) < lim:
                out.append((i, j))
    return out


def draw_mol(ax, x, z, keep=None, focus=None):
    """Orthographic view down the third axis; atoms drawn back to front."""
    keep = np.ones(len(x), bool) if keep is None else keep
    for i, j in bonds(x, z):
        a = 1.0 if keep[i] and keep[j] else 0.2
        ax.plot(x[[i, j], 0], x[[i, j], 1], color=INK, lw=0.6, alpha=a, zorder=1,
                solid_capstyle="round")
    for k in np.argsort(x[:, 2]):
        fc, s = ELEM[z[k]]
        ax.scatter(x[k, 0], x[k, 1], s=s, c=fc, edgecolors=INK, linewidths=0.4,
                   alpha=1.0 if keep[k] else 0.2, zorder=2 + x[k, 2] * 0.01)
    if focus is not None:
        ax.add_patch(Circle(x[focus, :2], R_CUT, fill=False, ls=(0, (2, 1.5)),
                            lw=0.6, ec=SITE[1], zorder=0))
        ax.scatter(*x[focus, :2], s=34, facecolors="none", edgecolors=SITE[1],
                   linewidths=0.9, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")


def inch_axes(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def box(ax, x, y, w, h, text, fc=FILL, ec=INK, ls="-", fs=7, color=INK, lw=0.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=color, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, color=INK, ls="-", lw=0.7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, ls=ls,
                                mutation_scale=6, shrinkA=0, shrinkB=0))


def site(ax, x, y, n, r=0.062):
    ax.add_patch(Circle((x, y), r, fc=SITE[n], ec="white", lw=0.5, zorder=10))
    ax.text(x, y - 0.004, str(n), ha="center", va="center", fontsize=5.6,
            color="white", fontweight="bold", zorder=11)


def main(data_path):
    d = np.load(data_path)
    z = d["nuclear_charges"]
    raw = d["coords"][FRAME].astype(np.float64)
    can = canonical(raw)
    dirs = load_icosphere()
    hs = np.linspace(-H_MAX, H_MAX, RES)

    # Arbitrary input orientation, so panels (a) and (b) visibly differ.
    a, b = 0.9, 0.6
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ \
          np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    shown_raw = (raw - raw.mean(0)) @ rot.T

    # Local panel: the heavy atom whose 5 Å ball leaves out the most atoms.
    dist = np.linalg.norm(can[:, None] - can[None], axis=-1)
    heavy = np.where(z > 1)[0]
    focus = heavy[np.argmin((dist[heavy] <= R_CUT).sum(1))]
    keep = dist[focus] <= R_CUT

    # Directions ordered by latitude, then longitude, so the grid reads smoothly.
    order = np.lexsort((np.arctan2(dirs[:, 1], dirs[:, 0]), -dirs[:, 2]))
    ect = soft_ect(can, dirs[order], hs)
    row = int(np.argmax(np.abs(dirs[order] @ np.array([1.0, 0, 0]))))

    fig = plt.figure(figsize=(W, H))
    ov = fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, W)
    ov.set_ylim(0, H)
    ov.axis("off")

    top_y, top_h, title_y = 1.78, 1.05, 3.12
    ov.text(0.02, title_y + 0.0, "ECT branch", fontsize=7.5, fontweight="bold",
            color=INK, va="center")
    ov.text(0.02, 1.47 - 0.02, "PONITA", fontsize=7.5, fontweight="bold",
            color=INK, va="center")

    # (a) molecule
    draw_mol(inch_axes(fig, 0.02, top_y, 0.78, top_h), shown_raw, z)
    ov.text(0.41, title_y - 0.17, "(a) molecule", ha="center", fontsize=7)
    arrow(ov, 0.80, top_y + top_h / 2, 0.93, top_y + top_h / 2)

    # (b) canonical frame
    draw_mol(inch_axes(fig, 0.93, top_y + 0.08, 0.80, top_h - 0.08), can, z)
    ox, oy = 0.98, top_y + 0.02                  # axis triad in the panel corner
    arrow(ov, ox, oy, ox + 0.24, oy, color=SITE[1])
    arrow(ov, ox, oy, ox, oy + 0.24, color=SITE[1])
    ov.text(ox + 0.27, oy, r"$\tilde x_1$", color=SITE[1], fontsize=6.5, va="center")
    ov.text(ox, oy + 0.30, r"$\tilde x_2$", color=SITE[1], fontsize=6.5, ha="center")
    ov.text(1.33, title_y - 0.17, "(b) PCA frame", ha="center", fontsize=7)
    arrow(ov, 1.73, top_y + top_h / 2, 1.86, top_y + top_h / 2)

    # (c) point set: whole molecule, or one atom's neighbourhood
    draw_mol(inch_axes(fig, 1.86, top_y + 0.53, 0.62, 0.50), can, z)
    draw_mol(inch_axes(fig, 1.86, top_y - 0.02, 0.62, 0.53), can, z, keep=keep, focus=focus)
    ov.text(2.17, title_y - 0.17, "(c) point set", ha="center", fontsize=7)
    ov.text(2.52, top_y + 0.78, "global", fontsize=6, color=MUTED, va="center")
    ov.text(2.52, top_y + 0.24, "local\n" + r"$r_c=5$ Å", fontsize=6, color=MUTED,
            va="center", linespacing=1.1)
    arrow(ov, 2.80, top_y + top_h / 2, 2.93, top_y + top_h / 2)

    # (d) soft ECT: one Euler characteristic curve, then the full grid
    axc = inch_axes(fig, 3.12, top_y + 0.70, 0.68, 0.30)
    exact = (can @ dirs[order][row] <= hs[:, None]).sum(1)
    axc.step(hs, exact, where="post", color=MUTED, lw=0.6)
    axc.plot(hs, ect[row], color=SITE[1], lw=0.8)
    axc.set_xlim(-H_MAX, H_MAX)
    axc.set_ylim(-1, len(z) + 2)
    axc.set_xticks([])
    axc.set_yticks([0, len(z)])
    axc.tick_params(length=1.5, pad=1, labelsize=5.5)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    axc.set_ylabel(r"$\chi$", fontsize=6, labelpad=0, rotation=0, va="center")

    axh = inch_axes(fig, 3.12, top_y + 0.06, 0.68, 0.58)
    axh.imshow(ect, aspect="auto", cmap="Blues", origin="upper",
               extent=(-H_MAX, H_MAX, len(order), 0), interpolation="nearest")
    axh.axhline(row + 0.5, color=SITE[1], lw=0.6)
    axh.set_xticks([-8, 0, 8])
    axh.set_yticks([])
    axh.tick_params(length=1.5, pad=1, labelsize=5.5)
    axh.set_xlabel(r"height $h$ (Å)", fontsize=6, labelpad=0.5)
    axh.set_ylabel(r"92 dir. $\xi$", fontsize=6, labelpad=1)
    ov.text(3.40, title_y - 0.17, "(d) soft ECT", ha="center", fontsize=7)
    ov.text(3.40, title_y - 0.31, r"92 $\times$ 64 grid", ha="center", fontsize=6, color=MUTED)
    arrow(ov, 3.84, top_y + top_h / 2, 3.97, top_y + top_h / 2)

    # (e) encoder, (f) training-only regulariser, then the token
    box(ov, 3.97, top_y + 0.30, 0.62, 0.46, "ECTBranch\nor MLP")
    ov.text(4.28, title_y - 0.17, "(e) encoder", ha="center", fontsize=7)
    arrow(ov, 4.59, top_y + top_h / 2, 4.71, top_y + top_h / 2)
    box(ov, 4.71, top_y + 0.24, 0.74, 0.58,
        "condition dropout\nor token noise\n" + r"$\it{training\ only}$",
        fc=TRAIN_FILL, ls=(0, (3, 1.5)), fs=6.3)
    ov.text(5.08, title_y - 0.17, "(f) regularizer", ha="center", fontsize=7)

    # Token and the bus that carries it to the injection sites
    tok_x, tok_y, bus_y = 5.08, 1.60, 1.47
    arrow(ov, tok_x, top_y + 0.24, tok_x, tok_y + 0.09)
    for k in range(8):
        ov.add_patch(plt.Rectangle((tok_x - 0.2 + k * 0.05, tok_y - 0.045), 0.045, 0.09,
                                   fc=plt.cm.Blues(0.25 + 0.09 * ((k * 5) % 8)), ec=INK, lw=0.3))
    ov.text(tok_x - 0.25, tok_y, r"token $e$ or $e^\ell_i$", fontsize=6.5, va="center", ha="right")
    ov.plot([tok_x, tok_x], [tok_y - 0.045, bus_y], color=INK, lw=0.8)
    ov.plot([0.72, tok_x], [bus_y, bus_y], color=INK, lw=0.8)

    # Bottom row: PONITA
    y0, hh = 0.48, 0.52
    yc = y0 + hh / 2
    box(ov, 0.05, y0, 0.50, hh, "one-hot\n" + r"$z_i$")
    arrow(ov, 0.55, yc, 0.65, yc)
    ov.add_patch(Circle((0.72, yc), 0.07, fc="white", ec=INK, lw=0.6))
    ov.text(0.72, yc - 0.005, r"$\Vert$", ha="center", va="center", fontsize=7)
    arrow(ov, 0.72, bus_y, 0.72, yc + 0.07, color=SITE[1])
    site(ov, 0.72, bus_y - 0.2, 1)
    arrow(ov, 0.79, yc, 0.90, yc)
    box(ov, 0.90, y0, 0.52, hh, "lift to\n" + r"$\mathbb{R}^3\times S^2$" + "\n20 orient.", fs=6.5)
    arrow(ov, 1.42, yc, 1.55, yc)

    # ConvNeXt block x5, with AdaLN modulation points
    ov.add_patch(FancyBboxPatch((1.55, y0 - 0.12), 1.62, hh + 0.24,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                fc="none", ec=RULE, lw=0.8))
    ov.text(3.13, y0 - 0.08, r"$\times 5$", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    ov.text(1.59, y0 - 0.08, "ConvNeXt block", ha="left", va="bottom", fontsize=6, color=MUTED)
    bx = [(1.66, "conv"), (2.10, "LN"), (2.50, "MLP")]
    for x, lab in bx:
        box(ov, x, y0 + 0.13, 0.32, 0.28, lab, fs=6.5)
    arrow(ov, 1.98, yc + 0.01, 2.10, yc + 0.01)
    arrow(ov, 2.42, yc + 0.01, 2.50, yc + 0.01)
    ov.add_patch(Circle((2.97, yc + 0.01), 0.055, fc="white", ec=INK, lw=0.6))
    ov.text(2.97, yc + 0.006, "+", ha="center", va="center", fontsize=7)
    arrow(ov, 2.82, yc + 0.01, 2.915, yc + 0.01)
    ov.plot([1.60, 1.60, 2.97, 2.97], [yc + 0.01, y0 + 0.07, y0 + 0.07, yc - 0.045],
            color=MUTED, lw=0.5)
    for x in (1.66 + 0.16, 2.10 + 0.16):          # modulation before conv and after LN
        arrow(ov, x, bus_y, x, y0 + 0.41, color=SITE[2])
    site(ov, 2.04, bus_y - 0.2, 2)
    ov.plot([1.82, 2.26], [bus_y - 0.2, bus_y - 0.2], color=SITE[2], lw=0.8, zorder=9)
    arrow(ov, 3.17, yc, 3.30, yc)

    # Readout: per-atom embedding, FiLM, concat, linear, sum
    ov.text(3.235, yc + 0.06, r"$h_i$", ha="center", va="bottom", fontsize=7)
    box(ov, 3.30, y0 + 0.08, 0.34, 0.36, "FiLM", fs=6.5)
    arrow(ov, 3.47, bus_y, 3.47, y0 + 0.44, color=SITE[3])
    site(ov, 3.47, bus_y - 0.2, 3)
    arrow(ov, 3.64, yc, 3.75, yc)
    ov.add_patch(Circle((3.82, yc), 0.07, fc="white", ec=INK, lw=0.6))
    ov.text(3.82, yc - 0.005, r"$\Vert$", ha="center", va="center", fontsize=7)
    arrow(ov, 3.82, bus_y, 3.82, yc + 0.07, color=SITE[4])
    site(ov, 3.82, bus_y - 0.2, 4)
    arrow(ov, 3.89, yc, 3.99, yc)
    box(ov, 3.99, y0 + 0.08, 0.40, 0.36, "linear\n" + r"$\to E_i$", fs=6.5)
    arrow(ov, 4.39, yc, 4.49, yc)
    ov.text(4.58, yc, r"$\sum_i$", ha="center", va="center", fontsize=8)
    arrow(ov, 4.68, yc, 4.78, yc)
    ov.text(4.84, yc, r"$E$", ha="center", va="center", fontsize=8)
    box(ov, 4.95, y0 + 0.04, 0.52, 0.44, r"$F=-\nabla_x E$" + "\nautograd", fc="white", fs=6.5)
    arrow(ov, 4.90, yc, 4.95, yc)

    # Legend
    items = [(1, "input concat"), (2, "adaptive LayerNorm (every block)"),
             (3, "FiLM on embedding"), (4, "readout concat")]
    for (n, lab), lx in zip(items, (0.10, 1.02, 2.80, 4.03)):
        site(ov, lx, 0.13, n)
        ov.text(lx + 0.10, 0.13, lab, va="center", fontsize=6.5, color=INK)

    out = HERE / "fig1_pipeline"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), "and .png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
