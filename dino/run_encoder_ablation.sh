#!/bin/bash
# Final encoder ablation on the winning model: the same model and settings, with the
# global ECT token encoded two ways, then a side-by-side table.
#   branch: ECTBranch, the invariant encoder (current setup)
#   mlp:    flatten -> 2-layer MLP (invariant ECT + MLP, as in the new paper)
# Both use the same PCA-canonical ECT; only the encoder differs.
#
# Usage: bash run_encoder_ablation.sh <winner cond-type> [winner's other train_ponita_md17.py flags]
#   bash run_encoder_ablation.sh hybrid-adaln --cond-drop 0.15
#   CKPT_DIR=/content/drive/MyDrive/dino_ckpts_official DATA_ROOT=/content/drive/MyDrive/dino_data/md17 \
#       bash run_encoder_ablation.sh hybrid-leap --cond-noise 0.5
#
# Set paths, molecule, seed and epochs via the env vars below, not as flags, so the
# results table finds the runs. With the same CKPT_DIR and flags as the winner's run,
# the branch arm resumes that run (only re-tested if finished) and only mlp trains.
# --max-hours, if passed, applies to each of the two runs separately.

set -euo pipefail

WINNER=${1:?usage: bash run_encoder_ablation.sh <winner cond-type> [extra flags]}
shift

MOLECULE=${MOLECULE:-revised-aspirin}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-5000}
DATA_ROOT=${DATA_ROOT:-./data/md17}
CKPT_DIR=${CKPT_DIR:-./checkpoints}

cd "$(dirname "$0")"
START=$(date +%s)

for ENCODER in branch mlp; do
    echo "========================================================"
    echo "  Encoder ablation  |  $WINNER  |  encoder: $ENCODER"
    echo "  molecule: $MOLECULE  |  seed: $SEED  |  extra: $*"
    echo "  Start: $(date)"
    echo "========================================================"
    python -u train_ponita_md17.py \
        --model         ect \
        --cond-type     "$WINNER" \
        --ect-encoder   "$ENCODER" \
        --molecule      "$MOLECULE" \
        --official \
        --epochs        "$EPOCHS" \
        --seed          "$SEED" \
        --log-interval  25 \
        --data-root     "$DATA_ROOT" \
        --ckpt-dir      "$CKPT_DIR" \
        "$@"
done

python - "$CKPT_DIR" "$WINNER" "$START" <<'PYEOF'
import glob, json, os, sys

ckpt_dir, winner, start = sys.argv[1], sys.argv[2], float(sys.argv[3])
runs = {}
for path in glob.glob(os.path.join(ckpt_dir, "*_results.json")):
    if os.path.getmtime(path) < start:
        continue   # only the two runs this script just trained or re-tested
    with open(path) as f:
        r = json.load(f)
    if r["label"].startswith(f"PONITA + ECT ({winner})"):
        runs[r["config"].get("ect_encoder", "branch")] = r

print(f"\n=== Encoder ablation — {winner} | MAE: E meV, F meV/Å ===")
print(f"  {'Encoder':<28} {'Ckpt':<7} {'Val E':>9} {'Val F':>9} {'Test E':>9} {'Test F':>9}")
print(f"  {'-' * 75}")
for enc, name in (("branch", "ECTBranch (current)"), ("mlp", "MLP (invariant ECT + MLP)")):
    r = runs.get(enc)
    if r is None:
        print(f"  {name:<28} not finished (stopped early?); re-run to resume")
        continue
    for which in ("best_f", "best_e"):
        if which in r:
            m = r[which]
            print(f"  {name:<28} {which:<7} {m['val_e']:>9.3f} {m['val_f']:>9.3f} "
                  f"{m['test_e']:>9.3f} {m['test_f']:>9.3f}")
PYEOF
