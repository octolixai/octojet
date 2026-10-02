#!/usr/bin/env bash
# Upload the verified release checkpoint to Hugging Face as a PRIVATE repo (the owner flips it public at launch).
#   1) stage README.md (docs/model-card.md), NOTICE, LICENSE (verbatim from the Vontra source) and assets/*.png into OUT
#   2) create REPO private if it does not exist, then `hf upload-large-folder` (resumable; safe to re-run)
# Needs `hf auth login` on this machine as the octolix account with a write token (never pass tokens as arguments).
# Usage: bash bench/spark/release-upload.sh [--dry-run]
set -euo pipefail
REPO_DIR=${REPO_DIR:-$HOME/octojet-f6}
OUT=${OUT:-$HOME/tensorfold/octojet-release}
MLX=${MLX:-$HOME/tensorfold/flashnext-mlx-4bit-mtp}
HF_REPO=${HF_REPO:-octolix/Qwen3.8-Flash-Next-Octojet-NVFP4}
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
command -v hf >/dev/null || { echo "error: hf CLI missing (pip install -U huggingface_hub)" >&2; exit 2; }
# a user-owned HF_HOME when ~/.cache/huggingface belongs to root (containers wrote it): log in with the same HF_HOME
[ -n "${HF_HOME:-}" ] || { [ -d "$HOME/.hf-octolix" ] && export HF_HOME=$HOME/.hf-octolix; } || true
# whoami prints "user=NAME" (older CLI) or "  user: NAME" with colours (2.x)
who=$(hf auth whoami 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | sed -n 's/^[[:space:]]*user[=:][[:space:]]*//p' | head -1 || true)
[ -n "$who" ] || { echo "error: not logged in to Hugging Face (run: hf auth login)" >&2; exit 2; }
echo "hugging face user: $who; repo: $HF_REPO; folder: $OUT"
[ -f "$OUT/octojet.json" ] && [ -f "$OUT/experts/model.safetensors.index.json" ] || { echo "error: $OUT is not a built release checkpoint" >&2; exit 2; }
[ -f "$MLX/LICENSE" ] || { echo "error: $MLX/LICENSE missing" >&2; exit 2; }
for f in docs/model-card.md docs/release-checkpoint/NOTICE docs/images/speed-vs-upstream.png docs/images/accuracy.png; do
  [ -f "$REPO_DIR/$f" ] || { echo "error: $REPO_DIR/$f missing" >&2; exit 2; }
done
if [ "$DRY" = 1 ]; then echo "dry-run OK: would stage README.md, NOTICE, LICENSE, assets/ into $OUT and upload to $HF_REPO (private)"; exit 0; fi
cp "$REPO_DIR/docs/model-card.md" "$OUT/README.md"
cp "$REPO_DIR/docs/release-checkpoint/NOTICE" "$OUT/NOTICE"
cp "$MLX/LICENSE" "$OUT/LICENSE"
mkdir -p "$OUT/assets"
cp "$REPO_DIR"/docs/images/*.png "$OUT/assets/"
if hf repos --help >/dev/null 2>&1; then          # hf 2.x: `repos`, and `upload` itself resumes when re-run
  hf repos create "$HF_REPO" --type model --private --exist-ok
  hf upload "$HF_REPO" "$OUT" . --type model --private --exclude ".cache/*" \
    --commit-message "Octojet NVFP4 release checkpoint (Qwen3.8 Flash Next)"
else                                               # hf 1.x
  hf repo create "$HF_REPO" --repo-type model --private --exist-ok
  hf upload-large-folder "$HF_REPO" "$OUT" --repo-type model
fi
echo "uploaded to https://huggingface.co/$HF_REPO (private): review it there; making it public is the owner's call"
