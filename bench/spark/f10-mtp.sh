#!/usr/bin/env bash
# F10 decode tuning: drafted decode speed per MTP setting on ~20 code and ~20 chat/prose prompts (bench/mtp_sweep.py).
# For each --mtp-confidence in CONFS (default 0.2 0.3 0.5 0.7) and each --mtp-drafts in DEPTHS (default 6, the CUDA
# default) a fresh test server with production's flags (text only: --kv-dtype int8 --parallel 3 --packed-cache) plus
# that setting; mtp_sweep runs 40 prompts x greedy/sampled x 200 tokens against it (-> $OUT/f10-c<conf>-d<depth>.jsonl),
# then one summary table over every setting with the recommendation rule (geometric mean of the four medians; the
# production setting 0.3 / 6 is the default) -> $OUT/f10-summary.txt (+ .json). A token mismatch between settings
# fails the run (drafting never changes a reply).
# Production oj-serve is STOPPED for the whole run (owner's standing test approval; PROD_STOP), the standby pause held
# and removed on every exit path; the operator's watcher restarts oj-serve. About 8 min a setting (start ~2.5 min,
# sweep ~5 min): ~35 min for the default four.
# Usage: bash bench/spark/f10-mtp.sh [--dry-run]     Knobs: CONFS, DEPTHS, TOKENS (200), CODE (20), CHAT (20), PROD_STOP
set -euo pipefail
REPO=${REPO:-$HOME/octojet-f10}; OUT=${OUT:-$HOME/octojet-runs/f10}; NAME=${NAME:-oj-f10}; PARALLEL=${PARALLEL:-3}
PREFIX=f10; HOLD_PAUSE=1
source "$(dirname "$0")/f2d-a-lib.sh"
PROD_STOP=${PROD_STOP:-docker stop -t 60 oj-serve}
CONFS=${CONFS:-0.2 0.3 0.5 0.7}; DEPTHS=${DEPTHS:-6}; TOKENS=${TOKENS:-200}; CODE=${CODE:-20}; CHAT=${CHAT:-20}
DEFAULT_LABEL=c0.3-d6
BASE_SERVE=("${SERVE[@]}")
DRYRUN=0
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRYRUN=1; shift;; *) echo "usage: $0 [--dry-run]" >&2; exit 2;; esac; done
for c in $CONFS; do [[ "$c" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || { echo "error: CONFS takes probabilities 0..1 (got '$c')" >&2; exit 2; }; done
for d in $DEPTHS; do [[ "$d" =~ ^[0-9]+$ ]] && [ "$d" -ge 1 ] || { echo "error: DEPTHS takes drafts >= 1 (got '$d')" >&2; exit 2; }; done
cd "$REPO"
sweep(){ python3 bench/mtp_sweep.py run http://127.0.0.1:8080 f1 --tokens "$TOKENS" --code "$CODE" --chat "$CHAT" "$@"; }
# before the window: the prompt set builds (HumanEval / GSM8K fetched into the eval cache) and the plan prints
sweep --label plan --out /dev/null --dry-run | tee -a "$TL"
if [ "$DRYRUN" = 1 ]; then
  for d in $DEPTHS; do for c in $CONFS; do echo "dry-run: would serve with ${BASE_SERVE[*]} --mtp-confidence $c --mtp-drafts $d"; done; done
  exit 0
fi
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true
pause_on
log "WINDOW: stopping production oj-serve ($PROD_STOP)"
bash -c "$PROD_STOP" >/dev/null 2>&1 || log "WINDOW: '$PROD_STOP' returned non-zero (already stopped?)"
port_free || { log "WINDOW: port 8080 still listening"; exit 1; }
FILES=()
for d in $DEPTHS; do
  for c in $CONFS; do
    label="c$c-d$d"; rows="$OUT/f10-$label.jsonl"; rm -f "$rows"
    SERVE=("${BASE_SERVE[@]}" --mtp-confidence "$c" --mtp-drafts "$d")
    start "$label"
    grep -h "MTP drafts a round\|warm-up:" "$OUT/f10-$label-server.log" | tee -a "$TL" || true
    sweep --label "$label" --out "$rows" --setting "confidence=$c" --setting "depth=$d" 2>&1 | tee -a "$TL" \
      || { log "$label: sweep FAILED (rc ${PIPESTATUS[0]})"; FAIL=1; }
    stop "$label"
    if [ -s "$rows" ]; then FILES+=("$rows"); fi
  done
done
pause_off
[ ${#FILES[@]} -gt 0 ] || { log "no sweep rows"; exit 1; }
python3 bench/mtp_sweep.py summary "${FILES[@]}" --default-label "$DEFAULT_LABEL" --json "$OUT/f10-summary.json" \
  | tee "$OUT/f10-summary.txt" | tee -a "$TL" || { log "summary: token mismatch across settings (see $OUT/f10-summary.txt)"; FAIL=1; }
log "f10 mtp sweep done: rc $FAIL (summary $OUT/f10-summary.txt)"
exit "$FAIL"
