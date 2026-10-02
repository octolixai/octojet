#!/usr/bin/env bash
# Re-run the post-pause reduction for one nsys report on the Spark host (no pause): usage f2c-phase1-reduce.sh <report name> [<report name> ...]
set -euo pipefail
REPO=$HOME/octojet-f2c; OUT=$HOME/octojet-runs/f2c-phase1; P=$HOME/tensorfold/octojet-profiles
mkdir -p "$OUT"; cd "$REPO"
TL="$OUT/f2c-phase1-timeline.txt"; log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$TL"; }
reduce(){  # $1 report name: kernel-row validation (nsys stats auto-exports sqlite), sqlite export + range reduction — CPU-heavy, AFTER restore
  docker run --rm -v "$HOME/tensorfold:/tf" --entrypoint /usr/local/cuda/bin/nsys tensorfold:0.3.6.2 stats --report cuda_gpu_kern_sum --format csv "/tf/octojet-profiles/$1.nsys-rep" > "$P/$1-kern.csv" 2>"$P/$1-stats.err" || { log "$1: nsys stats failed"; return 1; }
  python3 - "$P/$1-kern.csv" <<'PY' || { log "$1: no kernel rows in the report"; return 1; }
import csv, sys
raw = open(sys.argv[1]).read().splitlines()                                          # nsys stats prints "Generating SQLite file ..." /
start = next((i for i, l in enumerate(raw) if l.startswith("Time (%)")), None)       # "Processing [...]" before the CSV header
if start is None:
    print(f"{sys.argv[1]}: no 'Time (%)' header line found; first lines: {raw[:3]}"); sys.exit(1)
lines = [l for l in raw[start:] if l.strip() and not l.startswith("#")]
rows = list(csv.DictReader(lines))
cols = list(rows[0].keys()) if rows else []
inst = next((k for k in cols if k and k.startswith("Instances")), None)
ok = bool(rows) and inst is not None and "Name" in cols and any(
    r[inst].strip().isdigit() and int(r[inst]) > 0 and r["Name"].strip() for r in rows)
print(f"{sys.argv[1]}: {len(rows)} kernel rows; columns {cols}")
sys.exit(0 if ok else 1)
PY
  docker run --rm -v "$HOME/tensorfold:/tf" --entrypoint /usr/local/cuda/bin/nsys tensorfold:0.3.6.2 export --type sqlite -f true -o "/tf/octojet-profiles/$1.sqlite" "/tf/octojet-profiles/$1.nsys-rep" || { log "$1: sqlite export failed"; return 1; }
  python3 bench/nsys_ranges.py "$P/$1.sqlite" --out "$P/$1-ranges.json" | tee -a "$TL" || { log "$1: reducer rejected the capture (no admission bounds / no ranges)"; return 1; }; }
FAIL=0
for name in "$@"; do
  rm -f "$OUT/${name}-reduce-invalid"
  reduce "$name" || { log "$name: REDUCTION INVALID"; touch "$OUT/${name}-reduce-invalid"; FAIL=1; }
done
cp "$P"/*-ranges.json "$P"/*-kern.csv "$P"/*-stats.err "$P"/timing-records.jsonl "$OUT"/ 2>/dev/null || true
exit "$FAIL"
