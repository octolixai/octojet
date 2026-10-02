#!/bin/bash
# run_humaneval.sh RESULTS.jsonl — execute the HumanEval programs in a throwaway container with no network,
# 15 s per program. Prints pass@1. SANDBOX_IMAGE: any image with python3 (default nvcr.io/nvidia/pytorch:26.07-py3).
set -euo pipefail
IN=$(readlink -f "$1"); D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
python3 - "$IN" "$D" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if '"humaneval"' in l]
for i, r in enumerate(rows):
    open(f"{sys.argv[2]}/p{i:03d}.py", "w").write(r["program"])
EOF
docker run --rm --network none --memory 4g -v "$D":/p:ro --entrypoint bash "${SANDBOX_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}" -c '
pass=0; n=0; for f in /p/p*.py; do n=$((n+1)); if timeout 15 python "$f" >/dev/null 2>&1; then pass=$((pass+1)); echo "$(basename "$f") pass"; else echo "$(basename "$f") fail"; fi; done
echo "{\"humaneval_pass\": $pass, \"humaneval_n\": $n, \"pass_at_1\": $(python -c "print(round($pass/$n,4))")}"' | tee "$IN.humaneval.txt"
