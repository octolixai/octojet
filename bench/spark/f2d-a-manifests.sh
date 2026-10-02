#!/usr/bin/env bash
# F2d manifests (~2 min, no GPU): P (71,444 tokens, the classifier's cold call), the stage-B variants P' (shares 69,013)
# and P'' (P' + 2,083), and the multi-burst sequence P2 = P + 2,048, P2' (P2 minus its last 2,431 tokens + 2,470 fresh),
# P3 = P2 + 2,048. Copies the manifests into the run directory as receipts.
set -euo pipefail
source "$(dirname "$0")/f2d-a-lib.sh"
docker run --rm -v "$REPO:/octojet:ro" -v "$HOME/tensorfold:/tf" -e PYTHONPATH=/octojet/engine/src -w /octojet --entrypoint bash "$IMAGE" -c '
  set -euo pipefail; mkdir -p /tf/octojet-manifests; T=/tf/flashnext-nvfp4-mixed/tokenizer.json; D=/tf/octojet-manifests
  python3 bench/prefill_prompt.py --tokenizer $T --tokens 71444 --seed 11 --tag cP --out $D/cP.json
  python3 bench/prefill_prompt.py --from $D/cP.json  --take 69013 --tail 2470 --seed 12 --tag cPp  --out $D/cPp.json
  python3 bench/prefill_prompt.py --from $D/cPp.json --extend 2083            --seed 13 --tag cPpp --out $D/cPpp.json
  python3 bench/prefill_prompt.py --from $D/cP.json  --extend 2048            --seed 14 --tag cP2  --out $D/cP2.json
  python3 bench/prefill_prompt.py --from $D/cP2.json --take 71061 --tail 2470 --seed 15 --tag cP2p --out $D/cP2p.json
  python3 bench/prefill_prompt.py --from $D/cP2.json --extend 2048            --seed 16 --tag cP3  --out $D/cP3.json
  sha256sum $D/cP*.json' | tee -a "$TL"
cp "$M"/cP*.json "$OUT"/
log "manifests written to $M and copied to $OUT"
