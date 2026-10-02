docker run --rm -v "$HOME/octojet-f2c:/octojet:ro" -v "$HOME/tensorfold:/tf" -e PYTHONPATH=/octojet/engine/src -w /octojet --entrypoint bash tensorfold:0.3.6.2 -c '
  set -euo pipefail; mkdir -p /tf/octojet-manifests; T=/tf/flashnext-nvfp4-mixed/tokenizer.json
  python3 bench/prefill_prompt.py --tokenizer $T --tokens 32000  --seed 1 --tag m32k   --out /tf/octojet-manifests/m32k.json
  python3 bench/prefill_prompt.py --tokenizer $T --tokens 128000 --seed 2 --tag m128k  --out /tf/octojet-manifests/m128k.json
  python3 bench/prefill_prompt.py --tokenizer $T --tokens 210000 --seed 3 --tag m210k  --out /tf/octojet-manifests/m210k.json
  python3 bench/prefill_prompt.py --tokenizer $T --tokens 140000 --seed 7 --tag warm   --out /tf/octojet-manifests/warm140k.json
  sha256sum /tf/octojet-manifests/*.json'
