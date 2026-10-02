#!/usr/bin/env python3
"""Build a served directory for the NVFP4 mixed checkpoint: symlinks to every file of the MLX checkpoint (config,
tokenizer, index and shards, so every existing check and the base loader work unchanged) plus octojet.json.

  python tools/make_mixed_dir.py MLX_DIR EXPORT_DIR OUT_DIR
"""
import json
import os
import sys
from pathlib import Path

mlx, export, out = (Path(p).resolve() for p in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()):
    sys.exit(f"{out} is not empty: a reused directory could keep links to another base checkpoint; remove it first")
for f in sorted(mlx.iterdir()):
    os.symlink(f, out / f.name)
(out / "octojet.json").write_text(json.dumps({"format": "nvfp4-mixed", "experts": str(export), "base": str(mlx)}, indent=1))
print(f"{out}: {len(list(out.iterdir()))} entries")
