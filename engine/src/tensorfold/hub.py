"""Resolve local model directories or Hugging Face snapshots, downloading missing weights when requested."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_repo_id(name: str) -> bool:
    """``owner/name`` that is not an existing local path."""

    return bool(_REPO_ID.match(str(name))) and not Path(str(name)).expanduser().exists()


def cached(repo_id: str, *, cache_dir: Any = None) -> Path | None:
    """Use the cached snapshot, falling back to the newest config-bearing snapshot when refs/main is absent."""

    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, local_files_only=True, cache_dir=cache_dir))
    except Exception:  # noqa: BLE001 - not cached, or cached without a ref: look at the snapshots themselves
        pass
    if cache_dir is None:
        from huggingface_hub import constants

        cache_dir = constants.HF_HUB_CACHE
    snapshots = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    found = [s for s in snapshots.glob("*") if (s / "config.json").is_file()] if snapshots.is_dir() else []
    return max(found, key=lambda s: s.stat().st_mtime) if found else None


def pull(repo_id: str, *, cache_dir: Any = None) -> Path:
    """Download (or finish downloading) a repo into the cache; returns its snapshot directory."""

    from huggingface_hub import snapshot_download

    print(f"[octojet] downloading {repo_id} from Hugging Face", flush=True)
    return Path(snapshot_download(repo_id, cache_dir=cache_dir))


MIXED_MARKER = "octojet.json"
MIXED_INCOMPLETE = f"mixed checkpoint without {MIXED_MARKER}: the download is incomplete"
_ROUTER = re.compile(r"^language_model\.model\.layers\.\d+\.mlp\.gate\.weight$")
_DECODER_ROUTED = re.compile(r"^language_model\.model\.layers\.\d+\.mlp\.switch_mlp\.")


def _index_files(snapshot: Path, index: Path) -> tuple[dict | None, bool]:
    """(weight map, every shard it names present) of one safetensors index; (None, False) when unreadable."""

    try:
        weight_map = json.loads(index.read_text())["weight_map"]
    except (OSError, ValueError, KeyError, TypeError):
        return None, False
    if not isinstance(weight_map, dict) or not weight_map:
        return None, False
    try:
        files = set(weight_map.values())
    except TypeError:
        return None, False
    root = index.parent
    return weight_map, all(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts
                           and (root / name).is_file() for name in files)


def incomplete(snapshot: Path, *, required_files: tuple[str, ...] = ()) -> str | None:
    """Why a local snapshot cannot be served yet (None when complete). Hugging Face also returns partial snapshots.

    A mixed NVFP4 Flash Next checkpoint (octojet.json, spec 2026-09-29-f2-release-design.md 4.4) also needs
    experts/model.safetensors.index.json and every shard it names; without the marker, an experts/ directory or a
    Flash Next base index with routers but no decoder routed experts means the marker has not arrived yet."""

    missing = [name for name in required_files if not (snapshot / name).is_file()]
    if missing:
        return f"missing required files: {', '.join(missing)}"

    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        weight_map, present = _index_files(snapshot, index)
        if weight_map is None:
            return "unreadable model.safetensors.index.json"
        if not present:
            return "missing weight shards"
        marker = snapshot / MIXED_MARKER
        experts = snapshot / "experts"
        if marker.is_file():
            try:
                raw = json.loads(marker.read_text())
                where = raw.get("experts", "experts") if isinstance(raw, dict) else "experts"
            except (OSError, ValueError):
                return f"unreadable {MIXED_MARKER}"
            root = Path(where) if Path(str(where)).is_absolute() else snapshot / str(where)
            if not (root / "model.safetensors.index.json").is_file():
                return f"mixed checkpoint: {root.name}/model.safetensors.index.json missing"
            emap, epresent = _index_files(root, root / "model.safetensors.index.json")
            if emap is None:
                return f"mixed checkpoint: unreadable {root.name}/model.safetensors.index.json"
            if not epresent:
                return f"mixed checkpoint: missing expert shards in {root.name}/"
            return None
        if experts.is_dir():
            return MIXED_INCOMPLETE
        if any(_ROUTER.match(n) for n in weight_map) and not any(_DECODER_ROUTED.match(n) for n in weight_map):
            return MIXED_INCOMPLETE
        return None

    shards = list(snapshot.glob("model-*-of-*.safetensors"))
    if shards:
        matches = [re.fullmatch(r"model-(\d+)-of-(\d+)\.safetensors", path.name) for path in shards]
        if not all(matches):
            return "unexpected shard names"
        totals = {int(match.group(2)) for match in matches}
        if len(totals) != 1:
            return "shards of different totals"
        total = totals.pop()
        if len(shards) == total and {int(match.group(1)) for match in matches} == set(range(1, total + 1)):
            return None
        return "missing weight shards"

    return None if (snapshot / "model.safetensors").is_file() else "no weights"


def _cached_weights_complete(snapshot: Path, *, required_files: tuple[str, ...] = ()) -> bool:
    """Require complete weights before serving because Hugging Face also returns partial local snapshots."""

    return incomplete(snapshot, required_files=required_files) is None


def resolve(name: str, *, download: bool = True, cache_dir: Any = None,
            required_files: tuple[str, ...] = ()) -> Path:
    """A model directory for ``name``: the directory itself, or a repo id's snapshot (downloaded if needed)."""

    path = Path(str(name)).expanduser()
    if path.is_dir():
        return path
    if not is_repo_id(str(name)):
        raise FileNotFoundError(f"{name} is neither a directory nor a Hugging Face repo id (owner/name)")
    found = cached(str(name), cache_dir=cache_dir)
    if found is not None and (found / "config.json").is_file() and (
        not download or _cached_weights_complete(found, required_files=required_files)
    ):
        return found
    if not download:
        raise FileNotFoundError(f"{name} is not in the Hugging Face cache; run: octojet pull {name}")
    downloaded = pull(str(name), cache_dir=cache_dir)
    reason = incomplete(downloaded, required_files=required_files)
    if required_files and reason is not None and reason.startswith("missing required files"):
        raise FileNotFoundError(f"{name} is missing required files: {', '.join(required_files)}")
    if reason is not None and reason.startswith("mixed checkpoint"):
        raise FileNotFoundError(f"{name}: {reason}")
    return downloaded


def size_of(directory: Path) -> int:
    """Bytes of the files under ``directory`` (following the cache's symlinks)."""

    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())
