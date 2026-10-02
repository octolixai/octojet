"""The self-contained mixed Flash Next checkpoint (spec 2026-09-29-f2-release-design.md, section 4): which tensors the
release keeps from each source, and raw safetensors helpers that move payload bytes without decoding them (no torch).

Layout: the MLX checkpoint's tensors except the decoder layers' routed and shared experts at the root (resharded, a
weight's scales and biases never split from it), the NVFP4 export's routed experts and shared-expert weights under
``experts/``, and ``octojet.json`` naming both relatively ("base": ".", "experts": "experts")."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from pathlib import Path

INDEX = "model.safetensors.index.json"
EXPERTS_DIR = "experts"
REPOS = {"base": "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP", "experts": "RadixArk/Qwen3.8-Flash-Next-NVFP4"}

# MLX names (the base): the decoder layers' routed stacks, and what the mixed loader takes from the export instead
DECODER_ROUTED = re.compile(r"^language_model\.model\.layers\.\d+\.mlp\.switch_mlp\.")
DECODER_DROPPED = re.compile(r"^language_model\.model\.layers\.\d+\.mlp\."
                             r"(switch_mlp\.|shared_expert\.(gate|up|down)_proj\.)")
# export names: routed codes, block scales and global scale (no input_scale); the shared expert's weight (bf16 on
# HF main) and, in an fp8 export, the scale the loader multiplies it by
EXPORT_ROUTED = re.compile(r"^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\."
                           r"(weight|weight_scale|weight_scale_2)$")
EXPORT_SHARED = re.compile(r"^model\.language_model\.layers\.\d+\.mlp\.shared_expert\.(gate|up|down)_proj\.weight$")
EXPORT_SHARED_SCALE = re.compile(r"^model\.language_model\.layers\.\d+\.mlp\.shared_expert\.(gate|up|down)_proj\."
                                 r"(weight_scale|weight_scale_inv|weight_scale_2)$")
ROUTER = re.compile(r"^language_model\.model\.layers\.\d+\.mlp\.gate\.weight$")

_GROUP_SUFFIX = re.compile(r"\.(weight|scales|biases)$")
GIB = 1 << 30


def dropped_from_base(name: str) -> bool:
    return bool(DECODER_DROPPED.match(name))


def needed_from_export(name: str) -> bool:
    return bool(EXPORT_ROUTED.match(name) or EXPORT_SHARED.match(name) or EXPORT_SHARED_SCALE.match(name))


def group_of(name: str) -> str:
    """Tensors sharing this prefix (``X.weight``, ``X.scales``, ``X.biases``) stay in one shard."""

    return _GROUP_SUFFIX.sub("", name)


def read_header(path: str | Path) -> tuple[int, dict]:
    """(byte offset of the data section, header with ``__metadata__``) of a safetensors file."""

    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError(f"{path}: not a safetensors file")
        n = struct.unpack("<Q", head)[0]
        if not 0 < n <= min(100 << 20, os.fstat(f.fileno()).st_size - 8):
            raise ValueError(f"{path}: truncated or invalid safetensors header")
        header = json.loads(f.read(n))
    if not isinstance(header, dict):
        raise ValueError(f"{path}: the safetensors header is not a JSON object")
    return 8 + n, header


def payload_size(info: dict) -> int:
    begin, end = info["data_offsets"]
    return end - begin


def read_index(root: str | Path) -> dict[str, str]:
    return json.loads((Path(root) / INDEX).read_text())["weight_map"]


def _chunks(f, start: int, size: int, block: int = 64 << 20):
    f.seek(start)
    left = size
    while left:
        got = f.read(min(block, left))
        if not got:
            raise IOError(f"short read at {f.tell()} of {getattr(f, 'name', '?')}")
        left -= len(got)
        yield got


def tensor_sha256(path: str | Path, name: str, header: tuple[int, dict] | None = None) -> str:
    """sha256 of one tensor's payload bytes, streamed."""

    base, entries = header or read_header(path)
    info = entries[name]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in _chunks(f, base + info["data_offsets"][0], payload_size(info)):
            h.update(chunk)
    return h.hexdigest()


def drop_cache(*paths: str | Path) -> None:
    """Write back and drop these files' page cache (Linux): a 100 GB build or check beside a serving process on
    unified memory should not push out the server's mapped tables. A no-op where posix_fadvise is missing."""

    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (OSError, AttributeError):
            pass
        finally:
            os.close(fd)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            h.update(chunk)
    drop_cache(path)
    return h.hexdigest()


def write_shard(out: str | Path, tensors: list[tuple[str, Path, int, dict]], metadata: dict | None) -> int:
    """Write ``tensors`` [(name, source file, source data offset, source header entry)] in order into one safetensors
    file, copying each payload byte for byte (dtype, shape and name unchanged). Returns the payload bytes written."""

    header: dict = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    at = 0
    for name, _, _, info in tensors:
        size = payload_size(info)
        header[name] = {"dtype": info["dtype"], "shape": list(info["shape"]), "data_offsets": [at, at + size]}
        at += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)                               # data section 8-byte aligned, as safetensors writes it
    opened: dict[Path, object] = {}
    try:
        with open(out, "wb") as dst:
            dst.write(struct.pack("<Q", len(raw)))
            dst.write(raw)
            for name, src, base, info in tensors:
                f = opened.get(src)
                if f is None:
                    f = opened[src] = open(src, "rb")
                for chunk in _chunks(f, base + info["data_offsets"][0], payload_size(info)):
                    dst.write(chunk)
    finally:
        for f in opened.values():
            f.close()
    drop_cache(out, *opened)
    return at
