"""Mixed Flash Next checkpoint for the CUDA engine: routed experts from a ModelOpt NVFP4 export (RadixArk layout,
one shard per 128 experts a layer), everything else from the MLX affine checkpoint the served directory links to.
The shared expert is quantized to NVFP4 here from the export's bf16 copy (or, in the fp8hybrid export, its fp8 copy
dequantized with its scale first), so one table holds all 513 experts."""

from __future__ import annotations

import functools
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.capacity import SIZES
from tensorfold.cuda.load_timing import PHASES
from tensorfold.families.qwen4_exp.release import DECODER_ROUTED, EXPORT_ROUTED, EXPORT_SHARED, INDEX

MARKER = "octojet.json"
FORMAT = "nvfp4-mixed"
PREFIX = "model.language_model.layers."
NVFP4_PER_AFFINE = 144 / 160          # bytes of an NVFP4 block over an affine group-32 block


@dataclass(frozen=True)
class Mixed:
    experts: Path   # the NVFP4 export
    base: Path      # the MLX checkpoint (the served directory links to its files)


def is_mixed(model_dir: str | Path) -> bool:
    return (Path(model_dir) / MARKER).is_file()


def sources(model_dir: str | Path) -> Mixed:
    """The marker's two sources; relative paths (the self-contained release layout: "experts", ".") resolve against
    the served directory, absolute ones (a local symlink layout from tools/make_mixed_dir.py) stay as they are."""

    root = Path(model_dir)
    raw = json.loads((root / MARKER).read_text())
    if raw.get("format") != FORMAT:
        raise ValueError(f"{MARKER}: format {raw.get('format')!r}, expected {FORMAT!r}")
    experts, base = Path(raw["experts"]), Path(raw["base"])
    return Mixed(experts if experts.is_absolute() else root / experts, base if base.is_absolute() else root / base)




def release_expert_files(model_dir: str | Path) -> tuple[Path, ...]:
    """The export's shards when the served directory's own index holds no decoder routed experts (the
    self-contained release layout), so the startup estimate counts them; () for the symlink layout, whose MLX
    headers already stand in for them (``estimate_transform``)."""

    root = Path(model_dir)
    if not is_mixed(root):
        return ()
    names = json.loads((root / INDEX).read_text())["weight_map"]
    if any(DECODER_ROUTED.match(n) for n in names):
        return ()
    experts = sources(root).experts
    shards = sorted(set(json.loads((experts / INDEX).read_text())["weight_map"].values()))
    return tuple(experts / s for s in shards)


_DT = {"U8": torch.uint8, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
       "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2, "I32": torch.int32, "U32": torch.int32, "I64": torch.int64}


class SafetensorsDir:
    """Tensors of a sharded safetensors directory by name, read on the CPU; ``release()`` drops the read shards'
    page cache (unified memory: the cache competes with the model)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.where = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self._headers: dict[str, tuple[int, dict]] = {}
        self.touched: set[str] = set()

    def release(self) -> None:
        import os

        for shard in list(self.touched):
            try:
                fd = os.open(self.root / shard, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass
            finally:
                os.close(fd)
        self.touched.clear()

    def has(self, name: str) -> bool:
        return name in self.where

    def _header(self, shard: str) -> tuple[int, dict]:
        got = self._headers.get(shard)
        if got is None:
            with open(self.root / shard, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                got = (8 + n, json.loads(f.read(n)))
            self._headers[shard] = got
        return got

    def get(self, name: str) -> torch.Tensor:
        shard = self.where[name]
        base, header = self._header(shard)
        entry = header[name]
        begin, end = entry["data_offsets"]
        raw = torch.empty((end - begin,), dtype=torch.uint8)
        view = memoryview(raw.numpy())
        with open(self.root / shard, "rb", buffering=0) as f:
            f.seek(base + begin)
            at = 0
            while at < len(view):                       # a 400 MB expert stack can short-read
                got = f.readinto(view[at:at + (64 << 20)])
                if not got:
                    raise IOError(f"short read of {name}")
                at += got
        self.touched.add(shard)
        return raw.view(_DT[entry["dtype"]]).reshape(entry["shape"])


@functools.lru_cache(maxsize=2)
def _reader(root: str) -> SafetensorsDir:
    """One parsed index per export (its index lists ~300k tensors; 48 layers share it)."""

    return SafetensorsDir(root)


def routed_experts(rd: SafetensorsDir, layer: int, proj: str, n_experts: int = 512):
    """Stack expert 0..n_experts-1 of one projection: (packed u8 [E, N, K/2], scales fp8 [E, N, K/16], gscale [E])."""

    packed, scales, gscale = [], [], []
    for e in range(n_experts):
        p = f"{PREFIX}{layer}.mlp.experts.{e}.{proj}"
        packed.append(rd.get(p + ".weight"))
        scales.append(rd.get(p + ".weight_scale"))
        gscale.append(rd.get(p + ".weight_scale_2").reshape(()))
    return torch.stack(packed), torch.stack(scales), torch.stack(gscale).float()


_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)
SHARED_ABSMAX = 16.0      # Flash Next's shared-expert weights are O(1) (MLX absmax 0.5-0.8); raw fp8 codes reach 448


def _expand_blocks(scale: torch.Tensor, n: int, k: int, block: int = 128) -> torch.Tensor:
    """A [ceil(N/block), ceil(K/block)] block-scale grid expanded to [N, K] (the last blocks may be partial)."""

    return scale.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]


def _apply_scale(w: torch.Tensor, s: torch.Tensor, name: str) -> torch.Tensor:
    n, k = w.shape
    s = s.float()
    if s.dim() == 0 or s.numel() == 1:
        return w * s.reshape(())
    if s.dim() == 1 and s.shape[0] == n or s.dim() == 2 and tuple(s.shape) == (n, 1):
        return w * s.reshape(n, 1)
    if s.dim() == 2 and s.shape[0] == -(-n // 128) and s.shape[1] == -(-k // 128):
        return w * _expand_blocks(s, n, k)          # fine-grained fp8: 128x128 blocks
    raise ValueError(f"{name}: scale shape {tuple(s.shape)} does not fit weight {(n, k)}")


def shared_weight(rd: SafetensorsDir, layer: int, proj: str) -> torch.Tensor:
    """The shared expert's float weight in the model's real scale. HF ``main`` stores it in bf16; the local fp8hybrid
    export stores fp8 codes with a companion scale that MULTIPLIES them: ``weight_scale_inv`` as 128x128 block scales
    [ceil(N/128), ceil(K/128)] (verified: 448 x max scale = MLX absmax, despite the name), or ``weight_scale``
    per tensor / per row."""

    p = f"{PREFIX}{layer}.mlp.shared_expert.{proj}"
    w = rd.get(p + ".weight")
    if w.dtype in _FP8:
        for suffix in (".weight_scale_inv", ".weight_scale"):
            if rd.has(p + suffix):
                w = _apply_scale(w.float(), rd.get(p + suffix), p + suffix)
                print(f"[octojet] layer {layer}: shared expert dequantized from fp8 (scale {p + suffix})", file=sys.stderr, flush=True)
                break
        else:
            raise ValueError(f"{p}: fp8 weights without a weight_scale; cannot dequantize the shared expert")
    else:
        w = w.float()
    amax = w.abs().amax().item()
    if not math.isfinite(amax):
        raise ValueError(f"{p}: shared-expert weight has non-finite values (absmax {amax})")
    if amax > SHARED_ABSMAX:
        raise ValueError(f"{p}: shared-expert values up to {amax:.1f}; an fp8 tensor read without its scale?")
    return w


def shared_expert(rd: SafetensorsDir, layer: int, proj: str):
    packed, scales, gscale = grouped.quantize_nvfp4(shared_weight(rd, layer, proj))
    return packed[None], scales[None], gscale.reshape(1)


def make_layer(mixed: Mixed, layer: int, device, *, n_experts: int = 512, hidden: int = 2560, width: int = 640,
               shared_width: int = 640, limit: float = 0.0) -> grouped.Experts:
    """One layer's 513-expert NVFP4 table (the shared expert last), packed on the CPU and moved to ``device``."""

    if shared_width != width:
        raise ValueError("the shared expert must have the routed experts' width to share their table")
    rd = _reader(str(mixed.experts))

    def table(proj, n, k):
        with PHASES.phase("read_routed"):
            rp, rs, rg = routed_experts(rd, layer, proj, n_experts)
        with PHASES.phase("shared_expert"):
            sp, ss, sg = shared_expert(rd, layer, proj)
        if tuple(rp.shape[1:]) != (n, k // 2):
            raise ValueError(f"layer {layer} {proj}: expert shape {tuple(rp.shape[1:])}, expected {(n, k // 2)}")
        return torch.cat([rp, sp]), torch.cat([rs, ss]), torch.cat([rg, sg])

    up = [table("gate_proj", width, hidden), table("up_proj", width, hidden)]
    down = table("down_proj", hidden, width)
    with PHASES.phase("pack"):
        ex = grouped.make_nvfp4(up, down, limit=limit)
    rd.release()
    with PHASES.phase("upload"):
        return grouped.Experts(ex.up.to(device), ex.down.to(device), ex.gs, ex.width, ex.dims, ex.limit, fmt=ex.fmt,
                               gscale_up=ex.gscale_up.to(device), gscale_down=ex.gscale_down.to(device))


def agreement(a: torch.Tensor, b: torch.Tensor) -> dict:
    """How two dequantizations of one matrix agree. Cosine alone ignores magnitude (a lost per-matrix scale would pass),
    so the norm ratio and the relative error are checked too; 4-bit-vs-4-bit noise is ~10-20% relative."""

    a, b = a.float().reshape(-1), b.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a[None], b[None]).item()
    ratio = (b.norm() / a.norm().clamp_min(1e-30)).item()
    rel = ((a - b).norm() / a.norm().clamp_min(1e-30)).item()
    return {"cos": round(cos, 4), "norm_ratio": round(ratio, 4), "rel_err": round(rel, 4),
            "ok": cos >= 0.98 and 0.9 <= ratio <= 1.1 and rel <= 0.25}


def estimate_transform(inner: Callable) -> Callable:
    """The startup estimate for a mixed checkpoint. Symlink layout: the MLX headers stand in for the decoder's routed
    experts, rescaled to NVFP4 (144 bytes a block against 160). Release layout: the base has no decoder experts and
    the export's own shards (``release_expert_files``) are counted as packed: routed tensors as stored, the shared
    expert's weight requantized (144 bytes per 256 values), everything else in the export never loaded."""

    def transform(name: str, info: dict):
        if name.startswith("model.language_model."):     # the release layout's export shards (release_expert_files)
            values = math.prod(info["shape"])
            if EXPORT_ROUTED.match(name):                 # packed as stored: 4-bit codes, fp8 block scales, scale_2
                return values * SIZES[info["dtype"]], 0
            if EXPORT_SHARED.match(name):                 # requantized to NVFP4: 144 bytes per 256 values
                return values * 144 // 256, 0
            return 0, 0                                   # input scales and the rest: never loaded
        size, host = inner(name, info)
        # decoder layers only; the MTP head's experts (language_model.mtp.layers.0.mlp.switch_mlp.*) stay affine
        if name.startswith("language_model.model.layers.") and ".mlp.switch_mlp." in name:
            size = int(size) * 144 // 160
        return size, host

    return transform
