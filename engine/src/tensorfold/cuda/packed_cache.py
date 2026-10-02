"""Packed NVFP4 expert tables cached on disk, so a restart uploads instead of repacking.

Spec: docs/superpowers/specs/2026-09-29-f2-release-design.md section 3. A table is served from the cache only when
its file binds to this checkpoint's key, this table's name, this rank, the current packer version and the model's
geometry, and its tensors have the expected names, dtypes and shapes; anything else is rebuilt and overwritten.
Every file also records the *source generation* it was built from (sha256 over the export shards' full content
hashes): ``verify`` mode recomputes the generation and every cached payload hash and rebuilds what differs, so a
payload edited in place (which the sampled key cannot see) is caught there. Writes are atomic (unique temporary +
fsync + rename); nothing is deleted except temporaries older than an hour. Reads parse the header and check it
before any payload byte, then read each tensor with large sequential ``readinto`` calls (no mmap: a memory-mapped
table would be read through page faults during the device upload). Any cache failure is logged and serving
continues from the built table. Imports stay light (torch, safetensors, ``experts``): the module is tested on a
CPU-only host.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import save_file

from . import experts as grouped

MODES = ("on", "off", "verify")
SAMPLE = 1 << 20              # payload bytes hashed at each end of a shard for the key
ORPHAN_AGE_S = 3600           # a .tmp-* older than this belongs to no live writer (a table writes in seconds)
INDEX = "model.safetensors.index.json"
MARKER = "octojet.json"
TENSORS = ("up", "down", "gscale_up", "gscale_down")
DTYPES = {"up": torch.int32, "down": torch.int32, "gscale_up": torch.float32, "gscale_down": torch.float32}
ST_DTYPES = {"I32": torch.int32, "F32": torch.float32}   # the safetensors header's dtype strings the tables use
READ_CHUNK = 64 << 20         # bytes a readinto call asks for when a cached table is read
MAX_HEADER = 1 << 20          # a table's safetensors header is a few KB; a larger length is corruption


def default_root() -> Path:
    base = os.environ.get("OCTOJET_CACHE_DIR")
    return (Path(base) if base else Path.home() / ".cache" / "octojet") / "packed"


def parse_option(value: str | None) -> tuple[Path | None, str]:
    """``--packed-cache`` → (root, mode): None → default root, on; "off" → (None, off); "verify" → default root,
    verify; anything else is a directory, on."""

    if value is None:
        return default_root(), "on"
    if value == "off":
        return None, "off"
    if value == "verify":
        return default_root(), "verify"
    return Path(value).expanduser(), "on"


def _sha256(data) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    return _sha256(memoryview(t.contiguous().cpu().numpy()).cast("B"))


def full_sha256(path: Path, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def shard_fingerprint(path: Path) -> dict:
    """What the key knows about one shard: size, mtime, header hash and the payload's first and last MiB."""

    st = path.stat()
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n)
        start = 8 + n
        f.seek(start)
        head = f.read(SAMPLE)
        f.seek(max(start, st.st_size - SAMPLE))
        tail = f.read(SAMPLE)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "header_sha256": _sha256(header),
            "head_sha256": _sha256(head), "tail_sha256": _sha256(tail)}


def referenced_shards(index_path: Path) -> list[str]:
    return sorted(set(json.loads(index_path.read_text())["weight_map"].values()))


@dataclass(frozen=True)
class Geometry:
    """What every table of one model must look like; checked on every hit."""

    fmt: str
    gs: int
    width: int
    dims: int
    limit: float
    experts: int

    def as_metadata(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    def shapes(self) -> dict[str, tuple[int, ...]]:
        e, nb, kg = self.experts, self.width // 32, self.dims // 32
        return {"up": (e, nb, kg, 2, grouped.NVFP4_BLOCK), "down": (e, kg, nb, 1, grouped.NVFP4_BLOCK),
                "gscale_up": (e, 2), "gscale_down": (e, 1)}

    def table_bytes(self) -> int:
        words = sum(int(torch.Size(s).numel()) for k, s in self.shapes().items() if k in ("up", "down"))
        scales = sum(int(torch.Size(s).numel()) for k, s in self.shapes().items() if k.startswith("gscale"))
        return words * 4 + scales * 4


def checkpoint_key(model_dir: Path, experts_dir: Path, geometry: Geometry, *, rank: int = 0,
                   world: int = 1) -> tuple[str, dict]:
    """The cache key and the inputs it was computed from. No absolute path enters the key: renaming a directory
    keeps its cache; editing the marker, an index, or a shard's header, size, mtime, first or last MiB changes it.
    The base's shards enter the key too (cheap); ``experts_root`` and ``experts_dir_shards`` are added to the
    returned inputs *after* hashing, for the generation and meta.json."""

    model_dir, experts_dir = Path(model_dir), Path(experts_dir)
    marker = model_dir / MARKER
    sources = {}
    for label, root in (("base", model_dir), ("experts", experts_dir)):
        index = root / INDEX
        sources[label] = {"index_sha256": _sha256(index.read_bytes()),
                          "shards": {name: shard_fingerprint(root / name) for name in referenced_shards(index)}}
    inputs = {"pack_version": grouped.PACK_VERSION, "marker": marker.read_text() if marker.is_file() else None,
              "geometry": asdict(geometry), "rank": rank, "world": world, "sources": sources}
    key = _sha256(json.dumps(inputs, sort_keys=True).encode())
    inputs = {**inputs, "experts_root": os.path.abspath(experts_dir),     # logical: the path the builder reads
              "experts_dir_shards": sorted(sources["experts"]["shards"])}
    return key, inputs


def _version() -> str:
    try:
        from tensorfold import __version__
        return str(__version__)
    except Exception:                       # pragma: no cover - the package always has a version
        return "unknown"


def _sequential(fd: int) -> None:
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
    except (OSError, AttributeError):        # macOS has no posix_fadvise
        pass


def _read_exact(f, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        block = f.read(n - len(out))
        if not block:
            raise ValueError(f"short read: {len(out)} of {n} bytes")
        out += block
    return bytes(out)


def _read_into(f, offset: int, nbytes: int) -> torch.Tensor:
    """``nbytes`` from ``offset`` into a fresh uint8 tensor, READ_CHUNK bytes a call (a large read can short-read)."""

    raw = torch.empty((nbytes,), dtype=torch.uint8)
    view = memoryview(raw.numpy())
    f.seek(offset)
    at = 0
    while at < nbytes:
        got = f.readinto(view[at:at + READ_CHUNK])
        if not got:
            raise ValueError(f"short read: {at} of {nbytes} bytes")
        at += got
    return raw


def _dontneed(fd: int) -> None:
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError):        # macOS has no posix_fadvise
        pass


class TableCache:
    def __init__(self, root: Path | None, key: str, inputs: dict, geometry: Geometry, *, mode: str = "on",
                 rank: int = 0, world: int = 1, tables: int | None = None,
                 log: Callable[[str], None] = print) -> None:
        if mode not in MODES:
            raise ValueError(f"packed cache mode {mode!r}: one of {', '.join(MODES)}")
        if mode != "off" and root is None:
            raise ValueError("packed cache: a directory is needed unless the mode is off")
        self.root, self.key, self.inputs, self.geometry, self.mode = root, key, inputs, geometry, mode
        self.rank, self.world, self.tables, self.log = rank, world, tables, log
        self.dir = (Path(root) / key) if root is not None else Path(key)
        self.hits = self.builds = self.saved = 0
        self.seconds = 0.0
        self.notes: list[str] = []
        self.disabled = mode == "off"
        self.export_moved = False
        self._generation: str | None = None
        self._full: dict[str, str] | None = None
        if not self.disabled:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                self.disabled = True
                self.notes.append(f"cache disabled: cannot create {self.dir} ({e})")
                return
            self._cleanup_orphans()
            if mode == "verify":
                self.generation()                       # hash the sources once, up front

    # ---- names, metadata, generation --------------------------------------------------------------------------

    def path(self, name: str) -> Path:
        return self.dir / (name.replace(".", "-") + ".safetensors")

    def metadata(self, name: str) -> dict[str, str]:
        return {"pack_version": str(grouped.PACK_VERSION), "checkpoint_key": self.key, "table": name,
                "rank": str(self.rank), "world": str(self.world), **self.geometry.as_metadata()}

    def _experts_shards(self) -> dict[str, Path]:
        root = self.inputs.get("experts_root")
        return {n: Path(root) / n for n in (self.inputs.get("experts_dir_shards") or [])} if root else {}

    def generation(self) -> str | None:
        """sha256 over (shard name, full sha256) of the export's shards: the identity of the sources a table is
        built from. Computed once per process, on the first build or up front in verify mode. None when the
        sources cannot be hashed (then nothing is published this session; serving continues)."""

        if self._generation is None and not self.export_moved:
            try:
                self._full = {n: full_sha256(p) for n, p in self._experts_shards().items()}
                self._generation = _sha256(json.dumps(sorted(self._full.items())).encode())
            except Exception as e:                # OSError, struct.error on a truncated shard, ...
                self.export_moved = True
                self.notes.append(f"sources cannot be hashed ({type(e).__name__}: {e}); nothing is cached this session")
        return self._generation

    def _export_unchanged(self) -> bool:
        """The export's index and shard fingerprints (size, mtime, header, first/last MiB) still equal what the
        key was computed from. False also when they cannot be read. Once the export has moved under a running
        start, ``export_moved`` stays set: the key no longer describes the sources, so nothing more is published
        and, in verify mode, every remaining hit is rebuilt."""

        if self.export_moved:
            return False
        try:
            src = (self.inputs.get("sources") or {}).get("experts") or {}
            root = Path(self.inputs["experts_root"])
            same = _sha256((root / INDEX).read_bytes()) == src.get("index_sha256") and all(
                shard_fingerprint(p) == (src.get("shards") or {}).get(n) for n, p in self._experts_shards().items())
        except Exception:
            same = False
        if not same:
            self.export_moved = True
            self.notes.append("export changed under this start; nothing more is cached (restart to cache)")
        return same

    # ---- read ------------------------------------------------------------------------------------------------

    def load(self, name: str) -> grouped.Experts | None:
        if self.disabled:
            return None
        try:
            path = self.path(name)
            if not path.is_file():                # a probe that raises counts as a miss (handled below)
                return None
            with open(path, "rb", buffering=0) as f:
                _sequential(f.fileno())
                size = os.fstat(f.fileno()).st_size
                n = struct.unpack("<Q", _read_exact(f, 8))[0]
                # bounded before any header byte is read: a corrupt length never drives a large read
                if n > MAX_HEADER or 8 + n > size or size - (8 + n) != self.geometry.table_bytes():
                    raise ValueError("header invalid")
                header = json.loads(_read_exact(f, n))
                meta = header.pop("__metadata__", None) or {}
                want = self.metadata(name)
                bad = [k for k, v in want.items() if meta.get(k) != v]
                if bad:
                    raise ValueError("metadata differs: " + ", ".join(bad))
                if set(header) != set(TENSORS):
                    raise ValueError(f"tensors {sorted(header)}, expected {sorted(TENSORS)}")
                missing = [k for k in ("source_generation", *("sha256_" + t for t in TENSORS)) if not meta.get(k)]
                if missing:
                    raise ValueError("metadata incomplete: " + ", ".join(missing))
                shapes, base, spans = self.geometry.shapes(), 8 + n, {}
                for k in TENSORS:                     # every check before the first payload byte is read
                    entry = header[k]
                    dtype, shape = entry["dtype"], tuple(int(d) for d in entry["shape"])
                    if shape != shapes[k] or ST_DTYPES.get(dtype) != DTYPES[k]:
                        raise ValueError(f"{k}: {dtype} {shape}, expected {DTYPES[k]} {shapes[k]}")
                    begin, end = (int(o) for o in entry["data_offsets"])
                    spans[k] = (begin, end, int(torch.Size(shape).numel()) * DTYPES[k].itemsize)
                order = sorted(TENSORS, key=lambda t: spans[t][0])      # file order
                at = 0
                for k in order:                       # the payload as a whole: from 0, no gap, no overlap, to EOF
                    begin, end, nbytes = spans[k]
                    if begin != at or end - begin != nbytes:
                        raise ValueError("payload layout invalid")
                    at = end
                if at != size - base:
                    raise ValueError("payload layout invalid")
                tensors = {}
                for k in order:                       # one sequential pass
                    begin, _, nbytes = spans[k]
                    raw = _read_into(f, base + begin, nbytes)
                    tensors[k] = raw.view(DTYPES[k]).reshape(shapes[k])
            if self.mode == "verify":
                if not self._export_unchanged():
                    raise ValueError("export changed since this start hashed it")
                if meta.get("source_generation") != self.generation():
                    raise ValueError("source generation differs from the export's current contents")
                for k, t in tensors.items():
                    if meta.get("sha256_" + k) != tensor_sha256(t):
                        raise ValueError(f"{k}: payload hash differs from its metadata")
                self.notes.append(f"{name}: verified")
        except Exception as e:                    # a file that fails any check is rebuilt, never served
            self.notes.append(f"{name}: {e}; rebuilding")
            return None
        g = self.geometry
        return grouped.Experts(tensors["up"], tensors["down"], g.gs, g.width, g.dims, g.limit, fmt=g.fmt,
                               gscale_up=tensors["gscale_up"], gscale_down=tensors["gscale_down"])

    # ---- write -----------------------------------------------------------------------------------------------

    def save(self, name: str, ex: grouped.Experts) -> bool:
        if self.disabled or self.export_moved:    # the key no longer describes the sources: publish nothing
            return False
        tmp = None
        try:
            path = self.path(name)
            tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
            tensors = {"up": ex.up, "down": ex.down, "gscale_up": ex.gscale_up, "gscale_down": ex.gscale_down}
            tensors = {k: t.contiguous().cpu() for k, t in tensors.items()}
            meta = self.metadata(name)
            meta.update(built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), octojet_version=_version(),
                        source_generation=self.generation(),
                        **{"sha256_" + k: tensor_sha256(t) for k, t in tensors.items()})
            save_file(tensors, str(tmp), metadata=meta)
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
                _dontneed(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path)
            self.saved += 1
            return True
        except Exception as e:                    # OSError, SafetensorError (not an OSError), anything else
            self.notes.append(f"{name}: not cached ({type(e).__name__}: {e})")
            if tmp is not None:
                try:
                    tmp.unlink()
                except OSError:
                    pass
            return False

    def release(self, name: str) -> None:
        """After the upload: drop the file's pages (unified memory: the page cache competes with the model)."""

        if self.disabled:
            return
        try:
            fd = os.open(self.path(name), os.O_RDONLY)
        except OSError:
            return
        try:
            _dontneed(fd)
        finally:
            os.close(fd)

    # ---- the one entry point the loader uses -----------------------------------------------------------------

    def get_or_build(self, name: str, build: Callable[[], grouped.Experts]) -> grouped.Experts:
        t = time.perf_counter()
        ex = self.load(name)
        if ex is not None:
            self.hits += 1
        else:
            publish = not self.disabled and self._export_unchanged() and self.generation() is not None
            if publish and self.builds == 0 and self.tables:          # hashed BEFORE the build reads the sources
                gib = self.geometry.table_bytes() * self.tables / 2 ** 30
                self.notes.append(f"building the cache: {self.tables} tables, about {gib:.1f} GiB under {self.dir}")
            ex = build()
            self.builds += 1
            if publish and self._export_unchanged():          # unchanged across the build: the generation
                self.save(name, ex)                           # describes the bytes the build read
            elif publish:
                self.notes.append(f"{name}: export changed during the build; not cached")
        self.seconds += time.perf_counter() - t
        return ex

    def finish(self) -> str:
        """Write meta.json after builds, log the notes and the summary line; returns the summary ("" when off)."""

        if self.mode == "off":
            return ""
        if self.saved:
            self._write_meta()
        for n in self.notes:
            self._emit(f"[octojet] packed cache: {n}")
        if self.disabled:
            line = f"[octojet] packed tables: cache disabled, {self.builds} built and not saved"
        elif self.builds:
            line = (f"[octojet] packed tables: {self.builds} built, {self.saved} saved, {self.hits} from cache, "
                    f"in {self.seconds:.1f} s ({self.dir})")
        else:
            line = f"[octojet] packed tables: {self.hits} of {self.hits} from cache in {self.seconds:.1f} s"
        self._emit(line)
        return line

    def _emit(self, line: str) -> None:
        try:
            self.log(line)
        except Exception:                         # a failing logger never stops serving
            pass

    # ---- housekeeping ----------------------------------------------------------------------------------------

    def _write_meta(self) -> None:
        """An informational record of what was built from what; never used for validation (the generation in
        each table's metadata is)."""

        meta = {"key": self.key, "inputs": self.inputs, "generation": self.generation(), "full_sha256": self._full,
                "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "octojet_version": _version()}
        try:
            tmp = self.dir / f"meta.json.tmp-{os.getpid()}-{secrets.token_hex(4)}"
            tmp.write_text(json.dumps(meta, indent=1, sort_keys=True))
            os.replace(tmp, self.dir / "meta.json")
        except Exception as e:
            self.notes.append(f"meta.json not written ({type(e).__name__}: {e})")

    def _cleanup_orphans(self) -> None:
        try:
            now = time.time()
            for p in self.dir.glob("*.tmp-*"):
                try:
                    if now - p.stat().st_mtime > ORPHAN_AGE_S:
                        p.unlink()
                except OSError:
                    pass
        except Exception as e:                    # cleanup is optional: construction continues
            self.notes.append(f"orphan cleanup skipped ({type(e).__name__}: {e})")
