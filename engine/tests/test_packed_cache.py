"""The packed-table cache on tiny synthetic checkpoints: key sensitivity, binding, validation, atomic writes,
source generations, verify mode, failure handling. Spec: docs/superpowers/specs/2026-09-29-f2-release-design.md
section 3."""

import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tensorfold.cuda import experts, packed_cache as pc

D, I, E = 64, 32, 3          # hidden 64, width 32, 2 routed + 1 shared expert
GEO = pc.Geometry("nvfp4", 32, I, D, 0.0, E)
BIG = 3 << 20                # a shard whose middle byte lies outside both sampled MiBs


def tiny_table(seed: int = 1) -> experts.Experts:
    g = torch.Generator().manual_seed(seed)

    def proj(n, k):
        ws = [experts.quantize_nvfp4(torch.randn((n, k), generator=g) * 0.05) for _ in range(E)]
        return (torch.stack([w[0] for w in ws]), torch.stack([w[1] for w in ws]), torch.stack([w[2] for w in ws]))

    return experts.make_nvfp4([proj(I, D), proj(I, D)], proj(D, I))


def write_checkpoint(root: Path, *, big: bool = False) -> tuple[Path, Path]:
    """A served dir (base index + one shard + octojet.json) and an experts dir (index + two shards; the second is
    BIG bytes when ``big``)."""
    base, exp = root / "served", root / "export"
    base.mkdir(parents=True); exp.mkdir()
    g = torch.Generator().manual_seed(2)
    save_file({"language_model.model.embed_tokens.weight": torch.randn((8, D), generator=g)}, str(base / "model-00001.safetensors"))
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"language_model.model.embed_tokens.weight": "model-00001.safetensors"}}))
    (base / "octojet.json").write_text(json.dumps({"format": "nvfp4-mixed", "experts": str(exp), "base": str(base)}))
    wm = {}
    for s in (0, 1):
        n = BIG if (big and s == 1) else I * D // 2
        t = {f"model.language_model.layers.0.mlp.experts.{s}.gate_proj.weight": torch.full((n,), 7, dtype=torch.uint8)}
        name = f"layer-00000-experts-{s:04d}-{s:04d}.safetensors"
        save_file(t, str(exp / name))
        wm.update({k: name for k in t})
    (exp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    return base, exp


def keyed(tmp_path, **kw):
    base, exp = write_checkpoint(tmp_path, **kw)
    key, inputs = pc.checkpoint_key(base, exp, GEO)
    return key, inputs, base, exp


def cache_at(tmp_path, key, inputs, **kw):
    return pc.TableCache(tmp_path / "cache", key, inputs, GEO, **kw)


def equal(a: experts.Experts, b: experts.Experts) -> bool:
    return (torch.equal(a.up, b.up) and torch.equal(a.down, b.down)
            and torch.equal(a.gscale_up, b.gscale_up) and torch.equal(a.gscale_down, b.gscale_down)
            and (a.gs, a.width, a.dims, a.limit, a.fmt) == (b.gs, b.width, b.dims, b.limit, b.fmt))


def pin_mtime(path: Path, st: os.stat_result) -> None:
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def flip_middle_byte(shard: Path) -> None:
    st = shard.stat()
    data = bytearray(shard.read_bytes())
    data[len(data) // 2] ^= 0x55
    shard.write_bytes(data)
    pin_mtime(shard, st)


# ---- round trip, metadata, binding ----------------------------------------------------------------------------

def test_round_trip_is_bit_identical(tmp_path):
    key, inputs, *_ = keyed(tmp_path)
    cache = cache_at(tmp_path, key, inputs)
    built = tiny_table()
    calls = []
    got = cache.get_or_build("language_model.model.layers.0.mlp", lambda: (calls.append(1), built)[1])
    assert got is built and cache.builds == 1 and cache.saved == 1 and cache.hits == 0 and calls == [1]
    assert cache.path("language_model.model.layers.0.mlp").name == "language_model-model-layers-0-mlp.safetensors"
    again = cache_at(tmp_path, key, inputs)
    hit = again.get_or_build("language_model.model.layers.0.mlp", lambda: pytest.fail("must not rebuild"))
    assert again.hits == 1 and again.builds == 0 and equal(hit, built)
    assert hit.up.dtype == torch.int32 and hit.gscale_up.dtype == torch.float32


def test_metadata_is_complete_and_strings(tmp_path):
    key, inputs, *_ = keyed(tmp_path)
    cache = cache_at(tmp_path, key, inputs)
    cache.get_or_build("t", tiny_table)
    with safe_open(str(cache.path("t")), framework="pt") as f:
        meta = f.metadata()
    for k in ("pack_version", "checkpoint_key", "table", "rank", "world", "fmt", "gs", "width", "dims", "limit",
              "experts", "built_utc", "octojet_version", "source_generation",
              "sha256_up", "sha256_down", "sha256_gscale_up", "sha256_gscale_down"):
        assert isinstance(meta[k], str) and meta[k], k
    assert meta["checkpoint_key"] == key and meta["table"] == "t" and meta["pack_version"] == str(experts.PACK_VERSION)
    assert meta["source_generation"] == cache.generation()


def _built(tmp_path):
    key, inputs, *_ = keyed(tmp_path)
    cache = cache_at(tmp_path, key, inputs)
    cache.get_or_build("t", tiny_table)
    return key, inputs, cache


def _assert_rebuilt(reader, name="t"):
    rebuilt = []
    reader.get_or_build(name, lambda: (rebuilt.append(1), tiny_table(seed=9))[1])
    assert rebuilt == [1] and reader.builds == 1 and reader.hits == 0
    assert any("rebuilding" in n for n in reader.notes)


def test_other_checkpoint_key_is_rebuilt(tmp_path):
    key, inputs, cache = _built(tmp_path)
    reader = cache_at(tmp_path, "0" * 64, inputs)
    os.replace(cache.path("t"), reader.path("t"))
    _assert_rebuilt(reader)


def test_other_table_name_is_rebuilt(tmp_path):
    key, inputs, cache = _built(tmp_path)
    reader = cache_at(tmp_path, key, inputs)
    os.replace(cache.path("t"), reader.path("u"))
    _assert_rebuilt(reader, "u")


@pytest.mark.parametrize("kw", [{"rank": 1}, {"world": 2}])
def test_other_rank_or_world_is_rebuilt(tmp_path, kw):
    key, inputs, _ = _built(tmp_path)
    _assert_rebuilt(cache_at(tmp_path, key, inputs, **kw))


@pytest.mark.parametrize("geometry", [pc.Geometry("affine", 32, I, D, 0.0, E), pc.Geometry("nvfp4", 32, I, D, 0.0, E + 1),
                                      pc.Geometry("nvfp4", 32, I, D, 7.0, E)])
def test_other_geometry_is_rebuilt(tmp_path, geometry):
    key, inputs, _ = _built(tmp_path)
    _assert_rebuilt(pc.TableCache(tmp_path / "cache", key, inputs, geometry))


def test_other_pack_version_is_rebuilt(tmp_path):
    key, inputs, _ = _built(tmp_path)
    v = experts.PACK_VERSION
    experts.PACK_VERSION = v + 1
    try:
        _assert_rebuilt(cache_at(tmp_path, key, inputs))
    finally:
        experts.PACK_VERSION = v


def test_swapped_layer_files_are_detected(tmp_path):
    key, inputs, *_ = keyed(tmp_path)
    cache = cache_at(tmp_path, key, inputs)
    a, b = tiny_table(1), tiny_table(2)
    cache.get_or_build("layers.0.mlp", lambda: a)
    cache.get_or_build("layers.1.mlp", lambda: b)
    pa, pb, tmp = cache.path("layers.0.mlp"), cache.path("layers.1.mlp"), cache.dir / "swap"
    os.replace(pa, tmp); os.replace(pb, pa); os.replace(tmp, pb)
    reader = cache_at(tmp_path, key, inputs)
    got0 = reader.get_or_build("layers.0.mlp", lambda: a)
    assert reader.builds == 1 and equal(got0, a)


def test_wrong_tensor_set_is_rebuilt(tmp_path):
    key, inputs, cache = _built(tmp_path)
    meta = cache.metadata("t")
    save_file({"up": torch.zeros(GEO.shapes()["up"], dtype=torch.int32)}, str(cache.path("t")), metadata=meta)
    reader = cache_at(tmp_path, key, inputs)
    reader.get_or_build("t", tiny_table)
    assert reader.builds == 1 and any("rebuilding" in n for n in reader.notes)


@pytest.mark.parametrize("what", ["shape", "dtype"])
def test_wrong_shape_or_dtype_is_rebuilt(tmp_path, what):
    key, inputs, cache = _built(tmp_path)
    ex = tiny_table()
    if what == "shape":
        bad = experts.Experts(ex.up[:, :, :1].contiguous(), ex.down, ex.gs, ex.width, ex.dims, ex.limit, fmt=ex.fmt,
                              gscale_up=ex.gscale_up, gscale_down=ex.gscale_down)
    else:
        bad = experts.Experts(ex.up, ex.down, ex.gs, ex.width, ex.dims, ex.limit, fmt=ex.fmt,
                              gscale_up=ex.gscale_up.double(), gscale_down=ex.gscale_down)
    assert cache.save("t", bad)                             # right binding, wrong tensors
    reader = cache_at(tmp_path, key, inputs)
    got = reader.get_or_build("t", lambda: ex)
    assert reader.builds == 1 and equal(got, ex)
    reader2 = cache_at(tmp_path, key, inputs)
    reader2.get_or_build("t", lambda: pytest.fail("the rebuilt file must be valid"))
    assert reader2.hits == 1


# ---- key sensitivity -----------------------------------------------------------------------------------------

def test_key_is_stable_across_directory_rename_and_unreferenced_files(tmp_path):
    key, inputs, base, exp = keyed(tmp_path)
    os.rename(tmp_path / "served", tmp_path / "moved")           # contents and mtimes untouched
    moved = tmp_path / "moved"
    assert pc.checkpoint_key(moved, exp, GEO)[0] == key
    save_file({"junk": torch.zeros(4)}, str(moved / "unreferenced.safetensors"))
    save_file({"junk": torch.zeros(4)}, str(exp / "unreferenced.safetensors"))
    assert pc.checkpoint_key(moved, exp, GEO)[0] == key


def test_key_changes_with_geometry_rank_world(tmp_path):
    key, inputs, base, exp = keyed(tmp_path)
    assert pc.checkpoint_key(base, exp, pc.Geometry("nvfp4", 32, I, D, 0.0, E + 1))[0] != key
    assert pc.checkpoint_key(base, exp, GEO, rank=1, world=2)[0] != key
    assert pc.checkpoint_key(base, exp, GEO, rank=0, world=2)[0] != key


def test_key_changes_with_marker(tmp_path):
    key, inputs, base, exp = keyed(tmp_path)
    (base / "octojet.json").write_text((base / "octojet.json").read_text() + " ")
    assert pc.checkpoint_key(base, exp, GEO)[0] != key


def test_key_changes_with_pack_version(tmp_path):
    key, inputs, base, exp = keyed(tmp_path)
    v = experts.PACK_VERSION
    experts.PACK_VERSION = v + 1
    try:
        assert pc.checkpoint_key(base, exp, GEO)[0] != key
    finally:
        experts.PACK_VERSION = v
    assert pc.checkpoint_key(base, exp, GEO)[0] == key


def test_key_changes_with_shard_header_size_and_index(tmp_path):
    """Each edit with the shard's mtime pinned, so only the edited property moves the key."""
    key, inputs, base, exp = keyed(tmp_path)
    shard = exp / "layer-00000-experts-0000-0000.safetensors"
    st = shard.stat()
    original = shard.read_bytes()

    def write(data: bytes) -> str:
        shard.write_bytes(data)
        pin_mtime(shard, st)
        return pc.checkpoint_key(base, exp, GEO)[0]

    assert write(original) == key
    edited = bytearray(original); edited[10] ^= 0x01               # a header byte
    assert write(bytes(edited)) != key
    assert write(original + b"\0") != key                          # the size
    assert write(original) == key
    idx = exp / "model.safetensors.index.json"
    idx.write_text(idx.read_text().replace("}", " }", 1))         # the index text
    assert pc.checkpoint_key(base, exp, GEO)[0] != key


def test_key_changes_with_head_and_tail_bytes_of_a_big_shard(tmp_path):
    key, inputs, base, exp = keyed(tmp_path, big=True)
    shard = exp / "layer-00000-experts-0001-0001.safetensors"
    assert shard.stat().st_size > 2 * pc.SAMPLE
    st = shard.stat()
    original = shard.read_bytes()
    n = int.from_bytes(original[:8], "little")
    payload = 8 + n

    def write(data: bytes) -> str:
        shard.write_bytes(data)
        pin_mtime(shard, st)
        return pc.checkpoint_key(base, exp, GEO)[0]

    edited = bytearray(original); edited[payload] ^= 0xFF          # inside the first sampled MiB
    assert write(bytes(edited)) != key
    edited = bytearray(original); edited[-1] ^= 0xFF               # inside the last sampled MiB
    assert write(bytes(edited)) != key
    edited = bytearray(original); edited[len(original) // 2] ^= 0xFF   # outside both: the documented limitation
    assert write(bytes(edited)) == key
    assert write(original) == key


def test_key_changes_with_shard_mtime(tmp_path):
    key, inputs, base, exp = keyed(tmp_path)
    shard = exp / "layer-00000-experts-0001-0001.safetensors"
    st = shard.stat()
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    assert pc.checkpoint_key(base, exp, GEO)[0] != key


# ---- source generations and verify mode ----------------------------------------------------------------------

def test_key_misses_a_middle_payload_change_but_verify_mode_catches_it(tmp_path):
    key, inputs, base, exp = keyed(tmp_path, big=True)
    cache = cache_at(tmp_path, key, inputs)
    cache.get_or_build("t", tiny_table)
    old_gen = cache.generation()
    flip_middle_byte(exp / "layer-00000-experts-0001-0001.safetensors")
    key2, inputs2 = pc.checkpoint_key(base, exp, GEO)
    assert key2 == key                                                # the limitation
    plain = cache_at(tmp_path, key2, inputs2)
    plain.get_or_build("t", lambda: pytest.fail("plain mode cannot see a middle-byte change"))
    verify = cache_at(tmp_path, key2, inputs2, mode="verify")
    assert verify.generation() != old_gen
    rebuilt = []
    verify.get_or_build("t", lambda: (rebuilt.append(1), tiny_table())[1])
    assert rebuilt == [1] and any("source generation" in n for n in verify.notes)
    with safe_open(str(verify.path("t")), framework="pt") as f:
        assert f.metadata()["source_generation"] == verify.generation()


def test_verify_rebuilds_only_tables_of_a_stale_generation(tmp_path):
    key, inputs, base, exp = keyed(tmp_path, big=True)
    first = cache_at(tmp_path, key, inputs)
    first.get_or_build("a", tiny_table)                              # a: old generation
    flip_middle_byte(exp / "layer-00000-experts-0001-0001.safetensors")
    key2, inputs2 = pc.checkpoint_key(base, exp, GEO)
    second = cache_at(tmp_path, key2, inputs2)
    second.get_or_build("a", lambda: pytest.fail("plain mode hits a"))   # unvalidated hit
    second.get_or_build("b", tiny_table)                             # b: new generation
    second.finish()
    verify = cache_at(tmp_path, key2, inputs2, mode="verify")
    built = []
    verify.get_or_build("a", lambda: (built.append("a"), tiny_table())[1])
    verify.get_or_build("b", lambda: (built.append("b"), tiny_table())[1])
    assert built == ["a"] and verify.hits == 1 and verify.builds == 1
    assert any("a: source generation" in n for n in verify.notes)
    assert any(n.startswith("b: verified") for n in verify.notes)


def test_verify_mode_detects_cached_payload_corruption(tmp_path):
    key, inputs, cache = _built(tmp_path)
    cache.finish()
    p = cache.path("t")
    data = bytearray(p.read_bytes()); data[-1] ^= 0x01; p.write_bytes(data)
    plain = cache_at(tmp_path, key, inputs)
    plain.get_or_build("t", lambda: pytest.fail("plain mode does not hash payloads"))
    verify = cache_at(tmp_path, key, inputs, mode="verify")
    rebuilt = []
    verify.get_or_build("t", lambda: (rebuilt.append(1), tiny_table())[1])
    assert rebuilt == [1] and any("payload hash" in n for n in verify.notes)


@pytest.mark.parametrize("mode", ["on", "verify"])
def test_incomplete_metadata_is_rebuilt_in_every_mode(tmp_path, mode):
    key, inputs, cache = _built(tmp_path)
    ex = tiny_table()
    meta = {k: v for k, v in cache.metadata("t").items()}             # no source_generation, no payload hashes
    save_file({"up": ex.up, "down": ex.down, "gscale_up": ex.gscale_up, "gscale_down": ex.gscale_down},
              str(cache.path("t")), metadata=meta)
    reader = cache_at(tmp_path, key, inputs, mode=mode)
    _assert_rebuilt(reader)
    assert any("metadata incomplete" in n for n in reader.notes)


def test_build_during_source_change_is_not_cached(tmp_path):
    """The generation is hashed before the build reads the sources; if the export changes while the table is
    built, the table is served but not cached (its generation would not describe its bytes)."""
    key, inputs, base, exp = keyed(tmp_path, big=True)
    cache = cache_at(tmp_path, key, inputs)
    shard = exp / "layer-00000-experts-0001-0001.safetensors"

    def build_and_edit():
        assert cache._generation is not None                     # hashed before the build
        st = shard.stat()
        data = bytearray(shard.read_bytes()); data[len(data) // 2] ^= 0x33; shard.write_bytes(data)
        os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))     # an ordinary edit: the mtime advances
        return tiny_table()

    got = cache.get_or_build("t", build_and_edit)
    assert got is not None and cache.builds == 1 and cache.saved == 0
    assert any("export changed during the build" in n for n in cache.notes) and cache.export_moved
    assert not cache.path("t").exists()


def bump(shard: Path) -> None:
    """An ordinary in-place edit: a middle byte flipped and the mtime advanced by a second."""
    st = shard.stat()
    data = bytearray(shard.read_bytes()); data[len(data) // 2] ^= 0x33; shard.write_bytes(data)
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))


def test_export_change_after_start_stops_publication(tmp_path):
    key, inputs, base, exp = keyed(tmp_path, big=True)
    cache = cache_at(tmp_path, key, inputs)
    cache.get_or_build("a", tiny_table)                          # cached under the key's generation
    bump(exp / "layer-00000-experts-0001-0001.safetensors")
    got = cache.get_or_build("b", tiny_table)                    # built from the moved export
    assert got is not None and cache.saved == 1 and not cache.path("b").exists()
    assert cache.export_moved and any("export changed under this start" in n for n in cache.notes)
    assert "1 saved" in cache.finish()


def test_verify_hit_after_export_change_is_rebuilt(tmp_path):
    key, inputs, base, exp = keyed(tmp_path, big=True)
    cache_at(tmp_path, key, inputs).get_or_build("a", tiny_table)
    verify = cache_at(tmp_path, key, inputs, mode="verify")     # generation frozen here
    bump(exp / "layer-00000-experts-0001-0001.safetensors")
    _assert_rebuilt(verify, "a")                                 # the frozen generation no longer vouches for a
    assert verify.saved == 0 and verify.export_moved


def test_unhashable_sources_disable_publication_but_not_serving(tmp_path):
    key, inputs, base, exp = keyed(tmp_path)
    shard = exp / "layer-00000-experts-0000-0000.safetensors"
    st = shard.stat()
    shard.write_bytes(shard.read_bytes()[:4])                    # a truncated header: struct.error on fingerprinting
    pin_mtime(shard, st)
    cache = cache_at(tmp_path, key, inputs)
    got = cache.get_or_build("t", tiny_table)
    assert got is not None and cache.builds == 1 and cache.saved == 0
    assert cache.export_moved and any("nothing" in n for n in cache.notes)


def test_verify_mode_on_empty_cache_builds(tmp_path):
    key, inputs, *_ = keyed(tmp_path)
    verify = cache_at(tmp_path, key, inputs, mode="verify")
    got = verify.get_or_build("t", tiny_table)
    assert verify.builds == 1 and got is not None
    assert verify.finish().startswith("[octojet] packed tables: 1 built")


def test_finish_writes_meta_only_after_builds(tmp_path):
    key, inputs, *_ = keyed(tmp_path)
    lines = []
    cache = cache_at(tmp_path, key, inputs, log=lines.append)
    cache.get_or_build("a", tiny_table); cache.get_or_build("b", tiny_table)
    line = cache.finish()
    assert line.startswith("[octojet] packed tables: 2 built, 2 saved") and lines[-1] == line
    meta = json.loads((cache.dir / "meta.json").read_text())
    assert meta["key"] == key and meta["generation"] == cache.generation()
    assert set(meta["full_sha256"]) == {"layer-00000-experts-0000-0000.safetensors", "layer-00000-experts-0001-0001.safetensors"}
    before = (cache.dir / "meta.json").stat().st_mtime_ns
    reader = cache_at(tmp_path, key, inputs, log=lines.append)
    reader.get_or_build("a", tiny_table); reader.get_or_build("b", tiny_table)
    assert reader.finish().startswith("[octojet] packed tables: 2 of 2 from cache in ")
    assert (cache.dir / "meta.json").stat().st_mtime_ns == before


# ---- modes, roots, writes, failures ---------------------------------------------------------------------------

def test_off_mode_touches_nothing(tmp_path):
    key, inputs, *_ = keyed(tmp_path)
    cache = cache_at(tmp_path, key, inputs, mode="off")
    got = cache.get_or_build("t", tiny_table)
    assert got is not None and not (tmp_path / "cache").exists() and cache.finish() == ""


def test_parse_option_and_default_root(monkeypatch, tmp_path):
    monkeypatch.delenv("OCTOJET_CACHE_DIR", raising=False)
    assert pc.default_root() == Path.home() / ".cache" / "octojet" / "packed"
    monkeypatch.setenv("OCTOJET_CACHE_DIR", str(tmp_path))
    assert pc.default_root() == tmp_path / "packed"
    assert pc.parse_option(None) == (tmp_path / "packed", "on")
    assert pc.parse_option("off") == (None, "off")
    assert pc.parse_option("verify") == (tmp_path / "packed", "verify")
    assert pc.parse_option(str(tmp_path / "x")) == (tmp_path / "x", "on")
    with pytest.raises(ValueError):
        pc.TableCache(tmp_path, "k", {}, GEO, mode="sometimes")


def test_table_bytes_and_size_note(tmp_path):
    assert GEO.table_bytes() == (E * (1 * 2 * 2 + 2 * 1 * 1) * experts.NVFP4_BLOCK * 4) + E * 3 * 4
    key, inputs, *_ = keyed(tmp_path)
    cache = cache_at(tmp_path, key, inputs, tables=48)
    cache.get_or_build("t", tiny_table)
    assert any("48 tables" in n and "GiB" in n for n in cache.notes)


def test_orphan_tmp_is_removed_only_when_old(tmp_path):
    key, inputs, *_ = keyed(tmp_path)
    d = tmp_path / "cache" / key
    d.mkdir(parents=True)
    old, young = d / "a.safetensors.tmp-1-aa", d / "b.safetensors.tmp-2-bb"
    old.write_bytes(b"x"); young.write_bytes(b"y")
    past = time.time() - 2 * pc.ORPHAN_AGE_S
    os.utime(old, (past, past))
    cache_at(tmp_path, key, inputs)
    assert not old.exists() and young.exists()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
def test_write_failure_is_logged_and_serving_continues(tmp_path):
    key, inputs, *_ = keyed(tmp_path)
    cache = cache_at(tmp_path, key, inputs)
    os.chmod(cache.dir, stat.S_IRUSR | stat.S_IXUSR)
    try:
        got = cache.get_or_build("t", tiny_table)
        assert got is not None and cache.builds == 1 and cache.saved == 0
        assert any("not cached" in n for n in cache.notes)
        assert not list(cache.dir.glob("*.tmp-*"))
        assert "1 built, 0 saved" in cache.finish()
    finally:
        os.chmod(cache.dir, stat.S_IRWXU)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
def test_uncreatable_root_disables_the_cache(tmp_path):
    key, inputs, *_ = keyed(tmp_path)
    blocker = tmp_path / "blocked"
    blocker.mkdir(); os.chmod(blocker, stat.S_IRUSR | stat.S_IXUSR)
    try:
        cache = pc.TableCache(blocker / "cache", key, inputs, GEO)
        assert cache.disabled and any("disabled" in n for n in cache.notes)
        got = cache.get_or_build("t", tiny_table)
        assert got is not None and cache.builds == 1 and cache.saved == 0
        assert cache.finish().startswith("[octojet] packed tables: cache disabled")
    finally:
        os.chmod(blocker, stat.S_IRWXU)


def test_serialization_failure_is_logged_and_serving_continues(tmp_path, monkeypatch):
    key, inputs, *_ = keyed(tmp_path)
    cache = cache_at(tmp_path, key, inputs)

    def boom(*a, **k):
        raise RuntimeError("SafetensorError stand-in")

    monkeypatch.setattr(pc, "save_file", boom)
    got = cache.get_or_build("t", tiny_table)
    assert got is not None and cache.builds == 1 and cache.saved == 0
    assert any("not cached" in n for n in cache.notes) and not list(cache.dir.iterdir())


def test_concurrent_builders_leave_one_valid_file(tmp_path):
    """Four servers miss together (the barrier sits inside the build, so every one of them has already missed),
    all four publish, and one valid file remains."""
    key, inputs, *_ = keyed(tmp_path)
    ex = tiny_table()
    caches = [cache_at(tmp_path, key, inputs) for _ in range(4)]
    barrier = threading.Barrier(4)
    errors = []

    def build():
        barrier.wait(10)
        return ex

    def go(c):
        try:
            c.get_or_build("t", build)
        except Exception as e:                        # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=go, args=(c,)) for c in caches]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == [] and all(c.saved == 1 for c in caches)
    assert sorted(p.name for p in caches[0].dir.iterdir()) == ["t.safetensors"]
    reader = cache_at(tmp_path, key, inputs)
    assert equal(reader.get_or_build("t", lambda: pytest.fail("must hit")), ex)


def test_staggered_builders_both_publish(tmp_path, monkeypatch):
    """A second server starts (and runs orphan cleanup) while the first is mid-write: both files land."""
    key, inputs, *_ = keyed(tmp_path)
    ex = tiny_table()
    first = cache_at(tmp_path, key, inputs)
    mid_write, resume = threading.Event(), threading.Event()
    real_save_file = pc.save_file

    def slow_save(tensors, path, metadata=None):
        real_save_file(tensors, path, metadata=metadata)
        mid_write.set()                                # the temporary exists; the rename has not happened
        assert resume.wait(10)

    monkeypatch.setattr(pc, "save_file", slow_save)
    errors = []

    def writer():
        try:
            first.get_or_build("a", lambda: ex)
        except Exception as e:                        # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=writer); t.start()
    assert mid_write.wait(10)
    monkeypatch.setattr(pc, "save_file", real_save_file)
    second = cache_at(tmp_path, key, inputs)          # constructor: cleanup must leave the young temporary alone
    second.get_or_build("b", lambda: ex)
    resume.set(); t.join(10)
    assert errors == [] and first.saved == 1 and second.saved == 1
    assert sorted(p.name for p in first.dir.iterdir()) == ["a.safetensors", "b.safetensors"]


def test_tensor_sha256_and_experts_to():
    t = torch.arange(16, dtype=torch.int32)
    assert pc.tensor_sha256(t) == pc.tensor_sha256(t.clone()) and pc.tensor_sha256(t) != pc.tensor_sha256(t + 1)
    ex = tiny_table()
    moved = ex.to("cpu")
    assert moved is not ex and equal(moved, ex) and moved.fmt == "nvfp4"


# ---- review fixes: symlinked exports, stat failures, the latch in save(), contained optional failures ----------

def test_export_symlink_retarget_during_build(tmp_path):
    """The cache reads the export through the same logical path as the builder: retargeting a symlinked export
    during the build moves the post-build fingerprint, so nothing is published under the old generation."""
    base, exp_a = write_checkpoint(tmp_path / "a")
    _, exp_b = write_checkpoint(tmp_path / "b", big=True)          # different shard bytes (and size)
    link = tmp_path / "export-link"
    link.symlink_to(exp_a, target_is_directory=True)
    key, inputs = pc.checkpoint_key(base, link, GEO)
    assert inputs["experts_root"] == os.path.abspath(link)
    cache = cache_at(tmp_path, key, inputs)
    built = tiny_table()

    def build_and_retarget():
        link.unlink()
        link.symlink_to(exp_b, target_is_directory=True)
        return built

    got = cache.get_or_build("t", build_and_retarget)
    assert got is built and cache.builds == 1 and cache.saved == 0 and cache.export_moved
    assert not cache.path("t").exists()


def test_cache_stat_failure_returns_build(tmp_path, monkeypatch):
    key, inputs, *_ = keyed(tmp_path)
    cache = cache_at(tmp_path, key, inputs)
    target = cache.path("t")
    real_is_file = Path.is_file

    def is_file(self):
        if self == target:
            raise PermissionError("stat denied")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", is_file)
    built = tiny_table()
    got = cache.get_or_build("t", lambda: built)
    assert got is built and cache.builds == 1
    assert any("stat denied" in n and "rebuilding" in n for n in cache.notes)


def test_save_after_export_moved_is_rejected(tmp_path):
    key, inputs, cache = _built(tmp_path)
    before = cache.path("t").read_bytes()
    saved = cache.saved
    cache.export_moved = True
    assert cache.save("t", tiny_table(seed=9)) is False
    assert cache.save("u", tiny_table(seed=9)) is False
    assert cache.saved == saved and cache.path("t").read_bytes() == before and not cache.path("u").exists()
    assert not list(cache.dir.glob("*.tmp-*"))


def test_optional_cache_failures_are_contained(tmp_path, monkeypatch):
    key, inputs, *_ = keyed(tmp_path)
    built = tiny_table()

    def boom(*a, **k):
        raise OSError("injected")

    # temporary naming in save(): the built table is still returned
    with monkeypatch.context() as m:
        m.setattr(pc.secrets, "token_hex", boom)
        cache = cache_at(tmp_path, key, inputs, log=lambda line: None)
        assert cache.get_or_build("t", lambda: built) is built
        assert cache.saved == 0 and any("t: not cached" in n for n in cache.notes)
        assert isinstance(cache.finish(), str)

    # meta.json naming in finish(): a table was saved, then naming fails
    cache = cache_at(tmp_path, key, inputs, log=lambda line: None)
    assert cache.get_or_build("m", lambda: built) is built and cache.saved == 1
    with monkeypatch.context() as m:
        m.setattr(pc.secrets, "token_hex", boom)
        assert isinstance(cache.finish(), str)
    assert any("meta.json not written" in n for n in cache.notes)

    # a logger that raises
    def bad_log(line):
        raise RuntimeError("log sink down")

    cache = cache_at(tmp_path, key, inputs, log=bad_log)
    assert cache.get_or_build("n", lambda: built) is built
    assert cache.finish().startswith("[octojet] packed tables:")

    # orphan cleanup's glob during construction
    with monkeypatch.context() as m:
        m.setattr(Path, "glob", boom)
        cache = cache_at(tmp_path, key, inputs, log=lambda line: None)
    assert any("orphan cleanup skipped" in n for n in cache.notes)
    assert cache.get_or_build("o", lambda: built) is built
    assert isinstance(cache.finish(), str)

    # builder exceptions still propagate
    with pytest.raises(ValueError):
        cache_at(tmp_path, key, inputs).get_or_build("p", lambda: (_ for _ in ()).throw(ValueError("builder")))


# ---- fix round 2: explicit sequential reads (no mmap page faults during the upload) ---------------------------

def test_load_reads_sequentially_without_mmap(tmp_path, monkeypatch):
    import safetensors

    def no_mmap(*a, **k):
        raise AssertionError("safe_open must not be used on the read path")

    monkeypatch.setattr(pc, "safe_open", no_mmap, raising=False)
    monkeypatch.setattr(safetensors, "safe_open", no_mmap)
    key, inputs, *_ = keyed(tmp_path)
    built = tiny_table()
    cache_at(tmp_path, key, inputs).get_or_build("t", lambda: built)
    reader = cache_at(tmp_path, key, inputs)
    hit = reader.get_or_build("t", lambda: pytest.fail("must hit"))
    assert reader.hits == 1 and equal(hit, built)
    assert all(t.is_contiguous() for t in (hit.up, hit.down, hit.gscale_up, hit.gscale_down))


class _CountingFile:
    """A file wrapper counting the bytes read through read() and readinto()."""

    def __init__(self, f, counter):
        self._f, self._counter = f, counter

    def read(self, *a):
        data = self._f.read(*a)
        self._counter[0] += len(data)
        return data

    def readinto(self, b):
        got = self._f.readinto(b)
        self._counter[0] += got or 0
        return got

    def __getattr__(self, name):
        return getattr(self._f, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()


def test_bad_header_reads_no_payload(tmp_path, monkeypatch):
    import builtins

    key, inputs, cache = _built(tmp_path)
    reader = cache_at(tmp_path, "0" * 64, inputs)                  # the file binds to another key
    os.replace(cache.path("t"), reader.path("t"))
    target = reader.path("t")
    raw = target.read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    assert len(raw) > 8 + header_len + 1                           # there is a payload to (not) read
    counter, real_open = [0], builtins.open

    def counting_open(file, *a, **k):
        f = real_open(file, *a, **k)
        return _CountingFile(f, counter) if Path(os.fsdecode(file)) == target else f

    monkeypatch.setattr(builtins, "open", counting_open)
    assert reader.load("t") is None
    monkeypatch.setattr(builtins, "open", real_open)
    assert 0 < counter[0] < 8 + header_len + 1
    assert any("metadata differs" in n and "rebuilding" in n for n in reader.notes)


def test_read_chunking(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "READ_CHUNK", 4096)
    key, inputs, *_ = keyed(tmp_path)
    built = tiny_table()
    assert built.up.numel() * 4 > pc.READ_CHUNK                    # the up tensor spans several chunks
    cache_at(tmp_path, key, inputs).get_or_build("t", lambda: built)
    reader = cache_at(tmp_path, key, inputs)
    assert equal(reader.get_or_build("t", lambda: pytest.fail("must hit")), built) and reader.hits == 1


# ---- fix round 3: the payload layout is validated as a whole ---------------------------------------------------

def _rewrite(path: Path, edit_header, edit_payload=lambda b: b) -> None:
    """Rewrite a safetensors file with an edited header (re-padded to a multiple of 8 so it stays parseable) and an
    optionally edited payload. Offsets are relative to the payload, so a new header length keeps them meaningful."""
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    edit_header(header)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    path.write_bytes(len(text).to_bytes(8, "little") + text + edit_payload(raw[8 + n:]))


def _assert_layout_rebuild(tmp_path, key, inputs, note="t: payload layout invalid; rebuilding"):
    reader = cache_at(tmp_path, key, inputs)
    rebuilt = []
    got = reader.get_or_build("t", lambda: (rebuilt.append(1), tiny_table(seed=9))[1])
    assert rebuilt == [1] and reader.hits == 0 and equal(got, tiny_table(seed=9))
    assert any(n == note for n in reader.notes)


def test_overlapping_spans_are_rebuilt(tmp_path):
    key, inputs, cache = _built(tmp_path)

    def down_inside_up(h):
        ub = h["up"]["data_offsets"][0]
        db, de = h["down"]["data_offsets"]
        h["down"]["data_offsets"] = [ub, ub + (de - db)]           # the right length, inside up's span

    _rewrite(cache.path("t"), down_inside_up)
    _assert_layout_rebuild(tmp_path, key, inputs)


@pytest.mark.parametrize("fault", ["gap", "truncated"])
def test_gap_or_short_payload_is_rebuilt(tmp_path, fault):
    key, inputs, cache = _built(tmp_path)
    if fault == "gap":
        def shift_last(h):
            last = max(pc.TENSORS, key=lambda k: h[k]["data_offsets"][0])
            b, e = h[last]["data_offsets"]
            h[last]["data_offsets"] = [b + 4, e + 4]              # 4 unowned bytes before the last tensor

        def widen(payload):
            return payload + b"\0" * 4                             # the file still ends at the last span's end
        _rewrite(cache.path("t"), shift_last, widen)
    else:
        p = cache.path("t")
        p.write_bytes(p.read_bytes()[:-1])
    # the payload's size no longer equals the geometry's table bytes: rejected before the header is read
    _assert_layout_rebuild(tmp_path, key, inputs, note="t: header invalid; rebuilding")


# ---- final fix wave: cache setup failures, bounded header reads ------------------------------------------------

def test_checkpoint_key_raises_on_unreadable_shard(tmp_path):
    """The key fails loudly on a shard it cannot fingerprint (base or experts); the loader's cache-setup boundary
    catches that and continues uncached."""
    base, exp = write_checkpoint(tmp_path)
    (base / "model-00001.safetensors").unlink()                    # a missing base shard (unused by the experts)
    with pytest.raises(OSError):
        pc.checkpoint_key(base, exp, GEO)
    base2, exp2 = write_checkpoint(tmp_path / "b")
    shard = exp2 / "layer-00000-experts-0000-0000.safetensors"
    shard.write_bytes(shard.read_bytes()[:4])                      # a truncated header: struct.error
    with pytest.raises(Exception):
        pc.checkpoint_key(base2, exp2, GEO)


def _counted_load(reader, target, monkeypatch):
    import builtins

    counter, real_open = [0], builtins.open

    def counting_open(file, *a, **k):
        f = real_open(file, *a, **k)
        return _CountingFile(f, counter) if Path(os.fsdecode(file)) == target else f

    monkeypatch.setattr(builtins, "open", counting_open)
    try:
        got = reader.load("t")
    finally:
        monkeypatch.setattr(builtins, "open", real_open)
    return got, counter[0]


@pytest.mark.parametrize("length", ["huge", "within_file"])
def test_oversized_header_length_is_rejected_without_reading_it(tmp_path, monkeypatch, length):
    key, inputs, cache = _built(tmp_path)
    target = cache.path("t")
    raw = target.read_bytes()
    if length == "huge":
        n = 1 << 40
    else:                                                          # inside the file, above a lowered cap
        monkeypatch.setattr(pc, "MAX_HEADER", 64)
        n = len(raw) - 8 - 4
        assert n > pc.MAX_HEADER
    target.write_bytes(n.to_bytes(8, "little") + raw[8:])
    reader = cache_at(tmp_path, key, inputs)
    got, read = _counted_load(reader, target, monkeypatch)
    assert got is None and read < 8 + pc.MAX_HEADER
    assert "t: header invalid; rebuilding" in reader.notes


def test_wrong_payload_size_is_rejected_before_header_read(tmp_path, monkeypatch):
    key, inputs, cache = _built(tmp_path)
    target = cache.path("t")
    target.write_bytes(target.read_bytes() + b"\0" * 4)
    reader = cache_at(tmp_path, key, inputs)
    got, read = _counted_load(reader, target, monkeypatch)
    assert got is None and read == 8                               # only the length word
    assert "t: header invalid; rebuilding" in reader.notes
    _assert_layout_rebuild(tmp_path, key, inputs, note="t: header invalid; rebuilding")
