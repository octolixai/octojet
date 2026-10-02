"""The self-contained mixed checkpoint (spec 2026-09-29-f2-release-design.md section 4): tools/build_release_checkpoint.py
on synthetic two-shard sources, the engine's view of the result (nvfp4.sources, release_expert_files) and the
startup estimate of the release layout against the local symlink layout of the same sources."""

import json
from pathlib import Path

import pytest

from tensorfold import __version__
from tensorfold.families.qwen4_exp import release as R
from tensorfold.families.qwen4_exp.cuda import nvfp4
from tests.release_fakes import E, LAYERS, load_tool, symlink_layout, write_export, write_mlx

build_tool = load_tool("build_release_checkpoint")


def payloads(root: Path) -> dict[str, tuple[str, list, str]]:
    """name -> (dtype, shape, payload sha256) for every tensor an index lists."""

    out = {}
    for name, shard in R.read_index(root).items():
        head = R.read_header(root / shard)
        info = head[1][name]
        out[name] = (info["dtype"], info["shape"], R.tensor_sha256(root / shard, name, head))
    return out


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("src")
    write_mlx(root / "mlx")
    write_export(root / "export")
    return root / "mlx", root / "export"


@pytest.fixture(scope="module")
def built(sources, tmp_path_factory):
    mlx, export = sources
    out = tmp_path_factory.mktemp("out") / "release"
    report = build_tool.build(["--mlx", str(mlx), "--nvfp4", str(export), "--out", str(out),
                               "--max-shard-gib", str(48 * 1024 / 2**30), "--hash", "--mlx-revision", "abc123"])
    return out, report


# ---- nvfp4.sources / release_expert_files ------------------------------------------------------------------------------

def test_sources_relative_and_absolute(tmp_path):
    out = tmp_path / "served"
    out.mkdir()
    (out / nvfp4.MARKER).write_text(json.dumps({"format": "nvfp4-mixed", "experts": "experts", "base": "."}))
    src = nvfp4.sources(out)
    assert src.experts == out / "experts" and src.base == out
    (out / nvfp4.MARKER).write_text(json.dumps({"format": "nvfp4-mixed", "experts": "/abs/export", "base": "/abs/mlx"}))
    src = nvfp4.sources(out)
    assert src.experts == Path("/abs/export") and src.base == Path("/abs/mlx")
    (out / nvfp4.MARKER).write_text(json.dumps({"format": "other", "experts": "e", "base": "."}))
    with pytest.raises(ValueError, match="format"):
        nvfp4.sources(out)


def test_release_expert_files_per_layout(sources, built, tmp_path):
    mlx, export = sources
    out, _ = built
    assert nvfp4.release_expert_files(symlink_layout(mlx, export, tmp_path / "links")) == ()   # MLX headers stand in
    assert nvfp4.release_expert_files(mlx) == ()                                               # not mixed at all
    files = nvfp4.release_expert_files(out)
    assert files and all(f.parent == out / "experts" and f.is_file() for f in files)
    assert {f.name for f in files} == set(R.read_index(out / "experts").values())


def test_estimate_transform_counts_export_tensors():
    t = nvfp4.estimate_transform(lambda name, info: (999, 0))
    p = "model.language_model.layers.3.mlp"
    assert t(f"{p}.experts.5.up_proj.weight", {"shape": [640, 1280], "dtype": "U8"}) == (640 * 1280, 0)
    assert t(f"{p}.experts.5.up_proj.weight_scale", {"shape": [640, 160], "dtype": "F8_E4M3"}) == (640 * 160, 0)
    assert t(f"{p}.experts.5.up_proj.weight_scale_2", {"shape": [], "dtype": "F32"}) == (4, 0)
    assert t(f"{p}.experts.5.up_proj.input_scale", {"shape": [], "dtype": "F32"}) == (0, 0)
    assert t(f"{p}.shared_expert.down_proj.weight", {"shape": [2560, 640], "dtype": "BF16"}) == (2560 * 640 * 144 // 256, 0)
    assert t(f"{p}.gate.weight", {"shape": [512, 2560], "dtype": "BF16"}) == (0, 0)


def test_release_estimate_matches_symlink_layout(sources, built, tmp_path):
    from tensorfold.cuda.capacity import estimate_weights
    from tensorfold.cuda.geometry import indexed_weights

    mlx, export = sources
    out, _ = built
    links = symlink_layout(mlx, export, tmp_path / "links")
    transform = nvfp4.estimate_transform(indexed_weights(1, True))

    def total(d):
        w = estimate_weights(d, transform)
        extra = nvfp4.release_expert_files(d)
        if extra:
            x = estimate_weights(d, transform, files=list(extra))
            return w.resident + x.resident, max(w.staging, x.staging), w.mapped + x.mapped
        return w.resident, w.staging, w.mapped

    sym, rel = total(links), total(out)
    assert abs(rel[0] - sym[0]) <= 0.01 * sym[0], (rel, sym)
    assert rel[2] == sym[2] > 0                                    # the mapped n-gram tables, unchanged


# ---- the builder -------------------------------------------------------------------------------------------------------

def test_layout_marker_and_files(built, sources):
    out, report = built
    marker = json.loads((out / "octojet.json").read_text())
    assert marker["format"] == "nvfp4-mixed" and marker["experts"] == "experts" and marker["base"] == "."
    assert marker["sources"]["base"] == {"repo": R.REPOS["base"], "revision": "abc123"}
    assert marker["sources"]["experts"] == {"repo": R.REPOS["experts"], "revision": "unknown"}
    assert marker["built_utc"].endswith("Z") and marker["octojet"].startswith(__version__)
    names = {p.name for p in out.iterdir()}
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        assert f in names
    assert not names & {"README.md", "LICENSE", "NOTICE", ".gitattributes"}
    assert nvfp4.is_mixed(out)
    src = nvfp4.sources(out)
    assert src.experts == out / "experts" and src.base == out
    assert Path(report["report"]).is_file() and Path(report["report"]).parent == out.parent
    assert "/" not in json.dumps(marker["sources"]).replace("RadixArk/", "").replace("Vontra/", "")


def test_base_drops_decoder_experts_keeps_rest_byte_identical(built, sources):
    out, report = built
    mlx, _ = sources
    want, got = payloads(mlx), payloads(out)
    dropped = {n for n in want if R.dropped_from_base(n)}
    assert dropped and not dropped & set(got)
    assert all(R.DECODER_ROUTED.match(n) or ".shared_expert." in n for n in dropped)
    assert set(got) == set(want) - dropped
    assert all(got[n] == want[n] for n in got)                     # dtype, shape and payload bytes unchanged
    for keep in ("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
                 "language_model.mtp.layers.0.mlp.shared_expert.down_proj.biases",
                 "language_model.model.layers.1.mlp.shared_expert_gate.scales",
                 "language_model.model.layers.0.mlp.gate.weight",
                 "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.biases"):
        assert keep in got, keep
    idx = json.loads((out / R.INDEX).read_text())
    assert idx["metadata"]["total_size"] == report["base"]["total_size"]
    assert idx["metadata"]["total_size"] == sum(R.payload_size(R.read_header(out / s)[1][n]) for n, s in idx["weight_map"].items())
    assert report["base"]["dropped_tensors"] == len(dropped)


def test_base_reshards_within_limit_without_splitting_groups(built):
    out, report = built
    where = R.read_index(out)
    shards = sorted(set(where.values()))
    n = len(shards)
    assert n >= 3 and shards == [f"model-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]
    limit = 48 * 1024
    groups: dict[str, set] = {}
    for name, shard in where.items():
        groups.setdefault(R.group_of(name), set()).add(shard)
    assert all(len(s) == 1 for s in groups.values()), {g: s for g, s in groups.items() if len(s) > 1}
    # the embedding's weight and its scales/biases came from different source shards; they now share one
    assert len(groups["language_model.model.embed_tokens"]) == 1
    for s in shards:
        _, head = R.read_header(out / s)
        assert head["__metadata__"] == {"format": "mlx"}
        oversize = (out / s).stat().st_size > limit
        assert not oversize or len({R.group_of(k) for k in head if k != "__metadata__"}) == 1


def test_group_at_boundary_moves_whole(tmp_path):
    """A 4 KiB group, then a 3 x 2 KiB group whose weight alone would still fit the 10 KiB shard: it moves whole."""

    import torch
    from safetensors.torch import save_file

    root = tmp_path / "mlx"
    root.mkdir()
    kib = lambda n: torch.zeros((n * 512,), dtype=torch.bfloat16)  # noqa: E731
    t = {"a.weight": kib(4), "b.weight": kib(2), "b.scales": kib(2), "b.biases": kib(2), "c": kib(1)}
    save_file(t, str(root / "m.safetensors"), metadata={"format": "mlx"})
    (root / R.INDEX).write_text(json.dumps({"weight_map": {k: "m.safetensors" for k in t}}))
    plan, dropped, kept, notes = build_tool.plan_base(root, 10 * 1024 + 4096)
    names = [{x[0] for x in tensors} for tensors, _ in plan]
    assert names == [{"a.weight"}, {"b.weight", "b.scales", "b.biases", "c"}] and not notes


def test_experts_index_maps_every_needed_name(built, sources):
    out, report = built
    _, export = sources
    want = payloads(export)
    needed = {n for n in want if R.needed_from_export(n)}
    assert len(needed) == LAYERS * (E * 9 + 3)
    got = payloads(out / "experts")
    assert set(got) == needed and all(got[n] == want[n] for n in got)
    assert not any(n.endswith("input_scale") for n in got)
    rows = {r["shard"]: r for r in report["experts"]["shards"]}
    assert all(r["mode"] == "copy" and r["identical"] for s, r in rows.items() if s.startswith("layer-"))
    assert report["experts"]["wasted_bytes"] > 0                     # the input scales copied with their shards
    idx = json.loads((out / "experts" / R.INDEX).read_text())
    assert idx["metadata"]["total_size"] == report["experts"]["total_size"]


def test_export_shard_with_big_waste_is_rewritten(tmp_path, monkeypatch, sources):
    mlx, export = sources
    monkeypatch.setattr(build_tool, "WHOLE_COPY_WASTE", 64 * 1024)   # the bf16 shard's junk embedding exceeds this
    out = tmp_path / "rel"
    report = build_tool.build(["--mlx", str(mlx), "--nvfp4", str(export), "--out", str(out), "--hash"])
    rows = {r["shard"]: r for r in report["experts"]["shards"]}
    bf16 = rows["model-bf16-00001.safetensors"]
    assert bf16["mode"] == "rewrite" and not bf16["identical"]
    _, head = R.read_header(out / "experts" / "model-bf16-00001.safetensors")
    assert head["__metadata__"] == {"format": "pt"}
    assert set(head) - {"__metadata__"} == {n for n in R.read_index(out / "experts") if ".shared_expert." in n}
    want = payloads(export)
    assert all(v == want[n] for n, v in payloads(out / "experts").items())


def test_fp8_export_keeps_shared_scales(tmp_path, sources):
    mlx, _ = sources
    write_export(tmp_path / "fp8", shared_fp8=True)
    out = tmp_path / "rel"
    build_tool.build(["--mlx", str(mlx), "--nvfp4", str(tmp_path / "fp8"), "--out", str(out)])
    names = set(R.read_index(out / "experts"))
    assert "model.language_model.layers.1.mlp.shared_expert.up_proj.weight_scale" in names
    assert "model.language_model.layers.1.mlp.shared_expert.up_proj.input_scale" not in names


def test_refusals(tmp_path, sources):
    mlx, export = sources
    out = tmp_path / "busy"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        build_tool.build(["--mlx", str(mlx), "--nvfp4", str(export), "--out", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["x"]
    with pytest.raises(SystemExit, match="inside a source"):
        build_tool.build(["--mlx", str(mlx), "--nvfp4", str(export), "--out", str(mlx / "sub")])
    with pytest.raises(SystemExit, match="no model.safetensors.index.json"):
        build_tool.build(["--mlx", str(tmp_path), "--nvfp4", str(export), "--out", str(tmp_path / "o1")])
    short = tmp_path / "short"
    write_export(short, experts=E // 2)                              # the config wants E experts a layer
    with pytest.raises(SystemExit, match="lacks"):
        build_tool.build(["--mlx", str(mlx), "--nvfp4", str(short), "--out", str(tmp_path / "o2")])
    assert not (tmp_path / "o2").exists()


def test_revision_from_snapshot_path(tmp_path):
    snap = tmp_path / "models--x" / "snapshots" / ("7b71" + "0" * 36)
    snap.mkdir(parents=True)
    assert build_tool._revision(snap, None) == "7b71" + "0" * 36
    assert build_tool._revision(snap, "given") == "given"
    assert build_tool._revision(tmp_path, None) == "unknown"
