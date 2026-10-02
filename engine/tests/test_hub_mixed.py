"""Hub completeness for the self-contained mixed checkpoint (spec 2026-09-29-f2-release-design.md 4.4): the marker,
the experts index and every expert shard it names must be present; experts/ or an expert-less Flash Next base without
the marker is an incomplete download (named so), and a plain MLX layout stays accepted."""

import json
from pathlib import Path

import pytest

from tensorfold import hub
from tests.test_hub_and_checks import fake_repo

ROUTER = "language_model.model.layers.0.mlp.gate.weight"
ROUTED = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
EXPERT = "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"


def base_index(names) -> str:
    return json.dumps({"weight_map": {n: "model-00001-of-00001.safetensors" for n in names}})


def release_snapshot(cache: Path, *, marker=True, expert_shards=("e1.safetensors", "e2.safetensors"),
                     present=("e1.safetensors", "e2.safetensors")) -> Path:
    files = {"config.json": "{}", "model.safetensors.index.json": base_index([ROUTER, "x.weight"]),
             "model-00001-of-00001.safetensors": "w"}
    if marker:
        files["octojet.json"] = json.dumps({"format": "nvfp4-mixed", "experts": "experts", "base": "."})
    snap = fake_repo(cache, "owner/release", files)
    (snap / "experts").mkdir()
    (snap / "experts" / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {f"{EXPERT}.{i}": s for i, s in enumerate(expert_shards)}}))
    for s in present:
        (snap / "experts" / s).write_text("e")
    return snap


def test_complete_release_layout_is_accepted(tmp_path):
    assert hub.incomplete(release_snapshot(tmp_path)) is None


def test_missing_expert_shard_is_incomplete(tmp_path):
    snap = release_snapshot(tmp_path, present=("e1.safetensors",))
    assert "missing expert shards" in hub.incomplete(snap)
    assert not hub._cached_weights_complete(snap)


def test_missing_experts_index_is_incomplete(tmp_path):
    snap = release_snapshot(tmp_path)
    (snap / "experts" / "model.safetensors.index.json").unlink()
    assert "experts/model.safetensors.index.json missing" in hub.incomplete(snap)


def test_experts_dir_without_marker_is_incomplete(tmp_path):
    assert hub.incomplete(release_snapshot(tmp_path, marker=False)) == hub.MIXED_INCOMPLETE
    assert "octojet.json" in hub.MIXED_INCOMPLETE


def test_expert_less_flash_next_base_without_marker_is_incomplete(tmp_path):
    snap = fake_repo(tmp_path, "owner/partial", {"config.json": "{}", "model.safetensors.index.json": base_index([ROUTER]),
                                                 "model-00001-of-00001.safetensors": "w"})
    assert hub.incomplete(snap) == hub.MIXED_INCOMPLETE


def test_plain_mlx_layout_is_accepted(tmp_path):
    snap = fake_repo(tmp_path, "owner/plain", {"config.json": "{}",
                                               "model.safetensors.index.json": base_index([ROUTER, ROUTED]),
                                               "model-00001-of-00001.safetensors": "w"})
    assert hub.incomplete(snap) is None and hub._cached_weights_complete(snap)


def test_resolve_finishes_a_partial_release_and_names_the_marker(tmp_path, monkeypatch):
    snap = release_snapshot(tmp_path, marker=False)
    pulled = []

    def finish(repo_id, *, cache_dir=None):
        pulled.append(repo_id)
        (snap / "octojet.json").write_text(json.dumps({"format": "nvfp4-mixed", "experts": "experts", "base": "."}))
        return snap

    monkeypatch.setattr(hub, "pull", finish)
    monkeypatch.setattr(hub, "cached", lambda repo_id, *, cache_dir=None: snap)    # no huggingface_hub needed
    assert hub.resolve("owner/release", cache_dir=tmp_path) == snap and pulled == ["owner/release"]
    assert hub.resolve("owner/release", cache_dir=tmp_path) == snap and pulled == ["owner/release"]   # now complete


def test_resolve_refuses_a_release_still_without_marker(tmp_path, monkeypatch):
    snap = release_snapshot(tmp_path, marker=False)
    monkeypatch.setattr(hub, "pull", lambda repo_id, *, cache_dir=None: snap)
    monkeypatch.setattr(hub, "cached", lambda repo_id, *, cache_dir=None: snap)
    with pytest.raises(FileNotFoundError, match="without octojet.json"):
        hub.resolve("owner/release", cache_dir=tmp_path)
