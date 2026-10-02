"""tools/check_mixed_checkpoint.py on synthetic sources: reference arguments (required for a relative marker, defaulted
from an absolute one), --layers, and --source-bytes on the release and symlink layouts (spec 4.3). The numeric checks
need the CUDA stack and are skipped with --no-numeric."""

import json
import shutil
from pathlib import Path

import pytest

from tensorfold.families.qwen4_exp import release as R
from tests.release_fakes import load_tool, symlink_layout, write_export, write_mlx

check = load_tool("check_mixed_checkpoint")
build = load_tool("build_release_checkpoint")


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("chk")
    mlx, export = root / "mlx", root / "export"
    write_mlx(mlx)
    write_export(export)
    build.build(["--mlx", str(mlx), "--nvfp4", str(export), "--out", str(root / "release")])
    symlink_layout(mlx, export, root / "links")
    return root


def run(capsys, *argv):
    rc = check.main([str(a) for a in argv])
    return rc, json.loads(capsys.readouterr().out)


def test_relative_marker_requires_references(dirs, capsys):
    with pytest.raises(SystemExit) as e:
        check.main([str(dirs / "release"), "--no-numeric"])
    assert e.value.code == 2 and "--reference-mlx" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        check.main([str(dirs / "release"), "--reference-mlx", str(dirs / "mlx"), "--no-numeric"])


def test_absolute_marker_defaults_references(dirs):
    mlx, export, relative = check.references(dirs / "links", None, None)
    assert (mlx, export, relative) == ((dirs / "mlx").resolve(), (dirs / "export").resolve(), False)
    mlx, export, relative = check.references(dirs / "release", dirs / "mlx", dirs / "export")
    assert (mlx, export, relative) == (dirs / "mlx", dirs / "export", True)
    with pytest.raises(ValueError, match="no model.safetensors.index.json"):
        check.references(dirs / "release", dirs, dirs / "export")


def test_layers_argument():
    assert check.layer_list("all", 3) == [0, 1, 2]
    assert check.layer_list("0,2", 3) == [0, 2]
    for bad in ("3", "x", "", "-1"):
        with pytest.raises(ValueError, match="--layers"):
            check.layer_list(bad, 3)


def test_source_bytes_release_layout_passes(dirs, capsys):
    rc, rep = run(capsys, dirs / "release", "--reference-mlx", dirs / "mlx", "--reference-nvfp4", dirs / "export",
                  "--layers", "all", "--source-bytes", "--no-numeric")
    sb = rep["source_bytes"]
    assert rc == 0 and rep["ok"] and sb["layout"] == "release" and rep["layers"] == "all"
    assert sb["base"]["checked"] > 0 and sb["base"]["mismatched"] == 0 and sb["experts"]["mismatched"] == 0
    assert sb["dropped"]["count"] > 0 and sb["dropped"]["present"] == 0
    assert sb["kept"]["missing"] == 0 and sb["needed_experts"]["missing"] == 0


def test_source_bytes_symlink_layout_passes(dirs, capsys):
    rc, rep = run(capsys, dirs / "links", "--source-bytes", "--no-numeric", "--layers", "all")
    assert rc == 0 and rep["source_bytes"]["layout"] == "symlink" and "dropped" not in rep["source_bytes"]


def corrupt_one_byte(path: Path, name: str) -> None:
    base, head = R.read_header(path)
    at = base + head[name]["data_offsets"][0]
    with open(path, "r+b") as f:
        f.seek(at)
        b = f.read(1)
        f.seek(at)
        f.write(bytes([b[0] ^ 0xFF]))


def test_source_bytes_catches_a_changed_payload_and_a_missing_name(dirs, tmp_path, capsys):
    out = tmp_path / "release"
    shutil.copytree(dirs / "release", out)
    name = "language_model.model.layers.0.mlp.gate.weight"
    corrupt_one_byte(out / R.read_index(out)[name], name)
    idx = json.loads((out / "experts" / R.INDEX).read_text())
    gone = "model.language_model.layers.1.mlp.experts.3.down_proj.weight_scale"
    del idx["weight_map"][gone]
    (out / "experts" / R.INDEX).write_text(json.dumps(idx))
    rc, rep = run(capsys, out, "--reference-mlx", dirs / "mlx", "--reference-nvfp4", dirs / "export",
                  "--source-bytes", "--no-numeric", "--layers", "0")
    sb = rep["source_bytes"]
    assert rc == 1 and not rep["ok"]
    assert sb["base"]["mismatched_names"] == [name]
    assert sb["needed_experts"]["missing_names"] == [gone]


def test_source_bytes_catches_a_kept_dropped_name(dirs, tmp_path, capsys):
    """A base that still lists a decoder shared-expert projection (but no switch_mlp) is not a clean release."""

    out = tmp_path / "release"
    shutil.copytree(dirs / "release", out)
    idx = json.loads((out / R.INDEX).read_text())
    src = R.read_index(dirs / "mlx")
    extra = "language_model.model.layers.0.mlp.shared_expert.up_proj.scales"
    shard = sorted(set(idx["weight_map"].values()))[0]
    # append the reference tensor to a shard of the release (header rewritten, payload copied)
    base, head = R.read_header(out / shard)
    tensors = [(n, out / shard, base, i) for n, i in head.items() if n != "__metadata__"]
    rbase, rhead = R.read_header(dirs / "mlx" / src[extra])
    tensors.append((extra, dirs / "mlx" / src[extra], rbase, rhead[extra]))
    tmp = tmp_path / "s.safetensors"
    R.write_shard(tmp, tensors, head.get("__metadata__"))
    shutil.move(tmp, out / shard)
    idx["weight_map"][extra] = shard
    (out / R.INDEX).write_text(json.dumps(idx))
    rc, rep = run(capsys, out, "--reference-mlx", dirs / "mlx", "--reference-nvfp4", dirs / "export",
                  "--source-bytes", "--no-numeric", "--layers", "1")
    assert rc == 1 and rep["source_bytes"]["dropped"]["present_names"] == [extra]
    assert rep["source_bytes"]["base"]["mismatched"] == 0
