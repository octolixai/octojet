"""Octojet's product layer: CLI name, version, the per-reply stats key, owned_by and the log prefix.

The Python package stays ``tensorfold.*`` and env vars stay ``TENSORFOLD_*``; only what users see is renamed.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import tensorfold
from tensorfold import cli, update

ENGINE = Path(__file__).resolve().parents[1]
REPO = ENGINE.parent
SRC = ENGINE / "src" / "tensorfold"


def test_cli_prog_and_version():
    parser = cli.build_parser()
    assert parser.prog == "octojet"
    assert tensorfold.__version__ == "0.1.0"
    assert tensorfold.UPSTREAM == "TensorFold 0.3.6.2 (71377a5)"
    snapshot = next(a for a in parser._subparsers._group_actions[0].choices["serve"]._actions
                    if a.dest == "snapshot_dir")
    assert Path(snapshot.default).parts[-3:] == (".cache", "octojet", "prefix-snapshots")


def test_pyproject_names_the_octojet_script_only():
    project = tomllib.loads((ENGINE / "pyproject.toml").read_text())["project"]
    assert project["name"] == "octojet"
    assert project["scripts"] == {"octojet": "tensorfold.cli:main"}
    assert project["authors"] == [{"name": "Octolix"}]
    assert project["urls"]["Source"] == "https://github.com/octolixai/octojet"


def test_both_servers_use_the_octojet_stats_key_and_owned_by():
    cuda = (SRC / "cuda" / "server.py").read_text()
    mlx = (SRC / "server" / "http.py").read_text()
    for text in (cuda, mlx):
        assert '"owned_by": "octojet"' in text
        assert not re.search(r"""\[["']tensorfold["']\]|["']tensorfold["']\s*:""", text)
    assert '"octojet": result["stats"]' in cuda and 'end["octojet"]' in cuda
    assert 'extras["octojet"]' in mlx


def test_first_run_notice_names_octojet_not_upstream_releases(monkeypatch, tmp_path):
    monkeypatch.delenv("TENSORFOLD_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(update, "SEEN", tmp_path / "version-seen")
    line = update.first_run_notice()
    assert line.startswith("[octojet] this is Octojet 0.1.0")
    assert "octolixai/octojet" in line and "ashhart" not in line
    assert "octolixai/octojet" in update.FORK_NOTICE


def test_no_tensorfold_log_prefix_remains():
    found = []
    for root in (ENGINE / "src", ENGINE / "tools", REPO / "bench"):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh", ".md", ".json"} and "egg-info" not in str(path):
                if "[tensorfold]" in path.read_text(errors="replace"):
                    found.append(str(path.relative_to(REPO)))
    assert found == []
