"""engine/tools/release_check.py: leaks in tracked files and missing release files are findings (exit 1).

Sensitive-looking strings are assembled at run time so this file stays clean under the check itself.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "release_check.py"
spec = importlib.util.spec_from_file_location("release_check", TOOL)
release_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_check)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

HOME = "/" + "home" + "/alice"
MAC = "/" + "Users" + "/alice"
PRIVATE_IP = "192." + "168.1.50"
HF = "hf" + "_" + "A" * 24
SK = "sk" + "-" + "b" * 24
SOCKET = "/tmp/cc-socks/42" + ".sock"
UDS = "uds" + ":/run/operator"
SESSION = "session" + "_" + "0" * 24
EMAIL = "someone" + "@" + "corp.io"


def make_repo(tmp_path: Path, files: dict[str, str], *, release_files: bool = True) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    base = {}
    if release_files:
        base = {
            "LICENSE": "MIT License\n",
            "engine/LICENSE": "MIT License\n",
            "engine/THIRD_PARTY_NOTICES.md": "# Third-party notices\n",
            "engine/CHANGELOG.md": "# What's new\n\n## 0.1.0 (October 2026)\n\n- first\n",
            "engine/src/tensorfold/__init__.py": '__version__ = "0.1.0"\n',
            "docs/model-card.md": "# Card\n\n## License\n\nQwen.\n",
        }
    for name, text in {**base, **files}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return root


def run(root: Path, capsys) -> tuple[int, str]:
    code = release_check.main(["--root", str(root)])
    return code, capsys.readouterr().out


def test_a_clean_repo_passes(tmp_path, capsys):
    root = make_repo(tmp_path, {"engine/src/tensorfold/cli.py": 'HOST = "127.0.0.1"\nBIND = "0.0.0.0"\n',
                                "results/r.md": "TensorFold 0.3.6.2 under /tf/models and /srv/ai/x; master 192.0.2.1\n"
                                                "mail noreply@anthropic.com or user@example.com\n"
                                                "env NPP_VERSION 13.1.2.81; ~/octojet-runs/f2; $HOME/tensorfold\n"})
    assert run(root, capsys) == (0, "release check: clean\n")


@pytest.mark.parametrize("text, what", [
    (f"cache at {HOME}/tensorfold/cache", "home path"),
    (f"repo {MAC}/Documents/octojet", "home path"),
    (f"spark at {PRIVATE_IP}:8080", "IPv4 address"),
    (f"token {HF}", "Hugging Face token"),
    (f"key {SK}", "sk- key"),
    (f"socket {SOCKET}", "socket path"),
    (f"address {UDS}", "socket path"),
    ("Authori" + "zation: Bearer abcdef0123456789", "Authorization header"),
    (f"link https://claude.ai/code/{SESSION}", "session id"),
    (f"mail {EMAIL}", "email"),
])
def test_each_leak_is_a_finding(tmp_path, capsys, text, what):
    root = make_repo(tmp_path, {"docs/notes.md": f"line one\n{text}\n"})
    code, out = run(root, capsys)
    assert code == 1
    assert f"docs/notes.md:2: {what}" in out


def test_placeholders_and_the_allow_marker_pass(tmp_path, capsys):
    root = make_repo(tmp_path, {"docs/notes.md": "paths like /" + "home/<name> and ~/x; Authorization: Bearer $TOKEN\n"
                                                 f"{PRIVATE_IP} release-check: allow\n"
                                                 "connection.sock = sock\n"})
    assert run(root, capsys)[0] == 0


def test_untracked_and_binary_files_are_skipped(tmp_path, capsys):
    root = make_repo(tmp_path, {"data.bin": "x"})
    (root / "data.bin").write_bytes(b"\0" + HOME.encode())
    subprocess.run(["git", "-C", str(root), "add", "data.bin"], check=True)
    (root / "untracked.md").write_text(HOME + "\n")
    assert run(root, capsys)[0] == 0


def test_missing_release_files_and_the_old_prefix_are_findings(tmp_path, capsys):
    root = make_repo(tmp_path, {"engine/src/tensorfold/__init__.py": '__version__ = "0.2.0"\n',
                                "engine/src/tensorfold/log.py": 'print("[tensor' + 'fold] hi")\n',
                                "docs/model-card.md": "# Card\n"})
    (root / "LICENSE").unlink()
    code, out = run(root, capsys)
    assert code == 1
    assert "LICENSE: missing" in out
    assert "engine/CHANGELOG.md: no '## 0.2.0 ' entry" in out
    assert "docs/model-card.md: no '## License' section" in out
    assert "engine/src/tensorfold/log.py: '[tensorfold]' log prefix" in out


def test_the_repository_itself_is_clean(capsys):
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")
    code, out = run(root, capsys)
    assert code == 0, out
