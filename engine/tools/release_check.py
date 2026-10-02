"""Check the repository before a public release; exit 1 on any finding.

Run from the repository root: ``python engine/tools/release_check.py``. It scans every git-tracked text file for

- IPv4 addresses, except loopback (127.0.0.0/8), the bind-all address and ``0.x.y.z`` (which is also how four-part
  version numbers such as 0.3.6.2 look), the documentation range 192.0.2.0/24, well-known public or special-purpose
  addresses that name no private host (public DNS resolvers, cloud metadata endpoints, IETF 192.0.0.0/24,
  multicast, broadcast; the SSRF guard and its tests list them), and numbers right after the word "version";
- home-directory paths (``/Users/<name>``, ``/home/<name>``);
- ``.sock`` socket paths and ``uds:`` socket addresses (the operator's session socket);
- Hugging Face tokens (``hf_`` and 20+ characters) and ``sk-`` API keys;
- ``Authorization:`` headers carrying a literal credential;
- session ids (``session_`` and 20+ characters, Claude session links);
- email addresses, except the ones in ``ALLOWED_EMAILS`` and reserved example domains (RFC 2606);

and checks that the license files, the CHANGELOG entry for the current version and the model card's license section
exist, and that no old TensorFold log prefix (``OLD_PREFIX``) remains under ``engine/src``. Directory paths such as
``/tf/...`` or ``/srv/ai/...`` in result logs are not flagged: they name directories, not hosts or secrets. A line
that has to show one of these patterns (documentation of this check, say) can carry the marker
``release-check: allow``.
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import subprocess
import sys
from pathlib import Path

ALLOW_MARKER = "release-check: allow"
OLD_PREFIX = "[" + "tensorfold]"                   # spelled in two parts so this file passes its own scan
ALLOWED_EMAILS = {"noreply@anthropic.com"}
ALLOWED_NETWORKS = [ipaddress.ip_network(n) for n in (
    "0.0.0.0/8", "127.0.0.0/8", "192.0.2.0/24",                     # this host / versions, loopback, documentation
    "1.1.1.1/32", "1.0.0.1/32", "8.8.8.8/32", "8.8.4.4/32",          # public DNS resolvers
    "169.254.169.254/32", "168.63.129.16/32", "100.100.100.200/32",  # cloud metadata endpoints
    "192.0.0.0/24", "224.0.0.0/4", "255.255.255.255/32",             # IETF special, multicast, broadcast
)]
EXAMPLE_DOMAIN = re.compile(r"(?:^|\.)(?:example\.(?:com|org|net)|example|test|invalid|localhost)$", re.I)

IPV4 = re.compile(r"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\w.]*\d)")
HOME = re.compile(r"/(?:Users|home)/(?!<)[A-Za-z0-9_][A-Za-z0-9_.-]*")
SOCK = re.compile(r"(?:[\w.~$<>{}-]*/)+[\w.-]*\w\.sock\b|\buds:/\S+")
HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{20,}")
SK_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")
AUTH = re.compile(r"Authorization:\s*(?:Bearer|Basic|Token|token)\s+(?![$<{]|\.\.\.)[A-Za-z0-9._~+/=-]{8,}")
SESSION = re.compile(r"\bsession_[A-Za-z0-9]{20,}|claude\.ai/code/session")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")

TEXT_LIMIT = 8 * 1024 * 1024


def _ip_finding(match: re.Match, line: str) -> str | None:
    if any(int(part) > 255 for part in match.groups()):
        return None
    if re.search(r"version", line[max(0, match.start() - 40):match.start()], re.I):
        return None                                    # a library version such as NPP_VERSION 13.1.2.81
    address = ipaddress.ip_address(match.group(0))
    if any(address in network for network in ALLOWED_NETWORKS):
        return None
    return f"IPv4 address {match.group(0)}"


def scan_line(line: str) -> list[str]:
    """What one line leaks, as short descriptions (empty when nothing)."""

    if ALLOW_MARKER in line:
        return []
    found: list[str] = []
    for match in IPV4.finditer(line):
        what = _ip_finding(match, line)
        if what:
            found.append(what)
    found += [f"home path {m.group(0)}" for m in HOME.finditer(line)]
    found += [f"socket path {m.group(0)}" for m in SOCK.finditer(line)]
    found += ["Hugging Face token" for _ in HF_TOKEN.finditer(line)]
    found += ["sk- key" for _ in SK_TOKEN.finditer(line)]
    found += ["Authorization header with a credential" for _ in AUTH.finditer(line)]
    found += [f"session id {m.group(0)}" for m in SESSION.finditer(line)]
    found += [f"email {m.group(0)}" for m in EMAIL.finditer(line)
              if m.group(0).lower() not in ALLOWED_EMAILS and not EXAMPLE_DOMAIN.search(m.group(0).split("@")[1])]
    return found


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [root / name for name in out.decode().split("\0") if name]


def read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > TEXT_LIMIT:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None                                    # binary
    return data.decode("utf-8", errors="replace")


def scan_files(root: Path) -> list[str]:
    findings = []
    for path in tracked_files(root):
        text = read_text(path)
        if text is None:
            continue
        rel = path.relative_to(root)
        for number, line in enumerate(text.splitlines(), 1):
            for what in scan_line(line):
                findings.append(f"{rel}:{number}: {what}")
    return findings


def current_version(root: Path) -> str | None:
    init = root / "engine" / "src" / "tensorfold" / "__init__.py"
    try:
        match = re.search(r'^__version__\s*=\s*"([^"]+)"', init.read_text(), re.M)
    except OSError:
        return None
    return match.group(1) if match else None


def check_release_files(root: Path) -> list[str]:
    findings = []
    for name in ("LICENSE", "engine/LICENSE", "engine/THIRD_PARTY_NOTICES.md"):
        if not (root / name).is_file():
            findings.append(f"{name}: missing")
    version = current_version(root)
    changelog = root / "engine" / "CHANGELOG.md"
    if version is None:
        findings.append("engine/src/tensorfold/__init__.py: no __version__")
    elif not changelog.is_file() or not re.search(rf"^## {re.escape(version)} ", changelog.read_text(), re.M):
        findings.append(f"engine/CHANGELOG.md: no '## {version} ' entry")
    card = root / "docs" / "model-card.md"
    if not card.is_file() or not re.search(r"^## License\b", card.read_text(), re.M):
        findings.append("docs/model-card.md: no '## License' section")
    src = root / "engine" / "src"
    for path in sorted(src.rglob("*")) if src.is_dir() else []:
        if path.is_file() and "egg-info" not in path.parts[-2] and path.suffix not in {".pyc", ".so"}:
            text = read_text(path)
            if text and OLD_PREFIX in text:
                findings.append(f"{path.relative_to(root)}: {OLD_PREFIX!r} log prefix")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="the repository root (default: here)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    findings = check_release_files(root) + scan_files(root)
    for line in findings:
        print(line)
    if findings:
        print(f"release check: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print("release check: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
