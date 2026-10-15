"""bench/spark/f10-mtp.sh under the F2d stubs (docker, nvidia-smi, curl, ss, free, python3): the dry-run builds the
prompt plan and starts nothing; the window stops production once under a pause held throughout, serves one test
server per (confidence, depth) with production's flags plus that setting, runs mtp_sweep against each, summarises
every setting with the default named, and removes the pause on every exit path. No GPU, no network."""

import json, os, stat, subprocess, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_f2d_a_scripts import DOCKER, FAKE_RUNNER, HERE, SIMPLE, calls  # noqa: E402

PYTHON3 = r'''#!/usr/bin/env bash
echo "python3 $*" >> "$STUB_LOG"
case "$1" in
  bench/mtp_sweep.py) exec "$REAL_PY" "$STUB_DIR/fake_sweep.py" "${@:2}";;
  *) exec "$REAL_PY" "$STUB_DIR/fake_runner.py" "$@";;
esac
'''
FAKE_SWEEP = r'''
import json, os, sys
argv = sys.argv[1:]
def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default
if argv[0] == "summary":
    files = [a for a in argv[1:] if a.endswith(".jsonl")]
    print("summary of", len(files), "default", opt("--default-label"))
    json.dump({"files": files}, open(opt("--json"), "w"))
    sys.exit(1 if os.environ.get("STUB_MISMATCH") else 0)
if "--dry-run" in argv:
    print("40 prompts (20 code, 20 chat) x 2 samplings = 80 requests of", opt("--tokens"), "tokens"); sys.exit(0)
with open(opt("--out"), "a") as f:
    f.write(json.dumps({"label": opt("--label"), "decode_tps": 60.0}) + "\n")
print("swept", opt("--label"))
'''


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "stub"; stub.mkdir()
    for name, text in {"docker": DOCKER, "python3": PYTHON3, **SIMPLE}.items():
        p = stub / name; p.write_text(text); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (stub / "fake_runner.py").write_text(FAKE_RUNNER)
    (stub / "fake_sweep.py").write_text(FAKE_SWEEP)
    home = tmp_path / "home"; home.mkdir()
    (tmp_path / "repo").mkdir()
    e = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(home), "REPO": str(tmp_path / "repo"),
         "OUT": str(tmp_path / "out"), "STUB_LOG": str(tmp_path / "calls.log"), "STUB_DIR": str(stub),
         "REAL_PY": sys.executable, "READY_DEADLINE": "10"}
    for k in ("SHARED_GPU", "STUB_FAIL_START", "STUB_MISMATCH", "PARALLEL", "CONFS", "DEPTHS"):
        e.pop(k, None)
    return e


def run(env, *args, **extra):
    env = {**env, **{k: str(v) for k, v in extra.items()}}
    return subprocess.run(["bash", str(HERE / "f10-mtp.sh"), *args], env=env, capture_output=True, text=True, timeout=120)


def test_script_parses():
    subprocess.run(["bash", "-n", str(HERE / "f10-mtp.sh")], check=True)


def test_dry_run_starts_and_stops_nothing(env):
    r = run(env, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    assert not any(l.startswith(("server-start", "docker stop -t 60 oj-serve")) for l in log)
    assert sum("bench/mtp_sweep.py run" in l and "--dry-run" in l for l in log) == 1
    assert r.stdout.count("dry-run: would serve") == 4 and "--mtp-confidence 0.7 --mtp-drafts 6" in r.stdout
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_the_window(env):
    r = run(env, DEPTHS="6 4", CONFS="0.3 0.7")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    starts = [i for i, l in enumerate(log) if l.startswith("server-start")]
    assert len(starts) == 4 and all("pause=1 parallel=3" in log[i] for i in starts)
    assert sum(l.startswith("docker stop -t 60 oj-serve") for l in log) == 1
    assert log.index(next(l for l in log if l.startswith("docker stop -t 60 oj-serve"))) < starts[0]
    serves = [l for l in log if l.startswith("docker run") and " serve " in f" {l} "]
    want = [("0.3", "6"), ("0.7", "6"), ("0.3", "4"), ("0.7", "4")]
    for line, (c, d) in zip(serves, want):
        assert f"--mtp-confidence {c} --mtp-drafts {d}" in line and "--kv-dtype int8" in line and "--packed-cache" in line
        assert "--vision" not in line
    sweeps = [l for l in log if "bench/mtp_sweep.py run" in l and "--dry-run" not in l]
    assert [s.split("--label ")[1].split()[0] for s in sweeps] == ["c0.3-d6", "c0.7-d6", "c0.3-d4", "c0.7-d4"]
    assert all("--tokens 200" in s and "--setting confidence=" in s for s in sweeps)
    summary = [l for l in log if "bench/mtp_sweep.py summary" in l]
    assert len(summary) == 1 and summary[0].count(".jsonl") == 4 and "--default-label c0.3-d6" in summary[0]
    assert "summary of 4 default c0.3-d6" in Path(env["OUT"], "f10-summary.txt").read_text()
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_a_mismatch_fails_the_run_and_still_removes_the_pause(env):
    r = run(env, CONFS="0.3", STUB_MISMATCH="1")
    assert r.returncode == 1
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_a_failed_start_removes_the_pause(env):
    r = run(env, STUB_FAIL_START="2")
    assert r.returncode != 0
    assert sum(l.startswith("server-start") for l in calls(env)) == 2
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_bad_settings_are_refused_before_anything(env):
    for extra in ({"CONFS": "1.5"}, {"CONFS": "abc"}, {"DEPTHS": "0"}, {"DEPTHS": "x"}):
        r = run(env, **extra)
        assert r.returncode == 2, extra
    assert not any(l.startswith("server-start") for l in calls(env))
