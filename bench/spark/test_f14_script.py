"""bench/spark/f14-test.sh under stubs (docker, nvidia-smi, curl, ss, free, python3): the dry-run starts and stops
nothing; the tests step runs the F14 suites in one container beside production; the window stops production once
under a pause held for both servers (main, then f14), removes it, and writes the TTFT table; checkout and argument
refusals. No GPU, no network."""

import json, os, stat, subprocess, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_f2d_a_scripts import BENCH, DOCKER, FAKE_RUNNER, HERE, SIMPLE  # noqa: E402

PYTHON3 = r'''#!/usr/bin/env bash
echo "python3 $*" >> "$STUB_LOG"
case "$1" in
  -) exec "$REAL_PY" - "${@:2}";;
  *) exec "$REAL_PY" "$STUB_DIR/fake_runner.py" "$@";;
esac
'''


def _checkout(path: Path, f14: bool) -> None:
    gdn = path / "engine/src/tensorfold/cuda/kernels"
    gdn.mkdir(parents=True)
    (path / "engine/src/tensorfold/cli.py").write_text("")
    (gdn / "gdn.py").write_text('AHEAD_ENV = "OCTOJET_GDN_CHAIN_AHEAD"\n' if f14 else "")


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "stub"; stub.mkdir()
    for name, text in {"docker": DOCKER, "python3": PYTHON3, **SIMPLE}.items():
        p = stub / name; p.write_text(text); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (stub / "fake_runner.py").write_text(FAKE_RUNNER)
    home = tmp_path / "home"; home.mkdir()
    _checkout(tmp_path / "f14", True); _checkout(tmp_path / "main", False)
    manifests = tmp_path / "manifests"; manifests.mkdir()
    for m in ("m32k", "m128k", "m210k"):
        (manifests / f"{m}.json").write_text(json.dumps({"tokens": 3, "ids": [5, 6, 7]}))
    out = tmp_path / "out"; (out / "prompts").mkdir(parents=True)
    for m in ("m32k", "m128k", "m210k"):
        (out / "prompts" / f"{m}.txt").write_text(f"text of {m}")
    e = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(home), "REPO": str(tmp_path / "f14"),
         "REPO_MAIN": str(tmp_path / "main"), "OUT": str(out), "M": str(manifests),
         "PROF_CKPT": str(tmp_path / "no-mlx"), "STUB_LOG": str(tmp_path / "calls.log"), "STUB_DIR": str(stub),
         "REAL_PY": sys.executable, "BENCH_DIR": str(BENCH), "READY_DEADLINE": "10",
         "PROD_STOP": "echo PRODSTOP >> " + str(tmp_path / "calls.log")}
    for k in ("SHARED_GPU", "STUB_FAIL_START", "FORCE"):
        e.pop(k, None)
    return e


def run(env, *args, **extra):
    env = {**env, **{k: str(v) for k, v in extra.items()}}
    return subprocess.run(["bash", str(HERE / "f14-test.sh"), *args], env=env, capture_output=True, text=True,
                          timeout=180)


def calls(env) -> list[str]:
    p = Path(env["STUB_LOG"])
    return p.read_text().splitlines() if p.exists() else []


def test_the_dry_run_starts_and_stops_nothing(env):
    r = run(env, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    assert not any(l.startswith("server-start") or l == "PRODSTOP" or "drop_caches" in l or "pytest" in l for l in log)
    assert not (Path(env["HOME"]) / ".spark-standby.pause").exists()


def test_the_tests_step_runs_beside_production(env):
    r = run(env, "--steps", "tests")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    tests = [l for l in log if "-m pytest" in l]
    assert len(tests) == 1 and "test_gdn_chain_ahead.py" in tests[0] and "f14_gdn_bench.py" in tests[0]
    assert "/f14:/octojet" in tests[0]
    assert "PRODSTOP" not in log and not any(l.startswith("server-start") for l in log)


def test_the_window(env):
    r = run(env, "--steps", "ttft")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    starts = [l for l in log if l.startswith("server-start")]
    assert len(starts) == 2 and all("pause=1" in l and "parallel=3" in l for l in starts)
    assert log.count("PRODSTOP") == 1 and log.index("PRODSTOP") < log.index(starts[0])
    runs = [l for l in log if l.startswith("docker run") and " serve " in l]
    assert "/main:/octojet" in runs[0] and "/f14:/octojet" in runs[1] and "flashnext-nvfp4-mixed" in runs[1]
    assert not (Path(env["HOME"]) / ".spark-standby.pause").exists()
    text = (Path(env["OUT"]) / "f14-ttft.txt").read_text()
    assert "f14 / main" in text and "m210k" in text


def test_refusals(env):
    assert run(env, "--steps", "bogus").returncode == 2
    assert run(env, REPO_MAIN=env["REPO"]).returncode == 2              # main must not carry F14
    assert run(env, REPO=env["REPO_MAIN"]).returncode == 2              # the f14 checkout must carry it
