"""bench/spark/f12-profile.sh under stubs (docker, nvidia-smi, curl, ss, free, python3): the dry-run starts and stops
nothing; the window stops production once under a pause held for every server (three nsys-launched profile servers,
then main and f12 for TTFT), captures each size inside nsys start / stop, removes the pause, then compares the kernel
sums and the TTFTs; argument and checkout refusals. No GPU, no network."""

import json, os, stat, subprocess, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_f2d_a_scripts import BENCH, DOCKER, FAKE_RUNNER, HERE, SIMPLE  # noqa: E402

KERN = ('Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n'
        '60.0,6000000,3,1,1,1,1,0,"void <unnamed>::prefill_kernel<(int)32>(int)"\n40.0,4000000,9,1,1,1,1,0,"_hc_mix"\n')
EXTRA = r'''
if [ "$1" = exec ]; then
  if printf '%s\n' "$@" | grep -qx -- start; then
    rep=$(printf '%s\n' "$@" | sed -n 's|^--output=/prof/||p'); echo "$rep" > "$STUB_DIR/rep"; echo "nsys-start $rep pause=$(pause)" >> "$STUB_LOG"
  elif printf '%s\n' "$@" | grep -qx -- stop; then
    rep=$(cat "$STUB_DIR/rep"); printf 'report' > "$P/$rep.nsys-rep"; echo "nsys-stop $rep" >> "$STUB_LOG"
  fi
  exit 0
fi
if [ "$1" = run ] && printf '%s\n' "$@" | grep -qx -- stats; then printf '%s' "$KERN_CSV"; exit 0; fi
'''
STUB_DOCKER = DOCKER.replace("case \"$1\" in", EXTRA + "case \"$1\" in", 1)
PYTHON3 = r'''#!/usr/bin/env bash
echo "python3 $*" >> "$STUB_LOG"
case "$1" in
  -) exec "$REAL_PY" - "${@:2}";;
  bench/kern_compare.py) exec "$REAL_PY" "$BENCH_DIR/kern_compare.py" "${@:2}";;
  *) exec "$REAL_PY" "$STUB_DIR/fake_runner.py" "$@";;
esac
'''


def _checkout(path: Path, f12: bool) -> None:
    (path / "engine/src/tensorfold/families/qwen4_exp/cuda").mkdir(parents=True)
    (path / "engine/src/tensorfold/cli.py").write_text("")
    if f12:
        (path / "engine/src/tensorfold/families/qwen4_exp/cuda/hc_prefill.py").write_text("")


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "stub"; stub.mkdir()
    for name, text in {"docker": STUB_DOCKER, "python3": PYTHON3, **SIMPLE}.items():
        p = stub / name; p.write_text(text); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (stub / "fake_runner.py").write_text(FAKE_RUNNER)
    home = tmp_path / "home"; home.mkdir()
    _checkout(tmp_path / "f12", True); _checkout(tmp_path / "main", False)
    manifests = tmp_path / "manifests"; manifests.mkdir()
    for m in ("m32k", "m128k", "m210k"):
        (manifests / f"{m}.json").write_text(json.dumps({"tokens": 3, "ids": [5, 6, 7]}))
    out = tmp_path / "out"; (out / "prompts").mkdir(parents=True)
    for m in ("m32k", "m128k", "m210k"):
        (out / "prompts" / f"{m}.txt").write_text(f"text of {m}")
    ckpt = tmp_path / "mlx"; ckpt.mkdir(); (ckpt / "config.json").write_text("{}")
    up = tmp_path / "up"; (up / "tensorfold").mkdir(parents=True)
    nsys = tmp_path / "nsys"; (nsys / "bin").mkdir(parents=True)    # a host Nsight Systems install (bin/nsys)
    (nsys / "bin" / "nsys").write_text("#!/bin/sh\nexit 0\n"); (nsys / "bin" / "nsys").chmod(0o755)
    prof = tmp_path / "prof"
    e = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(home), "REPO": str(tmp_path / "f12"),
         "REPO_MAIN": str(tmp_path / "main"), "OUT": str(out), "M": str(manifests), "PROF_CKPT": str(ckpt),
         "UP_SRC": str(up), "NSYS_HOST": str(nsys), "P": str(prof), "POLL": "0", "KERN_CSV": KERN,
         "STUB_LOG": str(tmp_path / "calls.log"), "STUB_DIR": str(stub), "REAL_PY": sys.executable,
         "BENCH_DIR": str(BENCH), "READY_DEADLINE": "10", "PROD_STOP": "echo PRODSTOP >> " + str(tmp_path / "calls.log")}
    for k in ("SHARED_GPU", "STUB_FAIL_START", "OCTOJET_PREFILL_HC_TILE", "FORCE"):
        e.pop(k, None)
    return e


def run(env, *args, **extra):
    env = {**env, **{k: str(v) for k, v in extra.items()}}
    return subprocess.run(["bash", str(HERE / "f12-profile.sh"), *args], env=env, capture_output=True, text=True,
                          timeout=180)


def calls(env) -> list[str]:
    p = Path(env["STUB_LOG"])
    return p.read_text().splitlines() if p.exists() else []


def test_the_dry_run_starts_and_stops_nothing(env):
    r = run(env, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    assert not any(l.startswith("server-start") or l == "PRODSTOP" or "drop_caches" in l for l in log)
    assert not (Path(env["HOME"]) / ".spark-standby.pause").exists()


def test_the_window(env):
    r = run(env, "--steps", "profile,ttft")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    starts = [l for l in log if l.startswith("server-start")]
    assert len(starts) == 5 and all("pause=1" in l for l in starts)
    assert log.count("PRODSTOP") == 1 and log.index("PRODSTOP") < log.index(starts[0])
    runs = [l for l in log if l.startswith("docker run") and " serve " in l]
    assert [("nsys" in l and " launch " in l) for l in runs] == [True, True, True, False, False]
    assert ["/main:/octojet" in runs[0], "/f12:/octojet" in runs[1], "/up:/up" in runs[2]] == [True, True, True]
    assert "/main:/octojet" in runs[3] and "/f12:/octojet" in runs[4] and "flashnext-nvfp4-mixed" in runs[4]
    caps = [l.split()[1] for l in log if l.startswith("nsys-start")]
    assert caps == [f"f12-{e}-{s}" for e in ("main", "f12", "up") for s in ("m32k", "m128k")]
    assert not (Path(env["HOME"]) / ".spark-standby.pause").exists()
    out = Path(env["OUT"])
    text = (out / "f12-kernels-m128k.txt").read_text()
    assert "<unnamed>::prefill_kernel<(int)32>" in text and "f12 - main" in text and "up - main" in text
    assert "m210k" in (out / "f12-ttft.txt").read_text()


def test_refusals(env, tmp_path):
    assert run(env, "--steps", "bogus").returncode == 2
    assert run(env, REPO_MAIN=env["REPO"]).returncode == 2              # main must not carry F12
    assert run(env, "--steps", "profile", UP_SRC=str(tmp_path / "none")).returncode == 2
