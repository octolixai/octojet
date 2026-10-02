"""bench/spark/release-build.sh under the F2d stubs (docker, nvidia-smi, curl, ss, free, python3): input validation and
the dry-run (nothing built, stopped or started), then the full sequence: build, check and tests beside production,
production stopped once under the pause, the release server with production's text flags and its own packed cache,
the exactness and accuracy receipts, the summary gates, and the pause removed on every exit path. No GPU, no network."""

import json, os, stat, subprocess, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_f2d_a_scripts import BENCH, DOCKER, FAKE_RUNNER, HERE, PYTHON3, SIMPLE, calls  # noqa: E402

SCRIPT = HERE / "release-build.sh"
EXTRA = r'''    if printf '%s\n' "$@" | grep -q -- 'tools/build_release_checkpoint.py'; then
      out=$(printf '%s\n' "$@" | grep -A1 -x -- '--out' | tail -1); mkdir -p "$out"; echo '{}' > "$out/octojet.json"
      printf '%s' '{"total_size": 107000000000, "out_file_bytes": 107100000000, "base": {"total_size": 37000000000, "shards": [1, 2], "dropped_tensors": 288}, "experts": {"total_size": 70000000000, "copied_whole": 192, "rewritten": 1, "wasted_bytes": 1000}}' > "$out.build-report.json"
      echo "[build] stub"; exit 0; fi
    if printf '%s\n' "$@" | grep -q -- 'tools/check_mixed_checkpoint.py'; then
      echo "{\"ok\": ${STUB_CHECK_OK:-true}, \"routers_equal\": true, \"routed_layers_agree\": \"48/48\", \"shared_layers_agree\": \"48/48\", \"source_bytes\": {\"base\": {\"checked\": 9, \"mismatched\": 0}, \"experts\": {\"checked\": 9, \"mismatched\": 0}, \"dropped\": {\"present\": 0}, \"kept\": {\"missing\": 0}}}"; exit 0; fi
    if printf '%s\n' "$@" | grep -q -- '-m pytest'; then echo "F6 TESTS rc=0"; exit 0; fi
'''
STUB_DOCKER = DOCKER.replace('''    if printf '%s\\n' "$@" | grep -q -- '-m pytest'; then''', EXTRA + '''    if printf '%s\\n' "$@" | grep -q -- '-m pytest'; then''', 1)
STUB_DOCKER = STUB_DOCKER.replace('echo "2 streams of 262144 prompt/reply tokens"',
                                  'echo "[octojet] CUDA rank 0 startup estimate ${STUB_EST:-87.40} GiB within 110.00 GiB; native 262144"; '
                                  'echo "3 streams of 261872 prompt/reply tokens"')
FAKE = FAKE_RUNNER.replace('''if "--dry-run" in argv:''', '''if script.endswith("acc_eval.py"):
    import os
    n = int(os.environ.get("STUB_GSM", "244"))
    open(argv[3] + ".jsonl", "w").write("")
    print(json.dumps({"gsm8k_acc": n / 250, "gsm8k_n": 250, "humaneval_programs": 164})); sys.exit(0)
if "--dry-run" in argv:''', 1)
assert "build_release_checkpoint" in STUB_DOCKER and "STUB_EST" in STUB_DOCKER and "acc_eval" in FAKE


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "stub"; stub.mkdir()
    for name, text in {"docker": STUB_DOCKER, "python3": PYTHON3, **SIMPLE}.items():
        p = stub / name; p.write_text(text); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (stub / "fake_runner.py").write_text(FAKE)
    home = tmp_path / "home"; (home / "tensorfold").mkdir(parents=True)
    repo = tmp_path / "repo"; (repo / "bench").mkdir(parents=True)
    (repo / "bench" / "run_humaneval.sh").write_text(
        '#!/bin/bash\necho "{\\"humaneval_pass\\": ${STUB_HE:-156}, \\"humaneval_n\\": 164}" > "$1.humaneval.txt"\n')
    for name in ("mlx", "radix"):
        d = tmp_path / name; d.mkdir()
        (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "s.safetensors"}}))
        (d / "config.json").write_text("{}")
    e = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(home), "REPO": str(repo),
         "RUNS": str(tmp_path / "runs"), "OUT": str(tmp_path / "rel" / "octojet-release"), "MLX": str(tmp_path / "mlx"),
         "NVFP4": str(tmp_path / "radix"), "PACKED": str(tmp_path / "packed"), "NEED_GB": "0", "CACHE_GB": "0",
         "STUB_LOG": str(tmp_path / "calls.log"), "STUB_DIR": str(stub), "REAL_PY": sys.executable,
         "BENCH_DIR": str(BENCH), "READY_DEADLINE": "10"}
    for k in ("SHARED_GPU", "STUB_FAIL_START", "PARALLEL", "MLX_REVISION", "NVFP4_REVISION", "STUB_EST", "STUB_GSM",
              "STUB_HE", "STUB_CHECK_OK"):
        e.pop(k, None)
    return e


def run(env, *args, **extra):
    env = {**env, **{k: str(v) for k, v in extra.items()}}
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=120)


def test_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_dry_run_validates_and_touches_nothing(env):
    r = run(env, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    assert not any(l.startswith(("server-start", "docker stop -t 60 oj-serve")) for l in log)
    assert not any("build_release_checkpoint.py" in l or "check_mixed_checkpoint.py" in l for l in log)
    assert not Path(env["OUT"]).exists() and not Path(env["HOME"], ".spark-standby.pause").exists()
    assert "dry-run OK" in r.stdout and "--kv-dtype int8 --parallel 3" in r.stdout


def test_bad_inputs_are_refused(env, tmp_path):
    assert run(env, "--steps", "5").returncode == 2
    assert run(env, "--dry-run", MLX_REVISION="main").returncode == 2
    r = run(env, "--dry-run", NEED_GB=10**9)
    assert r.returncode == 2 and "needs ~" in r.stderr
    Path(env["OUT"]).mkdir(parents=True); Path(env["OUT"], "x").write_text("x")
    r = run(env, "--dry-run")
    assert r.returncode == 2 and "not empty" in r.stderr
    r = run(env, "--dry-run", "--steps", "2")                 # verifying an existing build needs its marker
    assert r.returncode == 2 and "octojet.json missing" in r.stderr
    Path(env["NVFP4"], "model.safetensors.index.json").unlink()
    r = run(env, "--dry-run", "--steps", "2")
    assert r.returncode == 2 and "index.json missing" in r.stderr
    assert not any(l.startswith("server-start") for l in calls(env))


def test_full_sequence_passes(env):
    r = run(env)
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    build = [i for i, l in enumerate(log) if "tools/build_release_checkpoint.py" in l]
    chk = [i for i, l in enumerate(log) if "tools/check_mixed_checkpoint.py" in l]
    tests = [i for i, l in enumerate(log) if l.startswith("docker run") and "python -m pytest" in l]
    prod = [i for i, l in enumerate(log) if l.startswith("docker stop -t 60 oj-serve")]
    starts = [i for i, l in enumerate(log) if l.startswith("server-start")]
    assert len(build) == len(chk) == len(tests) == len(prod) == len(starts) == 1
    assert build[0] < chk[0] < tests[0] < prod[0] < starts[0]
    b, c, t = log[build[0]], log[chk[0]], log[tests[0]]
    assert "--hash" in b and "--mlx-revision unknown" in b and "--nvfp4-revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594" in b
    assert "--user" in b
    assert "--layers all --source-bytes" in c and f"--reference-nvfp4 {env['NVFP4']}" in c
    assert f"OCTOJET_NVFP4_FLASHNEXT={env['OUT']}" in t and "test_qwen4_exp_nvfp4.py" in t and "test_prefix_reuse_flashnext.py" in t
    server = [l for l in log if l.startswith("docker run") and " serve " in f" {l} "][0]
    assert f"serve {env['OUT']}" in server and "--kv-dtype int8" in server and "--vision" not in server
    assert f"--packed-cache {env['PACKED']}" in server and "pause=1 parallel=3" in log[starts[0]]
    assert any("bench_openai.py" in l and "--expect-equal" in l for l in log)
    assert any("concurrent_equal.py" in l and "--prompts 2" in l for l in log)
    assert any("acc_eval.py" in l for l in log)
    summary = Path(env["RUNS"], "f6-summary.txt").read_text()
    assert "F6 release checks: PASS" in summary and "GSM8K 244/250" in summary and "HumanEval 156/164" in summary
    assert "3 x 261872" in summary and "48/48" in summary
    tl = Path(env["RUNS"], "f6-timeline.txt").read_text()
    assert tl.count("standby watcher paused") == 1 and tl.count("standby watcher resumed") == 1
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


@pytest.mark.parametrize("knob", [{"STUB_GSM": 241}, {"STUB_HE": 150}, {"STUB_EST": 90.1}])
def test_a_gate_miss_fails_and_restores(env, knob):
    Path(env["OUT"]).mkdir(parents=True); Path(env["OUT"], "octojet.json").write_text("{}")
    r = run(env, "--steps", "4", **knob)
    assert r.returncode == 1 and "F6 release checks: CHECK" in r.stdout
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_failed_check_and_failed_start_clean_up(env):
    r = run(env, "--steps", "1,2", STUB_CHECK_OK="false")
    assert r.returncode == 1 and "CHECK" in r.stdout
    r = run(env, "--steps", "4", STUB_FAIL_START=1)
    assert r.returncode != 0 and not Path(env["HOME"], ".spark-standby.pause").exists()
