"""bench/spark/f9-yarn.sh under the F2d stubs (docker, nvidia-smi, curl, ss, free, python3): the dry-run before anything,
the tests beside production, production stopped once under a pause held for the whole window, server A (--rope-yarn)
then server B (plain), the needle, cold-prompt and accuracy steps on each, the summary, and the pause removed on every
exit path. No GPU, no network."""

import json, os, stat, subprocess, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_f2d_a_scripts import BENCH, DOCKER, FAKE_RUNNER, HERE, PYTHON3, SIMPLE, calls  # noqa: E402

STARTUP = ('echo "[octojet] rope: YaRN factor 2.0, window 524288"; '
           'echo "[octojet] CUDA rank 0 startup estimate 101.20 GiB within 110.00 GiB; native 524288, '
           'allocated prompt/reply window 524288, cache slots 524296"; '
           'echo "[octojet] Flash Next on CUDA: x; 3 streams of 524281 prompt/reply tokens (2100 MiB a stream)"; ')
FAKE_DOCKER = DOCKER.replace('echo "[octojet] loaded in 59.2 s";', STARTUP + 'echo "[octojet] loaded in 59.2 s";')
assert STARTUP in FAKE_DOCKER

FAKE = FAKE_RUNNER.replace('''if "--dry-run" in argv:
    print("dry-run stub"); sys.exit(0)''', '''if script.endswith("needle.py"):
    if "--dry-run" in argv:
        print(json.dumps({"dry_run": True, "label": opt("--label")})); sys.exit(0)
    with open(opt("--out"), "a") as f:
        for size in opt("--sizes").split(","):
            for depth in opt("--depths").split(","):
                f.write(json.dumps({"label": opt("--label"), "target_tokens": int(size), "depth": float(depth),
                                    "prompt_tokens": int(size), "found": True, "reply": "1234-HELIX", "ttft_s": 99.0,
                                    "total_s": 100.0, "error": None}) + "\\n")
    sys.exit(0)
if script.endswith("acc_eval.py"):
    open(argv[3] + ".jsonl", "w").write(json.dumps({"task": "humaneval", "program": "pass"}) + "\\n")
    print(json.dumps({"gsm8k_acc": 0.98, "gsm8k_n": 250})); sys.exit(0)
if script.endswith("cmp_probe.py"):
    open(opt("--out"), "a").write(json.dumps({"label": opt("--label"), "ttft_s": 0.2}) + "\\n"); sys.exit(0)
if "--dry-run" in argv:
    print("dry-run stub"); sys.exit(0)''')
assert "needle.py" in FAKE


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "stub"; stub.mkdir()
    for name, text in {"docker": FAKE_DOCKER, "python3": PYTHON3, **SIMPLE}.items():
        p = stub / name; p.write_text(text); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (stub / "fake_runner.py").write_text(FAKE)
    home = tmp_path / "home"; home.mkdir()
    repo = tmp_path / "repo" / "bench"; repo.mkdir(parents=True)
    (repo / "run_humaneval.sh").write_text((BENCH / "run_humaneval.sh").read_text())
    e = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(home), "REPO": str(tmp_path / "repo"),
         "OUT": str(tmp_path / "out"), "STUB_LOG": str(tmp_path / "calls.log"), "STUB_DIR": str(stub),
         "REAL_PY": sys.executable, "BENCH_DIR": str(BENCH), "READY_DEADLINE": "10"}
    for k in ("SHARED_GPU", "STUB_FAIL_START", "FACTOR", "ACC", "TESTS", "PARALLEL"):
        e.pop(k, None)
    return e


def run(env, *args, **extra):
    env = {**env, **{k: str(v) for k, v in extra.items()}}
    return subprocess.run(["bash", str(HERE / "f9-yarn.sh"), *args], env=env, capture_output=True, text=True, timeout=120)


def servers(env):
    return [l for l in calls(env) if l.startswith("docker run") and " serve " in f" {l} "]


def test_script_parses():
    subprocess.run(["bash", "-n", str(HERE / "f9-yarn.sh")], check=True)


def test_dry_run_starts_and_stops_nothing(env):
    r = run(env, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    assert not any(l.startswith(("server-start", "docker stop -t 60 oj-serve")) for l in log)
    assert not any("-m pytest" in l for l in log)
    assert sum("bench/needle.py" in l and "--dry-run" in l for l in log) == 2
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_full_window(env):
    r = run(env)
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    tests = [i for i, l in enumerate(log) if l.startswith("docker run") and "python -m pytest" in l]
    prod = [i for i, l in enumerate(log) if l.startswith("docker stop -t 60 oj-serve")]
    starts = [i for i, l in enumerate(log) if l.startswith("server-start")]
    assert len(tests) == 1 and len(prod) == 1 and len(starts) == 2 and tests[0] < prod[0] < starts[0]
    for name in ("test_flashnext_yarn.py", "test_flashnext_multi.py", "test_qwen4_exp_nvfp4.py", "test_rope_yarn.py"):
        assert name in log[tests[0]], name
    assert all("pause=1 parallel=3" in log[i] for i in starts)                  # the pause is held across both servers
    a, b = servers(env)
    assert "--rope-yarn 2" in a and "--rope-yarn" not in b
    for s in (a, b):
        assert "--kv-dtype int8" in s and "--packed-cache" in s and "--vision" not in s
    needles = [(i, l) for i, l in enumerate(log) if "bench/needle.py run" in l and "--dry-run" not in l]
    assert len(needles) == 3
    assert "--sizes 450000 --depths 0.5" in needles[0][1] and starts[0] < needles[0][0] < starts[1]
    assert "--sizes 128000,256000,450000" in needles[1][1] and needles[1][0] < starts[1]
    assert "--sizes 128000,256000 " in needles[2][1] and needles[2][0] > starts[1]
    accs = [i for i, l in enumerate(log) if "bench/acc_eval.py" in l]
    assert len(accs) == 2 and accs[0] < starts[1] < accs[1]
    out = Path(env["OUT"])
    summary = (out / "f9-summary.txt").read_text()
    assert "rope: YaRN factor 2.0, window 524288" in summary
    assert "native 524288, allocated window 524288; 3 streams of 524281: FITS" in summary
    assert "yarn-cold | 450000 | 0.5" in summary and "base | 256000 | 0.9" in summary
    assert "TTFT 99.0 s" in summary and '"gsm8k_acc": 0.98' in summary
    tl = (out / "f9-timeline.txt").read_text()
    assert tl.count("standby watcher paused") == 1 and tl.count("standby watcher resumed") == 1
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_knobs_and_acc_off(env):
    r = run(env, FACTOR=4, ACC=0, TESTS=0, YARN_SIZES="100000", BASE_SIZES="50000", COLD=900000)
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    assert not any("-m pytest" in l for l in log) and not any("acc_eval.py" in l for l in log)
    assert "--rope-yarn 4" in servers(env)[0]
    assert any("--sizes 900000 --depths 0.5" in l for l in log)


def test_a_failed_start_cleans_up_and_bad_arguments(env):
    r = run(env, STUB_FAIL_START=1, TESTS=0)
    assert r.returncode != 0 and not Path(env["HOME"], ".spark-standby.pause").exists()
    assert run(env, "--bogus").returncode == 2
    assert run(env, "--dry-run", FACTOR=1).returncode == 2
    assert run(env, "--dry-run", FACTOR="two").returncode == 2
