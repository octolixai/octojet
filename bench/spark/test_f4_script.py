"""bench/spark/f4-test.sh under the F2d stubs (docker, nvidia-smi, curl, ss, free, python3): the dry-run before anything,
the tests beside production, production stopped once under a pause held for the whole window, the main server
(--vision, default checkpoints) then the checkpoints-off server, the live checks and their summary, and the pause
removed on every exit path. No GPU, no network."""

import json, os, stat, subprocess, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_f2d_a_scripts import BENCH, DOCKER, FAKE_RUNNER, HERE, PYTHON3, SIMPLE, calls  # noqa: E402

FAKE = FAKE_RUNNER.replace(
    '''    json.dump({"stub": "bench_openai", "expect_equal": "--expect-equal" in argv}, open(opt("--output"), "w")); sys.exit(0)''',
    '''    sha = "differs" if (opt("--label") == "off" and __import__("os").environ.get("STUB_DIFFER")) else "same"
    json.dump([{"prompt": "p", "temperature": 0.0, "token_sha_all": [sha]}], open(opt("--output"), "w")); sys.exit(0)
if script.endswith("f4_live.py"):
    if "--dry-run" in argv:
        print("dry-run f4_live"); sys.exit(0)
    check = argv[3]
    res = ({"check": "gap", "b_prompt_tokens": 3, "a_max_gap_s": 1.8, "a_tokens_during_b": 40, "b_ttft_s": 70.0}
           if check == "gap" else
           {"check": "twins", "prompt_tokens": 3, "b_reuse": "exact", "b_reuse_copy": True, "b_cached": 3,
            "b_after_a_s": 0.4, "b": {"ttft_s": 15.6}})
    json.dump(res, open(opt("--out"), "w")); print(json.dumps(res)); sys.exit(0)''')
FAKE = FAKE.replace('''        rows.append(row(None, None, 0, int(opt("--max-tokens", "2"))))''',
                    '''        kind = opt("--expect-reuse")
        rows.append(row(None, kind if kind not in (None, "none", "any") else None, 67584 if kind == "checkpoint" else 0,
                        int(opt("--max-tokens", "2"))))''')
assert "f4_live.py" in FAKE and "67584" in FAKE


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "stub"; stub.mkdir()
    for name, text in {"docker": DOCKER, "python3": PYTHON3, **SIMPLE}.items():
        p = stub / name; p.write_text(text); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (stub / "fake_runner.py").write_text(FAKE)
    home = tmp_path / "home"; home.mkdir()
    (tmp_path / "repo").mkdir()
    manifests = tmp_path / "manifests"; manifests.mkdir()
    for m in ("m32k", "m128k"):
        (manifests / f"{m}.json").write_text(json.dumps({"tokens": 3, "ids": [5, 6, 7], "sha256": "x"}))
    (manifests / "cP.json").write_text(json.dumps({"ids": list(range(10))}))
    (manifests / "cPp.json").write_text(json.dumps({"ids": list(range(6)) + [99, 98]}))
    deps = tmp_path / "deps"; deps.mkdir()
    e = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(home), "REPO": str(tmp_path / "repo"),
         "OUT": str(tmp_path / "out"), "M": str(manifests), "DEPS": str(deps),
         "STUB_LOG": str(tmp_path / "calls.log"), "STUB_DIR": str(stub), "REAL_PY": sys.executable,
         "BENCH_DIR": str(BENCH), "READY_DEADLINE": "10"}
    for k in ("SHARED_GPU", "STUB_FAIL_START", "STUB_DIFFER", "PARALLEL", "TENSORFOLD_PREFILL_ROWS", "CHECKPOINTS"):
        e.pop(k, None)
    return e


def run(env, *args, **extra):
    env = {**env, **{k: str(v) for k, v in extra.items()}}
    return subprocess.run(["bash", str(HERE / "f4-test.sh"), *args], env=env, capture_output=True, text=True, timeout=120)


def test_script_parses():
    subprocess.run(["bash", "-n", str(HERE / "f4-test.sh")], check=True)


def test_dry_run_starts_and_stops_nothing(env):
    r = run(env, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    assert not any(l.startswith(("server-start", "docker stop -t 60 oj-serve")) for l in log)
    assert sum("f4_live.py" in l and "--dry-run" in l for l in log) == 2
    assert sum("prefill_bench.py" in l and "--dry-run" in l for l in log) == 2
    assert any("bench/vision_probe.py" in l and "--quick" in l and "--dry-run" in l for l in log)
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_full_window(env):
    r = run(env)
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    tests = [i for i, l in enumerate(log) if l.startswith("docker run") and "python -m pytest" in l]
    prod = [i for i, l in enumerate(log) if l.startswith("docker stop -t 60 oj-serve")]
    starts = [i for i, l in enumerate(log) if l.startswith("server-start")]
    assert len(tests) == 1 and len(prod) == 1 and len(starts) == 2 and tests[0] < prod[0] < starts[0]
    for name in ("test_flashnext_multi.py", "test_qwen4_exp_nvfp4.py", "test_prefix_reuse_flashnext.py",
                 "test_prefix_checkpoints.py", "test_cuda_prompts_inside_rounds.py", "/deps"):
        assert name in log[tests[0]], name
    assert all("pause=1 parallel=3" in log[i] for i in starts)                  # the pause is held across both servers
    servers = [l for l in log if l.startswith("docker run") and " serve " in f" {l} "]
    assert all("--vision" in s for s in servers) and "--prefix-checkpoints 0" not in servers[0]
    assert "--prefix-checkpoints 0" in servers[1]
    live = [i for i, l in enumerate(log) if "f4_live.py" in l and "--dry-run" not in l]
    probe = [i for i, l in enumerate(log) if "bench/vision_probe.py" in l and "--dry-run" not in l]
    assert len(live) == 2 and len(probe) == 1 and all(starts[0] < i < starts[1] for i in live + probe)
    out = Path(env["OUT"])
    summary = (out / "f4-summary.txt").read_text()
    assert "F4 live checks: PASS" in summary and "reuse checkpoint, cached 67584" in summary
    assert "P' shares 6 tokens" in summary
    assert json.loads((out / "f4-text-exactness.json").read_text())["checkpoints_vs_off_equal"] is True
    tl = (out / "f4-timeline.txt").read_text()
    assert tl.count("standby watcher paused") == 1 and tl.count("standby watcher resumed") == 1
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_checkpoints_knob_and_a_differing_text_fails(env):
    r = run(env, "--steps", "7", STUB_DIFFER=1, CHECKPOINTS=8)
    assert r.returncode == 1 and "DIFFERENT" in r.stdout
    servers = [l for l in calls(env) if l.startswith("docker run") and " serve " in f" {l} "]
    assert "--prefix-checkpoints 8" in servers[0] and "--prefix-checkpoints 0" in servers[1]
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_a_failed_start_cleans_up_and_bad_arguments(env):
    r = run(env, "--steps", "2", STUB_FAIL_START=1)
    assert r.returncode != 0 and not Path(env["HOME"], ".spark-standby.pause").exists()
    assert run(env, "--steps", "8").returncode == 2
    Path(env["M"], "cPp.json").unlink()
    r = run(env)
    assert r.returncode == 2 and "cPp.json missing" in r.stderr
    assert run(env, "--steps", "1,2,3,4,6,7", "--dry-run").returncode == 0
