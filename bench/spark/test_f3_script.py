"""bench/spark/f3-test.sh under the F2d stubs (docker, nvidia-smi, curl, ss, free, python3): the dry-run before
anything, the tests beside production, production stopped once under a pause held for the whole window, the vision
server then the text server, the comparisons, and the pause removed on every exit path. No GPU, no network."""

import json, os, stat, subprocess, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_f2d_a_scripts import BENCH, DOCKER, FAKE_RUNNER, HERE, PYTHON3, SIMPLE, calls  # noqa: E402

FAKE = FAKE_RUNNER.replace(
    '''    json.dump({"stub": "bench_openai", "expect_equal": "--expect-equal" in argv}, open(opt("--output"), "w")); sys.exit(0)''',
    '''    sha = "differs" if (opt("--label") == "text" and __import__("os").environ.get("STUB_DIFFER")) else "same"
    json.dump([{"prompt": "p", "temperature": 0.0, "token_sha_all": [sha]}], open(opt("--output"), "w")); sys.exit(0)''')
assert FAKE != FAKE_RUNNER


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "stub"; stub.mkdir()
    for name, text in {"docker": DOCKER, "python3": PYTHON3, **SIMPLE}.items():
        p = stub / name; p.write_text(text); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (stub / "fake_runner.py").write_text(FAKE)
    home = tmp_path / "home"; home.mkdir()
    (tmp_path / "repo").mkdir()
    manifests = tmp_path / "manifests"; manifests.mkdir()
    for m in ("m32k", "m128k", "m210k"):
        (manifests / f"{m}.json").write_text(json.dumps({"tokens": 3, "ids": [5, 6, 7], "sha256": "x"}))
    media = tmp_path / "media"; media.mkdir()
    for f in ("robots-sim-observation.png", "towel-crumpled.jpg", "towel-flat.jpg", "towel-crumpled-1280.jpg", "towel-flat-1280.jpg"):
        (media / f).write_bytes(b"x")
    deps = tmp_path / "deps"; deps.mkdir()
    e = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(home), "REPO": str(tmp_path / "repo"),
         "OUT": str(tmp_path / "out"), "M": str(manifests), "MEDIA": str(media), "DEPS": str(deps),
         "STUB_LOG": str(tmp_path / "calls.log"), "STUB_DIR": str(stub), "REAL_PY": sys.executable,
         "BENCH_DIR": str(BENCH), "READY_DEADLINE": "10"}
    for k in ("SHARED_GPU", "STUB_FAIL_START", "STUB_DIFFER", "PARALLEL", "TENSORFOLD_PREFILL_ROWS"):
        e.pop(k, None)
    return e


def run(env, *args, **extra):
    env = {**env, **{k: str(v) for k, v in extra.items()}}
    return subprocess.run(["bash", str(HERE / "f3-test.sh"), *args], env=env, capture_output=True, text=True, timeout=120)


def test_dry_run_starts_and_stops_nothing(env):
    r = run(env, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    assert not any(l.startswith(("server-start", "docker stop -t 60 oj-serve")) for l in log)   # (the trap stops only oj-f3)
    assert sum("prefill_bench.py" in l and "--dry-run" in l for l in log) == 12     # 3 sizes x 2 reps x 2 servers
    assert any("bench/vision_probe.py" in l and "--dry-run" in l for l in log)
    assert not Path(env["HOME"], ".spark-standby.pause").exists()
    b = json.loads(Path(env["OUT"], "m32k-b.json").read_text())
    assert b["ids"][:-1] == [5, 6] and b["ids"][-1] != 7 and b["sha256"] != "x"


def test_full_window(env):
    r = run(env)
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    tests = [i for i, l in enumerate(log) if l.startswith("docker run") and "python -m pytest" in l]
    prod = [i for i, l in enumerate(log) if l.startswith("docker stop -t 60 oj-serve")]
    starts = [i for i, l in enumerate(log) if l.startswith("server-start")]
    assert len(tests) == 1 and len(prod) == 1 and len(starts) == 2 and tests[0] < prod[0] < starts[0]
    assert "OCTOJET_NVFP4_FLASHNEXT_LAYERS=4" in log[tests[0]] and "/deps" in log[tests[0]]
    assert "test_flashnext_vision.py" in log[tests[0]] and "register_width" in log[tests[0]]
    assert all("pause=1 parallel=3" in log[i] for i in starts)                  # the pause is held across both servers
    servers = [l for l in log if l.startswith("docker run") and " serve " in f" {l} "]
    assert "--vision" in servers[0] and "--vision" not in servers[1] and all("/deps:ro" in s for s in servers)
    probe = [i for i, l in enumerate(log) if "bench/vision_probe.py" in l and "--dry-run" not in l]
    assert len(probe) == 1 and starts[0] < probe[0] < starts[1]
    out = Path(env["OUT"])
    assert json.loads((out / "f3-text-exactness.json").read_text())["vision_vs_text_equal"] is True
    assert len((out / "f3-prefill-vision.jsonl").read_text().splitlines()) == 6
    assert "F2c baseline 227.01" in (out / "f3-prefill-summary.txt").read_text()
    tl = (out / "f3-timeline.txt").read_text()
    assert tl.count("standby watcher paused") == 1 and tl.count("standby watcher resumed") == 1
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_different_text_fails_and_a_failed_start_cleans_up(env):
    r = run(env, "--steps", "3", STUB_DIFFER=1)
    assert r.returncode == 1 and "DIFFERENT" in r.stdout
    assert not Path(env["HOME"], ".spark-standby.pause").exists()
    r = run({**env, "STUB_LOG": env["STUB_LOG"] + ".2"}, "--steps", "2", STUB_FAIL_START=1)
    assert r.returncode != 0 and not Path(env["HOME"], ".spark-standby.pause").exists()


def test_bad_arguments_and_missing_inputs(env):
    assert run(env, "--steps", "9").returncode == 2
    Path(env["M"], "m128k.json").unlink()
    r = run(env)
    assert r.returncode == 2 and "m128k.json missing" in r.stderr
    assert not any(l.startswith("server-start") for l in calls(env))
    assert run(env, "--steps", "1,2,3,5", "--dry-run").returncode == 0   # manifests are needed by step 4 only
