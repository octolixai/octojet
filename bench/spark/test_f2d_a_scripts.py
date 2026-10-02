"""The F2d stage-A operator scripts under stubs (docker, nvidia-smi, curl, ss, free, python3): the sequence of starts and
cache drops, the standby pause switch on every exit path, the dry-run before any start, the gate invocation, and the
argument / tenant refusals. Nothing here touches a GPU or the network."""

import json, os, re, stat, subprocess, sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent                      # bench/spark
BENCH = HERE.parent
SCRIPTS = ["f2d-a-lib.sh", "f2d-a-manifests.sh", "f2d-a-exactness.sh", "f2d-a-replay.sh"]

DOCKER = r'''#!/usr/bin/env bash
printf '%s\n' "docker $*" | tr '\n' ' ' >> "$STUB_LOG"; echo >> "$STUB_LOG"   # one line per call: -c scripts span lines
pause(){ [ -e "$HOME/.spark-standby.pause" ] && echo 1 || echo 0; }
case "$1" in
  run)
    if printf '%s\n' "$@" | grep -q -- '--privileged'; then echo "drop_caches pause=$(pause)" >> "$STUB_LOG"; exit 0; fi
    if printf '%s\n' "$@" | grep -qx -- 'serve'; then
      n=$(grep -c '^server-start' "$STUB_LOG" || true); n=$((n + 1))
      echo "server-start $n pause=$(pause) parallel=$(printf '%s\n' "$@" | grep -A1 -x -- '--parallel' | tail -1)" >> "$STUB_LOG"
      if [ "${STUB_FAIL_START:-0}" = "$n" ]; then echo "boom: injected start failure"; exit 1; fi
      echo "[octojet] loaded in 59.2 s"; echo "2 streams of 262144 prompt/reply tokens"; echo "serving f1 at http://0.0.0.0:8080"
      echo $$ > "$STUB_DIR/server.pid"; sleep 300 & CHILD=$!; trap 'kill $CHILD 2>/dev/null; exit 0' TERM; wait $CHILD; exit 0
    fi
    if printf '%s\n' "$@" | grep -q -- '-m pytest'; then echo "STAGE-A TESTS rc=0"; echo "CHUNK-SIZE TEST (stage-B evidence) rc=0"; exit 0; fi
    if printf '%s\n' "$@" | grep -q -- 'prefill_prompt.py'; then echo "$(printf '%s\n' "$@" | grep -o 'prefill_prompt.py' | wc -l | tr -d ' ') manifests"; exit 0; fi
    exit 0;;
  stop) if [ -f "$STUB_DIR/server.pid" ]; then kill "$(cat "$STUB_DIR/server.pid")" 2>/dev/null || true; rm -f "$STUB_DIR/server.pid"; fi; exit 0;;
  image) echo "sha256:stubimage"; exit 0;;
  ps) exit 0;;
esac
exit 0
'''
PYTHON3 = r'''#!/usr/bin/env bash
echo "python3 $*" >> "$STUB_LOG"
case "$1" in
  -) exec "$REAL_PY" - "${@:2}";;
  bench/f2d_gate.py) exec "$REAL_PY" "$BENCH_DIR/f2d_gate.py" "${@:2}";;
  *) exec "$REAL_PY" "$STUB_DIR/fake_runner.py" "$@";;
esac
'''
FAKE_RUNNER = r'''
"""Stands in for prefill_bench.py, bench_openai.py and concurrent_equal.py: writes plausible rows/receipts."""
import json, sys
argv = sys.argv[1:]
def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default
script = argv[0]
if script.endswith("bench_openai.py"):
    json.dump({"stub": "bench_openai", "expect_equal": "--expect-equal" in argv}, open(opt("--output"), "w")); sys.exit(0)
if script.endswith("concurrent_equal.py"):
    json.dump({"stub": "concurrent_equal"}, open(opt("--out"), "w")); sys.exit(0)
if "--dry-run" in argv:
    print("dry-run stub"); sys.exit(0)
out, stream = opt("--out"), "--no-stream" not in argv
meta = {"server_label": opt("--server-label"), "stage": opt("--stage"), "rep_id": int(opt("--rep")) if opt("--rep") else None}
def row(step, reuse, cached, max_tokens):
    cold = reuse is None and cached == 0
    ttft = 37.6 if cold else 1.2
    return {**meta, "step": step, "stream": stream, "reuse": reuse, "cached": cached, "max_tokens": max_tokens, "ttft_s": ttft,
            "reply_sha": "s", "prompt_sha_ok": True, "prompt_tokens": 71444, "arm": "clean", "draft": True, "rep": 1, "label": opt("--label"),
            "usage": {"prompt_tokens": 71444, "completion_tokens": max_tokens}, "decode_s": 0.9,
            "total_s": None if stream else ttft + 0.7, "client_send_utc": "2026-10-01T00:00:00.000Z",
            "client_complete_utc": "2026-10-01T00:00:40.000Z"}
rows = []
if "--replay" in argv:
    for s in json.load(open(opt("--replay"))):
        reuse = None if s["expect_reuse"] in ("none", "any") else s["expect_reuse"]
        cached = 0 if s["expect_cached"] == "any" else s["expect_cached"]
        rows.append(row(s["label"], reuse, cached, s["max_tokens"]))
else:
    for _ in range(int(opt("--concurrent", "1"))):
        rows.append(row(None, None, 0, int(opt("--max-tokens", "2"))))
with open(out, "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"fake runner: {len(rows)} rows -> {out}")
'''
SIMPLE = {"nvidia-smi": "#!/usr/bin/env bash\nexit 0\n", "ss": "#!/usr/bin/env bash\nexit 0\n",
          "free": "#!/usr/bin/env bash\necho 'Mem: stub'\n", "curl": "#!/usr/bin/env bash\necho '{\"data\":[{\"id\":\"f1\"}]}'\n"}


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "stub"; stub.mkdir()
    for name, text in {"docker": DOCKER, "python3": PYTHON3, **SIMPLE}.items():
        p = stub / name; p.write_text(text); p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (stub / "fake_runner.py").write_text(FAKE_RUNNER)
    home = tmp_path / "home"; home.mkdir()
    repo = tmp_path / "repo" / "bench" / "spark"; repo.mkdir(parents=True)
    for f in HERE.glob("f2d-a-*.json"):
        (repo / f.name).write_text(f.read_text())
    manifests = tmp_path / "manifests"; manifests.mkdir()
    for m in ("cP", "cPp", "cPpp", "cP2", "cP2p", "cP3"):
        (manifests / f"{m}.json").write_text(json.dumps({"tokens": 1, "ids": [1], "sha256": "x"}))
    out = tmp_path / "out"
    e = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(home), "REPO": str(tmp_path / "repo"),
         "OUT": str(out), "M": str(manifests), "STUB_LOG": str(tmp_path / "calls.log"), "STUB_DIR": str(stub),
         "REAL_PY": sys.executable, "BENCH_DIR": str(BENCH), "READY_DEADLINE": "10"}
    e.pop("SHARED_GPU", None); e.pop("STUB_FAIL_START", None)
    return e


def run(script, env, *args, **extra):
    env = {**env, **{k: str(v) for k, v in extra.items()}}
    return subprocess.run(["bash", str(HERE / script), *args], env=env, capture_output=True, text=True, timeout=120)


def calls(env):
    return Path(env["STUB_LOG"]).read_text().splitlines() if Path(env["STUB_LOG"]).exists() else []


def test_scripts_parse():
    for s in SCRIPTS:
        subprocess.run(["bash", "-n", str(HERE / s)], check=True)


def test_replay_refuses_bad_arguments_and_shared_gpu_without_starting(env):
    for args, extra in ([[], {}], [["--caller-timeout", "0"], {}], [["--caller-timeout", "abc"], {}],
                        [["--caller-timeout", "-3"], {}], [["--bogus"], {}], [["--caller-timeout", "30"], {"SHARED_GPU": "1"}]):
        r = run("f2d-a-replay.sh", env, *args, **extra)
        assert r.returncode == 2, (args, extra, r.stderr)
        assert not any(l.startswith("server-start") for l in calls(env)), (args, extra)
    Path(env["M"], "cP2p.json").unlink()
    r = run("f2d-a-replay.sh", env, "--caller-timeout", "30")
    assert r.returncode == 2 and "cP2p.json missing" in r.stderr and not any(l.startswith("server-start") for l in calls(env))
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_replay_sequence_pause_switch_dry_run_and_gate(env):
    r = run("f2d-a-replay.sh", env, "--caller-timeout", "30")
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    starts = [i for i, l in enumerate(log) if l.startswith("server-start")]
    assert len(starts) == 6 and all("pause=1 parallel=2" in log[i] for i in starts)
    for i in starts:                                                   # each start is preceded by a cache drop under the pause
        before = [l for l in log[:i] if l.startswith("drop_caches")]
        assert before and before[-1] == "drop_caches pause=1"
    assert sum(l.startswith("drop_caches") for l in log) == 6
    dry = [i for i, l in enumerate(log) if l.startswith("python3 bench/prefill_bench.py") and "--dry-run" in l]
    assert len(dry) == 8 and max(dry) < starts[0]                      # 4 replays + bursts + 3 probes, all before the first start
    runs = [l for l in log if l.startswith("python3 bench/prefill_bench.py") and "--dry-run" not in l]
    assert len(runs) == 8
    assert sum("f2d-a-replay-64.json" in l and "--no-stream" in l for l in runs) == 2
    assert sum("f2d-a-replay.json" in l for l in runs) == 2 and not any("f2d-a-replay.json" in l and "--no-stream" in l for l in runs)
    assert sum("f2d-a-bursts.json" in l for l in runs) == 1 and sum("--concurrent 2" in l for l in runs) == 1
    assert [l for l in runs if "--concurrent 2" in l][0].count("cPp.json") == 1
    third = [l for l in runs if "cP2.json" in l and "--manifest" in l]
    assert len(third) == 1 and "--expect-reuse none --expect-cached 0" in third[0]      # the retry probe's prediction
    gate = [l for l in log if l.startswith("python3 bench/f2d_gate.py")]
    assert len(gate) == 1 and "--caller-timeout 30" in gate[0] and "--prompt-tokens 71444" in gate[0]
    assert all(f"f2d-a-{n}.jsonl" in gate[0] for n in ("stream-1", "stream-2", "nostream-1", "nostream-2"))
    out = Path(env["OUT"])
    assert json.loads((out / "f2d-a-gate.json").read_text())["verdict"] == "PASS"
    meta = json.loads((out / "f2d-a-meta.json").read_text())
    assert meta["caller_timeout_s"] == 30.0 and meta["parallel"] == 2 and meta["shared_gpu"] is False and meta["image_id"] == "sha256:stubimage"
    assert "--parallel 2" in meta["server_args"] and "--kv-dtype int8" in meta["server_args"]
    tl = (out / "f2d-a-timeline.txt").read_text().splitlines()
    stamped = [l for l in tl if "page cache dropped" in l or ": server start" in l or "standby watcher" in l or ": stopped" in l]
    assert sum("page cache dropped" in l for l in tl) == 6 and len(stamped) >= 30
    assert all(re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ", l) for l in stamped), stamped[:3]
    assert sum("standby watcher paused" in l for l in tl) == 6 and sum("standby watcher resumed" in l for l in tl) == 6
    for n in ("stream-1", "stream-2", "nostream-1", "nostream-2", "bursts", "retry"):
        assert (out / f"f2d-a-{n}-server.log").exists() and (out / f"f2d-a-{n}-args.txt").exists()
    assert len((out / "f2d-a-retry.jsonl").read_text().splitlines()) == 4     # 1 + 2 (concurrent) + 1
    assert (out / "f2d-a-bursts.json").exists() and "__M__" not in (out / "f2d-a-bursts.json").read_text()
    assert sorted(p.name for p in out.glob("cP*.json")) == ["cP.json", "cP2.json", "cP2p.json", "cP3.json", "cPp.json", "cPpp.json"]
    assert not Path(env["HOME"], ".spark-standby.pause").exists()


def test_retry_probe_expectation_follows_parallel(env):
    r = run("f2d-a-replay.sh", env, "--caller-timeout", "30", PARALLEL=3)
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    assert all("parallel=3" in l for l in log if l.startswith("server-start"))
    runs = [l for l in log if l.startswith("python3 bench/prefill_bench.py") and "--dry-run" not in l]
    third = [l for l in runs if "cP2.json" in l and "--manifest" in l]
    assert len(third) == 1 and "--expect-reuse extend --expect-cached 71444" in third[0]
    assert json.loads((Path(env["OUT"]) / "f2d-a-meta.json").read_text())["parallel"] == 3


def test_replay_failed_start_removes_only_our_pause(env):
    r = run("f2d-a-replay.sh", env, "--caller-timeout", "30", STUB_FAIL_START=2)
    assert r.returncode != 0
    log = calls(env)
    assert sum(l.startswith("server-start") for l in log) == 2
    assert not Path(env["HOME"], ".spark-standby.pause").exists()
    assert not any(l.startswith("python3 bench/f2d_gate.py") for l in log)


def test_a_pre_existing_pause_is_left_in_place(env):
    Path(env["HOME"], ".spark-standby.pause").write_text("theirs")
    r = run("f2d-a-replay.sh", env, "--caller-timeout", "30")
    assert r.returncode == 0, r.stdout + r.stderr
    assert Path(env["HOME"], ".spark-standby.pause").read_text() == "theirs"
    assert "not ours" in (Path(env["OUT"]) / "f2d-a-timeline.txt").read_text()


def test_exactness_runs_tests_then_the_acceptance_server(env):
    r = run("f2d-a-exactness.sh", env)
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls(env)
    tests = [i for i, l in enumerate(log) if l.startswith("docker run") and "python -m pytest" in l]   # tmp paths contain "pytest" too
    assert len(tests) == 1, [log[i][:160] for i in tests]
    assert "OCTOJET_NVFP4_FLASHNEXT_LAYERS=4" in log[tests[0]] and "-k 'not chunk_sizes'" in log[tests[0]]
    assert "test_prefix_reuse_flashnext.py -k chunk_sizes" in log[tests[0]]
    starts = [i for i, l in enumerate(log) if l.startswith("server-start")]
    assert len(starts) == 1 and starts[0] > tests[0] and "pause=1" in log[starts[0]]
    runs = [l for l in log[starts[0]:] if l.startswith("python3")]
    assert "--concurrent 2" in runs[0] and "--expect-reuse any" in runs[0] and "cP.json" in runs[0]
    assert "--expect-reuse exact --expect-cached 71444" in runs[1] and "--concurrent" not in runs[1]
    assert runs[2].startswith("python3 engine/tools/bench_openai.py") and "--expect-equal" in runs[2] and "--output" in runs[2]
    assert runs[3].startswith("python3 bench/concurrent_equal.py") and "--prompts 2" in runs[3] and "--out" in runs[3]
    out = Path(env["OUT"])
    assert "STAGE-A TESTS rc=0" in (out / "f2d-a-cuda-tests.txt").read_text()
    for n in ("f2d-a-acceptance.jsonl", "f2d-a-bench-openai.json", "f2d-a-concurrent-equal.json", "f2d-a-acceptance-server.log", "cP.json"):
        assert (out / n).exists(), n
    assert not Path(env["HOME"], ".spark-standby.pause").exists()
    Path(env["M"], "cP.json").unlink()
    r = run("f2d-a-exactness.sh", env)
    assert r.returncode == 2 and "cP.json missing" in r.stderr


def test_exactness_failed_start_still_cleans_up(env):
    r = run("f2d-a-exactness.sh", env, STUB_FAIL_START=1)                # the acceptance server never gets healthy
    assert r.returncode != 0
    log = calls(env)
    assert sum(l.startswith("server-start") for l in log) == 1 and not any(l.startswith("python3 engine/tools/bench_openai.py") for l in log)
    assert any(l.startswith("docker stop") for l in log)                # cleanup ran: our container stopped ...
    assert not Path(env["HOME"], ".spark-standby.pause").exists()        # ... the pause removed ...
    tl = (Path(env["OUT"]) / "f2d-a-timeline.txt").read_text()
    assert "standby watcher resumed" in tl and "exit 1" in tl              # ... and the real exit code logged


def test_manifests_script_builds_six_manifests(env):
    r = run("f2d-a-manifests.sh", env)
    assert r.returncode == 0, r.stdout + r.stderr
    build = [l for l in calls(env) if l.startswith("docker run") and "prefill_prompt.py" in l]
    assert len(build) == 1 and build[0].count("prefill_prompt.py") == 6
    for flag in ("--tokens 71444", "--take 69013 --tail 2470", "--extend 2083", "--take 71061 --tail 2470"):
        assert flag in build[0], flag
    assert build[0].count("--extend 2048") == 2
    assert len(list(Path(env["OUT"]).glob("cP*.json"))) == 6
