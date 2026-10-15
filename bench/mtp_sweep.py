#!/usr/bin/env python3
"""F10 decode tuning: drafted decode speed of one running server over ~20 code and ~20 chat/prose prompts.

  mtp_sweep.py run BASE MODEL --label conf0.3-d6 --out rows.jsonl [--tokens 200] [--code 20] [--chat 20]
               [--samplings greedy,sampled] [--setting confidence=0.3 --setting depth=6] [--dry-run]
  mtp_sweep.py summary rows-a.jsonl rows-b.jsonl ... [--json summary.json]

Prompts (fixed, the same for every setting): code = the first N HumanEval prompts sent raw to /v1/completions (the
model completes the function, as the published "code prompt" cell does); chat = GSM8K test questions answered
briefly plus short prose requests on /v1/chat/completions with thinking off. Every request: `max_tokens` replies with
`ignore_eos` (equal lengths), greedy (temperature 0) and sampled (temperature 1, top_k 20, top_p 0.95, a fixed seed
per prompt, so every setting samples the same tokens: exact sampling), streamed, `return_token_ids`.

Per row: decode tok/s = (completion tokens - 1) / (last text chunk - first text chunk), the span the published
tables use; the server's own decode_s and rounds from its stats block when present, giving tokens per round and
accepted drafts per round (tokens per round - 1). The summary prints, per setting and (category, sampling), the
median and interquartile range of decode tok/s, the median accepted drafts per round, and a recommendation:

  RULE: the setting with the largest geometric mean of its four medians (code and chat, greedy and sampled); within
  1% of the best, the setting labelled as the default (--default-label, else the first file) wins, then the higher
  confidence. A winner that loses more than 3% on any one of the four cells against the default is reported with
  that cell named (a judgement call for the owner, not an automatic switch).

Token ids are hashed per (prompt, sampling); the summary checks every setting produced the same hashes (drafting
never changes a reply) and names any mismatch. Standard library only, so it runs on a bare host. HumanEval and GSM8K
come from acc_eval.py's cache (~/.cache/octojet/evaldata, OCTOJET_EVAL_CACHE), downloaded on first use.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acc_eval import GSM_URL, HE_URL, fetch  # noqa: E402

PROSE = [
    "Write a short paragraph about why the sky looks blue.",
    "Describe a quiet morning in a small coastal town in a few sentences.",
    "Give three tips for keeping houseplants healthy, one sentence each.",
    "Explain what a hash table is to a new programmer, briefly.",
    "Write a short, friendly email declining a meeting invitation.",
    "Summarise the plot of a classic fairy tale of your choice in a paragraph.",
    "Explain the difference between weather and climate in plain English.",
    "Write a four-line poem about autumn leaves.",
    "Describe how a bicycle's gears make climbing hills easier.",
    "Give a short overview of how vaccines train the immune system.",
    "Write a brief product description for a reusable water bottle.",
    "Explain why bread rises when it bakes, in a few sentences.",
]
GSM_SUFFIX = "\n\nAnswer briefly: a few short steps, then the final number."
SAMPLINGS = {"greedy": {"temperature": 0.0}, "sampled": {"temperature": 1.0, "top_k": 20, "top_p": 0.95}}


def build_prompts(n_code: int, n_chat: int, *, he=None, gsm=None) -> list[dict]:
    """[{id, category, kind, prompt}]: n_code HumanEval completions, then n_chat chat prompts (GSM8K questions and
    prose requests alternating, as many of each as the count asks)."""

    if n_code and he is None:
        he = [json.loads(line) for line in gzip.open(fetch(HE_URL, "HumanEval.jsonl.gz"), "rt")]
    n_prose = min(len(PROSE), n_chat // 2)
    n_gsm = n_chat - n_prose
    if n_gsm and gsm is None:
        gsm = [json.loads(line) for line in open(fetch(GSM_URL, "gsm8k_test.jsonl"))]
    out = [{"id": item["task_id"], "category": "code", "kind": "completion", "prompt": item["prompt"]}
           for item in (he or [])[:n_code]]
    chat = [{"id": f"gsm8k/{i}", "category": "chat", "kind": "chat", "prompt": item["question"] + GSM_SUFFIX}
            for i, item in enumerate((gsm or [])[:n_gsm])]
    prose = [{"id": f"prose/{i}", "category": "chat", "kind": "chat", "prompt": p} for i, p in enumerate(PROSE[:n_prose])]
    for i in range(max(len(chat), len(prose))):                      # alternate, so a short run still mixes both
        out += chat[i:i + 1] + prose[i:i + 1]
    return out


def body_for(model: str, item: dict, sampling: str, tokens: int, seed: int) -> tuple[str, dict]:
    body = {"model": model, "max_tokens": tokens, "stream": True, "stream_options": {"include_usage": True},
            "ignore_eos": True, "return_token_ids": True, **SAMPLINGS[sampling]}
    if sampling != "greedy":
        body["seed"] = seed
    if item["kind"] == "chat":
        body["messages"] = [{"role": "user", "content": item["prompt"]}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
        return "/v1/chat/completions", body
    body["prompt"] = item["prompt"]
    return "/v1/completions", body


def stream(base: str, path: str, body: dict, timeout: float = 900) -> dict:
    """Send one streamed request; client-side first/last text times, usage and the stats block."""

    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    usage, stats = {}, {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            stats = chunk.get("octojet") or chunk.get("tensorfold") or stats
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                if choice.get("text") or (choice.get("delta") or {}).get("content"):
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
    return {"start": start, "first": first, "last": last, "usage": usage, "stats": stats or {}}


def row_for(label: str, setting: dict, item: dict, sampling: str, got: dict) -> dict:
    n = got["usage"].get("completion_tokens")
    span = (got["last"] - got["first"]) if got["first"] is not None and got["last"] is not None else None
    st = got["stats"]
    rounds, server_s = st.get("rounds"), st.get("decode_s")
    ids = st.get("token_ids")
    tpr = (n - 1) / rounds if n and rounds else None
    return {"label": label, **{f"setting_{k}": v for k, v in setting.items()}, "id": item["id"],
            "category": item["category"], "sampling": sampling, "completion_tokens": n,
            "decode_tps": (n - 1) / span if n and span else None,
            "server_decode_tps": (n - 1) / server_s if n and server_s else None,
            "rounds": rounds, "tokens_per_round": tpr, "accepted_per_round": tpr - 1 if tpr is not None else None,
            "ttft_s": (got["first"] - got["start"]) if got["first"] is not None else None,
            "token_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16] if isinstance(ids, list) else None}


def cmd_run(a) -> int:
    setting = {}
    for kv in a.setting:
        k, _, v = kv.partition("=")
        setting[k] = float(v) if v.replace(".", "", 1).isdigit() else v
    items = build_prompts(a.code, a.chat)
    samplings = [s for s in a.samplings.split(",") if s]
    for s in samplings:
        if s not in SAMPLINGS:
            raise SystemExit(f"unknown sampling {s!r} (greedy, sampled)")
    plan = [(item, s) for s in samplings for item in items]
    print(f"{a.label}: {len(items)} prompts ({sum(i['category'] == 'code' for i in items)} code, "
          f"{sum(i['category'] == 'chat' for i in items)} chat) x {len(samplings)} samplings = {len(plan)} requests "
          f"of {a.tokens} tokens", flush=True)
    if a.dry_run:
        return 0
    path, body = body_for(a.model, items[0], "greedy", min(32, a.tokens), 0)
    stream(a.base, path, body)                                       # warm-up, excluded
    with open(a.out, "a") as f:
        for i, (item, s) in enumerate(plan):
            path, body = body_for(a.model, item, s, a.tokens, 1234 + items.index(item))
            row = row_for(a.label, setting, item, s, stream(a.base, path, body))
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(plan)}", flush=True)
    print(format_summary(summarise(load([a.out]), default=a.label)))
    return 0


def load(paths) -> list[dict]:
    return [json.loads(line) for p in paths for line in open(p) if line.strip()]


def quartiles(v: list[float]) -> tuple[float, float, float]:
    v = sorted(v)
    if len(v) == 1:
        return v[0], v[0], v[0]
    q = statistics.quantiles(v, n=4, method="inclusive")
    return q[0], statistics.median(v), q[2]


def summarise(rows: list[dict], default: str | None = None) -> dict:
    """{settings: {label: {cells: {"code/greedy": {...}}, geomean, confidence}}, recommendation, mismatches}."""

    labels = list(dict.fromkeys(r["label"] for r in rows))
    default = default if default in labels else (labels[0] if labels else None)
    settings = {}
    for label in labels:
        mine = [r for r in rows if r["label"] == label]
        cells = {}
        for cat in ("code", "chat"):
            for s in SAMPLINGS:
                tps = [r["decode_tps"] for r in mine if r["category"] == cat and r["sampling"] == s and r["decode_tps"]]
                if not tps:
                    continue
                q1, med, q3 = quartiles(tps)
                acc = [r["accepted_per_round"] for r in mine if r["category"] == cat and r["sampling"] == s
                       and r.get("accepted_per_round") is not None]
                cells[f"{cat}/{s}"] = {"n": len(tps), "median": med, "q1": q1, "q3": q3,
                                       "accepted_per_round": statistics.median(acc) if acc else None}
        meds = [c["median"] for c in cells.values()]
        conf = next((r.get("setting_confidence") for r in mine if r.get("setting_confidence") is not None), None)
        settings[label] = {"cells": cells, "confidence": conf,
                           "geomean": math.exp(sum(math.log(m) for m in meds) / len(meds)) if meds else None}
    scored = [lb for lb in labels if settings[lb]["geomean"]]
    rec = None
    if scored:
        best = max(settings[lb]["geomean"] for lb in scored)
        near = [lb for lb in scored if settings[lb]["geomean"] >= 0.99 * best]
        rec = default if default in near else max(near, key=lambda lb: (settings[lb]["confidence"] or 0,
                                                                         settings[lb]["geomean"]))
    losses = []
    if rec and default and rec != default:
        for cell, c in settings[rec]["cells"].items():
            base = settings[default]["cells"].get(cell)
            if base and c["median"] < 0.97 * base["median"]:
                losses.append(f"{cell} {c['median']:.1f} vs {base['median']:.1f}")
    by_key = {}
    for r in rows:
        if r.get("token_sha"):
            by_key.setdefault((r["id"], r["sampling"]), {}).setdefault(r["token_sha"], []).append(r["label"])
    mismatches = [f"{k[0]} {k[1]}: " + " / ".join(f"{','.join(v)}" for v in shas.values())
                  for k, shas in sorted(by_key.items()) if len(shas) > 1]
    return {"settings": settings, "default": default, "recommendation": rec, "recommendation_losses": losses,
            "mismatches": mismatches}


def format_summary(s: dict) -> str:
    cols = [f"{c}/{m}" for c in ("code", "chat") for m in SAMPLINGS]
    lines = ["setting | " + " | ".join(f"{c} median (IQR) acc/round" for c in cols) + " | geomean",
             "|".join(["---"] * (len(cols) + 2))]
    for label, st in s["settings"].items():
        parts = []
        for c in cols:
            cell = st["cells"].get(c)
            if cell is None:
                parts.append("-")
                continue
            acc = "-" if cell["accepted_per_round"] is None else f"{cell['accepted_per_round']:.2f}"
            parts.append(f"{cell['median']:.1f} ({cell['q1']:.1f}-{cell['q3']:.1f}) {acc}")
        g = "-" if st["geomean"] is None else f"{st['geomean']:.1f}"
        lines.append(f"{label}{' (default)' if label == s['default'] else ''} | " + " | ".join(parts) + f" | {g}")
    lines.append("rule: largest geometric mean of the four medians; within 1% the default, then the higher confidence")
    rec = s["recommendation"]
    lines.append(f"recommendation: {rec}" + (" (keep the default)" if rec == s["default"] else "")
                 + (f"; loses >3% vs the default on: {', '.join(s['recommendation_losses'])}"
                    if s["recommendation_losses"] else ""))
    if s["mismatches"]:
        lines.append(f"TOKEN MISMATCH across settings ({len(s['mismatches'])}): " + "; ".join(s["mismatches"][:5]))
    else:
        lines.append("token ids: every setting produced the same replies")
    return "\n".join(lines)


def cmd_summary(a) -> int:
    s = summarise(load(a.files), default=a.default_label)
    print(format_summary(s))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(s, f, indent=1)
    return 1 if s["mismatches"] else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("base"); r.add_argument("model")
    r.add_argument("--label", required=True); r.add_argument("--out", required=True)
    r.add_argument("--tokens", type=int, default=200)
    r.add_argument("--code", type=int, default=20); r.add_argument("--chat", type=int, default=20)
    r.add_argument("--samplings", default="greedy,sampled")
    r.add_argument("--setting", action="append", default=[], help="key=value recorded on every row (confidence=0.3)")
    r.add_argument("--dry-run", action="store_true", help="build the prompt set and print the plan; send nothing")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+"); s.add_argument("--json"); s.add_argument("--default-label")
    s.set_defaults(fn=cmd_summary)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
