#!/usr/bin/env python3
"""A/B accuracy for one OpenAI-compatible endpoint: GSM8K (first N of test) + HumanEval (164).

Greedy, thinking off, identical requests for every engine. Writes <out>.jsonl (one row per item, with the reply)
and prints a summary. HumanEval programs are only written out here; run_humaneval.sh executes them sandboxed.

  acc_eval.py BASE MODEL OUT [--gsm N] [--he N] [--workers W]
"""
import argparse, gzip, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

GSM_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
HE_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
CACHE = os.environ.get("OCTOJET_EVAL_CACHE", os.path.expanduser("~/.cache/octojet/evaldata"))


def fetch(url, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def chat(base, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.load(r)
            return d["choices"][0]["message"].get("content") or "", d.get("usage", {})
        except Exception as e:  # noqa: BLE001 - retry any transport error, then record it
            err = repr(e)
            time.sleep(3)
    return "", {"error": err}


def gsm_answer(text):
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", text)
    nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def code_block(text):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--gsm", type=int, default=250); ap.add_argument("--he", type=int, default=164); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    gsm = [json.loads(l) for l in open(fetch(GSM_URL, "gsm8k_test.jsonl"))][: a.gsm]
    he = [json.loads(l) for l in gzip.open(fetch(HE_URL, "HumanEval.jsonl.gz"), "rt")][: a.he]

    def do_gsm(i_item):
        i, it = i_item
        p = it["question"] + "\n\nSolve it step by step briefly, then end with a final line 'Answer: <number>'."
        text, usage = chat(a.base, a.model, p, 1024)
        gold = float(it["answer"].split("####")[-1].strip().replace(",", ""))
        pred = gsm_answer(text)
        return {"task": "gsm8k", "id": i, "ok": pred is not None and abs(pred - gold) < 1e-6,
                "gold": gold, "pred": pred, "usage": usage, "reply": text}

    def do_he(it):
        p = ("Complete the following Python function. Reply with the complete function (including the signature "
             "and any imports it needs) in one ```python code block.\n\n" + it["prompt"])
        text, usage = chat(a.base, a.model, p, 1024)
        prog = code_block(text) + "\n\n" + it["test"] + f"\n\ncheck({it['entry_point']})\n"
        return {"task": "humaneval", "id": it["task_id"], "program": prog, "usage": usage, "reply": text}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(do_gsm, enumerate(gsm))) + list(ex.map(do_he, he))
    with open(a.out + ".jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    g = [r for r in rows if r["task"] == "gsm8k"]
    errs = sum(1 for r in rows if "error" in r["usage"])
    print(json.dumps({"model": a.model, "gsm8k_acc": round(sum(r["ok"] for r in g) / len(g), 4), "gsm8k_n": len(g),
                      "humaneval_programs": len(rows) - len(g), "transport_errors": errs,
                      "wall_s": round(time.time() - t0)}))


if __name__ == "__main__":
    sys.exit(main())
