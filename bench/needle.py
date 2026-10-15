#!/usr/bin/env python3
"""Needle in a haystack over an OpenAI-compatible chat endpoint (F9: does YaRN keep long-context recall?).

  needle.py run BASE MODEL --out ROWS.jsonl [--sizes 128000,256000,450000] [--depths 0.1,0.5,0.9]
                [--filler FILE ...] [--seed N] [--label L] [--max-tokens 64] [--dry-run]
  needle.py report ROWS.jsonl [ROWS.jsonl ...]

Each case is one chat request (thinking off, greedy): a unique first line (so no kept prompt is reused: every case is a
cold prompt), a filler document of about SIZE tokens with one fact inserted at DEPTH of it ("The secret passcode for
the blue vault is 7391-HELIX.", random per run), then the question. A row records whether the reply contains the
passcode, the prompt tokens (the server's usage), the time to the first reply text and the total time.

Only the server needs to be reachable: the filler is engine/tools/prefill_cold.py's corpus (the Python standard
library's sources on this host) unless --filler names text files; special-token spellings (<|...|>) are removed. The
document is sized against the server's own tokenizer through Octojet's refusal receipt ("the rendered prompt has N
tokens"): a request asking for an impossible reply length is refused before any prefill, so sizing costs no GPU time.
A server without that receipt gets a 3.3 characters-a-token estimate. A case past the server's window is recorded
with the server's refusal.
"""
from __future__ import annotations

import argparse, json, os, random, re, sys, time, urllib.error, urllib.request
from pathlib import Path

COLORS = ["blue", "amber", "crimson", "jade", "violet", "silver", "ochre", "teal", "scarlet", "ivory"]
PLACES = ["vault", "archive", "lighthouse", "greenhouse", "observatory", "boathouse", "library", "foundry"]
WORDS = ["HELIX", "QUARTZ", "ORBIT", "MAPLE", "COBALT", "FALCON", "PRISM", "TUNDRA", "LOTUS", "EMBER"]
SPECIAL = re.compile(r"<\|[^|<>\n]{1,40}\|>")
RECEIPT = re.compile(r"rendered prompt has (\d+) tokens")
CHARS_PER_TOKEN = 3.3
IMPOSSIBLE_REPLY = 10**8


def corpus() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine" / "tools"))
    from prefill_cold import corpus as stdlib_corpus

    return stdlib_corpus()


def load_filler(paths: list[str] | None) -> str:
    text = "".join(Path(p).read_text(errors="ignore") + "\n" for p in paths) if paths else corpus()
    text = SPECIAL.sub("", text)
    if len(text) < 1000:
        raise ValueError("the filler text is too short (under 1,000 characters)")
    return text


def make_needle(rng: random.Random) -> dict:
    color, place = rng.choice(COLORS), rng.choice(PLACES)
    code = f"{rng.randint(1000, 9999)}-{rng.choice(WORDS)}"
    return {"fact": f"The secret passcode for the {color} {place} is {code}.",
            "question": (f"What is the secret passcode for the {color} {place}, stated somewhere in the document "
                         "above? Reply with the passcode only."),
            "answer": code}


def window(filler: str, offset: int, chars: int) -> str:
    """``chars`` characters of the filler from ``offset``, wrapping (with a separator) when it runs out."""

    out, at = [], offset % len(filler)
    while chars > 0:
        piece = filler[at:at + chars]
        out.append(piece)
        chars -= len(piece)
        at = 0
        if chars > 0:
            out.append("\n# ---- (the document continues) ----\n")
    return "".join(out)


def place(body: str, depth: float, fact: str) -> tuple[str, int]:
    """Insert the fact on a line of its own at the line boundary nearest ``depth`` of the body; returns the text and
    the insertion offset."""

    at = int(len(body) * depth)
    nl = body.rfind("\n", 0, at)
    at = nl + 1 if nl >= 0 and at - nl < 4096 else at
    return body[:at] + f"\n{fact}\n\n" + body[at:], at


def build(filler: str, chars: int, depth: float, needle: dict, nonce: str, offset: int) -> str:
    body, _ = place(window(filler, offset, chars), depth, needle["fact"])
    return (f"Document {nonce}. Read the whole document; a question about it follows at the end.\n\n" + body +
            "\n\n----\n" + needle["question"])


def _post(base: str, body: dict, timeout: float):
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat_body(model: str, text: str, max_tokens: int, stream: bool) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if stream:
        body.update(stream=True, stream_options={"include_usage": True})
    return body


def count_tokens(base: str, model: str, text: str) -> int | None:
    """The rendered prompt's tokens from the server's refusal of an impossible reply length (no prefill), or None."""

    try:
        with _post(base, chat_body(model, text, IMPOSSIBLE_REPLY, False), 600) as r:
            r.read()
        return None                                      # accepted: this server gives no receipt (should not happen)
    except urllib.error.HTTPError as exc:
        m = RECEIPT.search(exc.read().decode("utf-8", "replace"))
        return int(m.group(1)) if m else None


def fit(base: str, model: str, make, target: int, counter=count_tokens) -> tuple[str, int | None]:
    """The text ``make(chars)`` whose rendered prompt is within 0.5% of ``target`` tokens (up to six sizings)."""

    chars = int(target * CHARS_PER_TOKEN)
    text, n = make(chars), None
    for _ in range(6):
        n = counter(base, model, text)
        if n is None or abs(n - target) <= 0.005 * target:
            break
        chars = max(1000, int(chars * target / n))
        text = make(chars)
    return text, n


def ask(base: str, model: str, text: str, max_tokens: int) -> dict:
    """Stream the question: time to the first reply text, total time, usage and the reply (or the server's refusal)."""

    sent = time.time()
    first, usage, parts = None, {}, []
    try:
        with _post(base, chat_body(model, text, max_tokens, True), 7200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if d.get("error"):
                    raise RuntimeError(f"server error: {d['error']}")
                for ch in d.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        first = first or time.time()
                        parts.append(piece)
                usage = d.get("usage") or usage
    except urllib.error.HTTPError as exc:
        return {"error": exc.read().decode("utf-8", "replace")[:500], "total_s": round(time.time() - sent, 3)}
    except Exception as exc:  # noqa: BLE001 - recorded: the other cases still run
        return {"error": repr(exc)[:500], "total_s": round(time.time() - sent, 3)}
    end = time.time()
    return {"reply": "".join(parts), "usage": usage, "ttft_s": round(first - sent, 3) if first else None,
            "total_s": round(end - sent, 3)}


def found(reply: str, answer: str) -> bool:
    squash = lambda s: re.sub(r"[^0-9A-Z]", "", s.upper())   # noqa: E731
    return squash(answer) in squash(reply)


def floats(text: str) -> list[float]:
    values = [float(x) for x in text.split(",") if x.strip()]
    if not values or any(not 0.0 <= v <= 1.0 for v in values):
        raise argparse.ArgumentTypeError(f"depths are fractions from 0 to 1, not {text!r}")
    return values


def ints(text: str) -> list[int]:
    values = [int(float(x)) for x in text.split(",") if x.strip()]
    if not values or any(v < 1000 for v in values):
        raise argparse.ArgumentTypeError(f"sizes are token counts of 1000 or more, not {text!r}")
    return values


def cmd_run(a) -> int:
    filler = load_filler(a.filler)
    rng = random.Random(a.seed)
    run_id = f"{a.label}-{rng.randrange(16**8):08x}"
    cases = [(size, depth) for size in a.sizes for depth in a.depths]
    if a.dry_run:
        print(json.dumps({"dry_run": True, "label": a.label, "cases": len(cases), "sizes": a.sizes, "depths": a.depths,
                          "filler_chars": len(filler), "example": make_needle(rng)}))
        return 0
    bad = 0
    for i, (size, depth) in enumerate(cases):
        needle = make_needle(rng)
        nonce = f"{run_id}-{i}"
        offset = (i * 7919 * 1009) % len(filler)               # a different stretch of the filler each case
        text, counted = fit(a.base, a.model, lambda chars: build(filler, chars, depth, needle, nonce, offset), size)
        res = ask(a.base, a.model, text, a.max_tokens)
        usage = res.get("usage") or {}
        row = {"label": a.label, "case": i, "target_tokens": size, "depth": depth, "counted_tokens": counted,
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
               "answer": needle["answer"], "found": bool(res.get("reply")) and found(res["reply"], needle["answer"]),
               "reply": (res.get("reply") or "")[:300], "ttft_s": res.get("ttft_s"), "total_s": res.get("total_s"),
               "error": res.get("error"), "utc": time.strftime("%FT%TZ", time.gmtime())}
        bad += bool(row["error"])
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
    return 1 if bad == len(cases) else 0


def report(paths: list[str]) -> str:
    rows = [json.loads(line) for p in paths if os.path.exists(p) for line in open(p) if line.strip()]
    lines = ["label | target tokens | depth | prompt tokens | found | TTFT s | total s | note"]
    for r in rows:
        note = ("refused: " + r["error"][:90]) if r.get("error") else ("" if r["found"] else f"reply: {r['reply'][:60]!r}")
        lines.append(f"{r['label']} | {r['target_tokens']} | {r['depth']} | {r['prompt_tokens']} | "
                     f"{'yes' if r['found'] else 'NO'} | {r['ttft_s']} | {r['total_s']} | {note}")
    by: dict = {}
    for r in rows:
        k = (r["label"], r["target_tokens"])
        by.setdefault(k, []).append(r)
    for (label, size), rs in sorted(by.items()):
        lines.append(f"{label} {size}: {sum(r['found'] for r in rs)}/{len(rs)} found")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("base"); r.add_argument("model"); r.add_argument("--out", required=True)
    r.add_argument("--sizes", type=ints, default=[128000, 256000, 450000])
    r.add_argument("--depths", type=floats, default=[0.1, 0.5, 0.9])
    r.add_argument("--filler", nargs="*", default=None, help="text files (default: the stdlib corpus)")
    r.add_argument("--seed", type=int, default=None, help="needles and run id (default: random per run)")
    r.add_argument("--label", default="needle"); r.add_argument("--max-tokens", type=int, default=64)
    r.add_argument("--dry-run", action="store_true", help="build the filler and the plan; send nothing")
    p = sub.add_parser("report"); p.add_argument("rows", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        print(report(a.rows))
        return 0
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
