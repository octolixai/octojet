#!/usr/bin/env python3
"""F3 vision probe: chat requests with base64 data-URL images (and one video) against an OpenAI-compatible server started
with --vision; every reply is saved verbatim (reasoning and content) to OUT.md and OUT.jsonl with its TTFT.

  vision_probe.py BASE MODEL --media DIR --out PREFIX [--dry-run]

Cases: a generated red square (expect "red"); the robots observation PNG; the towel acceptance test (crumpled and flat,
originals and -1280 copies, then both together for a two-arm plan); a generated 4-frame video of a square moving left to
right (expect "right"); isolation (the same text with a red and a blue image must differ); and one image request while a
text request decodes (the text stream's largest gap between tokens). Generated media need Pillow, the video PyAV (run it
in the serving container with the vision dependencies on PYTHONPATH). Exit 1 if an expectation failed, 2 on bad input.
"""
import argparse, base64, io, json, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

DESCRIBE = "Describe this towel's current state: shape, how it is folded or bunched, where the edges and corners are."
PLAN = ("Image 1 is the current state, image 2 the target flat state. Give a step-by-step plan for a two-arm robot to "
        "spread it flat, then fold it in half twice.")
NO_THINK = {"enable_thinking": False}


def png(color, *, size=256, square=(64, 64, 192, 192)):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), "white")
    ImageDraw.Draw(img).rectangle(square, fill=color)
    out = io.BytesIO(); img.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def moving_square_mp4(frames=4, size=224, side=48):
    """A square moving left to right over ``frames`` frames at 2 fps, MPEG-4 Part 2 (FFmpeg's own encoder)."""
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    out = io.BytesIO()
    with av.open(out, mode="w", format="mp4") as box:
        stream = box.add_stream("mpeg4", rate=2)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            img = Image.new("RGB", (size, size), "white")
            x = 8 + i * (size - side - 16) // (frames - 1)
            ImageDraw.Draw(img).rectangle((x, size // 2 - side // 2, x + side, size // 2 + side // 2), fill="blue")
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")):
                box.mux(packet)
        for packet in stream.encode():
            box.mux(packet)
    return "data:video/mp4;base64," + base64.b64encode(out.getvalue()).decode()


def file_url(path):
    kind = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}[path.suffix.lower()]
    return f"data:image/{kind};base64," + base64.b64encode(path.read_bytes()).decode()


def image(url):
    return {"type": "image_url", "image_url": {"url": url}}


def message(text, *media):
    return [{"role": "user", "content": [*media, {"type": "text", "text": text}]}]


def cases(media: Path):
    """(name, messages, extra body fields, expectation) for every probe; media files are read here."""
    red, blue = png("red"), png("blue")
    out = [("red-square", message("What color is the square in this image? Answer in one word.", image(red)),
            {"chat_template_kwargs": NO_THINK, "max_tokens": 64, "temperature": 0}, ("contains", "red"))]
    robots = media / "robots-sim-observation.png"
    out.append(("robots", message("Describe this image: what is in the scene and where?", image(file_url(robots))),
                {"max_tokens": 4096}, None))
    pairs = []
    for suffix in ("", "-1280"):
        crumpled, flat = media / f"towel-crumpled{suffix}.jpg", media / f"towel-flat{suffix}.jpg"
        for p in (crumpled, flat):
            out.append((f"towel-describe-{p.stem}", message(DESCRIBE, image(file_url(p))), {"max_tokens": 4096}, None))
        pairs.append((f"towel-plan{suffix or '-original'}", message(PLAN, image(file_url(crumpled)), image(file_url(flat))),
                      {"max_tokens": 6144}, None))
    out += pairs
    video = {"type": "video_url", "video_url": {"url": moving_square_mp4()}}
    out.append(("video-direction", message("In which direction does the square move, left or right? Answer in one word.",
                                           video), {"chat_template_kwargs": NO_THINK, "max_tokens": 64, "temperature": 0},
                ("contains", "right")))
    same = "What is the main color of the shape in this image? Answer in one word."
    out.append(("isolation-red", message(same, image(red)), {"chat_template_kwargs": NO_THINK, "max_tokens": 64,
                                                            "temperature": 0}, ("contains", "red")))
    out.append(("isolation-blue", message(same, image(blue)), {"chat_template_kwargs": NO_THINK, "max_tokens": 64,
                                                              "temperature": 0}, ("contains", "blue")))
    return out


def chat(base, model, messages, extra, timeout=1800.0):
    """One streamed chat request: the reply's reasoning and content verbatim, TTFT, token arrival times, final stats."""
    body = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}, **extra}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    sent = time.perf_counter()
    first, arrivals, reasoning, content, final, error = None, [], [], [], {}, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                chunk = json.loads(line[5:])
                if "error" in chunk:
                    error = chunk["error"]
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    arrivals.append(now)
                reasoning.append(delta.get("reasoning_content") or "")
                content.append(delta.get("content") or "")
                if "usage" in chunk or "octojet" in chunk or "tensorfold" in chunk:
                    final = chunk
    except urllib.error.HTTPError as exc:
        error = {"status": exc.code, "body": exc.read().decode(errors="replace")}
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return {"ttft_s": None if first is None else round(first - sent, 3), "total_s": round(time.perf_counter() - sent, 3),
            "reasoning": "".join(reasoning), "content": "".join(content), "usage": final.get("usage"),
            "stats": final.get("octojet") or final.get("tensorfold"), "error": error, "max_gap_s": round(max(gaps), 3) if gaps else None,
            "arrivals": arrivals, "sent": sent}


def quick_cases():
    """The red square and the isolation pair: generated images only (F4's window)."""

    names = ("red-square", "isolation-red", "isolation-blue")
    red, blue = png("red"), png("blue")
    ask = "What color is the square in this image? Answer in one word."
    same = "What is the main color of the shape in this image? Answer in one word."     # as in cases()
    extra = {"chat_template_kwargs": NO_THINK, "max_tokens": 64, "temperature": 0}
    return [(names[0], message(ask, image(red)), dict(extra), ("contains", "red")),
            (names[1], message(same, image(red)), dict(extra), ("contains", "red")),
            (names[2], message(same, image(blue)), dict(extra), ("contains", "blue"))]


def check(expect, row):
    if expect is None:
        return None
    kind, word = expect
    return word in row["content"].lower()


def concurrent(base, model, media):
    """A text request decodes; one image request starts once its tokens flow. The text stream's largest gap is the
    image admission's stall (the tower and the image prefill run on the scheduler thread)."""
    text = {}

    def story():
        text.update(chat(base, model, message("Write a 500-word story about a lighthouse keeper."),
                         {"chat_template_kwargs": NO_THINK, "max_tokens": 900}))

    worker = threading.Thread(target=story)
    worker.start()
    time.sleep(4.0)                                  # the text stream is decoding by now
    img = chat(base, model, message(DESCRIBE, image(file_url(media / "towel-crumpled.jpg"))), {"max_tokens": 1024})
    worker.join()
    sent = img["sent"]
    window = [b - a for a, b in zip(text.get("arrivals", []), text.get("arrivals", [])[1:]) if b > sent]
    return text, img, (round(max(window), 3) if window else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base"); ap.add_argument("model")
    ap.add_argument("--media", required=True, type=Path)
    ap.add_argument("--out", help="output prefix: PREFIX.md and PREFIX.jsonl (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="build every request (media read and generated), send none")
    ap.add_argument("--quick", action="store_true", help="only the red square and isolation cases (no media files, "
                                                         "no concurrent case)")
    a = ap.parse_args(argv)
    if not a.dry_run and not a.out:
        ap.error("--out is required unless --dry-run")
    need = ["robots-sim-observation.png"] + [f"towel-{k}{s}.jpg" for k in ("crumpled", "flat") for s in ("", "-1280")]
    missing = [n for n in need if not (a.media / n).is_file()] if not a.quick else []
    if missing:
        print(f"error: missing media in {a.media}: {', '.join(missing)}", file=sys.stderr)
        return 2
    plan = quick_cases() if a.quick else cases(a.media)
    if a.dry_run:
        for name, messages, extra, expect in plan:
            size = len(json.dumps(messages))
            print(f"dry-run {name}: {size / 2**20:.2f} MiB body, expect {expect}")
        if not a.quick:
            print(f"dry-run concurrent: text story + image towel-crumpled.jpg")
        return 0
    base = a.base.rstrip("/")
    rows, failed = [], False
    md = [f"# F3 vision probe ({time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}, {base}, {a.model})\n"]
    for name, messages, extra, expect in plan:
        r = chat(base, a.model, messages, extra)
        ok = check(expect, r)
        failed |= ok is False or r["error"] is not None
        rows.append({"case": name, "expect": expect, "pass": ok, **{k: v for k, v in r.items() if k not in ("arrivals", "sent")}})
        print(json.dumps({"case": name, "ttft_s": r["ttft_s"], "total_s": r["total_s"], "pass": ok, "error": r["error"],
                          "content": r["content"][:120]}), flush=True)
    by = {r["case"]: r for r in rows}
    differ = by["isolation-red"]["content"].strip() != by["isolation-blue"]["content"].strip()
    failed |= not differ
    rows.append({"case": "isolation", "pass": differ, "red": by["isolation-red"]["content"],
                 "blue": by["isolation-blue"]["content"]})
    if a.quick:
        return _write(a, md, rows, failed)
    text, img, stall = concurrent(base, a.model, a.media)
    rows.append({"case": "concurrent", "text_max_gap_s": text.get("max_gap_s"), "text_gap_during_image_s": stall,
                 "image_ttft_s": img["ttft_s"], "text_error": text.get("error"), "image_error": img["error"],
                 "text_content": text.get("content", ""), "image_reasoning": img["reasoning"], "image_content": img["content"]})
    failed |= bool(text.get("error") or img["error"])
    print(json.dumps({"case": "concurrent", "text_max_gap_s": text.get("max_gap_s"), "text_gap_during_image_s": stall,
                      "image_ttft_s": img["ttft_s"]}), flush=True)
    return _write(a, md, rows, failed)


def _write(a, md, rows, failed) -> int:
    for r in rows:
        md.append(f"\n## {r['case']}\n")
        md.append("```json\n" + json.dumps({k: v for k, v in r.items() if k not in ("reasoning", "content",
                                            "text_content", "image_reasoning", "image_content", "red", "blue")},
                                           indent=1) + "\n```\n")
        for key in ("reasoning", "content", "red", "blue", "text_content", "image_reasoning", "image_content"):
            if r.get(key):
                md.append(f"\n**{key}**\n\n````text\n{r[key]}\n````\n")
    Path(a.out + ".md").write_text("".join(md))
    with open(a.out + ".jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"probe {'FAILED' if failed else 'ok'}: {a.out}.md, {a.out}.jsonl", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
