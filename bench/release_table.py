#!/usr/bin/env python3
"""The README / model-card results table from one compare-upstream run (one run per table: the owner's rule).

  release_table.py compare-summary.json [--configs ours up-lil-full up-radix-full] [--names "Octojet" "TF 0.6.2 + lil" ...]

Prints a Markdown table: one row per measure, seconds unless stated, the configs as columns in the order given.
"""
import argparse, json

ROWS = [
    ("cold-m32k", "Cold 32k-token prompt, first token", "{:.1f}"),
    ("cold-m128k", "Cold 128k-token prompt, first token", "{:.1f}"),
    ("cold-m210k", "Cold 210k-token prompt, first token", "{:.1f}"),
    ("P-cold", "Cold 71k-token prompt, first token", "{:.1f}"),
    ("Pp-variant", "Prompt sharing 69k of those 71k tokens", "{:.1f}"),
    ("P-resend", "The 71k prompt resent", "{:.2f}"),
    ("decode fibonacci-raw t=1", "Decode, code prompt, temperature 1 (tok/s)", "{:.1f}"),
    ("decode gpu-chat-no-think t=1", "Decode, chat prompt, temperature 1 (tok/s)", "{:.1f}"),
    ("decode fibonacci-raw t=0", "Decode, code prompt, greedy (tok/s)", "{:.1f}"),
    ("decode gpu-chat-no-think t=0", "Decode, chat prompt, greedy (tok/s)", "{:.1f}"),
    ("agent cold ttft", "Agent: 80k-token first turn, first token", "{:.1f}"),
    ("agent follow-up ttft (median)", "Agent: +5k-token follow-up, first token (median of 3)", "{:.2f}"),
    ("agent follow-up step total (median)", "Agent: +5k-token follow-up, whole step (median of 3)", "{:.1f}"),
    ("queue: cold short ttft during m128k", "A short prompt sent while a cold 128k prompt fills, first token", "{:.2f}"),
    ("queue: resend ttft during m128k", "A resend sent while a cold 128k prompt fills, first token", "{:.2f}"),
    ("live-stream gap during m128k", "Longest pause of a live reply while a 128k prompt arrives", "{:.2f}"),
    ("gap: m128k ttft while a reply streams", "That 128k prompt's first token while the reply streams", "{:.1f}"),
]


def find(d, key):
    """The summary's key for a measure (decode keys carry the prompt name; tolerate small naming differences)."""
    if key in d:
        return d[key]
    for k, v in d.items():
        if k.replace("_", " ").startswith(key.split(" (")[0]):
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--configs", nargs="+", default=["ours", "up-mlx", "up-lil-full", "up-radix-full"])
    ap.add_argument("--names", nargs="+", default=["Octojet", "TF 0.6.2 + MLX 4-bit", "TF 0.6.2 + lil", "TF 0.6.2 + RadixArk"])
    a = ap.parse_args()
    data = json.load(open(a.summary))
    print("| Measure | " + " | ".join(a.names) + " |")
    print("|---|" + "---:|" * len(a.names))
    for key, label, fmt in ROWS:
        vals = [find(data.get(c, {}), key) for c in a.configs]
        if all(v is None for v in vals):
            continue
        cells = [fmt.format(v) if isinstance(v, (int, float)) else "-" for v in vals]
        print(f"| {label} | " + " | ".join(cells) + " |")
    windows = [data.get(c, {}).get("window") for c in a.configs]
    if any(windows):
        print("| Context windows | " + " | ".join(w or "-" for w in windows) + " |")


if __name__ == "__main__":
    main()
