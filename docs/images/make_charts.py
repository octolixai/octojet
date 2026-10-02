#!/usr/bin/env python3
"""Render the benchmark charts used by the README and the Hugging Face model card.

  python3 docs/images/make_charts.py        # writes docs/images/*.png

Every number below is copied from docs/benchmarks.md (the single source); change them there first.
Versions: Octojet e53e17d / 78c1215; TensorFold v0.6.0 (c4646171) with local-inference-lab NVFP4 @ 7c4f1bc1 and
RadixArk NVFP4 @ 7b719225; one DGX Spark (GB10), int8 KV, --parallel 3.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
WHO = ["Octojet", "TensorFold 0.6 + local-inference-lab", "TensorFold 0.6 + RadixArk"]
COLOR = ["#1d5fa8", "#c2683f", "#e0b39b"]
INK, MUTED, RULE = "#16202c", "#5b6876", "#d9e0e7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.edgecolor": RULE})

# run 1, 1 Oct 2026 (docs/benchmarks.md "Head-to-head: speed")
SPEED = [
    ("Cold 128k-token prompt, first token", "s", True, [68.9, 89.5, 96.0]),
    ("Cold 210k-token prompt, first token", "s", True, [117.9, 152.3, 166.4]),
    ("Prompt sharing 69k of 71k tokens with an earlier one", "s", True, [3.2, 47.4, 50.0]),
    ("Agent follow-up step (+5k tokens), median", "s", True, [11.8, 14.9, 22.0]),
    ("Decode, chat, sampled", "tok/s", False, [62.0, 42.0, 31.4]),
    ("Decode, code, sampled", "tok/s", False, [64.0, 50.5, 40.4]),
    ("Decode, chat, greedy", "tok/s", False, [89.6, 40.2, 35.1]),
    ("Decode, code, greedy", "tok/s", False, [55.4, 51.2, 43.4]),
]


def speed_chart():
    fig, axes = plt.subplots(4, 2, figsize=(11, 9.2))
    for ax, (title, unit, lower, vals) in zip(axes.flat, SPEED):
        best = min(vals[1:]) if lower else max(vals[1:])
        ratio = best / vals[0] if lower else vals[0] / best
        y = range(len(vals))[::-1]
        ax.barh(list(y), vals, color=COLOR, height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.015, yi, f"{v:g} {unit}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, max(vals) * 1.28)
        ax.set_yticks([])
        ax.set_title(f"{title}\n{'lower' if lower else 'higher'} is better · Octojet {ratio:.2f}x the best upstream",
                     fontsize=10, loc="left", color=INK)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR]
    fig.legend(handles, WHO, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "One DGX Spark (GB10), int8 KV cache, 3 streams, text only, same prompts; client-side times. "
             "TensorFold v0.6.0 (c4646171); local-inference-lab @ 7c4f1bc1; RadixArk @ 7b719225. Run 1, 1 Oct 2026.",
             ha="center", fontsize=8, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT / "speed-vs-upstream.png", dpi=150, facecolor="white")
    plt.close(fig)


def accuracy_chart():
    # run 2 (docs/benchmarks.md "follow-up and accuracy"): Octojet vs TensorFold 0.6 + local-inference-lab
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for ax, (title, n, vals) in zip(axes, [("GSM8K, 250 problems", 250, [245, 246]),
                                          ("HumanEval pass@1, 164 programs", 164, [155, 158])]):
        ax.barh([1, 0], vals, color=COLOR[:2], height=0.6)
        for yi, v in zip([1, 0], vals):
            ax.text(v + n * 0.01, yi, f"{v} / {n}", va="center", fontsize=9)
        ax.set_xlim(0, n * 1.2)
        ax.set_yticks([1, 0], ["Octojet", "TF 0.6 + lil"], fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.text(0.5, 0.01, "Greedy, thinking off, identical requests. Differences are within noise (±2-3 GSM8K, ±3-4 "
             "HumanEval). Run 2, 1-2 Oct 2026.", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "accuracy.png", dpi=150, facecolor="white")
    plt.close(fig)


def history_chart():
    # docs/benchmarks.md "Octojet production history" (server-side times)
    rows = [("Cold 210k prompt", 227, 111, "s"), ("Cold 128k prompt", 68.5, 62.5, "s"),
            ("Live reply's pause while a 128k prompt arrives", 68, 2.5, "s"),
            ("Variant of a 71k prompt (shares 69k)", 35.7, 2.2, "s"), ("Warm start", 355, 138, "s")]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = list(range(len(rows)))[::-1]
    before = [r[1] for r in rows]
    after = [r[2] for r in rows]
    ax.barh([v + 0.18 for v in y], before, height=0.34, color="#b9c4cf", label="before")
    ax.barh([v - 0.18 for v in y], after, height=0.34, color=COLOR[0], label="after")
    for yi, b, a in zip(y, before, after):
        ax.text(b + 4, yi + 0.18, f"{b:g} s", va="center", fontsize=8, color=MUTED)
        ax.text(a + 4, yi - 0.18, f"{a:g} s", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, max(before) * 1.15)
    ax.set_xlabel("seconds (server side)", fontsize=8, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Octojet in production on one Spark, 29 Sep - 2 Oct 2026", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "production-history.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    speed_chart()
    accuracy_chart()
    history_chart()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
