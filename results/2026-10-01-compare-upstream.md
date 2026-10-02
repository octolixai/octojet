# Octojet vs upstream TensorFold 0.6.0 on the published NVFP4 checkpoints (2026-10-01)

Owner's question: is our build faster than the fastest published setup available today, and is the mixed checkpoint
worth releasing? Run by the operator on the Spark (GB10), 19:06-19:46 UTC, `bench/spark/compare-upstream.sh` at
cbb602a, rc 0, GPU exclusive, page cache dropped before each start.

| Config | Engine | Checkpoint |
|---|---|---|
| ours | Octojet e53e17d code (packed cache, 4 prefix checkpoints) | mixed: Vontra MLX 4-bit + RadixArk NVFP4 routed experts |
| up-lil | TensorFold v0.6.0 (latest release) | local-inference-lab/Qwen3.8-Flash-Next-NVFP4 rev 7c4f1bc |
| up-radix | TensorFold v0.6.0 | RadixArk/Qwen3.8-Flash-Next-NVFP4 rev 7b71922 |

All servers: `--kv-dtype int8 --parallel 3`, text only. Times are client side (request sent to first text), seconds;
decode is tok/s (`bench_openai`, 64 tokens, median of 10, drafting on). **Accuracy was not run (ACC=0).**

| Metric | ours | up-lil | up-radix | ours vs best upstream |
|---|---:|---:|---:|---|
| Cold 32k prompt | 27.7 | 21.4 | 22.5 | 1.30x slower (see note 2) |
| Cold 128k prompt | 68.9 | 89.5 | 96.0 | 1.30x faster |
| Cold 210k prompt | 117.9 | 152.3 | 166.4 | 1.29x faster |
| 71k prompt, cold | 32.4 | 46.1 | 52.6 | 1.42x faster |
| Variant sharing 69k of it | 3.2 | 47.4 | 50.0 | 14.6x faster |
| Identical resend | 3.2 | 0.49 | 0.28 | 11x slower (see note 1) |
| Decode, code, t=1 | 64.0 | 50.5 | 40.4 | 1.27x |
| Decode, chat, t=1 | 62.0 | 42.0 | 31.4 | 1.48x |
| Decode, code, t=0 | 55.4 | 51.2 | 43.4 | 1.08x |
| Decode, chat, t=0 | 89.6 | 40.2 | 35.1 | 2.23x |
| Agent: cold 80k turn, first token | 53.5 | 68.3 | 77.0 | 1.28x faster |
| Agent: follow-up first token (median) | 4.29 | 5.40 | 6.29 | 1.26x faster |
| Agent: follow-up step total (median) | 11.8 | 14.9 | 22.0 | 1.26x faster |
| Live stream's longest pause while a 128k prompt arrives | 2.34 | 1.64 | 2.26 | 1.43x longer |

Startup (from the server logs):

| | ours | up-lil | up-radix |
|---|---|---|---|
| Startup estimate | 87.22 GiB | 81.09 GiB | 97.39 GiB |
| Windows | 3 x 262,144 reserved (5,241 MiB a stream incl. 4 checkpoints) | up to 3 streams growing to 262,144; 30.9 GiB free for caches, 4.47 GiB per full window | 28.0 GiB free for caches |
| Loaded in | 150.2 s | 204.9 s | 86.9 s |

## Reading

- **Agent work is faster on ours:** about 26-30% on long prompts and agent steps, and 1.1-2.2x on decode. A prompt
  that shares a long prefix with an earlier one comes back in 3 s instead of about 47 s (our stage-B checkpoints;
  upstream has exact resume only).
- **No memory advantage:** local-inference-lab's checkpoint needs about 6 GiB less than ours and leaves more cache
  room, so a "more context on one Spark" claim for our checkpoint does not hold. The decode gain plausibly comes from
  our 4-bit non-expert weights (local-inference-lab keeps attention in MXFP8, RadixArk in bf16/FP8), so the
  checkpoint matters for speed, not memory. This is an inference, not measured separately.
- **Quality is unmeasured.** local-inference-lab's checkpoint is quantization-aware distilled; our non-expert weights
  are plain MLX 4-bit. Paired GSM8K and HumanEval are needed before any release claim.

## Notes

1. **Identical resend, 3.2 s (bug, ours).** The resend followed the variant. A variant that resumes from a checkpoint
   reuses the source entry's slot and drops the source entry (`MultiDecoder.admit`: the matched slot is filled in
   place), so the resend finds only a checkpoint near the end and re-reads about 1.8k tokens. Upstream keeps both.
   Fix: resume a non-exact match into a spare slot (copying the rows up to the resume point, as the twin copy does)
   when one is free, so the source's exact state survives. Classifier pattern (4 exact + 1 variant) is affected when
   the variant comes before the remaining repeats.
2. **Cold 32k, 27.7 s (unexplained).** Production measured about 13.8 s for 32k earlier today (F3), and our 128k/210k
   here are in line with production. The 32k run was the first long prompt after a 5-word warm-up, so a one-time
   compile for long-prompt shapes is the likely cause; unconfirmed. Re-run 32k twice to settle it.
3. Live-stream pause: all three stay within 1.6-2.3 s while a 128k prompt is read in; upstream 0.6 has the same
   prompts-inside-rounds design.
4. The summary's window row printed "-" for upstream (a parser gap, fixed after the run in `bench/compare_summary.py`).

## Follow-up run (2026-10-01 23:45 - 2026-10-02 00:17 UTC): the resend fix, cold 32k recheck, accuracy

`compare-upstream.sh MODE=short CONFIGS="ours up-lil"`, ours = f5-variant-beside 78c1215 (the resend fix), rc 0. The
fix's GPU tests passed first (66 + 52 + 8 + 53, FAIL=0, including the new beside-the-source test).

| Metric | ours | up-lil |
|---|---:|---:|
| Cold 32k prompt | 18.2 | 19.8 |
| Cold 32k, a second different prompt | 16.9 | 19.8 |
| 71k prompt, cold | 37.8 | 45.6 |
| Variant sharing 69k | 4.9 | 45.6 |
| Identical resend after the variant | 0.12 | 0.18 |
| GSM8K (250) | 0.980 (245) | 0.984 (246) |
| HumanEval pass@1 (164) | 0.945 (155) | 0.963 (158) |

- The resend is now an exact hit (note 1 fixed). The variant now pays for the row copy into a second slot: 4.9 s here
  against 3.2 s in place.
- Cold 32k is ahead of upstream; the earlier 27.7 s was a one-time cost on the first long prompt (note 2 settled).
- Accuracy passes the F1 parity rule (GSM8K within 3 net, HumanEval within 4 programs): -1 and -3. A per-problem
  HumanEval diff is still worth reading before any public claim; accuracy rows are in `~/octojet-runs/compare-short/`
  on the Spark.
