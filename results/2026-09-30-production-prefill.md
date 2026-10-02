# Production prefill speed, 2026-09-30 (router request log, operator's report)

Source: the operator's router log for production `oj-serve` (code 61986bf, int8 KV, packed cache, MTP on with 1-6 drafts),
last ~2 h before 05:00 UTC; no new load generated. TTFT here = router dispatch → first token (includes network and queueing);
"cold" is inferred (the engine reports no cached-token count). `--parallel` was 2 from 03:15 to 04:04 UTC, then a Qwen 27B
trial ran beside production 04:04-04:30 (excluded), then `--parallel 3` again from 04:30.

| time (UTC) | parallel | prompt tokens | prefill s | tok/s | overlap |
|---|---|---|---|---|---|
| 03:18:20 | 2 | 182,200 | 172.1 | 1,059 | 0 (first request after restart) |
| 03:26:33 | 2 | 195,301 | 198.4 | 984 | 0 |
| 04:53:47 | 3 | 103,021 | 56.4 | 1,826 | 0 (new conversation) |

No request near 32k and no cold request near 128k in the window. These agree with the earlier observations (137k cold ≈ 1,700
tok/s; 210k ≈ 925 tok/s) that F2c's per-resource bounds table starts from.

Follow-ups with prefix reuse (streamed calls): 182k-194k prompts adding 0.5-2k new tokens → TTFT 1.5-4.6 s. With one 60k
request overlapping, 103k-106k follow-ups adding 0.3-1k tokens took 11-21 s: contention costs a lot.

## Finding: non-streamed requests get no prefix reuse

Claude Code's permission classifier sends non-streamed calls (~60k-token prompt, 64-token reply). Three identical
60,230-token prompts in a row took 30 s each — a full re-read every time (≈ 2,000 tok/s), no reuse. It hit the
classifier's timeout and blocked the research session's tools; the operator moved classifier traffic to another
machine. Two capabilities would bring it back to the Spark: prefix reuse for non-streamed requests, and reuse across
different conversations that share the ~50k system/tools prefix. Not in F2c Phase 1's scope; recorded for the owner's
prioritisation (candidate: an F2c Phase 2 lever or its own item).

## Classifier request pattern (operator, router log fields + the other machine's per-request cache-hit counts; no prompt content)

- Body: `"stream": false`, ~64-token replies; the router forwards no `draft` field (default drafting on).
- Bursts within one classifier session: one cold call of 71,444 tokens, then four more at exactly 71,444 tokens
  (byte-identical: the other machine's cache re-read only the last 5 tokens), then one at 71,483 (+39) sharing 69,013
  tokens with the others. The next burst: 73,527 × 5, then 73,566, again sharing 69,013. Every burst: five identical
  calls, then one variant that shares a stable ~69k prefix and differs in the last ~2.5-4.7k tokens.
- Across bursts the new prompt usually extends the previous one; the main conversation shares no meaningful prefix with
  the classifier (its first call after a classifier call gets no hit elsewhere either): the classifier has its own prefix.
- Mapping to the engine's reuse rule (strict extension of a kept prompt, drafting on): the four identical calls are the
  "identical prompt = no hit" case (full re-read, ~30 s each); the +39 variant is the "shares a prefix, no checkpoint
  at the divergence point" case. Value: identical-prompt reuse covers 4 of every 5 calls; a checkpoint at the stable
  ~69k prefix (or a dedicated kept slot) covers the fifth (re-read ~2-6k tokens instead of ~71k).
