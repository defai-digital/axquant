# Qwen3.8-27B vs Qwen3.6-27B AXQ 4-bit MTP — practical head-to-head

| Field | Value |
| --- | --- |
| Status | Measured practical comparison; **not** a certification |
| Host | `df-macstudio-m2` (Apple M2 Ultra, 192 GB unified memory, 24 CPU / 60 GPU cores) |
| Runtime | AX Engine **6.16.1** |
| Date | 2026-08-14 |
| Protocol | Greedy, temperature `0`, **thinking off**, formal Qwen3.8 exact-async MTP profile on both packs |

This report answers a product question, not a quantization-retention question:

> On the same Mac Studio, with the same AXQ 4-bit MTP serving path, what does a user actually gain by moving from Qwen 3.6 27B to Qwen 3.8 27B?

**Short answer:** on short, greedy, non-thinking chatbot / coding / logic / easy-vision work, the two packs are **close**. Qwen3.8 is not a blowout. It is slightly better on the one visual counting item, tied on executable coding, and a little behind on strict last-line scoring. Cold prefill is the same. Decode is within a few percent. Official 3.8 gains live mainly in thinking-on, long-horizon, tool-using suites that this run does not reproduce.

## Bound artifacts

| Pack | Hub revision | Local path on host | Layout |
| --- | --- | --- | --- |
| Qwen3.8-27B AXQ 4-bit MTP | [`32f44846`](https://huggingface.co/AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP/tree/32f448461caf4aedcc3c16a77a63b6a94bf0667c) | `AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP` | 5.0667 main BPW; BF16 vision + MTP sidecars |
| Qwen3.6-27B AXQ 4-bit MTP | [`f44a9ee`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP/tree/f44a9eeebec0c488d0f42201c8763db770a1c0a8) | `AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP` | 5.4183 main BPW; BF16 vision + MTP sidecars |

Source `preprocessor_config.json` and `video_preprocessor_config.json` were copied next to each pack so image geometry is not left to the engine fallback. That is an evaluation convenience. The published Hub trees may still omit those files; see [qwen38-27b-axq-vl-retention.md](qwen38-27b-axq-vl-retention.md).

## Protocol

| Setting | Value | Why |
| --- | --- | --- |
| Decoding | temperature `0`, greedy | Reproducible A/B |
| Thinking | off | Matched instruct-mode cost; 3.8's default thinking-on path is a different product |
| MTP | Qwen3.8 exact async profile on **both** | Same dense `qwen3_5` kernels |
| Server | `ax-engine-server --mlx --support-tier mlx-certified` | Production serving path |
| Isolation | unique system `Request id` + user `[task …]` prefix | First passes leaked prior answers (`Osaka`, then `Amina` JSON) via prefix cache |
| Coding | extracted Python executed against hidden unit tests | “Does the function work?” |
| Vision | Pillow fixtures with exact labels | OCR / charts / counts are known |

Suite: [`data/eval/practical-qwen38-vs-qwen36/`](../data/eval/practical-qwen38-vs-qwen36/).
Runner: [`scripts/run_qwen38_vs_qwen36_practical.py`](../scripts/run_qwen38_vs_qwen36_practical.py).
Raw JSON: [`docs/eval/qwen38-vs-qwen36-27b-axq4-mtp-macstudio-m2/`](eval/qwen38-vs-qwen36-27b-axq4-mtp-macstudio-m2/).

Authoritative quality numbers are the **third** pass (isolated system prompt). Passes 1–2 are archived on the host as `pass1-prefix-cache` and `pass2-user-prefix` and must not be quoted as quality.

## Headline results

### Quality — strict (all checks on a task must pass)

| Category | Tasks | Qwen3.6-27B AXQ 4-bit MTP | Qwen3.8-27B AXQ 4-bit MTP | Δ (3.8 − 3.6) |
| --- | ---: | ---: | ---: | ---: |
| Chatbot QA | 24 | **21 / 24 (87.5%)** | 20 / 24 (83.3%) | −4.2 pp |
| Coding (executable) | 20 | **19 / 20 (95.0%)** | **19 / 20 (95.0%)** | 0 |
| Logic | 24 | **16 / 24 (66.7%)** | 13 / 24 (54.2%) | −12.5 pp |
| Computer vision | 18 | 17 / 18 (94.4%) | **18 / 18 (100%)** | +5.6 pp |
| **All tasks** | **86** | **73 / 86 (84.9%)** | 70 / 86 (81.4%) | **−3.5 pp** |

Mean check-score (partial credit): 3.6 **0.864**, 3.8 **0.828**.

### Quality — format-forgiven (last-answer contains the gold string)

Several logic items are **correct in the prose** and fail only because the model did not emit a bare `FINAL:` line (`38`, `7`, `3/10`, `90`, `no`, `knight`). Counting those, without crediting wrong last-line answers such as logic-09 (`4` instead of `8`):

| Category | Qwen3.6 | Qwen3.8 |
| --- | ---: | ---: |
| Chatbot | 21 / 24 | 20 / 24 |
| Coding | 19 / 20 | 19 / 20 |
| Logic | ~21 / 24 | ~19 / 24 |
| Vision | 17 / 18 | 18 / 18 |
| **All** | **~78 / 86** | **~76 / 86** |

Even with that credit, 3.8 does not pull ahead. The two packs stay within a few tasks.

### Throughput — cold prompts (prefix cache defeated)

Each speed trial prepends a unique salt. `cached_tokens` stays at the chat-template residue (16–32), not the whole prompt. Decode-512 often stopped early on EOS (357 / 392 tokens), so that row is “tokens actually emitted,” not a forced 512.

| Case | Prompt tokens | Qwen3.6 | Qwen3.8 | 3.8 / 3.6 |
| --- | ---: | ---: | ---: | ---: |
| Cold prefill ~512 | 545 | **156.5** tok/s | 156.1 tok/s | 1.00× |
| Cold prefill ~2k | 2004 | 177.3 tok/s | **177.7** tok/s | 1.00× |
| Cold prefill ~4k | 3945 | 176.5 tok/s | **177.2** tok/s | 1.00× |
| Decode 128 | 159 in / 128 out | **23.4** tok/s | 21.5 tok/s | 0.92× |
| Decode 256 | 159 in / 256 out | **27.4** tok/s | 26.8 tok/s | 0.97× |
| Decode 512 (EOS early) | 159 in / ~357–392 out | **28.9** tok/s | 28.6 tok/s | 0.99× |

A previous decode series, taken at the end of the quality session on a short shared prompt, had 3.8 about 7–9% faster (32.8 vs 35.2 tok/s at 512). That session was also prefix-warm. **The cold unique-prompt table above is the one to quote.** Prefill is a tie. Decode is a tie within measurement noise, with 3.6 slightly ahead on the shortest burst.

Do not quote the earlier 15k tok/s prefill figures. Those were almost-full prefix-cache hits (`cached_tokens ≈ prompt_tokens`).

## What changed task by task (strict, isolated pass)

**3.8 only (3 tasks):**

- `chat-01` — both listed capitals; 3.6 failed one format check.
- `code-13` — 3.8 fixed the zero-product bug; 3.6's function did not pass the hidden tests.
- `vl-01` — apple count: 3.8 said **7**, 3.6 said **6**. This is the only clean vision miss.

**3.6 only (6 tasks):**

- `chat-10` — red+blue: 3.6 `purple`, 3.8 `blue`.
- `chat-22` — unique vowels in *education*: 3.6 `5`, 3.8 `3`.
- `code-10` — sliding-window max: 3.6 passed execution, 3.8 did not.
- `logic-13` — day 32 of a Wednesday-start year: 3.6 `Saturday`, 3.8 `Thursday`.
- `logic-19` — truth-teller box: 3.6 `coin`, 3.8 `empty`.
- `logic-22` — 3:00 clock angle: both reasoned to 90°; 3.8's last line was `**Answer:** 90` and failed the strict extractor.

**Both failed (real mistakes or shared format misses):** exact 12-word summary, CSV header-on-its-own-line, several logic items whose last line was `4` or `85` instead of `8` / `83.3`, and a handful of correct answers wrapped in markdown.

Leak check on the isolated pass: **0** stray `Osaka` and **0** stray `Amina` completions.

## How this sits next to official Qwen numbers

Qwen's own 3.8-27B card (thinking / agent harnesses, not this run):

| Slice | Qwen3.8-27B | Qwen3.6-27B |
| --- | ---: | ---: |
| Terminal Bench 2.1 | 73.0 | 63.4 |
| SWE-bench Pro | 61.7 | 53.5 |
| DeepSWE 1.1 | 42.2 | 13.3 |
| IFBench | 79.5 | 69.1 |
| GPQA Diamond | 89.2 | 87.8 |
| BabyVision (no CI) | 65.7 | 28.9 |

Those suites are long-horizon and usually thinking-on. A 27B AXQ pack on a Studio, answering 86 greedy instruct prompts, cannot claim those scores. The local suite asks whether the **same direction** of improvement shows up in ordinary chat, a unit-tested function, a closed-form logic item, and a labeled image. On that narrower job, it mostly does not.

## Is this an AX Engine 3.8-support problem?

Unlikely as the main explanation. Evidence against a broken 3.8 route:

- Both packs load, stream, and return coherent answers on Engine 6.16.1.
- Executable coding is **tied at 95%**. A bad 3.8 kernel would not keep passing hidden tests.
- Easy vision is **18/18 vs 17/18**. 3.8 is the one that counted the apples.
- Cold prefill matches to **<1%**. Decode is within a few percent. 3.8 is not in a slow fallback path.
- The same engine binary already has a scoped 3.8 MTP exactness certificate on `df-macbookpro-m3`.

What the engine **did** affect, and what we corrected:

1. **Prefix reuse** can replay a previous completion when the system prompt is identical. Pass 1 leaked `Osaka`; pass 2 leaked `Amina` JSON into later coding items. Pass 3 unique `Request id` removed it. That is a serving hazard for batched eval, not a 3.8-only defect.
2. **Published AXQ trees omit processor JSON.** We copied the source files. End-to-end VL quality is still not certified ([retention note](qwen38-27b-axq-vl-retention.md)).
3. **Thinking off** is a product choice. 3.8 is trained to think by default. Turning thinking off can flatten it toward 3.6 on short items. That is protocol, not a load failure.

The better reading: **on this usage, the two 27B AXQ 4-bit MTP packs are similar**, and 3.8's advertised jump needs thinking-on / agent / long-horizon work to show up.

## Practical recommendation

| If you mostly… | Switch to 3.8 4-bit MTP? |
| --- | --- |
| Short chat, closed-form QA, one-shot functions, greedy, thinking off | Optional. Quality is tied-to-slightly-behind; speed is tied; pack is a bit smaller (5.07 vs 5.42 BPW). |
| Image questions of the “read this chart / count these objects / OCR this code” kind | Slight yes. 3.8 was perfect on this easy set; 3.6 missed one count. Not a BabyVision claim. |
| Agentic coding, terminal, long thinking traces | Not measured here. Use official numbers and a thinking-on harness; do not extrapolate this suite. |

Product default MTP remains direct fallback. These rates are the opt-in exact-async profile.

## Claim boundaries

- Not a Tier 1 or Tier 2 certificate. Historical 3.8 MTP certs are on
  `df-macbookpro-m3`; historical 3.6 MTP certs are on `df-macbookpro-m5`.
  All new Tier 1 and Tier 2 certifications run on `df-macstudio-m2`.
- Thinking-on, SWE, Terminal-Bench, and OSWorld are out of scope.
- Vision is 18 clean-room fixtures, not MathVision / CharXiv / BabyVision.
- Cold prefill/decode used unique salts. Cached-prefix rates are discarded.
- Isolation (unique system id) is required to reproduce the quality table. A naive sequential `/v1/chat/completions` loop on one server can leak answers.

## Appendix — host and evidence

```text
hostname     df-macstudio-m2
chip         Apple M2 Ultra
memory       192 GiB
os           macOS 26.6.1
engine       ax-engine-6.16.1  (6.16.1)
work         docs/eval/qwen38-vs-qwen36-27b-axq4-mtp-macstudio-m2
```

Copied JSON on this tree:

- [qwen36-quality.json](eval/qwen38-vs-qwen36-27b-axq4-mtp-macstudio-m2/qwen36-quality.json)
- [qwen38-quality.json](eval/qwen38-vs-qwen36-27b-axq4-mtp-macstudio-m2/qwen38-quality.json)
- [qwen36-speed.json](eval/qwen38-vs-qwen36-27b-axq4-mtp-macstudio-m2/qwen36-speed.json)
- [qwen38-speed.json](eval/qwen38-vs-qwen36-27b-axq4-mtp-macstudio-m2/qwen38-speed.json)
- [summary.json](eval/qwen38-vs-qwen36-27b-axq4-mtp-macstudio-m2/summary.json)
