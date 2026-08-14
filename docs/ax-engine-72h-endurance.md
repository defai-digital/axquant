# AX Engine 72-hour endurance soak

**Verdict:** pass. AX Engine **6.15.0** completed a continuous **72.00 h** AXQ
endurance soak on 2026-08-14.

This report records a **runtime soak**, not a new checkpoint certificate. It
does not recertify the Qwen 3.6 27B AXQ 6-bit pack, promote sibling models, or
claim MTP acceleration.

The checkpoint itself remains the published v3 artifact certified under
[Qwen 3.6 27B AXQ 6-bit Tier 1](certifications/qwen36-27b-axq6-tier1.md) /
[Tier 2](certifications/qwen36-27b-axq6-tier2.md). This soak asked a different
question: whether the engine stays healthy while serving that pack for three
days.

## Scope

| Item | Value |
| --- | --- |
| Verdict | `pass` (`status=completed`) |
| Soak schema | `ax.axq_endurance_soak.v4` |
| Runner | `run_axq_endurance.py` |
| Started | 2026-08-11 17:11 EDT (`2026-08-11T21:11:25Z`) |
| Finished | 2026-08-14 17:17 EDT (`2026-08-14T21:17:21Z`) |
| Target duration | 72.00 h (elapsed `259200.45` s) |
| Host | `df-macmini-03` — Mac mini (M4 Pro, 14-core, 64 GB), macOS 26.6 |
| Server | `ax-engine-server` **6.15.0** (SHA-256 `2345141cd71120b92c78dbc798c8d6023c54ddd232202794c791f588acd7475e`) |
| Model id | `qwen3.6-27b-axq-6bit` |
| Backend | MLX (`support_tier=mlx_preview`) |
| MLX stack | `mlx` 0.32.0, `mlx-lm` 0.31.3 |
| KV / admission | block size 16 tokens, 1024 blocks, max batch 2048 tokens, concurrency 1 |
| Power | AC, sleep prevented by `caffeinate` |

The server binary was launched from an AX Engine `v6.15.0` tree
(`28dbcd252331f8a0eca9829609f2975a1b4be6a8`). The working tree was dirty with
deleted local benchmark-result files; those deletions do not change the
stamped `6.15.0` server identity above.

Identity files of the served pack:

| File | SHA-256 |
| --- | --- |
| `config.json` | `9ac1f744078b6fbd8572a83ee7723261cbeddb30fde4115496084a1ccd2d4167` |
| `model-manifest.json` | `119b6accf40d2845c9d5237b78d7fd9d0fd728d7cbae3a1bb72d98c1a5da10e0` |
| `tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |

Safetensors payload: 6 files, 20,836,337,632 bytes.

## Result

| Gate | Result |
| --- | --- |
| Duration | 72.00 h continuous, observation continuity `continuous` |
| Endurance requests | **3643 / 3643** successful |
| Client error rate | `0.0000%` |
| Health failures | 0 |
| Consecutive request failures | 0 |
| Preflight | passed (cache-capacity rehearsal included) |
| Baseline | complete at 4 h; no stability or coverage alerts |
| Guardrail alerts | none (memory, performance, assessment, continuity) |
| Lifecycle | 0 drain timeouts, 0 inconclusive drains, 0 KV-report gaps |
| Swap / compressor | 0 bytes throughout |
| Server RSS | ended 20,143 MiB; **−88 MiB** vs baseline (no leak) |

Warm-up added 20 successful requests on top of the 3643 endurance requests.

Endurance mix:

| Shape | Successful |
| ---: | ---: |
| `short_unique` | 2551 |
| `medium_unique` | 546 |
| `shared_prefix` | 364 |
| `long_unique` | 182 |
| **Total** | **3643** |

Final 4-hour window (201 successful requests):

| Shape | OK | p95 TTFT | p05 decode | p05 effective prefill |
| --- | ---: | ---: | ---: | ---: |
| `short_unique` | 141 / 141 | 1.87 s | 13.90 tok/s | 85.04 tok/s |
| `medium_unique` | 30 / 30 | 9.01 s | 13.64 tok/s | 121.28 tok/s |
| `shared_prefix` | 20 / 20 | 9.77 s | 12.78 tok/s | 122.01 tok/s |
| `long_unique` | 10 / 10 | 34.63 s | 14.32 tok/s | 123.65 tok/s |

Window-wide p95 TTFT is 9.77 s because the mix includes medium/shared/long
shapes; short prompts stayed at ~1.87 s. Decode and prefill stayed in the same
band as the 16 h and 40 h checkpoints.

Shared-prefix work reused retained KV (20 retained cache hits / 21,440 reused
tokens in the final window). MLX prefix-cache hit counters stayed at 0; reuse
was through the engine's retained-block path, not a separate prefix-cache
store/hit pair.

## Decode path

The pack includes a native MTP sidecar (`mtp.safetensors`) and the engine
marked Qwen linear MTP as available on every endurance request. **MTP decode
was not used.**

| Telemetry (all 3643 endurance requests) | Value |
| --- | ---: |
| `ax_mtp_available` | 3643 |
| `ax_mlx_qwen_linear_mtp_direct_fallback` | 3643 |
| MTP draft / accepted tokens | 0 / 0 |
| N-gram accepted tokens (sum) | 133,655 |

This matches the product default for this pack: direct fallback, with n-gram
speculation providing the observed decode-side acceleration. Do not read this
soak as a Tier 2 MTP result.

## Memory and host health

4,314 resource samples. Peak server RSS was 20,381 MiB. Lifetime RSS slope was
about −0.24 MiB/h after the 4-hour baseline. MLX active bytes were flat at
20,572 MiB in the final checkpoint. Host swap and compressor stayed at 0.
No thermal or performance warnings were recorded during operator checks.

Work-normalized RSS was **−5.6 KiB per completed request**. Bounded KV state
grew only as retained cache filled (free blocks 225 → 5; cached blocks +220
over 72 h) and did not trend as a leak.

## What this does not claim

- Not a new AXQuant checkpoint certificate or Hub-edition change.
- Not an MTP acceleration or exactness claim.
- Not a multi-model, multi-host, or multi-concurrency matrix. Concurrency was 1
  on one Mac mini.
- Not a quality recertification. The runner's quality gate was `ready` after
  baseline; this soak is about uptime, errors, and memory, not suite scores.
- Not a claim that every later AX Engine build inherits this result. Re-run the
  soak for a new server SHA.

## Evidence

Host-local run id `axq-6bit-72h-v6.15.0-20260811T211124Z` on `df-macmini-03`
contains `summary.json`, `events.jsonl`, 4-hour periodic reports, the final
checkpoint `2026-08-14T211721+0000-final.md`, and diagnostics. Those files stay
on the soak host; this document is the public extract.
