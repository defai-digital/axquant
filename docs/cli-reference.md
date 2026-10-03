# AXQuant CLI reference

Every `axquant` subcommand in one table. Run `axquant COMMAND --help` for the full
options of any command. The six entry points most users need are `quantize`,
`inspect`, `analyze`, `plan`, `convert`, and `verify-cert`.

| Command | Purpose | Current maturity |
| --- | --- | --- |
| `feasibility` | Audit source and comparison checkpoints before conversion | Implemented |
| `source-checkpoint-manifest` | Derive and bind the immutable source revision, tokenizer, architecture, and file digests for exact-checkpoint certification | Implemented |
| `certification-policy` | Emit the frozen Qwen3-Next non-MTP certification policy and policy digest | Implemented |
| `prepare-coding-suite` | Build the checksum-bound 128-task Qwen3-Next coding suite, toolchain manifest, and calibration-overlap report (`axquant-token-5gram-v2`) | Implemented; formal use requires all pinned toolchains; regenerate manifests for Seatbelt policy v3 |
| `evaluate-coding-suite` | Run resumable generation and deny-default executable scoring for coding-suite v2 | Implemented; Apple Silicon/Seatbelt execution evidence required |
| `verify-coding-suite` | Self-test every coding oracle and scorer by requiring the reference to pass and an empty mutant to fail | Implemented; run before suite freeze |
| `evaluate-general-quality` | Evaluate the disjoint direct-track general holdout and archive every raw model output | Implemented; BF16 and candidate runs must use matched settings |
| `direct-validation-index` | Recompute policy-bound BF16/candidate quality retention for both direct-track profiles | Implemented; emits a fail-closed index for N4 |
| `prepare-general-overlap` | Recompute exact/near-duplicate separation between general holdout and calibration (`axquant-token-5gram-v2`) | Implemented; any match blocks direct validation |
| `inspect` | Inventory tensors, architecture, quantization, and MTP | Implemented |
| `calibrate` | Validate calibration input, record provenance, and build a tokenized cache (`--manifest-only` skips tokenization) | Implemented |
| `validate-calibration-dataset` | Check a calibration JSONL against the toolkit's domain/size/format bar (defaults to the bundled reference dataset) | Implemented |
| `tokenize-calibration` | Build and verify a deterministic tokenized cache | Implemented |
| `capture-activations` | Capture per-module Linear input activations from a verified tokenized cache into a checksum-bound artifact | Implemented |
| `analyze` | Generate architecture priors or measure resumable affine/DWQ/AWQ/GPTQ/BF16 sensitivity from a calibration cache | Implemented |
| `analyze-kv` | Measure per-layer KV-cache sensitivity over a tokenized calibration cache | Implemented; development evidence |
| `plan` | Allocate 4/6/8/BF16 from a sensitivity report | Implemented; release use requires measured evidence |
| `optimize` | Plan weights and optional KV cache under one explicit memory budget and runtime reserve | Implemented; architecture-prior inputs remain estimates |
| `diagnose-joint` | Measure weight x KV interaction I(W,KV) and memory-budget crossover across context lengths | 1.9.0 development evidence only; never a certification claim; 1.8 convert unchanged |
| `plan-joint` | I-gated WeightPlan x KVPlan search that writes a convert-ready development plan | 1.9.0; small I keeps 1.8 independent optimize; material I selects a coupled cell |
| `plan-replay` | Replay a measured plan against its current sensitivity report with exact tensor/signature/metric checks | Implemented; fail-closed migration path |
| `plan-manual` | Apply an explicit YAML precision recipe | Implemented for development |
| `plan-experimental-mix` | Measured mixed 2/3/4-bit on the robust trunk; fused switch modules upgrade as one unit | Development only; not plan-joint; not mlx-optiq |
| `quantize` | Simple development convert: positional `MODEL`, optional `--target-bpw` / `--output` / `--allow-download`; ladder `prior` multi-group default | Implemented; always development evidence (two-door) |
| `simple-convert-help` | Print simple-convert best practices (two-door model) | Implemented |
| `ladders` | List convert ladders (`prior` → `measured-lite` → `measured-full` → `refine-awq-dwq`) with cost/evidence | Implemented |
| `probe-capacity` | Recommend sensitivity probe mode under host memory (bf16-full / measured-lite / streaming / prior-only) | Implemented |
| `scoreboard` | Certification scoreboard from plan + optional size/quality/MTP evidence (MTP speed owned by AX Engine) | Implemented |
| `bind-sensitivity` | Bind weight (+ optional KV) sensitivity digests into one lineage artifact | Implemented |
| `recovery-rank` | Rank quantized tensors for opt-in recovery by sensitivity (not implied by convert) | Implemented |
| `deferred-features` | List fail-closed deferred expansion features (vision-tower quant, per-expert unfused, domain LoRA) | Implemented |
| `recipe-export` | Export a revision-pinned plan as a checksummed recipe bundle | Implemented |
| `support-matrix` | List families with tier, investment posture, priority, and policy notes | Implemented |
| `support-policy` | Print family investment best practices (primary/secondary/thin) | Implemented |
| `head-to-head` | Render the public comparison page from a bound benchmark evidence index | Implemented |
| `convert` | Create the mixed-precision MLX checkpoint and metadata | Implemented for checkpoints at the `convertible` tier or above |
| `runtime-check` | Run AX Engine readiness or actual MLX-LM, MLX-Audio, or MLX-VLM generation | Implemented |
| `prepare-suite` | Materialize deterministic disjoint benchmark inputs | Implemented |
| `evaluate-quality` | Run MLX perplexity and scored generation tasks | Implemented |
| `compare-quality` | Compare matched quality runs with per-task visibility | Implemented |
| `benchmark` | Collect AX Engine runtime evidence | Implemented |
| `benchmark-ab` | Compare one checkpoint with MTP disabled/enabled | Implemented |
| `compose-gemma4-assistant-mtp` | Compose a Tier 2 candidate: byte-identical AXQ Gemma target + `assistant/` drafter + `ax_gemma4_assistant_mtp.json` (does not mutate the Tier 1 pack) | Implemented; product Hub packs ship as bundles under `…-MLX-AXQ-*-MTP` |
| `prepare-grafted-mtp` | Extract and bind a Qwen3.5/3.6 MoE MTP donor head for a Holo3-class trunk | Implemented; graft provenance explicitly records that the donor was not co-trained |
| `compose-grafted-mtp` | Attach a prepared grafted MTP sidecar without mutating the certified trunk tensors | Implemented |
| `mtp-align-prepare-data` | Build trunk-greedy self-distillation labels and optional cached features for MTP adaptation | Implemented; development workflow |
| `mtp-align-teacher-force` | Measure offline depth-1 MTP top-1 agreement against trunk-greedy labels | Implemented; development diagnostic |
| `mtp-align-adapt-fc` | Adapt the grafted MTP FC and normalization tensors while freezing the transformer | Implemented; development workflow |
| `mtp-align-adapt-full` | Continue adaptation with all packed MTP tensors unfrozen | Implemented; development workflow |
| `mtp-align-evaluate` | Score MTP probe or engine A/B evidence against the alignment ladder | Implemented; decision support |
| `benchmark-kernels` | Measure host-scoped decode/prefill kernel latency per (bits, group size) for `plan --latency-table` | Implemented |
| `quantize-mtp-sidecar` | Emit an opt-in quantized MTP sidecar next to the untouched byte-preserved default, gated on a live or recorded AX Engine capability check | Implemented |
| `annotate-omlx-mtp` | Write `axquant_omlx_compat.json` for an existing Qwen MTP pack (`--directory`); does not requantize or merge `mtp.*` into the language index | Implemented |
| `relayout-ngram-table` | Build an MTPLX-compatible variant pack (`--directory` → `--output`) that moves sharded n-gram keys into a standalone `ngram-table.safetensors`; byte-preserving, new directory, `--dry-run` supported | Implemented |
| `kv-serving-quality` | Bind executed per-layer KV precisions to dual-profile quality retention as a report-only artifact | Implemented |
| `mtp-diagnose` | Run the MTP kill-switch diagnostic matrix | Implemented; diagnostic evidence only |
| `benchmark-index` | Bind every required baseline or record why it is unavailable | Implemented |
| `validation-index` | Require disjoint passing agent-coding and general evidence | Implemented |
| `refine` | Generate proxy-ranked bounded precision swaps | Development only |
| `recover` | Record optional post-PTQ recovery provenance | Implemented as identity-copy provenance; no weight mutation |
| `refine-measure` | Build checksum-bound complete-candidate evidence | Implemented |
| `refine-select` | Select only from checksum-bound, validated complete candidates | Implemented |
| `refine-export` | Export standalone executable plans from a refinement result | Implemented |
| `refine-run` | Resume complete conversion, quality, MTP, validation, and selection runs | Implemented |
| `pareto` | Report non-dominated validated candidates on named hardware | Implemented |
| `hardware-registry` | Certify checksum-bound kernel, version, power, and shape coverage | Implemented |
| `campaign-overlap` | Build privacy-preserving exact/5-gram overlap evidence (`axquant-token-5gram-v2`; repeatable `--id-field`, default `id` then `task_id`) | Implemented |
| `campaign-frontier` | Verify every cheapest-failure-first candidate gate and derive the eligible frontier | Implemented |
| `campaign-freeze` | Freeze one exact `qwen36-mtp-v2` source/candidate/evidence graph | Implemented |
| `campaign-preflight` | Verify frozen bindings, durable storage, and exact formal host identity (frozen campaign contract) | Implemented |
| `campaign-start-formal` | Start one budgeted formal cycle only after matching preflight | Implemented |
| `campaign-complete-formal` | Derive pass/fail from the bound completion and consume both formal holdouts | Implemented |
| `campaign-close-no-go` | Close a pre-formal campaign without consuming its blind holdout | Implemented |
| `campaign-record-publication` | Bind downloaded Hub bytes, revision, audit, claim, lifecycle, and runtime re-verification | Implemented |
| `artifact-lifecycle` | Append legal development → candidate → frozen → certified/superseded/revoked transitions | Implemented |
| `claim-render` | Generate measured-BPW public claims and the certified model card from bound evidence | Implemented |
| `release-audit` | Dispatch historical Qwen 3.6 v4, Qwen3-Next N0–N8, or additive `qwen36-mtp-v2` M0–M8 proof | Implemented |
| `compatibility-matrix` | Bind family-wide artifact, runtime, and validation evidence | Implemented |
| `validate` | Apply release thresholds to external benchmark evidence | Implemented |
| `size-evidence` | Bind authoritative candidate/uniform-4 or uniform-6 artifact sizes | Implemented |
| `release-exception` | Record an approved, expiring, evidence-bound size exception | Implemented |
| `report` | Render plan and validation reports | Implemented |
| `publish-prepare` | Assemble a release only after validation | Implemented |
| `bind-artifact-evidence` | Write the artifact evidence-binding sidecar that authorizes release claims | Implemented |
| `publish` | Preview or execute a guarded Hugging Face upload | Implemented |
| `verify-reproduction` | Verify regenerated weight bytes and bound provenance | Implemented |
| `verify-cert` | Offline-check a public certificate and optional artifact bundle with a machine-readable verdict | Implemented; exits nonzero on any inconsistent binding |
| `name` | Generate the recommended AXQuant model name | Implemented |
