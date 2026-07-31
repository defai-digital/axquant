# ADR 0002: Qwen 3.6 Product Boundary and Runtime Tiers

**Status:** Accepted; fixed checkpoint-count clauses superseded by ADR 0005  
**Date:** 2026-07-28  
**Amended:** 2026-07-30

## Context

AXQuant needs one vertical slice that can prove mixed-precision PTQ, agent/coding quality, MTP
preservation, and real Apple Silicon acceleration. Supporting unrelated model families in the
first release would multiply adapter, calibration, runtime, and benchmark uncertainty.

The Qwen3.6-27B public config has several implications:

- the external product name is Qwen 3.6 while `model_type` is `qwen3_5`;
- the language model has 64 layers with interleaved linear and full attention;
- `mtp_num_hidden_layers` declares an MTP component;
- a vision tower is present and `language_model_only` is false.

MLX-LM standard inference remains valuable for portability, but its current Qwen 3.6 MTP loading
path is not a safe dependency for the v0.1 completion gate. AX Engine already has a native
Qwen 3.6 graph, fused MTP sidecar contract, native model manifest, and readiness checks.

## Decision

AXQuant v0.x and v1 are Qwen 3.6 products.

- Qwen3.6-27B is the first and only conversion target through v0.4.
- v0.5 and v1 certify every official dense Qwen 3.6 parameter size present at release time, as
  defined by [ADR 0005](0005-release-time-official-dense-scope.md).
- The official catalog verified on 2026-07-30 contains one dense size, 27B.
- Gemma, arbitrary Qwen generations, MoE optimization, and VLM optimization are later adapters.
- Generic checkpoint inspection remains available, but it is inventory-only and cannot produce a
  release plan or conversion.

The product statement is:

> AXQuant is an AX Engine-optimized, MLX-compatible PTQ toolkit for Qwen 3.6.

## Runtime tiers

| Level | Runtime | Required behavior |
| --- | --- | --- |
| A | AX Engine | Native model manifest, full mixed precision, MTP, runtime checks, benchmark authority |
| B | MLX-LM | Standard language-model inference from portable MLX weights |

Level B does not promise feature or performance parity. When MLX-LM does not understand the MTP
sidecar or AX metadata, it may use ordinary decode. Failure to provide MLX-LM MTP is not a v0.1
blocker.

## Artifact ownership

AXQuant exports one directory with separate authorities:

```text
config.json and Safetensors
  portable MLX model and tokenizer contract

model-manifest.json
  AX Engine native model contract

mtp.safetensors, mtplx_runtime.json, ax_mtp_sidecar_manifest.json
  AX Engine Qwen MTP bundle and provenance

axquant_plan.json, axquant_manifest.json, axquant_runtime.json
  AXQuant allocation, provenance, runtime tier, and recommendations
```

AXQuant invokes `ax-engine-bench generate-manifest --validate` to create
`model-manifest.json`. It does not copy the Rust schema or inject AXQuant-only fields into that
manifest. `ax-engine doctor --json --mlx-model-artifacts-dir` is the readiness interface.

The v0.1 converter requires AX Engine manifest validation by default. Development runs may
explicitly select `if-available` or `skip`; such outputs are not publishable.

## Qwen 3.6 adapter policy

The adapter recognizes the external Qwen 3.6 identity and validates the current `qwen3_5` config
shape. It understands linear-attention projections, full-attention projections, MLPs, MTP
components, and vision components.

For Qwen3.6-27B v0.x:

- language-path 2-D tensors are eligible for planning;
- norms and vision components stay BF16;
- LM head and integrated or external MTP components use protected policy;
- external MTP bundles are copied byte-for-byte with checksum verification;
- vision execution and quality are not release claims.

An unknown Qwen 3.6 size, a Qwen MoE checkpoint, or a non-Qwen model remains inventory-only until
an explicit adapter version and benchmark matrix promote it.

## Release sequencing

| Version | Gate |
| --- | --- |
| v0.1 | Manual Qwen3.6-27B PTQ, MTP preservation, AX manifest, MLX-LM fallback |
| v0.2 | Trustworthy AX Engine MTP-on/off benchmark harness |
| v0.3 | Measured per-tensor and MTP-aware planner |
| v0.4 | AWQ/DWQ, global validation, agent-coding Pareto search |
| v0.5 | Catalog-complete dense Qwen 3.6 compatibility proof and MLX-LM hardening |
| v1.0 | Public Qwen 3.6 toolkit and validated reference checkpoints |

The MTP throughput gate first becomes authoritative in v0.2. v0.1 proves preservation and
loadability, not acceleration.

## Consequences

The narrower scope makes every product claim falsifiable against one coherent model family and
runtime. AX Engine can evolve MTP and kernel behavior without creating a closed weight format,
because standard MLX files remain the portable base. The cost is that generic model conversion,
Gemma comparison, and full multimodal quantization are intentionally deferred.

## References

- Qwen3.6-27B model and config:
  <https://huggingface.co/Qwen/Qwen3.6-27B>
- Official Qwen 3.6 collection:
  <https://huggingface.co/collections/Qwen/qwen36>
- MLX-LM Qwen3.6 MTP loader issue:
  <https://github.com/ml-explore/mlx-lm/issues/1462>
- MLX-LM learned quantization interfaces:
  <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md>
