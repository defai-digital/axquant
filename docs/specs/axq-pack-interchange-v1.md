# AXQ pack interchange v1 (`axq-affine-u32-v1`)

- **Status:** normative
- **Physical format:** MLX affine packed U32
- **Primary runtime:** AX Engine
- **Compatibility runtime:** stock MLX-LM text/backbone loading

This document freezes the existing AXQuant pack contract. It changes no packed bytes and permits
one unchanged checkpoint to serve both runtimes without an intermediate repack.

## 1. One physical weight format

Every quantized Linear or SwitchLinear weight is stored as MLX affine integer codes packed into a
Safetensors `U32` tensor. Each packed module has:

```text
<module>.weight    U32 packed codes
<module>.scales    affine group scales
<module>.biases    affine group biases / zero-point offsets
```

The module's `config.json` quantization entry records `bits`, `group_size`, and `mode: affine`.
Tensor shapes follow the public MLX quantized Linear/SwitchLinear layout. Safetensors sharding may
place modules in different files, and `model.safetensors.index.json` may map names to shards, but a
logical tensor name occurs exactly once and the bytes are not rewritten between runtimes.

AWQ, DWQ, GPTQ, and group-preserving act-order GPTQ are refinement methods. When used, their
execution record must attest that refinement was followed by portable affine packing. They do not
create a second physical format.

Non-affine quantized bodies—including GGUF, MXFP4 left in its source layout, NVFP4, FP8, or a
backend-native AutoRound format—do not conform. A pack never mixes physical quantized formats.

## 2. Protected tensors and sidecars

Protected norms, output heads, governed embeddings/routers, vision/audio tensors, and protected
MTP tensors remain BF16/F16. Ordinary floating Linear biases remain floating tensors and are not
the affine `.biases` metadata tensor.

External MTP sidecars are byte-preserved by default. A quantized MTP sidecar conforms only when its
dedicated manifest declares `mlx-affine-packed-u32`, preserves the byte-identical default, and
binds a passing AX Engine capability check. Vision/audio sidecars remain protected and do not
silently join the quantized text pack.

## 3. Required metadata and provenance

A complete published pack carries, when produced by the staged converter:

- `config.json` with affine per-module or global quantization metadata;
- Safetensors weights and an index when sharded;
- `axquant_plan.json` with the logical precision assignments;
- `axquant_manifest.json` binding the plan, weight bytes, logical parameters, and file hashes;
- `axquant_quantizer_execution.json` binding successful refinement/packing execution; and
- runtime metadata that AX Engine consumes and MLX-LM may ignore.

The manifest plan digest and quantizer-execution plan digest must match the semantic digest of the
packaged plan. Runtime-only metadata and optional protected sidecars do not change the affine
weight hashes.

## 4. Runtime contract

AX Engine is the primary runtime and may consume AXQuant runtime metadata, per-layer KV policy,
MTP metadata, and protected sidecars. MLX-LM is the compatibility runtime for the supported
text/backbone path and may ignore those AX-specific files. Both load the same affine U32 weight
tensors, scales, and biases; neither conformance path is allowed to repack the checkpoint first.

Runtime generation evidence is separate from format conformance. CI may skip live AX Engine tests
when the engine is unavailable, but it must not fabricate a runtime pass.

## 5. Offline conformance

`axquant.interchange.check_affine_u32_pack(directory)` reads JSON and Safetensors metadata without
importing MLX. It returns a list of issues and checks that:

- quantized bodies are U32 with matching `.scales` and `.biases`;
- every physical quantization declaration is affine with valid bits and group size;
- plan and execution records bind one another when present;
- refined methods attest final affine packing; and
- manifest and plan bindings agree when present.

Any reported issue means the directory does not conform to `axq-affine-u32-v1`.
