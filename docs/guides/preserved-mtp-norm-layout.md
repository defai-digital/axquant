# Preserved MTP norm layout

Preserving Safetensors payloads does not establish whether their RMSNorm weights
are raw Hugging Face deltas or already converted MLX multipliers. Converting an
already normalized head a second time can reduce speculative acceptance and
make MTP slower than direct decoding.

When extracting MTP tensors from an already quantized source, or copying an
external MTP bundle that includes a quantized `config.json`, AXQuant now requires
an explicit `mtp_norm_layout` in the source `mtplx_runtime.json`:

- `raw_hf_delta`: the runtime must add one to each MTP norm.
- `mlx_multiplier`: the runtime must use the stored multipliers unchanged.

The original unquantized HF conversion path retains its raw-delta default.
Explicit raw heads on quantized backbones remain supported. Unknown explicit
layout values fail validation. `ModelIdentity.format` is not proof of the norm
convention: its schema also labels original source inventories as `mlx`.

If conversion fails with `requantized MTP source requires an explicit
mtp_norm_layout`, establish the convention from the source conversion contract
or compare the norm tensors with the pinned original checkpoint. Add the verified
layout to a separate prepared source bundle; do not guess from a model name or a
single tensor's magnitude. Keep the original source and immutable Hub snapshots
unchanged. Existing explicit declarations are preserved, not silently corrected.

The September 19, 2026 Tiel exports demonstrate the failure mode: all seven MTP
norms already equal BF16(original Ornith norm + 1), but their runtime metadata
says `raw_hf_delta`. AX Engine's compatibility correction binds to the complete
hashes of those two audited sidecars. It does not repair published metadata or
qualify the models. Corrected future packs need a new immutable revision and
fresh runtime validation.
