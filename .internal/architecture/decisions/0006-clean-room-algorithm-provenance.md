# ADR 0006: Clean-Room Algorithm Provenance

**Status:** Accepted  
**Date:** 2026-08-01

## Context

AXQuant is developed under the clean-room policy recorded in
`.internal/policies/clean-room.md`, which prohibits importing, vendoring, translating, or
copying mlx-optiq implementation code, tests, documentation, calibration data, or generated
metadata. AXQuant builds only against the public MLX and MLX-LM interfaces and its own
independently defined schemas.

The policy establishes a negative boundary (what must not be reused) but does not, on its own,
demonstrate the positive case: that each core algorithm in the toolkit was derived independently
from published academic and engineering literature. Recording those independent derivation
sources strengthens the legal and engineering boundary by showing that every algorithm traces to
publicly available prior art rather than to any competitor's implementation. It also gives
reviewers a concrete, citable trail when auditing the toolkit.

This ADR documents the independent derivation sources for the core algorithms currently in the
pipeline. It is engineering provenance documentation, not a legal opinion.

## Decision

AXQuant records, for each core algorithm, the independent published source from which it was
derived. The following algorithms and sources are established as the clean-room provenance of
record.

### 1. Greedy marginal-efficiency budget allocation (`planner.py`)

The planner starts every tensor at its minimum permitted bit-width, then iteratively upgrades the
single tensor that yields the best quality-improvement-per-storage-cost ratio, repeating until the
bits-per-weight (BPW) budget is exhausted.

Independent source: the classical greedy algorithm for the fractional knapsack problem
(Cormen, Leiserson, Rivest, and Stein, *Introduction to Algorithms*, Ch. 16). The
marginal-efficiency variant — ranking upgrades by incremental gain per incremental cost — is
standard in rate-distortion optimization (Shannon, "A Mathematical Theory of Communication",
1948; Cover and Thomas, *Elements of Information Theory*, Ch. 10).

### 2. KL divergence as sensitivity metric (`probe.py`)

Per-tensor sensitivity is measured as the Kullback–Leibler divergence KL(P || Q) between the
reference and candidate logit distributions.

Independent source: Kullback and Leibler, "On Information and Sufficiency", 1951. Its application
as a quantization-sensitivity measure is standard in the post-training quantization literature
(for example, Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at
Scale", 2022; Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers", 2022).

### 3. Isolated-module probing (`probe.py`)

Each tensor is quantized independently while all other tensors remain at BF16, and the isolated
distortion introduced by that single tensor is measured.

Independent source: layer-wise reconstruction-error analysis in Li et al., "QED: A Framework for
Analyzing the Quantization Effects of Deep Neural Networks", 2016; and the per-layer sensitivity
approach in Dong et al., "HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision
Settings", 2019.

### 4. Fake-quant KV-cache probing (`kv_probe.py`)

KV-cache sensitivity is measured through a quantize-then-dequantize round trip (fake
quantization) rather than by executing packed kernels.

Independent source: fake quantization as a precision-loss proxy is standard in
quantization-aware training, Jacob et al., "Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference", 2018.

### 5. Cosine distance and token disagreement (`probe.py`)

Secondary metrics use `1 - cosine_similarity` between distributions and the argmax mismatch rate
between reference and candidate token selections.

Independent source: cosine similarity as a standard distributional distance, Singhal, "Modern
Information Retrieval: A Brief Overview", 2001; argmax disagreement as the standard zero-one loss
for classification.

### 6. Atomic staging conversion (`converter.py`)

Conversion writes into a temporary staging directory, verifies the result, and then renames the
staging directory to the final output path so that a partial checkpoint never appears at the
destination.

Independent source: the standard atomic-file-write pattern relying on POSIX `rename(2)`
atomicity semantics, used universally in package managers and database write-ahead-log systems.

## Consequences

- Every core algorithm in the current pipeline has a documented, citable independent derivation
  source, reinforcing the clean-room boundary defined in `.internal/policies/clean-room.md`.
- Future algorithms added to the toolkit should also record their derivation source, either by
  extending this ADR or in a supplementary ADR, before they are merged.
- This documentation establishes good-faith engineering provenance; it does not constitute legal
  advice and is not a substitute for legal review.
- The recorded sources are public academic and engineering prior art; they do not imply reuse of
  any competitor's code, data, or metadata, which remains prohibited by the clean-room policy.

## References

- Clean-room policy: `.internal/policies/clean-room.md`
- Cormen, Leiserson, Rivest, and Stein, *Introduction to Algorithms*, Ch. 16 (greedy algorithms).
- Shannon, "A Mathematical Theory of Communication", 1948.
- Cover and Thomas, *Elements of Information Theory*, Ch. 10 (rate-distortion theory).
- Kullback and Leibler, "On Information and Sufficiency", 1951.
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", 2022.
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
  Transformers", 2022.
- Li et al., "QED: A Framework for Analyzing the Quantization Effects of Deep Neural Networks",
  2016.
- Dong et al., "HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision
  Settings", 2019.
- Jacob et al., "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference", 2018.
- Singhal, "Modern Information Retrieval: A Brief Overview", 2001.
- POSIX `rename(2)` atomic-rename semantics.
