# AXQ model fleet v2 migration

Audit date: 2026-08-05
Hub owner: `AutomatosX`
Scope: every public model repository whose name contains `AXQ` and does not already carry the
`v2` edition marker

## Decision

Rebuild and republish all 28 existing AXQ development packs as v2 repositories. Do not overwrite
or delete the original repositories: they remain historical development artifacts, and their Hub
revisions stay available to existing users.

This is a real checkpoint rebuild, not a model-card-only refresh. The audited fleet is
598.301 GB across 28 repositories and 14 source checkpoints.

## Why a rebuild is required

The live Hub audit found:

- all 28 packs were produced by AXQuant 1.0.0 or 1.0.1, before the v1.1/v1.2 calibration,
  packed-expert, quantization-numerics, source-provenance, and runtime hardening;
- all 28 plans use `architecture_prior`, all 28 have no calibration binding, and none has a
  release audit; the cards correctly classify them as development evidence;
- 24 of 28 manifests do not pin the source revision;
- all 28 manifests say `target_class: 4bit`, which is wrong for the 11 six-bit and three
  eight-bit product variants;
- the MiniCPM5 and Qwen 3.5 pairs contain identical weight bytes because protection floors
  dominate both requested budgets;
- both Qwen3-Coder-Next repositories contain identical 156.7 GB weight payloads at 15.7300 BPW:
  the pre-fix adapter left fused experts at BF16;
- both Nemotron manifests identify a source repository that does not exist. The v2 source is
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` at an immutable revision.
- the unpinned Ministral 3 source names now resolve to FP8 checkpoints. V2 uses Mistral's
  dedicated `-BF16` repositories so AXQuant never re-quantizes an already quantized source.
- the pinned Ministral, Devstral, and Mistral Small source repositories contain both an
  authoritative indexed checkpoint and a redundant `consolidated.safetensors`. The v2 source
  preparation records the immutable Hub identity and materializes only the shards named by
  `model.safetensors.index.json`; AXQuant's ambiguity check remains fail-closed.

Architecture-prior allocation remains development evidence in v2. Rebuilding with a newer
toolkit does not turn it into measured sensitivity or certification. Certified releases still
require the family-specific measured pipeline and release audit.

## Audit issue codes

| Code | Meaning |
| --- | --- |
| `V` | Built with AXQuant 1.0.x; predates v1.2 hardening |
| `U` | Source revision is unpinned |
| `P` | Architecture-prior plan, no calibration, no release audit |
| `T` | Manifest target class conflicts with the 6-bit/8-bit Hub product class |
| `D` | Sibling product variants contain identical weight bytes |
| `E` | Qwen3-Coder-Next fused experts remained BF16; regeneration is mandatory |
| `S` | Manifest source repository is invalid and must be corrected |
| `R` | Unpinned source name now resolves to FP8; use the official BF16 source instead |

## Complete audited disposition

The abbreviated Hub revisions below identify the exact historical `main` state inspected on the
audit date. BPW is `measured_main_bpw` from that revision's `axquant_manifest.json`.

| Current repository | Hub rev | AXQuant | Source pinned | BPW | Issues | v2 repository |
| --- | --- | ---: | --- | ---: | --- | --- |
| `AX-Devstral-Small-2505-MLX-AXQ-4bit` | `6d04a0c65dbb` | 1.0.0 | no | 4.949963 | V,U,P | `AX-Devstral-Small-2505-MLX-AXQ-4bit-v2` |
| `AX-Devstral-Small-2505-MLX-AXQ-6bit` | `7086f12e3b3b` | 1.0.0 | no | 5.999989 | V,U,P,T | `AX-Devstral-Small-2505-MLX-AXQ-6bit-v2` |
| `AX-gemma-4-12b-MLX-AXQ-4bit` | `5192374d4daa` | 1.0.0 | no | 4.890033 | V,U,P | `AX-gemma-4-12b-MLX-AXQ-4bit-v2` |
| `AX-gemma-4-12b-MLX-AXQ-6bit` | `0003ab1be26a` | 1.0.0 | no | 6.000088 | V,U,P,T | `AX-gemma-4-12b-MLX-AXQ-6bit-v2` |
| `AX-MiniCPM5-1B-MLX-AXQ-4bit` | `9fc3fb996a35` | 1.0.0 | no | 7.380428 | V,U,P,D | `AX-MiniCPM5-1B-MLX-AXQ-4bit-v2` |
| `AX-MiniCPM5-1B-MLX-AXQ-6bit` | `f28d93155bb5` | 1.0.0 | no | 7.380428 | V,U,P,T,D | `AX-MiniCPM5-1B-MLX-AXQ-6bit-v2` |
| `AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit` | `a41e0128dd2e` | 1.0.1 | no | 5.279903 | V,U,P,R | `AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit-v2` |
| `AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit` | `8289fa74c71f` | 1.0.1 | no | 5.999990 | V,U,P,T,R | `AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit-v2` |
| `AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-4bit` | `7db481bd6258` | 1.0.1 | no | 5.490070 | V,U,P,R | `AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-4bit-v2` |
| `AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit` | `2e11a710290e` | 1.0.1 | no | 5.999935 | V,U,P,T,R | `AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit-v2` |
| `AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit` | `e30f2476cc20` | 1.0.0 | no | 5.150021 | V,U,P | `AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit-v2` |
| `AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit` | `0ae103ab4a51` | 1.0.0 | no | 5.999949 | V,U,P,T | `AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit-v2` |
| `AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit` | `a2bf4b597b75` | 1.0.0 | no | 4.789199 | V,U,P,S | `AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit-v2` |
| `AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit` | `84be62ed37b6` | 1.0.0 | no | 5.980108 | V,U,P,T,S | `AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit-v2` |
| `AX-Qwen3-Coder-Next-MLX-AXQ-4bit` | `edb845770821` | 1.0.1 | no | 15.730019 | V,U,P,D,E | `AX-Qwen3-Coder-Next-MLX-AXQ-4bit-v2` |
| `AX-Qwen3-Coder-Next-MLX-AXQ-6bit` | `9509c38ec927` | 1.0.1 | no | 15.730019 | V,U,P,T,D,E | `AX-Qwen3-Coder-Next-MLX-AXQ-6bit-v2` |
| `AX-Qwen3-Embedding-0.6B-MLX-AXQ-4bit` | `1d020493ec6d` | 1.0.1 | no | 5.550330 | V,U,P | `AX-Qwen3-Embedding-0.6B-MLX-AXQ-4bit-v2` |
| `AX-Qwen3-Embedding-0.6B-MLX-AXQ-8bit` | `c807a6091e7c` | 1.0.1 | no | 8.000275 | V,U,P,T | `AX-Qwen3-Embedding-0.6B-MLX-AXQ-8bit-v2` |
| `AX-Qwen3-Embedding-4B-MLX-AXQ-4bit` | `db0f06460456` | 1.0.1 | no | 4.890183 | V,U,P | `AX-Qwen3-Embedding-4B-MLX-AXQ-4bit-v2` |
| `AX-Qwen3-Embedding-4B-MLX-AXQ-8bit` | `32f24f1f354a` | 1.0.1 | no | 7.999979 | V,U,P,T | `AX-Qwen3-Embedding-4B-MLX-AXQ-8bit-v2` |
| `AX-Qwen3-Embedding-8B-MLX-AXQ-4bit` | `ed2873c7a533` | 1.0.1 | no | 4.830057 | V,U,P | `AX-Qwen3-Embedding-8B-MLX-AXQ-4bit-v2` |
| `AX-Qwen3-Embedding-8B-MLX-AXQ-8bit` | `a9778699834a` | 1.0.1 | no | 7.999911 | V,U,P,T | `AX-Qwen3-Embedding-8B-MLX-AXQ-8bit-v2` |
| `AX-Qwen3.5-9B-MLX-AXQ-4bit-MTP` | `0360978ffa26` | 1.0.0 | no | 6.736665 | V,U,P,D | `AX-Qwen3.5-9B-MLX-AXQ-4bit-v2-MTP` |
| `AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP` | `fba8cc8fc328` | 1.0.0 | no | 6.736665 | V,U,P,T,D | `AX-Qwen3.5-9B-MLX-AXQ-6bit-v2-MTP` |
| `AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP` | `c2ff69331547` | 1.0.0 | yes | 5.418315 | V,P | `AX-Qwen3.6-27B-MLX-AXQ-4bit-v2-MTP` |
| `AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP` | `469c7898e707` | 1.0.0 | yes | 5.844833 | V,P,T | `AX-Qwen3.6-27B-MLX-AXQ-6bit-v2-MTP` |
| `AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP` | `b87858dbcf12` | 1.0.0 | yes | 4.878782 | V,P | `AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-v2-MTP` |
| `AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP` | `de13b9b581a9` | 1.0.0 | yes | 5.759473 | V,P,T | `AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-v2-MTP` |

## v2 rebuild and publication contract

Each v2 variant must satisfy all of the following:

1. Build from the source repository and immutable revision listed below.
2. Use AXQuant 1.2.0 from base revision
   `fc48b59f13cfbddcf472a2656b5f10a3c39bd5a3` plus the narrowly scoped v2 migration fixes in
   this change (versioned naming/cards and strict MLX-LM output-name/shape verification).
3. Complete the fail-closed plan-predicate conversion with no recorded quantizer fallback.
4. Record the corrected target class derived from the effective budget.
5. Remove local source paths and pass the publication privacy scan.
6. Pass MLX-LM generation before upload.
7. Upload to a new private v2 repository, verify the uploaded tree, sizes, LFS SHA-256 values,
   and downloaded manifest bytes, then make that verified revision public.
8. Pass MLX-LM generation again with the public repository identity and immutable Hub revision.
9. Keep the development-evidence banner and avoid quality, speed, MTP, or certification claims
   that the artifact does not support.

The installed AX Engine benchmark CLI did not produce a validated, parseable ready manifest for
the representative fleet conversions. The v2 factory therefore uses
`--ax-engine-manifest skip` for every variant. Cards explicitly state that AX Engine execution is
not established when `model-manifest.json` is absent; the passing pre- and post-upload runtime
checks in this migration are MLX-LM standard text/backbone checks only.

The Ministral tokenizer metadata triggers a Transformers compatibility warning unless
`fix_mistral_regex` is passed explicitly. Inspection of the serialized backend confirmed that
the pinned 2512 BF16 tokenizers already contain the corrected split pattern and byte-level
pre-tokenizer, so the migration preserves the upstream tokenizer bytes instead of applying a
second, unrecorded tokenizer rewrite.

## Pinned source revisions

| Source | Revision |
| --- | --- |
| `google/gemma-4-12b` | `023679ed352de9bb66cc873c9009ce3482585c08` |
| `mistralai/Devstral-Small-2505` | `c2a9d81a2989af566682b4cecc828c84556076c5` |
| `mistralai/Ministral-3-14B-Instruct-2512-BF16` | `3cea74c1ebaf5ce5f5a2553de470e2ceab825142` |
| `mistralai/Ministral-3-8B-Instruct-2512-BF16` | `f6fae9795746f63c9be8344932f01275f3c63734` |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `68faf511d618ef198fef186659617cfd2eb8e33a` |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `2d59de1cbd51c0adf384eb906b766d1aee0e0517` |
| `openbmb/MiniCPM5-1B` | `4e9de7a0778dc1c362e983e6858f0e77542cbdca` |
| `Qwen/Qwen3-Coder-Next` | `a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb` |
| `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` |
| `Qwen/Qwen3-Embedding-8B` | `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| `Qwen/Qwen3.6-27B` | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| `Qwen/Qwen3.6-35B-A3B` | `995ad96eacd98c81ed38be0c5b274b04031597b0` |

## Execution phases

| Phase | Variants | Scope |
| --- | ---: | --- |
| 1 | 12 | MiniCPM5, Qwen3 Embedding, Qwen 3.5, Gemma-4 |
| 2 | 8 | Ministral 3, Devstral, Mistral Small 3.1 |
| 3 | 4 | Qwen 3.6 dense and MoE MTP packs |
| 4 | 4 | Nemotron 3 Nano and corrected Qwen3-Coder-Next |

## Completed migration

The migration completed on 2026-08-05. The final live-Hub audit established:

- 28/28 expected v2 repositories are public, current, and receipt-bound;
- all 28 pre-upload and post-upload MLX-LM runtime checks passed;
- 14/14 source checkpoints resolve at their pinned immutable revisions;
- remote trees, file sizes, non-LFS hashes, LFS SHA-256 values, downloaded manifests, plans,
  quantizer-execution records, source identities, target classes, cards, sibling links, and
  publication privacy all passed independent re-verification;
- the Qwen3-Coder-Next plans contain 73,872 low-bit expert assignments per variant, with distinct
  4-bit and 6-bit payload sizes and measured main BPW below 8;
- the fleet contains 394,448,965,023 weight bytes and 395,136,550,939 total repository bytes; and
- all 28 v2 repositories are present in the
  [AutomatosX MLX model catalog](https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog).

These remain development artifacts, not certified releases. None includes a validated native AX
Engine manifest; the runtime evidence recorded here is for MLX-LM standard text/backbone
inference.

| Public v2 repository | Audited Hub revision | Main BPW |
| --- | --- | ---: |
| [`AX-Devstral-Small-2505-MLX-AXQ-4bit-v2`](https://huggingface.co/AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-4bit-v2) | `b547d596fc5268f36029928a7fcd91b7cb60a8f7` | 4.949963 |
| [`AX-Devstral-Small-2505-MLX-AXQ-6bit-v2`](https://huggingface.co/AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-6bit-v2) | `2c14778cd319c14280f51cb6b4427647314d0fc6` | 5.999989 |
| [`AX-MiniCPM5-1B-MLX-AXQ-4bit-v2`](https://huggingface.co/AutomatosX/AX-MiniCPM5-1B-MLX-AXQ-4bit-v2) | `c28c752c81d96335d2a2cfab77357ff9b195a4bf` | 7.380428 |
| [`AX-MiniCPM5-1B-MLX-AXQ-6bit-v2`](https://huggingface.co/AutomatosX/AX-MiniCPM5-1B-MLX-AXQ-6bit-v2) | `26c50f8435aa52e96801496fd1cfef2d53a01457` | 7.380428 |
| [`AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit-v2`](https://huggingface.co/AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit-v2) | `4bf9b3bf9da726100b2d0642ba8342e6a69cab17` | 5.610033 |
| [`AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit-v2`](https://huggingface.co/AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit-v2) | `4854431159cef3d529dcef493c7699961f16caf7` | 5.999912 |
| [`AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-4bit-v2`](https://huggingface.co/AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-4bit-v2) | `2e5b138e62ee6e54eda6ff722e19e2b38588d72a` | 5.990115 |
| [`AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit-v2`](https://huggingface.co/AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit-v2) | `65dc793805794c2143c113f859fe9339a91962ec` | 5.999992 |
| [`AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit-v2`](https://huggingface.co/AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit-v2) | `9937e3461228e3adf75487738b5e60f8dd54ba95` | 5.150021 |
| [`AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit-v2`](https://huggingface.co/AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit-v2) | `ec0a752f1ccc8cede408399c34df1d5ef8facad9` | 5.999949 |
| [`AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit-v2`](https://huggingface.co/AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit-v2) | `f21bf44e24ce40cde066e2c9faeef3291f99324e` | 4.799310 |
| [`AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit-v2`](https://huggingface.co/AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit-v2) | `ea5b4fc88fc319a81c02000d818301626845bfcd` | 5.990219 |
| [`AX-Qwen3-Coder-Next-MLX-AXQ-4bit-v2`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-4bit-v2) | `dcebf1bf46011785f729f93fa59031aed99b15fa` | 4.797752 |
| [`AX-Qwen3-Coder-Next-MLX-AXQ-6bit-v2`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit-v2) | `df815f1ec80c81b62119f2072034d55bc7c3ea80` | 5.998996 |
| [`AX-Qwen3-Embedding-0.6B-MLX-AXQ-4bit-v2`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-AXQ-4bit-v2) | `63a185caa82078a2dd7712b04b2817169ad5583d` | 5.550330 |
| [`AX-Qwen3-Embedding-0.6B-MLX-AXQ-8bit-v2`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-AXQ-8bit-v2) | `1ca9a9b5fb0fde6a1e05090a0a7045c27adcd74c` | 8.000275 |
| [`AX-Qwen3-Embedding-4B-MLX-AXQ-4bit-v2`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-4B-MLX-AXQ-4bit-v2) | `6c2ea6af97b48668bb269ca58b135833d94cbd4f` | 4.890183 |
| [`AX-Qwen3-Embedding-4B-MLX-AXQ-8bit-v2`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-4B-MLX-AXQ-8bit-v2) | `4471a236ffc039bfbd692ae6e693cd834954e8dd` | 7.999979 |
| [`AX-Qwen3-Embedding-8B-MLX-AXQ-4bit-v2`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-8B-MLX-AXQ-4bit-v2) | `e1662c17f6ebebd30babfb92ad79b94c1a92ec5d` | 4.830057 |
| [`AX-Qwen3-Embedding-8B-MLX-AXQ-8bit-v2`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-8B-MLX-AXQ-8bit-v2) | `f9fff0c2b21270a5bbececf8560c6bcd1282b753` | 7.999911 |
| [`AX-Qwen3.5-9B-MLX-AXQ-4bit-v2-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.5-9B-MLX-AXQ-4bit-v2-MTP) | `246761f2d4b7a8a7a372bf293e080436b41ad7e6` | 6.736665 |
| [`AX-Qwen3.5-9B-MLX-AXQ-6bit-v2-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.5-9B-MLX-AXQ-6bit-v2-MTP) | `f402dfda014b11ef6c5c2c06fe4971d91c08c8f8` | 6.736665 |
| [`AX-Qwen3.6-27B-MLX-AXQ-4bit-v2-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-v2-MTP) | `7d6513a57145c693989cc9fda796f2e287774504` | 5.418315 |
| [`AX-Qwen3.6-27B-MLX-AXQ-6bit-v2-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-v2-MTP) | `d5ea5f4bf870aa943f42525704d82bb44e955e73` | 5.844833 |
| [`AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-v2-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-v2-MTP) | `37759d1ea5c61a714d3664391da7ff51eceda448` | 4.878782 |
| [`AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-v2-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-v2-MTP) | `947f16be1ff49ac0bf6a2f0233c78c0f7def0dac` | 5.759473 |
| [`AX-gemma-4-12b-MLX-AXQ-4bit-v2`](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-v2) | `9b01caa4c0bcff82fd15679a2a5bdfa0bb396f2e` | 4.890033 |
| [`AX-gemma-4-12b-MLX-AXQ-6bit-v2`](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-v2) | `86616f7f9bc70c595c8df9173890b0733cec43ee` | 6.000088 |
