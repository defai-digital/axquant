# AXQ model fleet v2 migration

Audit date: 2026-08-05
Hub owner: `AutomatosX`
Scope: every public model repository whose name contains `AXQ` and does not already carry the
v2 artifact edition

## Decision

Rebuild all 28 existing AXQ development packs as artifact edition v2 while preserving their
original repository identifiers. The audited v2 revision is served by `main` and tagged `v2`;
the exact artifact replaced on `main` is preserved at `legacy-pre-v2`. Edition-suffixed
repositories were temporary migration staging and are removed only after the stable-name fleet
passes a second complete Hub audit.

This is a real checkpoint rebuild, not a model-card-only refresh. The audited fleet is
598.301 GB across 28 repositories and 14 source checkpoints.

## Post-v2 certified supersession

On 2026-08-08, `AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP` was replaced on `main` by a materially new,
measured **artifact edition v3** after passing
[checkpoint Tier 1 certification](certifications/qwen36-27b-axq6-tier1.md). Its old v2 artifact
remains available at the immutable `v2` tag; certified v3 is Hub commit
`cdd13bf81cf21818a01cf59a31fc116ef84326bc`. This one supersession does not change the
historical v2 audit below and does not promote any other fleet entry.

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
  dominate both requested budgets (the misleading AXQ-4bit siblings were later removed from the
  public catalog; see [Floor-collapsed 4bit retirement](#floor-collapsed-4bit-retirement));
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

| Stable repository | Legacy Hub rev | AXQuant | Source pinned | BPW | Issues |
| --- | --- | ---: | --- | ---: | --- |
| `AX-Devstral-Small-2505-MLX-AXQ-4bit` | `6d04a0c65dbb` | 1.0.0 | no | 4.949963 | V,U,P |
| `AX-Devstral-Small-2505-MLX-AXQ-6bit` | `7086f12e3b3b` | 1.0.0 | no | 5.999989 | V,U,P,T |
| `AX-gemma-4-12b-MLX-AXQ-4bit` | `5192374d4daa` | 1.0.0 | no | 4.890033 | V,U,P |
| `AX-gemma-4-12b-MLX-AXQ-6bit` | `0003ab1be26a` | 1.0.0 | no | 6.000088 | V,U,P,T |
| `AX-MiniCPM5-1B-MLX-AXQ-4bit` | `9fc3fb996a35` | 1.0.0 | no | 7.380428 | V,U,P,D |
| `AX-MiniCPM5-1B-MLX-AXQ-6bit` | `f28d93155bb5` | 1.0.0 | no | 7.380428 | V,U,P,T,D |
| `AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit` | `a41e0128dd2e` | 1.0.1 | no | 5.279903 | V,U,P,R |
| `AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit` | `8289fa74c71f` | 1.0.1 | no | 5.999990 | V,U,P,T,R |
| `AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-4bit` | `7db481bd6258` | 1.0.1 | no | 5.490070 | V,U,P,R |
| `AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit` | `2e11a710290e` | 1.0.1 | no | 5.999935 | V,U,P,T,R |
| `AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit` | `e30f2476cc20` | 1.0.0 | no | 5.150021 | V,U,P |
| `AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit` | `0ae103ab4a51` | 1.0.0 | no | 5.999949 | V,U,P,T |
| `AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit` | `a2bf4b597b75` | 1.0.0 | no | 4.789199 | V,U,P,S |
| `AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit` | `84be62ed37b6` | 1.0.0 | no | 5.980108 | V,U,P,T,S |
| `AX-Qwen3-Coder-Next-MLX-AXQ-4bit` | `edb845770821` | 1.0.1 | no | 15.730019 | V,U,P,D,E |
| `AX-Qwen3-Coder-Next-MLX-AXQ-6bit` | `9509c38ec927` | 1.0.1 | no | 15.730019 | V,U,P,T,D,E |
| `AX-Qwen3-Embedding-0.6B-MLX-AXQ-4bit` | `1d020493ec6d` | 1.0.1 | no | 5.550330 | V,U,P |
| `AX-Qwen3-Embedding-0.6B-MLX-AXQ-8bit` | `c807a6091e7c` | 1.0.1 | no | 8.000275 | V,U,P,T |
| `AX-Qwen3-Embedding-4B-MLX-AXQ-4bit` | `db0f06460456` | 1.0.1 | no | 4.890183 | V,U,P |
| `AX-Qwen3-Embedding-4B-MLX-AXQ-8bit` | `32f24f1f354a` | 1.0.1 | no | 7.999979 | V,U,P,T |
| `AX-Qwen3-Embedding-8B-MLX-AXQ-4bit` | `ed2873c7a533` | 1.0.1 | no | 4.830057 | V,U,P |
| `AX-Qwen3-Embedding-8B-MLX-AXQ-8bit` | `a9778699834a` | 1.0.1 | no | 7.999911 | V,U,P,T |
| `AX-Qwen3.5-9B-MLX-AXQ-4bit-MTP` | `0360978ffa26` | 1.0.0 | no | 6.736665 | V,U,P,D |
| `AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP` | `fba8cc8fc328` | 1.0.0 | no | 6.736665 | V,U,P,T,D |
| `AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP` | `c2ff69331547` | 1.0.0 | yes | 5.418315 | V,P |
| `AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP` | `469c7898e707` | 1.0.0 | yes | 5.844833 | V,P,T |
| `AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP` | `b87858dbcf12` | 1.0.0 | yes | 4.878782 | V,P |
| `AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP` | `de13b9b581a9` | 1.0.0 | yes | 5.759473 | V,P,T |

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
7. Verify each isolated rebuild's uploaded tree, sizes, LFS SHA-256 values, and downloaded
   manifest bytes before promotion.
8. Tag the stable repository's current revision `legacy-pre-v2`, promote the exact audited v2
   tree to `main`, verify it again, and bind the resulting revision to tag `v2`.
9. Pass MLX-LM generation again with the stable public repository identity and immutable Hub
   revision.
10. Keep the development-evidence banner and avoid quality, speed, MTP, or certification claims
   that the artifact does not support.
11. Remove temporary edition-suffixed staging repositories only after every stable repository,
    both tags, the catalog, and all publication evidence pass the final fleet audit.

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

The migration and stable-name cleanup completed on 2026-08-05. The final live-Hub audit
established:

- 28/28 expected stable repositories are public, current, and receipt-bound;
- each stable `main` equals its immutable `v2` tag, while `legacy-pre-v2` resolves to the exact
  revision that `main` served before promotion;
- all 28 pre-upload and post-upload MLX-LM runtime checks passed;
- 14/14 source checkpoints resolve at their pinned immutable revisions;
- remote trees, file sizes, non-LFS hashes, LFS SHA-256 values, downloaded manifests, plans,
  quantizer-execution records, source identities, target classes, cards, sibling links, and
  publication privacy all passed independent re-verification;
- the Qwen3-Coder-Next plans contain 73,872 low-bit expert assignments per variant, with distinct
  4-bit and 6-bit payload sizes and measured main BPW below 8;
- the fleet contains 394,448,965,023 weight bytes and 395,136,555,823 total repository bytes;
- all 28 stable repositories are present in the
  [AutomatosX MLX model catalog](https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog);
  and
- all 28 temporary edition-suffixed repositories were removed from the catalog and deleted only
  after the canonical fleet audit passed; the post-deletion audit found none remaining.

The v2 artifacts recorded by this audit remain development artifacts. The later Qwen 3.6 27B AXQ
6-bit v3 supersession is the sole checkpoint Tier 1 certified exception and includes a validated
native AX Engine manifest; its MTP acceleration tier remains uncertified.

### Floor-collapsed 4bit retirement

On 2026-08-08 the following **AXQ-4bit** Hub repositories were **deleted** because they were
not a real storage/quality alternative to their 6bit siblings:

| Deleted 4bit repository | Canonical pack to use | Reason |
| --- | --- | --- |
| `AX-Qwen3.5-9B-MLX-AXQ-4bit-MTP` | [`AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP) | Identical weights; both budgets raised to ~6.97 BPW |
| `AX-MiniCPM5-1B-MLX-AXQ-4bit` | [`AX-MiniCPM5-1B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-MiniCPM5-1B-MLX-AXQ-6bit) | Identical weights; both budgets raised to ~7.38 BPW |
| `AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-4bit` | [`AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit) | Near-identical size/BPW (~5.99 vs ~6.00); no useful 4 vs 6 trade-off |

Most other AXQ bases still publish both 4bit and 6bit (or 4bit and 8bit for embeddings) when the
budgets produce distinct artifacts. Model-card generation records the no-4bit stems in
`axquant.model_card` so regenerated cards do not link to removed siblings.

The historical v2 audit table below still lists the pre-retirement 4bit repository IDs for
traceability; those three 4bit rows are **no longer public Hub models**.

| Stable public repository | Audited v2 revision | `legacy-pre-v2` revision | Main BPW |
| --- | --- | --- | ---: |
| [`AX-Devstral-Small-2505-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-4bit) | `17e0ce81a7d6aeb6729a0c84b92340e26fbe1a6d` | `6d04a0c65dbb201b9a80d12f98ba86defc711c7d` | 4.949963 |
| [`AX-Devstral-Small-2505-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-6bit) | `04be51a3173b94e0a0d859be871cfb7a749405d2` | `7086f12e3b3b0075b3668df30b712fbc7addb0e4` | 5.999989 |
| ~~`AX-MiniCPM5-1B-MLX-AXQ-4bit`~~ (deleted 2026-08-08) | `df7ace2359f2e42684e8f35d23e4f6df6c4810fc` | `9fc3fb996a3594c3fe7bee58de4d1d7119b36bda` | 7.380428 |
| [`AX-MiniCPM5-1B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-MiniCPM5-1B-MLX-AXQ-6bit) | `9687cba71d5ecacea70f0467e55a4c3411b7eb19` | `f28d93155bb565a05a0a2290c0091b31bf8449f1` | 7.380428 |
| [`AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit) | `669dda7a7d78e2fa167d6dae70128f8cf2fe778b` | `a41e0128dd2ee145caeb5cd6f1ba66ecc95c8617` | 5.610033 |
| [`AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit) | `74cc761a1f6f3e2d0e8bbb4d3d8c15cd17ef221a` | `8289fa74c71f46d62eb78e679cc343d95d3231d7` | 5.999912 |
| ~~`AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-4bit`~~ (deleted 2026-08-08) | `6dbeb485860fc2395204068419d081604d1bf759` | `7db481bd6258dad712a56cb3c578d6df190e44c0` | 5.990115 |
| [`AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit) | `0821405e77f4161424b09cffd8768e2f5453d95e` | `2e11a710290e27fc3c4971314d9a4dcad306f89e` | 5.999992 |
| [`AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit) | `91c20bd52f6c16b6b7e6f6e60b0a859ddd1ad8b0` | `e30f2476cc20dbb0a55883946f368fc815b57f88` | 5.150021 |
| [`AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit) | `f00654783b3e3b2a020a712161eb1ac7861da348` | `0ae103ab4a5163b3bf0e615e29d9476763a42970` | 5.999949 |
| [`AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit) | `cb2db117e80571afa466644e91ec39bd528ccf7f` | `a2bf4b597b7535fad8d64cdb6ad04a4bf291659c` | 4.799310 |
| [`AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit) | `a4dcc84b9b7318cc206f2b17dbc1555883cf67fd` | `84be62ed37b67dcd93fa4649886de28ad2ef2a4d` | 5.990219 |
| [`AX-Qwen3-Coder-Next-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-4bit) | `53dce509aa115e7fae583516b494a5dafebf31a9` | `edb845770821f874f343c61985af150f5587ba22` | 4.797752 |
| [`AX-Qwen3-Coder-Next-MLX-AXQ-6bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit) | `c6f3ae556f95ce13b7d319486ad2d4d753726216` | `9509c38ec927d6b1606bca21569b3af432840986` | 5.998996 |
| [`AX-Qwen3-Embedding-0.6B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-AXQ-4bit) | `af35a52e317dd12b6b70d847f8c170e823bee28d` | `1d020493ec6d5aa0ead13045ab187dd4cf27bef1` | 5.550330 |
| [`AX-Qwen3-Embedding-0.6B-MLX-AXQ-8bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-AXQ-8bit) | `2e6255546e5f45b7eca5debda547f15b84a30836` | `c807a6091e7c49948a9cdeba7e64372f653df4e7` | 8.000275 |
| [`AX-Qwen3-Embedding-4B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-4B-MLX-AXQ-4bit) | `05d1060acb93135650d08e65eb701653a1d9fa00` | `db0f064604567342c502837e3b66f2bfb4f133ee` | 4.890183 |
| [`AX-Qwen3-Embedding-4B-MLX-AXQ-8bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-4B-MLX-AXQ-8bit) | `4abc15919c3ffc00080a1857c50d55fd401d98ee` | `32f24f1f354a0777fa4373a2cd6c28ae93e3e5a6` | 7.999979 |
| [`AX-Qwen3-Embedding-8B-MLX-AXQ-4bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-8B-MLX-AXQ-4bit) | `a5cced82bfc1324b52eacbb499ac2f6463ba85f2` | `ed2873c7a533dfd76a56b2e735382cf2034275e6` | 4.830057 |
| [`AX-Qwen3-Embedding-8B-MLX-AXQ-8bit`](https://huggingface.co/AutomatosX/AX-Qwen3-Embedding-8B-MLX-AXQ-8bit) | `10854555717aa09e74c6e1b083004b399b58691e` | `a9778699834adaf73b095deb6d69c00934af9e00` | 7.999911 |
| ~~`AX-Qwen3.5-9B-MLX-AXQ-4bit-MTP`~~ (deleted 2026-08-08) | `c79379b1d22449c87bafcfa056082a2b9994dbc9` | `0360978ffa26e13483c788a53459d5e600fefd1d` | 6.736665 |
| [`AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.5-9B-MLX-AXQ-6bit-MTP) | `7acb8810588f2bb3380ca96daac2afaa1ced6d19` | `fba8cc8fc3283946e02e21af6368154aa33f29bd` | 6.736665 |
| [`AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP) | `6182ccbc41c7397ff90670f740c6d9eacfa4b09f` | `c2ff693315475640fa71a2fd6d6f95ce67ac86e9` | 5.418315 |
| [`AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP) | `8c37715c7b5f5ebca00eda6f73be47116a3e4ebc` | `469c7898e707c0e04241270bee8914323ce78270` | 5.844833 |
| [`AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP) | `3b496c6b1de84c700cbd2571a5a41671a8bb076c` | `b87858dbcf129c1991234683076ecf687833c6e1` | 4.878782 |
| [`AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP) | `d339afd98930abe4c1731f0b65336f6af78acccc` | `de13b9b581a9d5ca273a752ca44169611d9bd442` | 5.759473 |
| [`AX-gemma-4-12b-MLX-AXQ-4bit-MTP`](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP) | `585587770cb4a2fc541e4570afe009d9674e6755` | `5192374d4daa6dc45d783353ccff4a64d41e293f` | 4.890033 |
| [`AX-gemma-4-12b-MLX-AXQ-6bit-MTP`](https://huggingface.co/AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP) | `71d438a4016762ea9c40aaf097a420fa413bfef0` | `0003ab1be26ae5a51a824ed511847c3d540c9005` | 6.000088 |
