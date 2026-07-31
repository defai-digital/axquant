# AXQuant Expansion Technical Specification

**Document status:** Accepted for implementation  
**Applies to:** Expansion phases E1–E6 (`expansion-prd.md`), decisions AXQ-017–AXQ-022  
**Base document:** `technical-specification.md` remains authoritative for everything this
specification does not amend.  
**Last reviewed:** 2026-07-31

## 1. Scope

This specification defines the contracts for:

1. the tiered family-support model (AXQ-017);
2. the declarative dense-family adapter framework (AXQ-018);
3. the `axquant quantize` quick-conversion command (AXQ-019);
4. recipe bundles (AXQ-020);
5. the KV-cache plan extension (AXQ-021).

### 1.1 Capability truth table

| Capability | E1 | E2 | E3 | E4 | Later |
| --- | --- | --- | --- | --- | --- |
| `SupportTier` recorded in profile/inventory | ✅ | | | | |
| Tier enforcement in `convert` | ✅ | | | | |
| `DenseFamilySpec` + shared adapter | ✅ | | | | |
| Qwen 3.5 dense spec (`inspect-only`→`convertible`) | ✅ | | | | promotion evidence |
| Gemma-4 / MiniCPM5 / Nemotron 3 dense specs (`inspect-only`) | ✅ | | | | promotion evidence |
| `axquant quantize` one-command pipeline | | ✅ | | | |
| Recipe bundle schema + local resolution | | | ✅ | | |
| Remote `hf://` resolution + release bundle packaging + `support-matrix` | | | ✅ | | |
| KV plan section + prior allocator + AX metadata | | | | ✅ | |
| Measured KV probing + digest-bound measured allocation (AXQ-024) | | | | ✅ | release KV gates |
| Tier gate in `publish` | ✅ | | | | |

## 2. Tiered family support

### 2.1 Schema

`schema.py` gains:

```python
class SupportTier(StrEnum):
    CERTIFIED = "certified"
    CONVERTIBLE = "convertible"
    INSPECT_ONLY = "inspect-only"
```

`ArchitectureProfile` gains one additive field:

```python
support_tier: SupportTier = SupportTier.INSPECT_ONLY
```

The field is additive with a safe default, so `axquant.inventory.v1` and
`axquant.plan.v1` literals do not change; artifacts written by older versions
load with `inspect-only`, which is fail-closed (conversion refuses). Existing
`ArchitectureSupportLevel` is unchanged and continues to describe *mechanical*
adapter capability; `SupportTier` describes *evidence-backed permission*.
Invariant: `support_tier > inspect-only` requires
`support_level == supported`; the constructor path in each adapter maintains it
and `convert` re-checks it.

### 2.2 Tier semantics

| Tier | inspect | convert / quantize | publish (official catalog) | Claims |
| --- | --- | --- | --- | --- |
| `inspect-only` | ✅ | refused | refused | inventory facts only |
| `convertible` | ✅ | permitted | only through the release audit (see 2.3) | "convertible (development)" |
| `certified` | ✅ | permitted | existing gates apply | existing claim policy |

Tier assignment lives in the adapter (per family and, where relevant, per
size). Qwen3.6-27B maps its current "supported" state to `certified` scope
semantics only where existing release language already applies; the adapter
reports `convertible` until the formal release audit passes, at which point the
registry entry for that size is promoted in code with a reference to the audit
artifact. Promotions are code changes with test coverage, never runtime state.

### 2.3 Enforcement points (as implemented)

- `inspector` records the tier in the emitted `Inventory` (via the profile).
- The conversion scope guard (`assert_qwen36_conversion_scope`) refuses
  `inspect-only` plans with a message naming the tier and AXQ-017.
- `prepare_publication` refuses a packaged plan recording an `inspect-only`
  family. For `convertible`, the **certified gate is the existing
  executed-publication release-audit requirement**: passing M0–M8 is exactly
  the AXQ-017 certification evidence, so no duplicate tier check is layered on
  top of it. After a family's first audit passes, its adapter is promoted to
  report `certified` in code (with the audit reference), which is what future
  inspections and claims render.
- `report` renders the family and tier in the plan report table.

## 3. Declarative dense-family adapter framework

### 3.1 `DenseFamilySpec`

New module `architectures/dense_family.py`:

```python
@dataclass(frozen=True)
class DenseFamilySpec:
    adapter_id: str  # e.g. "qwen35-dense-v1"
    product_family: str  # e.g. "qwen3.5"
    model_types: tuple[str, ...]  # accepted config "model_type" values
    reference_pattern: str  # regex over model reference / _name_or_path
    support_tier: SupportTier  # default tier for matched checkpoints
    layer_count_keys: tuple[str, ...]  # config paths for num_hidden_layers
    text_config_key: str | None  # nested text config key, if any
    extra_role_patterns: tuple[tuple[str, TensorRole], ...] = ()
    notes: tuple[str, ...] = ()
```

`DenseFamilyAdapter(spec)` implements the `ArchitectureAdapter` protocol:

- `matches`: config `model_type` ∈ `spec.model_types` **and** reference
  pattern match (same double-keying as `Qwen36Adapter.matches` so a config
  from one family inside a mislabeled directory does not silently match);
- `profile`: extracts density (absence of MoE keys), layer count via
  `layer_count_keys`, vision presence, MTP declaration; emits the spec tier
  (downgraded to `inspect-only` whenever the checkpoint is non-dense or a
  required extraction fails — fail closed);
- `classify_tensor`: `spec.extra_role_patterns` first (most specific wins),
  then the shared dense classification table, which is the existing Qwen 3.6
  token table factored into `dense_family.py` and reused by `Qwen36Adapter`
  so there is exactly one protection-critical classifier.

### 3.2 Registry

`architectures/registry.py`:

- `_ADAPTERS` becomes the ordered tuple: `Qwen36Adapter()` first, then one
  `DenseFamilyAdapter` per spec, most specific patterns first.
- `adapter_for` collects **all** matches; more than one match raises
  `ArtifactError` naming the contending adapter ids (ambiguity is an error
  per AXQ-018). Zero matches returns `None` (generic inventory, unchanged).

### 3.3 Initial family specs

| Spec id | Family | model_type keys | Initial tier |
| --- | --- | --- | --- |
| `qwen35-dense-v1` | qwen3.5 | `qwen3_5` (non-3.6 reference) | `inspect-only` |
| `gemma4-dense-v1` | gemma-4 | `gemma4`, `gemma4_text` | `inspect-only` |
| `minicpm5-dense-v1` | minicpm5 | `minicpm5` | `inspect-only` |
| `nemotron3-dense-v1` | nemotron3 | `nemotron3` | `inspect-only` |

Ordering note: `qwen35-dense-v1` shares `model_type` `qwen3_5` with Qwen 3.6;
its reference pattern must exclude 3.6 references (negative match) and the
registry ambiguity check must have a regression test for a 3.6 checkpoint.
Tier promotion to `convertible` per family requires the AXQ-017 promotion
evidence and lands as a one-line spec change plus its test.

## 4. `axquant quantize` (quick conversion)

### 4.1 Contract

```text
axquant quantize
  --model PATH                # local MLX checkpoint directory
  --model-id ID               # optional; defaults from config/_name_or_path
  --revision REV              # optional for quick mode; recorded when given
  --output DIR                # output artifact directory
  [--recipe BUNDLE]           # recipe bundle file or directory (section 5); else priors
  [--calibration-manifest P]  # required by convert when the bundle plan is measured
  [--target-bpw N]            # planner target when no recipe; default 4.8
  [--kv-cache {off,prior}]    # default: off (section 6)
  [--profile NAME]            # default: general
  [--mtp-sidecar PATH]        # auto-discovered from the source dir when omitted
  [--runtime-smoke {mlx-lm,ax-engine,none}]  # default: none
  [--ax-engine-manifest {required,if-available,skip}]  # default: if-available
  [--json PATH]               # machine-readable axquant.quantize-summary.v1
```

Exit codes: `0` success; `1` runtime smoke failed; `2` any stage failure or
usage error (existing convention).

### 4.2 Pipeline

One process, existing stage implementations, no parallel pipeline:

1. **inspect** — in-memory `Inventory` (not written into the artifact
   directory, which stays exactly the converter's atomic output);
2. **tier gate** — refuse below `convertible`;
3. **plan** — with `--recipe`: verify and apply the bundle (section 5).
   Without: prior-based analyze → plan with `--target-bpw`. (Family default
   recipes ship as published recipe bundles rather than wheel-embedded
   `examples/` files.);
4. **convert** — the existing converter with development-evidence semantics
   (equivalent to `--allow-unmeasured`), atomic staging unchanged;
5. **runtime smoke** — optional, via the existing `runtime-check` machinery;
6. **summary** — human-readable block plus optional `--json`: tier, evidence
   kind, plan source (recipe id / default recipe / priors), measured BPW,
   output path, and the sentence
   `This artifact is development evidence; it is not a certified AXQuant release.`
   whenever evidence kind is not measured-release.

### 4.3 Guardrails

- Quick mode never writes release-gate artifacts (no validation, audit, or
  publication inputs) and its manifests carry the development evidence kind.
- `publish` continues to require the staged pipeline's evidence set; nothing
  emitted by quick mode satisfies it.
- Failure surfaces the underlying stage error unchanged (fail-closed coverage,
  unclassified tensors, tier refusals).

## 5. Recipe bundles

### 5.1 Schema

New artifact `axquant.recipe-bundle.v1`:

```python
class RecipeBundle(StrictModel):
    schema_version: Literal["axquant.recipe-bundle.v1"]
    bundle_id: str  # e.g. "qwen36-27b-measured-r1"
    source_model: ModelIdentity  # id + revision are mandatory here
    evidence_kind: EvidenceKind  # inherited from the payload; never upgraded
    payload_kind: Literal["plan", "manual-recipe"]
    payload_file: str  # relative path of the plan/recipe JSON/YAML
    payload_sha256: str  # SHA-256 over payload bytes
    lineage: dict[str, str]  # digests of producing sensitivity/calibration artifacts
    axquant_version: str
    notes: list[str]
    created_at: datetime
```

A bundle is a directory containing `axquant_recipe_bundle.json` plus the
payload named by `payload_file` (resolution also accepts the record file path
directly). Resolution re-verifies that the payload's evidence kind equals the
recorded `evidence_kind`, so editing the record cannot upgrade evidence.

### 5.2 Resolution and verification

`quantize --recipe` accepts a local path (file or bundle directory) or a
revision-pinned remote reference `hf://OWNER/REPO@REVISION[/PATH]` (AXQ-023);
an unpinned remote reference fails closed. Verification, all fail-closed and
identical for both origins:

1. schema parse (strict);
2. payload digest equals the recorded digest;
3. `source_model.model_id` equals the target's model id, and revision equals
   the target revision when the caller pins one;
4. evidence kind is recorded into the resulting plan/manifest unchanged —
   a bundle never upgrades evidence (AXQ-020).

### 5.3 Export

`axquant recipe-export --plan P --bundle-id ID --output-dir D [--lineage
NAME=SHA256 ...] [--note ...]` writes a bundle from a revision-pinned plan,
copying the plan's evidence kind. `prepare_publication` additionally packages
every prepared release's plan as `recipe/axquant_recipe_bundle.json` (bundle id
`<repo-name>-recipe`) with lineage digests of the packaged plan, calibration
manifest, and both profile validations; re-preparation verifies the existing
bundle instead of rewriting it.

## 6. KV-cache plan extension

### 6.1 Schema

Additive, optional section on `QuantizationPlan` (schema literal unchanged;
absent section ⇒ exact current behavior):

```python
class KvLayerAllocation(StrictModel):
    layer_index: int  # 0-based text-path layer
    bits: int  # 4, 6, 8, or 16
    group_size: int
    reason: str


class KvCachePlan(StrictModel):
    schema_version: Literal["axquant.kv-plan.v1"] = "axquant.kv-plan.v1"
    allocation_basis: Literal["architecture-prior", "measured"]
    min_bits: int  # policy floor, default 4
    default_bits: int
    default_group_size: int
    layers: list[KvLayerAllocation]  # complete cover of text layers, no gaps
    warnings: list[str]


# on QuantizationPlan:
kv_cache: KvCachePlan | None = None
```

Validation invariants (enforced in the allocator and re-checked at convert
preflight when the section is present): layer indices are exactly
`0..text_layer_count-1` with no duplicates; every `bits ≥ min_bits`;
`allocation_basis="measured"` is rejected until measured KV probing exists.

### 6.2 Prior-based allocator

`planner.py` gains `allocate_kv_cache(layer_count, *, default_bits, min_bits,
group_size)`: the first and last `ceil(0.1 × layer_count)` layers receive
`max(8, default_bits)`-bit KV (boundary layers carry disproportionate
attention mass under the same priors the weight path already uses), interior
layers receive `default_bits`; every allocation records its prior in
`reason`. This is deliberately simple: its purpose is the contract and the
metadata path, not quality claims (AXQ-021 phase one).

### 6.3 Runtime metadata

When a plan carries `kv_cache`, `runtime.py` includes in the AX Engine
metadata: the per-layer table, `allocation_basis`, and an advisory MLX-LM
fallback (`kv_bits` = the modal planned bits, `kv_group_size` = the modal
group size) marked `advisory: true`. MLX-LM behavior is unchanged; the
fallback is documentation for operators, not an executed contract.

### 6.4 Measured KV sensitivity (AXQ-024)

`kv_probe.py` measures per-layer KV sensitivity: for each text layer and
candidate bit-width, a forward pass runs with only that layer's KV quantized
(`QuantizedKVCache`; all other layers BF16) over the same verified tokenized
calibration cache the weight probe uses, comparing logits against the all-BF16
baseline (output KL and token disagreement at fixed metric positions). Results
are recorded as `axquant.kv-sensitivity.v1` with complete layer coverage and
calibration provenance; output is development evidence
(`measured_development`). `analyze-kv` is the CLI entry;
`plan --kv-cache measured --kv-analysis R [--kv-max-kl B]` allocates the
lowest per-layer bits within the KL budget via `allocate_kv_cache_measured`,
binding the plan to the report by `sensitivity_sha256`. Conversion accepts a
measured KV plan only with that digest present. Release validation gates for
KV quality claims remain deferred.

## 7. Testing

- `tests/test_architectures.py`: spec-driven adapter matching (positive,
  negative, ambiguity error, qwen3.5-vs-3.6 disambiguation), tier fail-closed
  downgrades, shared-classifier equivalence for the existing Qwen 3.6 fixtures.
- `tests/test_inspector.py` / `test_converter.py`: tier recorded; convert
  refusal below `convertible`.
- New `tests/test_quantize.py`: quick-mode orchestration against the synthetic
  fixtures (stage order, summary content, development labeling, tier refusal,
  `--json` output).
- New `tests/test_recipe_bundle.py`: schema round-trip, digest and identity
  verification, evidence-kind inheritance, tamper detection.
- `tests/test_planner.py`: KV allocator cover/floor/index invariants;
  `test_runtime.py`: KV metadata emission and advisory fallback.
- All suites run without MLX or real weights, per the existing fixture policy.

## 8. Compatibility and migration

- No schema literal changes; all new fields are additive with fail-closed
  defaults. Older artifacts load unchanged; older AXQuant versions reading new
  artifacts fail on strict parsing only where they would have anyway (strict
  models reject unknown fields — this is the existing, intended behavior:
  artifacts flow forward, not backward).
- `Qwen36Adapter` behavior is regression-pinned: identical classification and
  profile output for the existing fixture set, plus the new tier field.
- No changes to release-gate semantics; the tier gate is a new refusal in
  front of existing gates, never a bypass.
