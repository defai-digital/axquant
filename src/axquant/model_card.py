from __future__ import annotations

import re
from pathlib import Path

from axquant.artifact_paths import artifact_member_path, artifact_tree_files
from axquant.errors import ArtifactError
from axquant.schema import (
    ArtifactFile,
    ArtifactManifest,
    ModelIdentity,
    ProtectedTensorSidecarManifest,
    QuantizationPlan,
    QuantizerExecutionManifest,
)
from axquant.serde import file_sha256, load_model, read_data, stable_sha256, write_data, write_text

_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_AXQ_NAME = re.compile(r"^(?P<stem>AX-.+-MLX-AXQ)-(?P<product_class>4bit|6bit|8bit)(?:-MTP)?$")
_PUBLIC_METADATA_FILES = (
    "axquant_manifest.json",
    "axquant_plan.json",
    "axquant_quantizer_execution.json",
    "axquant_mtp_sidecar_manifest.json",
    "axquant_vision_sidecar_manifest.json",
)
_REFRESHABLE_FILES = frozenset(
    {
        "README.md",
        "LICENSE",
        "model-manifest.json",
        *_PUBLIC_METADATA_FILES[1:],
    }
)


def _public_identity(identity: ModelIdentity) -> ModelIdentity:
    return identity.model_copy(update={"local_path": None})


def _public_calibration_reference(value: str) -> str:
    normalized = value.replace("\\", "/")
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        normalized = relative.name
    if not normalized or normalized in {".", ".."}:
        raise ArtifactError("calibration evidence has no safe public reference")
    return normalized


def _load_optional_sidecar(path: Path) -> ProtectedTensorSidecarManifest | None:
    if not path.is_file():
        return None
    return load_model(path, ProtectedTensorSidecarManifest)


def _assert_safe_record_path(directory: Path, value: str, label: str) -> None:
    try:
        artifact_member_path(directory, value)
    except ValueError as exc:
        raise ArtifactError(f"{label} contains an unsafe artifact path: {value}") from exc


def _assert_input_bindings(
    directory: Path,
    manifest: ArtifactManifest,
    plan: QuantizationPlan,
    execution: QuantizerExecutionManifest,
    mtp_sidecar: ProtectedTensorSidecarManifest | None,
    vision_sidecar: ProtectedTensorSidecarManifest | None,
) -> None:
    plan_sha256 = stable_sha256(plan)
    if not _REPO_ID.fullmatch(plan.source_model.model_id):
        raise ArtifactError("development model cards require a Hub owner/name source identity")
    if manifest.plan_sha256 != plan_sha256:
        raise ArtifactError("artifact manifest does not bind the input plan")
    if execution.plan_sha256 != plan_sha256:
        raise ArtifactError("quantizer execution does not bind the input plan")
    if manifest.source_model != plan.source_model:
        raise ArtifactError("artifact manifest and plan use different source identities")
    for label, sidecar, expected_role in (
        ("MTP sidecar manifest", mtp_sidecar, "mtp"),
        ("vision sidecar manifest", vision_sidecar, "vision"),
    ):
        if sidecar is None:
            continue
        if sidecar.role != expected_role:
            raise ArtifactError(f"{label} declares the wrong protected tensor role")
        if sidecar.source_model != plan.source_model:
            raise ArtifactError(f"{label} uses a different source identity")
        _assert_safe_record_path(directory, sidecar.output.path, label)
        for source in sidecar.source_files:
            _assert_safe_record_path(directory, source.path, label)
    for record in manifest.files:
        _assert_safe_record_path(directory, record.path, "artifact manifest")


def _format_decimal_size(size_bytes: int) -> str:
    return f"{size_bytes / 1_000_000_000:.2f} GB"


def _format_parameters(parameters: int) -> str:
    if parameters >= 1_000_000_000:
        return f"{parameters / 1_000_000_000:.2f}B"
    if parameters >= 1_000_000:
        return f"{parameters / 1_000_000:.2f}M"
    return f"{parameters:,}"


def _precision_rows(manifest: ArtifactManifest) -> str:
    order = {"2bit": 2, "3bit": 3, "4bit": 4, "6bit": 6, "8bit": 8, "bf16": 16}
    rows: list[str] = []
    for precision, share in sorted(
        manifest.weight_distribution.items(),
        key=lambda item: (order.get(item[0], 99), item[0]),
    ):
        rows.append(
            f"| `{precision}` | {_format_parameters(share.parameters)} | "
            f"{share.fraction * 100:.2f}% |"
        )
    return "\n".join(rows)


def _context_length(directory: Path) -> str:
    config_path = directory / "config.json"
    if not config_path.is_file():
        return "unrecorded"
    payload = read_data(config_path)
    if not isinstance(payload, dict):
        return "unrecorded"
    text_config = payload.get("text_config")
    source = text_config if isinstance(text_config, dict) else payload
    value = source.get("max_position_embeddings")
    return f"{value:,}" if isinstance(value, int) and value > 0 else "unrecorded"


def _package_size(manifest: ArtifactManifest) -> int:
    return sum(record.size_bytes for record in manifest.files)


def _source_link(identity: ModelIdentity) -> str:
    if identity.revision:
        return f"https://huggingface.co/{identity.model_id}/tree/{identity.revision}"
    return f"https://huggingface.co/{identity.model_id}"


def _render_sidecar_detail(
    sidecar: ProtectedTensorSidecarManifest | None,
    *,
    fallback_bytes: int,
) -> str:
    if sidecar is None:
        return "not included"
    return (
        f"{sidecar.tensor_count} tensors, {_format_parameters(sidecar.parameters)} parameters, "
        f"{_format_decimal_size(sidecar.output.size_bytes or fallback_bytes)}, "
        f"{', '.join(sidecar.dtypes)}"
    )


def _evidence_summary(
    manifest: ArtifactManifest,
    plan: QuantizationPlan,
    execution: QuantizerExecutionManifest,
) -> tuple[str, str, str]:
    successful = sum(record.success for record in execution.records)
    fallbacks = sum(record.fallback for record in execution.records)
    conversion = (
        f"{successful}/{len(execution.records)} recorded module conversions succeeded; "
        f"{fallbacks} fallbacks"
    )
    calibration = (
        "none; the allocation is based on architecture priors"
        if plan.calibration is None
        else f"recorded in `{plan.calibration.reference}`"
    )
    mtp = (
        f"{manifest.mtp_measured_speedup:.4f}x measured speedup"
        if manifest.mtp_measured_speedup is not None
        else "not measured; no MTP speedup claim"
    )
    return conversion, calibration, mtp


def render_development_model_card(
    *,
    directory: Path,
    repo_id: str,
    product_class: str,
    manifest: ArtifactManifest,
    plan: QuantizationPlan,
    execution: QuantizerExecutionManifest,
    mtp_sidecar: ProtectedTensorSidecarManifest | None,
    vision_sidecar: ProtectedTensorSidecarManifest | None,
) -> str:
    """Render an evidence-safe Hub card for a development AXQ checkpoint."""

    name = repo_id.rsplit("/", 1)[-1]
    match = _AXQ_NAME.fullmatch(name)
    if match is None or match.group("product_class") != product_class:
        raise ArtifactError(
            "AXQ repository name and product class must agree "
            f"(repo={name!r}, product_class={product_class!r})"
        )
    source = manifest.source_model
    if not _REPO_ID.fullmatch(source.model_id):
        raise ArtifactError("development model cards require a Hub owner/name source identity")
    if (
        source.model_id != plan.source_model.model_id
        or source.revision != plan.source_model.revision
    ):
        raise ArtifactError("artifact manifest and plan use different source identities")
    stem = match.group("stem")
    suffix = "-MTP" if name.endswith("-MTP") else ""
    four_bit_repo = f"{repo_id.split('/', 1)[0]}/{stem}-4bit{suffix}"
    six_bit_repo = f"{repo_id.split('/', 1)[0]}/{stem}-6bit{suffix}"
    density = "dense" if plan.architecture_profile.dense else "mixture of experts (MoE)"
    product_family = plan.architecture_profile.product_family or "unknown"
    source_arch = source.architecture or plan.architecture_profile.config_model_type or "unrecorded"
    has_mtp = bool(manifest.mtp_present) or mtp_sidecar is not None
    has_vision = vision_sidecar is not None or bool(plan.architecture_profile.vision_present)
    context_length = _context_length(directory)
    group_sizes = sorted(
        {assignment.group_size for assignment in plan.assignments if assignment.group_size}
    )
    methods = sorted({assignment.method.value for assignment in plan.assignments})
    conversion, calibration, mtp_evidence = _evidence_summary(manifest, plan, execution)
    mlx_lm_version = manifest.software_versions.mlx_lm or "unrecorded"
    mlx_version = manifest.software_versions.mlx or "unrecorded"
    ax_engine_version = manifest.software_versions.ax_engine or "not recorded"
    model_manifest_status = (
        "included as `model-manifest.json`"
        if (directory / "model-manifest.json").is_file()
        else "not included"
    )
    license_link = "[`LICENSE`](LICENSE) and " if (directory / "LICENSE").is_file() else ""
    source_url = _source_link(source)
    package_size = _format_decimal_size(_package_size(manifest))
    mtp_detail = _render_sidecar_detail(
        mtp_sidecar,
        fallback_bytes=manifest.mtp_weight_file_size_bytes,
    )
    vision_detail = _render_sidecar_detail(
        vision_sidecar,
        fallback_bytes=manifest.protected_weight_file_size_bytes,
    )
    main_parameters = _format_parameters(manifest.main_logical_parameters)
    sibling_rows = "\n".join(
        (
            f"| [4bit sibling](https://huggingface.co/{four_bit_repo}) | "
            "Lower-storage AXQ budget; check its exact BPW |",
            f"| [6bit sibling](https://huggingface.co/{six_bit_repo}) | "
            "Higher average precision near a 6-BPW budget |",
        )
    )
    catalog_url = "https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog"
    long_context_status = (
        f"{context_length}-token capacity is config metadata, not a validated claim"
    )
    precision_tag = product_class.replace("bit", "-bit")
    family_tag = product_family.replace(" ", "-").lower()
    is_embedding_pack = "embedding" in name.lower() or "embedding" in source.model_id.lower()
    pipeline_tag = "feature-extraction" if is_embedding_pack else "text-generation"
    optional_tags = [family_tag, product_class, precision_tag]
    if is_embedding_pack:
        optional_tags.extend(("embedding", "sentence-similarity"))
    if has_mtp:
        optional_tags.append("mtp")
    if has_vision:
        optional_tags.append("vision")
    tag_block = "\n".join(
        f"- {tag}"
        for tag in (
            "mlx",
            "apple-silicon",
            "quantized",
            "mixed-precision",
            "axquant",
            "axq",
            "development",
            *optional_tags,
        )
    )
    sidecar_blurb_parts: list[str] = []
    if has_mtp:
        sidecar_blurb_parts.append("multi-token-prediction (MTP) head")
    if has_vision:
        sidecar_blurb_parts.append("vision tower")
    if sidecar_blurb_parts:
        sidecar_blurb = (
            "The language path is quantized while the "
            + " and ".join(sidecar_blurb_parts)
            + " are preserved as BF16 sidecars when present."
        )
    else:
        sidecar_blurb = (
            "The language path is quantized under AXQuant protection floors "
            "(embeddings, norms, and other protected tensors remain higher precision)."
        )
    total_bpw_label = "Measured total BPW, including MTP" if has_mtp else "Measured total BPW"
    mtp_contract_suffix = " and native MTP sidecar" if has_mtp else ""
    ax_engine_section = f"""## Serve with AX Engine{" and MTP" if has_mtp else ""}

After installing [AX Engine](https://github.com/defai-digital/ax-engine), download the complete
repository and serve the local directory:

```bash
ax-engine serve ./{name} --port 31418
```

AX Engine is the authority for the AXQ runtime contract{mtp_contract_suffix}.
This development package does not claim runtime speedups until identical-checkpoint benchmarks are
published. The artifact records AX Engine version `{ax_engine_version}`. Native
`model-manifest.json` status: {model_manifest_status}.
"""
    vision_quality = (
        "Not evaluated or claimed; vision tensors are preserved at BF16"
        if has_vision
        else "Not applicable (no vision sidecar in this package)"
    )
    mtp_limitation = (
        "- MTP may be ignored outside AX Engine and its speedup is unmeasured for this exact "
        "checkpoint.\n"
        if has_mtp
        else ""
    )
    vision_limitation = (
        "- Vision weights are byte-preserved at BF16, but this release does not claim validated "
        "VLM quality.\n"
        if has_vision
        else ""
    )
    provenance_sidecars = []
    if has_mtp:
        provenance_sidecars.append(
            "- [`axquant_mtp_sidecar_manifest.json`]"
            "(axquant_mtp_sidecar_manifest.json): MTP tensor provenance."
        )
    if has_vision:
        provenance_sidecars.append(
            "- [`axquant_vision_sidecar_manifest.json`](axquant_vision_sidecar_manifest.json): "
            "protected vision tensor provenance."
        )
    if (directory / "model-manifest.json").is_file():
        provenance_sidecars.append(
            "- [`model-manifest.json`](model-manifest.json): AX Engine native tensor manifest."
        )
    provenance_sidecar_block = "\n".join(provenance_sidecars)
    if provenance_sidecar_block:
        provenance_sidecar_block = "\n" + provenance_sidecar_block
    context_limitation = (
        "- The configured context window can require substantially more memory as the KV cache "
        "grows."
    )
    runtime_provenance = (
        "- [`axquant_runtime.json`](axquant_runtime.json): AX Engine and MLX-LM compatibility "
        "contract."
    )

    return f"""---
license: apache-2.0
library_name: mlx
base_model: {source.model_id}
base_model_relation: quantized
pipeline_tag: {pipeline_tag}
tags:
{tag_block}
---

# {name}

An **AXQuant (AXQ)** mixed-precision MLX checkpoint for Apple Silicon, converted directly from
the BF16 source model. {sidecar_blurb}

> **Development evidence — not a certified AXQuant release.** This package has conversion and
> artifact-integrity records, but it does not publish measured quality, long-context, kernel-speed,
> or MTP-speed evidence. Do not interpret the AXQ product label as a benchmark claim.

## Model details

| Property | Value |
| --- | --- |
| Base model | [{source.model_id}]({source_url}) |
| Source revision | `{source.revision or "unrecorded"}` |
| Product family | `{product_family}` |
| Source architecture | `{source_arch}` ({density}); text path optimized |
| Main-model parameters | {main_parameters} logical parameters |
| Quantizer | AXQuant `{manifest.axquant_version}` |
| Hub budget class | `{product_class}` |
| AXQuant base precision class | `{manifest.target_class}` |
| Planned storage-adjusted BPW | {manifest.effective_bpw:.4f} |
| Measured main-model BPW | {manifest.measured_main_bpw:.4f} |
| {total_bpw_label} | **{manifest.measured_total_bpw:.4f}** |
| Safetensors weight size | {_format_decimal_size(manifest.weight_file_size_bytes)} |
| Approximate complete download | {package_size} |
| Configured maximum context | {context_length} tokens; practical limits depend on unified memory |
| Primary runtime | AX Engine, compatibility level A |
| Compatible runtime | MLX-LM standard text inference, compatibility level B |
| MTP present | `{has_mtp}` |
| Vision sidecar present | `{has_vision}` |

This repository contains MLX Safetensors. It does **not** contain PyTorch or GGUF weights.

## Choosing an AXQ pack

AXQ names describe a **storage-budget product class**, not one uniform precision applied to every
tensor. Protected tensors remain at higher precision, so the exact measured BPW is authoritative.
In particular, a `6bit`-named mixed plan may retain `4bit` as its base precision while selecting
6-bit, 8-bit, or BF16 for other tensors to meet an approximately 6-BPW total budget. Protection
floors can also raise a `4bit`-named pack close to (or above) a `6bit` budget on small or heavily
protected models.

| Sibling | Intended trade-off |
| --- | --- |
{sibling_rows}

See the [AutomatosX MLX model catalog]({catalog_url})
for related MLX and OptiQ alternatives.

## Download

```bash
python -m pip install -U huggingface_hub
hf download {repo_id} --local-dir ./{name}
```

Allow at least {package_size} of free disk space. Pin the resulting Hub commit in reproducible
deployments rather than relying indefinitely on `main`.

## Run with MLX-LM

```bash
python -m pip install -U mlx-lm
mlx_lm.generate \\
  --model {repo_id} \\
  --prompt "Explain mixed-precision quantization in three sentences." \\
  --max-tokens 128 \\
  --temp 0.0
```

MLX-LM compatibility covers standard **text/backbone inference**. It may ignore AXQuant runtime
metadata and optional sidecars (`vision.safetensors`, `mtp.safetensors`); this command therefore
does not establish MTP acceleration or vision-language quality. The artifact records MLX
`{mlx_version}` and MLX-LM `{mlx_lm_version}` from conversion.

{ax_engine_section}
## Quantization layout

| Main-weight precision | Parameters | Share |
| --- | ---: | ---: |
{_precision_rows(manifest)}

- Quantization methods: `{", ".join(methods)}`.
- Group sizes used by quantized assignments: `{", ".join(map(str, group_sizes)) or "none"}`.
- MTP sidecar: {mtp_detail}.
- Vision sidecar: {vision_detail}.
- Optimization scope: `{plan.architecture_profile.optimization_scope.value}`.
- Support tier: `{plan.architecture_profile.support_tier.value}`.

BF16 sidecars, when present, are included in total download size. Their presence does not by itself
establish MTP acceleration or vision-language quality.

## Evidence and validation status

| Check | Status |
| --- | --- |
| Planning evidence | `{plan.evidence_kind.value}` |
| Calibration | {calibration} |
| Quantizer execution | {conversion} |
| AX Engine native manifest | {model_manifest_status} |
| Quality versus BF16 or uniform baselines | Not published; no quality-retention claim |
| MTP acceptance and speed | {mtp_evidence} |
| AX Engine kernel evidence | `{manifest.runtime.ax_engine.kernel_evidence}` |
| Vision-language quality | {vision_quality} |
| Long-context quality | {long_context_status} |
| Release certification | **Not certified**; formal AXQuant M0-M8 gates are not closed |

## Intended use and limitations

- Intended for local development and evaluation on Apple Silicon with MLX-compatible runtimes.
- No minimum unified-memory figure is claimed; loadability depends on model size, context length,
  KV-cache policy, runtime buffers, and other processes using unified memory.
- Architecture-prior allocation is not measured sensitivity. It must not be presented as measured
  model quality.
{mtp_limitation}{vision_limitation}{context_limitation}
- Upstream capabilities, limitations, biases, and responsible-use guidance still apply.

## Provenance and audit files

- [`axquant_manifest.json`](axquant_manifest.json): package identity, byte accounting, runtime
  contract, software versions, and file checksums.
- [`axquant_plan.json`](axquant_plan.json): per-tensor precision decisions and planning evidence.
- [`axquant_quantizer_execution.json`](axquant_quantizer_execution.json): conversion coverage and
  fallback records.
{runtime_provenance}{provenance_sidecar_block}

All published provenance uses repository-relative paths. Local source paths are stripped before
publication. The checkpoint was converted from BF16 rather than re-quantized from an OptiQ
artifact. Parallel OptiQ repositories use a different quantizer and should not be assumed to have
identical BPW or quality.

## License

The checkpoint follows the upstream model license where applicable (often Apache License 2.0). See
{license_link}the [{source.model_id} model card]({source_url}) for license terms, model
limitations, and responsible-use guidance.
"""


def _refresh_manifest_records(directory: Path, manifest: ArtifactManifest) -> None:
    records = {record.path: record for record in manifest.files}
    for relative in sorted(_REFRESHABLE_FILES):
        path = directory / relative
        if not path.is_file():
            continue
        records[relative] = ArtifactFile(
            path=relative,
            size_bytes=path.stat().st_size,
            sha256=file_sha256(path),
        )
    manifest.files = [records[path] for path in sorted(records)]


def _assert_public_consistency(directory: Path) -> None:
    manifest = load_model(directory / "axquant_manifest.json", ArtifactManifest)
    plan = load_model(directory / "axquant_plan.json", QuantizationPlan)
    execution = load_model(
        directory / "axquant_quantizer_execution.json",
        QuantizerExecutionManifest,
    )
    expected_plan_sha256 = stable_sha256(plan)
    if manifest.plan_sha256 != expected_plan_sha256:
        raise ArtifactError("public artifact manifest does not bind the sanitized plan")
    if execution.plan_sha256 != expected_plan_sha256:
        raise ArtifactError("public quantizer execution does not bind the sanitized plan")
    identities = [manifest.source_model, plan.source_model]
    if manifest.source_model != plan.source_model:
        raise ArtifactError("public artifact manifest and plan use different source identities")
    for sidecar_name in (
        "axquant_mtp_sidecar_manifest.json",
        "axquant_vision_sidecar_manifest.json",
    ):
        sidecar = _load_optional_sidecar(directory / sidecar_name)
        if sidecar is not None:
            if sidecar.source_model != plan.source_model:
                raise ArtifactError(f"public {sidecar_name} uses a different source identity")
            identities.append(sidecar.source_model)
    if any(identity.local_path for identity in identities):
        raise ArtifactError("public AXQuant metadata contains a local source path")
    records = {record.path: record for record in manifest.files}
    for relative in _REFRESHABLE_FILES:
        path = directory / relative
        if not path.is_file():
            continue
        record = records.get(relative)
        if record is None:
            raise ArtifactError(f"public artifact manifest omits {relative}")
        if record.size_bytes != path.stat().st_size or record.sha256 != file_sha256(path):
            raise ArtifactError(f"public artifact manifest contains a stale record for {relative}")


def prepare_development_model_card(
    *,
    artifact_dir: str | Path,
    repo_id: str,
    product_class: str | None = None,
) -> list[Path]:
    """Sanitize public provenance and materialize a detailed development model card.

    The operation is suitable for a complete artifact directory or a metadata-only staging
    directory. Existing weight-file records are preserved without rehashing multi-gigabyte shards;
    every metadata file changed here is re-bound in ``axquant_manifest.json``.
    """

    directory_input = Path(artifact_dir).expanduser()
    if directory_input.is_symlink():
        raise ArtifactError(f"artifact directory must not be a symlink: {directory_input}")
    directory = directory_input.resolve()
    if not directory.is_dir():
        raise ArtifactError(f"artifact directory does not exist: {directory}")
    try:
        artifact_tree_files(directory)
    except ValueError as exc:
        raise ArtifactError(f"artifact directory is unsafe: {exc}") from exc
    if not _REPO_ID.fullmatch(repo_id):
        raise ArtifactError("Hub repository must use the owner/name form")
    name = repo_id.rsplit("/", 1)[-1]
    match = _AXQ_NAME.fullmatch(name)
    if match is None:
        raise ArtifactError("development model cards require an MLX-AXQ repository name")
    inferred_class = match.group("product_class")
    resolved_class = product_class or inferred_class
    if resolved_class != inferred_class:
        raise ArtifactError("explicit product class does not match the Hub repository name")

    manifest_path = directory / "axquant_manifest.json"
    plan_path = directory / "axquant_plan.json"
    execution_path = directory / "axquant_quantizer_execution.json"
    manifest = load_model(manifest_path, ArtifactManifest)
    plan = load_model(plan_path, QuantizationPlan)
    execution = load_model(execution_path, QuantizerExecutionManifest)

    mtp_path = directory / "axquant_mtp_sidecar_manifest.json"
    vision_path = directory / "axquant_vision_sidecar_manifest.json"
    mtp_sidecar = _load_optional_sidecar(mtp_path)
    vision_sidecar = _load_optional_sidecar(vision_path)
    _assert_input_bindings(
        directory,
        manifest,
        plan,
        execution,
        mtp_sidecar,
        vision_sidecar,
    )

    plan.source_model = _public_identity(plan.source_model)
    if plan.calibration is not None:
        plan.calibration.reference = _public_calibration_reference(plan.calibration.reference)
    write_data(plan_path, plan)
    plan_sha256 = stable_sha256(plan)
    execution.plan_sha256 = plan_sha256
    write_data(execution_path, execution)

    for path, sidecar in ((mtp_path, mtp_sidecar), (vision_path, vision_sidecar)):
        if sidecar is None:
            continue
        sidecar.source_model = _public_identity(sidecar.source_model)
        write_data(path, sidecar)

    manifest.source_model = _public_identity(manifest.source_model)
    manifest.plan_sha256 = plan_sha256
    _refresh_manifest_records(directory, manifest)
    readme = directory / "README.md"
    write_text(
        readme,
        render_development_model_card(
            directory=directory,
            repo_id=repo_id,
            product_class=resolved_class,
            manifest=manifest,
            plan=plan,
            execution=execution,
            mtp_sidecar=mtp_sidecar,
            vision_sidecar=vision_sidecar,
        ),
    )
    _refresh_manifest_records(directory, manifest)
    write_data(manifest_path, manifest)
    _assert_public_consistency(directory)
    return [
        path
        for path in (manifest_path, plan_path, execution_path, mtp_path, vision_path, readme)
        if path.is_file()
    ]
