from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from axquant.artifact_paths import artifact_member_path, artifact_tree_files
from axquant.errors import ArtifactError
from axquant.identity import same_model_identity
from axquant.public_cert_index import claim_from_public_row, public_row_for_repo
from axquant.schema import (
    ArtifactFile,
    ArtifactManifest,
    CheckpointCertificationClaim,
    ModelIdentity,
    ProtectedTensorSidecarManifest,
    QuantizationPlan,
    QuantizerExecutionManifest,
)
from axquant.serde import file_sha256, load_model, read_data, stable_sha256, write_data, write_text

_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_AXQ_NAME = re.compile(
    r"^(?P<stem>AX-.+-MLX-AXQ)-"
    r"(?P<product_class>2bit(?:-experimental)?|3bit(?:-experimental)?|4bit|6bit|8bit)"
    r"(?P<edition>-v[1-9][0-9]*)?(?:-MTP)?$"
)
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

# Product stems that do not publish a distinct AXQ-4bit Hub sibling. Protection floors
# collapse the 4.8 and 6.0 BPW budgets to the same (or near-identical) artifact, so only
# the higher budget class remains in the public catalog. Values are short, base-specific
# reasons rendered into the development model card.
_NO_DISTINCT_4BIT_SIBLING_REASONS: dict[str, str] = {
    "AX-Qwen3.5-9B-MLX-AXQ": (
        "MTP and vision stay BF16 under protection floors, so both the ~4.8 and ~6.0 BPW "
        "budgets rise to ~6.97 BPW with identical weight files (~8.4 GB). A separate 4bit "
        "name would not reduce download size or runtime memory."
    ),
    "AX-MiniCPM5-1B-MLX-AXQ": (
        "On this ~1B model, protected high-precision tensors dominate storage, so both the "
        "~4.8 and ~6.0 BPW budgets land at ~7.38 BPW with identical weight files (~1.0 GB). "
        "A separate 4bit name would not reduce download size or runtime memory."
    ),
    "AX-Ministral-3-8B-Instruct-2512-MLX-AXQ": (
        "Protection floors lift the low-memory plan to ~5.99 BPW, essentially the same as "
        "the ~6.00 BPW pack (~6.7 GB). There is no meaningful size ladder between product "
        "names, so only this 6bit pack is published."
    ),
}
_NO_DISTINCT_4BIT_SIBLING_STEMS = frozenset(_NO_DISTINCT_4BIT_SIBLING_REASONS)


def _resolve_artifact_edition(
    repo_edition: str | None,
    artifact_edition: int | None,
) -> int | None:
    if artifact_edition is not None and (type(artifact_edition) is not int or artifact_edition < 1):
        raise ArtifactError("artifact edition must be a positive integer")
    inferred = int(repo_edition.removeprefix("-v")) if repo_edition else None
    if artifact_edition is not None and inferred is not None and artifact_edition != inferred:
        raise ArtifactError("explicit artifact edition does not match the Hub repository name")
    return artifact_edition if artifact_edition is not None else inferred


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
    if not same_model_identity(manifest.source_model, plan.source_model):
        raise ArtifactError("artifact manifest and plan use different source identities")
    for label, sidecar, expected_role in (
        ("MTP sidecar manifest", mtp_sidecar, "mtp"),
        ("vision sidecar manifest", vision_sidecar, "vision"),
    ):
        if sidecar is None:
            continue
        if sidecar.role != expected_role:
            raise ArtifactError(f"{label} declares the wrong protected tensor role")
        if not same_model_identity(sidecar.source_model, plan.source_model):
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


_DEVELOPMENT_BANNER = (
    "> **Development evidence — not a certified AXQuant release.** This package has conversion "
    "and\n"
    "> artifact-integrity records, but it does not publish measured quality, long-context, "
    "kernel-speed,\n"
    "> or MTP-speed evidence. Do not interpret the AXQ product label as a benchmark claim."
)
_DEVELOPMENT_RELEASE_ROW = (
    "| Release certification | **Not certified**; formal AXQuant M0-M8 gates are not closed |"
)

_MTP_ACCELERATION_STATUS_TEXT = {
    "certified": "certified on the certification host; see the certificate for its exact scope",
    "certified-scoped": (
        "certified for the certificate's authorizing profiles only; outside that scope there is "
        "no speedup claim"
    ),
    "not-certified": "**not certified**; no MTP speedup claim for this checkpoint",
}


def _certification_blocks(
    certified: CheckpointCertificationClaim | None,
) -> tuple[str, str, str | None]:
    """Banner, release-certification row, and MTP row override for a card."""

    if certified is None:
        return _DEVELOPMENT_BANNER, _DEVELOPMENT_RELEASE_ROW, None

    day = certified.certified_at.date().isoformat()
    link = (
        f"[checkpoint Tier 1 certificate]({certified.certificate_url})"
        if certified.certificate_url
        else "checkpoint Tier 1 certificate"
    )
    banner = (
        f"> **Checkpoint Tier 1 certified** on `{certified.host_id}` ({day}) for this exact\n"
        f"> revision — measured size against a matched uniform baseline, quality retention, and\n"
        "> conversion integrity. Tier 1 is a checkpoint claim, **not** a speed claim: MTP\n"
        f"> acceleration is {_MTP_ACCELERATION_STATUS_TEXT[certified.mtp_acceleration_status]}.\n"
        f"> See the {link} for the bound evidence and thresholds."
    )
    release_row = (
        f"| Release certification | **Checkpoint Tier 1 certified** on `{certified.host_id}` "
        f"({day}), Hub commit `{certified.hub_commit[:12]}`; the formal AXQuant M0-M8 release "
        "campaign is a separate process and is not implied |"
    )
    mtp_row = _MTP_ACCELERATION_STATUS_TEXT[certified.mtp_acceleration_status]
    if certified.mtp_acceleration_note:
        mtp_row = f"{mtp_row} ({certified.mtp_acceleration_note})"
    return banner, release_row, mtp_row


def _certification_for_repo(
    *,
    repo_id: str,
    certification: CheckpointCertificationClaim | None,
) -> CheckpointCertificationClaim | None:
    if certification is None:
        return None
    if certification.hub_repo_id != repo_id:
        raise ArtifactError(
            "checkpoint certification names a different repository "
            f"(certificate={certification.hub_repo_id!r}, card={repo_id!r})"
        )
    return certification


def _verified_certification(
    *,
    directory: Path,
    repo_id: str,
    certification: CheckpointCertificationClaim | None,
    manifest_sha256: str | None = None,
) -> CheckpointCertificationClaim | None:
    """Return the claim only when it provably binds to the rendered artifact."""

    certification = _certification_for_repo(repo_id=repo_id, certification=certification)
    if certification is None:
        return None
    measured = manifest_sha256 or file_sha256(directory / "axquant_manifest.json")
    if measured != certification.candidate_manifest_sha256:
        raise ArtifactError(
            "checkpoint certification does not bind this artifact: manifest SHA-256 "
            f"{measured} does not match certified {certification.candidate_manifest_sha256}"
        )
    return certification


def _render_development_model_card(
    *,
    directory: Path,
    repo_id: str,
    product_class: str,
    manifest: ArtifactManifest,
    plan: QuantizationPlan,
    execution: QuantizerExecutionManifest,
    mtp_sidecar: ProtectedTensorSidecarManifest | None,
    vision_sidecar: ProtectedTensorSidecarManifest | None,
    artifact_edition: int | None = None,
    certification: CheckpointCertificationClaim | None = None,
) -> str:
    """Render a Hub card after the caller has established claim provenance.

    Without ``certification`` the card carries the development-evidence banner.
    With one that binds to this exact artifact it states checkpoint Tier 1 and
    the certificate's own MTP acceleration status — which is normally still
    "not certified", and is rendered as such. Tier 1 is a size, quality, and
    conversion-integrity claim; it is never an acceleration claim.
    """

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
    repo_edition = match.group("edition") or ""
    resolved_edition = _resolve_artifact_edition(
        match.group("edition"),
        artifact_edition,
    )
    suffix = "-MTP" if name.endswith("-MTP") else ""
    # Match Qwen3-Embedding* and Nemotron-3-Embed* product stems.
    name_l = name.lower()
    source_l = source.model_id.lower()
    is_embedding_pack = (
        "embedding" in name_l
        or "embedding" in source_l
        or "-embed-" in name_l
        or "-embed-" in source_l
        or name_l.endswith("-embed")
        or source_l.endswith("-embed")
    )
    # Sibling ladder for Hub cross-links: low-bit experimental packs point at 4bit.
    base_class = product_class.removesuffix("-experimental")
    if is_embedding_pack:
        higher_precision_class = "8bit"
        low_precision_class = "4bit"
    elif base_class in {"2bit", "3bit"}:
        higher_precision_class = "4bit"
        low_precision_class = "2bit"
    else:
        higher_precision_class = "6bit"
        low_precision_class = "4bit"
    owner = repo_id.split("/", 1)[0]
    four_bit_repo = f"{owner}/{stem}-{low_precision_class}{repo_edition}{suffix}"
    higher_precision_repo = f"{owner}/{stem}-{higher_precision_class}{repo_edition}{suffix}"
    density = "dense" if plan.architecture_profile.dense else "mixture of experts (MoE)"
    product_family = plan.architecture_profile.product_family or "unknown"
    source_arch = source.architecture or plan.architecture_profile.config_model_type or "unrecorded"
    has_mtp = bool(manifest.mtp_present) or mtp_sidecar is not None
    has_vision = vision_sidecar is not None or bool(plan.architecture_profile.vision_present)
    has_audio = bool(plan.architecture_profile.audio_present)
    is_asr = product_family == "qwen3-asr"
    is_vlm = product_family == "qwen3-vl"
    context_length = _context_length(directory)
    group_sizes = sorted(
        {assignment.group_size for assignment in plan.assignments if assignment.group_size}
    )
    methods = sorted({assignment.method.value for assignment in plan.assignments})
    conversion, calibration, mtp_evidence = _evidence_summary(manifest, plan, execution)
    certified = _certification_for_repo(
        repo_id=repo_id,
        certification=certification,
    )
    evidence_banner, release_certification_row, certified_mtp_evidence = _certification_blocks(
        certified
    )
    if certified_mtp_evidence is not None:
        mtp_evidence = certified_mtp_evidence
    mlx_lm_version = manifest.software_versions.mlx_lm or "unrecorded"
    mlx_version = manifest.software_versions.mlx or "unrecorded"
    ax_engine_version = manifest.software_versions.ax_engine or "not recorded"
    has_native_manifest = (directory / "model-manifest.json").is_file()
    model_manifest_status = (
        "included as `model-manifest.json`" if has_native_manifest else "not included"
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
    no_4bit_reason = _NO_DISTINCT_4BIT_SIBLING_REASONS.get(stem)
    if no_4bit_reason is not None:
        sibling_rows = (
            f"| *(no distinct 4bit pack)* | {no_4bit_reason} The public catalog publishes only "
            f"[`{stem}-{higher_precision_class}{repo_edition}{suffix}`]"
            f"(https://huggingface.co/{higher_precision_repo}). |"
        )
        why_no_4bit_section = f"""## Why there is no AXQ-4bit pack

{no_4bit_reason}

"""
    else:
        sibling_rows = "\n".join(
            (
                f"| [4bit sibling](https://huggingface.co/{four_bit_repo}) | "
                "Lower-storage AXQ budget; check its exact BPW |",
                f"| [{higher_precision_class} sibling]"
                f"(https://huggingface.co/{higher_precision_repo}) | "
                f"Higher average precision near the "
                f"{higher_precision_class.removesuffix('bit')}-BPW budget |",
            )
        )
        why_no_4bit_section = ""
    catalog_url = "https://huggingface.co/collections/AutomatosX/automatosx-mlx-model-catalog"
    long_context_status = (
        f"{context_length}-token capacity is config metadata, not a validated claim"
    )
    precision_tag = product_class.replace("bit", "-bit")
    family_tag = product_family.replace(" ", "-").lower()
    pipeline_tag = (
        "automatic-speech-recognition"
        if is_asr
        else "image-text-to-text"
        if is_vlm
        else "feature-extraction"
        if is_embedding_pack
        else "text-generation"
    )
    library_name = "mlx-audio" if is_asr else "mlx"
    optional_tags = [family_tag, product_class, precision_tag]
    if resolved_edition is not None:
        optional_tags.append(f"v{resolved_edition}")
    if is_embedding_pack:
        optional_tags.extend(("embedding", "sentence-similarity"))
    if has_mtp:
        optional_tags.append("mtp")
    if has_vision:
        optional_tags.append("vision")
    if has_audio:
        optional_tags.extend(("audio", "speech-to-text", "mlx-audio"))
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
    if has_audio:
        sidecar_blurb_parts.append("audio tower")
    if sidecar_blurb_parts:
        sidecar_blurb = (
            "The language path is quantized while the "
            + " and ".join(sidecar_blurb_parts)
            + " are preserved at BF16 in the checkpoint (or a bound sidecar when present)."
        )
    else:
        sidecar_blurb = (
            "The language path is quantized under AXQuant protection floors "
            "(embeddings, norms, and other protected tensors remain higher precision)."
        )
    total_bpw_label = "Measured total BPW, including MTP" if has_mtp else "Measured total BPW"
    edition_row = (
        f"| Artifact edition | `v{resolved_edition}` |\n" if resolved_edition is not None else ""
    )
    stable_name_notice = (
        "\n> **Stable-name v2.** `main` serves the audited v2 artifact for backward "
        "compatibility. The same revision is tagged `v2`; when this repository replaced an "
        "earlier artifact, that prior revision remains recoverable at `legacy-pre-v2`.\n"
        if resolved_edition == 2 and not repo_edition
        else ""
    )
    mtp_contract_suffix = " and native MTP sidecar" if has_mtp else ""
    if has_native_manifest:
        ax_engine_runtime_status = (
            "Native manifest included; execution still requires a runtime check"
        )
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
        ax_engine_limitation = ""
    else:
        ax_engine_runtime_status = "Not established; no validated native manifest is included"
        ax_engine_section = f"""## AX Engine status

This package does **not** include a validated native `model-manifest.json`, so AX Engine execution
is not established by this release. The AX Engine fields in `axquant_runtime.json` describe the
intended compatibility contract, not observed runtime evidence. Use the architecture-specific MLX
runtime path above. The artifact records AX Engine version
`{ax_engine_version}`, but version discovery alone is not a runtime check.
"""
        ax_engine_limitation = (
            "- AX Engine execution is not established because this package has no validated "
            "native manifest.\n"
        )
    vision_quality = (
        "Not evaluated or claimed; vision tensors are preserved at BF16"
        if has_vision
        else "Not applicable (no vision tower in this package)"
    )
    speech_quality = (
        "Not evaluated or claimed; audio tensors are preserved at BF16"
        if has_audio
        else "Not applicable"
    )
    mtp_limitation = (
        "- MTP may be ignored outside AX Engine and its speedup is unmeasured for this exact "
        "checkpoint.\n"
        if has_mtp
        else ""
    )
    vision_limitation = (
        "- Vision weights are preserved at BF16, but this release does not claim validated "
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
    audio_limitation = (
        "- Audio weights are preserved at BF16, but this release does not claim measured "
        "transcription quality versus the BF16 source.\n"
        if has_audio
        else ""
    )
    if vision_sidecar is not None:
        provenance_sidecars.append(
            "- [`axquant_vision_sidecar_manifest.json`](axquant_vision_sidecar_manifest.json): "
            "protected vision tensor provenance."
        )
    if has_native_manifest:
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
        "- [`axquant_runtime.json`](axquant_runtime.json): declared AX Engine and MLX "
        "compatibility metadata; runtime checks remain separate evidence."
    )
    if is_asr:
        runtime_label = "MLX-Audio"
        runtime_section = f"""## Run with MLX-Audio

```bash
python -m pip install -U mlx-audio
python -m mlx_audio.stt.generate \\
  --model {repo_id} \\
  --audio ./audio.wav \\
  --output-path ./transcript \\
  --format txt
```

The protected audio tower and AXQ language decoder are loaded together by MLX-Audio. The artifact
records MLX `{mlx_version}`; conversion used the Qwen3-ASR implementation supplied by MLX-Audio.
"""
    elif is_vlm:
        runtime_label = "MLX-VLM"
        runtime_section = f"""## Run with MLX-VLM

```bash
python -m pip install -U mlx-vlm
python -m mlx_vlm.generate \\
  --model {repo_id} \\
  --image ./image.png \\
  --prompt "Describe this image." \\
  --max-tokens 128 \\
  --temperature 0.0
```

The protected vision tower and AXQ language decoder are loaded together by MLX-VLM. The artifact
records MLX `{mlx_version}`; runtime QA is reported separately from model-quality claims.
"""
    else:
        runtime_label = "MLX-LM"
        runtime_section = f"""## Run with MLX-LM

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
"""
    modality_layout_rows = ""
    if has_vision:
        vision_layout = (
            "protected BF16 sidecar"
            if vision_sidecar is not None
            else "protected BF16 in main shards"
        )
        modality_layout_rows += f"- Vision weights: {vision_layout}.\n"
    if has_audio:
        modality_layout_rows += "- Audio weights: protected BF16 in main shards.\n"

    return f"""---
license: apache-2.0
library_name: {library_name}
base_model: {source.model_id}
base_model_relation: quantized
pipeline_tag: {pipeline_tag}
tags:
{tag_block}
---

# {name}

An **AXQuant (AXQ)** mixed-precision MLX checkpoint for Apple Silicon, converted directly from
the BF16 source model. {sidecar_blurb}

{evidence_banner}
{stable_name_notice}

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
{edition_row}| AXQuant base precision class | `{manifest.target_class}` |
| Planned storage-adjusted BPW | {manifest.effective_bpw:.4f} |
| Measured main-model BPW | {manifest.measured_main_bpw:.4f} |
| {total_bpw_label} | **{manifest.measured_total_bpw:.4f}** |
| Safetensors weight size | {_format_decimal_size(manifest.weight_file_size_bytes)} |
| Approximate complete download | {package_size} |
| Configured maximum context | {context_length} tokens; practical limits depend on unified memory |
| Primary MLX runtime | {runtime_label} |
| AX Engine native execution | {ax_engine_runtime_status} |
| MTP present | `{has_mtp}` |
| Vision present | `{has_vision}` |
| Audio present | `{has_audio}` |

This repository contains MLX Safetensors. It does **not** contain PyTorch or GGUF weights.

{why_no_4bit_section}## Choosing an AXQ pack

AXQ names describe a **storage-budget product class**, not one uniform precision applied to every
tensor. Protected tensors remain at higher precision, so the exact measured BPW is authoritative.
In particular, a `6bit`-named mixed plan may retain `4bit` as its base precision while selecting
6-bit, 8-bit, or BF16 for other tensors to meet an approximately 6-BPW total budget. Protection
floors can also raise a `4bit`-named pack close to (or above) a `6bit` budget on small or heavily
protected models. When that collapse happens, AutomatosX does **not** publish a separate
misleading `4bit` sibling for that base.
{
        "For this base, see [Why there is no AXQ-4bit pack](#why-there-is-no-axq-4bit-pack)."
        if no_4bit_reason is not None
        else ""
    }

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

{runtime_section}
{ax_engine_section}
## Quantization layout

| Main-weight precision | Parameters | Share |
| --- | ---: | ---: |
{_precision_rows(manifest)}

- Quantization methods: `{", ".join(methods)}`.
- Group sizes used by quantized assignments: `{", ".join(map(str, group_sizes)) or "none"}`.
- MTP sidecar: {mtp_detail}.
- Vision sidecar: {vision_detail}.
{modality_layout_rows}- Optimization scope: `{plan.architecture_profile.optimization_scope.value}`.
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
| Speech-recognition quality | {speech_quality} |
| Long-context quality | {long_context_status} |
{release_certification_row}

## Intended use and limitations

- Intended for local development and evaluation on Apple Silicon with MLX-compatible runtimes.
- No minimum unified-memory figure is claimed; loadability depends on model size, context length,
  KV-cache policy, runtime buffers, and other processes using unified memory.
- Architecture-prior allocation is not measured sensitivity. It must not be presented as measured
  model quality.
{mtp_limitation}{vision_limitation}{audio_limitation}{context_limitation}
{ax_engine_limitation}
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
artifact. If an OptiQ repository is published separately, it uses a different quantizer and
should not be assumed to have identical BPW or quality.

## License

The checkpoint follows the upstream model license where applicable (often Apache License 2.0). See
{license_link}the [{source.model_id} model card]({source_url}) for license terms, model
limitations, and responsible-use guidance.
"""


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
    artifact_edition: int | None = None,
    certification: CheckpointCertificationClaim | None = None,
) -> str:
    """Render an evidence-safe Hub card for an existing AXQ checkpoint.

    Direct rendering verifies a supplied certification against the manifest
    currently on disk. Preparation uses the private renderer to calculate the
    final README-bound manifest first, then verifies that prospective digest.
    """

    certified = _verified_certification(
        directory=directory,
        repo_id=repo_id,
        certification=certification,
    )
    return _render_development_model_card(
        directory=directory,
        repo_id=repo_id,
        product_class=product_class,
        manifest=manifest,
        plan=plan,
        execution=execution,
        mtp_sidecar=mtp_sidecar,
        vision_sidecar=vision_sidecar,
        artifact_edition=artifact_edition,
        certification=certified,
    )


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


def _set_rendered_text_record(
    manifest: ArtifactManifest,
    *,
    relative: str,
    rendered: str,
) -> None:
    payload = rendered.encode("utf-8")
    records = {record.path: record for record in manifest.files}
    records[relative] = ArtifactFile(
        path=relative,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest.files = [records[path] for path in sorted(records)]


def _serialized_manifest_sha256(manifest: ArtifactManifest) -> str:
    """Hash the exact JSON bytes ``write_data`` will persist for a manifest."""

    rendered = (
        json.dumps(
            manifest.model_dump(mode="json"),
            allow_nan=False,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


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
    if not same_model_identity(manifest.source_model, plan.source_model):
        raise ArtifactError("public artifact manifest and plan use different source identities")
    for sidecar_name in (
        "axquant_mtp_sidecar_manifest.json",
        "axquant_vision_sidecar_manifest.json",
    ):
        sidecar = _load_optional_sidecar(directory / sidecar_name)
        if sidecar is not None:
            if not same_model_identity(sidecar.source_model, plan.source_model):
                raise ArtifactError(f"public {sidecar_name} uses a different source identity")
            identities.append(sidecar.source_model)
    if any(identity.local_path for identity in identities):
        raise ArtifactError("public AXQuant metadata contains a local source path")
    if manifest.calibration != plan.calibration:
        raise ArtifactError(
            "public artifact manifest and plan record different calibration evidence"
        )
    if plan.calibration is not None:
        reference = Path(plan.calibration.reference.replace("\\", "/"))
        if reference.is_absolute() or ".." in reference.parts:
            raise ArtifactError("public calibration evidence references a local path")
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


def resolve_public_certification_claim(
    repo_id: str,
    *,
    certifications_dir: str | Path | None = None,
) -> CheckpointCertificationClaim | None:
    """Load a Hub claim from the public certificate SSOT, if one exists.

    Returns ``None`` when there is no public record for ``repo_id``, or when the
    record is not checkpoint Tier 1 certified. Callers that bind the claim to an
    on-disk artifact must still verify the candidate manifest digest.
    """

    cert_dir = Path(certifications_dir).expanduser() if certifications_dir is not None else None
    row = public_row_for_repo(repo_id, cert_dir=cert_dir, listed_only=False)
    if row is None:
        return None
    return claim_from_public_row(row)


def prepare_development_model_card(
    *,
    artifact_dir: str | Path,
    repo_id: str,
    product_class: str | None = None,
    artifact_edition: int | None = None,
    certification: CheckpointCertificationClaim | None = None,
    use_public_certification: bool = True,
    certifications_dir: str | Path | None = None,
) -> list[Path]:
    """Sanitize public provenance and materialize a detailed development model card.

    The operation is suitable for a complete artifact directory or a metadata-only staging
    directory. Existing weight-file records are preserved without rehashing multi-gigabyte shards;
    every metadata file changed here is re-bound in ``axquant_manifest.json``.

    When ``use_public_certification`` is true (default) and ``certification`` is omitted, the
    public certificate index is consulted for ``repo_id``. A Tier 1 certified record that binds
    this artifact's ``axquant_manifest.json`` digest upgrades the card; a certified record that
    does not bind fails closed. Uncertified or missing records keep the development banner.
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
    resolved_edition = _resolve_artifact_edition(
        match.group("edition"),
        artifact_edition,
    )

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
    # Mirror the sanitized calibration evidence: the converter records
    # manifest.calibration as a copy of plan.calibration, and release-audit M1
    # requires them to stay equal — leaving the manifest copy unsanitized both
    # leaks the local reference and breaks that invariant.
    manifest.calibration = (
        plan.calibration.model_copy(deep=True) if plan.calibration is not None else None
    )
    # Persist sanitized metadata before rendering so every prospective file record
    # below is based on the exact public inputs that will remain on disk.
    _refresh_manifest_records(directory, manifest)
    write_data(manifest_path, manifest)
    resolved_certification = certification
    if resolved_certification is None and use_public_certification:
        resolved_certification = resolve_public_certification_claim(
            repo_id,
            certifications_dir=certifications_dir,
        )
    rendered_readme = _render_development_model_card(
        directory=directory,
        repo_id=repo_id,
        product_class=resolved_class,
        manifest=manifest,
        plan=plan,
        execution=execution,
        mtp_sidecar=mtp_sidecar,
        vision_sidecar=vision_sidecar,
        artifact_edition=resolved_edition,
        certification=resolved_certification,
    )
    prospective_manifest = manifest.model_copy(deep=True)
    _refresh_manifest_records(directory, prospective_manifest)
    _set_rendered_text_record(
        prospective_manifest,
        relative="README.md",
        rendered=rendered_readme,
    )
    prospective_sha256 = _serialized_manifest_sha256(prospective_manifest)
    # A certification must bind the final README-aware manifest, not the stale
    # development-card manifest that happens to be on disk before this write.
    _verified_certification(
        directory=directory,
        repo_id=repo_id,
        certification=resolved_certification,
        manifest_sha256=prospective_sha256,
    )

    readme = directory / "README.md"
    write_text(readme, rendered_readme)
    write_data(manifest_path, prospective_manifest)
    if file_sha256(manifest_path) != prospective_sha256:
        raise ArtifactError("serialized public artifact manifest digest changed during write")
    _assert_public_consistency(directory)
    return [
        path
        for path in (manifest_path, plan_path, execution_path, mtp_path, vision_path, readme)
        if path.is_file()
    ]
