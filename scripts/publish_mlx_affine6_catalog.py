#!/usr/bin/env python3
"""Resume source-pinned affine6 factory jobs; publish only verified task smokes.

Input is an English JSON list of jobs with key, source_id, source_revision,
source_path, repo_id, output, task, license, and mtp fields. Optional artifact_path
adopts an existing pure-affine6 pack into a separate publication view. Publication
is explicit (--publish); existing repositories require this campaign's receipt.
Model payloads and state directories must live outside version-controlled files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from axquant.factory import FACTORY_MODELS, require_factory_host
from axquant.gemma4_assistant_compose import (
    Gemma4AssistantComposeRequest,
    compose_gemma4_assistant_mtp,
    refresh_gemma4_assistant_composite_manifest,
    validate_gemma4_assistant_composite,
)
from axquant.model_card import prepare_development_model_card
from axquant.model_support import require_model_supported
from axquant.runtime_compatibility import (
    require_runtime_compatibility_record,
    write_runtime_compatibility,
)
from axquant.schema import ArtifactFile, ArtifactManifest, RuntimeMetadata
from axquant.serde import file_sha256 as _file_sha256
from axquant.serde import load_model, stable_sha256, write_data

TASKS = {"embedding", "text", "vision", "ocr", "translation", "decision"}
BASE_COLLECTIONS = (
    "AutomatosX/6-bit-6adc1885130a725e8595678b",
    "AutomatosX/mlx-6ab432f9724c4d8a1037df1c",
)
_DIGEST_CACHE: dict[tuple[int, ...], str] = {}


def file_sha256(path: Path) -> str:
    """Reuse a measured digest only while the inode's change metadata is identical."""
    before = path.stat()
    key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    if key not in _DIGEST_CACHE:
        digest = _file_sha256(path)
        after = path.stat()
        if key != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"Artifact changed during hashing: {path.name}")
        _DIGEST_CACHE[key] = digest
    return _DIGEST_CACHE[key]


def validate_job(job: dict[str, Any]) -> None:
    """Reject ambiguous identities, task substitutions and unsafe publication names."""
    require_model_supported(
        job.get("source_id"),
        job.get("repo_id"),
        job.get("artifact_path"),
        job.get("source_path"),
        job.get("output"),
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", job["key"]):
        raise ValueError("Unsafe job key")
    if not re.fullmatch(r"[a-f0-9]{40}", job["source_revision"]):
        raise ValueError("Source requires an immutable revision")
    if job["task"] not in TASKS:
        raise ValueError("Unsupported task: implement a task-specific smoke first")
    suffix = "-MTP" if job["mtp"] else ""
    if not re.fullmatch(r"AutomatosX/AX-[A-Za-z0-9_.-]+-MLX-AXQ-6bit" + suffix, job["repo_id"]):
        raise ValueError("Repository name must agree with physical format and MTP")
    if job["repo_id"].endswith("-MTP") != bool(job["mtp"]):
        raise ValueError("MTP name must match packaged payload")
    if not job["license"] or not job.get("source_id"):
        raise ValueError("Source and license metadata are required")
    if job.get("runtime", "mlx-vlm") not in {"mlx-vlm", "ax-engine-stream"}:
        raise ValueError("Unknown runtime smoke profile")
    if job.get("assistant_path") and (
        not job["mtp"]
        or job["task"] != "vision"
        or not re.fullmatch(r"[^/@]+/[^/@]+@[a-f0-9]{40}", job.get("assistant_source_id", ""))
    ):
        raise ValueError("External assistant requires a pinned source and an MTP vision job")
    output = Path(job["output"]).resolve()
    source = Path(job["source_path"]).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Source and output must be disjoint")


def safe_member(directory: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or not name
        or any(char in name for char in "*?[]")
    ):
        raise ValueError(f"Unsafe artifact member: {name}")
    path = directory / name
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != directory.parent):
        raise ValueError(f"Symlink artifact member: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Missing artifact member: {name}")
    if (
        any(part.startswith(".") for part in relative.parts)
        or relative.name == "runtime_check.json"
    ):
        raise ValueError(f"Private artifact member: {name}")
    return path


def validate_pack(
    job: dict[str, Any], directory: Path, *, historical: bool = False
) -> ArtifactManifest:
    manifest = load_model(directory / "axquant_manifest.json", ArtifactManifest)
    require_model_supported(
        job.get("source_id"), job.get("repo_id"), manifest.source_model.model_id
    )
    source = manifest.source_model
    if (source.model_id, source.revision) != (job["source_id"], job["source_revision"]):
        raise ValueError("Artifact/source identity mismatch")
    distribution = manifest.weight_distribution
    if "6bit" not in distribution or distribution["6bit"].parameters <= 0:
        raise ValueError("No actual six-bit allocation")
    if set(distribution) - {"6bit", "8bit", "bf16"}:
        raise ValueError("Legacy low-bit allocations cannot be relabeled affine6")
    config = json.loads((directory / "config.json").read_text())
    require_model_supported(source.model_id, config=config)
    quantization = config.get("quantization", {})
    declarations = [quantization, *[v for v in quantization.values() if isinstance(v, dict)]]
    if not any(item.get("bits") == 6 for item in declarations):
        raise ValueError("No physical six-bit quantization declaration")
    for item in declarations:
        if item.get("bits") is not None and (
            item["bits"] not in {6, 8, 16} or item.get("mode", "affine") != "affine"
        ):
            raise ValueError("Non-affine6 physical packing")
    external = bool(job.get("assistant_path"))
    if external:
        if not job["mtp"] or manifest.mtp_present:
            raise ValueError("External assistant must be distinguished from a native head")
        composite = validate_gemma4_assistant_composite(directory)
        if composite.get("assistant_source_id") != job["assistant_source_id"]:
            raise ValueError("Assistant source identity mismatch")
        assistant_weight_accounting(directory / "assistant")
    elif manifest.mtp_present != job["mtp"]:
        raise ValueError("MTP payload/name mismatch")
    if job["mtp"] and not external:
        sidecar = directory / "axquant_mtp_sidecar_manifest.json"
        if manifest.mtp_weight_file_size_bytes <= 0 or not sidecar.is_file():
            raise ValueError("MTP requires an actual bound payload")
    if not historical:
        require_runtime_compatibility_record(directory)
    return manifest


def assistant_weight_accounting(directory: Path) -> tuple[int, int]:
    parameters = 0
    storage = 0
    for path in directory.rglob("*.safetensors"):
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
        storage += path.stat().st_size
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if tensor["dtype"] not in {"BF16", "F16", "F32"}:
                raise ValueError("External MTP assistant must preserve floating protection")
            count = 1
            for dimension in tensor["shape"]:
                count *= dimension
            parameters += count
    if not parameters:
        raise ValueError("External assistant contains no floating weights")
    return parameters, storage


def payload_digest(directory: Path, manifest: ArtifactManifest) -> str:
    """Bind runtime inputs; card prose and evidence metadata do not alter inference."""
    members = {entry.path for entry in manifest.files}
    if (directory / "ax_composite_pack_manifest.json").is_file():
        composite = validate_gemma4_assistant_composite(directory)
        members.update(composite["assistant_weight_digests"])
        members.add("ax_gemma4_assistant_mtp.json")
    return stable_sha256(
        [
            {"path": name, "sha256": file_sha256(safe_member(directory, name))}
            for name in sorted(members)
            if not name.startswith("axquant_") and not name.endswith("README.md")
        ]
    )


def require_smoke(job: dict[str, Any], smoke: dict[str, Any], digest: str) -> None:
    if smoke.get("passed") is not True or smoke.get("task") != job["task"]:
        raise ValueError("A passing task-specific smoke is required")
    if smoke.get("payload_sha256") != digest:
        raise ValueError("Smoke is stale: runtime payload changed")


def verify_files(directory: Path, manifest: ArtifactManifest) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for entry in manifest.files:
        if entry.path in records:
            raise ValueError("Duplicate artifact member")
        path = safe_member(directory, entry.path)
        if path.stat().st_size != entry.size_bytes or file_sha256(path) != entry.sha256:
            raise ValueError(f"Artifact drift: {entry.path}")
        records[entry.path] = entry.model_dump()
    path = directory / "axquant_manifest.json"
    records[path.name] = {"size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if (directory / "ax_composite_pack_manifest.json").is_file():
        composite = validate_gemma4_assistant_composite(directory)
        names = set(composite["assistant_weight_digests"]) | {
            "ax_gemma4_assistant_mtp.json",
            "ax_composite_pack_manifest.json",
        }
        for name in names:
            member = safe_member(directory, name)
            records[name] = {"size_bytes": member.stat().st_size, "sha256": file_sha256(member)}
    return records


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(
    command: list[str], log: Path, timeout: int, *, overrides: dict[str, str] | None = None
) -> None:
    environment = {**os.environ, **(overrides or {})}
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        stream.write(
            f"PID {process.pid}; timeout_seconds={timeout}; HF_XET_HIGH_PERFORMANCE="
            f"{environment.get('HF_XET_HIGH_PERFORMANCE', 'unset')}; "
            f"AXQUANT_STREAMING_CONVERT={environment.get('AXQUANT_STREAMING_CONVERT', 'auto')}\n"
        )
        stream.flush()
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if result:
            raise subprocess.CalledProcessError(result, command)


def convert(job: dict[str, Any], state: Path) -> Path:
    directory = Path(job["output"])
    if (directory / "axquant_manifest.json").is_file():
        validate_pack(job, directory)
        return directory
    if directory.exists():
        raise ValueError("Existing incomplete output requires manual review")
    if job.get("assistant_path"):
        base = {
            key: value
            for key, value in job.items()
            if key not in {"assistant_path", "assistant_source_id"}
        }
        base.update(
            mtp=False,
            repo_id=job["repo_id"].removesuffix("-MTP"),
            output=str(directory.with_name(directory.name + "-target")),
        )
        target = convert(base, state)
        assistant = Path(job["assistant_path"])
        config = json.loads((assistant / "config.json").read_text())
        if config.get("quantization") or config.get("quantization_config"):
            raise ValueError("External assistant requires a floating protected source")
        compose_gemma4_assistant_mtp(
            Gemma4AssistantComposeRequest(
                target_dir=target,
                assistant_dir=assistant,
                output_dir=directory,
                target_model_id=job["source_id"],
                assistant_model_id=job["source_id"] + "-assistant",
                assistant_source_id=job["assistant_source_id"],
            )
        )
        # Runtime metadata was intentionally changed by composition.
        manifest = load_model(directory / "axquant_manifest.json", ArtifactManifest)
        manifest.runtime = load_model(directory / "axquant_runtime.json", RuntimeMetadata)
        for name in ("axquant_runtime.json",):
            path = directory / name
            manifest.files = [entry for entry in manifest.files if entry.path != name]
            manifest.files.append(
                ArtifactFile(path=name, size_bytes=path.stat().st_size, sha256=file_sha256(path))
            )
        write_data(directory / "axquant_manifest.json", manifest)
        refresh_gemma4_assistant_composite_manifest(directory)
    elif job.get("artifact_path"):
        original = Path(job["artifact_path"])
        manifest = validate_pack(job, original, historical=True)
        verify_files(original, manifest)
        directory.mkdir(parents=True)
        for entry in [
            *manifest.files,
            ArtifactFile(path="axquant_manifest.json", size_bytes=0, sha256="0" * 64),
        ]:
            source = safe_member(original, entry.path)
            target = directory / entry.path
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".safetensors":
                os.link(source, target)
            else:
                shutil.copy2(source, target)
        write_runtime_compatibility(directory)
        compatibility = directory / "axquant_compatibility.json"
        manifest.files = [entry for entry in manifest.files if entry.path != compatibility.name]
        manifest.files.append(
            ArtifactFile(
                path=compatibility.name,
                size_bytes=compatibility.stat().st_size,
                sha256=file_sha256(compatibility),
            )
        )
        write_data(directory / "axquant_manifest.json", manifest)
    else:
        command = [
            sys.executable,
            "-m",
            "axquant",
            "quantize",
            job["source_path"],
            "--model-id",
            job["source_id"],
            "--revision",
            job["source_revision"],
            "--format",
            "6bit",
            "--profile",
            "general",
            "--runtime-smoke",
            "none",
            "--ax-engine-manifest",
            "skip",
            "--output",
            str(directory),
            "--json",
            str(state / "conversion.json"),
        ]
        if job.get("source_quantized"):
            command.append("--allow-quantized")
        log = state / "conversion.log"
        overrides = {"AXQUANT_STREAMING_CONVERT": "1"} if job["task"] == "decision" else {}
        try:
            run_logged(command, log, job.get("convert_timeout", 7200), overrides=overrides)
        except subprocess.CalledProcessError:
            can_retry = (
                not job.get("source_quantized")
                and os.environ.get("AXQUANT_FORCE_CPU") != "1"
                and not (directory / "axquant_manifest.json").exists()
                and log.is_file()
                and "GPU Timeout Error" in log.read_text()
            )
            if not can_retry:
                raise
            log.rename(state / "conversion-metal-timeout.log")
            write_data(
                state / "conversion-fallback.json",
                {
                    "reason": "Metal command buffer timeout",
                    "retry_device": "cpu",
                    "source_revision": job["source_revision"],
                    "precision_plan_unchanged": True,
                },
            )
            run_logged(
                command,
                log,
                job.get("convert_timeout", 7200),
                overrides={**overrides, "AXQUANT_FORCE_CPU": "1"},
            )
    validate_pack(job, directory)
    return directory


def engine_cli_version() -> str:
    response = subprocess.run(
        ["ax-engine-bench", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    match = re.search(r"AX Engine v([A-Za-z0-9.+-]+)", response.stdout + response.stderr)
    return match.group(1) if match else "unknown build"


def atem_user_text(output: str) -> str | None:
    """Read the source protocol's user channel; never accept reasoning text."""
    marker = "<|start|>assistant to=user<|message|>"
    if marker in output:
        answer = output.rsplit(marker, 1)[1]
    elif output.lstrip().startswith("to=user<|message|>"):
        answer = output.lstrip().removeprefix("to=user<|message|>")
    else:
        return None
    return answer.split("<|eot|>", 1)[0].split("<|eom|>", 1)[0].strip()


def qwen_user_text(output: str, formatted_prompt: str) -> str | None:
    """Honor Qwen's reasoning boundary, including an opening tag in the prompt."""
    if "</think>" in output:
        return output.rsplit("</think>", 1)[1].strip()
    if "<think>" in output or formatted_prompt.rfind("<think>") > formatted_prompt.rfind(
        "</think>"
    ):
        return None
    return output.strip()


def smoke_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Public runtime APIs, deliberately isolated from the campaign parent."""
    require_model_supported(
        job.get("source_id"),
        job.get("repo_id"),
        job.get("artifact_path"),
        job.get("source_path"),
        job.get("output"),
    )
    import mlx.core as mx

    directory = Path(job["output"])
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    runtime = "mlx-vlm"
    config_type = json.loads((directory / "config.json").read_text()).get("model_type")
    if job.get("runtime") == "ax-engine-stream":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(directory)
        runtime_environment = {"AX_STREAM_EXPERTS": "on"}
        if os.environ.get("AX_MLX_PIPELINE_GRANULARITY"):
            runtime_environment["AX_MLX_PIPELINE_GRANULARITY"] = os.environ[
                "AX_MLX_PIPELINE_GRANULARITY"
            ]
        for prompt, expected in (
            ("What is 19 plus 23? Reply with only the number.", "42"),
            ("Reply with exactly the word READY.", "READY"),
        ):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_tokens = tokenizer.encode(formatted, add_special_tokens=False)
            result = subprocess.run(
                [
                    "ax-engine-bench",
                    "generate",
                    "--model-id",
                    job["repo_id"],
                    "--tokens",
                    ",".join(str(token) for token in input_tokens),
                    "--max-output-tokens",
                    str(max(1, len(tokenizer.encode(expected, add_special_tokens=False)))),
                    "--mlx",
                    "--mlx-model-artifacts-dir",
                    str(directory),
                    "--json",
                ],
                env={**os.environ, **runtime_environment},
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"AX Engine generation failed: {detail[:4000]}")
            response = json.loads(result.stdout)
            if response.get("runtime", {}).get("selected_backend") != "mlx":
                raise RuntimeError("AX Engine smoke did not select the requested MLX backend")
            output = response.get("output_text") or tokenizer.decode(
                response.get("output_tokens", []), skip_special_tokens=True
            )
            rows.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "output": output,
                    "output_tokens": response.get("output_tokens", []),
                    "passed": output.strip() == expected,
                    "route": response.get("route"),
                    "selected_backend": response.get("runtime", {}).get("selected_backend"),
                    "bound_weight_bytes": response.get("runtime", {})
                    .get("mlx_model", {})
                    .get("buffer_bytes"),
                }
            )
        return {
            "task": job["task"],
            "runtime": "ax-engine",
            "runtime_environment": runtime_environment,
            "device": "Metal / SSD layer-stack",
            "version": engine_cli_version(),
            "mlx_version": "native binding unverified; host Python package "
            + importlib.metadata.version("mlx"),
            "cases": rows,
            "passed": all(row["passed"] for row in rows),
            "elapsed_seconds": time.monotonic() - started,
            "mtp_execution_verified": False,
            "quality_certified": False,
        }
    if job["task"] == "translation" or (
        job["task"] == "text" and config_type in {"muse_glimmer", "qwen3_next"}
    ):
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        if config_type == "hy_v3":
            sys.path.insert(0, str(directory))
            import hy_v3_mlx

            hy_v3_mlx.register_hy_v3_model()
        model, tokenizer = load(str(directory))
        runtime = "mlx-lm"
        cases = [
            ("What is 19 plus 23? Reply with only the number.", "42"),
            ("Reply with exactly the word READY.", "READY"),
        ]
        if job["task"] == "translation":
            cases = [
                (
                    "Translate the following segment into English, without additional "
                    "explanation.\n\nBonjour le monde.",
                    "Hello world",
                )
            ]
        for prompt, expected in cases:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                **({"reasoning_strength": "low"} if config_type == "muse_glimmer" else {}),
            )
            output = generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=128,
                sampler=make_sampler(temp=0),
                verbose=False,
            )
            normalized = re.sub(r"[^a-z ]", "", output.lower())
            answer = atem_user_text(output) if config_type == "muse_glimmer" else output.strip()
            passed = (
                expected.lower() in normalized
                if job["task"] == "translation"
                else answer == expected
            )
            rows.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "output": output,
                    "final_answer": answer,
                    "passed": passed,
                }
            )
    elif job["task"] == "decision":
        sys.path.insert(0, str(directory))
        import clef_mlx

        model = clef_mlx.load(directory)
        runtime = "mlx-vlm" if model.processor is not None else "mlx-lm"
        response = model.systemone(
            {
                "model": "clef",
                "state": "Our checkout started returning errors and orders are blocked.",
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "Which team should help?",
                        "criteria": {"billing": "Payments", "technical": "Outages"},
                    },
                    "urgency": {"type": "score", "criteria": ["Can wait", "Today"]},
                    "outage": {"type": "noul", "instructions": "Is a service down?"},
                },
            }
        )
        answers = response["answers"]
        probabilities = list(answers["department"]["probabilities"].values())
        passed = (
            all(0 <= value <= 1 for value in probabilities)
            and abs(sum(probabilities) - 1) < 0.02
            and 0 <= answers["outage"]["noul"] <= 1
            and "urgency" in answers
        )
        rows.append({"answers": answers, "passed": passed})
    elif job["task"] == "embedding":
        from mlx_vlm.embedding_loader import load_embedding_model
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(directory)
        texts = [
            "task: search result | query: Which planet is known as the Red Planet?",
            "title: none | text: Mars is known as the Red Planet.",
            "title: none | text: Venus is often called Earth's twin.",
        ]
        arrays = {
            key: mx.array(value)
            for key, value in tokenizer(texts, padding=True, return_tensors="np").items()
        }

        def embed(path: Path) -> Any:
            model = load_embedding_model(path)
            vectors = model(**arrays).text_embeds.astype(mx.float32)
            mx.eval(vectors)
            return vectors

        quant = embed(directory)
        reference = embed(Path(job["source_path"]))
        cosine = mx.sum(quant * reference, axis=-1) / (
            mx.linalg.norm(quant, axis=-1) * mx.linalg.norm(reference, axis=-1)
        )
        finite = bool(mx.all(mx.isfinite(quant)).item())
        unit = bool(mx.all(mx.abs(mx.linalg.norm(quant, axis=-1) - 1) < 1e-2).item())
        rank = bool(((quant[0] @ quant[1]) > (quant[0] @ quant[2])).item())
        min_cosine = float(mx.min(cosine).item())
        rows.append(
            {
                "shape": list(quant.shape),
                "finite": finite,
                "unit_norm": unit,
                "mars_ranks_above_venus": rank,
                "min_cosine_vs_source": min_cosine,
                "passed": finite and unit and rank and min_cosine >= 0.90,
            }
        )
    else:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor = load(str(directory))
        config = json.loads((directory / "config.json").read_text())
        cases = [
            ("What is 19 plus 23? Reply with only the number.", "42"),
            ("Reply with exactly the word READY.", "READY"),
        ]
        if job["task"] == "translation":
            cases = [
                (
                    "Translate the following segment into English, without additional "
                    "explanation.\n\nBonjour le monde.",
                    "Hello",
                )
            ]
        image = None
        if job["task"] in {"vision", "ocr"}:
            from PIL import Image, ImageDraw, ImageFont

            image = Image.new("RGB", (640, 240), "white")
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 80)
            ImageDraw.Draw(image).text((40, 60), "AXQ 7319", fill="black", font=font)
            cases = [("Read all the text in the image. Output only the text.", "AXQ 7319")]
            if job["task"] == "ocr":
                # apply_chat_template inserts the single image token itself.
                cases = [("<|grounding|>Convert the document to markdown.", "7319")]
        for prompt, expected in cases:
            formatted = apply_chat_template(
                processor, config, prompt, num_images=int(image is not None), enable_thinking=False
            )
            options = {"image": image} if image is not None else {}
            result = generate(
                model,
                processor,
                formatted,
                max_tokens=128,
                temperature=0.0,
                verbose=False,
                **options,
            )
            output = result.text if hasattr(result, "text") else str(result)
            answer = (
                qwen_user_text(output, formatted)
                if job["source_id"].startswith("Qwen/")
                else output.strip()
            )
            passed = (
                answer is not None and expected.lower() in answer.lower()
                if job["task"] in {"translation", "ocr"}
                else answer == expected
            )
            rows.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "output": output,
                    "final_answer": answer,
                    "passed": passed,
                }
            )
    return {
        "task": job["task"],
        "runtime": runtime,
        "device": str(mx.default_device()),
        "version": importlib.metadata.version(runtime),
        "mlx_version": importlib.metadata.version("mlx"),
        "cases": rows,
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "elapsed_seconds": time.monotonic() - started,
        "mtp_execution_verified": False,
        "quality_certified": False,
    }


def attach_decision_head(job: dict[str, Any], directory: Path) -> None:
    """Retain exact source head bytes and account for this protected BF16 payload."""
    origin = Path(job.get("source_origin", job["source_path"]))
    head = origin / "joint_head.safetensors"
    if not head.is_file():
        raise ValueError("Decision source has no joint head")
    manifest = load_model(directory / "axquant_manifest.json", ArtifactManifest)
    records = {entry.path: entry for entry in manifest.files}
    if "joint_head.safetensors" in records and (
        records["joint_head.safetensors"].sha256 != file_sha256(head)
    ):
        raise ValueError("Protected source head changed during resume")
    if "joint_head.safetensors" not in records:
        with head.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
        parameters = 0
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if tensor["dtype"] != "BF16":
                raise ValueError("Decision head accounting requires a BF16 source")
            count = 1
            for dimension in tensor["shape"]:
                count *= dimension
            parameters += count
        size = head.stat().st_size
        total_parameters = manifest.logical_parameters + parameters
        main_parameters = manifest.main_logical_parameters + parameters
        total_bytes = manifest.weight_file_size_bytes + size
        main_bytes = manifest.main_weight_file_size_bytes + size
        counts = {key: value.parameters for key, value in manifest.weight_distribution.items()}
        counts["bf16"] = counts.get("bf16", 0) + parameters
        from axquant.schema import PrecisionShare

        changes = {
            "logical_parameters": total_parameters,
            "main_logical_parameters": main_parameters,
            "weight_file_size_bytes": total_bytes,
            "main_weight_file_size_bytes": main_bytes,
            "protected_weight_file_size_bytes": manifest.protected_weight_file_size_bytes + size,
            "weight_distribution": {
                key: PrecisionShare(parameters=value, fraction=value / total_parameters)
                for key, value in counts.items()
            },
            "measured_total_bpw": 8 * total_bytes / total_parameters,
            "measured_main_bpw": 8 * main_bytes / main_parameters,
            "effective_bpw": sum(
                counts.get(key, 0) * bits for key, bits in (("6bit", 6), ("8bit", 8), ("bf16", 16))
            )
            / total_parameters,
        }
        # StrictModel validates accounting on assignment. Replace the related
        # fields together so no intermediate inconsistent manifest is created.
        manifest = ArtifactManifest.model_validate(manifest.model_dump() | changes)
    for name in (
        "joint_head.safetensors",
        "joint_head_config.json",
        "decision_config.json",
        "joint_schema_model.py",
        "lora_merge.json",
    ):
        source = origin / name
        if not source.is_file() and name == "lora_merge.json":
            source = Path(job["source_path"]) / name
        if source.is_file():
            target = directory / name
            shutil.copy2(source, target)
            records[name] = ArtifactFile(
                path=name, size_bytes=target.stat().st_size, sha256=file_sha256(target)
            )
    loader = directory / "clef_mlx.py"
    shutil.copy2(Path(__file__).with_name("clef_mlx.py"), loader)
    records[loader.name] = ArtifactFile(
        path=loader.name, size_bytes=loader.stat().st_size, sha256=file_sha256(loader)
    )
    provenance = directory / "axquant_protected_head_provenance.json"
    write_data(
        provenance,
        {
            "source_id": job["source_id"],
            "source_revision": job["source_revision"],
            "head": records["joint_head.safetensors"].model_dump(),
            "transform": "byte-identical-protected-bf16",
            "quality_certified": False,
        },
    )
    records[provenance.name] = ArtifactFile(
        path=provenance.name,
        size_bytes=provenance.stat().st_size,
        sha256=file_sha256(provenance),
    )
    manifest.files = [records[name] for name in sorted(records)]
    write_data(directory / "axquant_manifest.json", manifest)


def prepare(job: dict[str, Any], directory: Path) -> None:
    if job["task"] == "decision":
        attach_decision_head(job, directory)
    manifest = validate_pack(job, directory)
    # Some upstream VLM converters copy Hub download-cache metadata. It is not
    # a runtime input, and must never enter a public upload allowlist.
    records = {
        entry.path: entry
        for entry in manifest.files
        if PurePosixPath(entry.path).parts[0] != ".cache"
        and PurePosixPath(entry.path).name != "runtime_check.json"
    }
    if json.loads((directory / "config.json").read_text()).get("model_type") == "hy_v3":
        project = Path(__file__).resolve().parents[1]
        for source, name in (
            (project / "src/axquant/mlx_models/hy_v3.py", "hy_v3_mlx.py"),
            (project / "LICENSE", "AXQuant-Adapter-LICENSE"),
        ):
            target = directory / name
            shutil.copy2(source, target)
            records[name] = ArtifactFile(
                path=name, size_bytes=target.stat().st_size, sha256=file_sha256(target)
            )
    if job.get("runtime") == "ax-engine-stream":
        native = directory / "model-manifest.json"
        if not native.is_file():
            subprocess.run(
                ["ax-engine-bench", "generate-manifest", str(directory), "--json"],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        records[native.name] = ArtifactFile(
            path=native.name,
            size_bytes=native.stat().st_size,
            sha256=file_sha256(native),
        )
    source = Path(job["source_path"])
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if (
            path.is_file()
            and not any(part.startswith(".") for part in relative.parts)
            and (
                path.name.upper().startswith(("LICENSE", "NOTICE", "COPYING"))
                or path.name.upper().endswith("_POLICY.MD")
                or relative.parts[0] in {"1_Dense", "2_Dense", "2_Normalize", "1_Pooling"}
            )
        ):
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
            name = relative.as_posix()
            records[name] = ArtifactFile(
                path=name, size_bytes=target.stat().st_size, sha256=file_sha256(target)
            )
    manifest.files = [records[key] for key in sorted(records)]
    write_data(directory / "axquant_manifest.json", manifest)
    prepare_development_model_card(
        artifact_dir=directory,
        repo_id=job["repo_id"],
        product_class="6bit",
        use_public_certification=False,
    )


def license_frontmatter(job: dict[str, Any]) -> str:
    """Retain a custom license's exact upstream identity and link."""
    if job["license"] != "other":
        return "license: " + job["license"]
    import yaml

    readme = Path(job.get("source_origin", job["source_path"])) / "README.md"
    text = readme.read_text()
    if not text.startswith("---") or len(text.split("---", 2)) != 3:
        raise ValueError("Custom license requires source model-card metadata")
    metadata = yaml.safe_load(text.split("---", 2)[1])
    if metadata.get("license") != "other" or not all(
        isinstance(metadata.get(key), str) and metadata[key].strip()
        for key in ("license_name", "license_link")
    ):
        raise ValueError("Custom license requires its original name and link")
    return "\n".join(
        f"{key}: {json.dumps(metadata[key])}" for key in ("license", "license_name", "license_link")
    )


def finish_card(job: dict[str, Any], directory: Path, smoke: dict[str, Any]) -> None:
    manifest = validate_pack(job, directory)
    mtp = (
        "Protected MTP weights included; execution, acceptance and speed unverified."
        if job["mtp"]
        else "No MTP payload is advertised."
    )
    weight_bytes = manifest.weight_file_size_bytes
    parameters = manifest.logical_parameters
    assistant_note = ""
    if job.get("assistant_path"):
        manifest.runtime = load_model(directory / "axquant_runtime.json", RuntimeMetadata)
        extra_parameters, extra_bytes = assistant_weight_accounting(directory / "assistant")
        weight_bytes += extra_bytes
        parameters += extra_parameters
        mtp = "Exact-pair floating external assistant included; execution and speed unverified."
        assistant_note = (
            f"The external assistant comes from `{job['assistant_source_id']}`. "
            "Complete storage/BPW above includes the assistant. The base artifact manifest "
            "accounts for target weights; `ax_composite_pack_manifest.json` additionally "
            "binds the separate assistant and exact-pair contract."
        )
    pipeline = (
        "feature-extraction"
        if job["task"] == "embedding"
        else "image-text-to-text"
        if job["task"] in {"vision", "ocr"}
        else "text-generation"
    )
    peers = require_runtime_compatibility_record(directory)
    peer_rows = "\n".join(
        f"| {key} | {value['status']} (static only) |" for key, value in peers["targets"].items()
    )
    details = json.dumps(
        {key: smoke[key] for key in ("task", "runtime", "version", "mlx_version", "cases")},
        indent=2,
    )
    if smoke.get("runtime_environment"):
        details = json.dumps(
            {
                **json.loads(details),
                "runtime_environment": smoke["runtime_environment"],
                "elapsed_seconds": smoke["elapsed_seconds"],
            },
            indent=2,
        )
    source_url = f"https://huggingface.co/{job['source_id']}/tree/{job['source_revision']}"
    origin_note = (
        "The source was already quantized; re-quantization does not restore BF16 information."
        if job.get("source_quantized")
        else "Converted from the pinned original source with existing precision protection floors."
    )
    if smoke["runtime"] == "ax-engine":
        runtime_rows = (
            f"| AX Engine {smoke['version']} | Passed text smoke with SSD streaming requested |\n"
            "| MLX-VLM / MLX-LM | Not separately tested |"
        )
    else:
        tested = "MLX-LM" if smoke["runtime"] == "mlx-lm" else "MLX-VLM"
        other = "MLX-VLM" if tested == "MLX-LM" else "MLX-LM"
        qualifier = " + bundled Hy-V3 adapter" if job["task"] == "translation" else ""
        other_status = (
            "Unsupported hy_v3 architecture in this build"
            if job["task"] == "translation"
            else "Not separately tested"
        )
        runtime_rows = (
            f"| {tested} {smoke['version']}{qualifier} | Passed the task-specific smoke above |\n"
            f"| {other} | {other_status} |\n"
            "| AX Engine | Not tested; architecture metadata is not execution evidence |"
        )
    pipeline_metadata = "" if job["task"] == "decision" else f"pipeline_tag: {pipeline}"
    usage_note = (
        "Use `mlx_vlm.embedding_loader.load_embedding_model` for embeddings, "
        "preserving source pooling and task prefixes."
        if job["task"] == "embedding"
        else "Use the included `clef_mlx.load` and `model.systemone` API for decision heads."
        if job["task"] == "decision"
        else "Hy-MT2 requires the included MIT-licensed `hy_v3_mlx.py` adapter. "
        "After downloading, import that module and call `register_hy_v3_model()` before "
        "`mlx_lm.load`; unmodified upstream MLX-LM/VLM does not provide hy_v3. "
        "The adapter license is retained in `AXQuant-Adapter-LICENSE`."
        if job["task"] == "translation"
        else "Use the architecture-aware runtime above and preserve the source processor/template."
    )
    if smoke["runtime"] == "ax-engine":
        settings = smoke.get("runtime_environment", {})
        usage_note += (
            " The measured native MLX preview requires pre-tokenized input tokens from the "
            "original chat template; its raw text input path is unsupported. "
            "Reproduce the runtime environment recorded above: "
            + ", ".join(f"`{key}={value}`" for key, value in sorted(settings.items()))
            + ". Other execution settings were not separately verified. "
            "The elapsed time above covers two cold-start requests and SSD paging; "
            "it is not a throughput benchmark or a practical latency guarantee."
        )
    if job["source_id"].startswith("meta-models/Muse"):
        usage_note += (
            " The tested MLX-LM port is text-only and discards vision weights; image inference "
            "is unsupported by that path. Preserve ATEM channel framing and read only the "
            "explicit `to=user` final message. The smoke checks that message and retains "
            "the raw response; reasoning-only text does not pass."
        )
    vision_sidecar = directory / "vision.safetensors"
    index_path = directory / "model.safetensors.index.json"
    if job["task"] == "text" and vision_sidecar.is_file() and index_path.is_file():
        index_files = set(json.loads(index_path.read_text())["weight_map"].values())
        if vision_sidecar.name not in index_files:
            usage_note += (
                " The tested default text loading path omits the unindexed vision sidecar. "
                "Image inference is unsupported by that path for this revision and requires "
                "a separately verified vision loader or export."
            )
    install_packages = (
        "huggingface_hub transformers"
        if smoke["runtime"] == "ax-engine"
        else "huggingface_hub mlx-vlm"
    )
    license_front = license_frontmatter(job)
    card = f"""---
{license_front}
library_name: mlx
base_model: {job["source_id"]}
base_model_relation: quantized
{pipeline_metadata}
tags:
- mlx
- axquant
- 6-bit
- mixed-precision
- development
{"- mtp" if job["mtp"] else ""}
---

# {job["repo_id"].split("/")[1]}

Research development artifact. Base affine6 packing with higher-precision
protection, not MXFP6 or uniformly six-bit storage. No quality certificate is issued.

| Property | Value |
| --- | --- |
| Source | [{job["source_id"]}]({source_url}) |
| Immutable source revision | `{job["source_revision"]}` |
| Complete weight storage | {weight_bytes:,} bytes |
| Complete measured BPW | {8 * weight_bytes / parameters:.6f} |
| Manifest budget class | `{manifest.target_class}` |
| MTP | {mtp} |

The precision distribution and exact per-module settings are recorded in
`axquant_manifest.json`, `axquant_plan.json`, and `config.json`. Complete BPW
includes protected tensors and storage overhead. File size does not establish
peak runtime memory requirements. {origin_note}
{assistant_note}

## Observed runtime evidence

The exact runtime payload passed the following scoped check on Metal. The receipt
binds its runtime input files. This is a functional smoke, not a quality benchmark,
MTEB evaluation, coding evaluation, long-context test or MTP acceleration result.

```json
{details}
```

| Runtime | Evidence for this artifact |
| --- | --- |
{runtime_rows}
{peer_rows}

Unexercised modalities and MTP behavior remain unverified. Peer runtime-specific
exports require distinct variants and their own testing. See
[the manifest-bound static report](axquant_compatibility.json).

## Use and provenance

```bash
python -m pip install -U {install_packages}
hf download {job["repo_id"]} --local-dir ./{job["repo_id"].split("/")[1]}
```

Pin the immutable Hub revision. {usage_note}
The source license and notices are retained where supplied. The upstream
repository and its license govern downstream use; no additional commercial grant
or upstream benchmark/certification transfer is implied.
"""
    (directory / "README.md").write_text(card)
    write_data(directory / "axquant_smoke.json", smoke)
    for name in ("README.md", "axquant_smoke.json"):
        path = directory / name
        manifest.files = [entry for entry in manifest.files if entry.path != name]
        manifest.files.append(
            ArtifactFile(path=name, size_bytes=path.stat().st_size, sha256=file_sha256(path))
        )
    manifest.files.sort(key=lambda entry: entry.path)
    write_data(directory / "axquant_manifest.json", manifest)
    if job.get("assistant_path"):
        refresh_gemma4_assistant_composite_manifest(directory)


def publish(job: dict[str, Any], directory: Path, state: Path, smoke: dict[str, Any]) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    manifest = validate_pack(job, directory)
    require_smoke(job, smoke, payload_digest(directory, manifest))
    records = verify_files(directory, manifest)
    for name in records:
        if Path(name).suffix in {".json", ".md", ".jinja"}:
            content = (directory / name).read_text()
            if any(
                marker in content
                for marker in ("/Volumes/", "/Users/", "df-macstudio", "/home/devop")
            ):
                raise ValueError(f"Private location in public metadata: {name}")
    api = HfApi()
    if api.whoami()["name"] != "AutomatosX":
        raise ValueError("Unexpected Hub account")
    repo = job["repo_id"]
    intent = state / "upload-intent.json"
    receipt_path = state / "publication.json"
    identity = {"repo_id": repo, "manifest_sha256": records["axquant_manifest.json"]["sha256"]}
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing publication receipt identity mismatch")
        return
    try:
        existing = api.model_info(repo)
    except RepositoryNotFoundError:
        api.create_repo(repo, repo_type="model", private=True)
        write_data(intent, identity)
    else:
        if (
            not existing.private
            or not intent.is_file()
            or json.loads(intent.read_text()) != identity
        ):
            raise ValueError("Refusing to replace an unrelated existing repository")
    api.upload_large_folder(
        repo_id=repo,
        folder_path=directory,
        repo_type="model",
        allow_patterns=list(records),
        num_workers=4,
        print_report_every=60,
    )
    remote = api.model_info(repo, files_metadata=True)
    remote_files = {entry.rfilename: entry for entry in remote.siblings}
    if set(remote_files) - set(records) - {".gitattributes"}:
        raise ValueError("Unexpected remote payload")
    for name, record in records.items():
        entry = remote_files.get(name)
        if entry is None or entry.size != record["size_bytes"]:
            raise ValueError(f"Remote size mismatch: {name}")
        lfs = entry.lfs
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if digest:
            matched = digest == record["sha256"]
        else:
            matched = entry.blob_id == git_blob_sha1(directory / name)
        if not matched:
            raise ValueError(f"Remote content mismatch: {name}")
    api.update_repo_settings(repo, repo_type="model", private=False)
    public = HfApi(token=False).model_info(repo, revision=remote.sha)
    if public.private or public.sha != remote.sha:
        raise ValueError("Unauthenticated immutable-revision check failed")
    write_data(
        receipt_path,
        {
            **identity,
            "revision": remote.sha,
            "public": True,
            "verified_files": len(records),
            "quality_certified": False,
            "mtp_execution_verified": False,
        },
    )


def family_titles(job: dict[str, Any]) -> set[str]:
    source = job["source_id"]
    titles: set[str] = set()
    for prefix, title in (
        ("Qwen/", "Qwen"),
        ("nvidia/", "Nemotron"),
        ("Cloudflare/clef", "Clef"),
        ("google/gemma-4", "Gemma"),
        ("ornith-ai/", "Ornith"),
        ("meta-models/Muse", "Muse Glimmer"),
        ("openbmb/MiniCPM", "MiniCPM-V 4.7"),
        ("tencent/Hy-MT2", "Hy-MT2"),
        ("deepseek-ai/", "DeepSeek"),
        ("baidu/Unlimited-OCR", "Unlimited-OCR"),
    ):
        if source.startswith(prefix):
            titles.add(title)
    if job["task"] == "embedding":
        titles.add("Embeddings")
    if job["task"] == "ocr":
        titles.add("OCR")
    if "Qwen3-VL" in source:
        titles.add("Qwen3-VL")
    if job["task"] == "decision":
        titles.add("Clef")
    return titles


def update_collections(job: dict[str, Any], state: Path) -> None:
    """Add only this published model; never reconcile away unrelated members."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo = job["repo_id"]
    if api.model_info(repo).private:
        raise ValueError("Collections require an already public repository")
    slugs = set(BASE_COLLECTIONS) | set(job.get("collection_slugs", []))
    module = f"{__package__}.sync_hf_collections" if __package__ else "sync_hf_collections"
    catalog = importlib.import_module(module)
    for title in sorted(family_titles(job)):
        slug = catalog.KNOWN_SLUGS.get(title)
        if not slug:
            slug = catalog._find_existing(api, title)
        if not slug:
            slug = api.create_collection(
                title=title,
                namespace="AutomatosX",
                exists_ok=True,
                description=(
                    f"{title} AXQ research packs. See per-artifact runtime evidence; not certified."
                ),
            ).slug
        slugs.add(slug)
    if job["mtp"]:
        slugs.add("AutomatosX/mtp-6ac1a4e2daa11b99c2da9338")
    results = []
    note = "Affine6 base with precision protection. Scoped task smoke passed; not certified."
    for slug in sorted(slugs):
        if not slug.startswith("AutomatosX/"):
            raise ValueError("Collection must belong to the publication namespace")
        before = {item.item_id for item in api.get_collection(slug).items}
        api.add_collection_item(slug, repo, "model", note=note, exists_ok=True)
        after = {item.item_id for item in api.get_collection(slug).items}
        if not before <= after or repo not in after:
            raise ValueError("Additive collection verification failed")
        results.append({"slug": slug, "previous_members_preserved": True, "repo_id": repo})
    write_data(state / "collections.json", results)


def _execute(
    job: dict[str, Any],
    root: Path,
    do_publish: bool,
    *,
    publish_ready: bool = False,
    smoke_timeout: int | None = None,
) -> None:
    validate_job(job)
    state = root / job["key"]
    state.mkdir(parents=True, exist_ok=True)
    identity_path = state / "job.json"
    if identity_path.is_file() and json.loads(identity_path.read_text()) != job:
        raise ValueError("Job identity changed: use a separate campaign state")
    write_data(identity_path, job)
    if (state / "publication.json").is_file():
        if do_publish:
            update_collections(job, state)
        return
    ready_for_publication = False
    try:
        if publish_ready:
            status = json.loads((state / "status.json").read_text())
            if status["status"] not in {"smoke-passed", "publication-failed"}:
                raise ValueError("Publication-only mode requires a prepared passing candidate")
            directory = Path(job["output"])
            smoke = json.loads((state / "smoke.json").read_text())
            # Regenerate factual prose with the current renderer; this changes
            # no inference input and still requires the exact payload receipt.
            finish_card(job, directory, smoke)
            ready_for_publication = True
        else:
            import fcntl

            with (Path(FACTORY_MODELS) / ".affine6-factory.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                directory = convert(job, state)
                prepare(job, directory)
                manifest = validate_pack(job, directory)
                digest = payload_digest(directory, manifest)
                smoke_path = state / "smoke.json"
                if smoke_path.is_file():
                    smoke = json.loads(smoke_path.read_text())
                    require_smoke(job, smoke, digest)
                else:
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(identity_path),
                        "--worker-result",
                        str(smoke_path),
                    ]
                    run_logged(
                        command,
                        state / "smoke.log",
                        smoke_timeout
                        if smoke_timeout is not None
                        else job.get("smoke_timeout", 900),
                    )
                    smoke = json.loads(smoke_path.read_text())
                    require_smoke(job, smoke, digest)
                finish_card(job, directory, smoke)
                ready_for_publication = True
        if do_publish:
            publish(job, directory, state, smoke)
            update_collections(job, state)
            cards = Path(__file__).resolve().parents[1] / "docs" / "hub-cards"
            shutil.copy2(directory / "README.md", cards / (job["repo_id"].split("/")[1] + ".md"))
        write_data(
            state / "status.json",
            {"status": "published" if do_publish else "smoke-passed", "repo_id": job["repo_id"]},
        )
    except Exception as exc:
        write_data(
            state / "status.json",
            {
                "status": "publication-failed" if ready_for_publication else "blocked",
                "error": str(exc),
                "repo_id": job["repo_id"],
            },
        )
        raise


def execute(
    job: dict[str, Any],
    root: Path,
    do_publish: bool,
    *,
    publish_ready: bool = False,
    smoke_timeout: int | None = None,
) -> None:
    validate_job(job)
    if do_publish:
        import fcntl

        state = root / job["key"]
        state.mkdir(parents=True, exist_ok=True)
        with (state / ".publication.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _execute(
                job, root, do_publish, publish_ready=publish_ready, smoke_timeout=smoke_timeout
            )
    else:
        _execute(job, root, do_publish, publish_ready=publish_ready, smoke_timeout=smoke_timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--key", action="append", default=[])
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--publish-ready",
        action="store_true",
        help="Publish an already prepared passing candidate without GPU work",
    )
    parser.add_argument(
        "--smoke-timeout",
        type=int,
        help="Override the worker deadline in seconds without changing job identity",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.smoke_timeout is not None and args.smoke_timeout <= 0:
        parser.error("--smoke-timeout must be positive")
    require_factory_host(socket.gethostname())
    if args.worker:
        job = json.loads(args.worker.read_text())
        result = smoke_worker(job)
        result["payload_sha256"] = payload_digest(
            Path(job["output"]), validate_pack(job, Path(job["output"]))
        )
        write_data(args.worker_result, result)
        return 0 if result["passed"] else 1
    if not args.jobs or not args.state:
        parser.error("--jobs and --state are required")
    jobs = json.loads(args.jobs.read_text())
    if args.key and set(args.key) - {job["key"] for job in jobs}:
        parser.error("Unknown job key")
    failures = 0
    for job in jobs:
        if args.key and job["key"] not in args.key:
            continue
        print(f"Starting {job['key']}", flush=True)
        try:
            execute(
                job,
                args.state,
                args.publish or args.publish_ready,
                publish_ready=args.publish_ready,
                smoke_timeout=args.smoke_timeout,
            )
        except Exception as exc:
            failures += 1
            print(f"Blocked {job['key']}: {exc}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
