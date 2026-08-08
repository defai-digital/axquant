from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog
from pydantic import ValidationError

from axquant.analyzer import architecture_prior_report
from axquant.architectures.registry import support_matrix
from axquant.benchmark import (
    parse_runtime_env_items,
    result_to_evaluation_bundle,
    run_benchmark,
    run_mtp_ab,
    run_mtp_diagnostics,
)
from axquant.calibration import calibration_manifest_sha256, prepare_calibration
from axquant.cli._parser import _build_parser
from axquant.converter import convert_model
from axquant.errors import ArtifactError, AxquantError, PlanningError
from axquant.feasibility import ArtifactTarget, assess_feasibility, feasibility_markdown
from axquant.identity import same_model_identity
from axquant.inspector import inspect_model, resolve_model_dir
from axquant.logging import configure_logging
from axquant.manual import manual_quantization_plan
from axquant.naming import model_name
from axquant.planner import allocate_kv_cache, allocate_kv_cache_measured, plan_quantization
from axquant.profiles import thresholds_for
from axquant.publisher import publish_model
from axquant.quantize import DEVELOPMENT_NOTE
from axquant.recipes import export_recipe_bundle
from axquant.reporting import plan_markdown, prepare_publication, validation_markdown
from axquant.reproduction import verify_reproduction
from axquant.revisions import is_immutable_revision
from axquant.runtime import (
    check_ax_engine,
    check_mlx_audio_transcription,
    check_mlx_lm_generation,
    check_mlx_lm_static,
    check_mlx_vlm_generation,
)
from axquant.schema import (
    BaselineKind,
    BenchmarkConfig,
    CalibrationManifest,
    EvaluationBundle,
    HardwareProfile,
    Inventory,
    KernelLatencyTable,
    KvSensitivityReport,
    ManualPlanRecipe,
    ModelIdentity,
    MtpAbComparison,
    MtpPolicy,
    MtpSidecarLayout,
    PlanRequest,
    ProfileName,
    QualityEvaluationResult,
    QuantizationPlan,
    QuantMethod,
    RuntimeName,
    SensitivityReport,
    ValidationReport,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data, write_text
from axquant.validator import validate_evaluations


def _load_matching_quality_evaluation(
    path: str | Path | None,
    model: ModelIdentity,
) -> QualityEvaluationResult | None:
    if path is None:
        return None
    result = load_model(path, QualityEvaluationResult)
    if not same_model_identity(result.model, model):
        raise ValueError("quality evaluation model does not match benchmark model")
    return result


def _output_json(path: str | Path, default_name: str) -> Path:
    output = Path(path).expanduser()
    return output if output.suffix.lower() == ".json" else output / default_name


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if (
            not separator
            or not name
            or not raw_path
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in name)
        ):
            raise ValueError("exception evidence must use lowercase-name=path")
        if name in result:
            raise ValueError(f"duplicate exception evidence name: {name}")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def _toolchain_overrides(values: list[str]) -> dict[str, str]:
    allowed = {"python", "node", "typescript", "rust", "go", "sandbox"}
    result: dict[str, str] = {}
    for value in values:
        name, separator, executable = value.partition("=")
        if not separator or name not in allowed or not executable:
            choices = ", ".join(sorted(allowed))
            raise ValueError(f"toolchain must use NAME=EXECUTABLE where NAME is one of: {choices}")
        if name in result:
            raise ValueError(f"duplicate toolchain override: {name}")
        result[name] = executable
    return result


def _run(args: argparse.Namespace) -> int:
    log = structlog.get_logger()
    if args.command == "feasibility":
        output_paths = [
            Path(args.output).expanduser().resolve(),
            Path(args.markdown_output).expanduser().resolve(),
        ]
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("feasibility JSON and markdown outputs must use different paths")
        local_checkpoint_roots = [
            candidate.resolve()
            for value in (
                args.reference_4bit,
                args.reference_6bit,
                args.mixed_baseline,
                args.source_bf16,
            )
            if value is not None
            for candidate in (Path(value).expanduser(),)
            if candidate.is_dir()
        ]
        for output_path in output_paths:
            if any(
                output_path == checkpoint_root or output_path.is_relative_to(checkpoint_root)
                for checkpoint_root in local_checkpoint_roots
            ):
                raise ValueError(
                    "feasibility outputs must not be written inside an audited checkpoint"
                )
        report = assess_feasibility(
            reference_4bit=ArtifactTarget(
                model=args.reference_4bit,
                kind=BaselineKind.UNIFORM_4BIT,
                model_id=args.reference_4bit_id,
                revision=args.reference_4bit_revision,
            ),
            reference_6bit=ArtifactTarget(
                model=args.reference_6bit,
                kind=BaselineKind.UNIFORM_6BIT,
                model_id=args.reference_6bit_id,
                revision=args.reference_6bit_revision,
            ),
            mixed_baseline=(
                ArtifactTarget(
                    model=args.mixed_baseline,
                    kind=BaselineKind.MIXED_PRECISION,
                    model_id=args.mixed_baseline_id,
                    revision=args.mixed_baseline_revision,
                )
                if args.mixed_baseline
                else None
            ),
            source_bf16=(
                ArtifactTarget(
                    model=args.source_bf16,
                    kind=BaselineKind.BF16_SOURCE,
                    model_id=args.source_bf16_id,
                    revision=args.source_bf16_revision,
                )
                if args.source_bf16
                else None
            ),
            run_runtime_checks=args.run_runtime_checks,
            ax_engine=args.ax_engine,
        )
        write_data(args.output, report)
        write_text(args.markdown_output, feasibility_markdown(report))
        log.info(
            "feasibility_completed",
            output=str(args.output),
            markdown_output=str(args.markdown_output),
            status=report.status,
        )
        if report.status == "blocked":
            return 1
        if args.require_ready and report.status != "ready-for-conversion":
            return 1
        return 0

    if args.command == "inspect":
        inventory = inspect_model(
            args.model,
            model_id=args.model_id,
            revision=args.revision,
            allow_download=args.allow_download,
            allow_quantized=args.allow_quantized,
        )
        write_data(args.output, inventory)
        log.info(
            "inspection_completed",
            output=str(args.output),
            tensors=len(inventory.tensors),
            mtp=inventory.mtp_present,
        )
        return 0

    if args.command == "source-checkpoint-manifest":
        from axquant.certification.common import build_source_checkpoint_manifest

        inventory = load_model(args.inventory, Inventory)
        source_checkpoint_manifest = build_source_checkpoint_manifest(
            args.model, inventory=inventory
        )
        write_data(args.output, source_checkpoint_manifest)
        log.info(
            "source_checkpoint_manifest_written",
            output=str(args.output),
            files=len(source_checkpoint_manifest.files),
            source_revision=source_checkpoint_manifest.source_model.revision,
        )
        return 0

    if args.command == "certification-policy":
        from axquant.certification.policy import direct_policy, direct_policy_sha256

        policy = direct_policy()
        write_data(args.output, policy)
        log.info(
            "certification_policy_written",
            output=str(args.output),
            policy_sha256=direct_policy_sha256(),
        )
        return 0

    if args.command == "direct-validation-index":
        from axquant.certification.qwen3_next_direct import (
            build_direct_release_validation_index,
        )

        direct_validation = build_direct_release_validation_index(args.request)
        write_data(args.output, direct_validation)
        log_method = log.info if direct_validation.release_ready else log.warning
        log_method(
            "direct_release_validation_index_written",
            output=str(args.output),
            release_ready=direct_validation.release_ready,
            issues=len(direct_validation.issues),
        )
        return 0 if direct_validation.release_ready else 1

    if args.command == "prepare-general-overlap":
        from axquant.coding_suite import build_general_overlap_report

        general_overlap = build_general_overlap_report(
            general_dataset_path=args.general_dataset,
            calibration_path=args.calibration,
        )
        write_data(args.output, general_overlap)
        log_method = log.info if general_overlap.passed else log.warning
        log_method(
            "general_overlap_report_written",
            output=str(args.output),
            passed=general_overlap.passed,
            matches=len(general_overlap.matches),
        )
        return 0 if general_overlap.passed else 1

    if args.command == "prepare-coding-suite":
        from axquant.coding_suite import build_coding_suite

        coding_suite = build_coding_suite(
            args.output_dir,
            calibration_path=args.calibration,
            random_seed=args.seed,
            toolchain_executables=_toolchain_overrides(args.toolchain),
        )
        unavailable = sorted(
            name for name, identity in coding_suite.toolchains.items() if identity == "unavailable"
        )
        log.info(
            "coding_suite_prepared",
            output=str(Path(args.output_dir) / "coding-suite-manifest.json"),
            tasks=len(coding_suite.tasks),
            target_tokens=sum(task.target_tokens for task in coding_suite.tasks),
            unavailable_toolchains=unavailable,
        )
        return 1 if unavailable else 0

    if args.command == "evaluate-coding-suite":
        from axquant.coding_sandbox import evaluate_coding_suite

        coding_model_path = Path(args.model).expanduser()
        output_path = Path(args.output).expanduser().resolve()

        def beside_output(value: str) -> Path:
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (output_path.parent / path).resolve()

        coding_evaluation = evaluate_coding_suite(
            model=ModelIdentity(
                model_id=args.model_id or args.model,
                revision=args.revision,
                local_path=(
                    str(coding_model_path.resolve()) if coding_model_path.is_dir() else None
                ),
            ),
            model_artifact_sha256=args.model_artifact_sha256,
            manifest_path=args.manifest,
            tokenizer_sha256=args.tokenizer_sha256,
            output_path=output_path,
            state_path=beside_output(args.state),
            raw_log_dir=beside_output(args.raw_log_dir),
            work_root=Path(args.work_root).expanduser().resolve(),
            max_sequence_length=args.max_seq_length,
            random_seed=args.seed,
            executable_overrides=_toolchain_overrides(args.toolchain),
        )
        model_errors = sum(outcome.model_error for outcome in coding_evaluation.outcomes)
        infrastructure_errors = sum(
            outcome.infrastructure_error for outcome in coding_evaluation.outcomes
        )
        log.info(
            "coding_suite_evaluated",
            output=str(output_path),
            outcomes=len(coding_evaluation.outcomes),
            model_errors=model_errors,
            infrastructure_errors=infrastructure_errors,
        )
        return 1 if model_errors or infrastructure_errors else 0

    if args.command == "evaluate-general-quality":
        from axquant.direct_quality import evaluate_general_quality

        general_model_path = Path(args.model).expanduser()
        output_path = Path(args.output).expanduser().resolve()
        raw_log_path = Path(args.raw_log_dir).expanduser()
        if not raw_log_path.is_absolute():
            raw_log_path = output_path.parent / raw_log_path
        state_path = Path(args.state).expanduser()
        if not state_path.is_absolute():
            state_path = output_path.parent / state_path
        general_evaluation = evaluate_general_quality(
            model=ModelIdentity(
                model_id=args.model_id or args.model,
                revision=args.revision,
                local_path=(
                    str(general_model_path.resolve()) if general_model_path.is_dir() else None
                ),
            ),
            model_artifact_sha256=args.model_artifact_sha256,
            dataset_path=args.dataset,
            tokenizer_sha256=args.tokenizer_sha256,
            output_path=output_path,
            state_path=state_path.resolve(),
            raw_log_dir=raw_log_path.resolve(),
            max_sequence_length=args.max_seq_length,
            max_generation_tokens=args.max_generation_tokens,
            random_seed=args.seed,
        )
        model_errors = sum(outcome.model_error for outcome in general_evaluation.outcomes)
        log_method = log.info if model_errors == 0 else log.warning
        log_method(
            "general_quality_evaluated",
            output=str(output_path),
            outcomes=len(general_evaluation.outcomes),
            model_errors=model_errors,
            perplexity=general_evaluation.perplexity,
        )
        return 1 if model_errors else 0

    if args.command == "verify-coding-suite":
        from axquant.coding_sandbox import verify_coding_suite

        output_path = Path(args.output).expanduser().resolve()
        raw_log_path = Path(args.raw_log_dir).expanduser()
        if not raw_log_path.is_absolute():
            raw_log_path = output_path.parent / raw_log_path
        self_test = verify_coding_suite(
            manifest_path=args.manifest,
            output_path=output_path,
            raw_log_dir=raw_log_path,
            work_root=Path(args.work_root).expanduser().resolve(),
            executable_overrides=_toolchain_overrides(args.toolchain),
        )
        log.info(
            "coding_suite_verified",
            output=str(output_path),
            tasks=len(self_test.oracle_outcomes),
            passed=self_test.passed,
            issues=self_test.issues,
        )
        return 0 if self_test.passed else 1

    if args.command == "calibrate":
        model_path = Path(args.model).expanduser()
        model_identity = ModelIdentity(
            model_id=args.model_id or args.model,
            revision=args.revision,
            local_path=str(model_path.resolve()) if model_path.is_dir() else None,
        )
        manifest = prepare_calibration(
            model=model_identity,
            dataset=args.dataset,
            output_dir=args.output,
            profile=args.profile,
            domains=args.domains,
            sequence_length=args.max_seq_length,
            random_seed=args.seed,
            tokenizer_revision=args.tokenizer_revision,
            separation_attested=args.attest_calibration_eval_separation,
        )
        tokenized_manifest = None
        if not args.manifest_only:
            from axquant.activation_cache import tokenize_calibration

            tokenized_manifest = tokenize_calibration(
                model=model_identity,
                dataset_path=args.dataset,
                output_dir=args.output,
                profile=args.profile,
                sequence_length=args.max_seq_length,
                random_seed=args.seed,
                tokenizer_revision=args.tokenizer_revision,
                calibration_manifest_sha256=calibration_manifest_sha256(manifest),
                separation_attested=args.attest_calibration_eval_separation,
                domains=args.domains,
            )
        log.info(
            "calibration_created",
            output=str(Path(args.output) / "calibration_manifest.json"),
            samples=manifest.samples,
            tokenized=tokenized_manifest is not None,
            tokens=tokenized_manifest.total_tokens if tokenized_manifest is not None else 0,
        )
        return 0

    if args.command == "analyze":
        if (args.base_sensitivity or args.target_tensor) and not args.calibration:
            raise ValueError(
                "--base-sensitivity/--target-tensor require measured --calibration probing"
            )
        if args.calibration_activations and not args.calibration:
            raise ValueError("--calibration-activations requires measured --calibration probing")
        inventory = inspect_model(
            args.model,
            model_id=args.model_id,
            revision=args.revision,
            allow_download=args.allow_download,
        )
        if args.calibration:
            from axquant.probe import probe_tensor_sensitivity
            from axquant.schema import ProbeConfig

            calibration_activations = None
            if args.calibration_activations:
                from axquant.capture import load_capture_activations

                calibration_activations = load_capture_activations(
                    args.calibration_activations,
                    model=inventory.model.model_id,
                    revision=args.revision or inventory.model.revision,
                )
            analysis_report = probe_tensor_sensitivity(
                inventory,
                config=ProbeConfig(
                    model=inventory.model,
                    calibration_cache=args.calibration,
                    profile=args.profile,
                    candidate_bits=args.bits,
                    candidate_methods=args.methods,
                    target_tensors=tuple(args.target_tensor),
                    group_size=args.group_size,
                    candidate_group_sizes=getattr(args, "candidate_group_sizes", ()),
                    token_budget_per_candidate=args.token_budget,
                    replay_batch_size=args.replay_batch_size,
                    metric_positions_per_sample=args.metric_positions,
                    long_context_min_tokens=args.long_context_min_tokens,
                    warmup_replays=args.warmup_replays,
                    capture_points=(
                        tuple(args.capture_points)
                        if args.capture_points is not None
                        else ("output", "hidden")
                    ),
                ),
                state_path=(args.state or str(Path(args.output).with_suffix(".progress.json"))),
                base_report=(
                    load_model(args.base_sensitivity, SensitivityReport)
                    if args.base_sensitivity
                    else None
                ),
                calibration_activations=calibration_activations,
            )
        else:
            analysis_report = architecture_prior_report(
                inventory,
                profile=args.profile,
                candidate_bits=args.bits,
                group_size=args.group_size,
                candidate_group_sizes=getattr(args, "candidate_group_sizes", ()),
            )
        write_data(args.output, analysis_report)
        log_method = log.info if analysis_report.evidence_kind.release_quality else log.warning
        log_method(
            "sensitivity_analysis_created",
            output=str(args.output),
            entries=len(analysis_report.entries),
            evidence=analysis_report.evidence_kind.value,
        )
        return 0

    if args.command == "analyze-kv":
        from axquant.kv_probe import measure_kv_sensitivity

        inventory = inspect_model(
            args.model,
            model_id=args.model_id,
            revision=args.revision,
        )
        kv_report = measure_kv_sensitivity(
            inventory,
            model_dir=args.model,
            calibration_cache=args.calibration,
            profile=args.profile,
            candidate_bits=args.bits,
            group_size=args.group_size,
            token_budget=args.token_budget,
            metric_positions=args.metric_positions,
        )
        write_data(args.output, kv_report)
        log.warning(
            "kv_sensitivity_created",
            output=str(args.output),
            layers=kv_report.text_layer_count,
            evidence=kv_report.evidence_kind.value,
        )
        return 0

    if args.command == "plan":
        from axquant.ladders import get_ladder, plan_request_for_ladder
        from axquant.unified_sensitivity import attach_binding_warning, bind_unified_sensitivity

        analysis_report = load_model(args.analysis, SensitivityReport)
        if args.ladder is not None:
            ladder = get_ladder(args.ladder)
            base_request = plan_request_for_ladder(
                ladder,
                profile=analysis_report.profile,
                target_bpw=args.target_bpw,
                allow_unmeasured=args.allow_unmeasured,
            )
            candidate_bits = args.bits if args.bits is not None else base_request.candidate_bits
            group_size = args.group_size if args.group_size is not None else base_request.group_size
            candidate_group_sizes = (
                args.candidate_group_sizes
                if args.candidate_group_sizes is not None
                else base_request.candidate_group_sizes
            )
            methods = args.methods if args.methods is not None else base_request.candidate_methods
            target_bpw = args.target_bpw if args.target_bpw is not None else base_request.target_bpw
            allow_unmeasured = args.allow_unmeasured or base_request.allow_unmeasured
        else:
            candidate_bits = args.bits if args.bits is not None else (4, 6, 8, 16)
            group_size = args.group_size if args.group_size is not None else 64
            candidate_group_sizes = (
                args.candidate_group_sizes if args.candidate_group_sizes is not None else ()
            )
            methods = (
                args.methods
                if args.methods is not None
                else (
                    QuantMethod.AFFINE,
                    QuantMethod.AWQ,
                    QuantMethod.DWQ,
                    QuantMethod.BF16,
                )
            )
            target_bpw = args.target_bpw if args.target_bpw is not None else 4.8
            allow_unmeasured = args.allow_unmeasured
        request = PlanRequest(
            profile=analysis_report.profile,
            target_bpw=target_bpw,
            candidate_bits=candidate_bits,
            group_size=group_size,
            candidate_group_sizes=candidate_group_sizes,
            candidate_methods=methods,
            allow_unmeasured=allow_unmeasured,
            candidate_count=args.candidates,
            random_seed=args.seed,
            target_mode=args.mode,
            hardware=HardwareProfile(
                # Sort by method value so plan digests stay stable across PYTHONHASHSEED.
                supported_methods=tuple(
                    sorted(
                        set(methods) | {QuantMethod.BF16},
                        key=lambda method: method.value,
                    )
                )
            ),
            max_model_size_ratio_to_uniform4=args.max_size_ratio,
            minimum_quality_retention=args.minimum_quality,
            minimum_mtp_acceptance_retention=args.minimum_mtp_retention,
            minimum_mtp_speedup=args.minimum_mtp_speedup,
            lm_head_min_bits=8 if args.lm_head_floor == "8bit" else 16,
            mtp=MtpPolicy(
                mode=args.mtp,
                candidate_bits=args.mtp_bits,
                min_bits=args.mtp_min_bits,
            ),
        )
        kernel_latency_table = (
            load_model(args.latency_table, KernelLatencyTable) if args.latency_table else None
        )
        plan = plan_quantization(analysis_report, request, kernel_latency=kernel_latency_table)
        if args.ladder is not None:
            plan.warnings.append(f"convert ladder: {args.ladder.value}")
        if args.kv_cache == "prior":
            layer_count = analysis_report.architecture_profile.text_layer_count
            if layer_count is None:
                raise PlanningError("KV-cache planning requires a known text layer count")
            plan.kv_cache = allocate_kv_cache(
                layer_count,
                default_bits=args.kv_default_bits,
                group_size=group_size,
            )
        elif args.kv_cache == "measured":
            if not args.kv_analysis:
                raise PlanningError("--kv-cache measured requires --kv-analysis")
            kv_report = load_model(args.kv_analysis, KvSensitivityReport)
            if kv_report.model.model_id != plan.source_model.model_id:
                raise PlanningError("KV sensitivity report model does not match the plan model")
            plan.kv_cache = allocate_kv_cache_measured(
                kv_report,
                max_output_kl=args.kv_max_kl,
            )
        if args.bind_kv_sensitivity or args.kv_cache == "measured":
            kv_path = args.bind_kv_sensitivity or args.kv_analysis
            binding = bind_unified_sensitivity(
                analysis_report,
                kv_sensitivity=kv_path,
                plan=plan,
            )
            attach_binding_warning(plan, binding)
            write_data(args.unified_binding_output, binding)
            log.info("unified_sensitivity_bound", output=str(args.unified_binding_output))
        output = _output_json(args.output, "plan-01.json")
        write_data(output, plan)
        log.info(
            "plan_created",
            output=str(output),
            effective_bpw=round(plan.effective_bpw, 6),
            evidence=plan.evidence_kind.value,
            kv_cache=(plan.kv_cache.allocation_basis if plan.kv_cache else "off"),
        )
        return 0

    if args.command == "plan-manual":
        inventory = load_model(args.inventory, Inventory)
        recipe = load_model(args.recipe, ManualPlanRecipe)
        plan = manual_quantization_plan(inventory, recipe)
        write_data(args.output, plan)
        if args.markdown_output:
            write_text(args.markdown_output, plan_markdown(plan))
        log.warning(
            "unmeasured_manual_plan_created",
            output=str(args.output),
            effective_bpw=round(plan.effective_bpw, 6),
        )
        return 0

    if args.command == "plan-replay":
        from axquant.plan_replay import replay_measured_plan_file

        analysis_report = load_model(args.sensitivity, SensitivityReport)
        plan = replay_measured_plan_file(
            analysis_report,
            args.source_plan,
            ax_engine_executable=args.ax_engine_bench,
        )
        write_data(args.output, plan)
        log.info(
            "measured_plan_replayed",
            output=str(args.output),
            effective_bpw=round(plan.effective_bpw, 6),
            sensitivity_sha256=plan.analysis_sha256,
        )
        return 0

    if args.command == "convert":
        plan = load_model(args.plan, QuantizationPlan)
        calibration_activations = None
        if args.calibration_activations:
            from axquant.capture import load_capture_activations

            calibration_activations = load_capture_activations(
                args.calibration_activations,
                model=plan.source_model.model_id,
                revision=args.revision or plan.source_model.revision,
            )
        convert_model(
            model=args.model,
            plan=plan,
            output=args.output,
            revision=args.revision,
            mtp_sidecar=args.mtp_sidecar,
            mtp_layout=MtpSidecarLayout(args.mtp_layout),
            calibration_manifest=args.calibration_manifest,
            kv_sensitivity=args.kv_sensitivity,
            calibration_activations=calibration_activations,
            allow_unmeasured=args.allow_unmeasured,
            ax_engine_manifest=args.ax_engine_manifest,
            ax_engine_bench=args.ax_engine_bench,
        )
        return 0

    if args.command == "quantize":
        from axquant.simple_convert import simple_convert

        model_ref = args.model_option or args.model_positional
        if not model_ref:
            raise PlanningError(
                "quantize requires a model: `axquant quantize MODEL` or `--model PATH|HUB_ID`"
            )
        if (
            args.model_option
            and args.model_positional
            and args.model_option != args.model_positional
        ):
            raise PlanningError("conflicting positional MODEL and --model values")
        summary = simple_convert(
            model_ref,
            output=args.output,
            model_id=args.model_id,
            revision=args.revision,
            profile=(ProfileName(args.profile) if args.profile is not None else None),
            target_bpw=args.target_bpw,
            ladder=args.ladder,
            kv_cache=args.kv_cache,
            recipe=args.recipe,
            allow_download=args.allow_download,
            allow_quantized=args.allow_quantized,
            calibration_manifest=args.calibration_manifest,
            kv_sensitivity=args.kv_sensitivity,
            mtp_sidecar=args.mtp_sidecar,
            runtime_smoke=args.runtime_smoke,
            ax_engine=args.ax_engine,
            mlx_lm=args.mlx_lm,
            python=args.python,
            audio_input=args.audio_input,
            image_input=args.image_input,
            ax_engine_manifest=args.ax_engine_manifest,
        )
        if args.json_output:
            write_data(args.json_output, summary)
        log.info(
            "quantize_completed",
            output=summary.output_path,
            family=summary.product_family,
            tier=summary.support_tier.value,
            evidence=summary.evidence_kind.value,
            plan_source=summary.plan_source,
            ladder=summary.convert_ladder,
            measured_bpw=round(summary.measured_total_bpw, 4),
            runtime_smoke=summary.runtime_smoke,
            runtime_smoke_passed=summary.runtime_smoke_passed,
        )
        if summary.development_evidence:
            log.warning("development_evidence", note=DEVELOPMENT_NOTE)
        return 0 if summary.runtime_smoke_passed is not False else 1

    if args.command == "simple-convert-help":
        from axquant.simple_convert import simple_convert_help_markdown

        text = simple_convert_help_markdown()
        if args.output:
            write_text(args.output, text)
            log.info("simple_convert_help_written", output=str(args.output))
        else:
            print(text)
        return 0

    if args.command == "head-to-head":
        from axquant.head_to_head import render_head_to_head

        page = render_head_to_head(args.benchmark_index, title=args.title)
        write_text(args.output, page)
        log.info("head_to_head_written", output=str(args.output))
        return 0

    if args.command == "scoreboard":
        from axquant.scoreboard import (
            build_scoreboard,
            require_scoreboard_inputs_for_certification,
            scoreboard_markdown,
        )

        scoreboard_report = build_scoreboard(
            plan=args.plan,
            profile=args.profile,
            evaluation_profile=args.evaluation_profile,
            title=args.title,
            candidate_size=args.candidate_size,
            size_reference=args.size_reference,
            quality_comparison=args.quality_comparison,
            validation_report=args.validation_report,
            mtp_ab=args.mtp_ab,
            candidate_evaluation=args.candidate_evaluation,
            reference_evaluation=args.reference_evaluation,
            minimum_quality_retention=args.minimum_quality,
            max_size_ratio_to_uniform4=args.max_size_ratio,
            minimum_mtp_acceptance_retention=args.minimum_mtp_retention,
            minimum_mtp_speedup=args.minimum_mtp_speedup,
            minimum_mtp_prompt_median_speedup=args.minimum_mtp_prompt_speedup,
            require_mtp_acceleration=args.require_mtp_acceleration,
        )
        write_data(args.output, scoreboard_report)
        write_text(args.markdown_output, scoreboard_markdown(scoreboard_report))
        log.info(
            "scoreboard_written",
            output=str(args.output),
            overall=scoreboard_report.overall_status,
            missing=len(scoreboard_report.missing_mandatory),
        )
        if args.require_complete:
            require_scoreboard_inputs_for_certification(scoreboard_report)
        return 0 if scoreboard_report.overall_status != "fail" else 1

    if args.command == "probe-capacity":
        from axquant.probe_capacity import (
            assess_probe_capacity,
            assess_probe_capacity_from_inventory,
            probe_capacity_markdown,
        )

        if args.inventory is None and args.parameter_count is None:
            raise PlanningError("probe-capacity requires --inventory or --parameter-count")
        if args.inventory is not None:
            capacity = assess_probe_capacity_from_inventory(
                args.inventory,
                available_memory_bytes=args.available_memory_bytes,
                headroom_fraction=args.headroom_fraction,
            )
        else:
            capacity = assess_probe_capacity(
                parameter_count=args.parameter_count,
                model=(ModelIdentity(model_id=args.model_id) if args.model_id else None),
                available_memory_bytes=args.available_memory_bytes,
                headroom_fraction=args.headroom_fraction,
            )
        write_data(args.output, capacity)
        write_text(args.markdown_output, probe_capacity_markdown(capacity))
        log.info(
            "probe_capacity_written",
            output=str(args.output),
            recommended=capacity.recommended_mode.value,
        )
        return 0

    if args.command == "quantize-mtp-sidecar":
        import shlex

        from axquant.mtp_sidecar import (
            probe_ax_engine_mtp_capability,
            quantize_qwen36_mtp_sidecar,
        )
        from axquant.schema import AxEngineMtpCapabilityCheck

        if bool(args.capability_command) == bool(args.capability_result):
            raise ArtifactError(
                "quantize-mtp-sidecar requires exactly one of --capability-command "
                "(live AX Engine probe) or --capability-result (recorded check)"
            )
        if args.capability_command:
            capability = probe_ax_engine_mtp_capability(shlex.split(args.capability_command))
        else:
            capability = load_model(args.capability_result, AxEngineMtpCapabilityCheck)
        sidecar_manifest = quantize_qwen36_mtp_sidecar(
            args.sidecar,
            args.output,
            bits=args.bits,
            group_size=args.group_size,
            capability=capability,
        )
        write_data(args.manifest_output, sidecar_manifest)
        if args.runtime_json:
            from axquant.mtp_sidecar import annotate_mtp_runtime_sidecar_bits

            annotate_mtp_runtime_sidecar_bits(args.runtime_json, args.bits)
        log.info(
            "quantized_mtp_sidecar_written",
            output=str(args.output),
            manifest=str(args.manifest_output),
            default_bits=sidecar_manifest.default_bits,
            quantized_tensors=sum(1 for t in sidecar_manifest.tensors if t.quantized),
            layout=sidecar_manifest.capability.layout,
        )
        return 0

    if args.command == "kv-serving-quality":
        import json as _json

        from axquant.kv_quality import build_kv_serving_quality_report
        from axquant.schema import KvServingQualityProfileResult

        kv_plan_source = load_model(args.plan, QuantizationPlan)
        if kv_plan_source.kv_cache is None:
            raise ArtifactError("kv-serving-quality requires a plan with a kv_cache section")
        summary_payload = _json.loads(
            Path(args.execution_summary).expanduser().read_text(encoding="utf-8")
        )
        if not isinstance(summary_payload, dict):
            raise ArtifactError("KV execution summary must be a JSON object")
        results_payload = _json.loads(Path(args.results).expanduser().read_text(encoding="utf-8"))
        if not isinstance(results_payload, list):
            raise ArtifactError("KV quality results must be a JSON array")
        kv_results = [
            KvServingQualityProfileResult.model_validate(item) for item in results_payload
        ]
        kv_serving_report = build_kv_serving_quality_report(
            model=kv_plan_source.source_model,
            kv_plan=kv_plan_source.kv_cache,
            execution_summary=summary_payload,
            results=kv_results,
            kv_sensitivity_sha256=(
                args.kv_sensitivity_sha256
                if args.kv_sensitivity_sha256
                else kv_plan_source.kv_cache.sensitivity_sha256
            ),
        )
        write_data(args.output, kv_serving_report)
        log.info(
            "kv_serving_quality_written",
            output=str(args.output),
            kv_plan_sha256=kv_serving_report.kv_plan_sha256,
            results=len(kv_serving_report.results),
        )
        return 0

    if args.command == "benchmark-kernels":
        from axquant.kernel_latency import measure_mlx_kernel_latency

        if args.from_ax_engine:
            import json as _json

            from axquant.schema import KernelLatencyEntry
            from axquant.versioning import collect_versions

            raw_document = _json.loads(
                Path(args.from_ax_engine).expanduser().read_text(encoding="utf-8")
            )
            if (
                not isinstance(raw_document, dict)
                or raw_document.get("schema_version") != "ax-engine.kernel-latency-raw.v1"
            ):
                raise ArtifactError(
                    "--from-ax-engine expects an ax-engine.kernel-latency-raw.v1 document"
                )
            raw_entries = raw_document.get("entries")
            if not isinstance(raw_entries, list) or not raw_entries:
                raise ArtifactError("ax-engine kernel latency document has no entries")
            unknown_methods = sorted(
                {
                    str(item.get("method"))
                    for item in raw_entries
                    if isinstance(item, dict) and item.get("method") not in ("bf16", "affine")
                }
            )
            if unknown_methods:
                raise ArtifactError(
                    "ax-engine kernel latency document reports methods this toolkit "
                    f"does not recognize: {unknown_methods}; refusing to relabel them"
                )
            try:
                ingested = [
                    KernelLatencyEntry(
                        runtime=RuntimeName.AX_ENGINE,
                        bits=int(item["bits"]),
                        group_size=(
                            int(item["group_size"]) if item.get("group_size") is not None else None
                        ),
                        method=(
                            QuantMethod.BF16 if item.get("method") == "bf16" else QuantMethod.AFFINE
                        ),
                        hidden_size=int(item["hidden_size"]),
                        decode_median_us=float(item["decode_median_us"]),
                        prefill_median_us=float(item["prefill_median_us"]),
                        dispersion=float(item.get("dispersion") or 0.0),
                        iterations=int(item["iterations"]),
                    )
                    for item in raw_entries
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise ArtifactError(f"ax-engine kernel latency entry is malformed: {exc}") from exc
            reported_engine = str(raw_document.get("ax_engine_version") or "")
            table = KernelLatencyTable(
                host_id=args.host_id,
                chip=args.chip,
                os_version=args.os_version,
                software_versions=collect_versions().model_copy(
                    update={"ax_engine": reported_engine or None}
                ),
                warmup_iterations=int(raw_document.get("warmup_iterations") or 0),
                entries=ingested,
                warnings=[str(item) for item in raw_document.get("warnings") or []],
            )
            write_data(args.output, table)
            log.info(
                "kernel_latency_table_written",
                output=args.output,
                entries=len(table.entries),
                host_id=table.host_id,
                source="ax-engine",
            )
            return 0

        table = measure_mlx_kernel_latency(
            host_id=args.host_id,
            chip=args.chip,
            os_version=args.os_version,
            bits_grid=args.bits,
            group_sizes=args.group_sizes,
            hidden_sizes=args.hidden_sizes,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
        )
        write_data(args.output, table)
        for warning in table.warnings:
            log.warning("kernel_latency_skip", detail=warning)
        log.info(
            "kernel_latency_table_written",
            output=args.output,
            entries=len(table.entries),
            host_id=table.host_id,
        )
        return 0

    if args.command == "ladders":
        from axquant.ladders import ladder_markdown, list_ladders

        ladders = list_ladders()
        write_text(args.markdown_output, ladder_markdown(ladders))
        if args.output:
            write_data(
                args.output,
                {
                    "schema_version": "axquant.convert-ladders.v1",
                    "ladders": [
                        {
                            "name": item.name.value,
                            "evidence_kind": item.evidence_kind.value,
                            "default_target_bpw": item.default_target_bpw,
                            "candidate_bits": list(item.candidate_bits),
                            "candidate_group_sizes": list(item.candidate_group_sizes),
                            "candidate_methods": [
                                method.value for method in item.candidate_methods
                            ],
                            "estimated_relative_cost": item.estimated_relative_cost,
                            "requires_calibration": item.requires_calibration,
                            "description": item.description,
                        }
                        for item in ladders
                    ],
                },
            )
        log.info("ladders_listed", count=len(ladders), markdown=str(args.markdown_output))
        return 0

    if args.command == "deferred-features":
        from axquant.deferred import deferred_feature_matrix

        deferred_matrix = deferred_feature_matrix()
        for entry in deferred_matrix:
            log.info("deferred_feature", feature=entry["feature"], status=entry["status"])
        if args.output:
            write_data(
                args.output,
                {
                    "schema_version": "axquant.deferred-features.v1",
                    "features": deferred_matrix,
                },
            )
        return 0

    if args.command == "recovery-rank":
        from axquant.recovery import rank_recovery_targets

        ranking = rank_recovery_targets(
            args.plan,
            sensitivity=args.sensitivity,
            limit=args.limit,
        )
        write_data(args.output, ranking)
        log.info(
            "recovery_ranking_written",
            output=str(args.output),
            targets=len(ranking.targets),
        )
        return 0

    if args.command == "bind-sensitivity":
        from axquant.unified_sensitivity import bind_unified_sensitivity

        binding = bind_unified_sensitivity(
            args.sensitivity,
            kv_sensitivity=args.kv_sensitivity,
            plan=args.plan,
        )
        write_data(args.output, binding)
        log.info("unified_sensitivity_bound", output=str(args.output))
        return 0

    if args.command == "support-matrix":
        families = support_matrix(args.certification_registry)
        for family_entry in families.entries:
            log.info(
                "support_matrix_entry",
                adapter=family_entry.adapter_id,
                family=family_entry.product_family,
                tier=family_entry.support_tier.value,
                posture=family_entry.investment_posture,
                priority=family_entry.priority,
                cert_track=family_entry.cert_track,
            )
        if args.output:
            write_data(args.output, families)
            log.info("support_matrix_written", output=str(args.output))
        return 0

    if args.command == "support-policy":
        from axquant.support_policy import support_policy_markdown

        text = support_policy_markdown()
        if args.output:
            write_text(args.output, text)
            log.info("support_policy_written", output=str(args.output))
        else:
            print(text)
        return 0

    if args.command == "recipe-export":
        lineage: dict[str, str] = {}
        for item in args.lineage:
            name, separator, digest = item.partition("=")
            if not separator or not name or not digest:
                raise ValueError(f"lineage entries use NAME=SHA256 form, got {item!r}")
            if name in lineage:
                raise ValueError(f"duplicate lineage name: {name}")
            lineage[name] = digest
        bundle_path = export_recipe_bundle(
            plan=args.plan,
            output_dir=args.output_dir,
            bundle_id=args.bundle_id,
            lineage=lineage,
            notes=args.note,
        )
        log.info(
            "recipe_bundle_exported",
            bundle=str(bundle_path),
            bundle_id=args.bundle_id,
        )
        return 0

    if args.command == "validate":
        reference = load_model(args.reference_evaluation, EvaluationBundle)
        candidate_direct = load_model(args.candidate_direct_evaluation, EvaluationBundle)
        candidate = load_model(args.candidate_evaluation, EvaluationBundle)
        mtp_ab = load_model(args.mtp_ab, MtpAbComparison) if args.mtp_ab else None
        calibration = (
            load_model(args.calibration_manifest, CalibrationManifest)
            if args.calibration_manifest
            else None
        )
        from axquant.schema import ArtifactSizeEvidence

        size_reference = (
            load_model(args.size_reference, ArtifactSizeEvidence) if args.size_reference else None
        )
        candidate_size = (
            load_model(args.candidate_size, ArtifactSizeEvidence) if args.candidate_size else None
        )
        validation_report = validate_evaluations(
            reference,
            candidate_direct,
            candidate,
            profile=args.profile,
            thresholds=thresholds_for(args.profile),
            target_class=args.target_class,
            calibration=calibration,
            size_reference=size_reference,
            candidate_size=candidate_size,
            mtp_ab=mtp_ab,
        )
        if args.release_exception:
            if not args.plan or size_reference is None or candidate_size is None:
                raise ValueError(
                    "--release-exception requires --plan, --size-reference, and --candidate-size"
                )
            from axquant.release_exceptions import apply_release_exception
            from axquant.schema import ReleaseException

            exception = load_model(args.release_exception, ReleaseException)
            exception_evidence = _named_paths(args.exception_evidence)
            reserved_evidence = {
                "plan": Path(args.plan).expanduser().resolve(),
                "candidate_size": Path(args.candidate_size).expanduser().resolve(),
                "size_reference": Path(args.size_reference).expanduser().resolve(),
            }
            duplicated = sorted(set(exception_evidence) & set(reserved_evidence))
            if duplicated:
                raise ValueError(f"reserved exception evidence names were supplied: {duplicated}")
            validation_report = apply_release_exception(
                validation_report,
                exception,
                plan=load_model(args.plan, QuantizationPlan),
                evidence_files={**reserved_evidence, **exception_evidence},
            )
        elif args.exception_evidence:
            raise ValueError("--exception-evidence requires --release-exception")
        write_data(args.output, validation_report)
        log.info(
            "validation_completed",
            output=str(args.output),
            passed=validation_report.passed,
        )
        return 0 if validation_report.passed else 1

    if args.command == "size-evidence":
        from axquant.schema import (
            ArtifactManifest,
            ArtifactSizeEvidence,
            FeasibilityReport,
        )

        sources = [
            value
            for value in (args.artifact_manifest, args.feasibility_report)
            if value is not None
        ]
        if len(sources) != 1:
            raise ValueError(
                "size-evidence requires exactly one of --artifact-manifest or --feasibility-report"
            )
        if args.artifact_manifest:
            if not args.model_id or not is_immutable_revision(args.revision):
                raise ValueError(
                    "candidate size evidence requires --model-id and immutable --revision"
                )
            artifact = load_model(args.artifact_manifest, ArtifactManifest)
            evidence = ArtifactSizeEvidence(
                kind="candidate",
                model=ModelIdentity(
                    model_id=args.model_id,
                    revision=args.revision,
                    local_path=str(Path(args.artifact_manifest).expanduser().resolve().parent),
                ),
                logical_parameters=artifact.logical_parameters,
                weight_bytes=artifact.weight_file_size_bytes,
                measured_bpw=artifact.measured_total_bpw,
                source_sha256=file_sha256(args.artifact_manifest),
            )
        else:
            feasibility = load_model(args.feasibility_report, FeasibilityReport)
            reference_baseline_kind = (
                BaselineKind.UNIFORM_6BIT
                if args.reference_kind == "uniform-6bit"
                else BaselineKind.UNIFORM_4BIT
            )
            baseline = next(
                (audit for audit in feasibility.baselines if audit.kind == reference_baseline_kind),
                None,
            )
            if baseline is None or not baseline.complete:
                raise ValueError(
                    f"feasibility report has no complete {args.reference_kind} baseline"
                )
            evidence = ArtifactSizeEvidence(
                kind=args.reference_kind,
                model=baseline.model,
                logical_parameters=baseline.logical_parameters,
                weight_bytes=baseline.weight_bytes,
                measured_bpw=baseline.effective_bpw,
                source_sha256=file_sha256(args.feasibility_report),
            )
        write_data(args.output, evidence)
        log.info(
            "artifact_size_evidence_written",
            output=str(args.output),
            kind=evidence.kind,
            weight_bytes=evidence.weight_bytes,
        )
        return 0

    if args.command == "release-exception":
        from axquant.schema import (
            ArtifactSizeEvidence,
            ReleaseException,
            ReleaseExceptionTarget,
        )

        plan_path = Path(args.plan).expanduser().resolve()
        candidate_size_path = Path(args.candidate_size).expanduser().resolve()
        size_reference_path = Path(args.size_reference).expanduser().resolve()
        tradeoff_path = Path(args.tradeoff_evidence).expanduser().resolve()
        plan = load_model(plan_path, QuantizationPlan)
        candidate_size = load_model(candidate_size_path, ArtifactSizeEvidence)
        size_reference = load_model(size_reference_path, ArtifactSizeEvidence)
        if candidate_size.kind != "candidate" or size_reference.kind != "uniform-4bit":
            raise ValueError("release exception requires candidate and uniform-4bit size evidence")
        if candidate_size.logical_parameters != size_reference.logical_parameters:
            raise ValueError("release exception size evidence parameter counts differ")
        additional_evidence = _named_paths(args.evidence)
        reserved_evidence = {
            "plan": plan_path,
            "candidate_size": candidate_size_path,
            "size_reference": size_reference_path,
            "tradeoff": tradeoff_path,
        }
        duplicated = sorted(set(additional_evidence) & set(reserved_evidence))
        if duplicated:
            raise ValueError(f"reserved release exception evidence names: {duplicated}")
        evidence_files = {**reserved_evidence, **additional_evidence}
        missing_evidence = sorted(
            name for name, path in evidence_files.items() if not path.is_file()
        )
        if missing_evidence:
            raise ValueError(f"release exception evidence files are missing: {missing_evidence}")
        weight_size_ratio = candidate_size.weight_bytes / size_reference.weight_bytes
        exception = ReleaseException(
            exception_id=args.exception_id,
            candidate_model=candidate_size.model,
            plan_sha256=stable_sha256(plan),
            targets=[
                ReleaseExceptionTarget(
                    metric="artifact.weight_size_ratio",
                    observed_value=weight_size_ratio,
                    required_maximum=args.max_weight_size_ratio,
                    requirement="candidate weight bytes must be at most the uniform-4-bit limit",
                ),
                ReleaseExceptionTarget(
                    metric="artifact.candidate_measured_bpw",
                    observed_value=candidate_size.measured_bpw,
                    required_minimum=args.minimum_measured_bpw,
                    required_maximum=args.maximum_measured_bpw,
                    requirement="candidate measured BPW must remain within the target range",
                ),
            ],
            measured_tradeoff=args.measured_tradeoff,
            owner=args.owner,
            approved_by=args.approved_by,
            approval_reference=args.approval_reference,
            approved_at=args.approved_at,
            expires_at=args.expires_at,
            evidence_sha256={
                name: file_sha256(path) for name, path in sorted(evidence_files.items())
            },
        )
        write_data(args.output, exception)
        log.warning(
            "governed_release_exception_recorded",
            output=str(args.output),
            exception_id=exception.exception_id,
            expires_at=exception.expires_at.isoformat(),
        )
        return 0

    if args.command == "report":
        plan = load_model(args.plan, QuantizationPlan)
        sections = [plan_markdown(plan)]
        if args.validation:
            validation = load_model(args.validation, ValidationReport)
            sections.append(validation_markdown(validation))
        write_text(args.output, "\n\n".join(sections))
        log.info("report_created", output=str(args.output))
        return 0

    if args.command == "publish-prepare":
        if args.release_audit_request:
            from axquant.certification.dispatch import load_certification_request
            from axquant.schema import (
                FlagshipReleaseAuditRequest,
                Qwen3NextReleaseAuditRequest,
            )

            certification_request = load_certification_request(args.release_audit_request)
            if isinstance(certification_request, FlagshipReleaseAuditRequest):
                from axquant.publisher import prepare_flagship_publication

                missing = [
                    option
                    for option, value in (
                        ("--release-audit", args.release_audit),
                        ("--validation-index", args.validation_index),
                        ("--hardware-registry", args.hardware_registry),
                        ("--pareto-report", args.pareto_report),
                    )
                    if value is None
                ]
                if missing:
                    raise ValueError("flagship publish-prepare requires " + ", ".join(missing))
                prepared_files = prepare_flagship_publication(
                    model_dir=args.model,
                    repo_id=args.repo,
                    request_path=args.release_audit_request,
                    audit_path=args.release_audit,
                    validation_index_path=args.validation_index,
                    hardware_registry_path=args.hardware_registry,
                    pareto_report_path=args.pareto_report,
                )
            elif isinstance(certification_request, Qwen3NextReleaseAuditRequest):
                from axquant.certification.packaging import prepare_direct_publication

                prepared_files = prepare_direct_publication(
                    model_dir=args.model,
                    repo_id=args.repo,
                    request_path=args.release_audit_request,
                )
            else:
                raise ValueError(
                    "legacy M0-M8 requests use validation, hardware, and Pareto inputs "
                    "without --release-audit-request"
                )
        else:
            missing = [
                option
                for option, value in (
                    ("--validation-index", args.validation_index),
                    ("--hardware-registry", args.hardware_registry),
                    ("--pareto-report", args.pareto_report),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "publish-prepare requires "
                    + ", ".join(missing)
                    + " unless --release-audit-request is supplied"
                )
            prepared_files = prepare_publication(
                model_dir=args.model,
                repo_id=args.repo,
                validation_index_path=args.validation_index,
                hardware_registry_path=args.hardware_registry,
                pareto_report_path=args.pareto_report,
            )
        log.info(
            "publication_prepared",
            model=args.model,
            files=len(prepared_files),
        )
        return 0

    if args.command == "publish":
        published_files = publish_model(
            model_dir=args.model,
            repo_id=args.repo,
            validation_index_path=args.validation_index,
            hardware_registry_path=args.hardware_registry,
            pareto_report_path=args.pareto_report,
            release_audit_path=args.release_audit,
            release_audit_request_path=args.release_audit_request,
            certification_registry_path=args.certification_registry,
            execute=args.yes,
            private=args.private,
        )
        log.info(
            "publication_finished" if args.yes else "publication_previewed",
            repo=args.repo,
            files=len(published_files),
        )
        return 0

    if args.command == "verify-reproduction":
        verification = verify_reproduction(
            recipe_path=args.recipe,
            artifact_dir=args.artifact,
        )
        write_data(args.output, verification)
        log_method = log.info if verification.passed else log.error
        log_method(
            "reproduction_verification_completed",
            artifact=args.artifact,
            output=str(args.output),
            passed=verification.passed,
            verified_weight_files=len(verification.verified_weight_files),
        )
        return 0 if verification.passed else 1

    if args.command == "name":
        name = model_name(
            args.base,
            target_class=args.target_class,
            mtp=args.mtp_suffix,
            include_mlx=not args.no_mlx,
        )
        sys.stdout.write(f"{args.owner}/{name}\n")
        return 0

    if args.command == "runtime-check":
        runtime_model_path = Path(args.model).expanduser()
        runtime_model = ModelIdentity(
            model_id=args.model_id or args.model,
            revision=args.revision,
            local_path=(str(runtime_model_path.resolve()) if runtime_model_path.is_dir() else None),
        )
        if args.static_only and args.runtime != RuntimeName.MLX_LM.value:
            raise PlanningError("--static-only is only valid with --runtime mlx-lm")
        if args.runtime == "mlx-lm-kv":
            from axquant.runtime import check_mlx_lm_kv_layered

            result = check_mlx_lm_kv_layered(args.model, model_identity=runtime_model)
            write_data(args.output, result)
            log_method = log.info if result.passed else log.warning
            log_method(
                "runtime_check_completed",
                runtime="mlx-lm-kv",
                passed=result.passed,
                output=str(args.output),
            )
            return 0 if result.passed else 1
        runtime = RuntimeName(args.runtime)
        if runtime == RuntimeName.AX_ENGINE:
            result = check_ax_engine(
                args.model,
                executable=args.ax_engine,
                model_identity=runtime_model,
            )
        elif runtime == RuntimeName.MLX_AUDIO:
            if not args.audio_input:
                raise PlanningError("--runtime mlx-audio requires --audio-input")
            result = check_mlx_audio_transcription(
                args.model,
                audio=args.audio_input,
                executable=args.python,
                model_identity=runtime_model,
            )
        elif runtime == RuntimeName.MLX_VLM:
            if not args.image_input:
                raise PlanningError("--runtime mlx-vlm requires --image-input")
            result = check_mlx_vlm_generation(
                args.model,
                image=args.image_input,
                executable=args.python,
                model_identity=runtime_model,
            )
        else:
            result = (
                check_mlx_lm_static(args.model, model_identity=runtime_model)
                if args.static_only
                else check_mlx_lm_generation(
                    args.model,
                    executable=args.mlx_lm,
                    model_identity=runtime_model,
                )
            )
        write_data(args.output, result)
        log.info(
            "runtime_check_completed",
            runtime=runtime.value,
            output=str(args.output),
            passed=result.passed,
        )
        return 0 if result.passed else 1

    if args.command == "benchmark":
        from axquant.serde import file_sha256 as _file_sha256

        dataset_path = Path(args.prompts).expanduser().resolve()
        dataset_sha = _file_sha256(dataset_path)
        benchmark_model_path = Path(args.model).expanduser()
        model_identity = ModelIdentity(
            model_id=args.model_id or args.model,
            revision=args.revision,
            local_path=(
                str(benchmark_model_path.resolve()) if benchmark_model_path.is_dir() else None
            ),
        )
        runtime_env = parse_runtime_env_items(args.runtime_env)
        config = BenchmarkConfig(
            model=model_identity,
            mtp_enabled=args.mtp,
            baseline_kind=args.baseline_kind,
            workload=args.workload,
            dataset_sha256=dataset_sha,
            prompt_count=args.trials,
            warmup_trials=args.warmup,
            measured_trials=args.trials,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            draft_depth=args.draft_depth,
            power_mode=args.power_mode,
            quantizer=args.quantizer,
            quantizer_version=args.quantizer_version,
            random_seed=args.seed,
            timeout_seconds=args.timeout,
            runtime_env=runtime_env,
        )
        quality_result = _load_matching_quality_evaluation(
            args.quality_evaluation,
            model_identity,
        )
        bench_result = run_benchmark(
            config,
            dataset_path=dataset_path,
            executable=args.ax_engine,
            output_dir=args.log_dir,
        )
        bundle = result_to_evaluation_bundle(bench_result)
        if quality_result is not None:
            bundle.quality = quality_result.metrics
            bundle.benchmark_metadata["quality_dataset_sha256"] = quality_result.dataset_sha256
        write_data(args.output, bundle)
        log.info(
            "benchmark_completed",
            output=str(args.output),
            mtp=args.mtp,
            measured=bench_result.measured_count,
            failed=bench_result.failed_count,
        )
        return 0

    if args.command == "benchmark-ab":
        from axquant.serde import file_sha256 as _file_sha256

        dataset_path = Path(args.prompts).expanduser().resolve()
        dataset_sha = _file_sha256(dataset_path)
        model_identity = ModelIdentity(
            model_id=args.model_id or args.model,
            revision=args.revision,
            local_path=(
                str(Path(args.model).expanduser().resolve())
                if Path(args.model).expanduser().is_dir()
                else None
            ),
        )
        runtime_env = parse_runtime_env_items(args.runtime_env)
        if args.qwen36_exact_profile:
            from axquant.benchmark import QWEN36_EXACT_MTP_PROFILE_ENV

            runtime_env = {**QWEN36_EXACT_MTP_PROFILE_ENV, **runtime_env}
        config_direct = BenchmarkConfig(
            model=model_identity,
            mtp_enabled=False,
            baseline_kind=args.direct_baseline_kind,
            workload=args.workload,
            dataset_sha256=dataset_sha,
            prompt_count=args.trials,
            warmup_trials=args.warmup,
            measured_trials=args.trials,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            draft_depth=args.draft_depth,
            power_mode=args.power_mode,
            quantizer=args.quantizer,
            quantizer_version=args.quantizer_version,
            random_seed=args.seed,
            timeout_seconds=args.timeout,
            runtime_env=runtime_env,
        )
        config_mtp = BenchmarkConfig(
            model=model_identity,
            mtp_enabled=True,
            baseline_kind=args.mtp_baseline_kind,
            workload=args.workload,
            dataset_sha256=dataset_sha,
            prompt_count=args.trials,
            warmup_trials=args.warmup,
            measured_trials=args.trials,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            draft_depth=args.draft_depth,
            power_mode=args.power_mode,
            quantizer=args.quantizer,
            quantizer_version=args.quantizer_version,
            random_seed=args.seed,
            timeout_seconds=args.timeout,
            runtime_env=runtime_env,
        )
        quality_result = _load_matching_quality_evaluation(
            args.quality_evaluation,
            model_identity,
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        direct_bundle, mtp_bundle = run_mtp_ab(
            config_direct,
            config_mtp,
            dataset_path=dataset_path,
            executable=args.ax_engine,
            output_dir=output_dir,
            enforce_speedup=not args.record_failed_speedup,
            minimum_speedup=args.minimum_speedup,
            speedup_metric=args.speedup_metric,
            minimum_prompt_median_speedup=args.minimum_prompt_median_speedup,
        )
        if quality_result is not None:
            direct_bundle.quality = quality_result.metrics
            mtp_bundle.quality = quality_result.metrics
            direct_bundle.benchmark_metadata["quality_dataset_sha256"] = (
                quality_result.dataset_sha256
            )
            mtp_bundle.benchmark_metadata["quality_dataset_sha256"] = quality_result.dataset_sha256
        write_data(output_dir / "evaluation_mtp_off.json", direct_bundle)
        write_data(output_dir / "evaluation_mtp_on.json", mtp_bundle)
        log.info(
            "benchmark_ab_completed",
            output_dir=str(output_dir),
            runtime_env=runtime_env,
        )
        if args.record_failed_speedup:
            mtp_ab_comparison = load_model(
                output_dir / "mtp_ab_comparison.json",
                MtpAbComparison,
            )
            if not mtp_ab_comparison.speedup_pass:
                log.warning(
                    "benchmark_ab_speedup_gate_failed",
                    output_dir=str(output_dir),
                    speedup=mtp_ab_comparison.speedup,
                    speedup_metric=mtp_ab_comparison.speedup_metric,
                    minimum_speedup=mtp_ab_comparison.minimum_speedup,
                )
                return 1
        return 0

    if args.command == "mtp-diagnose":
        from axquant.serde import file_sha256 as _file_sha256

        dataset_path = Path(args.prompts).expanduser().resolve()
        dataset_sha = _file_sha256(dataset_path)
        model_path = Path(args.model).expanduser()
        model_identity = ModelIdentity(
            model_id=args.model_id or args.model,
            revision=args.revision,
            local_path=str(model_path.resolve()) if model_path.is_dir() else None,
        )
        runtime_env = parse_runtime_env_items(args.runtime_env)
        base_config = BenchmarkConfig(
            model=model_identity,
            mtp_enabled=False,
            baseline_kind="axquant-mtp-off",
            workload=args.workload,
            dataset_sha256=dataset_sha,
            prompt_count=args.trials,
            warmup_trials=args.warmup,
            measured_trials=args.trials,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            draft_depth=args.draft_depth,
            power_mode=args.power_mode,
            quantizer=args.quantizer,
            quantizer_version=args.quantizer_version,
            random_seed=args.seed,
            timeout_seconds=args.timeout,
            runtime_env=runtime_env,
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        diagnostic = run_mtp_diagnostics(
            base_config,
            dataset_path=dataset_path,
            executable=args.ax_engine,
            output_dir=output_dir,
            profiles=args.profiles or None,
            minimum_speedup=args.minimum_mtp_speedup,
        )
        report_path = Path(args.output).expanduser()
        if not report_path.is_absolute():
            report_path = output_dir / report_path
        write_data(report_path, diagnostic)
        log.info(
            "mtp_diagnose_completed",
            output=str(report_path),
            any_exactness_pass=diagnostic.any_exactness_pass,
            any_release_ready=diagnostic.any_release_ready,
            profiles=len(diagnostic.profiles),
        )
        return 0 if diagnostic.any_release_ready else 1

    if args.command == "evaluate-quality":
        from axquant.quality import evaluate_quality

        quality_model_path = Path(args.model).expanduser()
        quality_result = evaluate_quality(
            model=ModelIdentity(
                model_id=args.model_id or args.model,
                revision=args.revision,
                local_path=(
                    str(quality_model_path.resolve()) if quality_model_path.is_dir() else None
                ),
            ),
            dataset_path=args.dataset,
            max_sequence_length=args.max_seq_length,
            max_generation_tokens=args.max_tokens,
            random_seed=args.seed,
            max_samples=args.max_samples,
        )
        write_data(args.output, quality_result)
        log.info(
            "quality_evaluation_written",
            output=str(args.output),
            samples=quality_result.samples,
            perplexity=quality_result.metrics.perplexity,
        )
        return 0

    if args.command == "compare-quality":
        from axquant.quality import compare_quality

        comparison = compare_quality(
            load_model(args.reference, QualityEvaluationResult),
            load_model(args.candidate, QualityEvaluationResult),
        )
        write_data(args.output, comparison)
        log.info(
            "quality_comparison_written",
            output=str(args.output),
            aggregate_retention=comparison.aggregate.retention,
            perplexity_ratio=comparison.perplexity_ratio,
        )
        return 0

    if args.command == "tokenize-calibration":
        from axquant.activation_cache import tokenize_calibration

        tokenization_model_path = Path(args.model).expanduser()
        token_manifest = tokenize_calibration(
            model=ModelIdentity(
                model_id=args.model_id or args.model,
                revision=args.revision,
                local_path=(
                    str(tokenization_model_path.resolve())
                    if tokenization_model_path.is_dir()
                    else None
                ),
            ),
            dataset_path=args.dataset,
            output_dir=args.output,
            profile=args.profile,
            sequence_length=args.max_seq_length,
            random_seed=args.seed,
            tokenizer_revision=args.tokenizer_revision,
            separation_attested=args.attest_calibration_eval_separation,
            domains=args.domains,
        )
        log.info(
            "tokenization_completed",
            output=str(args.output),
            shards=token_manifest.shard_count,
            samples=token_manifest.samples,
            complete=token_manifest.complete,
        )
        return 0

    if args.command == "capture-activations":
        from axquant.activation_cache import load_cache_manifest
        from axquant.capture import CAPTURE_MANIFEST_NAME, capture_calibration_activations

        capture_cache = Path(args.calibration).expanduser().resolve()
        cache_manifest = load_cache_manifest(capture_cache)
        if cache_manifest is None:
            raise ValueError(f"calibration cache manifest is missing or invalid: {capture_cache}")
        if args.revision is not None and args.revision != cache_manifest.model.revision:
            raise ValueError("--revision does not match the tokenized calibration cache revision")
        capture_model_dir = resolve_model_dir(
            args.model,
            revision=args.revision or cache_manifest.model.revision,
            allow_download=args.allow_download,
        )
        capture_output = Path(args.output).expanduser().resolve()
        capture_manifest = capture_calibration_activations(
            model_dir=capture_model_dir,
            cache_dir=capture_cache,
            output_dir=capture_output,
            target_modules=tuple(args.target_modules) or None,
            max_rows=args.max_rows,
            token_budget=args.token_budget,
            segment_batches=args.segment_batches,
            modules_per_shard=args.modules_per_shard,
        )
        log.info(
            "capture_activations_completed",
            output=str(capture_output / CAPTURE_MANIFEST_NAME),
            modules=len(capture_manifest.entries),
            max_rows=capture_manifest.max_rows,
        )
        return 0

    if args.command == "refine":
        from axquant.refinement import refine_candidates
        from axquant.schema import RefinementConfig

        analysis_report = load_model(args.analysis, SensitivityReport)
        request = PlanRequest(
            profile=analysis_report.profile,
            target_bpw=args.target_bpw,
            candidate_bits=args.bits,
            group_size=args.group_size,
            allow_unmeasured=args.allow_unmeasured,
            candidate_count=args.top_n,
            random_seed=args.seed,
            lm_head_min_bits=8 if args.lm_head_floor == "8bit" else 16,
        )
        refine_config = RefinementConfig(
            top_n=args.top_n,
            max_iterations=args.max_iterations,
            evaluation_budget=args.eval_budget,
            wall_clock_seconds=args.wall_clock,
            convergence_threshold=args.convergence,
            swap_radius=args.swap_radius,
            random_seed=args.seed,
            holdout_measurement_set_sha256=getattr(args, "holdout_measurement_set_sha256", None),
        )
        refine_result = refine_candidates(analysis_report, request, refine_config)
        write_data(args.output, refine_result)
        log.info(
            "refinement_completed",
            output=str(args.output),
            iterations=refine_result.iterations_used,
            evaluations=refine_result.evaluations_used,
            converged=refine_result.converged,
            evidence_label=refine_result.evidence_label,
        )
        return 0

    if args.command == "recover":
        from axquant.recovery import ParameterUpdateScope, RecoveryRequest, recover_checkpoint

        recovery_request = RecoveryRequest(
            source_artifact=args.artifact,
            plan_path=args.plan,
            calibration_dataset_id=args.calibration_dataset_id,
            calibration_dataset_sha256=args.calibration_dataset_sha256,
            output=args.output,
            random_seed=args.seed,
            steps=args.steps,
            learning_rate=args.learning_rate,
            parameter_update_scope=ParameterUpdateScope(args.scope),
            quality_before_sha256=args.quality_before_sha256,
            quality_after_sha256=args.quality_after_sha256,
        )
        recovery_manifest = recover_checkpoint(recovery_request)
        log.info(
            "recovery_completed",
            output=str(args.output),
            claim=recovery_manifest.claim,
            algorithm_id=recovery_manifest.algorithm_id,
            development_evidence=recovery_manifest.development_evidence,
            weight_mutation_applied=recovery_manifest.weight_mutation_applied,
        )
        return 0

    if args.command == "refine-select":
        from axquant.refinement import (
            optimize_candidate_interactions,
            select_complete_candidate,
        )
        from axquant.schema import RefinementMeasurementSet, RefinementResult

        refinement = load_model(args.refinement, RefinementResult)
        measurements = load_model(args.measurements, RefinementMeasurementSet)
        if args.interaction:
            selected = optimize_candidate_interactions(refinement, measurements)
        else:
            selected = select_complete_candidate(refinement, measurements)
        write_data(args.output, selected)
        log.info(
            "complete_candidate_selected",
            output=str(args.output),
            candidate_id=selected.selected_candidate_id,
            plan_sha256=selected.selected_plan_sha256,
            evidence_label=selected.evidence_label,
        )
        return 0

    if args.command == "refine-measure":
        from axquant import __version__
        from axquant.refinement import (
            COMPLETE_OBJECTIVE_VERSION,
            build_complete_candidate_measurement,
        )
        from axquant.schema import (
            ArtifactManifest,
            QualityComparisonReport,
            RefinementMeasurementSet,
            RefinementResult,
        )

        refinement = load_model(args.refinement, RefinementResult)
        refine_measure_plan = refinement.candidate_plans.get(args.candidate_id)
        if refine_measure_plan is None:
            raise ValueError(f"unknown refinement candidate {args.candidate_id!r}")
        artifact = load_model(args.artifact_manifest, ArtifactManifest)
        quality = load_model(args.quality_comparison, QualityComparisonReport)
        validation = load_model(args.validation, ValidationReport)
        measurement = build_complete_candidate_measurement(
            candidate_id=args.candidate_id,
            measurement_id=args.measurement_id,
            plan=refine_measure_plan,
            artifact=artifact,
            artifact_sha256=file_sha256(args.artifact_manifest),
            quality=quality,
            quality_sha256=file_sha256(args.quality_comparison),
            validation=validation,
            validation_sha256=file_sha256(args.validation),
        )
        refinement_sha256 = stable_sha256(refinement)
        evaluator_version = f"{__version__}:{COMPLETE_OBJECTIVE_VERSION}"
        refine_measure_records = [measurement]
        if args.existing:
            existing_measurement_set = load_model(args.existing, RefinementMeasurementSet)
            if existing_measurement_set.refinement_sha256 != refinement_sha256:
                raise ValueError("existing measurement set does not match the refinement result")
            if existing_measurement_set.evaluator_version != evaluator_version:
                raise ValueError("existing measurement set uses another evaluator version")
            refine_measure_records = [*existing_measurement_set.measurements, measurement]
        measurement_set_result = RefinementMeasurementSet(
            refinement_sha256=refinement_sha256,
            evaluator_version=evaluator_version,
            measurements=refine_measure_records,
        )
        write_data(args.output, measurement_set_result)
        log.info(
            "complete_candidate_measured",
            output=str(args.output),
            candidate_id=args.candidate_id,
            measurement_id=measurement.measurement_id,
            validation_passed=measurement.validation_passed,
            objective_loss=measurement.objective_loss,
        )
        return 0

    if args.command == "refine-export":
        from axquant.schema import RefinementResult

        refinement = load_model(args.refinement, RefinementResult)
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        targets = {
            output_dir / f"{candidate_id}.json": plan
            for candidate_id, plan in refinement.candidate_plans.items()
        }
        targets[output_dir / "selected-plan.json"] = refinement.selected_plan
        export_existing = [str(path) for path in targets if path.exists()]
        if export_existing:
            raise ValueError(f"refinement export targets already exist: {export_existing[:5]}")
        for path, plan in targets.items():
            write_data(path, plan)
        log.info(
            "refinement_plans_exported",
            output_dir=str(output_dir),
            candidates=len(refinement.candidate_plans),
            selection_basis=refinement.selection_basis,
        )
        return 0

    if args.command == "refine-run":
        from axquant.refinement_runner import execute_refinement

        execution = execute_refinement(
            request_path=args.request,
            output_dir=args.output_dir,
            execute=args.execute,
        )
        log_method = (
            log.info if not args.execute or execution.selected_result is not None else log.warning
        )
        log_method(
            "refinement_execution_updated",
            output_dir=str(args.output_dir),
            execute=args.execute,
            complete=execution.complete,
            measured=len(execution.measured_candidate_ids),
            failed=len(execution.failed_candidate_ids),
            selected=execution.selected_result,
        )
        if not args.execute:
            return 0
        return (
            0
            if execution.complete
            and execution.selected_result is not None
            and execution.pareto_report is not None
            else 1
        )

    if args.command == "pareto":
        from axquant.pareto import build_pareto_report
        from axquant.schema import RefinementMeasurementSet

        measurement_set = load_model(args.measurements, RefinementMeasurementSet)
        pareto_report = build_pareto_report(measurement_set)
        write_data(args.output, pareto_report)
        log.info(
            "pareto_report_written",
            output=str(args.output),
            candidates=len(pareto_report.points),
            frontier=len(pareto_report.frontier_candidate_ids),
        )
        return 0

    if args.command == "hardware-registry":
        from axquant.hardware_registry import build_hardware_profile_registry

        registry = build_hardware_profile_registry(args.request)
        write_data(args.output, registry)
        log_method = log.info if registry.release_ready else log.warning
        log_method(
            "hardware_profile_registry_written",
            output=str(args.output),
            candidates=len(registry.entries),
            named_hosts=registry.distinct_named_hosts,
            release_ready=registry.release_ready,
        )
        return 0 if registry.release_ready else 1

    if args.command == "compatibility-matrix":
        from axquant.compatibility import build_compatibility_matrix

        matrix = build_compatibility_matrix(args.request)
        write_data(args.output, matrix)
        log_method = log.info if matrix.release_ready else log.warning
        log_method(
            "compatibility_matrix_written",
            output=str(args.output),
            candidates=len(matrix.entries),
            distinct_dense_sources=matrix.distinct_dense_source_checkpoints,
            release_ready=matrix.release_ready,
        )
        return 0 if matrix.release_ready else 1

    if args.command == "benchmark-index":
        from axquant.benchmark_evidence import build_benchmark_evidence_index

        index = build_benchmark_evidence_index(args.request)
        write_data(args.output, index)
        log_method = log.info if index.release_ready else log.warning
        log_method(
            "benchmark_evidence_index_written",
            output=str(args.output),
            profile=index.profile.value,
            release_ready=index.release_ready,
            unavailable=sum(entry.status == "unavailable" for entry in index.entries),
        )
        return 0 if index.release_ready else 1

    if args.command == "validation-index":
        from axquant.release_validation import build_release_validation_index

        release_index = build_release_validation_index(args.request)
        write_data(args.output, release_index)
        log_method = log.info if release_index.release_ready else log.warning
        log_method(
            "release_validation_index_written",
            output=str(args.output),
            release_ready=release_index.release_ready,
            profiles=[entry.profile.value for entry in release_index.entries],
        )
        return 0 if release_index.release_ready else 1

    if args.command == "release-audit":
        from axquant.certification.dispatch import build_certification_audit

        audit = build_certification_audit(args.request)
        write_data(args.output, audit)
        log_method = log.info if audit.release_ready else log.warning
        log_method(
            "release_audit_written",
            output=str(args.output),
            release_ready=audit.release_ready,
            passed=sum(check.passed for check in audit.checks),
            blockers=len(audit.blockers),
        )
        return 0 if audit.release_ready else 1

    if args.command == "campaign-overlap":
        from axquant.dataset_overlap import (
            DEFAULT_ID_FIELDS,
            DEFAULT_TEXT_FIELDS,
            build_campaign_overlap_report,
        )

        overlap = build_campaign_overlap_report(
            dataset_path=args.dataset,
            compared_paths=args.compare,
            similarity_threshold=args.threshold,
            id_fields=tuple(args.id_field or DEFAULT_ID_FIELDS),
            text_fields=tuple(args.text_field or DEFAULT_TEXT_FIELDS),
            max_comparison_pairs=args.max_comparison_pairs,
        )
        write_data(args.output, overlap)
        log_method = log.info if overlap.passed else log.warning
        log_method(
            "campaign_overlap_report_written",
            output=str(args.output),
            passed=overlap.passed,
            exact_matches=overlap.exact_match_count,
            near_duplicates=overlap.near_duplicate_count,
            comparison_pairs=overlap.comparison_pair_count,
        )
        return 0 if overlap.passed else 1

    if args.command == "campaign-frontier":
        from axquant.campaign import build_flagship_frontier

        frontier = build_flagship_frontier(
            request_path=args.request,
            output_path=args.output,
        )
        log.info(
            "flagship_frontier_written",
            output=str(args.output),
            candidates=len(frontier.entries),
            eligible=len(frontier.feasible_candidate_sha256),
            search_budget=frontier.search_budget,
            search_used=frontier.search_used,
        )
        return 0

    if args.command == "campaign-freeze":
        from axquant.campaign import freeze_campaign

        campaign = freeze_campaign(request_path=args.request, output_path=args.output)
        log.info(
            "flagship_campaign_frozen",
            output=str(args.output),
            campaign_id=campaign.campaign_id,
            formal_host=campaign.formal_host.host_id,
        )
        return 0

    if args.command == "campaign-preflight":
        from axquant.campaign import preflight_campaign

        preflight = preflight_campaign(
            campaign_path=args.campaign,
            output_path=args.output,
        )
        log_method = log.info if preflight.passed else log.warning
        log_method(
            "flagship_campaign_preflight_written",
            output=str(args.output),
            campaign_id=preflight.campaign_id,
            passed=preflight.passed,
            issues=len(preflight.issues),
        )
        return 0 if preflight.passed else 1

    if args.command == "campaign-start-formal":
        from axquant.campaign import start_formal_campaign
        from axquant.schema import CampaignPreflight, FlagshipCampaign

        campaign = load_model(args.campaign, FlagshipCampaign)
        preflight = load_model(args.preflight, CampaignPreflight)
        running = start_formal_campaign(
            campaign=campaign,
            preflight=preflight,
            output_path=args.output,
        )
        log.info(
            "flagship_formal_cycle_started",
            output=str(args.output),
            campaign_id=running.campaign_id,
        )
        return 0

    if args.command == "campaign-complete-formal":
        from axquant.campaign import complete_formal_campaign
        from axquant.schema import FlagshipCampaign

        formal_campaign = load_model(args.campaign, FlagshipCampaign)
        completed_campaign = complete_formal_campaign(
            campaign=formal_campaign,
            completion_path=args.completion,
            output_path=args.output,
        )
        log.info(
            "flagship_formal_cycle_completed",
            output=str(args.output),
            campaign_id=completed_campaign.campaign_id,
            state=completed_campaign.state.value,
        )
        return 0

    if args.command == "campaign-close-no-go":
        from axquant.campaign import close_campaign_no_go

        no_go_campaign = close_campaign_no_go(
            campaign_path=args.campaign,
            no_go_record_path=args.no_go_record,
            output_path=args.output,
        )
        log.info(
            "flagship_campaign_closed_no_go",
            output=str(args.output),
            campaign_id=no_go_campaign.campaign_id,
        )
        return 0

    if args.command == "campaign-record-publication":
        from axquant.campaign import record_campaign_publication

        published_campaign = record_campaign_publication(
            campaign_path=args.campaign,
            verification_path=args.verification,
            output_path=args.output,
        )
        log.info(
            "flagship_campaign_publication_verified",
            output=str(args.output),
            campaign_id=published_campaign.campaign_id,
        )
        return 0

    if args.command == "artifact-lifecycle":
        from axquant.lifecycle import transition_lifecycle
        from axquant.schema import (
            ArtifactLifecycleRegistry,
            ArtifactLifecycleState,
            BoundFile,
            CandidateKey,
            LifecycleReason,
            SemanticImpactScan,
        )

        lifecycle_output_path = Path(args.output).expanduser().resolve()
        unresolved_lifecycle_evidence = Path(args.evidence).expanduser()
        if unresolved_lifecycle_evidence.is_symlink():
            raise ValueError("lifecycle evidence must not be a symlink")
        lifecycle_evidence_path = unresolved_lifecycle_evidence.resolve()
        if not lifecycle_evidence_path.is_file():
            raise ValueError("lifecycle evidence must be an existing file")
        try:
            lifecycle_evidence_relative = lifecycle_evidence_path.relative_to(
                lifecycle_output_path.parent
            ).as_posix()
        except ValueError as exc:
            raise ValueError("lifecycle evidence must be inside the output registry root") from exc
        lifecycle_registry = load_model(args.registry, ArtifactLifecycleRegistry)
        lifecycle_candidate = load_model(args.candidate, CandidateKey)
        replacement_candidate = (
            load_model(args.replacement_candidate, CandidateKey)
            if args.replacement_candidate
            else None
        )
        impact_scan = load_model(args.impact_scan, SemanticImpactScan) if args.impact_scan else None
        if impact_scan is not None:
            impact_scan_path = Path(args.impact_scan).expanduser().resolve()
            impact_evidence_unresolved = impact_scan_path.parent / impact_scan.evidence.path
            impact_evidence_path = impact_evidence_unresolved.resolve()
            try:
                impact_evidence_path.relative_to(impact_scan_path.parent)
            except ValueError as exc:
                raise ValueError("impact scan evidence escapes its evidence root") from exc
            if (
                impact_evidence_unresolved.is_symlink()
                or not impact_evidence_path.is_file()
                or impact_evidence_path.stat().st_size != impact_scan.evidence.size_bytes
                or file_sha256(impact_evidence_path) != impact_scan.evidence.sha256
            ):
                raise ValueError("impact scan evidence is missing, unsafe, or changed")
        updated = transition_lifecycle(
            registry=lifecycle_registry,
            candidate=lifecycle_candidate,
            new_state=ArtifactLifecycleState(args.to),
            actor=args.actor,
            reviewer=args.reviewer,
            reason=LifecycleReason(args.reason),
            narrative=args.narrative,
            authorizing_evidence=BoundFile(
                path=lifecycle_evidence_relative,
                sha256=file_sha256(lifecycle_evidence_path),
                size_bytes=lifecycle_evidence_path.stat().st_size,
            ),
            replacement_candidate=replacement_candidate,
            public_repository=args.public_repository,
            public_revision=args.public_revision,
            impact_scan=impact_scan,
            output_path=lifecycle_output_path,
        )
        log.info(
            "artifact_lifecycle_transition_written",
            output=str(lifecycle_output_path),
            state=args.to,
            events=len(updated.events),
        )
        return 0

    if args.command == "claim-render":
        from axquant.claims import render_public_claim_request

        public_claim = render_public_claim_request(
            request_path=args.request,
            claim_path=args.output,
            model_card_path=args.model_card,
        )
        log.info(
            "certified_public_claim_rendered",
            output=str(args.output),
            model_card=str(args.model_card),
            repository=public_claim.public_repository,
        )
        return 0

    if args.command == "prepare-suite":
        from axquant.suites import build_benchmark_suites

        suite_manifest = build_benchmark_suites(args.output_dir, random_seed=args.seed)
        log.info(
            "benchmark_suite_created",
            output=str(args.output_dir),
            samples=sum(suite_manifest.samples.values()),
        )
        return 0

    if args.command == "validate-calibration-dataset":
        from axquant.calibration_dataset import validate_calibration_dataset

        resolved: str = args.path
        if resolved is None:
            from importlib.resources import files

            resolved = str(files("axquant.data") / "reference_calibration.jsonl")
        issues = validate_calibration_dataset(Path(resolved))
        if issues:
            for issue in issues:
                log.error("dataset_validation_failed", issue=issue)
            return 2
        log.info("dataset_validation_passed", path=resolved)
        return 0

    raise AssertionError(f"unhandled command {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        return _run(args)
    except (AxquantError, ValidationError, OSError, ValueError) as exc:
        structlog.get_logger().error(
            "command_failed",
            command=args.command,
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=args.verbose,
        )
        return 2


def entrypoint() -> None:
    raise SystemExit(main())
