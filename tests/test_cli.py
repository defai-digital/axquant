from __future__ import annotations

from pathlib import Path

import pytest

import axquant.cli as cli_module
from axquant.cli import _build_parser, main
from axquant.schema import (
    ActivationCaptureManifest,
    ModelIdentity,
    ProfileName,
    QualityEvaluationResult,
    QuantizationPlan,
    SoftwareVersions,
    TokenizedCacheManifest,
)
from axquant.serde import load_model, write_data


def test_name_command_uses_product_naming(capsys) -> None:
    result = main(["name", "--base", "Qwen/Qwen3.6-27B"])
    assert result == 0
    assert capsys.readouterr().out.strip() == "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit"


def test_command_failure_shows_traceback_only_when_verbose(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # A genuine internal bug can surface as any of the caught exception
    # types (ValueError is common and not exclusive to deliberate user-input
    # validation). Normal users should see one clean line; --verbose users
    # need the traceback to actually debug it.
    def _boom(_args: object) -> int:
        raise ValueError("synthetic internal failure")

    monkeypatch.setattr(cli_module, "_run", _boom)

    quiet_result = main(["name", "--base", "Qwen/Qwen3.6-27B"])
    quiet_err = capsys.readouterr().err
    assert quiet_result == 2
    assert "synthetic internal failure" in quiet_err
    assert "Traceback" not in quiet_err

    verbose_result = main(["--verbose", "name", "--base", "Qwen/Qwen3.6-27B"])
    verbose_err = capsys.readouterr().err
    assert verbose_result == 2
    assert "synthetic internal failure" in verbose_err
    assert "Traceback" in verbose_err


def test_benchmark_ab_defaults_to_release_speedup_gate() -> None:
    args = _build_parser().parse_args(
        ["benchmark-ab", "--model", "/model", "--prompts", "/prompts.jsonl"]
    )
    assert args.minimum_speedup == 1.20
    assert args.record_failed_speedup is False
    assert args.direct_baseline_kind == "axquant-mtp-off"
    assert args.mtp_baseline_kind == "axquant-mtp-on"


def test_benchmark_ab_accepts_reference_baseline_kinds() -> None:
    args = _build_parser().parse_args(
        [
            "benchmark-ab",
            "--model",
            "/model",
            "--prompts",
            "/prompts.jsonl",
            "--direct-baseline-kind",
            "uniform-6bit",
            "--mtp-baseline-kind",
            "uniform-6bit",
        ]
    )
    assert args.direct_baseline_kind == "uniform-6bit"
    assert args.mtp_baseline_kind == "uniform-6bit"


def test_benchmark_ab_accepts_failed_speedup_recording() -> None:
    args = _build_parser().parse_args(
        [
            "benchmark-ab",
            "--model",
            "/model",
            "--prompts",
            "/prompts.jsonl",
            "--record-failed-speedup",
        ]
    )
    assert args.record_failed_speedup is True


@pytest.mark.parametrize(
    "argv",
    [
        [
            "plan",
            "--sensitivity",
            "/evidence/weights.json",
            "--kv-default-bits",
            "5",
        ],
        [
            "plan",
            "--sensitivity",
            "/evidence/weights.json",
            "--kv-default-bits",
            "3",
        ],
        [
            "analyze-kv",
            "--model",
            "/model",
            "--calibration",
            "/calibration",
            "--bits",
            "4,5",
        ],
        [
            "analyze-kv",
            "--model",
            "/model",
            "--calibration",
            "/calibration",
            "--group-size",
            "7",
        ],
    ],
)
def test_cli_rejects_non_executable_kv_grid(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _build_parser().parse_args(argv)

    assert exc_info.value.code == 2


def test_cli_accepts_executable_kv_grid() -> None:
    plan_args = _build_parser().parse_args(
        [
            "plan",
            "--sensitivity",
            "/evidence/weights.json",
            "--kv-default-bits",
            "6",
        ]
    )
    analyze_args = _build_parser().parse_args(
        [
            "analyze-kv",
            "--model",
            "/model",
            "--calibration",
            "/calibration",
            "--bits",
            "2,6,16",
            "--group-size",
            "128",
        ]
    )

    assert plan_args.kv_default_bits == 6
    assert analyze_args.bits == (2, 6, 16)
    assert analyze_args.group_size == 128


def test_direct_publish_prepare_does_not_require_mtp_track_evidence() -> None:
    args = _build_parser().parse_args(
        [
            "publish-prepare",
            "--model",
            "/model",
            "--repo",
            "org/model",
            "--release-audit-request",
            "/evidence/request.json",
        ]
    )

    assert args.validation_index is None
    assert args.hardware_registry is None
    assert args.pareto_report is None


def test_mtp_publish_prepare_requires_track_evidence_at_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "prepare_publication",
        lambda **_kwargs: pytest.fail("missing arguments must fail before preparation"),
    )
    args = _build_parser().parse_args(
        ["publish-prepare", "--model", "/model", "--repo", "org/model"]
    )

    with pytest.raises(
        ValueError,
        match=r"--validation-index.*--hardware-registry.*--pareto-report",
    ):
        cli_module._run(args)


def test_recipe_export_rejects_duplicate_lineage_names() -> None:
    args = _build_parser().parse_args(
        [
            "recipe-export",
            "--plan",
            "/plan.json",
            "--bundle-id",
            "bundle",
            "--output-dir",
            "/output",
            "--lineage",
            "sensitivity=" + "a" * 64,
            "--lineage",
            "sensitivity=" + "b" * 64,
        ]
    )

    with pytest.raises(ValueError, match="duplicate lineage name"):
        cli_module._run(args)


def test_quantize_planning_overrides_are_unset_unless_explicit() -> None:
    implicit = _build_parser().parse_args(["quantize", "/model"])
    explicit = _build_parser().parse_args(
        [
            "quantize",
            "/model",
            "--profile",
            "agent-coding",
            "--kv-cache",
            "prior",
        ]
    )

    assert implicit.profile is None
    assert implicit.kv_cache is None
    assert explicit.profile == "agent-coding"
    assert explicit.kv_cache == "prior"


@pytest.mark.parametrize("runtime", ["ax-engine", "mlx-audio", "mlx-vlm", "mlx-lm-kv"])
def test_runtime_check_rejects_static_only_for_non_mlx_lm(runtime: str) -> None:
    assert (
        main(
            [
                "runtime-check",
                "--model",
                "/model",
                "--runtime",
                runtime,
                "--static-only",
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    ("runtime", "required_option"),
    [
        ("mlx-audio", "--audio-input"),
        ("mlx-vlm", "--image-input"),
    ],
)
def test_runtime_check_requires_matching_media_input(
    runtime: str,
    required_option: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["runtime-check", "--model", "/model", "--runtime", runtime]) == 2
    assert required_option in capsys.readouterr().err


@pytest.mark.parametrize("command", ["benchmark", "benchmark-ab"])
def test_benchmark_rejects_mismatched_quality_before_backend_execution(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("{}\n", encoding="utf-8")
    quality = QualityEvaluationResult.model_construct(
        model=ModelIdentity(
            model_id="org/different",
            revision="different",
            local_path=str(model),
        )
    )
    monkeypatch.setattr(cli_module, "load_model", lambda *_args: quality)

    def unexpected_backend_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("benchmark backend ran before quality identity validation")

    monkeypatch.setattr(cli_module, "run_benchmark", unexpected_backend_call)
    monkeypatch.setattr(cli_module, "run_mtp_ab", unexpected_backend_call)

    result = main(
        [
            command,
            "--model",
            str(model),
            "--model-id",
            "org/model",
            "--revision",
            "pinned",
            "--prompts",
            str(prompts),
            "--quality-evaluation",
            str(tmp_path / "quality.json"),
        ]
    )

    assert result == 2


def test_foundation_pipeline_emits_versioned_artifacts(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "architecture_report.json"
    sensitivity = tmp_path / "sensitivity_map.json"
    plan_path = tmp_path / "plan.json"
    assert (
        main(
            [
                "inspect",
                "--model",
                str(tiny_model_dir),
                "--model-id",
                "org/tiny",
                "--revision",
                "abc",
                "--output",
                str(inventory),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "analyze",
                "--model",
                str(tiny_model_dir),
                "--model-id",
                "org/tiny",
                "--profile",
                "agent-coding",
                "--output",
                str(sensitivity),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "plan",
                "--analysis",
                str(sensitivity),
                "--target-bpw",
                "15.5",
                "--allow-unmeasured",
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )
    plan = load_model(plan_path, QuantizationPlan)
    assert plan.profile.value == "agent-coding"
    assert plan.effective_bpw <= 15.5
    tied = [
        allocation
        for allocation in plan.assignments
        if allocation.tensor in {"model.embed_tokens.weight", "lm_head.weight"}
    ]
    assert len({(item.bits, item.method, item.group_size) for item in tied}) == 1
    assert plan.software_versions.axquant


def test_analyze_with_calibration_fails_until_probe_backend_exists(
    tiny_model_dir: Path,
    tmp_path: Path,
) -> None:
    result = main(
        [
            "analyze",
            "--model",
            str(tiny_model_dir),
            "--calibration",
            str(tmp_path / "cache"),
        ]
    )
    assert result == 2


def test_validate_calibration_dataset_passes_on_bundled_data() -> None:
    result = main(["validate-calibration-dataset"])
    assert result == 0


def test_analyze_capture_points_flag_parsing() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        ["analyze", "--model", "m", "--calibration", "c", "--capture-points", "output"]
    )
    assert args.capture_points == ("output",)

    args = parser.parse_args(
        ["analyze", "--model", "m", "--capture-points", "output,hidden,output"]
    )
    assert args.capture_points == ("output", "hidden")

    args = parser.parse_args(["analyze", "--model", "m"])
    assert args.capture_points is None

    with pytest.raises(SystemExit):
        parser.parse_args(["analyze", "--model", "m", "--capture-points", "bogus"])


def test_capture_activations_accepts_immutable_revision() -> None:
    revision = "a" * 40
    args = _build_parser().parse_args(
        [
            "capture-activations",
            "--model",
            "org/model",
            "--calibration",
            "cache",
            "--revision",
            revision,
        ]
    )
    assert args.revision == revision


def test_capture_activations_resolves_model_at_cache_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import axquant.capture as capture_module

    revision = "a" * 40
    cache = tmp_path / "cache"
    write_data(
        cache / "tokenized_cache_manifest.json",
        TokenizedCacheManifest(
            cache_key_sha256="b" * 64,
            model=ModelIdentity(model_id="org/model", revision=revision),
            dataset_sha256="c" * 64,
            profile=ProfileName.AGENT_CODING,
            sequence_length=32,
            samples=1,
            shard_count=1,
            total_tokens=8,
            software_versions=SoftwareVersions(
                axquant="1.1.1",
                python="3.13",
                safetensors="0.5",
                pydantic="2.0",
            ),
            complete=True,
        ),
    )
    resolved: dict[str, object] = {}

    def fake_resolve(
        model: str,
        *,
        revision: str | None = None,
        allow_download: bool = False,
    ) -> Path:
        resolved.update(model=model, revision=revision, allow_download=allow_download)
        return tmp_path / "source"

    monkeypatch.setattr(cli_module, "resolve_model_dir", fake_resolve)
    monkeypatch.setattr(
        capture_module,
        "capture_calibration_activations",
        lambda **_kwargs: ActivationCaptureManifest(
            model="org/model",
            revision=revision,
            tokenized_cache_manifest_sha256="d" * 64,
            cache_key_sha256="b" * 64,
            calibration_dataset_id="dataset",
            max_rows=8,
        ),
    )
    args = _build_parser().parse_args(
        [
            "capture-activations",
            "--model",
            "org/model",
            "--calibration",
            str(cache),
            "--output",
            str(tmp_path / "capture"),
            "--max-rows",
            "8",
            "--allow-download",
        ]
    )

    assert cli_module._run(args) == 0
    assert resolved == {
        "model": "org/model",
        "revision": revision,
        "allow_download": True,
    }


def test_validate_calibration_dataset_fails_on_missing_file(tmp_path: Path) -> None:
    result = main(["validate-calibration-dataset", "--path", str(tmp_path / "nope.jsonl")])
    assert result == 2
