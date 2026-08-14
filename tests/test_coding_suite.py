from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from axquant.coding_sandbox import (
    _CommandSpec,
    _compile_and_test_commands,
    _native_toolchain_executable,
    _resolved_executables,
    _sandbox_profile,
    _toolchain_read_root,
    _wait_with_limits,
    evaluate_coding_suite,
    score_coding_task,
    verify_coding_suite,
)
from axquant.coding_suite import (
    SANDBOX_POLICY_CONTRACT,
    SANDBOX_PROFILE_SHA256,
    build_coding_suite,
    build_general_overlap_report,
    build_overlap_report,
    coding_general_overlap_issues,
    load_coding_payloads,
    probe_toolchains,
    reference_coding_payloads,
)
from axquant.direct_quality import evaluate_general_quality
from axquant.errors import ArtifactError, BenchmarkError
from axquant.schema import (
    CodingEvaluationState,
    CodingModelOutput,
    CodingOverlapReport,
    CodingScorer,
    CodingSuiteManifest,
    CodingTaskManifest,
    CodingTaskPayload,
    ModelIdentity,
    QualityCheck,
    QualityTask,
)
from axquant.serde import file_sha256, load_model, stable_sha256, write_data, write_text


def _calibration(path: Path, text: str = "A deliberately unrelated calibration record.") -> Path:
    path.write_text(
        json.dumps({"id": "calibration-001", "text": text}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_reference_coding_suite_has_frozen_quotas_and_bindings(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    manifest = build_coding_suite(
        output,
        calibration_path=_calibration(tmp_path / "calibration.jsonl"),
    )
    payloads = load_coding_payloads(output / "coding-suite-manifest.json", manifest)

    assert len(manifest.tasks) == 128
    assert len(payloads) == 128
    assert sum(task.target_tokens for task in manifest.tasks) == 58_112
    assert manifest.calibration_overlap_attested
    assert manifest.sandbox_profile_sha256 == SANDBOX_PROFILE_SHA256
    assert stable_sha256(SANDBOX_POLICY_CONTRACT) == SANDBOX_PROFILE_SHA256
    assert SANDBOX_POLICY_CONTRACT["default"] == "deny"
    assert set(task.category for task in manifest.tasks) == {
        "python",
        "javascript-typescript",
        "rust",
        "go",
        "repository-context",
        "json-tool",
        "algorithm-reasoning",
        "long-context",
    }
    assert (
        sum(
            task.scorer in {CodingScorer.UNIT_TEST, CodingScorer.COMPILE} for task in manifest.tasks
        )
        >= 64
    )


@pytest.mark.parametrize(
    ("language", "candidate_path", "toolchain"),
    [
        ("rust", "candidate.rs", "rustc"),
        ("go", "candidate.go", "go"),
    ],
)
def test_compile_only_native_tasks_do_not_require_test_fixtures(
    tmp_path: Path,
    language: str,
    candidate_path: str,
    toolchain: str,
) -> None:
    source_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    (source_dir / candidate_path).write_text("", encoding="utf-8")
    payload = CodingTaskPayload(
        task_id=f"{language}-compile",
        category=language,
        language=language,
        scorer=CodingScorer.COMPILE,
        prompt="Compile this source.",
        candidate_path=candidate_path,
        target_tokens=1,
    )

    commands, _environment = _compile_and_test_commands(
        payload,
        source_dir=source_dir,
        output_dir=output_dir,
        executables={language: toolchain},
        completion_token="a" * 32,
    )

    assert len(commands) == 1
    assert commands[0].argv[0] == toolchain


def test_sandbox_profile_is_deny_default_with_split_read_write_scopes(tmp_path: Path) -> None:
    source_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    command = _CommandSpec(argv=("/usr/bin/python3", "-c", "pass"))

    profile = _sandbox_profile(
        input_dir=source_dir,
        output_dir=output_dir,
        toolchain_paths=[Path("/usr")],
        command=command,
    )

    assert profile.startswith('(version 1) (deny default) (import "system.sb")')
    assert "(allow default)" not in profile
    assert "(deny network*)" in profile
    assert f'(allow file-write* (subpath "{output_dir.resolve()}"))' in profile
    assert f'(allow file-write* (subpath "{source_dir.resolve()}"))' not in profile
    assert "(allow process-fork)" not in profile
    assert '(literal "/usr/bin/python3")' in profile
    assert f'(literal "{Path("/usr/bin/python3").resolve()}")' in profile


def test_python_framework_toolchain_root_includes_sibling_dynamic_library() -> None:
    executable = Path(
        "/Library/Frameworks/Python.framework/Versions/3.13/Resources/"
        "Python.app/Contents/MacOS/Python"
    )

    assert _toolchain_read_root(str(executable)) == Path(
        "/Library/Frameworks/Python.framework/Versions/3.13"
    )


@pytest.mark.parametrize("control", ["\n", "\r", "\t", "\x7f"])
def test_sandbox_profile_rejects_control_characters_in_paths(
    tmp_path: Path,
    control: str,
) -> None:
    command = _CommandSpec(argv=(f"/usr/bin/python{control}3", "-c", "pass"))

    with pytest.raises(ValueError, match="control characters"):
        _sandbox_profile(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            toolchain_paths=[Path("/usr")],
            command=command,
        )


def test_toolchain_symlink_invocation_is_preserved_and_identity_binds_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "rustup"
    target.write_text(
        '#!/bin/sh\ncase "$0" in\n  *rustc) echo "rustc 1.99.0" ;;\n  *) exit 9 ;;\nesac\n',
        encoding="utf-8",
    )
    target.chmod(0o755)
    rustc = tmp_path / "rustc"
    rustc.symlink_to(target)

    resolved = _resolved_executables({"rust": str(rustc)})
    identities = probe_toolchains({"rust": str(rustc)})

    assert resolved["rust"] == str(rustc)
    assert identities["rust"] == f"{rustc} -> {target} :: rustc 1.99.0"


@pytest.mark.parametrize(
    ("name", "query", "binary_name"),
    [("rust", "--print sysroot", "rustc"), ("go", "env GOROOT", "go")],
)
def test_native_toolchain_uses_reported_root_instead_of_credential_directory(
    tmp_path: Path,
    name: str,
    query: str,
    binary_name: str,
) -> None:
    credential_dir = tmp_path / ".cargo"
    shim_dir = credential_dir / "bin"
    shim_dir.mkdir(parents=True)
    (credential_dir / "credentials.toml").write_text("secret", encoding="utf-8")
    toolchain_root = tmp_path / "toolchain"
    runtime = toolchain_root / "bin" / binary_name
    runtime.parent.mkdir(parents=True)
    runtime.write_text("runtime", encoding="utf-8")
    runtime.chmod(0o755)
    shim = shim_dir / binary_name
    shim.write_text(
        f'#!/bin/sh\n[ "$*" = "{query}" ] || exit 9\nprintf "%s\\n" "{toolchain_root}"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)

    resolved = _native_toolchain_executable(name, str(shim))

    assert resolved == str(runtime)
    assert credential_dir not in Path(resolved).parents


def test_coding_suite_rejects_a_tampered_task_shard(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    manifest = build_coding_suite(
        output,
        calibration_path=_calibration(tmp_path / "calibration.jsonl"),
    )
    shard = output / next(iter(manifest.task_shards))
    shard.write_text(shard.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="missing or stale"):
        load_coding_payloads(output / "coding-suite-manifest.json", manifest)


def test_coding_suite_rejects_unbound_dataset_policy_and_symlinked_shard(
    tmp_path: Path,
) -> None:
    output = tmp_path / "suite"
    manifest = build_coding_suite(
        output,
        calibration_path=_calibration(tmp_path / "calibration.jsonl"),
    )
    manifest_path = output / "coding-suite-manifest.json"

    unbound = manifest.model_copy(update={"dataset_sha256": "0" * 64})
    write_data(manifest_path, unbound)
    with pytest.raises(ArtifactError, match="does not bind"):
        load_coding_payloads(manifest_path)

    wrong_policy = manifest.model_copy(update={"sandbox_profile_sha256": "0" * 64})
    write_data(manifest_path, wrong_policy)
    with pytest.raises(ArtifactError, match="sandbox policy"):
        load_coding_payloads(manifest_path)

    write_data(manifest_path, manifest)
    shard = output / next(iter(manifest.task_shards))
    outside = tmp_path / shard.name
    shard.replace(outside)
    shard.symlink_to(outside)
    with pytest.raises(ArtifactError, match=r"unsafe|missing or stale"):
        load_coding_payloads(manifest_path)


@pytest.mark.parametrize("unsafe_path", [".", "nested//file.py", "nested/./file.py"])
def test_coding_payload_rejects_noncanonical_paths(unsafe_path: str) -> None:
    payload = reference_coding_payloads()[0].model_dump(mode="json")
    payload["candidate_path"] = unsafe_path
    with pytest.raises(ValueError, match="safe relative paths"):
        CodingTaskPayload.model_validate(payload)


def test_coding_task_ids_cannot_escape_raw_evidence_directories() -> None:
    payload = reference_coding_payloads()[0].model_dump(mode="json")
    payload["task_id"] = "../escape"

    with pytest.raises(ValueError, match="task_id"):
        CodingTaskPayload.model_validate(payload)


def test_overlap_report_fails_on_exact_normalized_content(tmp_path: Path) -> None:
    payload = reference_coding_payloads()[0]
    calibration_text = f"{payload.prompt}\n{payload.reference or ''}"
    report = build_overlap_report(
        payloads=[payload],
        suite_dataset_sha256="a" * 64,
        calibration_path=_calibration(tmp_path / "calibration.jsonl", calibration_text),
    )

    assert not report.passed
    assert len(report.matches) == 1
    assert report.matches[0].exact


def test_general_overlap_treats_distinct_cjk_as_non_duplicates(tmp_path: Path) -> None:
    """v2 tokenization: pure CJK strings must not collapse to empty and match."""
    general_task = QualityTask(
        task_id="cjk-general",
        category="general",
        prompt="東京の港湾物流計画を要約してください。",
        reference="コンテナ取扱量と潮汐スケジュールを含める。",
        checks=[QualityCheck(kind="contains", value="港湾")],
    )
    general_dataset = tmp_path / "general.jsonl"
    write_text(general_dataset, general_task.model_dump_json() + "\n")
    calibration = _calibration(
        tmp_path / "calibration.jsonl",
        "京都の茶道文化について短く説明してください。",
    )

    report = build_general_overlap_report(
        general_dataset_path=general_dataset,
        calibration_path=calibration,
    )

    assert report.passed
    assert report.matches == []


def test_general_overlap_and_coding_separation_fail_on_normalized_duplicates(
    tmp_path: Path,
) -> None:
    coding_payload = reference_coding_payloads()[0]
    general_task = QualityTask(
        task_id="general-duplicate",
        category="general",
        prompt=coding_payload.prompt,
        reference=coding_payload.reference,
        checks=[QualityCheck(kind="contains", value="marker")],
    )
    general_dataset = tmp_path / "general.jsonl"
    write_text(general_dataset, general_task.model_dump_json() + "\n")
    calibration = _calibration(
        tmp_path / "calibration.jsonl",
        f"{general_task.prompt}\n{general_task.reference or ''}",
    )

    general_overlap = build_general_overlap_report(
        general_dataset_path=general_dataset,
        calibration_path=calibration,
    )
    coding_general_issues = coding_general_overlap_issues(
        coding_payloads=[coding_payload],
        general_tasks=[general_task],
    )

    assert not general_overlap.passed
    assert general_overlap.matches[0].exact
    assert coding_general_issues


def _python_task() -> tuple[CodingTaskManifest, CodingTaskPayload]:
    payload = reference_coding_payloads()[0]
    task = CodingTaskManifest(
        task_id=payload.task_id,
        category=payload.category,
        language=payload.language,
        prompt_sha256="a" * 64,
        reference_sha256="b" * 64,
        payload_sha256="c" * 64,
        scorer=payload.scorer,
        license_id="CC0-1.0",
        provenance="clean-room test fixture",
        target_tokens=payload.target_tokens,
        timeout_seconds=10,
        cpu_time_seconds=5,
        memory_limit_bytes=512 * 1024**2,
        process_limit=16,
        output_limit_bytes=64 * 1024,
        file_size_limit_bytes=64 * 1024**2,
        open_file_limit=128,
        long_context=False,
    )
    return task, payload


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is a macOS facility")
def test_python_unit_test_runs_in_deny_default_sandbox(tmp_path: Path) -> None:
    task, payload = _python_task()
    external_secret = tmp_path / "outside-secret.txt"
    external_secret.write_text("must remain unreadable", encoding="utf-8")
    socket_id = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    agent_socket = Path("/tmp") / f"axquant-agent-{socket_id}.sock"
    tests = payload.fixture_files[payload.test_path]
    payload = payload.model_copy(
        update={
            "fixture_files": {
                payload.test_path: tests
                + "\nfrom candidate import (\n"
                + "    AGENT_SOCKET_BLOCKED, EXTERNAL_READ_BLOCKED, EXTERNAL_WRITE_BLOCKED,\n"
                + "    INPUT_WRITE_BLOCKED, LIBRARY_READ_BLOCKED, NETWORK_BLOCKED,\n"
                + "    PROCESS_SPAWN_BLOCKED,\n"
                + "    SYSTEM_READ_BLOCKED, USERS_READ_BLOCKED, VOLUMES_READ_BLOCKED,\n"
                + ")\n"
                + "assert AGENT_SOCKET_BLOCKED\n"
                + "assert EXTERNAL_READ_BLOCKED\n"
                + "assert EXTERNAL_WRITE_BLOCKED\n"
                + "assert INPUT_WRITE_BLOCKED\n"
                + "assert LIBRARY_READ_BLOCKED\n"
                + "assert NETWORK_BLOCKED\n"
                + "assert PROCESS_SPAWN_BLOCKED\n"
                + "assert SYSTEM_READ_BLOCKED\n"
                + "assert USERS_READ_BLOCKED\n"
                + "assert VOLUMES_READ_BLOCKED\n"
            }
        }
    )
    output = f"""```python
import os
import socket
import subprocess

try:
    socket.create_connection(("1.1.1.1", 53), timeout=0.1)
except OSError:
    NETWORK_BLOCKED = True
else:
    NETWORK_BLOCKED = False

try:
    open({str(external_secret)!r}, encoding="utf-8").read()
except OSError:
    EXTERNAL_READ_BLOCKED = True
else:
    EXTERNAL_READ_BLOCKED = False

try:
    open({str(external_secret)!r}, "w", encoding="utf-8").write("overwritten")
except OSError:
    EXTERNAL_WRITE_BLOCKED = True
else:
    EXTERNAL_WRITE_BLOCKED = False

try:
    open("/etc/hosts", encoding="utf-8").read()
except OSError:
    SYSTEM_READ_BLOCKED = True
else:
    SYSTEM_READ_BLOCKED = False

try:
    os.listdir("/Library")
except OSError:
    LIBRARY_READ_BLOCKED = True
else:
    LIBRARY_READ_BLOCKED = False

try:
    os.listdir("/Users/Shared")
except OSError:
    USERS_READ_BLOCKED = True
else:
    USERS_READ_BLOCKED = False

try:
    os.listdir("/Volumes")
except OSError:
    VOLUMES_READ_BLOCKED = True
else:
    VOLUMES_READ_BLOCKED = False

agent = None
try:
    agent = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    agent.connect({str(agent_socket)!r})
except OSError:
    AGENT_SOCKET_BLOCKED = True
else:
    AGENT_SOCKET_BLOCKED = False
finally:
    if agent is not None:
        agent.close()

try:
    os.chmod(__file__, 0o600)
except OSError:
    INPUT_WRITE_BLOCKED = True
else:
    INPUT_WRITE_BLOCKED = False

try:
    subprocess.run(["/bin/true"], check=False)
except OSError:
    PROCESS_SPAWN_BLOCKED = True
else:
    PROCESS_SPAWN_BLOCKED = False

def normalize_records_00(records, modulus):
    if modulus <= 0:
        raise ValueError("modulus")
    seen = set()
    result = []
    for key, value in records:
        if key not in seen:
            seen.add(key)
            result.append((key, (value * 2 + 1) % modulus))
    return sorted(result)
"""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(agent_socket))
            listener.listen(1)
            result = score_coding_task(
                task=task,
                payload=payload,
                model_output=CodingModelOutput(
                    task_id=task.task_id,
                    output=output,
                    generated_tokens=200,
                    perplexity_loss=1.0,
                    perplexity_tokens=10,
                ),
                raw_log_dir=tmp_path / "logs",
                work_root=tmp_path / "work",
                # Prefer the active interpreter: Apple's /usr/bin/python3 is an
                # xcrun shim and fails under Seatbelt without a full Xcode install.
                executable_overrides={"python": sys.executable},
            )
    finally:
        agent_socket.unlink(missing_ok=True)

    assert result.score == 1.0
    assert result.syntax_valid
    assert result.unit_tests_passed
    assert result.sandboxed
    assert result.network_disabled
    assert not result.infrastructure_error
    assert external_secret.read_text(encoding="utf-8") == "must remain unreadable"


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is a macOS facility")
def test_python_unit_test_rejects_clean_early_exit(tmp_path: Path) -> None:
    task, payload = _python_task()
    output = """import os
os._exit(0)

def normalize_records_00(records, modulus):
    return []
"""

    result = score_coding_task(
        task=task,
        payload=payload,
        model_output=CodingModelOutput(
            task_id=task.task_id,
            output=output,
            generated_tokens=20,
            perplexity_loss=1.0,
            perplexity_tokens=10,
        ),
        raw_log_dir=tmp_path / "logs",
        work_root=tmp_path / "work",
        executable_overrides={"python": sys.executable},
    )

    assert result.score == 0.0
    assert result.syntax_valid
    assert result.unit_tests_passed is False
    assert result.exit_code == 0
    assert result.stderr_excerpt == "trusted test completion evidence missing"


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is a macOS facility")
def test_python_unit_test_cannot_replace_parent_logs_with_symlinks(tmp_path: Path) -> None:
    task, payload = _python_task()
    external_secret = tmp_path / "outside-parent-secret.txt"
    external_secret.write_text("parent-only-secret", encoding="utf-8")
    output = f"""import os
import sys

try:
    os.unlink("command.stdout")
except FileNotFoundError:
    pass
os.symlink(sys.argv[0], "command.stdout")
try:
    os.unlink("command.stderr")
except FileNotFoundError:
    pass
os.symlink({str(external_secret)!r}, "command.stderr")
os._exit(0)

def normalize_records_00(records, modulus):
    return []
"""

    result = score_coding_task(
        task=task,
        payload=payload,
        model_output=CodingModelOutput(
            task_id=task.task_id,
            output=output,
            generated_tokens=40,
            perplexity_loss=1.0,
            perplexity_tokens=10,
        ),
        raw_log_dir=tmp_path / "logs",
        work_root=tmp_path / "work",
        executable_overrides={"python": shutil.which("python3") or "python3"},
    )

    assert result.score == 0.0
    assert result.unit_tests_passed is False
    assert result.stderr_file is not None
    assert "parent-only-secret" not in (tmp_path / "logs" / result.stderr_file).read_text()


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is a macOS facility")
@pytest.mark.parametrize(
    ("language", "toolchain", "executable"),
    [
        ("python", "python", "python3"),
        ("javascript", "node", "node"),
        ("typescript", "typescript", "tsc"),
        ("rust", "rust", "rustc"),
        ("go", "go", "go"),
    ],
)
def test_reference_executable_tasks_run_under_deny_default_policy(
    tmp_path: Path,
    language: str,
    toolchain: str,
    executable: str,
) -> None:
    # Use the active interpreter for Python: Apple's /usr/bin/python3 is an
    # xcrun shim and fails under Seatbelt without a full Xcode install.
    resolved = sys.executable if language == "python" else shutil.which(executable)
    if resolved is None:
        pytest.skip(f"{executable} is not installed")
    payload = next(item for item in reference_coding_payloads() if item.language == language)
    task = CodingTaskManifest(
        task_id=payload.task_id,
        category=payload.category,
        language=payload.language,
        prompt_sha256="a" * 64,
        reference_sha256="b" * 64,
        payload_sha256="c" * 64,
        scorer=payload.scorer,
        license_id="CC0-1.0",
        provenance="clean-room toolchain fixture",
        target_tokens=payload.target_tokens,
        timeout_seconds=30,
        cpu_time_seconds=20,
        memory_limit_bytes=1024**3,
        process_limit=32,
        output_limit_bytes=1024**2,
        file_size_limit_bytes=128 * 1024**2,
        open_file_limit=256,
        long_context=False,
    )

    result = score_coding_task(
        task=task,
        payload=payload,
        model_output=CodingModelOutput(
            task_id=task.task_id,
            output=payload.reference or "",
            generated_tokens=payload.target_tokens,
            perplexity_loss=1.0,
            perplexity_tokens=10,
        ),
        raw_log_dir=tmp_path / f"{language}-logs",
        work_root=tmp_path / f"{language}-work",
        executable_overrides={toolchain: resolved},
    )

    assert result.score == 1.0, result.stderr_excerpt
    assert result.syntax_valid
    assert not result.infrastructure_error


def test_executable_scorer_fails_closed_without_sandbox(tmp_path: Path) -> None:
    task, payload = _python_task()
    result = score_coding_task(
        task=task,
        payload=payload,
        model_output=CodingModelOutput(
            task_id=task.task_id,
            output="def normalize_records_00(records, modulus): return []",
            generated_tokens=10,
            perplexity_loss=1.0,
            perplexity_tokens=10,
        ),
        raw_log_dir=tmp_path / "logs",
        work_root=tmp_path / "work",
        executable_overrides={"sandbox": str(tmp_path / "missing-sandbox")},
    )

    assert result.infrastructure_error
    assert result.score == 0.0
    assert not result.sandboxed


def test_executable_scorer_rejects_an_untrusted_sandbox_binary(tmp_path: Path) -> None:
    task, payload = _python_task()
    result = score_coding_task(
        task=task,
        payload=payload,
        model_output=CodingModelOutput(
            task_id=task.task_id,
            output="def normalize_records_00(records, modulus): return []",
            generated_tokens=10,
            perplexity_loss=1.0,
            perplexity_tokens=10,
        ),
        raw_log_dir=tmp_path / "logs",
        work_root=tmp_path / "work",
        executable_overrides={"sandbox": shutil.which("true") or sys.executable},
    )

    assert result.infrastructure_error
    assert result.score == 0.0
    assert not result.sandboxed
    assert result.stderr_excerpt is not None
    assert "trusted macOS sandbox" in result.stderr_excerpt


def test_coding_raw_output_symlink_is_rejected_before_write(tmp_path: Path) -> None:
    task, payload = _python_task()
    logs = tmp_path / "logs"
    logs.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("preserve", encoding="utf-8")
    (logs / f"{task.task_id}.model-output.txt").symlink_to(external)
    with pytest.raises(BenchmarkError, match="symbolic link"):
        score_coding_task(
            task=task,
            payload=payload,
            model_output=CodingModelOutput(
                task_id=task.task_id,
                output="candidate",
                generated_tokens=1,
                perplexity_loss=1.0,
                perplexity_tokens=1,
            ),
            raw_log_dir=logs,
            work_root=tmp_path / "work",
        )
    assert external.read_text(encoding="utf-8") == "preserve"


def test_coding_output_limit_checks_final_process_bytes() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    result = _wait_with_limits(
        process,
        wall_seconds=5,
        memory_limit_bytes=1024**3,
        process_limit=8,
        output_limit_bytes=16,
        stdout_pipe=process.stdout,
        stderr_pipe=process.stderr,
    )
    process.stdout.close()
    process.stderr.close()
    assert result[4] is True
    assert len(result[6]) + len(result[7]) == 17


class _FakeCodingBackend:
    def __init__(self) -> None:
        self.generations = 0

    def load_model(self, model: str, revision: str | None) -> None:
        assert model
        assert revision

    def perplexity_loss(self, text: str, max_length: int) -> tuple[float, int]:
        assert text
        assert max_length > 0
        return 2.0, 10

    def generate(self, prompt: str, max_tokens: int, random_seed: int) -> str:
        assert "marker" in prompt
        assert max_tokens == 32
        assert random_seed >= 0
        self.generations += 1
        return "AXQ-MARKER"

    def generation_metadata(self) -> tuple[str, str | None]:
        return "raw", None

    def count_tokens(self, text: str) -> int:
        return len(text.split("-"))


def test_coding_evaluation_is_resumable_and_emits_direct_raw_outcomes(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    payload = CodingTaskPayload(
        task_id="long-context-000",
        category="long-context",
        language="text",
        scorer=CodingScorer.TEXT_EXACT,
        prompt="Return the marker.",
        reference="AXQ-MARKER",
        candidate_path="answer.txt",
        expected_text="AXQ-MARKER",
        target_tokens=32,
    )
    shard_name = "tasks-long-context.jsonl"
    shard_path = suite_dir / shard_name
    write_text(shard_path, payload.model_dump_json() + "\n")
    shards = {shard_name: file_sha256(shard_path)}
    overlap = CodingOverlapReport(
        suite_dataset_sha256=stable_sha256(shards),
        calibration_dataset_sha256="d" * 64,
        similarity_threshold=0.85,
        passed=True,
    )
    overlap_path = suite_dir / "coding-overlap-report.json"
    write_data(overlap_path, overlap)
    task = CodingTaskManifest(
        task_id=payload.task_id,
        category=payload.category,
        language=payload.language,
        prompt_sha256=hashlib.sha256(payload.prompt.encode()).hexdigest(),
        reference_sha256=hashlib.sha256((payload.reference or "").encode()).hexdigest(),
        payload_sha256=stable_sha256(payload),
        scorer=payload.scorer,
        license_id="CC0-1.0",
        provenance="clean-room test fixture",
        target_tokens=32,
        timeout_seconds=5,
        cpu_time_seconds=2,
        memory_limit_bytes=128 * 1024**2,
        process_limit=4,
        output_limit_bytes=4096,
        file_size_limit_bytes=64 * 1024**2,
        open_file_limit=128,
        long_context=True,
    )
    manifest = CodingSuiteManifest(
        suite_id="fixture",
        version="1",
        dataset_sha256=stable_sha256(shards),
        tasks=[task],
        task_shards=shards,
        calibration_overlap_attested=True,
        calibration_overlap_report=overlap_path.name,
        calibration_overlap_report_sha256=file_sha256(overlap_path),
        toolchains=probe_toolchains(),
        sandbox_profile_sha256=SANDBOX_PROFILE_SHA256,
        near_duplicate_threshold=0.85,
        random_seed=7,
    )
    manifest_path = suite_dir / "coding-suite-manifest.json"
    write_data(manifest_path, manifest)
    model = ModelIdentity(model_id="fixture/model", revision="a" * 40)
    backend = _FakeCodingBackend()
    kwargs = {
        "model": model,
        "model_artifact_sha256": "c" * 64,
        "manifest_path": manifest_path,
        "tokenizer_sha256": "b" * 64,
        "output_path": tmp_path / "evaluation.json",
        "state_path": tmp_path / "state.json",
        "raw_log_dir": tmp_path / "logs",
        "work_root": tmp_path / "work",
        "random_seed": 7,
    }
    result = evaluate_coding_suite(**kwargs, backend=backend)
    resumed = evaluate_coding_suite(**kwargs, backend=backend)

    assert backend.generations == 1
    assert result.model_dump(exclude={"created_at"}) == resumed.model_dump(exclude={"created_at"})
    assert result.outcomes[0].score == 1.0
    assert result.outcomes[0].scored_tokens == 2
    assert result.outcomes[0].output_file == "logs/long-context-000.model-output.txt"
    output_path = tmp_path / result.outcomes[0].output_file
    assert output_path.read_text(encoding="utf-8") == "AXQ-MARKER"
    assert file_sha256(output_path) == result.outcomes[0].output_sha256

    self_test = verify_coding_suite(
        manifest_path=manifest_path,
        output_path=tmp_path / "self-test.json",
        raw_log_dir=tmp_path / "self-test-logs",
        work_root=tmp_path / "self-test-work",
    )
    assert self_test.passed
    assert self_test.oracle_outcomes[0].score == 1.0
    assert self_test.empty_mutant_outcomes[0].score == 0.0

    state = load_model(tmp_path / "state.json", CodingEvaluationState)
    state.outputs.append(
        CodingModelOutput(
            task_id="unexpected-task",
            output="tampered",
            generated_tokens=1,
            perplexity_loss=1.0,
            perplexity_tokens=1,
        )
    )
    write_data(tmp_path / "state.json", state)
    with pytest.raises(BenchmarkError, match="unexpected task IDs"):
        evaluate_coding_suite(**kwargs, backend=backend)


def test_coding_evaluation_rejects_raw_logs_outside_evidence_root(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="beneath the quality evaluation directory"):
        evaluate_coding_suite(
            model=ModelIdentity(model_id="fixture/model", revision="a" * 40),
            model_artifact_sha256="c" * 64,
            manifest_path=tmp_path / "not-loaded.json",
            tokenizer_sha256="b" * 64,
            output_path=tmp_path / "evidence" / "evaluation.json",
            state_path=tmp_path / "state.json",
            raw_log_dir=tmp_path / "outside-logs",
            work_root=tmp_path / "work",
            backend=_FakeCodingBackend(),
        )


def test_general_quality_evaluation_archives_provenance_bound_outputs(tmp_path: Path) -> None:
    dataset = tmp_path / "general.jsonl"
    task = QualityTask(
        task_id="general-001",
        category="instruction",
        prompt="Return the marker.",
        reference="AXQ-MARKER",
        checks=[QualityCheck(kind="exact", value="AXQ-MARKER")],
    )
    write_text(dataset, task.model_dump_json() + "\n")
    output = tmp_path / "evidence" / "general-quality.json"
    backend = _FakeCodingBackend()

    evaluation = evaluate_general_quality(
        model=ModelIdentity(model_id="fixture/model", revision="a" * 40),
        model_artifact_sha256="c" * 64,
        dataset_path=dataset,
        tokenizer_sha256="b" * 64,
        output_path=output,
        state_path=output.parent / "state.json",
        raw_log_dir=output.parent / "raw",
        max_generation_tokens=32,
        random_seed=7,
        backend=backend,
    )
    resumed = evaluate_general_quality(
        model=ModelIdentity(model_id="fixture/model", revision="a" * 40),
        model_artifact_sha256="c" * 64,
        dataset_path=dataset,
        tokenizer_sha256="b" * 64,
        output_path=output,
        state_path=output.parent / "state.json",
        raw_log_dir=output.parent / "raw",
        max_generation_tokens=32,
        random_seed=7,
        backend=backend,
    )

    assert evaluation.profile.value == "general"
    assert backend.generations == 1
    assert evaluation.model_dump(exclude={"created_at"}) == resumed.model_dump(
        exclude={"created_at"}
    )
    assert evaluation.model_artifact_sha256 == "c" * 64
    assert evaluation.evaluation_manifest_sha256 == file_sha256(dataset)
    assert evaluation.dataset_sha256 == file_sha256(dataset)
    assert evaluation.outcomes[0].score == 1.0
    assert evaluation.outcomes[0].scored_tokens == 2
    raw_output = output.parent / (evaluation.outcomes[0].output_file or "")
    assert raw_output.read_text(encoding="utf-8") == "AXQ-MARKER"
    assert file_sha256(raw_output) == evaluation.outcomes[0].output_sha256


def test_general_quality_rejects_raw_outputs_outside_evidence_root(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="beneath the quality evaluation directory"):
        evaluate_general_quality(
            model=ModelIdentity(model_id="fixture/model", revision="a" * 40),
            model_artifact_sha256="c" * 64,
            dataset_path=tmp_path / "not-loaded.jsonl",
            tokenizer_sha256="b" * 64,
            output_path=tmp_path / "evidence" / "general-quality.json",
            state_path=tmp_path / "evidence" / "state.json",
            raw_log_dir=tmp_path / "outside",
            backend=_FakeCodingBackend(),
        )
