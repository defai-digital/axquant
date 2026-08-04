from __future__ import annotations

import ast
import json
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import structlog

from axquant.coding_suite import (
    SANDBOX_PROFILE_SHA256,
    load_coding_payloads,
    probe_toolchains,
)
from axquant.errors import BackendUnavailableError, BenchmarkError
from axquant.quality import MlxQualityBackend, QualityBackend
from axquant.schema import (
    CodingEvaluationState,
    CodingModelOutput,
    CodingScorer,
    CodingSuiteManifest,
    CodingSuiteSelfTestReport,
    CodingTaskManifest,
    CodingTaskPayload,
    DirectQualityEvaluation,
    DirectQualityTaskOutcome,
    ModelIdentity,
    ProfileName,
)
from axquant.serde import file_sha256, load_model, write_data, write_text
from axquant.versioning import collect_versions

log = structlog.get_logger()

_EXCERPT_LIMIT = 4096
_DEFAULT_EXECUTABLES = {
    "python": "python3",
    "node": "node",
    "typescript": "tsc",
    "rust": "rustc",
    "go": "go",
    "sandbox": "sandbox-exec",
}


@dataclass(frozen=True)
class _CommandResult:
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    stdout: bytes
    stderr: bytes
    memory_exceeded: bool = False
    process_limit_exceeded: bool = False
    output_limit_exceeded: bool = False
    infrastructure_error: str | None = None


def _unfenced(value: str) -> str:
    stripped = value.strip()
    match = re.match(
        r"^```(?:[A-Za-z0-9_+.-]+)?[ \t]*(?:\r?\n)?(.*?)(?:\r?\n```(?:\s.*)?|\Z)",
        stripped,
        re.DOTALL,
    )
    return match.group(1).rstrip() if match else stripped


def _seatbelt_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _sandbox_profile(
    *,
    work_dir: Path,
    toolchain_paths: list[Path],
) -> str:
    home = Path.home().resolve()
    allowed_paths = [work_dir, *toolchain_paths]
    metadata_paths = {home}
    for allowed_path in allowed_paths:
        resolved_path = allowed_path.resolve()
        try:
            resolved_path.relative_to(home)
        except ValueError:
            continue
        current = resolved_path
        while current != home:
            metadata_paths.add(current)
            current = current.parent
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny file-write*)",
        f'(allow file-write* (subpath "{_seatbelt_string(str(work_dir))}"))',
        f'(deny file-read* (subpath "{_seatbelt_string(str(home))}"))',
        f'(allow file-read* (subpath "{_seatbelt_string(str(work_dir))}"))',
        '(allow file-read* file-write* (literal "/dev/null"))',
        '(allow file-read* (literal "/dev/urandom"))',
    ]
    for path in sorted(metadata_paths):
        rules.append(f'(allow file-read-metadata (literal "{_seatbelt_string(str(path))}"))')
    for path in sorted({path.resolve() for path in toolchain_paths}):
        rules.append(f'(allow file-read* (subpath "{_seatbelt_string(str(path))}"))')
    return " ".join(rules)


def _limit_process(task: CodingTaskManifest) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (task.cpu_time_seconds, task.cpu_time_seconds + 1))
    if sys.platform != "darwin":
        resource.setrlimit(
            resource.RLIMIT_AS,
            (task.memory_limit_bytes, task.memory_limit_bytes),
        )
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_NPROC, (task.process_limit, task.process_limit))
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (task.file_size_limit_bytes, task.file_size_limit_bytes),
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (task.open_file_limit, task.open_file_limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _process_group_usage(process_group: int) -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pgid=,rss="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    total_kib = 0
    processes = 0
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            pgid, rss_kib = (int(field) for field in fields)
        except ValueError:
            continue
        if pgid == process_group:
            total_kib += rss_kib
            processes += 1
    return total_kib * 1024, processes


def _wait_with_limits(
    process: subprocess.Popen[bytes],
    *,
    wall_seconds: float,
    memory_limit_bytes: int,
    process_limit: int,
    output_limit_bytes: int,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, bool, bool, bool, bool, str | None]:
    deadline = time.monotonic() + wall_seconds
    next_memory_sample = 0.0
    timed_out = False
    memory_exceeded = False
    process_limit_exceeded = False
    output_limit_exceeded = False
    monitor_error: str | None = None
    while process.poll() is None:
        now = time.monotonic()
        if now >= deadline:
            timed_out = True
            break
        output_bytes = sum(
            path.stat().st_size for path in (stdout_path, stderr_path) if path.exists()
        )
        if output_bytes > output_limit_bytes:
            output_limit_exceeded = True
            break
        if sys.platform == "darwin" and now >= next_memory_sample:
            usage = _process_group_usage(process.pid)
            if usage is None:
                if process.poll() is None:
                    monitor_error = "cannot measure scorer process-group resource usage"
                    break
                continue
            rss_bytes, process_count = usage
            if rss_bytes > memory_limit_bytes:
                memory_exceeded = True
                break
            if process_count > process_limit:
                process_limit_exceeded = True
                break
            next_memory_sample = now + 0.1
        time.sleep(0.02)
    if (
        timed_out
        or memory_exceeded
        or process_limit_exceeded
        or output_limit_exceeded
        or monitor_error is not None
    ):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    return (
        process.wait(),
        timed_out,
        memory_exceeded,
        process_limit_exceeded,
        output_limit_exceeded,
        monitor_error,
    )


def _run_command(
    command: list[str],
    *,
    task: CodingTaskManifest,
    work_dir: Path,
    environment: dict[str, str],
    profile: str,
    sandbox_executable: str,
) -> _CommandResult:
    stdout_path = work_dir / "command.stdout"
    stderr_path = work_dir / "command.stderr"
    started = time.monotonic()
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [sandbox_executable, "-p", profile, *command],
                cwd=work_dir,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                start_new_session=True,
                preexec_fn=lambda: _limit_process(task),
            )
            (
                exit_code,
                timed_out,
                memory_exceeded,
                process_limit_exceeded,
                output_limit_exceeded,
                monitor_error,
            ) = _wait_with_limits(
                process,
                wall_seconds=task.timeout_seconds,
                memory_limit_bytes=task.memory_limit_bytes,
                process_limit=task.process_limit,
                output_limit_bytes=task.output_limit_bytes,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return _CommandResult(
            exit_code=None,
            timed_out=False,
            duration_seconds=time.monotonic() - started,
            stdout=b"",
            stderr=str(exc).encode(),
            infrastructure_error=f"cannot launch sandboxed scorer: {exc}",
        )
    stdout_bytes = stdout_path.read_bytes() if stdout_path.exists() else b""
    stderr_bytes = stderr_path.read_bytes() if stderr_path.exists() else b""
    infrastructure_error = monitor_error
    if exit_code in {64, 65, 69, 70, 71, 72, 78} and b"sandbox" in stderr_bytes.lower():
        infrastructure_error = "sandbox initialization failed"
    return _CommandResult(
        exit_code=exit_code,
        timed_out=timed_out,
        duration_seconds=time.monotonic() - started,
        stdout=stdout_bytes,
        stderr=stderr_bytes,
        memory_exceeded=memory_exceeded,
        process_limit_exceeded=process_limit_exceeded,
        output_limit_exceeded=output_limit_exceeded,
        infrastructure_error=infrastructure_error,
    )


def _resolved_executables(overrides: dict[str, str] | None) -> dict[str, str]:
    configured = {**_DEFAULT_EXECUTABLES, **(overrides or {})}
    resolved: dict[str, str] = {}
    for name, executable in configured.items():
        path = shutil.which(executable)
        if path is not None:
            resolved[name] = path
    return resolved


def _required_toolchain(payload: CodingTaskPayload) -> str | None:
    return {
        "python": "python",
        "javascript": "node",
        "typescript": "typescript",
        "rust": "rust",
        "go": "go",
    }.get(payload.language)


def _compile_and_test_commands(
    payload: CodingTaskPayload,
    *,
    fixture_dir: Path,
    output_dir: Path,
    executables: dict[str, str],
) -> tuple[list[list[str]], dict[str, str]]:
    environment_updates: dict[str, str] = {}
    candidate = output_dir / payload.candidate_path
    if payload.language == "python":
        python = executables["python"]
        compile_command = [python, "-B", "-m", "py_compile", str(candidate)]
        if payload.scorer is CodingScorer.COMPILE:
            return [compile_command], environment_updates
        assert payload.test_path is not None
        environment_updates["PYTHONPATH"] = os.pathsep.join([str(output_dir), str(fixture_dir)])
        return [compile_command, [python, "-B", str(fixture_dir / payload.test_path)]], (
            environment_updates
        )
    if payload.language == "javascript":
        node = executables["node"]
        compile_command = [node, "--check", str(candidate)]
        if payload.scorer is CodingScorer.COMPILE:
            return [compile_command], environment_updates
        assert payload.test_path is not None
        environment_updates["AXQ_CANDIDATE_PATH"] = str(candidate)
        return [compile_command, [node, str(fixture_dir / payload.test_path)]], (
            environment_updates
        )
    if payload.language == "typescript":
        compiler = executables["typescript"]
        copied_fixtures: list[str] = []
        for relative_path, content in payload.fixture_files.items():
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
            copied_fixtures.append(str(destination))
        command = [
            compiler,
            "--strict",
            "--noEmit",
            "--module",
            "commonjs",
            "--target",
            "es2020",
            str(candidate),
            *copied_fixtures,
        ]
        return [command], environment_updates
    if payload.language == "rust":
        compiler = executables["rust"]
        assert payload.test_path is not None
        harness = output_dir / "axquant_harness.rs"
        harness.write_text(
            candidate.read_text(encoding="utf-8")
            + "\n"
            + (fixture_dir / payload.test_path).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        test_binary = output_dir / "axquant-rust-tests"
        compile_command = [
            compiler,
            "--edition",
            "2021",
            "--test",
            str(harness),
            "-o",
            str(test_binary),
        ]
        return [compile_command, [str(test_binary)]], environment_updates
    if payload.language == "go":
        go = executables["go"]
        assert payload.test_path is not None
        test_copy = output_dir / payload.test_path
        test_copy.parent.mkdir(parents=True, exist_ok=True)
        test_copy.write_text(
            (fixture_dir / payload.test_path).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (output_dir / "go.mod").write_text("module axquant.task\n\ngo 1.23\n", encoding="utf-8")
        environment_updates.update(
            {
                "GOCACHE": str(output_dir / ".gocache"),
                "GOPATH": str(output_dir / ".gopath"),
                "GOMODCACHE": str(output_dir / ".gomodcache"),
                "GOENV": "off",
                "GOROOT": str(Path(go).resolve().parent.parent),
                "GOTELEMETRY": "off",
            }
        )
        compile_command = [go, "test", "-p=4", "-run", "^$", str(candidate), str(test_copy)]
        test_command = [go, "test", "-p=4", str(candidate), str(test_copy)]
        return [compile_command, test_command], environment_updates
    raise BenchmarkError(f"unsupported executable coding language: {payload.language}")


def _write_raw_logs(
    *,
    raw_log_dir: Path,
    task_id: str,
    stdout: bytes,
    stderr: bytes,
    evidence_root: Path | None = None,
) -> tuple[str, str, str, str]:
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_name = f"{task_id}.stdout.txt"
    stderr_name = f"{task_id}.stderr.txt"
    stdout_path = raw_log_dir / stdout_name
    stderr_path = raw_log_dir / stderr_name
    write_text(stdout_path, stdout.decode("utf-8", errors="replace"))
    write_text(stderr_path, stderr.decode("utf-8", errors="replace"))
    root = (evidence_root or raw_log_dir).resolve()
    try:
        stdout_name = stdout_path.resolve().relative_to(root).as_posix()
        stderr_name = stderr_path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise BenchmarkError("raw coding logs must be inside the evidence root") from exc
    return stdout_name, stderr_name, file_sha256(stdout_path), file_sha256(stderr_path)


def _write_model_output(
    *,
    raw_log_dir: Path,
    task_id: str,
    output: str,
    evidence_root: Path | None = None,
) -> tuple[str, str]:
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_log_dir / f"{task_id}.model-output.txt"
    write_text(output_path, output)
    root = (evidence_root or raw_log_dir).resolve()
    try:
        output_name = output_path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise BenchmarkError("raw coding outputs must be inside the evidence root") from exc
    return output_name, file_sha256(output_path)


def _non_executable_score(
    payload: CodingTaskPayload,
    output: str,
) -> tuple[float, bool | None, bool | None]:
    unfenced = _unfenced(output)
    if payload.scorer is CodingScorer.AST:
        if payload.language != "python":
            raise BenchmarkError("AST scorer currently requires Python")
        try:
            ast.parse(unfenced)
            return 1.0, True, None
        except SyntaxError:
            return 0.0, False, None
    if payload.scorer in {CodingScorer.JSON_SCHEMA, CodingScorer.TOOL_EXACT}:
        try:
            value = json.loads(unfenced)
        except json.JSONDecodeError:
            return 0.0, None, False
        if payload.scorer is CodingScorer.TOOL_EXACT:
            valid = value == payload.expected_json
        else:
            valid = isinstance(value, dict) and set(payload.json_required_keys) <= set(value)
        return float(valid), None, valid
    if payload.scorer is CodingScorer.TEXT_EXACT:
        valid = unfenced.strip() == (payload.expected_text or "").strip()
        return float(valid), None, None
    raise BenchmarkError(f"unsupported non-executable scorer: {payload.scorer.value}")


def score_coding_task(
    *,
    task: CodingTaskManifest,
    payload: CodingTaskPayload,
    model_output: CodingModelOutput,
    raw_log_dir: Path,
    work_root: Path,
    executable_overrides: dict[str, str] | None = None,
    evidence_root: Path | None = None,
) -> DirectQualityTaskOutcome:
    output_file, output_sha256 = _write_model_output(
        raw_log_dir=raw_log_dir,
        task_id=task.task_id,
        output=model_output.output,
        evidence_root=evidence_root,
    )
    empty_stdout = b""
    empty_stderr = b""
    if model_output.model_error is not None:
        stdout_file, stderr_file, stdout_sha, stderr_sha = _write_raw_logs(
            raw_log_dir=raw_log_dir,
            task_id=task.task_id,
            stdout=empty_stdout,
            stderr=model_output.model_error.encode(),
            evidence_root=evidence_root,
        )
        return DirectQualityTaskOutcome(
            task_id=task.task_id,
            score=0.0,
            scored_tokens=model_output.generated_tokens,
            scorer=task.scorer,
            model_error=True,
            output_file=output_file,
            output_sha256=output_sha256,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
            stderr_excerpt=model_output.model_error[:_EXCERPT_LIMIT],
        )
    if task.scorer not in {CodingScorer.UNIT_TEST, CodingScorer.COMPILE}:
        score, syntax_valid, tool_valid = _non_executable_score(payload, model_output.output)
        stdout_file, stderr_file, stdout_sha, stderr_sha = _write_raw_logs(
            raw_log_dir=raw_log_dir,
            task_id=task.task_id,
            stdout=empty_stdout,
            stderr=empty_stderr,
            evidence_root=evidence_root,
        )
        return DirectQualityTaskOutcome(
            task_id=task.task_id,
            score=score,
            scored_tokens=model_output.generated_tokens,
            scorer=task.scorer,
            syntax_valid=syntax_valid,
            tool_valid=tool_valid,
            output_file=output_file,
            output_sha256=output_sha256,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
        )

    executables = _resolved_executables(executable_overrides)
    required = _required_toolchain(payload)
    sandbox = executables.get("sandbox")
    if required is None or required not in executables or sandbox is None:
        missing = required if required not in executables else "sandbox"
        message = f"required coding scorer toolchain is unavailable: {missing}"
        stdout_file, stderr_file, stdout_sha, stderr_sha = _write_raw_logs(
            raw_log_dir=raw_log_dir,
            task_id=task.task_id,
            stdout=empty_stdout,
            stderr=message.encode(),
            evidence_root=evidence_root,
        )
        return DirectQualityTaskOutcome(
            task_id=task.task_id,
            score=0.0,
            scored_tokens=model_output.generated_tokens,
            scorer=task.scorer,
            infrastructure_error=True,
            output_file=output_file,
            output_sha256=output_sha256,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
            stderr_excerpt=message,
        )

    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{task.task_id}-", dir=work_root) as temporary:
        workspace = Path(temporary).resolve()
        fixture_dir = workspace / "fixture"
        output_dir = workspace / "output"
        fixture_dir.mkdir()
        output_dir.mkdir()
        for relative_path, content in payload.fixture_files.items():
            fixture = fixture_dir / relative_path
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_text(content, encoding="utf-8")
            fixture.chmod(0o444)
        candidate = output_dir / payload.candidate_path
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(_unfenced(model_output.output), encoding="utf-8")
        commands, environment_updates = _compile_and_test_commands(
            payload,
            fixture_dir=fixture_dir,
            output_dir=output_dir,
            executables=executables,
        )
        environment = {
            "PATH": os.pathsep.join(
                sorted({str(Path(path).parent) for path in executables.values()})
            ),
            "HOME": str(output_dir / "home"),
            "TMPDIR": str(output_dir / "tmp"),
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "SOURCE_DATE_EPOCH": "0",
            **environment_updates,
        }
        Path(environment["HOME"]).mkdir()
        Path(environment["TMPDIR"]).mkdir()
        toolchain_paths = [Path(path).parent.parent for path in executables.values()]
        profile = _sandbox_profile(work_dir=workspace, toolchain_paths=toolchain_paths)
        results: list[_CommandResult] = []
        for command in commands:
            result = _run_command(
                command,
                task=task,
                work_dir=workspace,
                environment=environment,
                profile=profile,
                sandbox_executable=sandbox,
            )
            results.append(result)
            if result.exit_code != 0 or result.timed_out or result.infrastructure_error:
                break

    stdout = b"\n".join(result.stdout for result in results)
    stderr = b"\n".join(result.stderr for result in results)
    stdout_file, stderr_file, stdout_sha, stderr_sha = _write_raw_logs(
        raw_log_dir=raw_log_dir,
        task_id=task.task_id,
        stdout=stdout,
        stderr=stderr,
        evidence_root=evidence_root,
    )
    compile_passed = bool(
        results
        and results[0].exit_code == 0
        and not results[0].timed_out
        and not results[0].memory_exceeded
        and not results[0].process_limit_exceeded
        and not results[0].output_limit_exceeded
        and results[0].infrastructure_error is None
    )
    all_passed = bool(
        len(results) == len(commands)
        and all(
            result.exit_code == 0
            and not result.timed_out
            and not result.memory_exceeded
            and not result.process_limit_exceeded
            and not result.output_limit_exceeded
            and result.infrastructure_error is None
            for result in results
        )
    )
    infrastructure_error = next(
        (result.infrastructure_error for result in results if result.infrastructure_error),
        None,
    )
    return DirectQualityTaskOutcome(
        task_id=task.task_id,
        score=float(all_passed),
        scored_tokens=model_output.generated_tokens,
        scorer=task.scorer,
        syntax_valid=compile_passed,
        unit_tests_passed=(all_passed if task.scorer is CodingScorer.UNIT_TEST else None),
        infrastructure_error=infrastructure_error is not None,
        output_file=output_file,
        output_sha256=output_sha256,
        sandboxed=True,
        network_disabled=True,
        timed_out=any(result.timed_out for result in results),
        exit_code=results[-1].exit_code if results else None,
        duration_seconds=sum(result.duration_seconds for result in results),
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        stdout_sha256=stdout_sha,
        stderr_sha256=stderr_sha,
        stdout_excerpt=stdout.decode("utf-8", errors="replace")[:_EXCERPT_LIMIT],
        stderr_excerpt=(
            infrastructure_error
            or (
                "memory limit exceeded" if any(result.memory_exceeded for result in results) else ""
            )
            or (
                "process limit exceeded"
                if any(result.process_limit_exceeded for result in results)
                else ""
            )
            or (
                "output limit exceeded"
                if any(result.output_limit_exceeded for result in results)
                else ""
            )
            or stderr.decode("utf-8", errors="replace")[:_EXCERPT_LIMIT]
        ),
        toolchain=probe_toolchains(executable_overrides).get(required),
        sandbox_profile_sha256=SANDBOX_PROFILE_SHA256,
    )


def _load_or_create_state(
    *,
    state_path: Path,
    suite_manifest_sha256: str,
    model: ModelIdentity,
    model_artifact_sha256: str,
    tokenizer_sha256: str,
    random_seed: int,
    max_sequence_length: int,
) -> CodingEvaluationState:
    expected = CodingEvaluationState(
        suite_manifest_sha256=suite_manifest_sha256,
        model=model,
        model_artifact_sha256=model_artifact_sha256,
        tokenizer_sha256=tokenizer_sha256,
        random_seed=random_seed,
        max_sequence_length=max_sequence_length,
    )
    if not state_path.exists():
        return expected
    state = load_model(state_path, CodingEvaluationState)
    if (
        state.suite_manifest_sha256 != expected.suite_manifest_sha256
        or state.model != expected.model
        or state.model_artifact_sha256 != expected.model_artifact_sha256
        or state.tokenizer_sha256 != expected.tokenizer_sha256
        or state.random_seed != expected.random_seed
        or state.max_sequence_length != expected.max_sequence_length
    ):
        raise BenchmarkError("coding evaluation state does not match the requested run")
    return state


def evaluate_coding_suite(
    *,
    model: ModelIdentity,
    model_artifact_sha256: str,
    manifest_path: str | Path,
    tokenizer_sha256: str,
    output_path: str | Path,
    state_path: str | Path,
    raw_log_dir: str | Path,
    work_root: str | Path,
    max_sequence_length: int = 4096,
    random_seed: int = 20260803,
    backend: QualityBackend | None = None,
    executable_overrides: dict[str, str] | None = None,
) -> DirectQualityEvaluation:
    manifest_file = Path(manifest_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    raw_logs = Path(raw_log_dir).expanduser().resolve()
    try:
        raw_logs.relative_to(output_file.parent)
    except ValueError as exc:
        raise BenchmarkError(
            "raw coding logs must be stored beneath the quality evaluation directory"
        ) from exc
    manifest = load_model(manifest_file, CodingSuiteManifest)
    payloads = load_coding_payloads(manifest_file, manifest)
    task_by_id = {task.task_id: task for task in manifest.tasks}
    current_toolchains = probe_toolchains(executable_overrides)
    required_names = {
        required for payload in payloads if (required := _required_toolchain(payload)) is not None
    } | {"sandbox"}
    for name in required_names:
        if manifest.toolchains.get(name) != current_toolchains.get(name):
            raise BackendUnavailableError(
                f"coding suite toolchain identity differs for {name}: "
                f"expected {manifest.toolchains.get(name)!r}, got {current_toolchains.get(name)!r}"
            )
    state_file = Path(state_path).expanduser().resolve()
    state = _load_or_create_state(
        state_path=state_file,
        suite_manifest_sha256=file_sha256(manifest_file),
        model=model,
        model_artifact_sha256=model_artifact_sha256,
        tokenizer_sha256=tokenizer_sha256,
        random_seed=random_seed,
        max_sequence_length=max_sequence_length,
    )
    active_backend = backend or MlxQualityBackend()
    active_backend.load_model(model.local_path or model.model_id, model.revision)
    prompt_format, chat_template_sha256 = active_backend.generation_metadata()
    completed = {output.task_id: output for output in state.outputs}
    unexpected_output_ids = set(completed) - set(task_by_id)
    if unexpected_output_ids:
        raise BenchmarkError(
            f"coding evaluation state contains unexpected task IDs: {sorted(unexpected_output_ids)}"
        )
    for index, payload in enumerate(payloads):
        if payload.task_id in completed:
            continue
        loss_text = f"{payload.prompt}\n{payload.reference or ''}"
        loss, perplexity_tokens = active_backend.perplexity_loss(
            loss_text,
            max_sequence_length,
        )
        try:
            output = active_backend.generate(
                payload.prompt,
                payload.target_tokens,
                random_seed + index,
            )
            generated_tokens = active_backend.count_tokens(output)
            model_error = None
        except (BenchmarkError, RuntimeError, ValueError) as exc:
            output = ""
            generated_tokens = 0
            model_error = str(exc)
        completed[payload.task_id] = CodingModelOutput(
            task_id=payload.task_id,
            output=output,
            generated_tokens=generated_tokens,
            perplexity_loss=loss,
            perplexity_tokens=perplexity_tokens,
            model_error=model_error,
        )
        state = state.model_copy(
            update={
                "outputs": [
                    completed[item.task_id] for item in manifest.tasks if item.task_id in completed
                ]
            }
        )
        write_data(state_file, state)
        log.info(
            "coding_generation_completed",
            task_id=payload.task_id,
            task=index + 1,
            tasks=len(payloads),
            generated_tokens=generated_tokens,
            model_error=model_error,
        )
    ordered_outputs = [completed[task.task_id] for task in manifest.tasks]
    total_loss = sum(output.perplexity_loss for output in ordered_outputs)
    total_perplexity_tokens = sum(output.perplexity_tokens for output in ordered_outputs)
    if total_perplexity_tokens == 0:
        raise BenchmarkError("coding evaluation produced no perplexity tokens")
    sandbox_root = Path(work_root).expanduser().resolve()
    outcomes = [
        score_coding_task(
            task=task_by_id[payload.task_id],
            payload=payload,
            model_output=completed[payload.task_id],
            raw_log_dir=raw_logs,
            work_root=sandbox_root,
            executable_overrides=executable_overrides,
            evidence_root=output_file.parent,
        )
        for payload in payloads
    ]
    evaluation = DirectQualityEvaluation(
        profile=ProfileName.AGENT_CODING,
        model=model,
        model_artifact_sha256=model_artifact_sha256,
        evaluation_manifest_sha256=file_sha256(manifest_file),
        dataset_sha256=manifest.dataset_sha256,
        tokenizer_sha256=tokenizer_sha256,
        generation={
            "prompt_format": prompt_format,
            "chat_template_sha256": chat_template_sha256,
            "thinking_enabled": False,
            "max_sequence_length": max_sequence_length,
            "max_generation_tokens": max(task.target_tokens for task in manifest.tasks),
        },
        random_seed=random_seed,
        evaluated_tokens=total_perplexity_tokens,
        software_versions=collect_versions(),
        perplexity=math.exp(total_loss / total_perplexity_tokens),
        outcomes=outcomes,
    )
    write_data(output_file, evaluation)
    write_data(state_file, state.model_copy(update={"completed": True}))
    return evaluation


def verify_coding_suite(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    raw_log_dir: str | Path,
    work_root: str | Path,
    executable_overrides: dict[str, str] | None = None,
) -> CodingSuiteSelfTestReport:
    manifest_file = Path(manifest_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    raw_logs = Path(raw_log_dir).expanduser().resolve()
    try:
        raw_logs.relative_to(output_file.parent)
    except ValueError as exc:
        raise BenchmarkError(
            "raw coding self-test logs must be stored beneath the report directory"
        ) from exc
    manifest = load_model(manifest_file, CodingSuiteManifest)
    payloads = load_coding_payloads(manifest_file, manifest)
    tasks = {task.task_id: task for task in manifest.tasks}
    current_toolchains = probe_toolchains(executable_overrides)
    if current_toolchains != manifest.toolchains:
        raise BackendUnavailableError(
            "coding suite self-test toolchain identities differ from the frozen manifest"
        )
    sandbox_root = Path(work_root).expanduser().resolve()

    def score_phase(
        phase: str,
        output_for: Callable[[CodingTaskPayload], str],
    ) -> list[DirectQualityTaskOutcome]:
        outcomes: list[DirectQualityTaskOutcome] = []
        for payload in payloads:
            generated = output_for(payload)
            outcomes.append(
                score_coding_task(
                    task=tasks[payload.task_id],
                    payload=payload,
                    model_output=CodingModelOutput(
                        task_id=payload.task_id,
                        output=generated,
                        generated_tokens=0,
                        perplexity_loss=0.0,
                        perplexity_tokens=0,
                    ),
                    raw_log_dir=raw_logs / phase,
                    work_root=sandbox_root / phase,
                    executable_overrides=executable_overrides,
                    evidence_root=output_file.parent,
                )
            )
        return outcomes

    oracle_outcomes = score_phase("oracle", lambda payload: payload.reference or "")
    mutant_outcomes = score_phase("empty-mutant", lambda _payload: "")
    issues: list[str] = []
    for oracle in oracle_outcomes:
        if oracle.infrastructure_error:
            issues.append(f"oracle scorer infrastructure failed: {oracle.task_id}")
        elif oracle.model_error or oracle.score != 1.0:
            issues.append(f"reference oracle did not pass: {oracle.task_id}")
    for mutant in mutant_outcomes:
        if mutant.infrastructure_error:
            issues.append(f"mutant scorer infrastructure failed: {mutant.task_id}")
        elif mutant.model_error or mutant.score != 0.0:
            issues.append(f"empty mutant was not rejected: {mutant.task_id}")
    report = CodingSuiteSelfTestReport(
        suite_manifest_sha256=file_sha256(manifest_file),
        toolchains=current_toolchains,
        sandbox_profile_sha256=SANDBOX_PROFILE_SHA256,
        oracle_outcomes=oracle_outcomes,
        empty_mutant_outcomes=mutant_outcomes,
        passed=not issues,
        issues=issues,
    )
    write_data(output_file, report)
    return report
