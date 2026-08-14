from __future__ import annotations

import ast
import json
import math
import os
import re
import resource
import secrets
import selectors
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
from typing import IO, Any

import structlog

from axquant.coding_suite import (
    load_coding_payloads,
    probe_toolchains,
)
from axquant.errors import BackendUnavailableError, BenchmarkError
from axquant.identity import same_model_identity
from axquant.quality import MlxQualityBackend, QualityBackend
from axquant.sandbox_policy import (
    SANDBOX_PROFILE_SHA256,
    TRUSTED_SANDBOX_EXECUTABLE,
    render_sandbox_profile,
)
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
    "sandbox": "/usr/bin/sandbox-exec",
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
    completion_evidence_missing: bool = False
    infrastructure_error: str | None = None


@dataclass(frozen=True)
class _CommandSpec:
    argv: tuple[str, ...]
    completion_markers: tuple[bytes, ...] = ()
    allow_subprocesses: bool = False


def _unfenced(value: str) -> str:
    stripped = value.strip()
    match = re.match(
        r"^```(?:[A-Za-z0-9_+.-]+)?[ \t]*(?:\r?\n)?(.*?)(?:\r?\n```(?:\s.*)?|\Z)",
        stripped,
        re.DOTALL,
    )
    return match.group(1).rstrip() if match else stripped


def _sandbox_profile(
    *,
    input_dir: Path,
    output_dir: Path,
    toolchain_paths: list[Path],
    command: _CommandSpec,
) -> str:
    return render_sandbox_profile(
        input_dir=input_dir,
        output_dir=output_dir,
        toolchain_paths=toolchain_paths,
        entrypoint=Path(command.argv[0]),
        allow_subprocesses=command.allow_subprocesses,
    )


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
    stdout_pipe: IO[Any],
    stderr_pipe: IO[Any],
) -> tuple[int, bool, bool, bool, bool, str | None, bytes, bytes]:
    deadline = time.monotonic() + wall_seconds
    next_memory_sample = 0.0
    timed_out = False
    memory_exceeded = False
    process_limit_exceeded = False
    output_limit_exceeded = False
    monitor_error: str | None = None
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    total_output_bytes = 0
    selector = selectors.DefaultSelector()
    selector.register(stdout_pipe, selectors.EVENT_READ, stdout_buffer)
    selector.register(stderr_pipe, selectors.EVENT_READ, stderr_buffer)

    def read_ready_streams(timeout: float) -> None:
        nonlocal monitor_error, output_limit_exceeded, total_output_bytes
        try:
            events = selector.select(timeout)
        except OSError as exc:
            monitor_error = f"cannot monitor scorer output: {exc}"
            return
        for key, _mask in events:
            try:
                chunk = os.read(key.fd, 64 * 1024)
            except BlockingIOError:
                continue
            except OSError as exc:
                monitor_error = f"cannot read scorer output: {exc}"
                with suppress(Exception):
                    selector.unregister(key.fileobj)
                continue
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            remaining = max(0, output_limit_bytes + 1 - total_output_bytes)
            if remaining:
                key.data.extend(chunk[:remaining])
            total_output_bytes += len(chunk)
            if total_output_bytes > output_limit_bytes:
                output_limit_exceeded = True

    for stream in (stdout_pipe, stderr_pipe):
        os.set_blocking(stream.fileno(), False)
    while process.poll() is None:
        now = time.monotonic()
        if now >= deadline:
            timed_out = True
            break
        read_ready_streams(min(0.02, max(0.0, deadline - now)))
        if output_limit_exceeded or monitor_error is not None:
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
    if (
        timed_out
        or memory_exceeded
        or process_limit_exceeded
        or output_limit_exceeded
        or monitor_error is not None
    ):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    exit_code = process.wait()
    # A compiler must not leave a background descendant holding either capture pipe open.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    drain_deadline = time.monotonic() + 1.0
    while selector.get_map() and time.monotonic() < drain_deadline:
        read_ready_streams(0.02)
    if selector.get_map() and monitor_error is None:
        monitor_error = "cannot finish reading scorer output"
    selector.close()
    return (
        exit_code,
        timed_out,
        memory_exceeded,
        process_limit_exceeded,
        output_limit_exceeded,
        monitor_error,
        bytes(stdout_buffer),
        bytes(stderr_buffer),
    )


def _run_command(
    command: _CommandSpec,
    *,
    task: CodingTaskManifest,
    work_dir: Path,
    environment: dict[str, str],
    profile: str,
    sandbox_executable: str,
    output_limit_bytes: int,
) -> _CommandResult:
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [sandbox_executable, "-p", profile, *command.argv],
            cwd=work_dir,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
            preexec_fn=lambda: _limit_process(task),
        )
        if process.stdout is None or process.stderr is None:
            raise OSError("cannot create scorer output pipes")
        (
            exit_code,
            timed_out,
            memory_exceeded,
            process_limit_exceeded,
            output_limit_exceeded,
            monitor_error,
            stdout_bytes,
            stderr_bytes,
        ) = _wait_with_limits(
            process,
            wall_seconds=task.timeout_seconds,
            memory_limit_bytes=task.memory_limit_bytes,
            process_limit=task.process_limit,
            output_limit_bytes=output_limit_bytes,
            stdout_pipe=process.stdout,
            stderr_pipe=process.stderr,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return _CommandResult(
            exit_code=None,
            timed_out=False,
            duration_seconds=time.monotonic() - started,
            stdout=b"",
            stderr=str(exc).encode(),
            infrastructure_error=f"cannot launch sandboxed scorer: {exc}",
        )
    finally:
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
    infrastructure_error = monitor_error
    if exit_code in {64, 65, 69, 70, 71, 72, 78} and b"sandbox" in stderr_bytes.lower():
        infrastructure_error = "sandbox initialization failed"
    completion_evidence_missing = exit_code == 0 and any(
        marker not in stdout_bytes for marker in command.completion_markers
    )
    return _CommandResult(
        exit_code=exit_code,
        timed_out=timed_out,
        duration_seconds=time.monotonic() - started,
        stdout=stdout_bytes,
        stderr=stderr_bytes,
        memory_exceeded=memory_exceeded,
        process_limit_exceeded=process_limit_exceeded,
        output_limit_exceeded=output_limit_exceeded,
        completion_evidence_missing=completion_evidence_missing,
        infrastructure_error=infrastructure_error,
    )


def _resolved_executables(overrides: dict[str, str] | None) -> dict[str, str]:
    configured = {**_DEFAULT_EXECUTABLES, **(overrides or {})}
    resolved: dict[str, str] = {}
    for name, executable in configured.items():
        path = shutil.which(executable)
        if path is not None:
            invocation_path = Path(path).absolute()
            runtime_path = invocation_path
            if name == "python" and sys.platform == "darwin":
                resolved_path = invocation_path.resolve()
                runtime_path = resolved_path
                app_binary = (
                    resolved_path.parent.parent
                    / "Resources"
                    / "Python.app"
                    / "Contents"
                    / "MacOS"
                    / "Python"
                )
                if app_binary.is_file():
                    runtime_path = app_binary
            resolved[name] = str(runtime_path)
    return resolved


def _required_toolchain(payload: CodingTaskPayload) -> str | None:
    return {
        "python": "python",
        "javascript": "node",
        "typescript": "typescript",
        "rust": "rust",
        "go": "go",
    }.get(payload.language)


def _native_toolchain_executable(name: str, executable: str) -> str:
    query = {
        "rust": ("--print", "sysroot"),
        "go": ("env", "GOROOT"),
    }.get(name)
    binary_name = {"rust": "rustc", "go": "go"}.get(name)
    if query is None or binary_name is None:
        return executable
    try:
        result = subprocess.run(
            [executable, *query],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BenchmarkError(f"cannot resolve the {name} toolchain root: {exc}") from exc
    root = Path(result.stdout.strip()).expanduser().resolve()
    runtime = root / "bin" / binary_name
    if not root.is_dir() or not runtime.is_file():
        raise BenchmarkError(f"{name} reported an invalid toolchain root: {root}")
    return str(runtime)


def _toolchain_read_root(executable: str) -> Path:
    raw_path = Path(executable).expanduser().absolute()
    resolved_path = raw_path.resolve()
    for prefix in (Path("/opt/homebrew"), Path("/usr/local")):
        try:
            resolved_path.relative_to(prefix)
        except ValueError:
            continue
        return prefix
    for ancestor in resolved_path.parents:
        if ancestor.parent.name == "Versions" and ancestor.parent.parent.name.endswith(
            ".framework"
        ):
            # Framework executables (notably GitHub Actions' Python.app) load a sibling
            # library from the version root rather than from the app bundle itself.
            return ancestor
    root = resolved_path.parent.parent
    home = Path.home().resolve()
    if root in {Path("/"), Path("/Users"), home}:
        return resolved_path.parent
    return root


def _darwin_developer_tools() -> tuple[list[Path], dict[str, str], Path | None]:
    if sys.platform != "darwin":
        return [], {}, None
    try:
        developer = Path(
            subprocess.run(
                ["/usr/bin/xcode-select", "-p"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        ).resolve()
        sdk = Path(
            subprocess.run(
                ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        ).resolve()
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return [], {}, None
    xcode_contents = next(
        (path for path in (developer, *developer.parents) if path.name == "Contents"), developer
    )
    clang = developer / "Toolchains" / "XcodeDefault.xctoolchain" / "usr" / "bin" / "clang"
    return (
        [xcode_contents, sdk],
        {"DEVELOPER_DIR": str(developer), "SDKROOT": str(sdk)},
        clang if clang.is_file() else None,
    )


def _sandbox_runtime_paths(
    executables: dict[str, str],
) -> tuple[list[Path], dict[str, str], Path | None]:
    roots = {_toolchain_read_root(path) for name, path in executables.items() if name != "sandbox"}
    developer_paths, developer_environment, rust_linker = _darwin_developer_tools()
    return sorted({*roots, *developer_paths}), developer_environment, rust_linker


def _seal_input_tree(root: Path) -> None:
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise BenchmarkError("coding sandbox input tree cannot contain symbolic links")
    for path in paths:
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((path for path in paths if path.is_dir()), reverse=True):
        path.chmod(0o555)
    root.chmod(0o555)


def _unseal_input_tree(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(0o700)
    paths = list(root.rglob("*"))
    for path in (path for path in paths if path.is_dir() and not path.is_symlink()):
        path.chmod(0o700)
    for path in (path for path in paths if path.is_file() and not path.is_symlink()):
        path.chmod(0o600)


def _compile_and_test_commands(
    payload: CodingTaskPayload,
    *,
    source_dir: Path,
    output_dir: Path,
    executables: dict[str, str],
    completion_token: str,
    rust_linker: Path | None = None,
) -> tuple[list[_CommandSpec], dict[str, str]]:
    environment_updates: dict[str, str] = {}
    candidate = source_dir / payload.candidate_path
    completion_line = f"AXQUANT_COMPLETION:{completion_token}"

    def command(
        *argv: str,
        completion_markers: tuple[bytes, ...] = (),
        allow_subprocesses: bool = False,
    ) -> _CommandSpec:
        return _CommandSpec(
            argv=tuple(argv),
            completion_markers=completion_markers,
            allow_subprocesses=allow_subprocesses,
        )

    def required_test_path() -> str:
        if payload.test_path is None:
            raise BenchmarkError(f"{payload.language} unit-test task is missing test_path")
        return payload.test_path

    if payload.language == "python":
        python = executables["python"]
        environment_updates["PYTHONPYCACHEPREFIX"] = str(output_dir / ".pycache")
        compile_command = command(python, "-B", "-m", "py_compile", str(candidate))
        if payload.scorer is CodingScorer.COMPILE:
            return [compile_command], environment_updates
        test_path = required_test_path()
        runner = source_dir / test_path
        runner.write_text(
            runner.read_text(encoding="utf-8") + f"\nprint({completion_line!r})\n",
            encoding="utf-8",
        )
        environment_updates["PYTHONPATH"] = str(source_dir)
        test_command = command(
            python,
            "-B",
            str(runner),
            completion_markers=(completion_line.encode(),),
        )
        return [compile_command, test_command], environment_updates
    if payload.language == "javascript":
        node = executables["node"]
        compile_command = command(node, "--check", str(candidate))
        if payload.scorer is CodingScorer.COMPILE:
            return [compile_command], environment_updates
        test_path = required_test_path()
        runner = source_dir / test_path
        runner.write_text(
            runner.read_text(encoding="utf-8")
            + f"\nprocess.stdout.write({json.dumps(completion_line + chr(10))});\n",
            encoding="utf-8",
        )
        environment_updates["AXQ_CANDIDATE_PATH"] = str(candidate)
        test_command = command(
            node,
            str(runner),
            completion_markers=(completion_line.encode(),),
        )
        return [compile_command, test_command], environment_updates
    if payload.language == "typescript":
        compiler = executables["typescript"]
        fixtures = [str(source_dir / relative_path) for relative_path in payload.fixture_files]
        return [
            command(
                compiler,
                "--strict",
                "--noEmit",
                "--module",
                "commonjs",
                "--target",
                "es2020",
                str(candidate),
                *fixtures,
                allow_subprocesses=True,
            )
        ], environment_updates
    if payload.language == "rust":
        compiler = executables["rust"]
        if payload.scorer is CodingScorer.COMPILE:
            library = output_dir / "libaxquant_candidate.rlib"
            return [
                command(
                    compiler,
                    "--edition",
                    "2021",
                    "--crate-type",
                    "lib",
                    str(candidate),
                    "-o",
                    str(library),
                    allow_subprocesses=True,
                )
            ], environment_updates
        test_path = required_test_path()
        harness = source_dir / "axquant_harness.rs"
        completion_test = f"axquant_completion_{completion_token}"
        harness.write_text(
            candidate.read_text(encoding="utf-8")
            + "\n"
            + (source_dir / test_path).read_text(encoding="utf-8")
            + "\n#[cfg(test)]\n"
            + "mod zzz_axquant_completion {\n"
            + "    #[test]\n"
            + f"    fn {completion_test}() {{}}\n"
            + "}\n",
            encoding="utf-8",
        )
        test_binary = output_dir / "axquant-rust-tests"
        compile_argv = [
            compiler,
            "--edition",
            "2021",
            "--test",
            str(harness),
            "-o",
            str(test_binary),
        ]
        if rust_linker is not None:
            compile_argv.extend(["-C", f"linker={rust_linker}"])
        expected = f"test zzz_axquant_completion::{completion_test} ... ok".encode()
        return [
            command(*compile_argv, allow_subprocesses=True),
            command(
                str(test_binary),
                completion_markers=(expected, b"test result: ok."),
            ),
        ], environment_updates
    if payload.language == "go":
        go = executables["go"]
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
        if payload.scorer is CodingScorer.COMPILE:
            return [
                command(
                    go,
                    "test",
                    "-c",
                    "-o",
                    str(output_dir / "axquant-go-compile"),
                    str(candidate),
                    allow_subprocesses=True,
                )
            ], environment_updates
        test_path = required_test_path()
        test_file = source_dir / test_path
        test_names = tuple(
            sorted(
                set(re.findall(r"(?m)^\s*func\s+(Test[A-Za-z0-9_]+)\s*\(", test_file.read_text()))
            )
        )
        if not test_names:
            raise BenchmarkError("Go unit-test task contains no discoverable Test functions")
        completion_test = f"TestZZZAXQuantCompletion{completion_token}"
        completion_file = source_dir / "zzzz_axquant_completion_test.go"
        completion_file.write_text(
            f'package candidate\n\nimport "testing"\n\nfunc {completion_test}(t *testing.T) {{}}\n',
            encoding="utf-8",
        )
        test_binary = output_dir / "axquant-go-tests"
        sources = [str(candidate), str(test_file), str(completion_file)]
        compile_command = command(
            go,
            "test",
            "-c",
            "-o",
            str(test_binary),
            *sources,
            allow_subprocesses=True,
        )
        markers = (
            *(f"--- PASS: {name}".encode() for name in (*test_names, completion_test)),
            b"PASS",
        )
        test_command = command(
            str(test_binary),
            "-test.v",
            completion_markers=markers,
        )
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
    if raw_log_dir.is_symlink():
        raise BenchmarkError("raw coding log directory cannot be a symbolic link")
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    stdout_name = f"{task_id}.stdout.txt"
    stderr_name = f"{task_id}.stderr.txt"
    stdout_path = raw_log_dir / stdout_name
    stderr_path = raw_log_dir / stderr_name
    root = (evidence_root or raw_log_dir).resolve()
    for path in (stdout_path, stderr_path):
        if path.is_symlink():
            raise BenchmarkError("raw coding log file cannot be a symbolic link")
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise BenchmarkError("raw coding logs must be inside the evidence root") from exc
    write_text(stdout_path, stdout.decode("utf-8", errors="replace"))
    write_text(stderr_path, stderr.decode("utf-8", errors="replace"))
    stdout_name = stdout_path.resolve().relative_to(root).as_posix()
    stderr_name = stderr_path.resolve().relative_to(root).as_posix()
    return stdout_name, stderr_name, file_sha256(stdout_path), file_sha256(stderr_path)


def _write_model_output(
    *,
    raw_log_dir: Path,
    task_id: str,
    output: str,
    evidence_root: Path | None = None,
) -> tuple[str, str]:
    if raw_log_dir.is_symlink():
        raise BenchmarkError("raw coding output directory cannot be a symbolic link")
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_log_dir / f"{task_id}.model-output.txt"
    root = (evidence_root or raw_log_dir).resolve()
    if output_path.is_symlink():
        raise BenchmarkError("raw coding output file cannot be a symbolic link")
    try:
        output_path.resolve().relative_to(root)
    except ValueError as exc:
        raise BenchmarkError("raw coding outputs must be inside the evidence root") from exc
    write_text(output_path, output)
    output_name = output_path.resolve().relative_to(root).as_posix()
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
    trusted_sandbox = bool(
        sandbox is not None
        and sys.platform == "darwin"
        and Path(sandbox).resolve() == TRUSTED_SANDBOX_EXECUTABLE
    )
    if required is None or required not in executables or not trusted_sandbox:
        missing = required if required not in executables else "trusted macOS sandbox"
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

    if required in {"rust", "go"}:
        executables[required] = _native_toolchain_executable(required, executables[required])

    if work_root.is_symlink():
        raise BenchmarkError("coding sandbox work root cannot be a symbolic link")
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{task.task_id}-", dir=work_root) as temporary:
        workspace = Path(temporary).resolve()
        source_dir = workspace / "input"
        output_dir = workspace / "output"
        source_dir.mkdir()
        output_dir.mkdir()
        if payload.candidate_path in payload.fixture_files:
            raise BenchmarkError("coding candidate path collides with a fixture path")
        for relative_path, content in payload.fixture_files.items():
            fixture = source_dir / relative_path
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_text(content, encoding="utf-8")
        candidate = source_dir / payload.candidate_path
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(_unfenced(model_output.output), encoding="utf-8")
        scorer_executables = {required: executables[required]}
        if required == "typescript" and "node" in executables:
            scorer_executables["node"] = executables["node"]
        toolchain_paths, developer_environment, rust_linker = _sandbox_runtime_paths(
            scorer_executables
        )
        commands, environment_updates = _compile_and_test_commands(
            payload,
            source_dir=source_dir,
            output_dir=output_dir,
            executables=executables,
            completion_token=secrets.token_hex(16),
            rust_linker=rust_linker,
        )
        environment = {
            "PATH": os.pathsep.join(
                sorted({str(Path(path).parent) for path in scorer_executables.values()})
            ),
            "HOME": str(output_dir / "home"),
            "TMPDIR": str(output_dir / "tmp"),
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "SOURCE_DATE_EPOCH": "0",
            **developer_environment,
            **environment_updates,
        }
        Path(environment["HOME"]).mkdir()
        Path(environment["TMPDIR"]).mkdir()
        results: list[_CommandResult] = []
        remaining_output_bytes = task.output_limit_bytes
        _seal_input_tree(source_dir)
        try:
            for command in commands:
                profile = _sandbox_profile(
                    input_dir=source_dir,
                    output_dir=output_dir,
                    toolchain_paths=toolchain_paths,
                    command=command,
                )
                result = _run_command(
                    command,
                    task=task,
                    work_dir=output_dir,
                    environment=environment,
                    profile=profile,
                    sandbox_executable=str(TRUSTED_SANDBOX_EXECUTABLE),
                    output_limit_bytes=remaining_output_bytes,
                )
                results.append(result)
                remaining_output_bytes = max(
                    0,
                    remaining_output_bytes - len(result.stdout) - len(result.stderr),
                )
                if (
                    result.exit_code != 0
                    or result.timed_out
                    or result.memory_exceeded
                    or result.process_limit_exceeded
                    or result.output_limit_exceeded
                    or result.completion_evidence_missing
                    or result.infrastructure_error
                ):
                    break
        finally:
            _unseal_input_tree(source_dir)

    stdout = b"".join(result.stdout for result in results)
    stderr = b"".join(result.stderr for result in results)
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
        and not results[0].completion_evidence_missing
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
            and not result.completion_evidence_missing
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
            or (
                "trusted test completion evidence missing"
                if any(result.completion_evidence_missing for result in results)
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
        or not same_model_identity(state.model, expected.model)
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
