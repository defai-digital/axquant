from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from axquant.serde import stable_sha256

TRUSTED_SANDBOX_EXECUTABLE = Path("/usr/bin/sandbox-exec")

_BASE_RULES = (
    "(version 1)",
    "(deny default)",
    '(import "system.sb")',
    "(deny network*)",
    "(allow sysctl-read)",
    "(allow system-sched)",
    "(allow signal (target self))",
)
_COMPILE_PROCESS_RULES = (
    "(allow process-fork)",
    "(allow process-exec)",
)

SANDBOX_POLICY_CONTRACT = {
    "id": "axquant-macos-seatbelt-v3",
    "renderer": "render_sandbox_profile-v2",
    "base_rules": list(_BASE_RULES),
    "default": "deny",
    "system_runtime": 'import "system.sb"',
    "network": "deny-all",
    "read_scopes": [
        "sealed-task-input",
        "task-output",
        "explicit-toolchain-roots",
        "system.sb-runtime-paths",
    ],
    "write_scopes": ["task-output", "system.sb-standard-devices"],
    "candidate_processes": "entrypoint-exec-only-no-fork",
    "compiler_processes": list(_COMPILE_PROCESS_RULES),
    "path_ancestors": "metadata-only",
    "limits": ["cpu", "wall", "address-space", "process", "file-size", "open-files"],
}
SANDBOX_PROFILE_SHA256 = stable_sha256(SANDBOX_POLICY_CONTRACT)


def _seatbelt_string(value: str) -> str:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("sandbox paths cannot contain control characters")
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _metadata_ancestors(paths: Iterable[Path]) -> list[Path]:
    ancestors: set[Path] = set()
    for path in paths:
        current = path.resolve()
        while current != current.parent:
            ancestors.add(current)
            current = current.parent
    return sorted(ancestors)


def render_sandbox_profile(
    *,
    input_dir: Path,
    output_dir: Path,
    toolchain_paths: Iterable[Path],
    entrypoint: Path,
    allow_subprocesses: bool,
) -> str:
    """Render the policy whose stable contract is bound by SANDBOX_PROFILE_SHA256."""

    source = input_dir.resolve()
    output = output_dir.resolve()
    runtime_paths = sorted({path.resolve() for path in toolchain_paths})
    # ``Path.absolute()`` followed symlinks on some supported Python/platform
    # combinations. Preserve the lexical invocation as well as the resolved target:
    # Seatbelt may match either during exec, and version-manager shims rely on argv[0].
    invocation = Path(os.path.abspath(os.fspath(entrypoint)))
    entrypoints = sorted({invocation, entrypoint.resolve()})
    rules = [*_BASE_RULES]
    if allow_subprocesses:
        rules.extend(_COMPILE_PROCESS_RULES)
    else:
        selectors = " ".join(f'(literal "{_seatbelt_string(str(path))}")' for path in entrypoints)
        rules.append(f"(allow process-exec {selectors})")
    rules.extend(
        [
            f'(allow file-read* file-test-existence (subpath "{_seatbelt_string(str(source))}"))',
            f'(allow file-read* file-test-existence (subpath "{_seatbelt_string(str(output))}"))',
            f'(allow file-write* (subpath "{_seatbelt_string(str(output))}"))',
        ]
    )
    for path in _metadata_ancestors([source, output, *runtime_paths, *entrypoints]):
        rules.append(
            "(allow file-read-metadata file-test-existence "
            f'(literal "{_seatbelt_string(str(path))}"))'
        )
    for path in runtime_paths:
        rules.append(
            f'(allow file-read* file-test-existence (subpath "{_seatbelt_string(str(path))}"))'
        )
    return " ".join(rules)
