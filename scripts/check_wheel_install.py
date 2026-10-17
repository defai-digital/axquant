#!/usr/bin/env python3
"""Build and smoke-test the pip distribution outside the source checkout."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="axquant-wheel-check-") as temporary:
        work = Path(temporary)
        wheels = work / "wheels"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(wheels),
                str(root),
            ],
            cwd=work,
            env=env,
            check=True,
            timeout=300,
        )
        artifacts = list(wheels.glob("axquant-*.whl"))
        if len(artifacts) != 1:
            raise RuntimeError("expected exactly one AXQuant wheel")
        install = work / "venv"
        venv.EnvBuilder(with_pip=True).create(install)
        scripts = install / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        cli = scripts / ("axquant.exe" if os.name == "nt" else "axquant")
        subprocess.run(
            [str(python), "-m", "pip", "install", str(artifacts[0])],
            cwd=work,
            env=env,
            check=True,
            timeout=300,
        )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                """
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
import axquant
from axquant.cli._parser import _build_parser
from axquant.conversion_formats import MLX_FORMATS, format_ladder
from axquant.ladders import list_ladders

assert Path(axquant.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert axquant.__version__ == importlib.metadata.version('axquant')
assert list_ladders(), 'packaged ladder data missing'
for name in MLX_FORMATS:
    assert _build_parser().parse_args(['quantize', 'local-model', '--format', name]).format == name
    assert format_ladder(name).candidate_bits
for name in ('mlx', 'mlx_lm', 'torch'):
    assert importlib.util.find_spec(name) is None, f'base wheel unexpectedly installs {name}'
""",
            ],
            cwd=work,
            env=env,
            check=True,
            timeout=60,
        )
        for args, expected in [
            (["--version"], "axquant "),
            (["formats", "--backend", "mlx"], "mxfp8"),
            (["formats", "--backend", "cuda"], "NVFP4"),
            (["quantize", "--help"], "--format"),
            (["quantize-cuda", "--help"], "nvfp4"),
        ]:
            result = subprocess.run(
                [str(cli), *args],
                cwd=work,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if expected not in result.stdout:
                raise RuntimeError(f"installed CLI failed smoke: {args}")
    print("Wheel install passed: isolated base dependencies, packaged resources and console CLI")


if __name__ == "__main__":
    main()
