from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _script_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "smoke_gemma4_omlx.py"
    spec = importlib.util.spec_from_file_location("smoke_gemma4_omlx", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_stats_uses_last_completed_request() -> None:
    module = _script_module()

    parsed = module._parse_stats(  # type: ignore[attr-defined]
        "vlm_mtp stats: request=a rounds=2 accepted=1/2\n"
        "vlm_mtp stats: request=b rounds=6 accepted=6/7\n"
    )

    assert parsed == {"rounds": 6, "accepted": 6, "proposed": 7}


def test_parse_stats_rejects_missing_runtime_evidence() -> None:
    module = _script_module()

    assert module._parse_stats("request completed without MTP") is None  # type: ignore[attr-defined]
