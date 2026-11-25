from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from scripts.repair_pack_layout import _runtime_contract_fix, plan_repair


def _pack(directory: Path, names: list[str], runtime: dict[str, object]) -> Path:
    directory.mkdir()
    config = {"model_type": "qwen3_5_moe", "architectures": ["X"]}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {name: np.zeros(2, dtype=np.float32) for name in names}, directory / "mtp.safetensors"
    )
    (directory / "mtplx_runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    return directory


def test_runtime_contract_gains_the_qwen_arch_id(tmp_path: Path) -> None:
    names = ["language_model.mtp.fc.weight", "language_model.mtp.norm.weight"]
    pack = _pack(tmp_path / "p", names, {"mtp_depth_max": 1})
    fixed = _runtime_contract_fix(pack, "qwen3_5_moe")
    assert fixed == {"mtp_depth_max": 1, "arch_id": "qwen3-next-mtp", "mtp_tensor_count": 2}


def test_current_runtime_contract_is_left_alone(tmp_path: Path) -> None:
    runtime = {"arch_id": "qwen3-next-mtp", "mtp_tensor_count": 1}
    pack = _pack(tmp_path / "p", ["mtp.fc.weight"], runtime)
    assert _runtime_contract_fix(pack, "qwen3_5_moe") is None


def test_foreign_runtime_contract_is_refused_not_overwritten(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "p", ["mtp.fc.weight"], {"arch_id": "deepseek-nextn"})
    plan = plan_repair(pack)
    assert not any("arch_id" in action for action in plan["actions"])
    assert any("foreign arch_id" in refusal for refusal in plan["refusals"])
