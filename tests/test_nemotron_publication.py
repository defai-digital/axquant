from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from axquant.serde import file_sha256, write_data


def load_driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_nemotron3_mx_mtp.py"
    spec = importlib.util.spec_from_file_location("nemotron_publication", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cuda_driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_nemotron3_nvfp4_mtp.py"
    spec = importlib.util.spec_from_file_location("nemotron_cuda_publication", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cards_keep_mtp_execution_unverified_and_close_code_fences() -> None:
    driver = load_driver()
    for key in driver.PACKS:
        card = driver._card(key)
        assert card.count("```") == 2
        assert "Runtime compatibility is unverified" in card
        assert driver.SOURCES[driver.PACKS[key]["family"]]["revision"] in card


@pytest.mark.parametrize("drift", ["none", "manifest", "weights", "extra", "missing"])
def test_publication_rejects_stale_load_and_incomplete_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    driver = load_driver()
    output = tmp_path / "pack"
    output.mkdir()
    monkeypatch.setattr(driver, "pack_path", lambda _: output)
    monkeypatch.setattr(driver, "WORK", tmp_path / "work")
    manifest = output / "axquant_manifest.json"
    weights = output / "model.safetensors"
    write_data(manifest, {"status": "development"})
    weights.write_bytes(b"original weights")
    hashes = {p.name: file_sha256(p) for p in (manifest, weights)}
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    )
    record = driver.WORK / "lightning/mxfp4/mlx-lm-load.json"
    write_data(record, {"status": "passed", "manifest_sha256": hashes[manifest.name]})
    if drift == "manifest":
        write_data(manifest, {"status": "changed"})
    elif drift == "weights":
        weights.write_bytes(b"changed weights")
    elif drift == "extra":
        (output / "untracked.txt").write_text("unexpected member")
    elif drift == "missing":
        record.unlink()
    if drift == "none":
        verified = driver.verify_publication_inputs("lightning-mxfp4")
        assert verified == {**hashes, "SHA256SUMS.txt": file_sha256(output / "SHA256SUMS.txt")}
    else:
        with pytest.raises((SystemExit, FileNotFoundError)):
            driver.verify_publication_inputs("lightning-mxfp4")


def test_cuda_nemotron_preserves_integrated_mtp_and_detects_payload_drift(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    from axquant.architectures.nemotron3 import _LIGHTNING_30B_A3B_SIGNATURE
    from axquant.cuda import _quantization_config, convert_cuda_nvfp4, plan_cuda_nvfp4

    driver = load_cuda_driver()
    source = tmp_path / "source"
    source.mkdir()
    write_data(
        source / "config.json",
        {
            **_LIGHTNING_30B_A3B_SIGNATURE,
            "model_type": "nemotron_h",
            "torch_dtype": "bfloat16",
            "mtp_hybrid_override_pattern": "*E",
        },
    )
    names = [
        "backbone.layers.0.mixer.in_proj.weight",
        "backbone.layers.1.mixer.experts.0.up_proj.weight",
        "backbone.layers.1.mixer.experts.0.down_proj.weight",
        "mtp.layers.1.mixer.experts.0.up_proj.weight",
        "mtp.layers.0.norm.weight",
    ]
    tensors = {name: torch.ones((8, 32), dtype=torch.bfloat16) for name in names}
    tensors[names[-1]] = torch.ones(32, dtype=torch.bfloat16)
    save_file(tensors, source / "model.safetensors")
    write_data(
        source / "model.safetensors.index.json",
        {"weight_map": {name: "model.safetensors" for name in names}},
    )
    plan = plan_cuda_nvfp4(
        source,
        model_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        revision="a" * 40,
        keep_patterns=driver.KEEP_PATTERNS,
        allow_unmeasured=True,
    )
    selected = {item.tensor_name for item in plan.allocations if item.method == "nvfp4"}
    assert selected == set(names[1:3])
    assert "mtp.layers.1.mixer.experts" in _quantization_config(plan)["ignore"]
    output = tmp_path / "output"
    convert_cuda_nvfp4(source, plan, output, device="cpu", allow_unmeasured=True)
    records = driver.verify_integrated_mtp(source, output, plan, 2)
    assert {record["name"] for record in records} == set(names[-2:])
    assert all(record["dtype"] == "BF16" for record in records)
    changed = load_file(output / "model.safetensors")
    changed[names[-2]] = torch.zeros_like(changed[names[-2]])
    save_file(changed, output / "model.safetensors")
    with pytest.raises(SystemExit, match="payload changed"):
        driver.verify_integrated_mtp(source, output, plan, 2)


def test_cuda_cards_distinguish_checkpoint_format_from_gpu_execution() -> None:
    driver = load_cuda_driver()
    for family in driver.MTP_COUNTS:
        card = driver.card(family, "numpy-reference")
        assert "Activations remain BF16" in card
        assert "MTP runtime compatibility is unverified" in card
        assert "numpy-reference" in card
        assert "no AWQ or third-party quantizer" in card
