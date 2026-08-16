from __future__ import annotations

import json
from pathlib import Path

import pytest

from axquant.factory import (
    FACTORY_CERT_ROOT,
    FACTORY_DATASETS,
    FACTORY_HF_HOME,
    FACTORY_HOST_ID,
    FACTORY_MODELS,
    FactoryHostError,
    is_historical_cert_host,
    normalize_host_id,
    require_factory_host,
)
from axquant.public_cert_index import load_public_cert_rows

_ROOT = Path(__file__).resolve().parents[1]
_CERTS = _ROOT / "docs" / "certifications"


def test_factory_disk_defaults_are_ext12t() -> None:
    assert FACTORY_HF_HOME.startswith("/Volumes/Ext12T/")
    assert FACTORY_MODELS == "/Volumes/Ext12T/models"
    assert FACTORY_CERT_ROOT == "/Volumes/Ext12T/axquant-certification"
    assert FACTORY_DATASETS.startswith(FACTORY_CERT_ROOT)


def test_require_factory_host_accepts_studio_fqdn() -> None:
    assert require_factory_host("df-macstudio-m2") == "df-macstudio-m2"
    assert require_factory_host("df-macstudio-m2.defai.digital") == FACTORY_HOST_ID
    assert require_factory_host("devopsmacstudio.defai.digital") == FACTORY_HOST_ID


def test_require_factory_host_rejects_other_machines() -> None:
    with pytest.raises(FactoryHostError, match="df-macbookpro-m5"):
        require_factory_host("df-macbookpro-m5")
    with pytest.raises(FactoryHostError, match="observed df-macbookpro-m3"):
        require_factory_host("df-macbookpro-m3.defai.digital")


def test_normalize_host_id_strips_domain() -> None:
    assert normalize_host_id("  df-macstudio-m2.defai.digital\n") == "df-macstudio-m2"


def test_historical_cert_hosts_are_recognized_and_not_rewritten() -> None:
    rows = load_public_cert_rows(_CERTS)
    assert rows, "expected published cert rows"
    hosts = {row.host_id for row in rows}
    assert hosts <= {"df-macstudio-m2", "df-macbookpro-m5", "df-macbookpro-m3"}
    # Immutable historical hosts must remain present in the published set.
    assert "df-macbookpro-m5" in hosts or "df-macbookpro-m3" in hosts
    for host in hosts:
        assert is_historical_cert_host(host)


def test_qwen38_4bit_mtp_historical_host_json_not_rewritten() -> None:
    payload = json.loads((_CERTS / "qwen38-27b-axq4-mtp-tier1.json").read_text())
    assert payload["host_id"] == "df-macbookpro-m3"
    assert payload["artifact"]["hub_repo_id"].endswith("4bit-MTP")
