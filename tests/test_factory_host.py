from __future__ import annotations

from pathlib import Path

import pytest

from axquant.factory import (
    FACTORY_CERT_ROOT,
    FACTORY_DATASETS,
    FACTORY_HF_HOME,
    FACTORY_HOST_ID,
    FACTORY_MODELS,
    LARGE_MEMORY_CERT_HOST_ID,
    FactoryHostError,
    is_historical_cert_host,
    normalize_host_id,
    require_factory_host,
    require_large_memory_cert_host,
)
from axquant.public_cert_index import load_public_cert_rows

_ROOT = Path(__file__).resolve().parents[1]
_CERTS = _ROOT / "docs" / "certifications"


def test_factory_disk_defaults_are_ext16tr0() -> None:
    assert FACTORY_HF_HOME.startswith("/Volumes/Ext16TR0/")
    assert FACTORY_MODELS == "/Volumes/Ext16TR0/models"
    assert FACTORY_CERT_ROOT == "/Volumes/Ext16TR0/axquant-certification"
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
    # The public catalog was withdrawn pending re-certification; the
    # historical host set itself stays recognized without live rows.
    assert load_public_cert_rows(_CERTS) == []
    for host in (
        "df-macstudio-m2",
        "df-macbookpro-m5",
        "df-macbookpro-m3",
        "tn-macstudio-m3",
    ):
        assert is_historical_cert_host(host)
    assert not is_historical_cert_host("synthetic-test-host")


def test_require_large_memory_cert_host_accepts_m3_studio_aliases() -> None:
    assert require_large_memory_cert_host("tn-macstudio-m3") == LARGE_MEMORY_CERT_HOST_ID
    assert require_large_memory_cert_host("localadacStudio") == LARGE_MEMORY_CERT_HOST_ID
    assert (
        require_large_memory_cert_host("localadmins-Mac-Studio.local") == LARGE_MEMORY_CERT_HOST_ID
    )
    with pytest.raises(FactoryHostError, match="tn-macstudio-m3"):
        require_large_memory_cert_host("df-macstudio-m2")
