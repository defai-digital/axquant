"""Factory host contract for convert and new public certificates."""

from __future__ import annotations

FACTORY_HOST_ID = "df-macstudio-m2"
FACTORY_HOST_ALIASES = frozenset({"df-macstudio-m2", "devopsmacstudio"})
FACTORY_HF_HOME = "/Volumes/Ext12T/huggingface"
FACTORY_MODELS = "/Volumes/Ext12T/models"
FACTORY_CERT_ROOT = "/Volumes/Ext12T/axquant-certification"
FACTORY_DATASETS = "/Volumes/Ext12T/axquant-certification/datasets"
# 512 GB Studio: DeepSeek Flash-0731 packs that cannot generate on 192 GB.
LARGE_MEMORY_CERT_HOST_ID = "tn-macstudio-m3"
LARGE_MEMORY_CERT_HOST_ALIASES = frozenset(
    {"tn-macstudio-m3", "localadacStudio", "localadmins-Mac-Studio"}
)
HISTORICAL_CERT_HOSTS = frozenset(
    {"df-macstudio-m2", "df-macbookpro-m5", "df-macbookpro-m3", "tn-macstudio-m3"}
)


class FactoryHostError(ValueError):
    """Raised when convert/cert work is attempted off the factory host."""


def normalize_host_id(hostname: str) -> str:
    """Return the short host id (first DNS label)."""
    return hostname.strip().split(".", 1)[0]


def require_factory_host(observed_hostname: str, *, host_id: str = FACTORY_HOST_ID) -> str:
    """Fail closed unless *observed_hostname* is the factory convert/cert host."""
    observed = normalize_host_id(observed_hostname)
    allowed = FACTORY_HOST_ALIASES if host_id == FACTORY_HOST_ID else {host_id}
    if observed not in allowed:
        raise FactoryHostError(f"factory convert/cert must run on {host_id}; observed {observed}")
    return host_id


def require_large_memory_cert_host(observed_hostname: str) -> str:
    """Fail closed unless this is the 512 GB recert host for memory-blocked packs."""
    observed = normalize_host_id(observed_hostname)
    if observed not in LARGE_MEMORY_CERT_HOST_ALIASES:
        raise FactoryHostError(
            f"large-memory recert must run on {LARGE_MEMORY_CERT_HOST_ID}; observed {observed}"
        )
    return LARGE_MEMORY_CERT_HOST_ID


def is_historical_cert_host(host_id: str) -> bool:
    """True if *host_id* is a recorded certificate host (immutable)."""
    return host_id in HISTORICAL_CERT_HOSTS
