"""SSRF guard for the Clef MLX server's remote image fetch (scripts/clef_mlx.py)."""

from __future__ import annotations

import importlib.util
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "clef_mlx.py"


@pytest.fixture(scope="module")
def clef() -> Any:
    """Load the script with stub MLX modules; the fetch guard is stdlib only."""
    mx = ModuleType("mlx.core")
    nn = ModuleType("mlx.nn")
    nn.Module = object  # type: ignore[attr-defined]
    mx.__getattr__ = lambda name: object()  # type: ignore[attr-defined]
    package = ModuleType("mlx")
    package.core = mx  # type: ignore[attr-defined]
    package.nn = nn  # type: ignore[attr-defined]
    name = "clef_mlx_under_test"
    saved = {key: sys.modules.get(key) for key in ("mlx", "mlx.core", "mlx.nn", name)}
    sys.modules.update({"mlx": package, "mlx.core": mx, "mlx.nn": nn})
    try:
        spec = importlib.util.spec_from_file_location(name, _SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Dataclasses resolve string annotations through the defining module.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


@pytest.mark.parametrize(
    "address",
    [
        "8.8.8.8",
        "93.184.216.34",
        "2606:4700:4700::1111",
        "::ffff:8.8.8.8",
    ],
)
def test_public_addresses_are_allowed(clef: Any, address: str) -> None:
    assert clef._is_public_address(address)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.9",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "::",
        "fe80::1%en0",
        "fc00::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "2002:7f00:1::1",
        "64:ff9b::a00:1",
    ],
)
def test_non_public_addresses_are_refused(clef: Any, address: str) -> None:
    assert not clef._is_public_address(address)


@pytest.fixture
def no_network(clef: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a socket connection was attempted")

    monkeypatch.setattr(clef.socket, "create_connection", refuse)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://127.0.0.1:8080/admin",
        "http://localhost/",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::ffff:169.254.169.254]/",
        "http://10.0.0.5/",
        "http://0.0.0.0/",
    ],
)
def test_internal_destinations_are_refused_before_connecting(
    clef: Any, no_network: None, url: str
) -> None:
    with pytest.raises(ValueError, match="public host"):
        clef._fetch_remote_image(url)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/a.png",
        "file:///etc/passwd",
        "http://user:secret@example.com/a.png",
        "http://example.com:notaport/a.png",
        "http:///a.png",
    ],
)
def test_unsupported_urls_are_refused(clef: Any, no_network: None, url: str) -> None:
    with pytest.raises(ValueError):
        clef._fetch_remote_image(url)


def test_hostname_resolving_to_private_address_is_refused(
    clef: Any, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def resolve(host: str, port: int, **kwargs: object) -> list[tuple[Any, ...]]:
        return [(2, 1, 6, "", ("10.0.0.7", port))]

    monkeypatch.setattr(clef.socket, "getaddrinfo", resolve)
    with pytest.raises(ValueError, match="public host"):
        clef._fetch_remote_image("http://rebind.example/a.png")


def test_one_private_answer_among_public_ones_is_refused(
    clef: Any, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def resolve(host: str, port: int, **kwargs: object) -> list[tuple[Any, ...]]:
        return [
            (2, 1, 6, "", ("93.184.216.34", port)),
            (2, 1, 6, "", ("127.0.0.1", port)),
        ]

    monkeypatch.setattr(clef.socket, "getaddrinfo", resolve)
    with pytest.raises(ValueError, match="public host"):
        clef._fetch_remote_image("http://mixed.example/a.png")


def test_resolution_failure_is_reported_generically(
    clef: Any, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def resolve(host: str, port: int, **kwargs: object) -> list[tuple[Any, ...]]:
        raise OSError("name or service not known")

    monkeypatch.setattr(clef.socket, "getaddrinfo", resolve)
    with pytest.raises(ValueError, match="could not fetch image URL"):
        clef._fetch_remote_image("http://missing.example/a.png")


class _Origin:
    """Loopback HTTP origin; the guard's address check is stubbed to admit it."""

    def __init__(self, routes: dict[str, tuple[int, dict[str, str], bytes]]) -> None:
        self.requests: list[tuple[str, str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.requests.append((self.path, self.headers.get("Host", "")))
                status, headers, body = routes.get(self.path, (404, {}, b""))
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> _Origin:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def admit_loopback(clef: Any, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Treat names ending in .public as public and pin them to loopback."""
    resolved: list[str] = []

    def resolve(host: str, port: int) -> str:
        resolved.append(host)
        if not host.endswith(".public"):
            raise ValueError("image URL must point to a public host")
        return "127.0.0.1"

    monkeypatch.setattr(clef, "_resolve_public_host", resolve)
    return resolved


def test_fetch_connects_to_the_validated_address(clef: Any, admit_loopback: list[str]) -> None:
    with _Origin({"/img.png?x=1": (200, {}, b"pixels")}) as origin:
        data = clef._fetch_remote_image(f"http://img.public:{origin.port}/img.png?x=1#frag")
    assert data == b"pixels"
    assert origin.requests == [("/img.png?x=1", f"img.public:{origin.port}")]
    assert admit_loopback == ["img.public"]


def test_redirect_to_internal_host_is_refused(clef: Any, admit_loopback: list[str]) -> None:
    routes = {"/start": (302, {"Location": "http://169.254.169.254/latest/meta-data/"}, b"")}
    with _Origin(routes) as origin, pytest.raises(ValueError, match="public host"):
        clef._fetch_remote_image(f"http://img.public:{origin.port}/start")
    assert [path for path, _ in origin.requests] == ["/start"]
    assert admit_loopback == ["img.public", "169.254.169.254"]


def test_public_redirects_are_followed_and_revalidated(
    clef: Any, admit_loopback: list[str]
) -> None:
    routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
    with _Origin(routes) as origin:
        routes["/a"] = (301, {"Location": f"http://cdn.public:{origin.port}/b"}, b"")
        routes["/b"] = (200, {}, b"final")
        assert clef._fetch_remote_image(f"http://img.public:{origin.port}/a") == b"final"
    assert admit_loopback == ["img.public", "cdn.public"]


def test_redirect_loops_are_bounded(clef: Any, admit_loopback: list[str]) -> None:
    routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
    with _Origin(routes) as origin:
        routes["/loop"] = (302, {"Location": f"http://img.public:{origin.port}/loop"}, b"")
        with pytest.raises(ValueError, match="too many times"):
            clef._fetch_remote_image(f"http://img.public:{origin.port}/loop")
    assert len(origin.requests) == clef._REMOTE_IMAGE_MAX_REDIRECTS + 1


def test_oversized_response_is_refused(
    clef: Any, admit_loopback: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(clef, "_REMOTE_IMAGE_MAX_BYTES", 8)
    with (
        _Origin({"/big": (200, {}, b"x" * 64)}) as origin,
        pytest.raises(ValueError, match="too large"),
    ):
        clef._fetch_remote_image(f"http://img.public:{origin.port}/big")


def test_error_status_is_refused(clef: Any, admit_loopback: list[str]) -> None:
    with _Origin({}) as origin, pytest.raises(ValueError, match="HTTP 404"):
        clef._fetch_remote_image(f"http://img.public:{origin.port}/missing")


def test_decode_image_routes_urls_through_the_guard(
    clef: Any, no_network: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PIL.Image")
    with pytest.raises(ValueError, match="public host"):
        clef._decode_image("http://169.254.169.254/latest/meta-data/")


def test_script_has_no_unguarded_url_fetch() -> None:
    source = _SCRIPT.read_text()
    assert "urlopen" not in source
    assert "urllib.request" not in source
