"""Global deterministic test boundaries."""

from __future__ import annotations

import socket
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def block_unmarked_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    allowed = any(request.node.get_closest_marker(name) for name in ("network", "live", "paid"))
    if allowed:
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def is_loopback(address: object) -> bool:
        return (
            isinstance(address, tuple)
            and bool(address)
            and str(address[0]).lower() in {"127.0.0.1", "::1", "localhost"}
        )

    def connect(sock: socket.socket, address: Any) -> Any:
        if is_loopback(address):
            return original_connect(sock, address)
        raise AssertionError(
            "network access requires an explicit network, live, or paid pytest marker"
        )

    def connect_ex(sock: socket.socket, address: Any) -> Any:
        if is_loopback(address):
            return original_connect_ex(sock, address)
        raise AssertionError(
            "network access requires an explicit network, live, or paid pytest marker"
        )

    def create_connection(address: Any, *args: object, **kwargs: object) -> Any:
        if is_loopback(address):
            return original_create_connection(address, *args, **kwargs)
        raise AssertionError(
            "network access requires an explicit network, live, or paid pytest marker"
        )

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
