import socket

import pytest


def test_external_connections_are_blocked():
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        socket.create_connection(("example.com", 443), timeout=1)


def test_raw_socket_connect_is_blocked():
    with socket.socket() as sock, pytest.raises(RuntimeError, match="Network access is disabled"):
        sock.connect(("93.184.216.34", 80))
