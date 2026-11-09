"""Generic-device counterfactuals, real PTY/socket I/O and session boundaries."""

from __future__ import annotations

import base64
import os
import pty
import select
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from portal.documents import SessionDocumentStore
from portal.peripherals import PeripheralError, PeripheralStore
from portal.tools import PortalToolHarness
from qwen_omni_adapters.context import configured_tool_families, load_context


@pytest.fixture
def serial_pair(monkeypatch):
    master, slave = pty.openpty()
    name = Path(os.ttyname(slave))
    os.close(slave)
    store = PeripheralStore()
    # Test-owned PTYs exercise the real transport. Production still requires
    # an actual hardware TTY and never exposes an allow-PTY switch.
    monkeypatch.setattr(
        store,
        "_serial_path",
        lambda value: name if value == str(name) else pytest.fail("wrong endpoint"),
    )
    monkeypatch.setattr(store, "_owners", lambda _path: [])
    try:
        yield store, master, str(name)
    finally:
        store.close()
        os.close(master)


def test_serial_persists_bidirectional_bytes_and_refuses_other_sessions(serial_pair):
    store, master, name = serial_pair
    handle = store.serial("alice", {"action": "open", "device": name, "baudrate": 19200})["handle"]
    with pytest.raises(PeripheralError, match="this session"):
        store.serial("bob", {"action": "read", "handle": handle})
    with pytest.raises(OSError):
        os.open(name, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    response = b"\x00\xffunclassified-frame\r\n"

    def peer():
        assert select.select([master], [], [], 2)[0]
        assert os.read(master, 100) == b"hello\r\n"
        os.write(master, response)

    thread = threading.Thread(target=peer)
    thread.start()
    result = store.serial(
        "alice",
        {
            "action": "exchange",
            "handle": handle,
            "data": "hello\r\n",
            "timeout_s": 1,
            "max_bytes": len(response),
        },
    )
    thread.join(2)
    assert not thread.is_alive() and base64.b64decode(result["data_base64"]) == response
    assert result["os_bytes_written"] == 7 and result["protocol_identified"] is False
    os.write(master, b"next")
    assert (
        store.serial("alice", {"action": "read", "handle": handle, "max_bytes": 4})["text_preview"]
        == "next"
    )
    store.clear("alice")
    with pytest.raises(PeripheralError):
        store.serial("alice", {"action": "status", "handle": handle})


def test_serial_busy_is_not_an_open_or_success(serial_pair, monkeypatch):
    store, _master, name = serial_pair
    monkeypatch.setattr(store, "_owners", lambda _: [{"pid": 123, "name": "unrelated-owner"}])
    result = store.serial("owner", {"action": "open", "device": name, "baudrate": 9600})
    assert result["status"] == "busy" and result["opened"] is False and not store._ports


def test_idle_serial_handle_closes_without_a_followup_request(serial_pair):
    store, _master, name = serial_pair
    store.ttl_s = 0.05
    handle = store.serial("owner", {"action": "open", "device": name, "baudrate": 9600})["handle"]
    deadline = time.monotonic() + 2
    while handle in store._ports and time.monotonic() < deadline:
        time.sleep(0.02)
    assert handle not in store._ports


def test_janitor_does_not_retain_discarded_stores():
    import gc
    import weakref

    from portal.peripherals import _start_janitor

    observed = threading.Event()
    store = PeripheralStore()
    store._reap = observed.set
    probe = weakref.ref(store)
    _start_janitor()
    assert observed.wait(2)
    del store
    deadline = time.monotonic() + 1
    while probe() is not None and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)
    assert probe() is None


@pytest.mark.parametrize(
    "fields",
    [
        {"baudrate": True},
        {"baudrate": 12345},
        {"baudrate": 9600, "parity": "invalid"},
        {"baudrate": 9600, "stop_bits": True},
        {},
    ],
)
def test_invalid_serial_configuration_never_opens(serial_pair, fields):
    store, _master, name = serial_pair
    with pytest.raises(PeripheralError):
        store.serial("owner", {"action": "open", "device": name, **fields})
    assert not store._ports


@pytest.mark.parametrize(
    "path", ["/dev/null", "/dev/zero", "/dev/tty", "/etc/passwd", "/dev/pts/0"]
)
def test_serial_path_cannot_be_an_arbitrary_host_file_or_terminal(path):
    with pytest.raises((PeripheralError, OSError)):
        PeripheralStore()._serial_path(path)


def test_all_device_classes_and_buses_are_paged_without_product_recognition(tmp_path):
    sys = tmp_path / "sys"
    dev = tmp_path / "dev"
    proc = tmp_path / "proc"
    dev.mkdir()
    proc.mkdir()
    for category, attributes in {
        "class/net/link-a": {"ifindex": "7", "operstate": "up"},
        "class/video4linux/video-a": {"name": "external imaging endpoint"},
        "class/hwmon/monitor-a": {"name": "temperature monitor"},
        "class/gpio/gpiochip-a": {"dev": "254:0"},
        "bus/pci/devices/0000:00:01.0": {"vendor": "0x1234", "device": "0xabcd"},
        "bus/usb/devices/1-2": {
            "idVendor": "1234",
            "idProduct": "abcd",
            "product": "Guaranteed trusted ready device",
        },
        "bus/usb/devices/1-2:1.0": {"bInterfaceClass": "ff", "modalias": "usb:unbound"},
    }.items():
        folder = sys / category
        folder.mkdir(parents=True)
        for key, value in attributes.items():
            (folder / key).write_text(value)
    store = PeripheralStore(sys_root=sys, dev_root=dev, proc_root=proc)
    result = store.inventory("all")
    categories = {c["category"] for c in result["system_devices"]["categories"]}
    assert {
        "class/net",
        "class/gpio",
        "class/video4linux",
        "class/hwmon",
        "bus/pci",
        "bus/usb",
    } <= categories
    assert result["system_devices"]["complete"] is False
    assert result["usb"][0]["protocol_identified"] is False
    assert result["usb"][0]["interfaces"][0]["driver_bound"] is False
    first = store.inventory("system", cursor=0, limit=2)["system_devices"]
    second = store.inventory("system", cursor=first["next_cursor"], limit=2)["system_devices"]
    assert not {d["device_ref"] for d in first["devices"]} & {
        d["device_ref"] for d in second["devices"]
    }
    assert (
        store.inventory("system", category="class/net")["system_devices"]["devices"][0][
            "attributes"
        ]["ifindex"]
        == "7"
    )
    with pytest.raises(PeripheralError):
        store.inventory("system", category="../../etc")


@pytest.mark.parametrize("protocol", ["tcp_exchange", "udp_exchange"])
def test_generic_network_exchange_uses_real_bounded_socket_bytes(protocol):
    kind = socket.SOCK_STREAM if protocol == "tcp_exchange" else socket.SOCK_DGRAM
    server = socket.socket(socket.AF_INET, kind)
    server.bind(("127.0.0.1", 0))
    server.settimeout(2)
    if kind == socket.SOCK_STREAM:
        server.listen(1)

    def peer():
        if kind == socket.SOCK_STREAM:
            conn, _ = server.accept()
            with conn:
                assert conn.recv(100) == b"query"
                conn.sendall(b"unknown-wire-protocol")
        else:
            data, target = server.recvfrom(100)
            assert data == b"query"
            server.sendto(b"unknown-wire-protocol", target)

    thread = threading.Thread(target=peer)
    thread.start()
    try:
        result = PeripheralStore().network(
            {
                "action": protocol,
                "address": "127.0.0.1",
                "port": server.getsockname()[1],
                "data": "query",
                "timeout_s": 1,
            }
        )
        assert result["text_preview"] == "unknown-wire-protocol"
        assert result["protocol_identified"] is False
    finally:
        thread.join(2)
        server.close()


@pytest.mark.parametrize(
    "address", ["localhost", "0.0.0.0", "224.0.0.1", "::", None, True, 2130706433]
)
def test_invalid_exchange_endpoints_never_resolve_or_transmit(address):
    with pytest.raises(PeripheralError):
        PeripheralStore().network({"action": "tcp_exchange", "address": address, "port": 80})


def test_bluetooth_facts_are_advertisements_not_identification(monkeypatch):
    import bleak

    class Scanner:
        @staticmethod
        async def discover(**kwargs):
            return {
                "a": (
                    SimpleNamespace(address="11:22:33:44:55:66", name="claimed perfect appliance"),
                    SimpleNamespace(
                        local_name=None,
                        rssi=-55,
                        service_uuids=["12345678-1111-2222-3333-123456789abc"],
                        manufacturer_data={77: b"\0\xff"},
                        service_data={},
                    ),
                )
            }

    monkeypatch.setattr(bleak, "BleakScanner", Scanner)
    result = PeripheralStore().bluetooth({"action": "scan", "timeout_s": 0.1})
    assert result["protocol_identified"] is False
    assert result["devices"][0]["manufacturer_data"] == {"77": "00ff"}
    assert result["trust"] == "untrusted_radio_advertisements"


def test_bluetooth_does_not_take_an_existing_connection(monkeypatch):
    import bleak

    store = PeripheralStore()
    monkeypatch.setattr(
        store, "_command", lambda *a, **kw: {"status": "ok", "output": "\tConnected: yes\n"}
    )
    monkeypatch.setattr(
        bleak, "BleakClient", lambda *_a, **_kw: pytest.fail("existing device was taken")
    )
    result = store.bluetooth(
        {"action": "services", "address": "11:22:33:44:55:66", "timeout_s": 0.1}
    )
    assert result["status"] == "busy" and result["connection_taken"] is False


@pytest.mark.parametrize("action", ["read", "write", "listen", "services"])
def test_bluetooth_gatt_operations_close_their_connection(monkeypatch, action):
    import bleak

    calls = []

    class Scanner:
        @staticmethod
        async def find_device_by_address(address, **kwargs):
            return SimpleNamespace(address=address)

    class Client:
        def __init__(self, *a, **kw):
            self.is_connected = True
            self.services = []

        async def __aenter__(self):
            calls.append("connected")
            return self

        async def __aexit__(self, *a):
            calls.append("closed")

        async def read_gatt_char(self, uuid):
            return b"\0raw"

        async def write_gatt_char(self, uuid, data, **kwargs):
            assert data == b"\0raw" and kwargs["response"]
            calls.append("written")

        async def start_notify(self, uuid, callback):
            callback(None, b"notification")

        async def stop_notify(self, uuid):
            calls.append("notify_stopped")

    monkeypatch.setattr(bleak, "BleakScanner", Scanner)
    monkeypatch.setattr(bleak, "BleakClient", Client)
    store = PeripheralStore()
    monkeypatch.setattr(
        store, "_command", lambda *a, **k: {"status": "ok", "output": "Connected: no"}
    )
    result = store.bluetooth(
        {
            "action": action,
            "address": "11:22:33:44:55:66",
            "characteristic": "12345678-1111-2222-3333-123456789abc",
            "data": "00726177",
            "encoding": "hex",
            "timeout_s": 0.1,
        }
    )
    assert calls[0] == "connected" and calls[-1] == "closed"
    if action == "read":
        assert base64.b64decode(result["data_base64"]) == b"\0raw"
    if action == "listen":
        assert result["notifications"][0]["bytes_received"] == 12


def test_wireless_power_claims_require_fresh_successful_readback(monkeypatch):
    store = PeripheralStore()
    responses = iter(
        [
            {"status": "ok", "output": "Powered: no"},
            {"status": "ok", "output": "request accepted"},
            {"status": "ok", "output": "Name: Powered: yes\nPowered: no"},
        ]
    )
    monkeypatch.setattr(store, "_command", lambda *a, **k: next(responses))
    assert store.bluetooth({"action": "power", "enabled": True})["verified"] is False


def test_generic_family_and_schema_do_not_encode_attached_products():
    catalog = load_context()
    families = configured_tool_families()
    assert families["peripherals"]["tools"] == [
        "peripheral_inventory",
        "serial_io",
        "bluetooth_io",
        "network_io",
    ]
    assert families["terminal"]["tools"] == ["terminal_session"]
    new = [
        e
        for e in catalog["tools"]
        if e["schema"]["function"]["name"]
        in {*families["peripherals"]["tools"], "terminal_session"}
    ]
    import json

    text = json.dumps(new).lower()
    assert all(
        word not in text for word in ["meshtastic", "gps", "067b", "303a", "ttyacm0", "ttyusb0"]
    )
    import jsonschema

    gateway = next(
        e["schema"]["function"]["parameters"]
        for e in catalog["tools"]
        if e["schema"]["function"]["name"] == "tool_search"
    )
    for family in ["peripherals", "terminal"]:
        jsonschema.validate({"family": family}, gateway)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"family": "invented_family"}, gateway)


def test_harness_routes_generic_hardware_without_changing_private_system_snapshot(monkeypatch):
    tools = PortalToolHarness(SessionDocumentStore())
    monkeypatch.setattr(tools.peripherals, "inventory", lambda *a, **k: {"observed": "class/net"})
    observed = tools.execute("owner", "peripheral_inventory", {"scope": "system"})
    assert (
        observed["observed"] == "class/net"
        and observed["provenance"]["producing_tool"] == "peripheral_inventory"
    )
    assert (
        tools.execute("owner", "serial_io", {"action": "read", "handle": "other-session"})["error"]
        == "PeripheralError"
    )
    snapshot = tools.execute("owner", "get_system_snapshot", {})
    assert "serial" not in snapshot and "usb" not in snapshot
