"""Transient windows outside the snapshot are attributed only by the owner's X client range."""
from types import SimpleNamespace

import pytest

from runtime import owned_flatpak_firefox as owned

OWNED = str(0x04400016)


def row(parent):
    return {"parent": parent, "map_state": 2, "override_redirect": False}


BEFORE = {"root": "1", "complete": True, "windows": {"1": row("0")}}
AFTER = {"root": "1", "complete": True, "windows": {"1": row("0"), OWNED: row("1")}}
PROOF = {"window": {"process_id": 42, "process_start": "420"}}


def backend(*, current=None, ranges=None):
    value = owned._SystemBackend.__new__(owned._SystemBackend)
    ranges = list(ranges or [{"base": 0x04400000, "mask": 0x1FFFFF}] * 2)
    value.identity = SimpleNamespace(
        xres_client_range=lambda _xid, _env: ranges.pop(0),
        xres_owner=lambda _xid, _env: {"pid": 42, "start": "420"})
    value.snapshot = lambda: current or AFTER
    return value


def events(*xids):
    return [{"window_id": OWNED, "type": 16}, {"window_id": OWNED, "type": 19}] + [
        {"window_id": xid, "type": kind} for xid in xids for kind in (16, 17)]


def test_destroyed_windows_from_the_owners_client_range_are_accepted():
    backend().validate_events(events(str(0x04400003), str(0x04400010)), OWNED, BEFORE, AFTER, PROOF)


@pytest.mark.parametrize("case", ["foreign_range", "still_present", "client_reconnected"])
def test_unattributable_transient_windows_reject(case):
    xid = str(0x04400003)
    kwargs = {}
    if case == "foreign_range":
        xid = str(0x04600003)
    elif case == "still_present":
        kwargs["current"] = {**AFTER, "windows": {**AFTER["windows"], xid: row("1")}}
    else:
        kwargs["ranges"] = [{"base": 0x04400000, "mask": 0x1FFFFF}, {"base": 0x04800000, "mask": 0x1FFFFF}]
    with pytest.raises(owned.OwnershipUnavailable, match="unknown_window_event"):
        backend(**kwargs).validate_events(events(xid), OWNED, BEFORE, AFTER, PROOF)


def test_preexisting_window_events_still_reject():
    with pytest.raises(owned.OwnershipUnavailable, match="concurrent_preexisting_event"):
        backend().validate_events(events("1"), OWNED, BEFORE, AFTER, PROOF)
