"""The daemon has to come up on its own, including with no internet.

A machine with no network still has a microphone, a camera and speakers, and
everything this daemon serves locally happens over loopback. Publishing a
tunnel is a convenience on top of that, and a convenience must not be able to
take the conversation down with it.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_omni_adapters import daemon  # noqa: E402


def test_a_tunnel_that_cannot_publish_does_not_stop_the_daemon() -> None:
    """Offline, cloudflared never prints a URL. That used to be fatal."""

    source = inspect.getsource(daemon.OmniDaemon.start_children)

    assert "cloudflared did not publish a URL" not in source
    assert "cloudflared exited before publishing a URL" not in source
    assert "serving locally at" in source


def test_a_missing_cloudflared_only_disables_publishing() -> None:
    source = inspect.getsource(daemon.OmniDaemon._preflight)

    # It is not in the list whose absence aborts startup.
    assert 'required = ["ffmpeg", "ollama"]' in source
    assert "serving locally only" in source


def test_a_stale_instance_is_replaced_rather_than_refused() -> None:
    """Refusing left the ports held until a human noticed and intervened."""

    source = inspect.getsource(daemon.OmniDaemon._reclaim_from_prior_instance)

    assert "SIGTERM" in source
    assert "SIGKILL" in source
    # The old refusal is gone from the module entirely.
    assert "daemon is already running as pid" not in inspect.getsource(daemon)


def test_the_port_check_waits_for_a_replaced_instance_to_let_go() -> None:
    """A port just given up takes a moment; failing instantly wedged restarts."""

    source = inspect.getsource(daemon.OmniDaemon._preflight)

    assert "time.sleep(0.5)" in source
    assert "required loopback port is already in use" in source


def test_a_model_already_present_is_never_pulled() -> None:
    """Offline, a pull cannot succeed -- and does not need to."""

    source = inspect.getsource(daemon.OmniDaemon._ensure_model)

    assert "ollama" in source and "pull" in source
    # The pull is reached only when `ollama show` fails.
    assert "if shown.returncode != 0:" in source
