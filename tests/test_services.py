from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def test_linux_service_template_uses_broker_or_explicit_direct_launcher() -> None:
    template = Path("services/linux/qwen-omni-adapters.service.in").read_text(encoding="utf-8")
    installer = Path("services/linux/install.sh").read_text(encoding="utf-8")

    assert "ExecStart=@EXEC_START@" in template
    assert "EnvironmentFile=-@REPO_ROOT@/.env" in template
    assert "portal/start.sh --foreground" in installer
    assert "qwen-omni-daemon serve --allow-direct-gpu" in installer
    assert "docker gpu discover" in installer
    assert "Refusing --direct" in installer


def test_macos_launchd_template_is_valid_after_substitution() -> None:
    template = Path("services/macos/ai.robit.qwen-omni-adapters.plist.in").read_text(
        encoding="utf-8"
    )
    rendered = template.replace("@REPO_ROOT@", "/opt/qwen-omni-adapters").replace(
        "@PATH@", "/usr/local/bin:/usr/bin:/bin"
    )

    root = ET.fromstring(rendered)

    assert root.tag == "plist"
    assert "qwen-omni-daemon" in rendered
    assert "serve" in rendered


def test_windows_installer_supports_user_task_and_true_service_modes() -> None:
    installer = Path("services/windows/install.ps1").read_text(encoding="utf-8")
    wrapper = Path("src/qwen_omni_adapters/windows_service.py").read_text(encoding="utf-8")

    assert 'ValidateSet("Task", "Service")' in installer
    assert "Register-ScheduledTask" in installer
    assert "qwen_omni_adapters.windows_service" in installer
    assert '_svc_name_ = "QwenOmniAdapters"' in wrapper
    assert "ServiceFramework" in wrapper


def test_platform_deployers_bootstrap_before_service_install() -> None:
    macos = Path("deploy-macos.sh").read_text(encoding="utf-8")
    windows = Path("deploy.ps1").read_text(encoding="utf-8")

    assert macos.index("scripts/bootstrap.sh") < macos.index("services/macos/install.sh")
    assert windows.index("scripts\\bootstrap.ps1") < windows.index("services\\windows\\install.ps1")
