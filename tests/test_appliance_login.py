import pytest

from qwen_omni_adapters.appliance_login import updated_config


def test_appliance_autologin_enables_only_the_desktop_account():
    original = "# Jetson settings\n[daemon]\n#AutomaticLoginEnable=True\nAutomaticLoginEnable=False\nWaylandEnable=false\n[debug]\nEnable=true\n"
    result = updated_config(original, "egg")
    assert "[daemon]\nAutomaticLoginEnable=True\nAutomaticLogin=egg\n" in result
    assert "AutomaticLoginEnable=False" not in result
    assert "WaylandEnable=false\n[debug]\nEnable=true\n" in result
    assert updated_config(result, "egg") == result


def test_appliance_install_preserves_an_existing_other_user():
    source = "[daemon]\nAutomaticLoginEnable=True\nAutomaticLogin=existing\n"
    with pytest.raises(ValueError, match="different account"):
        updated_config(source, "egg")
