"""Configure GDM autologin for the local Jetson appliance account."""

from __future__ import annotations

import argparse
import configparser
import os
import pwd
import re
import stat
import tempfile
from pathlib import Path

GDM_CONFIG = Path("/etc/gdm3/custom.conf")
_SETTINGS = re.compile(r"^\s*(?:AutomaticLoginEnable|AutomaticLogin)\s*=")


def updated_config(source: str, account: str) -> str:
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_.-]*", account):
        raise ValueError("Use a local desktop account name")

    parser = configparser.ConfigParser(interpolation=None, strict=False, inline_comment_prefixes=("#", ";"))
    parser.optionxform = str
    parser.read_string(source)
    if parser.has_section("daemon"):
        daemon = parser["daemon"]
        if daemon.getboolean("AutomaticLoginEnable", fallback=False):
            current = daemon.get("AutomaticLogin", "")
            if current == account:
                return source
            if current:
                raise ValueError("GDM already automatically logs in a different account")
        if daemon.getboolean("TimedLoginEnable", fallback=False) and daemon.get("TimedLogin", account) != account:
            raise ValueError("GDM already has a timed login for a different account")

    lines = []
    in_daemon = False
    settings = f"AutomaticLoginEnable=True\nAutomaticLogin={account}\n"
    for line in source.splitlines(keepends=True):
        header = re.fullmatch(r"\s*\[([^\]]+)\]\s*(?:[#;].*)?", line.strip())
        if header:
            in_daemon = header[1] == "daemon"
            lines.append(line.rstrip("\n") + "\n" + (settings if in_daemon else ""))
        elif not (in_daemon and _SETTINGS.match(line)):
            lines.append(line)
    if not parser.has_section("daemon"):
        lines.append("\n[daemon]\n" + settings)
    return "".join(lines)


def configure(config: Path, account: str) -> bool:
    identity = pwd.getpwnam(account)
    if identity.pw_uid < 1000 or not Path(identity.pw_dir).is_dir():
        raise ValueError("GDM autologin requires an existing local desktop account")
    metadata = config.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0:
        raise ValueError("GDM config must be a root-owned regular file")
    source = config.read_text(encoding="utf-8")
    updated = updated_config(source, account)
    if updated == source:
        return False

    backup = config.with_name(config.name + ".before-omni-autologin")
    try:
        descriptor = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(source)

    descriptor, filename = tempfile.mkstemp(prefix=".omni-gdm-", dir=config.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), stat.S_IMODE(metadata.st_mode))
            os.fchown(handle.fileno(), metadata.st_uid, metadata.st_gid)
            handle.write(updated)
        temporary.replace(config)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("account")
    args = parser.parse_args()
    try:
        configure(GDM_CONFIG, args.account)
    except (OSError, KeyError, ValueError, configparser.Error) as error:
        parser.exit(1, f"GDM autologin: {error}\n")
    print(f"Jetson desktop autologin is configured for {args.account}.")


if __name__ == "__main__":
    main()
