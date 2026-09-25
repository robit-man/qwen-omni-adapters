#!/usr/bin/env python3
"""Guard Kestrel's post-Torch-2.4 MAIA enum for JetPack 6 builds."""

from __future__ import annotations

import argparse
from pathlib import Path

MAIA_CASES = (
    """    case at::DeviceType::MAIA:
      device.device_type = kDLMAIA;
      break;""",
    """    case kDLMAIA:
      return at::Device(at::DeviceType::MAIA, device.device_id);""",
)


def patch_source(source: str) -> tuple[str, int]:
    """Wrap the two MAIA references missing guards in Kestrel 0.7.3."""

    changed = 0
    for block in MAIA_CASES:
        guarded = (
            "#if (TORCH_VERSION_MAJOR > 2) || "
            "(TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)\n"
            + block
            + "\n#endif"
        )
        if guarded in source:
            continue
        if block not in source:
            raise RuntimeError("Kestrel DLPack source has an unknown MAIA layout")
        source = source.replace(block, guarded, 1)
        changed += 1
    return source, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    target = args.target.resolve()
    source = target.read_text(encoding="utf-8")
    patched, changed = patch_source(source)
    if changed:
        target.write_text(patched, encoding="utf-8")
    print(f"kestrel JetPack compatibility guards: {changed} added in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
