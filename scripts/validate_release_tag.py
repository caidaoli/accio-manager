from __future__ import annotations

import re
from pathlib import Path


def read_project_version(pyproject_path: Path) -> str:
    text = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
    if not match:
        raise ValueError(f"version not found in {pyproject_path}")
    return match.group(1)


def validate_release_tag(tag: str, version: str) -> str:
    if tag != f"v{version}":
        raise ValueError(
            f"tag {tag!r} must match project version {version!r} (expected v{version})"
        )
    return tag
