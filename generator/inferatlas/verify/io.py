"""Stable wire-format IO shared by the CLI and verification stages."""

import json
import subprocess
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[3]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wire(value):
    return value.model_dump(mode="json", by_alias=True) if isinstance(value, BaseModel) else value


def write_json(path: Path, value) -> None:
    content = json.dumps(wire(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def source_root(root: Path = ROOT) -> Path:
    import os

    if value := os.environ.get("INFERATLAS_SRC"):
        return Path(value)
    local = root / "experiments/a2_qwen3moe/src"
    if local.is_dir():
        return local
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip()).parent / "experiments/a2_qwen3moe/src"
