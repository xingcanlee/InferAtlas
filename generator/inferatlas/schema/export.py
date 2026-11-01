"""Export deterministic Draft 2020-12 schemas: uv run python -m inferatlas.schema.export."""

import json
from pathlib import Path

from . import (
    IR,
    Annotation,
    Annotations,
    Arch,
    Catalog,
    Eval,
    Extract,
    Index,
    Report,
    Rule,
    Rules,
    Snapshot,
    Triage,
    View,
)

ROOT = Path(__file__).resolve().parents[3]
MODELS = {
    "arch": Arch,
    "annotation": Annotation,
    "annotations": Annotations,
    "catalog": Catalog,
    "eval": Eval,
    "extract": Extract,
    "index": Index,
    "ir": IR,
    "report": Report,
    "rule": Rule,
    "rules": Rules,
    "snapshot": Snapshot,
    "triage": Triage,
    "view": View,
}


def schema_text(name: str) -> str:
    """Stable object-key order, preserved semantic array order, no timestamps."""
    schema = MODELS[name].model_json_schema(by_alias=True, mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def export(directory: Path | None = None) -> None:
    """Write one self-contained schema per wire document."""
    destination = directory or ROOT / "schemas"
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(MODELS):
        (destination / f"{name}.schema.json").write_text(schema_text(name), encoding="utf-8")


if __name__ == "__main__":
    export()
