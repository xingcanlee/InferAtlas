"""Validate YAML, invalidate changed anchors, and apply explicitly bound overlays."""

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter
from ruamel.yaml import YAML

from inferatlas.schema import IR, Annotation
from inferatlas.schema.models import AscendTag, SourceRef, StaleAnnotation

from .source import SourceTree, resolve_target


def load_annotations(path: str | Path) -> list[Annotation]:
    """Load one YAML list or a directory recursively; reject duplicate ids."""
    path = Path(path)
    files = sorted([*path.rglob("*.yaml"), *path.rglob("*.yml")]) if path.is_dir() else [path]
    annotations = []
    yaml = YAML(typ="safe")
    for file in files:
        annotations.extend(TypeAdapter(list[Annotation]).validate_python(yaml.load(file)))
    ids = [annotation.id for annotation in annotations]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate annotation id")
    return sorted(annotations, key=lambda annotation: annotation.id)


@dataclass(frozen=True)
class ResolvedAnnotation:
    annotation: Annotation
    src: tuple[SourceRef, ...]


def check_annotations(
    annotations: Iterable[Annotation], sources: Mapping[str, SourceTree]
) -> tuple[list[ResolvedAnnotation], list[StaleAnnotation]]:
    """Return usable anchors and triage items. Unresolvable symbols fail the build.

    Hash target and every additional anchor; any mismatch invalidates the note.
    Link resolves_to as well, even if it is not one of the hashed declarations.
    """
    valid = []
    stale = []
    seen = set()
    for annotation in sorted(annotations, key=lambda annotation: annotation.id):
        if annotation.id in seen:
            raise ValueError(f"Duplicate annotation id: {annotation.id}")
        seen.add(annotation.id)
        expected = [(annotation.target, annotation.anchor_hash)] + [
            (anchor.target, anchor.anchor_hash) for anchor in annotation.anchors
        ]
        if len({target for target, _ in expected}) != len(expected):
            raise ValueError(f"Duplicate annotation anchor: {annotation.id}")
        anchors = [(resolve_target(target, sources), digest) for target, digest in sorted(expected)]
        replacement = resolve_target(annotation.resolves_to, sources)
        for anchor, digest in anchors:
            if anchor.anchor_hash != digest:
                stale.append(
                    StaleAnnotation(
                        kind="stale_annotation",
                        id=annotation.id,
                        target=anchor.src.symbol,
                        anchor_hash_old=digest,
                        anchor_hash_new=anchor.anchor_hash,
                    )
                )
                break
        else:
            refs = {anchor.src.symbol: anchor.src for anchor, _ in anchors}
            refs[replacement.src.symbol] = replacement.src
            valid.append(ResolvedAnnotation(annotation, tuple(refs[key] for key in sorted(refs))))
    return valid, stale


def apply_annotations(
    ir: IR,
    annotations: Iterable[ResolvedAnnotation],
    *,
    bindings: Mapping[str, str],
    active_ids: Collection[str],
) -> IR:
    """Return a deep copy with overlays; never read files or evaluate predicates.

    bindings maps IR node id to annotation id using extraction's observed class
    swaps/patches. active_ids is the caller's predicate selection from the shared
    TS evaluator. Multiple notes per node need an explicit caller choice because
    the wire format has one annotation_id; never choose by file order/severity.
    """
    resolved = list(annotations)
    by_id = {item.annotation.id: item for item in resolved}
    if len(by_id) != len(resolved):
        raise ValueError("Duplicate annotation id")
    node_ids = {node.id for node in ir.nodes}
    if set(bindings) - node_ids:
        raise ValueError("Annotation binding references an unknown IR node")
    if (set(bindings.values()) | set(active_ids)) - by_id.keys():
        raise ValueError("Annotation selection references an unknown or stale annotation")
    result = ir.model_copy(deep=True)
    for node in result.nodes:
        annotation_id = bindings.get(node.id)
        if annotation_id is None:
            continue
        if annotation_id not in active_ids:
            continue
        item = by_id[annotation_id]
        for src in item.src:
            revision = {"vllm": ir.snapshot.vllm, "vllm-ascend": ir.snapshot.ascend}.get(src.repo)
            if revision is None or src.commit != revision.commit:
                raise ValueError(
                    f"Annotation source commit differs from IR snapshot: {annotation_id}"
                )
        annotation = item.annotation
        existing = node.ascend
        refs = {
            (src.repo, src.commit, src.file, src.line, src.symbol): src
            for src in [*(existing.src if existing else []), *item.src]
        }
        overlay = {
            "tag": sorted(set(annotation.tag) | set(existing.tag if existing else [])),
            "level": annotation.level,
            "note": annotation.note,
            "annotation_id": annotation.id,
            "src": [refs[key].model_copy(deep=True) for key in sorted(refs)],
        }
        node.ascend = existing.model_copy(update=overlay) if existing else AscendTag(**overlay)
    return result
