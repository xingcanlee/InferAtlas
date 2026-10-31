"""Source-anchored Ascend annotations; predicates use the shared TS DSL."""

from .pipeline import (
    AnchoredAnnotation,
    AnnotationAnchor,
    ResolvedAnnotation,
    apply_annotations,
    check_annotations,
    load_annotations,
)
from .source import SourceAnchor, SourceTree, anchor_hash, resolve_target

__all__ = [
    "AnchoredAnnotation",
    "AnnotationAnchor",
    "ResolvedAnnotation",
    "SourceAnchor",
    "SourceTree",
    "anchor_hash",
    "apply_annotations",
    "check_annotations",
    "load_annotations",
    "resolve_target",
]
