"""Validate immutable source links without importing upstream libraries."""

import re
import subprocess
from pathlib import Path, PurePosixPath

from inferatlas.schema import Snapshot
from inferatlas.schema.models import SourceLinkFailure, SourceRef

from .io import wire


def source_refs(*documents) -> list[SourceRef]:
    found = {}

    def visit(value):
        if isinstance(value, dict):
            if {"repo", "file", "line", "commit", "symbol"} <= value.keys():
                ref = SourceRef.model_validate(value)
                found[(ref.repo, ref.file, ref.line, ref.symbol, ref.commit)] = ref
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for document in documents:
        visit(wire(document))
    return [found[key] for key in sorted(found)]


def check_sources(
    documents: list, snapshot: Snapshot, repos: dict[str, Path]
) -> list[SourceLinkFailure]:
    failures = []
    files = {}
    commits = {"vllm": snapshot.vllm.commit, "vllm-ascend": snapshot.ascend.commit}
    for ref in source_refs(*documents):
        reason = None
        path = PurePosixPath(ref.file)
        if ref.repo in commits and ref.commit != commits[ref.repo]:
            reason = "source commit disagrees with snapshot"
        elif path.is_absolute() or ".." in path.parts or not path.parts:
            reason = "source file must be repository-relative"
        elif ref.repo not in repos or not repos[ref.repo].is_dir():
            reason = "source repository unavailable"
        else:
            key = (ref.repo, ref.commit, ref.file)
            if key not in files:
                result = subprocess.run(
                    ["git", "-C", str(repos[ref.repo]), "show", f"{ref.commit}:{ref.file}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                files[key] = result.stdout.splitlines() if result.returncode == 0 else None
            lines = files[key]
            if lines is None:
                reason = "file or commit unavailable via git show"
            elif ref.line > len(lines):
                reason = "declaration line outside file"
            else:
                name = re.escape(ref.symbol.rsplit(".", 1)[-1])
                declaration = rf"^\s*(?:class\s+{name}\b|(?:async\s+)?def\s+{name}\s*\()"
                if not re.search(declaration, lines[ref.line - 1]):
                    reason = "line does not declare the referenced class/function"
        if reason:
            failures.append(SourceLinkFailure(src=ref, reason=reason))
    return failures
