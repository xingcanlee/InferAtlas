"""Resolve declarations at immutable revisions without importing upstream code."""

import ast
import hashlib
import io
import json
import subprocess
import textwrap
import tokenize
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from inferatlas.schema.models import SourceRef


def anchor_hash(source: str) -> str:
    """Hash tokens of dedented source, ignoring comments and formatting whitespace.

    Block boundaries and logical newlines remain significant; whitespace inside
    string literals is data. Decorators and docstrings are part of the anchor.
    """
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(textwrap.dedent(source)).readline):
        if token.type in ignored:
            continue
        value = (
            ""
            if token.type in {tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE}
            else token.string
        )
        tokens.append((token.type, value))
    normalized = json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceAnchor:
    src: SourceRef
    source: str

    @property
    def anchor_hash(self) -> str:
        return anchor_hash(self.source)


@dataclass(frozen=True)
class SourceTree:
    """Read a Git tree at commit, including when the checkout has local edits."""

    root: Path
    repo: str
    commit: str

    def resolve(self, symbol: str) -> SourceAnchor:
        parts = symbol.split(".")
        if not all(part.isidentifier() for part in parts):
            raise ValueError(f"Invalid qualified symbol: {symbol}")
        for split in range(len(parts) - 1, 0, -1):
            module = "/".join(parts[:split])
            for file in (f"{module}.py", f"{module}/__init__.py"):
                result = subprocess.run(
                    ["git", "-C", str(self.root), "show", f"{self.commit}:{file}"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                )
                if result.returncode:
                    continue
                source = result.stdout
                scope = ast.parse(source).body
                declaration = None
                for name in parts[split:]:
                    # Walk conditional blocks, never another declaration's scope.
                    matches = []
                    pending = list(scope)
                    while pending:
                        node = pending.pop()
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            if node.name == name:
                                matches.append(node)
                        else:
                            pending.extend(ast.iter_child_nodes(node))
                    if len(matches) != 1:
                        raise ValueError(f"Missing or ambiguous declaration: {symbol}")
                    declaration = matches[0]
                    scope = declaration.body
                assert declaration is not None
                start = declaration.lineno
                if declaration.decorator_list:
                    first = min(d.lineno for d in declaration.decorator_list)
                    # A parenthesized decorator's AST can start below its @ token.
                    start = max(
                        token.start[0]
                        for token in tokenize.generate_tokens(io.StringIO(source).readline)
                        if token.type == tokenize.OP
                        and token.string == "@"
                        and token.start[1] == declaration.col_offset
                        and token.start[0] <= first
                    )
                fragment = "".join(
                    source.splitlines(keepends=True)[start - 1 : declaration.end_lineno]
                )
                return SourceAnchor(
                    src=SourceRef(
                        repo=self.repo,
                        file=file,
                        line=declaration.lineno,
                        commit=self.commit,
                        symbol=symbol,
                    ),
                    source=textwrap.dedent(fragment),
                )
        raise ValueError(f"Cannot resolve declaration: {symbol}")


def resolve_target(target: str, sources: Mapping[str, SourceTree]) -> SourceAnchor:
    """Sources are keyed by Python package (vllm / vllm_ascend), not repo label."""
    package = target.split(".", 1)[0]
    if package not in sources:
        raise ValueError(f"No source tree configured for package: {package}")
    return sources[package].resolve(target)
