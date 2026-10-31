"""Source provenance, stale-note safety, and side-effect-free IR overlays."""

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from inferatlas.annotations import (
    ResolvedAnnotation,
    SourceTree,
    anchor_hash,
    apply_annotations,
    check_annotations,
    load_annotations,
    resolve_target,
)
from inferatlas.schema import IR, Annotation, Triage

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures/qwen3-30b-a3b/ir.json"


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def source_tree(tmp_path):
    git(tmp_path, "init", "-q")
    package = tmp_path / "sample"
    package.mkdir()
    (package / "__init__.py").write_text("class Exported:\n    pass\n")
    (package / "layer.py").write_text(
        'raise RuntimeError("AST resolution must never import this module")\n'
        "\nclass Layer:\n"
        "    @decorator\n"
        "    async def forward(self, x):\n"
        "        return x + 1  # implementation\n"
        "\nif enabled:\n"
        "    def conditional():\n"
        "        return True\n"
        "\ndef outer():\n"
        "    def inner():\n"
        "        return 'nested'\n"
        "    return inner\n"
    )
    git(tmp_path, "add", ".")
    git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "source",
    )
    return SourceTree(tmp_path, "vllm-ascend", git(tmp_path, "rev-parse", "HEAD"))


def annotation_for(anchor, id="example"):
    return Annotation(
        id=id,
        target=anchor.src.symbol,
        resolves_to=anchor.src.symbol,
        anchor_hash=anchor.anchor_hash,
        applies_when="true",
        level="KERNEL",
        tag=["PATCH", "OOT", "PATCH"],
        note="测试标注",
        reviewed_by="pending",
    )


@pytest.mark.parametrize(
    "source",
    [
        "def f(x):\n    return x + 1\n",
        "# before\ndef f( x ): # signature\n    # body\n    return x+1 # after\n",
        "    def f(x):\n        return (\n            x + 1\n        )\n",
        "def f(x):\n\n\treturn x + 1",
    ],
)
def test_hash_ignores_formatting(source):
    # Parentheses are actual tokens and therefore not a formatting-only change.
    reference = (
        "def f(x):\n    return (x + 1)\n" if "return (" in source else "def f(x):\n    return x+1\n"
    )
    assert anchor_hash(source) == anchor_hash(reference)


@pytest.mark.parametrize(
    "before,after",
    [
        ("def f(x):\n    return x+1", "def f(x):\n    return x+2"),
        ("def f():\n    return '# data'", "def f():\n    return '# other'"),
        ("@first\ndef f(): pass", "@second\ndef f(): pass"),
        ("def f():\n    '''doc'''\n    pass", "def f():\n    '''changed'''\n    pass"),
        (
            "def f(x):\n    if x:\n        x()\n    return x",
            "def f(x):\n    if x:\n        x()\n        return x",
        ),
    ],
)
def test_hash_changes_on_code_or_scope_changes(before, after):
    assert anchor_hash(before) != anchor_hash(after)


def test_ast_resolution_is_immutable_and_does_not_import(source_tree):
    symbol = "sample.layer.Layer.forward"
    anchor = resolve_target(symbol, {"sample": source_tree})
    assert anchor.src.line == 5  # Declaration, not the preceding decorator.
    assert anchor.src.file == "sample/layer.py"
    assert anchor.src.commit == source_tree.commit
    assert anchor.source.startswith("@decorator\nasync def forward")
    (source_tree.root / "sample/layer.py").write_text("broken working tree\n")
    assert source_tree.resolve(symbol) == anchor


@pytest.mark.parametrize(
    "symbol", ["sample.Exported", "sample.layer.conditional", "sample.layer.outer.inner"]
)
def test_resolves_packages_conditionals_and_nested_functions(source_tree, symbol):
    assert source_tree.resolve(symbol).src.symbol == symbol


@pytest.mark.parametrize(
    "symbol", ["sample.layer.Missing", "sample.layer.Layer.absent", "sample.layer", "../escape.f"]
)
def test_resolution_fails_closed(source_tree, symbol):
    with pytest.raises(ValueError):
        source_tree.resolve(symbol)


def test_unknown_source_package_fails(source_tree):
    with pytest.raises(ValueError, match="No source tree"):
        resolve_target("unknown.Class", {"sample": source_tree})


def test_multiline_decorators_are_included_in_anchor(source_tree):
    path = source_tree.root / "sample/decorated.py"
    path.write_text("@(\n    decorator\n)\nclass Layer:\n    pass\n")
    git(source_tree.root, "add", ".")
    git(
        source_tree.root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "decorator",
    )
    tree = SourceTree(
        source_tree.root, source_tree.repo, git(source_tree.root, "rev-parse", "HEAD")
    )
    anchor = tree.resolve("sample.decorated.Layer")
    assert anchor.source == path.read_text()
    assert anchor.anchor_hash == anchor_hash(path.read_text())
    assert anchor.src.line == 4


def test_comment_only_commit_keeps_annotation_valid_and_updates_source_line(source_tree):
    annotation = annotation_for(source_tree.resolve("sample.layer.Layer.forward"))
    path = source_tree.root / "sample/layer.py"
    path.write_text("# shifted source\n" + path.read_text().replace("x + 1", "x+1"))
    git(source_tree.root, "add", ".")
    git(
        source_tree.root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "format",
    )
    tree = SourceTree(
        source_tree.root, source_tree.repo, git(source_tree.root, "rev-parse", "HEAD")
    )
    valid, stale = check_annotations([annotation], {"sample": tree})
    assert stale == []
    assert valid[0].src[0].line == 6
    assert valid[0].src[0].commit == tree.commit


def test_distinct_replacement_is_resolved_and_linked(source_tree):
    annotation = annotation_for(source_tree.resolve("sample.layer.Layer.forward"))
    annotation.resolves_to = "sample.Exported"
    valid, stale = check_annotations([annotation], {"sample": source_tree})
    assert stale == []
    assert [src.symbol for src in valid[0].src] == ["sample.Exported", annotation.target]
    annotation.resolves_to = "sample.Removed"
    with pytest.raises(ValueError):
        check_annotations([annotation], {"sample": source_tree})


def test_ambiguous_conditional_declarations_fail(source_tree):
    (source_tree.root / "sample/ambiguous.py").write_text(
        "if enabled:\n    def f(): pass\nelse:\n    def f(): pass\n"
    )
    git(source_tree.root, "add", ".")
    git(
        source_tree.root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "ambiguity",
    )
    changed = SourceTree(
        source_tree.root, source_tree.repo, git(source_tree.root, "rev-parse", "HEAD")
    )
    with pytest.raises(ValueError, match="ambiguous"):
        changed.resolve("sample.ambiguous.f")


def test_load_schema_sorting_and_duplicate_ids(tmp_path):
    fixture = json.loads((ROOT / "fixtures/annotation.json").read_text())
    yaml = YAML()
    for filename, id in [("z.yaml", "first"), ("a.yml", "last")]:
        with (tmp_path / filename).open("w") as stream:
            yaml.dump([{**fixture, "id": id}], stream)
    assert [a.id for a in load_annotations(tmp_path)] == ["first", "last"]
    with (tmp_path / "duplicate.yaml").open("w") as stream:
        yaml.dump([{**fixture, "id": "first"}], stream)
    with pytest.raises(ValueError, match="Duplicate"):
        load_annotations(tmp_path)


@pytest.mark.parametrize(
    "change",
    [{"anchor_hash": "invented"}, {"line": 10}, {"level": "UNKNOWN"}, {"applies_when": 42}],
)
def test_invalid_yaml_annotations_rejected(tmp_path, change):
    fixture = json.loads((ROOT / "fixtures/annotation.json").read_text())
    path = tmp_path / "invalid.yaml"
    with path.open("w") as stream:
        YAML().dump([{**fixture, **change}], stream)
    with pytest.raises(ValidationError):
        load_annotations(path)


def test_stale_detection_across_commits_and_schema(source_tree):
    symbol = "sample.layer.Layer.forward"
    annotation = annotation_for(source_tree.resolve(symbol))
    sources = {"sample": source_tree}
    valid, stale = check_annotations([annotation], sources)
    assert len(valid) == 1 and stale == []
    path = source_tree.root / "sample/layer.py"
    path.write_text(path.read_text().replace("x + 1", "x + 2"))
    git(source_tree.root, "add", ".")
    git(
        source_tree.root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "behavior",
    )
    sources["sample"] = SourceTree(
        source_tree.root, source_tree.repo, git(source_tree.root, "rev-parse", "HEAD")
    )
    valid, stale = check_annotations([annotation], sources)
    assert valid == [] and len(stale) == 1
    assert stale[0].anchor_hash_new != stale[0].anchor_hash_old
    assert stale[0].id == annotation.id
    snapshot = IR.model_validate_json(FIXTURE.read_text()).snapshot
    assert Triage(snapshot=snapshot, items=stale).items[0].kind == "stale_annotation"
    with pytest.raises(ValueError, match="Duplicate"):
        check_annotations([annotation, annotation], sources)
    missing = annotation.model_copy(update={"target": "sample.layer.Removed"})
    with pytest.raises(ValueError):
        check_annotations([missing], sources)


def overlay_for_ir(source_tree):
    ir = IR.model_validate_json(FIXTURE.read_text())
    anchor = source_tree.resolve("sample.layer.Layer.forward")
    annotation = annotation_for(anchor)
    ref = anchor.src.model_copy(update={"commit": ir.snapshot.ascend.commit})
    return ir, ResolvedAnnotation(annotation, (ref,))


def test_apply_is_pure_conditional_deterministic_and_idempotent(source_tree):
    ir, item = overlay_for_ir(source_tree)
    before = ir.model_dump_json(by_alias=True)
    bindings = {"attn.qkv": item.annotation.id, "attn.o": item.annotation.id}
    result = apply_annotations(ir, [item], bindings=bindings, active_ids={item.annotation.id})
    assert ir.model_dump_json(by_alias=True) == before
    assert result is not ir
    assert IR.model_validate(result.model_dump(by_alias=True)) == result
    nodes = {n.id: n for n in result.nodes}
    assert nodes["attn.qkv"].ascend.annotation_id == item.annotation.id
    assert nodes["attn.qkv"].ascend.tag == ["OOT", "PATCH"]
    assert nodes["attn.qkv"].ascend.src[0].line == item.src[0].line
    assert (
        apply_annotations(result, [item], bindings=bindings, active_ids={item.annotation.id})
        == result
    )
    inactive = apply_annotations(result, [item], bindings=bindings, active_ids=set())
    assert all(n.ascend is None for n in inactive.nodes if n.id in bindings)
    nodes["attn.qkv"].ascend.src[0].line += 1
    assert nodes["attn.o"].ascend.src[0].line == item.src[0].line


@pytest.mark.parametrize(
    "bindings,active",
    [({"missing": "example"}, {"example"}), ({"attn.qkv": "stale"}, set()), ({}, {"stale"})],
)
def test_apply_rejects_unknown_and_stale_bindings(source_tree, bindings, active):
    ir, item = overlay_for_ir(source_tree)
    with pytest.raises(ValueError):
        apply_annotations(ir, [item], bindings=bindings, active_ids=active)


def test_apply_rejects_wrong_snapshot_and_duplicate_annotations(source_tree):
    ir, item = overlay_for_ir(source_tree)
    bad = ResolvedAnnotation(
        item.annotation, (item.src[0].model_copy(update={"commit": "0" * 40}),)
    )
    with pytest.raises(ValueError, match="snapshot"):
        apply_annotations(
            ir, [bad], bindings={"attn.qkv": item.annotation.id}, active_ids={item.annotation.id}
        )
    with pytest.raises(ValueError, match="Duplicate"):
        apply_annotations(ir, [item, item], bindings={}, active_ids=set())
