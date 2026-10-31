"""Source provenance, stale-note safety, and side-effect-free IR overlays."""

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from inferatlas.annotations import (
    AnchoredAnnotation,
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


def test_column_zero_comments_do_not_block_dedent_or_change_string_data():
    source = (
        "    def f(x):\n"
        '        """A multiline docstring.\n'
        "        # This is string data.\n"
        '        """\n'
        "        if x:\n"
        "            return x + 1\n"
    )
    commented = "# before\n" + source.replace("        if x:", "# body\n\n        if x:")
    assert anchor_hash(commented) == anchor_hash(source)
    assert anchor_hash(commented.replace("string data", "different data")) != anchor_hash(source)


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
    path.write_text(
        "# shifted source\n"
        + path.read_text().replace("        return x + 1", "# body comment\n        return x+1")
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
    [
        {"anchor_hash": "invented"},
        {"line": 10},
        {"level": "UNKNOWN"},
        {"applies_when": 42},
        {"anchors": [{"target": "sample.helper", "anchor_hash": "invented"}]},
        {"anchors": [{"target": "sample.helper", "anchor_hash": "sha256:" + "0" * 64, "line": 1}]},
    ],
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


@pytest.mark.parametrize("changed", ["Layer.forward", "outer.inner", "conditional"])
def test_all_anchors_are_linked_and_any_change_invalidates_note(source_tree, tmp_path, changed):
    primary = annotation_for(source_tree.resolve("sample.layer.Layer.forward"))
    helpers = [
        source_tree.resolve("sample.layer." + name) for name in ["outer.inner", "conditional"]
    ]
    data = {
        **primary.model_dump(),
        "anchors": [
            {"target": helper.src.symbol, "anchor_hash": helper.anchor_hash} for helper in helpers
        ],
    }
    path = tmp_path / "notes.yaml"
    with path.open("w") as stream:
        YAML().dump([data], stream)
    annotations = load_annotations(path)
    valid, stale = check_annotations(annotations, {"sample": source_tree})
    assert stale == []
    assert [ref.symbol for ref in valid[0].src] == sorted(
        [primary.target, *(helper.src.symbol for helper in helpers)]
    )

    code = source_tree.root / "sample/layer.py"
    before, after = {
        "Layer.forward": ("x + 1", "x + 2"),
        "outer.inner": ("'nested'", "'changed'"),
        "conditional": ("return True", "return False"),
    }[changed]
    code.write_text(code.read_text().replace(before, after))
    git(source_tree.root, "add", "sample/layer.py")
    git(
        source_tree.root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "changed anchor",
    )
    tree = SourceTree(
        source_tree.root, source_tree.repo, git(source_tree.root, "rev-parse", "HEAD")
    )
    valid, stale = check_annotations(annotations, {"sample": tree})
    assert valid == [] and len(stale) == 1
    assert stale[0].id == primary.id
    assert stale[0].anchor_hash_new == tree.resolve("sample.layer." + changed).anchor_hash
    assert stale[0].anchor_hash_old != stale[0].anchor_hash_new
    # Anchor input order never changes the stale record.
    annotations[0].anchors.reverse()
    assert check_annotations(annotations, {"sample": tree}) == (valid, stale)


def test_extra_anchors_reject_duplicates_and_missing_declarations(source_tree):
    primary = annotation_for(source_tree.resolve("sample.layer.Layer.forward"))
    duplicate = AnchoredAnnotation.model_validate(
        {
            **primary.model_dump(),
            "anchors": [{"target": primary.target, "anchor_hash": primary.anchor_hash}],
        }
    )
    with pytest.raises(ValueError, match="Duplicate annotation anchor"):
        check_annotations([duplicate], {"sample": source_tree})
    duplicate.anchors[0].target = "sample.layer.Removed"
    with pytest.raises(ValueError, match="Missing"):
        check_annotations([duplicate], {"sample": source_tree})


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
    assert item.src[0] in nodes["attn.qkv"].ascend.src
    assert (
        apply_annotations(result, [item], bindings=bindings, active_ids={item.annotation.id})
        == result
    )
    inactive = apply_annotations(result, [item], bindings=bindings, active_ids=set())
    assert inactive == result
    nodes["attn.qkv"].ascend.src[0].line += 1
    assert item.src[0] in nodes["attn.o"].ascend.src


def test_inactive_annotation_preserves_extracted_ascend_facts(source_tree):
    ir, item = overlay_for_ir(source_tree)
    node = next(node for node in ir.nodes if node.id == "attn.qkv")
    assert node.ascend is not None and node.ascend.src and "OOT" in node.ascend.tag
    result = apply_annotations(ir, [item], bindings={node.id: item.annotation.id}, active_ids=set())
    assert result == ir and result is not ir
    assert next(n for n in result.nodes if n.id == node.id).ascend is not node.ascend


def test_active_annotation_merges_existing_tags_and_extracted_source(source_tree):
    ir, item = overlay_for_ir(source_tree)
    node = next(node for node in ir.nodes if node.id == "attn.qkv")
    node.ascend.tag = ["QUANT", "OOT"]
    existing_sources = [src.model_copy(deep=True) for src in node.ascend.src]
    item.annotation.tag = ["PATCH"]
    before = ir.model_dump_json(by_alias=True)
    result = apply_annotations(
        ir, [item], bindings={node.id: item.annotation.id}, active_ids={item.annotation.id}
    )
    updated = next(n for n in result.nodes if n.id == node.id)
    assert updated.ascend.tag == ["OOT", "PATCH", "QUANT"]
    assert all(src in updated.ascend.src for src in [*existing_sources, *item.src])
    assert len(updated.ascend.src) == len(existing_sources) + len(item.src)
    assert updated.ascend.note == item.annotation.note
    assert updated.ascend.level == item.annotation.level
    assert updated.ascend.annotation_id == item.annotation.id
    assert ir.model_dump_json(by_alias=True) == before
    assert (
        apply_annotations(
            result, [item], bindings={node.id: item.annotation.id}, active_ids={item.annotation.id}
        )
        == result
    )
    updated.ascend.src[-1].line += 1
    assert ir.model_dump_json(by_alias=True) == before
    assert item.src[0].line == 5


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
