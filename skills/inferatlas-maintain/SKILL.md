---
name: inferatlas-maintain
description: Maintain InferAtlas rules and Ascend annotations for upstream releases, new model onboarding, golden failures, and stale or conditional overlays.
---

# InferAtlas maintenance

Use this skill when a maintainer asks to handle InferAtlas triage or update its source-based rules and Ascend overlays. Read `docs/design.md`, `docs/dev-plan.md`, the relevant schemas and triage before editing. Source at the snapshot's paired commits is authoritative; patch comments explain intent but are not a complete target inventory.

## Hard constraints

- Author only rules YAML, annotations YAML, and PR text. Run deterministic scripts to generate other artifacts; do not hand-edit source references, IR, numeric results, catalog data, or golden outputs. Describe required changes outside those outputs in PR text.
- Never write source line numbers or numeric values. Refer to qualified symbols and use symbolic expressions; scripts obtain config values, quantities, hashes and source locations. Mathematical coefficients also come from existing rules or source-backed generation, never guesses. Digits within existing symbol names, dtype names and script-computed hashes are identifiers/provenance, not authored measurements.
- Resolve source symbols through the generator. Include external implementations of the described behavior as independently hashed `anchors` in the shared Annotation contract. After reviewing a source change, run a deterministic script using `SourceTree.resolve(target).anchor_hash` at the paired snapshot commits to recompute the primary hash and every additional anchor for each affected annotation. Never patch hashes by value, including search/replace or copying `anchor_hash_new` from triage: hashes can be shared by different symbols or annotations. Reread the changed implementation identified by the stale item's `target` before regenerating all its anchors; inspect every anchor if a legacy item omits `target`.
- Preserve unknowns: missing predicates, unmatched classes, and unresolved branches stay in triage. Do not use an unconditional annotation to suppress a conditional override. Never interpret absent evidence as zero memory or no communication.
- Keep output deterministic: stable ids, sorted YAML keys and entries, sorted tags, no timestamps or host paths. Set `reviewed_by: pending` for drafts; never claim another person's approval.
- Use CPU/meta extraction and the existing upstream mock environment. Do not download weights or run NPU operations. Do not add a Python expression evaluator: predicates and quantities use the shared TypeScript evaluator.
- A nonempty triage blocks the affected model's publication. Produce PR text for maintainer review; publishing, messaging, and merging are outside this skill's workflow.

## New upstream version

Run `uv run inferatlas resolve` in `generator/` to establish a compatible pair, then `scripts/scan.sh` and `scripts/triage.sh` with the CLI's snapshot/model arguments. Read the upstream diff for each reported symbol. Update only rules or annotations justified by that diff, using [rule-schema.md](references/rule-schema.md) and [ascend-overlay.md](references/ascend-overlay.md). Run `scripts/verify.sh` and `scripts/golden_diff.sh`; explain unresolved cases and verification in draft PR text.

## New model onboarding

Use the config already pinned by the maintainer in the catalog; resolving may obtain config metadata, never weights. Run resolve, scan and triage. Read [vllm-parallel-layers.md](references/vllm-parallel-layers.md) for unknown reachable classes and add symbolic rules using the schema reference. Have the generator create the initial golden; compare extraction shapes and the accounting in [memory-accounting.md](references/memory-accounting.md). Include the evidence and remaining gaps in PR text for human review.

## Golden failure

Run `scripts/golden_diff.sh` and read the changed source, rule and config provenance. Use the memory reference to distinguish actual upstream changes from broken rules or double counting. For a real change, ask the deterministic golden command to regenerate the candidate and explain it in PR text; for a rule regression, add the appropriate version-scoped rule. Never edit a golden number to make a check pass. Rerun verify and the golden comparison.

## Ascend overlay review

Read the runtime identity diff, the actual patch/replacement, and the upstream implementation. Use the overlay reference to select significance, badge tags, and declared `applies_when` predicates. Recheck all stale anchors, including decorators and nested implementations. Generate hashes and locations with the annotation module, write Chinese notes explaining the source-observed difference, and run verify. Use `scripts/preview.sh` to inspect the generated overlay if the site command is available. State review status and conditional limitations in PR text.

## Command wrappers

The scripts locate the repo and run the corresponding `uv run inferatlas` command from `generator/`, forwarding arguments unchanged: `scan.sh` → `extract`, `triage.sh` → `lower`, `verify.sh` → `verify`, `golden_diff.sh` → `golden`, `preview.sh` → `site`. Resolve first; `uv run inferatlas build` performs the composed pipeline. Check `--help` for the installed CLI flags; wrappers do not invent a second CLI or interpret its results.

Use `UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` for dependency installation. If a JS command is needed, use pnpm workspaces with `npm_config_registry=https://registry.npmmirror.com`.
