# C2 full catalog release

The verified source pair is vLLM
`544f182d9f0c9bfe8bd7645525951d5e3692a542` / vllm-ascend
`24a76e84394ae3427195e3715fc08e90769d398c`. Generation uses Python 3.12,
CPU/meta tensors and upstream Ascend mocks, without weights or accelerator kernels.

The all-device review build covers **3 models × 5 devices**: **4 ok, 11 triage,
0 hard errors**. See [C2 progress](progress/C2.md) for the cell-by-cell causes,
validation and hosted runs. H3's earlier A2/H100-only evidence remains in
[H3 progress](progress/H3.md) and `docs/screenshots/release/`; those screenshots
and `evidence.json` describe H3, not this expanded matrix.

## Matrix and outputs

| Model | 310P | A2 | A3 | A5 | H100 |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-30B-A3B | triage | ok, TP2 | triage | triage | ok, TP1 |
| Qwen/Qwen3-8B | triage | ok, TP1 | triage | triage | ok, TP1 |
| deepseek-ai/DeepSeek-V3 | triage | triage | triage | triage | triage |

Qwen's 310P implementations need distinct rules and annotations. A3's observed
replacements lack applicable annotations; existing predicates explicitly require
A2. A5's board SKU and memory capacity are unconfirmed: after extraction and
lowering, `hardware_profile_incomplete` blocks numeric publication while preserving
unannotated `conditional_override` items. Direct lowerer calls use the same gate,
with zero as an explicit unknown-capacity sentinel, never a 64 GB default.
DeepSeek's Ascend construction (including A5) requests an NPU allocation under
CPU mocks; its H100 baseline reaches unsupported MLA lowering.
All these cells exit **2**, retain diagnostics and publish no numeric data.
Annotations remain `reviewed_by: pending`.

`release` defaults remain **all catalog models × A2/H100**, because Qwen does
not yet build cleanly on every Ascend device. Explicitly select all devices for
the full audit. No goldens were regenerated: the supported cells' IR and numeric
formulas are unchanged. The device-aware TP rule still selects the smallest
valid TP with positive estimated KV capacity: MoE A2 uses TP2, all other usable
cells use TP1. These are estimates without hardware calibration.

Uncommitted generated outputs:

- `dist/index.html`: offline hub, 4 usable links and 11 unlinked triage cells.
- `dist/inferatlas-qwen3-30b-a3b-{a2,h100}.html` and
  `dist/inferatlas-qwen3-8b-{a2,h100}.html`: four device-qualified offline exports.
- `dist/.inferatlas/`: private manifests for all 15 attempted cells.
- `web/dist/`: complete Pages site; its catalog includes the unbuilt `v0.32.0`
  pair, so the full site matrix is 4 ok / 11 triage / 15 missing.

Start with a fresh output directory. Do not copy stale unsupported exports from
an older build: the public HTML set is exactly the hub plus the four Qwen files.
Each single file inlines its model data and returns to the sibling hub through
**All models**. Pages keeps its hash routes and includes numeric bundles only
for the four ok cells.

## Reproduce

From the repository root:

```bash
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export npm_config_registry=https://registry.npmmirror.com
export INFERATLAS_SRC=/home/cyberspace/projects/InferAtlas/experiments/a2_qwen3moe/src
export INFERATLAS_PY=/home/cyberspace/projects/InferAtlas/experiments/a2_qwen3moe/.venv/bin/python
# Both pins are required together, only for --pair main-verified (the default).
# The Ascend pin must declare exactly the vLLM pin in its main-verified pairing file.
# Tagged pairs reject pin overrides; unset both pins to resolve a catalog release.
export INFERATLAS_VLLM_COMMIT=544f182d9f0c9bfe8bd7645525951d5e3692a542
export INFERATLAS_ASCEND_COMMIT=24a76e84394ae3427195e3715fc08e90769d398c
pnpm install --frozen-lockfile
uv sync --project generator --python 3.12
pnpm --filter '@inferatlas/*' -r build

# Expected exit 2 after rendering the hub and four exports.
uv run --project generator inferatlas release \
  --devices 310P A2 A3 A5 H100 --out dist
# Or a fresh temporary output: --out /tmp/inferatlas-c2-release
# Default subset remains: just release
# Refresh existing exports without extraction: just hub --dir dist
```

Pages must use the same all-device results. Stage blocked cells as triage-only
directories, copying their original diagnostics unchanged. This avoids exposing
partial numeric IR from unsupported lowering:

```bash
uv run --project generator python - <<'PY'
import shutil
from pathlib import Path
from inferatlas.schema import Report, Triage
from inferatlas.verify import exit_code
from inferatlas.verify.io import read_json, write_json

root = Path('.cache/release/site-builds')
shutil.rmtree(root, ignore_errors=True)
for path in sorted(Path('.cache/build').glob('*/*/triage.json')):
    triage = Triage.model_validate(read_json(path))
    target = root / path.parent.parent.name / path.parent.name
    if triage.items:
        write_json(target / 'triage.json', triage)
    else:
        report = Report.model_validate(read_json(path.with_name('report.json')))
        assert exit_code(report, triage) == 0
        shutil.copytree(path.parent, target)
PY

# Expected exit 2 after a successful Pages build with 11 triage cells.
uv run --project generator inferatlas site \
  --build-root .cache/release/site-builds \
  --data-dir .cache/release/site-data --include-blocked

just lint && just test
INFERATLAS_RELEASE=1 pnpm --filter web e2e
```

Use a build cache containing one build per version/model/device cell; archive
older pin sets before staging. The release extracts TP1/4. Existing acceptance
tests and committed numeric golden grids also cover TP1/2/4/8. Default selection
uses the shared Node expression evaluator without an additional extraction.

## Hosted validation and publication boundary

See [CI documentation](ci.md) for dispatch commands and the verified run.
`build.yml` uses the same 15-cell catalog matrix. It now honors both dispatched
SHAs, builds its evaluator/layout prerequisites and handles fresh source-only
vLLM version metadata through upstream SCM versioning.

The hosted collector gates each model×device cell. A triaged cell is blocked;
H100 baseline triage also blocks all Ascend cells of that model. Missing or error
cells block the whole publication. The full audit yields `clean=4`, `errors=[]`
and 11 `blocked_cells`, matching the four local Qwen A2/H100 exports.
`blocked_cells` is retained in both bundle summary and publication state;
`blocked_models` remains an aggregate for maintenance retries. `publish=false`
prevents deployment and state updates. Publication and scheduled automation remain disabled.

Remaining gaps are MLA, 310P rules/annotations and dtype validation, A3 annotations,
a confirmed A5 profile, calibration, and a second real version pair. Null dtype
peaks suppress unavailable rooflines; they do not establish dtype support.

Source-only main version detection uses `setuptools-scm==10.3.4`. In a shallow,
tagless checkout SCM reports a placeholder such as `0.1.dev1`; this is only an
Ascend import compatibility value, not a claimed upstream release. Snapshot
identity remains the validated source SHAs. Release-tag version overrides are
used only for resolved release sources. Local SCM metadata can differ with Git
history; environment facts and snapshot identity never substitute that placeholder
for a source revision.
