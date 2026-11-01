# H3 release artifacts

Regenerated with F3's offline landing/navigation and G1's release CLI on the
verified pair vLLM `544f182d9f0c9bfe8bd7645525951d5e3692a542` /
vllm-ascend `24a76e84394ae3427195e3715fc08e90769d398c`.
Extraction uses Python 3.12, CPU/meta tensors and upstream Ascend mocks.
No weights were downloaded and no accelerator kernels ran.

## Changes since H2

- `just release` builds all catalog models × A2/H100 and an offline
  `dist/index.html` hub beside four device-qualified single files. Each export
  opens directly on its model; **All models** returns to the hub.
- Build defaults use the shared Node evaluator to select the smallest valid TP
  choice whose existing `memory.kv_tokens` formula is positive on the snapshot
  device, keeping every other default. If no choice fits, the smallest valid TP
  remains selected and the zero-capacity warning remains truthful. The rule is
  recorded in the IR TP description and report's `default_parameters` check.
- Missing Node keeps the extracted TP default and records a skipped
  `default_parameters` check for fixture builds. Real evaluator failures propagate;
  only declared parameter domain/constraint violations skip a TP candidate.
- MoE A2 now opens at TP=2 with **471,936 KV-capacity tokens**, rather than TP=1
  with zero capacity. MoE H100 stays at TP=1 (80,128 tokens); dense A2/H100 stay
  at TP=1 (253,312 / 356,352 tokens). These are formula estimates, not calibration.
- Four IR goldens record the selection rule; only MoE A2 changes its TP default.
  Re-evaluating every existing TP grid left all four `eval.json` files unchanged.
  Memory formulas, rules, layouts and public contracts are unchanged.
- DeepSeek A2 now has a real `extraction_failure` record with stage, exception
  class and normalized upstream message. Site deep links say “Needs triage” and
  state the unresolved count and absence of numeric data. Blocked site/hub cells
  have no links.

## Generated matrix and files

| Main-verified model | A2 | H100 |
| --- | --- | --- |
| Qwen/Qwen3-30B-A3B | verified; default TP=2 | verified; default TP=1 |
| Qwen/Qwen3-8B | verified; default TP=1 | verified; default TP=1 |
| deepseek-ai/DeepSeek-V3 | triage: 1 `extraction_failure` | triage: 668 lowering items; numeric checks/layout skipped |

DeepSeek differs by device: A2's mock rotary-cache constructor requests an NPU
allocation and raises `NotImplementedError`. H100 baseline extraction succeeds,
then unsupported MLA lowering produces 668 blocking triage items. As required by
design §5.5, weights, golden, evaluation, calibration and layout checks skip
with “Blocked by triage”; no view is generated and default selection does not
run. `release` exits **2**, after producing the hub and four verified exports.
Source-link and provenance failures still exit 3. Neither DeepSeek cell publishes
numeric data.

The Pages matrix has 4 `ok`, 2 `triage`, and 24 `missing` cells. The hub includes
the selected main-verified pair: 4 `ok`, 2 `triage`, and 9 `missing` cells. Other
devices and the `v0.32.0` pair were not built. `dist/` has exactly five public HTML files;
private `.inferatlas/` sidecars record all six attempted exports. There are no
DeepSeek files or device-less stale copies.

| Local output (gitignored) | Bytes |
| --- | ---: |
| `web/dist/` — complete Pages site, 27 files | 2,103,567 |
| `web/dist/index.html` | 871 |
| `web/dist/data/index.json` | 6,177 |
| `dist/index.html` — offline hub | 709,082 |
| `dist/inferatlas-qwen3-30b-a3b-a2.html` | 1,829,969 |
| `dist/inferatlas-qwen3-30b-a3b-h100.html` | 1,803,520 |
| `dist/inferatlas-qwen3-8b-a2.html` | 1,811,595 |
| `dist/inferatlas-qwen3-8b-h100.html` | 1,789,171 |

The Pages site retains its existing hash routes (`/#/`,
`/#/m/main-verified/A2/qwen3-8b`, `/#/diff`). It contains one shell and 14 data
JSON files: index, hardware, and four IR/view/report bundles. Each offline export
inlines scripts, styles, fonts and exactly one model/device bundle. The hub
inlines only its matrix, hardware and export mapping. Config fingerprints,
artifact hashes, default checks and verification evidence are recorded in
[evidence.json](screenshots/release/evidence.json). Outputs remain local;
`dist/` and `web/dist/` are not committed or deployed.

## Reproduce

Run from the repository root:

```bash
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export npm_config_registry=https://registry.npmmirror.com
export INFERATLAS_SRC=/home/cyberspace/projects/InferAtlas/experiments/a2_qwen3moe/src
export INFERATLAS_PY=/home/cyberspace/projects/InferAtlas/experiments/a2_qwen3moe/.venv/bin/python
pnpm install --frozen-lockfile
uv sync --project generator --python 3.12
pnpm --filter '@inferatlas/*' -r build

# All catalog models × A2/H100; expected exit 2 for blocked DeepSeek triage.
just release
# Equivalent: uv run --project generator inferatlas release --out dist
# Optional subset: just release --models qwen3-8b --devices A2 H100
# Refresh the hub from existing sidecars: just hub --dir dist
```

Before copying/sharing `dist/`, remove old HTML files that are absent from the
current release's export mapping. H3 started with no generated output and checked
that the final public HTML set is exactly the hub plus the four Qwen exports.
Do not reuse an old DeepSeek page or a device-less export from H1.

Build Pages from the same release results. As in H2, stage blocked cells as
triage-only directories. This retains the existing status-only staging procedure
and copies real triage without editing reports or inventing blockers:

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

# Expected exit 2: successful site build with two blocked cells.
uv run --project generator inferatlas site \
  --build-root .cache/release/site-builds \
  --data-dir .cache/release/site-data --include-blocked
```

The release command extracts TP 1/4 for each Qwen cell. The committed goldens
and real extraction acceptance tests cover TP 1/2/4/8; no extra extraction is
needed just to choose the first-view TP, which uses the symbolic memory formula.

## Verification and screenshot review

```bash
just lint && just test
pnpm --filter web exec tsc -p tsconfig.json
INFERATLAS_RELEASE=1 pnpm --filter web e2e
```

The browser inventory covers site matrix/status messages, model routes, initial
graph fit at four desktop sizes, light/dark themes, overlay comparison,
drill-down/keyboard navigation, EP/TP coupling, dtype annotation conditions,
TP recompute, URL state, analysis tabs, and the honest unavailable diff state.
The real hub is opened over file:// at 1280×800, checked in both themes, and
followed to every named single file and back through “All models”. Each export
opens on the embedded model without a hash and selects its device-aware TP.
The Pages MoE A2 route also explicitly asserts default TP2 and no zero-capacity warning.
Offline navigation and recomputation reject HTTP(S) requests, console errors
and page exceptions. The intentional site DeepSeek deep link allows exactly its
status-derived loader console error; additional errors fail the test.

- Python: 811 passed, including the real CPU/meta extraction acceptance cases.
- Workspace: contracts 2, expressions 499, layout 25 and web 45 passed.
  Ruff/Biome and TypeScript passed.
- Chromium release suite: 32 passed; two demo-only cases skipped. All four
  hub → export → hub round trips and all four offline recompute cases passed
  with zero console errors, page exceptions or HTTP(S) requests.
- Every published report has no failed checks, golden differences or source-link
  failures. Calibration is explicitly skipped; every published extract has
  `fixture: false`.
- Re-rendering each single file from its original inline inputs, refreshing the
  hub and rebuilding Pages produced byte-identical output: four single files,
  one hub and all 27 site files.

All release PNGs were regenerated and opened for visual inspection. The hub
shows four readable export links and two unlinked DeepSeek triage cells for the
selected version pair. The default MoE A2 view shows TP=2.
At 1280×800 the graph and routed edges fit before Fit, analysis is collapsed to
tabs, and header/breadcrumb controls do not overlap. Wide subgraph labels remain
small at this width; users can zoom for detail.

| Evidence | Observation |
| --- | --- |
| [Hub 1280](screenshots/release/hub-1280.png), [dark](screenshots/release/hub-dark-1280.png) | Four sibling exports; honest unlinked DeepSeek triage |
| [Home A2](screenshots/release/home-a2.png), [H100](screenshots/release/home-h100.png) | Four usable Qwen cells across the two devices |
| [Light](screenshots/release/model-light.png), [dark](screenshots/release/model-dark.png) | Whole L0 graph; MoE A2 default TP=2 |
| [A2 overlay](screenshots/release/model-overlay-a2.png) | Norm replacement and comparison card |
| [1280 initial](screenshots/release/model-1280.png), [after Fit](screenshots/release/model-1280-fit.png) | Whole graph visible before interaction |
| [MoE A2 TP4](screenshots/release/single-qwen3-30b-a3b-a2-tp4.png), [dense A2 TP4](screenshots/release/single-qwen3-8b-a2-tp4.png) | Offline parameter recompute |
| [MoE H100 TP4](screenshots/release/single-qwen3-30b-a3b-h100-tp4.png), [dense H100 TP4](screenshots/release/single-qwen3-8b-h100-tp4.png) | H100 preset; no Ascend overlay |
| [Diff](screenshots/release/diff-unavailable.png) | “Nothing to compare” without synthetic data |

## Known limitations

DeepSeek/MLA remains unsupported. There is only one real version pair per usable
cell, so a real two-version diff is unavailable. A3/A5/310P are unbuilt. H100 is
a hardware preset over CPU baseline facts, not measured H100 execution.
Activation/runtime/allocator deductions are estimates; no NPU/GPU calibration
was performed. Ascend annotations remain drafts marked `reviewed_by: pending`.
Configs use catalog-pinned inline hashes rather than newly fetched HF revisions.

Browser evidence uses Chromium on desktop sizes; other engines, mobile, every
parameter combination and hosted Pages deployment were not checked. Immutable
source links are validated against the pinned local clones; live upstream HTTP
fetching was not checked. The Playwright runner is used because this session
has no `js_repl` tool.
