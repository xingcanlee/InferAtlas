# H2 release artifacts

Regenerated from main `f849934` (F2 graph-fit and review fixes) plus H2's
small release changes, following the real H1 pipeline.
The verified pair is vLLM `544f182d9f0c9bfe8bd7645525951d5e3692a542` /
vllm-ascend `24a76e84394ae3427195e3715fc08e90769d398c`.
All extraction used Python 3.12, CPU/meta tensors and the upstream Ascend mocks.
No model weights were downloaded and no accelerator kernels ran.

Since H1, F2 fits graph content on first load, collapses the analysis panel on
short windows, and prevents header overlap. H2 regenerates that UI with real data,
uses device-qualified default filenames, documents null triage data paths, and
fails release tests on console errors as well as page exceptions. H100 standalone
exports and their offline checks are now included alongside A2.

## Generated matrix and files

| Main-verified model | A2 | H100 |
| --- | --- | --- |
| Qwen/Qwen3-30B-A3B | verified | verified |
| Qwen/Qwen3-8B | verified | verified |
| deepseek-ai/DeepSeek-V3 | triage: 1 recorded baseline MLA blocker; Ascend extraction also fails | triage: 668 lowering items; weight verification and layout fail |

Each Qwen build extracted TP 1/2/4/8. Its rank-local weight bytes match the
symbolic IR exactly at all four points; goldens and immutable source links pass.
The MoE model has 30,532,122,624 parameters at TP 1. H100 identifies the reference
hardware preset applied to CPU baseline facts, not a measured H100 execution.

The index contains 4 `ok`, 2 `triage` and 24 `missing` cells. The other catalog
devices and the `v0.32.0` version pair were not generated. Both DeepSeek cells
have `data_path: null`: there is no DeepSeek numeric page or data bundle.

| Local output (gitignored) | Bytes |
| --- | ---: |
| `web/dist/` — complete Pages site, 27 files | 2,100,687 |
| `web/dist/index.html` | 871 |
| `web/dist/data/index.json` — version/model/device matrix | 6,177 |
| `dist/inferatlas-qwen3-30b-a3b-a2.html` | 1,827,374 |
| `dist/inferatlas-qwen3-8b-a2.html` | 1,809,000 |
| `dist/inferatlas-qwen3-30b-a3b-h100.html` | 1,800,924 |
| `dist/inferatlas-qwen3-8b-h100.html` | 1,786,574 |

The site has home, model and diff pages through the existing hash router
(`/#/`, `/#/m/main-verified/A2/qwen3-8b`, `/#/diff`). It uses one HTML shell,
as specified in the development plan, rather than separate HTML files per URL.
Its 14 data JSON files are index + hardware + four IR/view/report bundles.
There are no fixture or synthetic previous-version files in the release tree.
The four standalone HTMLs each inline exactly one model/device cell, including
scripts, styles, fonts and data. `web/dist-single/index.html` holds the last
standalone build (MoE A2). There are no device-less copies in `dist/`.

Exact artifact hashes, config fingerprints and all TP weight-check results are
in [evidence.json](screenshots/release/evidence.json). Build outputs are local
and are not committed or deployed to a hosted Pages environment.

## Reproduce

Run from the repository root. The equivalent CLI from `generator/` is
`uv run inferatlas ...`; paths in this transcript are relative to the root.

```bash
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export npm_config_registry=https://registry.npmmirror.com
export INFERATLAS_SRC=/home/cyberspace/projects/InferAtlas/experiments/a2_qwen3moe/src
export INFERATLAS_PY=/home/cyberspace/projects/InferAtlas/experiments/a2_qwen3moe/.venv/bin/python
pnpm install --frozen-lockfile
uv sync --project generator --python 3.12
just schemas
pnpm --filter '@inferatlas/*' -r build
mkdir -p .cache/release

for model in Qwen/Qwen3-30B-A3B Qwen/Qwen3-8B; do
  for device in A2 H100; do
    uv run --project generator inferatlas build --model "$model" --device "$device" \
      --tp 1 --tp 2 --tp 4 --tp 8
  done
done

# Expected exit 3 for each: retain the actual failures for investigation.
uv run --project generator inferatlas build --model deepseek-ai/DeepSeek-V3 \
  --device A2 --tp 1 --tp 2 --tp 4 --tp 8
uv run --project generator inferatlas build --model deepseek-ai/DeepSeek-V3 \
  --device H100 --tp 1 --tp 2 --tp 4 --tp 8
uv run --project generator inferatlas resolve --model deepseek-ai/DeepSeek-V3 \
  --device A2 --output .cache/release/deepseek-a2-snapshot.json
```

Stage the four verified builds separately from the failed DeepSeek H100 build.
H100's triage is copied unchanged. A2's single recorded blocker is the missing
baseline `MLAAttention` rule also observed on H100; it is a curated blocker,
not a successful A2 extraction or a complete A2 triage inventory. The A2 attempt
fails in Ascend's rotary-cache constructor before it can publish an extract.

```bash
uv run --project generator python - <<'PY'
import shutil
from pathlib import Path
from inferatlas.cli import snapshot_key
from inferatlas.schema import Snapshot, Triage
from inferatlas.verify.io import read_json, write_json

root = Path('.cache/release/site-builds')
shutil.rmtree(root, ignore_errors=True)
for path in sorted(Path('.cache/build').glob('*/*/triage.json')):
    triage = Triage.model_validate(read_json(path))
    target = root / path.parent.parent.name / path.parent.name
    if path.parent.name == 'deepseek-v3':
        h100 = triage
        write_json(target / 'triage.json', triage)
    else:
        report = read_json(path.with_name('report.json'))
        assert not triage.items
        assert all(c['status'] != 'fail' for c in report['checks'])
        shutil.copytree(path.parent, target)
snapshot = Snapshot.model_validate(read_json(Path('.cache/release/deepseek-a2-snapshot.json')))
blocker = next(i for i in h100.items if i.kind == 'unknown_layer_class'
               and i.symbol == 'vllm.model_executor.layers.attention.mla_attention.MLAAttention')
write_json(root / snapshot_key(snapshot, 'main-verified') / 'deepseek-v3/triage.json',
           Triage(snapshot=snapshot, items=[blocker]))
PY

# Each single command exits 0 and performs a fresh real TP 1/4 build.
# Defaults include device; leave MoE A2 last for the existing smoke test.
for device in H100 A2; do
  for model in Qwen/Qwen3-8B Qwen/Qwen3-30B-A3B; do
    uv run --project generator inferatlas single --model "$model" --device "$device"
  done
done

# Restore the full matrix after single-file staging. Expected exit 2 means
# the site was built successfully with blocked status cells; it is not all-ok.
uv run --project generator inferatlas site --build-root .cache/release/site-builds \
  --data-dir .cache/release/site-data --include-blocked
```

`site` without `--include-blocked` still refuses a blocked matrix. An IR/report
with failed verification still causes exit 3, even with that flag. Only the
explicit triage-only staging above omits failed numeric artifacts; no failed
report was edited into a pass. H1's triage-only publishing support keeps blocked
cells non-navigable. Empty triage-only directories are rejected. H2 changes the
CLI default export name,
removes the web plugin's device-less fallback copy, documents nullable triage
data paths in regenerated contracts, and strengthens browser checks. F2 supplies
the graph-fit/panel/header implementation; formulas, rules and goldens are unchanged.

## Verification and screenshot review

```bash
just lint
just test
pnpm --filter web exec tsc -p tsconfig.json
# Playwright starts and stops the Vite preview server on port 4318.
INFERATLAS_RELEASE=1 pnpm --filter web e2e
```

The QA inventory covered the requested matrix, four model/device routes,
light/dark switching, A2 overlay and comparison card, drill-down and keyboard
navigation, TP recompute, URL state, memory/KV/roofline/rank tabs, offline files,
and the available diff state. Exploratory cases exercised EP/TP coupling,
inactive dtype annotations, a direct blocked-model URL and a smaller desktop.

- Python: 732 tests passed, including the real CPU/meta acceptance cases.
- Node: contracts 2, expressions 499, layout 25 and web 36 tests.
- Browser: 30 tests passed in Chromium. All four device-qualified exports were
  opened with networking disabled, with no HTTP requests, console errors or page
  exceptions. Published model, matrix and diff cases reject console errors too.
  The intentional DeepSeek deep link asserts its one known loader console error
  (`deepseek-v3 @ main-verified on A2 has no artifacts (missing)`); all page
  exceptions and additional console errors still fail the test.
  Every A2 view fits before any Fit click at 1280×800, 1440×900, 1600×1000 and
  1920×1080, at zoom ≥0.6. Header boxes do not overlap at 1280×800.
  Normal TP-button input changed `attn.o` resident bytes from 16 to 4 MiB (MoE)
  and 32 to 8 MiB (dense); the corresponding memory ledgers also updated.
- All four published reports have empty golden/source-link failures and no
  failed checks. Calibration is explicitly skipped. Every published source
  extract has `fixture: false`.
- A repeat site build produced the same bytes for every output file. The
  standalone renderer was also rebuilt from each file's original inline inputs
  and compared byte-for-byte.

All twelve committed PNGs were opened and visually inspected, including initial
viewports and post-interaction states. The home matrix is readable and unclipped,
themes remain usable, and the selected A2 comparison card shows both
implementations. At **1280×800 the whole graph, including routed edges and the
decoder, is visible on first load without clicking Fit**, with the analysis panel
collapsed to its tabs. Header/breadcrumb/layer controls do not overlap. No broken
images or missing fonts were found. The widest graphs have small labels at this
width; zooming remains useful for reading details.

| Evidence | Observation |
| --- | --- |
| [Home A2](screenshots/release/home-a2.png), [H100](screenshots/release/home-h100.png) | Two usable Qwen cells and one non-clickable DeepSeek blocker per main row |
| [Light](screenshots/release/model-light.png), [dark](screenshots/release/model-dark.png) | Whole L0 graph and routed edges visible at the default camera |
| [A2 overlay](screenshots/release/model-overlay-a2.png) | Selected norm, replacement badges and readable Ascend comparison card |
| [1280×800 initial](screenshots/release/model-1280.png), [after fit view](screenshots/release/model-1280-fit.png) | Whole graph visible before Fit; no header overlap; analysis collapsed |
| [MoE offline TP4](screenshots/release/single-qwen3-30b-a3b-a2-tp4.png), [dense offline TP4](screenshots/release/single-qwen3-8b-a2-tp4.png) | Changed shard sizes, FLOPs and memory after normal input |
| [MoE H100 offline TP4](screenshots/release/single-qwen3-30b-a3b-h100-tp4.png), [dense H100 offline TP4](screenshots/release/single-qwen3-8b-h100-tp4.png) | Same offline recompute with the H100 preset and no Ascend overlay |
| [Diff](screenshots/release/diff-unavailable.png) | Honest “Nothing to compare” state |

## Known limitations

1. Widest graph labels are small on narrow desktops (about 9px in the
   1280×800 attention view; see [F2 measurements](progress/F2.md#measurements-real-build-data-main-verified-a2-first-load-nothing-selected)).
   The whole graph fits before interaction, and users can zoom for details.
2. DeepSeek is unsupported. A2's upstream mock constructor requests
   `device='npu'` for its rotary cache and raises `NotImplementedError` on CPU.
   H100 has unsupported MLA classes, heterogeneous layer structure, incomplete
   weight accounting and an over-budget layout. Neither publishes numeric data.
   Direct blocked-model URLs currently describe absent artifacts as “missing”
   even though the matrix correctly says “triage”.
3. There is only one real version per model/device. The other catalog version
   has no artifacts, so a real two-version diff cannot be verified. No synthetic
   snapshot was added to manufacture one.
4. There is no NPU/H100 measurement or deployment calibration. Activation and
   runtime/allocator deductions remain estimates; hardware presets retain their
   documented provisional fields. A2 annotations remain drafts marked
   `reviewed_by: pending`. Configs are catalog-pinned inline hashes, not freshly
   fetched HF revisions. MoE at TP 1 correctly warns that no KV cache fits the
   default A2 memory budget.
5. Browser verification used Chromium screenshots at 1600×1000 and 1280×800,
   plus fit assertions at 1440×900 and 1920×1080.
   Other browser engines, mobile layout, all parameter combinations, hosted
   Pages deployment and live HTTP fetching of upstream source links were not
   checked. Source file/line targets were validated against the pinned local
   clones. The Playwright runner was used because this session has no `js_repl`.
