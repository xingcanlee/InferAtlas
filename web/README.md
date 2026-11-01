# web — InferAtlas frontend (F1)

Vite + React 19 + TypeScript (strict), Tailwind CSS v4 + shadcn/ui (Radix), TanStack Router
(hash history, zod-validated search params), Zustand, @xyflow/react, shadcn Charts (Recharts),
vite-plugin-singlefile. Expressions are evaluated only by `@inferatlas/expr` (via `src/lib/expr.ts`).

```sh
# release (real pipeline output required — fails without it)
INFERATLAS_DATA_DIR=<dir> pnpm --filter web build          # dist/  (static site; data under dist/data)
INFERATLAS_DATA_DIR=<dir> pnpm --filter web build:single   # dist-single/index.html, works from file://
INFERATLAS_DATA_DIR=<dir> pnpm --filter web build:hub      # dist-hub/index.html: landing page for dist/

# local demo on contract fixtures (never for publication)
pnpm --filter web dev                 # Vite dev server
pnpm --filter web build:demo          # dist/ with fixture + synthetic demo data
pnpm --filter web build:single:demo   # Qwen3-30B-A3B main-verified A2 only
pnpm --filter web build:hub:demo      # hub over the demo index + demo hub.json
pnpm --filter web test                # Vitest (eval.json oracle, graph/diff/router tests)
pnpm --filter web e2e                 # demo builds + Playwright smoke + docs/screenshots/
pnpm --filter web lint                # Biome
```

`INFERATLAS_DATA_DIR` is the pipeline's public data directory: it contains `index.json`,
`hardware.json` (catalog `hardware_presets`) and every `data_path` the index names, each with
`ir.json`, `view.json`, `report.json` and optionally `arch.json` (structure; a cell may ship
`arch.json` alone). Relative paths resolve from where pnpm was invoked.
`scripts/sync-data.mjs` copies it to `public/data/`; without it (and without `--demo`) the sync
step exits 1, so a release build can never ship fixtures.

Demo mode (`--demo`, used by `dev`, `build:demo`, `e2e`) copies `fixtures/`, lays the views out
with the real build-time `packages/layout` CLI, and derives one clearly labelled synthetic
snapshot (`fixture-prev`, "synthetic · diff demo") so the diff page has two versions.
In single-file mode the Vite plugin inlines index + every bundle + hardware as
`window.__INFERATLAS__`; `INFERATLAS_SINGLE_MODEL=<model-id>` restricts it to one model.
The output is `dist-single/index.html`; `INFERATLAS_SINGLE_OUTPUT` requests an additional copy.
The generator CLI sets that path and defaults to `dist/inferatlas-<model>-<device>.html`.
`INFERATLAS_SINGLE_VERSION=<version-id>` optionally restricts the export to one version pair.
A single-file export opens directly on its embedded model page (the first inlined `ok` cell,
else the first inlined cell); hash deep links still work. Its header links "All models" to
`./index.html`, the hub that sits beside the exports in a release `dist/`.

Hub mode (`vite build --mode hub`, entry `src/hub/main.tsx`, no router/graph code) inlines
`index.json`, `hardware.json` and `hub.json` = `{"exports":[{"version","model","device","file"}]}`
from the data directory and renders a model × device matrix per version pair. A cell with an
export is a plain relative `<a href="./<file>">` (works on file://; only bare `*.html` names
are linked); every other cell shows its honest status. Output: `dist-hub/index.html`, plus a
copy at `INFERATLAS_HUB_OUTPUT` when set. Demo sync also writes a demo `hub.json` and
synthetic status-only H100 cells (one triage-only cell, like the release's blocked cells).

Model page tabs: **Architecture** (default; `src/pages/model/ArchitectureView.tsx`, figure built
from `@inferatlas/arch-ui` over `arch.json`, `?layer=<i>&sub=<id>` / `?stage=<id>`, symbol
overrides such as `?T=8192&tp=8`) and **Graph & analysis** (`?tab=graph`; links with graph-only
keys such as `view`/`node`/`side` open it directly). See `docs/progress/U1.md`.

URL state (all shareable): `#/m/<version>/<device>/<model>?view=L2:attn&node=attn.o&phase=decode&overlay=true&side=roofline&shapes=sym&heat=weight&hw=<preset>&tp=4&T=8192…`.
Any non-UI search key is a runtime-parameter override validated against `ir.params`; defaults are
omitted from the URL. Keyboard: `/` find node, `Esc` up one level (focus returns to the node/anchor you left),
`Space` selects the focused graph node, `Enter` selects / opens it.
