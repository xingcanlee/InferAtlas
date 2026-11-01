# web — InferAtlas frontend (F1)

Vite + React 19 + TypeScript (strict), Tailwind CSS v4 + shadcn/ui (Radix), TanStack Router
(hash history, zod-validated search params), Zustand, @xyflow/react, shadcn Charts (Recharts),
vite-plugin-singlefile. Expressions are evaluated only by `@inferatlas/expr` (via `src/lib/expr.ts`).

```sh
# release (real pipeline output required — fails without it)
INFERATLAS_DATA_DIR=<dir> pnpm --filter web build          # dist/  (static site; data under dist/data)
INFERATLAS_DATA_DIR=<dir> pnpm --filter web build:single   # dist-single/index.html, works from file://

# local demo on contract fixtures (never for publication)
pnpm --filter web dev                 # Vite dev server
pnpm --filter web build:demo          # dist/ with fixture + synthetic demo data
pnpm --filter web build:single:demo
pnpm --filter web test                # Vitest (eval.json oracle, graph/diff/router tests)
pnpm --filter web e2e                 # demo builds + Playwright smoke + docs/screenshots/
pnpm --filter web lint                # Biome
```

`INFERATLAS_DATA_DIR` is the pipeline's public data directory: it contains `index.json`,
`hardware.json` (catalog `hardware_presets`) and every `data_path` the index names, each with
`ir.json`, `view.json`, `report.json`. Relative paths resolve from where pnpm was invoked.
`scripts/sync-data.mjs` copies it to `public/data/`; without it (and without `--demo`) the sync
step exits 1, so a release build can never ship fixtures.

Demo mode (`--demo`, used by `dev`, `build:demo`, `e2e`) copies `fixtures/`, lays the views out
with the real build-time `packages/layout` CLI, and derives one clearly labelled synthetic
snapshot (`fixture-prev`, "synthetic · diff demo") so the diff page has two versions.
In single-file mode the Vite plugin inlines index + every bundle + hardware as
`window.__INFERATLAS__`; `INFERATLAS_SINGLE_MODEL=<model-id>` restricts it to one model.

URL state (all shareable): `#/m/<version>/<device>/<model>?view=L2:attn&node=attn.o&phase=decode&overlay=true&side=roofline&shapes=sym&heat=weight&hw=<preset>&tp=4&T=8192…`.
Any non-UI search key is a runtime-parameter override validated against `ir.params`; defaults are
omitted from the URL. Keyboard: `/` find node, `Esc` up one level (focus returns to the node/anchor you left),
`Space` selects the focused graph node, `Enter` selects / opens it.
