# web — InferAtlas frontend (F1)

Vite + React 19 + TypeScript (strict), Tailwind CSS v4 + shadcn/ui (Radix), TanStack Router
(hash history, zod-validated search params), Zustand, @xyflow/react, shadcn Charts (Recharts),
vite-plugin-singlefile. Expressions are evaluated only by `@inferatlas/expr` (via `src/lib/expr.ts`).

```sh
pnpm --filter web dev            # sync fixtures → public/data, start Vite
pnpm --filter web build          # dist/ (static site; data under dist/data)
pnpm --filter web build:single   # dist-single/index.html — one self-contained file, works from file://
pnpm --filter web test           # Vitest (oracle check against fixtures/*/eval.json, components)
pnpm --filter web e2e            # Playwright smoke + screenshots → docs/screenshots/ (build first)
pnpm --filter web lint           # Biome
```

Data: `scripts/sync-fixtures.mjs` copies `fixtures/` into `public/data/` following dev-plan §3.7
(`data/index.json`, `data/hardware.json`, `data/<version>-<device>/<model>/{ir,view,report}.json`)
and derives one clearly labelled synthetic snapshot (`fixture-prev`) so the diff page has two
versions. Set `INFERATLAS_DATA_DIR=<dir>` to copy real pipeline output instead. In single-file mode
the Vite plugin inlines index + every bundle + hardware as `window.__INFERATLAS__`;
`INFERATLAS_SINGLE_MODEL=<model-id>` restricts it to one model.

URL state (all shareable): `#/m/<version>/<device>/<model>?view=L2:attn&node=attn.o&phase=decode&overlay=true&side=roofline&shapes=sym&heat=weight&hw=<preset>&tp=4&T=8192…`.
Any non-UI search key is a runtime-parameter override validated against `ir.params`; defaults are
omitted from the URL. Keyboard: `/` find node, `Esc` up one level, `Enter` drill into selection.
