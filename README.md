# InferAtlas

Source-linked model diagrams and symbolic memory, FLOP and communication accounting for
vLLM and vllm-ascend. Extraction uses CPU/meta tensors and the upstream Ascend unit-test
mocks. It downloads no weights and runs no NPU kernels.

Requires Node 22.14+, pnpm 10, uv, just and Python 3.12 (selected by uv).
The npm and Python mirrors are configured in `.npmrc` and `justfile`.

```sh
just setup
just lint
just test                  # pytest, contracts, expression, layout and web unit tests
just extract-all           # real CPU/meta extraction matrix only
```

Real extraction also needs the read-only upstream clones and the CPU torch/vLLM
interpreter prepared by `experiments/a2_qwen3moe/`. A normal checkout or worktree finds
them automatically. For a different installation, set:

```sh
export INFERATLAS_SRC=/path/to/sources  # contains vllm/ and vllm-ascend/
export INFERATLAS_PY=/path/to/cpu-venv/bin/python
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
```

Build the supported main-verified matrix, then publish its data and web assets:

```sh
just build MODEL=Qwen/Qwen3-30B-A3B DEVICE=A2
just build MODEL=Qwen/Qwen3-30B-A3B DEVICE=H100
just build MODEL=Qwen/Qwen3-8B DEVICE=A2
just build MODEL=Qwen/Qwen3-8B DEVICE=H100
uv run --project generator inferatlas site   # same as just site
pnpm --filter web preview                  # serves web/dist
```

`build` compiles the shared Node packages and writes real extract/IR/view/triage/report
artifacts to `.cache/build/<snapshot>/<model>/`. `site` reads those builds, checks their
provenance and verification status, publishes `web/public/data/` (including the hardware
presets), and produces `web/dist/`. From `generator/`, the equivalent command is
`uv run inferatlas site`. Exit codes are 0 for verified output, 2 for blocking triage,
and 3 for failed verification or execution. Fixture provenance cannot be published.

Generate a single-model offline HTML file and run browser checks against real data:

```sh
uv run --project generator inferatlas single --model Qwen/Qwen3-8B --device A2
uv run --project generator inferatlas single --model Qwen/Qwen3-30B-A3B --device A2
# dist/inferatlas-{qwen3-8b,qwen3-30b-a3b}-a2.html
# The last single command also stages web/dist-single/index.html.
uv run --project generator inferatlas site   # restore the full matrix after single
pnpm --filter web exec playwright install chromium
pnpm --filter web e2e
```

`single` rechecks the selected real model and inlines its data. It passes
`INFERATLAS_SINGLE_MODEL` to the web builder; `--output` changes the destination.
For browser development without upstream sources, use `pnpm --filter web dev` or
`pnpm --filter web e2e:demo`. These explicitly use fixtures and a labelled synthetic
previous version. Release web builds require `INFERATLAS_DATA_DIR` and never silently
substitute demo data. Real e2e uses the existing site and single-file artifacts.

To check a particular build against its committed numerical baseline:

```sh
uv run --project generator inferatlas golden check --ir .cache/build/<snapshot>/<model>/ir.json
just schemas               # regenerate JSON Schema and TypeScript after contract changes
```

The pinned pair is vLLM `544f182` / vllm-ascend `24a76e8`. Dense and MoE A2 annotations
are source-anchored drafts marked `reviewed_by: pending`. Hardware overhead deductions
are documented catalog estimates, not measurements: A2 defaults to 1 GB runtime buffers
plus 2 GB graph/allocator reserve; H100 uses 0.2 GB plus 2 GB. They are editable in the
page and require startup-log calibration for a deployment. Unknown model rules, stale
notes and unannotated observed Ascend swaps block publication. DeepSeek-V3 is catalogued
but its lowering remains unsupported and is shown as missing.

See [design](docs/design.md), [development contracts](docs/dev-plan.md),
[expression DSL](docs/expr-dsl.md), [annotations](annotations/README.md) and
[integration verification transcript](docs/progress/I1.md).
