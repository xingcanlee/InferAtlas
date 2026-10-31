# InferAtlas

Source-linked diagrams and symbolic memory, FLOP and communication accounting for vLLM
and vllm-ascend. Generate on CPU from pinned source/configs, then publish a static site or
an offline HTML file. No model weights or NPU are needed for extraction.

This branch delivers **T0: shared contracts and scaffold**. The extraction, evaluator,
lowering, layout, verification and UI implementations are assigned to subsequent tasks.
The curated Qwen3-MoE fixture matches **61,064,245,248 bf16 weight bytes** at TP=1 and
15,285,252,096 at TP=4; hardware-only overheads are explicitly uncalibrated.

Requires Node 22+, pnpm 10, uv and just. uv selects Python 3.12; package mirrors are
configured in `.npmrc` and the justfile.

```sh
just setup                 # pnpm workspace + independent generator environment
just schemas               # Pydantic → JSON Schema → TypeScript
just test                  # pytest + Node contract tests + skeleton type checks
just lint                  # ruff + Biome
pnpm -r build              # contracts build + strict skeleton type checks
uv run --project generator inferatlas version
```

`just build [MODEL] [DEVICE]`, `just site` and `just single` reserve the pipeline/UI entry
points for T6/F1. They do not yet produce a site: web's build is a skeleton type check,
single-file build reports that it is unimplemented, and the CLI currently has only `version`.
For Python commands outside just, set
`UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` before the first dependency sync.

Start with [design](docs/design.md), [dev plan](docs/dev-plan.md),
[wire contracts](packages/contracts/README.md), and [expression DSL](docs/expr-dsl.md).
`catalog.yaml` pins three inline configurations, two version pairs and device presets.
`fixtures/` supplies validated extract/IR/view/triage/report/evaluation/index examples.
See [T0 progress](docs/progress/T0.md) for decisions, limits and handoff notes.
