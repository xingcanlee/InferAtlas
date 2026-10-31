set dotenv-load := false
export UV_INDEX_URL := "https://mirrors.aliyun.com/pypi/simple/"

# Install Node workspace and the independent Python 3.12 generator environment.
setup:
    pnpm install
    uv sync --project generator --python 3.12

# Both contract stacks; new task packages add their own test scripts.
test:
    uv run --project generator pytest
    pnpm -r test

lint:
    uv run --project generator ruff check --config generator/pyproject.toml generator fixtures/build.py
    uv run --project generator ruff format --check --config generator/pyproject.toml generator fixtures/build.py
    pnpm lint

# Pydantic → JSON Schema → committed TypeScript contracts.
schemas:
    uv run --project generator python -m inferatlas.schema.export
    pnpm --filter @inferatlas/contracts gen

build MODEL="Qwen/Qwen3-30B-A3B" DEVICE="A2":
    uv run --project generator inferatlas build --model "{{MODEL}}" --device "{{DEVICE}}"
    pnpm -r build

site:
    pnpm --filter web build

single:
    pnpm --filter web build:single
