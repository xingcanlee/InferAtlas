set dotenv-load := false
set positional-arguments
export UV_INDEX_URL := "https://mirrors.aliyun.com/pypi/simple/"
export npm_config_registry := "https://registry.npmmirror.com"
MODEL := "Qwen/Qwen3-30B-A3B"
DEVICE := "A2"

# Install Node workspace and the independent Python 3.12 generator environment.
setup:
    pnpm install
    uv sync --project generator --python 3.12

# Both contract stacks; new task packages add their own test scripts.
test:
    uv run --project generator pytest -c generator/pyproject.toml
    pnpm -r test

lint:
    uv run --project generator ruff check --config generator/pyproject.toml generator fixtures/build.py
    uv run --project generator ruff format --check --config generator/pyproject.toml generator fixtures/build.py
    pnpm lint

# Pydantic → JSON Schema → committed TypeScript contracts.
schemas:
    uv run --project generator python -m inferatlas.schema.export
    pnpm --filter @inferatlas/contracts gen

build *args:
    #!/usr/bin/env bash
    set -eu
    model="{{MODEL}}"
    device="{{DEVICE}}"
    for arg in "$@"; do
        case "$arg" in
            MODEL=*) model="${arg#MODEL=}" ;;
            DEVICE=*) device="${arg#DEVICE=}" ;;
            *) echo "Expected MODEL=... or DEVICE=..., got $arg" >&2; exit 1 ;;
        esac
    done
    pnpm --filter '@inferatlas/*' -r build
    uv run --project generator inferatlas build --model "$model" --device "$device"

site:
    uv run --project generator inferatlas site

single:
    uv run --project generator inferatlas single --model "{{MODEL}}" --device "{{DEVICE}}"

hub *args:
    uv run --project generator inferatlas hub "$@"

release *args:
    uv run --project generator inferatlas release "$@"

# Real CPU/meta acceptance tests (external experiment venv required).
extract-all:
    uv run --project generator pytest -c generator/pyproject.toml -m extract
