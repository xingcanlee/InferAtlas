#!/usr/bin/env bash
# CPU-only verification: Qwen3-MoE (Qwen3-30B-A3B config) on Ascend A2 profile.
# No NPU, no model weights. Results go to ./out/*.json and ./out/*.log
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
SRC=${INFERATLAS_SRC:-$HERE/src}
export INFERATLAS_SRC=$SRC
mkdir -p out "$SRC" .tmp
# /tmp may be a small tmpfs; keep build/temp files on disk.
export TMPDIR=$HERE/.tmp

# 1. sources (latest main; commits recorded in out/commits.txt)
for r in vllm vllm-ascend; do
  [ -d "$SRC/$r/.git" ] || git clone -q --depth 1 "https://github.com/vllm-project/$r.git" "$SRC/$r"
done
# vllm-ascend main only works with the vLLM commit it declares as verified.
PIN=$(tr -d '[:space:]' < "$SRC/vllm-ascend/.github/vllm-main-verified.commit")
if [ "$(git -C "$SRC/vllm" rev-parse HEAD)" != "$PIN" ]; then
  git -C "$SRC/vllm" fetch -q --depth 1 origin "$PIN" && git -C "$SRC/vllm" checkout -q "$PIN"
fi
for r in vllm vllm-ascend; do echo "$r $(git -C "$SRC/$r" rev-parse HEAD)"; done | tee out/commits.txt

# 2. venv: CPU torch matching vllm-ascend's pin, vLLM as a pure-Python build
if [ ! -x .venv/bin/python ]; then
  uv venv -q --python 3.12 .venv
fi
PY=$HERE/.venv/bin/python
# Mirrors: direct pypi.org / download.pytorch.org are ~200 KB/s here. Override via env if needed.
export UV_INDEX_URL=${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}
TORCH_LINKS=${TORCH_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cpu/}
TORCH_VER=$(grep -oE '^torch==[0-9.]+' "$SRC/vllm-ascend/requirements.txt" | head -1 | cut -d= -f3)
# Pin the CPU build for every install so no dependency drags in CUDA torch (+nvidia libs, GBs).
echo "torch==${TORCH_VER}+cpu" > out/constraints.txt
UVI=(uv pip install --python "$PY" -c out/constraints.txt --find-links "$TORCH_LINKS" --index-strategy unsafe-best-match)
echo "torch: ${TORCH_VER}+cpu  index: $UV_INDEX_URL" | tee out/install.log
"${UVI[@]}" "torch==${TORCH_VER}+cpu" 2>&1 | tee -a out/install.log
# vLLM is NOT built/installed: even VLLM_TARGET_DEVICE=empty runs cmake + a Rust
# extension (3+ GB of build output). We only need its Python sources, so install
# its Python deps and put the source tree on sys.path via a .pth file.
"${UVI[@]}" -r "$SRC/vllm/requirements/common.txt" 2>&1 | tee -a out/install.log
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$SRC/vllm" > "$SITE/inferatlas_vllm_src.pth"
# vllm-ascend runtime deps, minus NPU-only and torch packages (it stays on sys.path, not installed)
grep -vE '^\s*(#|$)|^(torch|torch_npu|torch-npu|torchvision|torchaudio|triton|triton-ascend)\b' \
  "$SRC/vllm-ascend/requirements.txt" > out/ascend-reqs.txt
while read -r req; do
  "${UVI[@]}" -q "$req" >>out/install.log 2>&1 || echo "skip dep: $req" | tee -a out/install.log
done < out/ascend-reqs.txt
"${UVI[@]}" -q pytest "torchvision==0.25.0+cpu" runai-model-streamer librosa soundfile 2>&1 | tee -a out/install.log
"$PY" -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)" | tee -a out/install.log

# 3. experiments (each isolated; a failure is recorded in its json, not fatal)
run() { local name=$1; shift; echo "== $name"; timeout 600 "$PY" -I "$@" >"out/$name.log" 2>&1; echo "   exit=$? (log: out/$name.log)"; }
run exp1_tp1 exp1_vllm_meta.py --tp 1
run exp1_tp4 exp1_vllm_meta.py --tp 4
run exp2_A2  exp2_ascend_overlay.py --device A2
run exp3_A2  exp3_ascend_meta.py --device A2

echo; echo "results:"; ls -1 out/*.json
