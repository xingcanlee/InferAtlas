# InferAtlas 开发设计方案 v1.0

> 依据：`docs/design.md`（产品与架构设计）与 `experiments/a2_qwen3moe/`（CPU 上 meta 实例化、TP shim、昇腾 mock 叠加已验证）。
> 本文是**可执行的开发计划**：技术选型、模块契约、任务拆分、并行方式、验收标准。

---

## 0. 执行模型（多 agent 并行）

| 角色 | 执行者 | 负责 |
|---|---|---|
| 编排 / 跟进 | Claude（主会话） | 拆任务、启动 agent、跟进进度、合并分支、处理冲突 |
| 后端、构建、CI、规则、skill、最终 HTML 生成 | **Codex CLI**（`codex exec -m gpt-6.1-sol`） | 除前端与 review 外的全部任务 |
| 前端 | Claude subagent | `web/` |
| 代码 review | Claude subagent | 每个分支合入 `main` 前 review |

**并行纪律**
1. 每个任务在独立 git worktree + 分支 `task/<id>` 上开发，只改自己**拥有的路径**（见 §4 表格）。
2. 需要改别人拥有的文件时，不直接改，写到 `docs/requests/<task-id>.md`，由编排者协调。
3. 跨任务依赖**只通过契约**（`schemas/` + `packages/contracts/` + `fixtures/`），契约由 T0 一次性定稿；之后修改契约必须走编排者。
4. 每个任务完成时：测试全绿、`docs/progress/<task-id>.md` 写明完成项 / 未完成项 / 偏离契约之处，然后在分支上 commit。
5. 合入顺序：review 通过 → rebase 到 `main` → fast-forward 合入。

---

## 1. 技术选型（优先框架能力）

| 层 | 选型 | 理由 |
|---|---|---|
| 仓库 | pnpm workspaces（TS）+ uv（Python）单仓 | 两个生态各用原生工具 |
| Python 生成器 | Python 3.12、**pydantic v2**（数据模型，导出 JSON Schema）、**typer**（CLI）、ruamel.yaml、pytest、ruff | 契约由 pydantic 单一定义，导出 JSON Schema，再生成 TS 类型 |
| TS 类型 | `json-schema-to-typescript` 从 `schemas/*.json` 生成 `packages/contracts` | 前后端同一契约来源 |
| 表达式 DSL | **peggy** 语法 → TS 解析器/求值器（`packages/expr`）；Python 侧用同一语法的 `lark` 实现做**语法校验 + 符号检查**，不做求值 | 设计要求只有一套求值实现；双端共享 `test-vectors.json` |
| 布局 | **elkjs**（Layered + ORTHOGONAL 路由），构建期 Node 运行 | 确定性坐标，离线不跑布局 |
| 前端框架 | **Vite + React 19 + TypeScript** | 生态成熟 |
| 样式 | **Tailwind CSS v4** + **shadcn/ui**（Radix 原语）+ CSS 变量语义 token（明暗主题） | 现代 UI，组件可控 |
| 路由 / URL 状态 | **TanStack Router**（hash history，离线 file:// 可用）+ zod 校验 search params | 目标中的 URL 状态与单文件离线 |
| 状态 | Zustand | 参数面板 ↔ 图 ↔ 侧栏联动 |
| 图渲染 | **@xyflow/react**（React Flow），节点坐标与边折点来自 view.json，自定义正交边 | 平移缩放、选中、小地图开箱即用 |
| 图表 | shadcn/ui Charts（Recharts）：显存瀑布、roofline（log-log）、KV 堆叠条 | 与 shadcn 主题一致 |
| 打包 | vite-plugin-singlefile（单文件 HTML）+ 普通多页站点 | 两种产物同一代码 |
| 测试 | pytest、Vitest、Playwright（冒烟 + 截图） | |
| Lint | ruff、Biome | 快、零配置 |
| CI | GitHub Actions + `actions/deploy-pages` | |

---

## 2. 仓库布局（最终形态）

```
InferAtlas/
  package.json  pnpm-workspace.yaml  biome.json  justfile
  schemas/                      # 由 pydantic 导出的 JSON Schema（生成物，入库）
  fixtures/                     # 手写/生成的契约样例，供并行开发
    qwen3-30b-a3b/{extract.json, ir.json, view.json, eval.json}
  catalog.yaml                  # 模型清单 + 版本配对 + 设备预设 + 硬件预设
  rules/                        # 规则 YAML（按版本区间）
  annotations/                  # 昇腾标注 YAML（anchor_hash）
  golden/                       # IR + 参数网格求值结果
  generator/                    # Python 包 inferatlas
    pyproject.toml
    inferatlas/
      schema/      # pydantic 模型 = 契约唯一来源
      cli.py       # typer：resolve / extract / lower / verify / golden / build
      resolve/     # catalog、版本配对（.github/*.commit）、源码 checkout、HF config
      extract/     # 环境、mock、meta_build、TP shim、registry、源码定位、ascend_scan
      expr/        # lark 语法校验（与 packages/expr 同语法）
      lower/       # 规则加载与匹配、IR 生成、显存/KV/通信、triage、覆盖率闸门
      annotations/ # 标注加载、anchor_hash、失效检测
      verify/      # 源码行校验、golden diff、校准比对
    tests/
  packages/
    contracts/     # 由 schemas 生成的 TS 类型 + 少量常量
    expr/          # peggy 语法、parse/evaluate/simplify/format、test-vectors.json
    layout/        # IR → 视图模板 → elkjs → view.json（Node CLI）
  web/             # Vite + React + Tailwind + shadcn
  skills/inferatlas-maintain/
  .github/workflows/  ci.yml poll.yml build.yml nightly.yml draft-pr.yml pages.yml
  experiments/     # 已有验证脚本（保留，只读参考）
  docs/            # design.md  dev-plan.md  progress/  requests/
```

---

## 3. 模块契约（T0 定稿）

数据流：`catalog.yaml` →(resolve)→ `snapshot.json` →(extract)→ `extract.json` →(lower + rules + annotations)→ `ir.json` + `triage.json` →(verify)→ `report.json` →(layout)→ `view.json` →(web build)→ 站点 / 单文件 HTML

### 3.1 `snapshot.json`
`{vllm: {tag, commit}, ascend: {tag, commit}, model: {hf_id, config_rev}, device, generator_version}`。
配对规则：vllm-ascend release tag → `.github/vllm-release-tag.commit`；main → `.github/vllm-main-verified.commit`（已验证）。

### 3.2 `extract.json`（脚本产出，事实层，无规则）
```jsonc
{
  "snapshot": {...},
  "config": {...HF config 原文...},
  "parallel": {"tp": 4, "pp": 1, "dp": 1, "ep": false},     // 实例化时的设置
  "modules": [{                       // 模块树，扁平化，按 path 排序
    "path": "model.layers.0.self_attn.qkv_proj",
    "class": "vllm.model_executor.layers.linear.QKVParallelLinear",
    "src": {"file": "vllm/model_executor/layers/linear.py", "line": 812, "commit": "544f182…"},
    "attrs": {"input_size": 2048, "output_size": 5120, "tp_size": 4, "...": "…"},  // 从模块实例白名单属性读取
    "params": [{"name": "weight", "shape": [1280, 2048], "dtype": "bfloat16"}],
    "ascend": {"class": "vllm_ascend.ops.linear.AscendQKVParallelLinear", "src": {...}} // 叠加时
  }],
  "layer_types": ["full_attention", "..."],     // 每层的类型序列，用于 layer band
  "ascend": {"device": "A2", "profile": {...}, "custom_ops": {...}, "patched": {"symbol": "origin_file"}},
  "env": {"torch": "2.10.0", "python": "3.12"}
}
```
同一模型按 `tp ∈ {1, 2, 4, 8}` 各抽取一次，lower 用多份 extract 交叉校验规则表达式（tp 作为符号参数，数值必须与对应 extract 中的 shape 一致）。

### 3.3 `ir.json`（见 design.md §4.1，以下为定稿字段）
- `snapshot`、`symbols`（来自 config 的常量）、`params`（运行时可调参数及默认值、取值域）
- `nodes[]`：`id, kind(compute|comm|io|group), class, src, rule, label, in[], out[]`（shape 为表达式数组）、`weight_bytes, flops{prefill,decode}, bytes_moved{prefill,decode}, comm{op,group,logical_bytes,bus_bytes,when}, parallel{tp|ep|pp:{dim}}, ascend{tag[],level,note,src,annotation_id}`
- `edges[]`：`from, to, tensor(shape 表达式), kind(data|kv|residual)`
- `groups[]`：layer band `{id, layers:[start,end], template, representative_layer}`
- `memory`：`{weights, kv_per_token, kv_tokens, activation_peak, non_torch, reserved, unmodelled[]}` 全部为表达式
- `views`：`{family: "dense|moe|mla|hybrid|vlm"}` 交给 layout 选模板
- 所有数值字段都是 **表达式字符串**；求值只在 TS。

### 3.4 表达式 DSL
- 语法：数字、标识符、`+ - * / % **`、括号、`max min ceil floor round_up(x, m) cdiv(x, y)`、`a ? b : c`、比较与 `&& || !`、`bytes(dtype)`。
- 语义：浮点 double；整数运算在 2^53 内精确；`cdiv` 整除向上取整。
- 双端共享 `packages/expr/test-vectors.json`：`[{expr, env, value | error}]`。

### 3.5 `view.json`（layout 产出）
`{ views: [{id: "L0"|"L1:<band>"|"L2:<node>", nodes:[{id, ir_node, x, y, w, h, label}], edges:[{id, from, to, points:[[x,y]...], label_pos}], width, height, collapsed_of: [...] }] }`

### 3.6 `triage.json` 与 `annotations/*.yaml`
见 design.md §5.2，T0 用 pydantic 定稿。

### 3.7 Web 数据入口
构建时生成 `web/public/data/index.json`（版本 × 模型矩阵、每格状态）和 `data/<snapshot>/<model>/{ir,view,report}.json`；单文件模式下由 Vite 插件把当前模型的数据内联为 `window.__INFERATLAS__`。

---

## 4. 任务拆分

### 4.1 任务表

| ID | 名称 | 执行 | 拥有路径 | 依赖 | 验收 |
|---|---|---|---|---|---|
| **T0** | 脚手架 + 契约 | Codex | 根配置、`schemas/`、`generator/inferatlas/schema/`、`generator/pyproject.toml`、`packages/contracts/`、`fixtures/`、`catalog.yaml`、`justfile` | — | `uv run pytest`、`pnpm -r build` 通过；fixture 全部通过 schema 校验；Python 依赖一次加齐 |
| **T1** | extract | Codex | `generator/inferatlas/{resolve,extract}/`、对应 tests | T0 | Qwen3-30B-A3B 在 tp=1/4 下产出合规 extract.json；参数总数 = 30,532,122,624；源码行 100% 指向 `class`/`def`；A2 叠加：31 个 custom op、patched 集合与实验一致 |
| **T2** | 表达式 DSL | Codex | `packages/expr/`、`generator/inferatlas/expr/` | T0 | ≥150 条 test vectors 双端通过；evaluate 性能：1 万节点 × 表达式 < 50ms |
| **T3** | 规则 + lower | Codex | `rules/`、`generator/inferatlas/lower/` | T0（用 fixture extract 开发） | Qwen3-8B、Qwen3-30B-A3B 的 IR 合规；IR 在 tp=1/2/4/8 求值的每节点权重 shape 与 extract 一致；权重字节与实验精确一致；KV/激活按 vLLM 口径；未匹配类进 triage |
| **T4** | 布局 | Codex | `packages/layout/` | T0, T2 | Dense/MoE 模板的 L0/L1/L2；同输入两次运行字节一致；节点 ≤9、边 ≤12 的预算检查 |
| **T5** | 昇腾标注 + skill | Codex | `annotations/`、`generator/inferatlas/annotations/`、`skills/inferatlas-maintain/` | T0 | anchor_hash 计算与失效检测有测试；Qwen3-MoE A2 的 12 个替换类都有 level/note 标注；SKILL.md + references + scripts |
| **T6** | verify + golden + CLI 编排 | Codex | `generator/inferatlas/{verify,cli.py}`、`golden/` | T0 | `inferatlas build --model … --device A2` 串起全流程（缺失模块用 fixture 降级）；源码行校验；golden diff 输出可读 |
| **T7** | CI / 发布 | Codex | `.github/workflows/` | T0 | ci.yml 跑全部测试；poll.yml cron 检测两仓库新 tag；build.yml 生成并部署 Pages；draft-pr.yml 调用 skill；`act` 或 dry-run 校验 YAML |
| **F1** | 前端 | Claude subagent | `web/` | T0（fixture + contracts）；T2 前用 stub | 见 §4.2 |
| **I1** | 集成 | Codex | 全仓（编排者指派） | T1–T7, F1 | 真实数据端到端：2 个模型 × 1 个版本对 × A2，站点和单文件 HTML 产出；所有测试通过 |
| **H1** | 生成 HTML | Codex | `dist/` | I1 | `just site` 产出 Pages 站点 + `dist/inferatlas-<model>.html` 单文件，Playwright 冒烟通过 |
| **R\*** | 代码 review | Claude subagent | 只读 | 每个分支 | 无阻断问题后合入 |

### 4.2 F1 前端功能清单
1. 首页：版本 × 模型矩阵（状态徽章：ok / triage / missing），筛选设备。
2. 模型页三栏：左参数面板（tp/pp/dp/ep、T、S、batch、dtype_w/kv、device、gpu_memory_utilization、max_num_batched_tokens…，shadcn Form + Slider），中图（React Flow，L0→L1→L2 钻取，面包屑，layer band），右详情面板（design.md §3.5 的顺序）。
3. 侧栏视图：显存瀑布图、KV 堆叠条、roofline、rank 泳道（TP/EP 切片矩阵）。
4. 昇腾叠加开关：角标 `OOT/PATCH/QUANT/FUSED`，虚线外框，对照卡片。
5. 版本 diff 页：左右对照，变动节点/边强调色。
6. 实时重算：参数变化 → `packages/expr` 求值 → 图边标签、详情、图表同步更新（<16ms）。
7. URL 状态：所有参数、当前视图、选中节点进 search params；可分享。
8. 视觉：design.md §3.3 语义 token；明暗主题；无阴影；4px 网格；mono 用于 shape。
9. 单文件构建模式 `pnpm --filter web build:single`。
10. Vitest 组件测试 + Playwright 冒烟。

### 4.3 并行波次

```
波次 0 ── T0 契约（阻塞，约 1 个 agent 会话）
           │
波次 1 ── T1 extract ─┐  T2 expr ─┐  T3 rules+lower ─┐  T5 标注+skill  T7 CI   F1 前端(fixture)
           │          │           └─ T4 layout       │
           │          └──────────────────────────────┤
波次 2 ── T6 verify+golden+CLI（需 T1/T3 接口）        │
           │                                         │
波次 3 ── I1 集成 → R 全量 review → H1 生成 HTML
```
- 每个任务完成后立即进入 review（R），通过即合入 `main`，后续任务基于新 `main`。
- 波次 1 七个 agent 并行（6 Codex + 1 Claude）。

---

## 5. 关键实现约定

- **抽取环境**：复用实验的 venv 与源码（`experiments/a2_qwen3moe/.venv`、`src/`）；generator 通过 `INFERATLAS_SRC`、`INFERATLAS_PY` 环境变量定位；子进程隔离每次抽取（一个 snapshot × tp × device 一个进程，`python -I`）。
- **TP>1**：用实验验证过的 `_ShimTP`（真实 1-rank gloo + 覆盖 world_size/rank_in_group），不用 FakeProcessGroup。EP 同理对 `_EP` 做 shim。
- **mock**：执行上游 `tests/ut/conftest.py`（剥离 `adapt_patch()` 与 `register_ascend_customop()` 两行后显式调用），设置 `_build_info.__device_type__`。
- **确定性**：所有 JSON 输出 `sort_keys`、稳定排序；不写时间戳；elk 固定 seed 与选项。
- **agent 产物约束**（skill）：不写行号、不写数值；标注带 anchor_hash。
- **不建模项**明确列入 `memory.unmodelled`，页面显示。

---

## 6. 里程碑与验收

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M-A（波次 0–1） | 各模块独立可测 | 各任务验收项全部通过 |
| M-B（波次 2） | CLI 端到端 | `just build MODEL=Qwen/Qwen3-30B-A3B DEVICE=A2` 产出 ir/view/report |
| M-C（波次 3） | 站点 + 单文件 | Pages 站点本地预览可用；单文件离线打开可交互重算；review 无阻断项 |
