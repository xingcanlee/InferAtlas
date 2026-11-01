# InferAtlas 设计方案 v0.1

> 状态：草案（2026-10-31）。依据目标文档 1–23 条及非目标整理。
> 本文分五部分：①同类产品与可借鉴点 ②AI Infra 视角需补充的内容 ③图的设计规范（参考 diagram-design skill）④系统架构 ⑤Skill + 脚本的分工；最后是分期计划与风险。

---

## 1. 同类产品调研

| 产品 | 做什么 | 可借鉴 | 与 InferAtlas 的差别 |
|---|---|---|---|
| **LLM-Viewer**（hahnyuan，论文 arXiv 2402.16363） | 浏览器内的逐层 roofline 分析：FLOPs、访存、算术强度、按数据依赖算峰值显存，点击节点看详情 | 逐层 roofline 模型；**按数据依赖（tensor 生命周期）算激活峰值**，正好满足目标 12；节点详情面板 | 模型结构是手写的通用模板，不绑定推理框架源码和版本，也没有并行切分和通信 |
| **Google Model Explorer**（ai-edge） | 大规模计算图的分层可视化，可折叠/展开命名空间，支持 torch.export，可做双图对比，可通过 JSON 叠加节点数据 | ① **按 namespace 嵌套折叠**；② **Node Data Provider**：结构和数值分开存，数值以 JSON 叠加，对应我们的"主图 + 昇腾叠加层"；③ 双图对比视图 | 图来自算子级 trace，节点成千上万，信息密度太高，10 分钟讲不清 |
| **Netron** | 模型文件查看器，本身就是静态单页 | 纯静态、零后端；单文件分发 | 只看权重文件，没有运行时语义 |
| **bbycroft llm-viz / Transformer Explainer** | 面向教学的 Transformer 动画讲解 | 引导式"导览"叙事，让人能一步步跟着走 | 只覆盖玩具模型，不能用于工程 |
| **torchview / torchinfo** | 用 meta tensor 跑一次前向，得到模块树和 shape | 目标 21 的 meta-device 追踪可以直接参考它的 hook 方案 | 不懂 TP，也不懂 vLLM 的 attention metadata |
| **llm-analysis / llm_counts / HF accelerate estimate-memory** | 命令行版的参数、FLOPs、显存理论计算器 | 公式库可以拿来交叉校验 | 只有文本输出，没有结构图，也不跟版本走 |
| **kvfit / ApX VRAM Calculator / 各类 HF Space** | KV cache 和显存容量规划，kvfit 能对照真实 vLLM 运行做校准 | ① **显式列出"未建模项"**（CUDA/ACL graph、allocator workspace 等），诚实标注；② **用真实 serve 日志校准**，对应目标 22 | 只算总数，没有结构 |
| **DeepWiki 类代码 wiki** | 自动生成源码解读 | 源码永久链接的呈现方式 | 由 LLM 生成，不确定、不可复现，与目标 18、20 相反 |

**结论：** 目前没有产品同时做到"**绑定推理框架源码和版本** + **TP/EP 通信** + **昇腾适配层对照**"，这是 InferAtlas 的定位空间。具体借鉴方式：
- 交互模型参考 Model Explorer：分层折叠、数据叠加、双图 diff；
- 计算模型参考 LLM-Viewer：逐层 roofline，按 tensor 生命周期算激活峰值；
- 可信度参考 kvfit：未建模项清单，用真实日志校准；
- 讲解方式参考 llm-viz：做"10 分钟导览"。

---

## 2. AI Infra 视角：目标里需要补充或修正的点

### 2.1 并行维度不完整
目标只列了 tp/pp/dp/ep，vLLM 和 vllm-ascend 实际还有下面这些，都会改变 shape、通信或显存：

| 维度/开关 | 影响 |
|---|---|
| **Sequence Parallel**（vllm-ascend 的 flashcomm 类优化） | all-reduce 改为 reduce-scatter + all-gather，激活按 token 切分 |
| **DCP / PCP**（decode / prefill context parallel） | KV cache 沿 token 维切分，MLA 模型的 KV 占用随之下降 |
| **DP-attention + EP-MoE**（DeepSeek 类） | attention 走 DP、MoE 走 EP，层间多一次 all-gather 或 all2all |
| **lm_head / o_proj 独立 TP**（vllm-ascend additional_config） | 局部 TP 度和全局 TP 度不同 |
| **PP 不均匀切分**（`VLLM_PP_LAYER_PARTITION` / `get_pp_indices`） | 每个 rank 的层数不同，所以"单 rank 显存"要按 **PP 内最重的那个 rank** 给 |

→ **建议：** 并行配置用 `ParallelSpec` 对象描述，每一维都由规则表声明作用在哪些节点上，不要把维度写死成四个。

### 2.2 显存估算要与 vLLM 的算法对齐，否则做不到 10% 以内
vLLM 的 KV 可用显存是这样算的：
```
available_kv = total_hbm × gpu_memory_utilization
             − weights
             − peak_activation(profile_run 用 max_num_batched_tokens 跑一次)
             − non_torch（通信 buffer、HCCL/NCCL workspace）
             − graph_capture（CUDA/ACL graph，新版本计入）
num_blocks   = available_kv // (page_size_bytes × num_layers_per_group)
```
因此：
- **交互参数必须补上：** `gpu_memory_utilization`、`max_num_batched_tokens`、`max_num_seqs`、`block_size`、设备 HBM 总量（设备预设）。
- **激活峰值的口径是 `max_num_batched_tokens` 次 chunked prefill**，不是"上下文长度"。另外 sampler 预热时 `max_num_seqs × vocab × fp32` 的 logits 常常就是峰值所在。
- **校验口径：** 对照启动日志里的 `Model loading took X GiB`、`Available KV cache memory`、`GPU KV cache size: N tokens`、`Maximum concurrency`，而不是只对 block 数。
- **未建模项要列出来**（参考 kvfit）：graph capture、allocator 碎片、HCCL buffer。给的是区间，或者明确注明"不含"。

### 2.3 容易算错的 KV/权重细节（规则表必须覆盖）
- **KV 头复制：** `tp > num_kv_heads` 时每个 rank 持有 1 个 KV 头的副本，每 rank 的 KV 头数是 `max(1, kv_heads/tp)`，不是 `kv_heads/tp`。
- **MLA：** 缓存的是 latent `(kv_lora_rank + qk_rope_head_dim)`，**在 TP 维上不切分**，每个 rank 都存全量；昇腾把 nope 和 rope 分成两个 cache tensor 存。
- **混合 KV 管理器：** sliding window 层、full attention 层、Mamba/linear-attention 状态层（Qwen3-Next 等）分属不同 KV group，page size 不同。"每 token KV" 要分组给出。
- **稀疏 attention**（DeepSeek V3.2 的 indexer cache）、**MTP/EAGLE draft 层**：都会多出权重和 KV。
- **Vocab padding：** `VocabParallelEmbedding` 先把 vocab pad 到 64 的倍数，再按 TP 切分；tied embedding 的权重不能重复计入。
- **量化：** weight-only 量化（GPTQ/AWQ）要算 group scale/zero 的开销；W8A8 有 per-token 动态 scale；FP8 block-wise scale；KV cache 也可能量化（fp8 / 昇腾 int8 C8）。
- **昇腾特有：** 权重转成 FRACTAL_NZ 格式时有 16/32 对齐 padding；不同 attention 后端的 block_size 约束（如 128）不同。这些放在叠加层里，标为"昇腾差异"。

### 2.4 通信量口径
- 每个通信节点给两个数：**逻辑字节**（张量大小）和**总线字节**（ring all-reduce 是 `2(n−1)/n ×`，all-gather / reduce-scatter 是 `(n−1)/n ×`）。
- EP all2all（dispatch/combine）的量取决于路由分布，默认按均匀路由估算，并注明这一假设。
- 昇腾的 **MC2 融合算子**（matmul 与通信融合）会改变通信节点的形态：叠加层把它画成"融合节点"。

### 2.5 Roofline 需要硬件预设
目标 8 要求标出"偏计算瓶颈还是访存瓶颈"，这需要硬件的 ridge point（峰值算力 ÷ 带宽）。因此要提供**设备预设表**（Atlas 800I A2/A3 各型号、H100/H20 作参照），包括各 dtype 的峰值 TFLOPS、HBM 带宽和容量，用户也可以自定义。这样做不违反"不测耗时"，因为只给出瓶颈分类，不给时间。

### 2.6 版本、配置和环境的固化
- **HF config 也要固化：** 页面结构由 vLLM commit 和 `config.json` 共同决定，所以快照键里要加上 HF repo 的 revision（sha）。
- **生成器版本：** 生成器自己修 bug 后，旧快照是否重建？建议快照的身份是 `(vllm, ascend, model, config_rev)`，页头再标 `generator vX`。生成器修复后发布 `-r2`，旧版本保留，不覆盖。
- **vLLM 与 vllm-ascend 的版本配对：** vllm-ascend 的 release 对应某个特定的 vLLM 版本，配对规则由矩阵文件维护，不做笛卡尔积。
- **GitHub Actions 没法订阅别人仓库的 release 事件**（目标 17）：只能用定时任务（cron）轮询两个仓库的 releases API，发现新 tag 再触发生成。

### 2.7 不依赖 NPU：生成走静态 + CPU，NPU 只用于校准
**结论：** 生成页面不需要 NPU，只有显存校准（目标 22）需要。CI 全部跑在 CPU runner 上。vllm-ascend 不能直接 import，但可以复用**上游自己维护的 CPU 单测 mock 环境**，按设备型号加载插件和 patch，读出确定性事实（见下方核对结果 ①）。自己写 stub 很脆弱，不采用；复用上游 mock 的维护成本由上游 CI 承担。

| 要得到的信息 | 来源 | 环境 | NPU |
|---|---|---|---|
| ModelRegistry → 类 → 源码 file:line | AST 扫描 `vllm/model_executor/models/registry.py` 和各模型文件 | 纯静态 | 否 |
| 模块树、参数 shape、权重字节数 | `torch.device("meta")` + **FakeProcessGroup**（`world_size=tp`，`rank=0`）实例化 vLLM 模型 | CPU（vLLM CPU 版） | 否 |
| 前向 shape | 对 MLP / Linear / MoE gate 等子模块做 meta 前向，校验规则；attention 需要 metadata，shape 由规则推导、golden 对照 | CPU | 否 |
| 昇腾 CustomOp 替换 | 在上游 CPU mock 环境里调用 `register_ascend_customop()` 后读取 `REGISTERED_ASCEND_OPS`（AST 读取字面量作为交叉校验） | CPU + mock | 否 |
| 昇腾 patch 目标集合 | 在 mock 环境里对 patch 前后的 vllm 符号做 identity diff | CPU + mock | 否 |
| 昇腾 patch 的相关性与说明 | `patch/__init__.py` 的半结构化注释 + patch 源码 → agent 判定与提炼（见 §5） | 静态 + agent | 否 |
| 设备差异（A2/A3/A5/310P） | `device/hardware_profile.py` 的声明式表，按 `_build_info.__device_type__` 切换 | CPU + mock | 否 |
| attention 后端、量化方法 | `platform.get_attn_backend_cls`、`AscendQuantConfig`（读 msmodelslim 的量化描述） | 静态，带**运行时条件**（待核实） | 否 |
| 型号以外的条件分支（环境变量、`additional_config`、图模式、模型类型判断） | 得到"候选项 + 触发条件"后，由**标注文件**给出判定（见 §5） | 静态 + 一次性 agent 判定 | 否 |
| NZ 对齐 padding、ACL graph 占用、HCCL buffer、真实 KV block 数 | 真机启动日志 | NPU | **是**，仅用于校准 |

**显存校准：** 由有 NPU 的 self-hosted runner 定期（而不是每个 release）跑 `vllm serve`、解析日志，生成 `calibration/*.json` 提交入库。CI 只读这些文件；某个快照没有校准数据时，页面标"未校准"，不阻塞发布。

#### 源码核对结果（vllm `e5196e2`、vllm-ascend `24a76e8`，2026-10-31）

**① 不能直接 import，但上游维护了一套 CPU mock 环境，可以复用。**
- `vllm_ascend/utils.py:34` 顶层 `import torch_npu`，而 `platform.py` 又 import 了 utils，所以普通 CPU 环境 import 会直接失败。
- 但 vllm-ascend 自己的 CPU 单测（`tests/ut/conftest.py`）做了这几件事：
  - mock 掉 `torch_npu` / `torch.npu` / `acl` / `fla_npu`；
  - 通过 `vllm_ascend._build_info.__device_type__` 指定设备型号（默认 `"A2"`）；
  - 执行 `adapt_patch()`、`adapt_patch(True)` 和 `register_ascend_customop()`。
- 这套环境由上游为自己的 CI 维护，InferAtlas 可以按版本复用。
- vLLM 本体可用 `VLLM_TARGET_DEVICE=empty` 构建成纯 Python 包（见 `utils.py` 中 `vllm_version_is` 的注释），不需要 CUDA 内核。

**② CustomOp 替换表是字面量字典，可以确定性提取。**
- 位置：`utils.py:817-949` 的 `register_ascend_customop`。
- 共约 30 项，覆盖主图几乎所有节点：
  - Column/Row/QKV/MergedColumn/Replicated 各类 Linear；
  - `VocabParallelEmbedding`、`ParallelLMHead`、`LogitsProcessor`；
  - `RMSNorm` / `GemmaRMSNorm`、各类 RotaryEmbedding、`SiluAndMul`；
  - MLA wrapper、`MoERunner` / `RoutedExperts` / `GateLinear`、GatedDeltaNet 等。
- 唯一的条件分支是 `HardwareCapability.COMPATIBILITY_OP_IMPLEMENTATIONS`（310P 覆盖为 `*310` 实现）。
- **对叠加层设计的影响：** 昇腾几乎替换了每个节点，"有替换就打角标"会让角标失去信号。叠加层需要分级：
  - `KERNEL`：同语义，只是换了 kernel，默认隐藏；
  - `LAYOUT`：权重或 KV 布局变化，例如 NZ；
  - `SEMANTIC`：shape、通信或计算图有变化，例如 MC2、FlashComm。

**③ 硬件差异集中在一张声明式表里，条件分支大多能确定性解析。**
- 位置：`device/hardware_profile.py`，冻结的 `_HARDWARE_PROFILES` 表，覆盖 A2 / A3 / A5 / 310P。
- 每个型号声明了：`attention_backend_family`、`weight_layout_policy`（310P 为 `FORCE_NZ`）、`moe_comm_policy`（A3 为 `FUSED_OR_CAPACITY`，310P 为 `ALLGATHER`）、`quantization_backend_family`，以及约 50 个 `HardwareCapability` 中各自支持哪些（MC2、MegaMoE、FP8_ATTENTION 等）。
- **因此"设备型号"应当是页面的一个维度**（页头切换 A2/A3/A5/310P）。按型号做条件解析是确定性的，不需要 agent。

**④ patch 体系：目标清单可以确定性得到，相关性需要判断。**
- **加载入口：** `patch/platform/__init__.py` 和 `patch/worker/__init__.py` 中的 import 列表。条件包括硬件 capability、环境变量（如 `DYNAMIC_EPLB`）和 `HAS_TRITON`。
- **说明文档：** `patch/__init__.py`（约 1800 行注释），每条有 Target / Why / How / Related PR / Future Plan 字段。但格式**只是半结构化**：编号有重复也有缺号，部分条目写在 `platform/__init__.py` 里，不能当作机器可读的事实来源。
- **得到精确的 patch 目标集合：** 在 mock 环境里对 `sys.modules` 中的 vllm 符号做 patch 前后的 identity diff。这是运行时事实，比解析注释可靠。
- **相关性：** 绝大多数 patch 改的是调度器、引擎、API、结构化输出（属于非目标）。和模型结构或显存相关的是少数，例如：
  - `patch_fused_moe`：`FusedMoEFactory` → `AscendMoERunner`；
  - `worker/patch_deepseek_v2`：Indexer 的构造条件；
  - `patch_kv_cache_utils`：DSV4/GLM5 多块分配布局的每 block 字节数；
  - `patch_mamba_config`：block size 改为 128；
  - `worker/patch_kv_cache_dtype`：fp8 dtype；
  - `patch_mla_prefill_backend`。
- **"是否影响结构/显存"的判定和说明文字的提炼**属于理解性工作，适合 agent（见 §5）。

**⑤ 实测验证：Qwen3-30B-A3B × A2，纯 CPU（2026-10-31，`experiments/a2_qwen3moe/`）**

环境：vllm-ascend `24a76e8`，vllm `544f182`（vllm-ascend 声明验证过的 commit），torch `2.10.0+cpu`。没有 NPU，没有权重，没有访问 HF。

| 验证项 | 结果 |
|---|---|
| 纯 vLLM 在 meta 上构造（tp=1） | ✅ 参数 30,532,122,624，**与闭式公式完全一致**（差值 0），bf16 下 61.06 GB |
| tp=4 时单 rank 的切分 shape | ✅ qkv `[5120,2048]→[1280,2048]`；o_proj `[2048,4096]→[2048,1024]`；experts w13 `[128,1536,2048]→[128,384,2048]`、w2 `[128,2048,768]→[128,2048,192]`（无 EP 时专家按 intermediate 维 TP 切分）；单 rank 7.64B 参数、15.29 GB |
| 上游 mock 下加载 A2 profile、全部 platform/worker patch、CustomOp 注册 | ✅ 全部成功；A2 有 26 项 capability；CustomOp 替换表 31 项 |
| 带昇腾 CustomOp 在 meta 上构造 | ✅ 参数名与 shape **与纯 vLLM 完全一致**，没有 MagicMock 污染；第 0 层有 12 个模块被替换为 Ascend 类（RMSNorm、QKV/Row Linear、RotaryEmbedding、ApplyRotaryEmb、GateLinear、MoERunner、RoutedExperts、UnquantizedFusedMoEMethod） |
| patch 目标的运行时检测 | ✅ 检出 130 个被 patch 的符号，涉及 50 余个文件，其中包括 **`Qwen3MoeAttention.forward`**（来自 `patch_qwen3vl.py`），直接关系到本模型页面 |

由此得到的结论：
1. **主图参数可以完全确定性地生成**，单 rank 切分也可以，所以"权重显存"这一项可以做到精确值，而不是估算。
2. **tp>1 不能用 FakeProcessGroup**：vLLM 初始化时的组间握手会卡死（实测超时）。可行的做法是真实初始化 1 rank 的 gloo 组，再只把 TP `GroupCoordinator` 的 `world_size` 和 `rank_in_group` 换成目标值。构造层只读取这两个值。
3. **昇腾 CustomOp 不改变参数 shape**（至少对 Qwen3-MoE 的 bf16 非量化路径如此）。可以直接用 mock 环境构造，同时拿到"替换后的类"和参数，不需要兜底方案。
4. **patch 注释不能作为事实来源**：注释列出的 131 个目标中，运行时只命中 64 个；另有 58 个检出符号没有写进注释。部分原因是注释里的路径已经过时（例如 `get_mla_prefill_backend` 已经搬到 `layers.attention.mla_attention`），部分是条件 patch 在 A2 下不生效，部分是注释漏写。**运行时 identity diff 才是事实来源**，注释只用来给 agent 提炼"为什么这样改"。
5. **版本配对已在上游声明**：`.github/vllm-main-verified.commit`（main 对应的 vLLM commit）和 `.github/vllm-release-tag.commit`（当前为 `v0.32.0`）。用 vLLM main HEAD 搭配 vllm-ascend main 会直接 import 失败（`get_rope_shape_decorate` 不存在）。§2.6 的配对矩阵应直接读取这两个文件。

**仍未验证：**
- attention 后端的选择：`Attention` 不走 CustomOp 替换，而是由平台的 `get_attn_backend_cls` 在运行时决定；这次用的是 vLLM CPU 平台，不代表 NPU 上的选择。
- 量化模型（W8A8 等）在 mock 下构造时，参数 shape 是否依然可信。
- MLA / DeepSeek 类模型。
- 检测方法的局限：通过 pydantic validator 实现的 patch、对常量或 dtype 的替换，用"函数代码来源"的办法检测不到。

### 2.8 模型覆盖策略
vLLM 注册了约 200 个架构类，没法每个都配一份 config。建议：
- **策展清单** `catalog.yaml`：首批约 15–20 个代表模型（Llama3、Qwen3 Dense/MoE、Qwen3-Next、DeepSeek-V3/V3.2、Kimi-K2、GLM-4.5、gpt-oss、Qwen2.5-VL 等），每个都有 golden；
- **架构级兜底页：** 其余在 Registry 里的架构只生成"结构 + 源码链接"页（不带数值），并标为 "unverified"。
- **规则覆盖率闸门：** 遇到规则表里没有的层类时，CI 标红该模型，不发布错误的数字。

### 2.9 其他建议
- **"10 分钟"需要一条叙事线：** 首页每个模型提供一个 6 步导览——数据怎么走 → 哪层算得最多 → 显存花在哪 → 通信在哪 → 昇腾改了什么 → 与上一版有何不同。
- **URL 承载状态：** `?tp=8&ep=8&dtype=w8a8&tokens=4096&node=layers.mlp.experts`，链接一贴就能复现同一视图，方便讨论。
- **多模态：** encoder（ViT）+ projector 作为主干之前的一个分支画出，不要塞进 decoder 主图。

---

## 3. 图的设计规范（借鉴 diagram-design skill）

diagram-design skill 的核心思路有五条：**做减法**、**强调色克制使用**、**每张图有复杂度预算**、**只走正交连线**、**用语义 token 而不是色值**；导入时**提取再重画，不做直接转换**。映射到 InferAtlas：

### 3.1 "提取再重画"：不渲染原始模块树
Netron / Model Explorer 的做法是把 trace 原样画出来，所以节点密度高。InferAtlas 的做法是：**IR 来自真实代码，视图是按架构族策展的模板**（Dense / MoE / MLA / Hybrid / VLM）。模板决定哪些节点合并（比如 `q_proj/k_proj/v_proj` 在 vLLM 里本来就是 `QKVParallelLinear`，画成一个节点）、哪些下沉到详情面板（比如 rotary、bias）。另外参照 skill 的"保真度台账"，在页面角落列出"本视图合并了哪些节点"。

### 3.2 复杂度预算 → 三级视图
skill 的预算是每张图 ≤9 个节点、≤12 条箭头、≤2 处强调色。按这个预算切分层级：

| 层级 | 内容 | 节点数 |
|---|---|---|
| L0 主干 | tokens → embed → [decoder ×N] → norm → lm_head → logits → sampler | 7 |
| L1 decoder layer | input_norm → attention → residual → post_norm → MLP/MoE → residual | ≤8 |
| L2 子图 | attention 内部（qkv / rope / attn backend / o_proj + all-reduce）、MoE 内部（gate / topk / dispatch / experts / combine / shared expert） | ≤9 |

**层异构的压缩：** 所有层都一样时画一个代表层并标 `×N`；有异构时（DeepSeek 前 k 层是 dense、Gemma 的 sliding/full 交替、Qwen3-Next 的 linear/full 交替），用 **layer band**（如 `[dense ×3][MoE ×58]`）表示，点击某一段再展开对应的代表层。

### 3.3 视觉语法（语义 token）
| 语义 | 编码 |
|---|---|
| 计算节点 | 实线边框矩形 `rx=6`；节点名用 sans 600，类名和 shape 用 mono |
| 通信节点 | 不同的形状（六边形或双竖线），横跨"rank 边界"那条线 |
| 张量边 | 正交连线；**边标签就是 shape**（符号或数值），用 mono 字体，带不透明底衬 |
| KV cache 流 | 虚线，从侧栏引入 attention 节点 |
| 昇腾叠加 | **不改变主图颜色**。被替换或打了 patch 的节点在右上角加角标（`OOT` / `PATCH` / `QUANT` / `FUSED`），叠加层打开时显示虚线外框和对照卡片 |
| 强调色 | 只用于当前焦点：选中节点的数据通路，或导览当前步骤。≤2 处 |
| 热度 | FLOPs / 显存占比用节点内的细条（sparkline bar）表示，不给整个节点涂色 |

其他规则：不加阴影；结构几何对齐 4px 网格；图例放在底部横条；明暗主题都通过 CSS 变量映射到语义 token。

### 3.4 布局算法
- 主干和子图是有向无环、带 compound 和端口的图，按 node-link layout 的经验应选 **分层布局（Sugiyama）+ 正交路由** → 使用 **ELK Layered**（elkjs）。
- **布局在构建期计算**（Node 端跑 elkjs），坐标写进页面数据。好处是：①输出确定（目标 18）；②离线单文件里不需要跑布局引擎；③折叠和展开两种状态各算一次，切换时不会跳动。
- 侧栏的并行切分用**rank 泳道或矩阵视图**（行是 rank，列是层组，单元格里画张量切片），KV cache 用**分组堆叠条**，显存用**瀑布图**（权重 → 激活 → KV → 预留 → 未建模），版本 diff 用**左右对照，变动的节点和边用强调色标出**。

### 3.5 详情面板（目标 15）
内容自上而下：类名 + 源码永久链接 → 输入/输出 shape（可在符号和数值间切换）→ 参数（切分前 / 切分后 / 字节）→ FLOPs（prefill/decode）→ 算术强度与 roofline 上的位置点 → 通信（逻辑字节/总线字节）→ 昇腾差异卡片 → 适用的规则 id 和公式原文（方便核查）。

---

## 4. 系统架构

```
              ┌────────── cron 轮询 releases（两个仓库）──────────┐
              ▼                                                    │
  [resolve] 配对矩阵 → (vllm_commit, ascend_commit, model, config_rev)
              │
  [extract]   Python，CPU runner
   ├─ registry_scan   ModelRegistry → arch → (module, class, file:line)
   ├─ meta_build      meta device + FakePG 实例化 → 模块树、参数 shape、层类
   ├─ ascend_scan     AST：CustomOp oot 映射 / patch 目标 / attn backend / quant
   └─ sub_trace       子模块 meta 前向，校验 shape 规则
              │
  [lower]     模块树 + 规则表(按版本区间) + 标注文件 → InferAtlas IR（JSON，全部是符号表达式）
              │                         └→ 未匹配层类 / 失效标注 → 待判定清单(triage.json)
              │                            → 覆盖率闸门失败，交给维护 skill 处理（§5）
  [verify]    golden diff · 源码行校验 · 校准数据比对(±10%)
              │
  [layout]    Node：elkjs 计算各视图/折叠态坐标 → view.json
              │
  [render]    Vite 静态站 + 单文件 HTML（vite-plugin-singlefile，内联数据和字体）
              ▼
           GitHub Pages：/v/{vllm}+{ascend}/{model}/   index.json   diff 索引
```

### 4.1 IR（核心契约，也是 golden 的载体）
```jsonc
{
  "snapshot": {"vllm": "v0.11.0@abc123", "ascend": "v0.11.0rc1@def456",
               "model": "Qwen/Qwen3-30B-A3B", "config_rev": "…", "generator": "0.3.0"},
  "symbols": {"H": 2048, "L": 48, "E": 128, "topk": 8, "Hq": 32, "Hkv": 4, "D": 128, "V": 151936},
  "params":  ["tp","pp","dp","ep","T","S","dtype_w","dtype_kv", "..."],   // 运行时可调
  "nodes": [{
    "id": "L.attn.qkv", "kind": "compute", "class": "QKVParallelLinear",
    "src": "vllm/model_executor/models/qwen3_moe.py#L210", "rule": "linear.column@>=0.10",
    "in": ["T","H"], "out": ["T","(Hq+2*max(1,Hkv/tp)*tp/tp)*D/tp"],   // 表达式字符串
    "weight_bytes": "...", "flops": {"prefill": "2*T*H*...", "decode": "..."},
    "parallel": {"tp": {"dim": "out", "comm": null}},
    "ascend": {"tag": "QUANT", "note": "W8A8 → AscendW8A8LinearMethod", "src": "vllm_ascend/…#L88"}
  }],
  "edges": [...], "groups": [...layer bands...], "unmodelled": ["acl_graph", "hccl_buffer"]
}
```
- **只用一种表达式 DSL**（四则运算、`max/min/ceil`、条件），浏览器端用 TS 求值，实现目标 14 的实时重算。Python 侧只负责**生成表达式**，不重复实现求值，避免两套实现结果不一致。golden 测试在 CI 里用 Node 对固定的参数网格求值后比对。
- golden = IR + 若干参数点上的求值结果，提交入库。上游变动表现为 diff，CI 报警并贴出差异（目标 23）。

### 4.2 规则表
`rules/{range}/*.yaml`，键是 `layer class × version range`：
```yaml
- id: linear.row
  match: {class: RowParallelLinear, vllm: ">=0.9,<0.13"}
  shard: {tp: in}
  out: "[T, out]"
  weight: "in*out/tp * bytes(dtype_w) + scale_bytes(quant, in/tp, out)"
  flops: "2*T*in*out/tp"
  comm: {op: all_reduce, when: "reduce_results && tp>1", bytes: "T*out*bytes(dtype_a)"}
```
引入新规则区间的触发条件是 golden diff 报警，由人工确认后提交，以此对应目标 19 的"只在大重构时更新"。

### 4.3 源码定位 100%（目标 20）
- 行号来自 `inspect.getsourcelines` 加 AST，链接里用 commit hash。
- CI 校验：取 `file@commit` 的对应行，必须出现 `class X` / `def f`，否则构建失败。
- 昇腾 patch 节点给出两条链接：原函数和 patch 实现。

### 4.4 仓库布局
```
inferatlas/
  generator/   (python) resolve/ extract/ lower/ verify/
  rules/       按版本区间的规则 YAML
  annotations/ 条件分支判定、昇腾差异说明（agent 起草、人审合入，带源码 hash 锚点）
  catalog.yaml 模型清单 + 版本配对矩阵 + 设备预设
  golden/      IR + 求值结果
  calibration/ NPU 真实启动日志解析结果
  web/         (ts) expr 求值器 / elk 布局脚本 / 视图组件 / 单文件打包
  skills/inferatlas-maintain/  维护者 skill（§5）
  .github/workflows/ poll.yml  build.yml  nightly.yml  calibrate.yml(self-hosted)  draft-pr.yml
```

---

## 5. Skill + 脚本的分工

### 5.1 原则：agent 只在编写期工作，构建期只跑确定性脚本
如果让 agent 直接产出页面上的数字或结构，会同时破坏三个目标：目标 18（确定性：同样输入得到同样输出）、目标 20（源码定位 100% 正确：agent 写的行号不可信）、目标 23（golden 一旦会抖动，回归测试就失去意义）。

因此分工是 **"agent 写数据文件，脚本消费这些数据"**：

| | 脚本（generator/，确定性） | Skill（agent，编写期） |
|---|---|---|
| 职责 | AST 扫描、meta 实例化、源码定位、规则匹配、表达式求值、布局、校验、打包 | 阅读源码和上游 diff，起草规则和标注，解释"为什么" |
| 输入 | 源码 @commit、HF config、`rules/`、`annotations/` | 脚本产出的 `triage.json`、上游 diff、相关源码 |
| 输出 | IR、golden diff、`triage.json`、站点 | `rules/*.yaml`、`annotations/*.yaml`、PR 描述 |
| 能否写行号 | 能（唯一来源） | **不能**，只用符号名引用代码 |
| 能否写数值 | 能（由公式求值得出） | **不能**，只写公式和判定，数值一律由脚本算 |
| 在 CI 构建中运行 | 是 | 否（只在维护流程中运行，产物经人审合入） |

按源码核对结果（§2.7），昇腾叠加层的信息分成两类：

| 信息 | 谁来产出 | 原因 |
|---|---|---|
| 某个 CustomOp 被替换成哪个类、某型号的 capability、实际被 patch 的符号集合 | 脚本 | 字面量表或运行时 identity diff，有唯一正确答案 |
| 替换属于 `KERNEL` / `LAYOUT` / `SEMANTIC` 中的哪一级 | agent 起草，人审 | 要读懂 Ascend 实现是否改变了 shape、通信或布局 |
| 某个 patch 是否影响结构或显存（大多数只影响调度/API，应过滤掉） | agent 起草，人审 | 判断性工作，注释是半结构化的 |
| patch 函数体内的模型类型判断、环境变量、`additional_config` 等条件 | agent 写 `applies_when`，脚本校验符号存在 | 条件散落在函数体内，没有统一的声明 |
| 叠加层卡片上的"为什么这样改"说明 | agent 从 Why/How 注释和源码提炼 | 文字工作 |

产品本身不是 skill：用户打开站点就能看，全程不经过 agent（目标 1、16）。skill 是**维护者的工具**，正好承接目标 18 里"内部分支、未发布模型、调试规则由维护者本地运行"的几种情况，以及目标 19 里"上游大重构时更新规则表"这件阅读理解类工作。

### 5.2 衔接点：待判定清单与标注文件
脚本遇到自己判定不了的情况时不猜，而是写入 `triage.json`：
```jsonc
{
  "snapshot": "vllm@abc123+ascend@def456",
  "items": [
    {"kind": "unknown_layer_class", "symbol": "vllm.model_executor.layers.foo.BarLinear",
     "used_by": ["Qwen3NextForCausalLM"]},
    {"kind": "conditional_override", "symbol": "vllm_ascend.platform.NPUPlatform.get_attn_backend_cls",
     "candidates": ["AscendAttentionBackend", "AscendMLABackend", "..."], "conditions": ["use_mla", "..."]},
    {"kind": "stale_annotation", "id": "ascend.moe.mc2", "anchor_hash_old": "…", "anchor_hash_new": "…"},
    {"kind": "golden_diff", "model": "deepseek-ai/DeepSeek-V3", "diff": "…"}
  ]
}
```
agent 处理后写入标注文件，格式由 schema 约束：
```yaml
- id: ascend.attn.backend.mla
  target: vllm_ascend.platform.NPUPlatform.get_attn_backend_cls   # 只写符号，不写行号
  anchor_hash: "sha256:…"            # 脚本对目标源码片段计算；源码一变即失效
  applies_when: "use_mla && !use_sparse"
  resolves_to: vllm_ascend.attention.mla_v1.AscendMLABackend
  level: LAYOUT                      # KERNEL | LAYOUT | SEMANTIC
  note: "MLA 走昇腾专用后端，KV 以 nope / rope 两个 cache 分开存放"
  reviewed_by: "<maintainer>"
```
规则：
- 源码位置统一由脚本把 `target` 解析为 `file:line@commit`；解析失败即构建失败。
- `anchor_hash` 不匹配时，标注自动失效、回到 `triage.json`，所以上游改动不会悄悄沿用旧判定。
- schema 校验、golden、覆盖率闸门对 agent 产物与人写产物一视同仁。

### 5.3 Skill 结构
```
skills/inferatlas-maintain/
  SKILL.md                    # 入口：何时使用、四个工作流、硬约束（不写行号/数值）
  references/
    rule-schema.md            # 规则 DSL、标注 schema、示例
    vllm-parallel-layers.md   # 并行层语义速查（Column/Row/QKV/Merged/Vocab/FusedMoE…）
    ascend-overlay.md         # customop / patch / quant / attn backend 的判定方法
    memory-accounting.md      # vLLM 显存账本口径（§2.2），用于判断 golden diff
  scripts/                    # 薄封装，调用 generator/ 的入口
    scan.sh  triage.sh  verify.sh  golden_diff.sh  preview.sh
```

### 5.4 四个工作流
| 工作流 | 触发 | 步骤 |
|---|---|---|
| **新版本上线** | cron 发现新 tag | `scan` → `triage` → agent 逐项处理 → `verify` → 草稿 PR |
| **新模型接入** | 维护者往 catalog 加模型 | 拉取 config → `scan` → 给未知层类补规则 → 首次生成 golden → 人工核对后入库 |
| **golden 报警修复** | golden diff | agent 判断是上游真实变化（更新 golden，PR 里说明原因）还是规则失效（新增版本区间规则） |
| **昇腾叠加层审阅** | 出现新 patch / 条件分支 / 失效标注 | agent 读 patch 和上游被 patch 的函数，写 `applies_when`、`tag`、`note` |

### 5.5 CI 中的位置
```
poll.yml (cron) ──新 tag──▶ build.yml（纯脚本）
                              ├─ triage 为空  → verify → 发布
                              └─ triage 非空  → 不发布受影响的模型×设备单元格 → draft-pr.yml
                                                   └─ claude-code-action 运行 skill → 草稿 PR → 人审合入 → 重新 build
```
- triage 非空时只拦下受影响的模型×设备单元格（affected model×device cell），其余单元格照常发布。
- H100 baseline 有 triage 时，同时拦下该模型全部 Ascend 单元格；缺失或硬错误单元格阻止整次发布。
- 发布产物只依赖已合入的文件，任何人在同一 commit 上重跑 build 都得到同样结果。

---

## 6. 分期计划

| 阶段 | 范围 | 验收 |
|---|---|---|
| **M0 原型（2 周）** | 1 个版本 × Qwen3-8B（Dense）；meta 实例化 + 规则 → IR → 手工布局的 L0/L1 视图；TP 可调 | 权重显存与真实日志偏差 ≤5%；源码链接全部有效 |
| **M1 核心（4 周）** | MoE（Qwen3-MoE、DeepSeek-V3/MLA）；L2 子图；FLOPs、roofline、通信；elk 构建期布局；单文件导出；URL 状态 | 5 个模型有 golden；KV token 数偏差 ≤10% |
| **M2 昇腾叠加（3 周）** | AST 扫描、角标、对照卡片、NZ/量化/KV 布局差异；`triage.json` + 标注文件 + anchor_hash 失效机制；维护 skill 初版（昇腾审阅、新模型接入两个工作流）；校准 runner | 叠加层与人工审阅一致；标注随源码变化正确失效 |
| **M3 发布自动化（2 周）** | cron 轮询、配对矩阵、版本矩阵首页、版本 diff 视图、nightly；skill 补齐新版本上线、golden 修复两个工作流，接入 draft-pr.yml | triage 为空的新 tag 24 小时内自动上线；非空时自动生成草稿 PR |
| **M4 扩展** | Hybrid/Mamba、VLM、MTP、DCP/SP、导览模式 | — |

## 7. 主要风险
1. **meta 实例化的兼容性**：vLLM 模型构造依赖 `VllmConfig` 和分布式状态，跨版本的 API 会变 → 把实例化做成"按版本区间的适配器"，与规则表的版本区间同步维护。
2. **昇腾行为是运行时决定的**：AST 扫不到条件分支（比如按 SoC 型号选 kernel）→ 进入 `triage.json`，由 skill 起草 `applies_when` 标注、人审合入；页面标注"条件生效"，并以校准 runner 的结果为准。
3. **10% 精度的边界**：graph capture、碎片化等开销没有建模 → 只承诺"权重"和"KV token 数"两项，其余以区间加注释的形式给出。
4. **agent 起草的规则或标注有误**：agent 不写行号和数值，产物要过 schema、golden、覆盖率闸门和人审；标注带 anchor_hash，源码变化后自动失效、重新判定。
