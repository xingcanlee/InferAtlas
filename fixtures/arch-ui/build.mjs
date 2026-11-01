#!/usr/bin/env node
/**
 * Hand-authored arch.json fixtures for the U1 architecture UI (docs/arch-contract.md, v1).
 * NOT generator output: written by hand from the public HF configs (catalog.yaml where present)
 * and vLLM 544f182 class locations, so the UI can be built before the S1 emitter lands.
 *
 *   node fixtures/arch-ui/build.mjs        # rewrites fixtures/arch-ui/<model>/arch.json
 *
 * Shapes are global (tp is a symbol, default 1) expression strings in the expr DSL.
 * Contract v1.3: `io.internals` with canonical names, de-duplicated notes, head-sharing attention
 * variants (locality/sparsity in extras), per-period pattern labels, mHC (`mhc_pre_attn` …,
 * generic components, `residual_streams [T, HC, H]`), stages in dataflow array order.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const VLLM = "544f182d9f0c9bfe8bd7645525951d5e3692a542";

const src = (file, line) => ({
  file,
  line,
  url: `https://github.com/vllm-project/vllm/blob/${VLLM}/${file}#L${line}`,
});
const SRC = {
  rmsnorm: src("vllm/model_executor/layers/layernorm.py", 39),
  gemmaRms: src("vllm/model_executor/layers/layernorm.py", 145),
  embed: src("vllm/model_executor/layers/vocab_parallel_embedding.py", 206),
  lmHead: src("vllm/model_executor/layers/vocab_parallel_embedding.py", 578),
  qwen3Attn: src("vllm/model_executor/models/qwen3.py", 65),
  qwen2Mlp: src("vllm/model_executor/models/qwen2.py", 81),
  qwen3MoeAttn: src("vllm/model_executor/models/qwen3_moe.py", 244),
  qwen3MoeBlock: src("vllm/model_executor/models/qwen3_moe.py", 133),
  dsMla: src("vllm/model_executor/models/deepseek_v2.py", 1000),
  dsMlp: src("vllm/model_executor/models/deepseek_v2.py", 244),
  dsMoe: src("vllm/model_executor/models/deepseek_v2.py", 291),
  dsMtp: src("vllm/model_executor/models/deepseek_mtp.py", 76),
  v4Attn: src("vllm/models/deepseek_v4/attention.py", 132),
  v4Moe: src("vllm/models/deepseek_v4/nvidia/model.py", 205),
  v4Layer: src("vllm/models/deepseek_v4/nvidia/model.py", 585),
  v4Model: src("vllm/models/deepseek_v4/nvidia/model.py", 811),
  v4Mtp: src("vllm/models/deepseek_v4/nvidia/mtp.py", 79),
  mhcPre: src("vllm/model_executor/layers/mhc.py", 109),
  mhcPost: src("vllm/model_executor/layers/mhc.py", 564),
  hcHead: src("vllm/model_executor/layers/mhc.py", 649),
  ossAttn: src("vllm/model_executor/models/gpt_oss.py", 92),
  ossMlp: src("vllm/model_executor/models/gpt_oss.py", 359),
  nextGdn: src("vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py", 527),
  nextAttn: src("vllm/model_executor/models/qwen3_next.py", 280),
  nextMoe: src("vllm/model_executor/models/qwen3_next.py", 131),
};

const BF16 = "bfloat16";
const t = (name, shape, dtype = BF16) => ({ name, shape, dtype });
const p = (name, shape, dtype = BF16) => ({ name, shape, dtype });
const io = (inputs, outputs) => ({ inputs, outputs });
/** Fully qualified class name (contract v1.1 §10) from the defining source file. */
const qualify = (cls, source) =>
  source ? `${source.file.replace(/\.py$/, "").replaceAll("/", ".")}.${cls}` : cls;
const mod = (path, cls, source, params, status = "ruled", replaced = null) => ({
  path,
  class_name: qualify(cls, source),
  source,
  status,
  replaced,
  params,
});
const HID = ["T", "H"];
const hidden = (name = "hidden_states") => t(name, HID);
const sym = (value, description) => ({ value, description });

// ------------------------------------------------------------------------- shared pieces

const rms = (path, dim = "H", cls = "RMSNorm", source = SRC.rmsnorm, extra = {}) => ({
  component: { kind: "norm", variant: "rmsnorm", eps: 1e-6 },
  io: io([hidden()], [hidden()]),
  module: mod(path, cls, source, [p("weight", [dim])], extra.status, extra.replaced),
});

const normSub = (id, path, opts = {}) => ({
  id,
  role: "norm",
  ...rms(path, "H", opts.cls, opts.source, opts),
});

const addSub = (id) => ({
  id,
  role: "residual_add",
  component: null,
  io: io([hidden(), hidden("residual")], [hidden()]),
  module: null,
});

const swigluSub = (path, cls, source, inter) => ({
  id: "mlp",
  role: "ffn",
  component: { kind: "mlp", variant: "swiglu", intermediate: inter.value },
  io: io([hidden()], [hidden()]),
  module: mod(path, cls, source, [
    p("gate_up_proj.weight", [`2*${inter.sym}/tp`, "H"]),
    p("down_proj.weight", ["H", `${inter.sym}/tp`]),
  ]),
});

const embedStage = (vocab, hidden_, tied = false) => ({
  id: "embed",
  kind: "embedding",
  label: "Token embedding",
  component: { kind: "embedding", vocab, hidden: hidden_, tied },
  module: mod("model.embed_tokens", "VocabParallelEmbedding", SRC.embed, [
    p("weight", ["cdiv(V, tp)", "H"]),
  ]),
  io: io([t("input_ids", ["T"], "int64")], [hidden()]),
});

const finalNorm = (cls = "RMSNorm", source = SRC.rmsnorm) => ({
  id: "final_norm",
  kind: "norm",
  label: "Final norm",
  component: { kind: "norm", variant: "rmsnorm", eps: 1e-6 },
  module: mod("model.norm", cls, source, [p("weight", ["H"])]),
  io: io([hidden(), hidden("residual")], [hidden()]),
});

const lmHead = (vocab, hidden_, tied = false) => ({
  id: "lm_head",
  kind: "lm_head",
  label: "LM head",
  component: { kind: "lm_head", vocab, hidden: hidden_, tied },
  module: mod("lm_head", "ParallelLMHead", SRC.lmHead, [p("weight", ["cdiv(V, tp)", "H"])]),
  io: io([hidden()], [t("logits", ["T", "V"])]),
});

const band = (id, layers, block, pattern = null) => ({
  id,
  layers,
  count: layers[1] - layers[0] + 1,
  pattern,
  block,
});

const tSym = sym(4096, "tokens in the batch (prefill chunk)");
const tpSym = sym(1, "tensor-parallel size");

// ---------------------------------------------------------------------------- Qwen3-8B

function qwen3_8b() {
  const L = 0;
  const lp = `model.layers.${L}`;
  return {
    schema_version: 1,
    model: {
      id: "qwen3-8b",
      hf_id: "Qwen/Qwen3-8B",
      architecture: "Qwen3ForCausalLM",
      family: "dense",
    },
    device: "A2",
    symbols: {
      D: sym(128, "head dim"),
      H: sym(4096, "hidden size"),
      Hkv: sym(8, "key/value heads"),
      Hq: sym(32, "query heads"),
      I: sym(12288, "MLP intermediate size"),
      L: sym(36, "decoder layers"),
      T: tSym,
      V: sym(151936, "vocabulary size"),
      tp: tpSym,
    },
    input: t("input_ids", ["T"], "int64"),
    output: t("logits", ["T", "V"]),
    stages: [
      embedStage(151936, 4096),
      {
        id: "layers",
        kind: "stack",
        label: "Decoder layers",
        bands: [
          band("dense", [0, 35], {
            kind: "decoder",
            residual: "pre_norm",
            sublayers: [
              normSub("input_layernorm", `${lp}.input_layernorm`, {
                status: "unannotated_override",
                replaced: { baseline_class: "RMSNorm", ascend_class: "AscendRMSNorm" },
              }),
              {
                id: "self_attn",
                role: "mixer",
                component: {
                  kind: "attention",
                  variant: "gqa",
                  heads_q: 32,
                  heads_kv: 8,
                  head_dim: 128,
                  rope: { kind: "rope", theta: 1000000, scaling: null },
                  qk_norm: true,
                  extras: {},
                },
                io: io([t("positions", ["T"], "int64"), hidden()], [hidden()]),
                module: mod(`${lp}.self_attn`, "Qwen3Attention", SRC.qwen3Attn, [
                  p("qkv_proj.weight", ["(Hq+2*Hkv)*D/tp", "H"]),
                  p("o_proj.weight", ["H", "Hq*D/tp"]),
                  p("q_norm.weight", ["D"]),
                  p("k_norm.weight", ["D"]),
                ]),
              },
              addSub("residual_1"),
              normSub("post_attention_layernorm", `${lp}.post_attention_layernorm`),
              swigluSub(`${lp}.mlp`, "Qwen2MLP", SRC.qwen2Mlp, { sym: "I", value: 12288 }),
              addSub("residual_2"),
            ],
          }),
        ],
      },
      finalNorm(),
      lmHead(151936, 4096),
    ],
    notes: [
      {
        severity: "warn",
        target: "dense/input_layernorm",
        text: "vllm-ascend replaces RMSNorm with AscendRMSNorm (fused add + norm); no reviewed annotation yet.",
        triage_kind: "unannotated_override",
      },
      {
        severity: "warn",
        target: "lm_head",
        text: "Golden diff: lm_head weight bytes differ from the golden by vocab padding at tp=8.",
        triage_kind: "golden_diff",
      },
      {
        severity: "info",
        target: "dense/self_attn",
        text: "Per-head RMSNorm on q and k (qk_norm) before RoPE.",
        triage_kind: null,
      },
    ],
  };
}

// ----------------------------------------------------------------------- Qwen3-30B-A3B

function qwen3_30b() {
  const lp = "model.layers.0";
  return {
    schema_version: 1,
    model: {
      id: "qwen3-30b-a3b",
      hf_id: "Qwen/Qwen3-30B-A3B",
      architecture: "Qwen3MoeForCausalLM",
      family: "moe",
    },
    device: "A2",
    symbols: {
      D: sym(128, "head dim"),
      E: sym(128, "routed experts"),
      H: sym(2048, "hidden size"),
      Hkv: sym(4, "key/value heads"),
      Hq: sym(32, "query heads"),
      I_moe: sym(768, "expert intermediate size"),
      L: sym(48, "decoder layers"),
      T: tSym,
      V: sym(151936, "vocabulary size"),
      topk: sym(8, "experts per token"),
      tp: tpSym,
    },
    input: t("input_ids", ["T"], "int64"),
    output: t("logits", ["T", "V"]),
    stages: [
      embedStage(151936, 2048),
      {
        id: "layers",
        kind: "stack",
        label: "Decoder layers",
        bands: [
          band("moe", [0, 47], {
            kind: "decoder",
            residual: "pre_norm",
            sublayers: [
              normSub("input_layernorm", `${lp}.input_layernorm`),
              {
                id: "self_attn",
                role: "mixer",
                component: {
                  kind: "attention",
                  variant: "gqa",
                  heads_q: 32,
                  heads_kv: 4,
                  head_dim: 128,
                  rope: { kind: "rope", theta: 1000000, scaling: null },
                  qk_norm: true,
                  extras: {},
                },
                io: io([t("positions", ["T"], "int64"), hidden()], [hidden()]),
                module: mod(`${lp}.self_attn`, "Qwen3MoeAttention", SRC.qwen3MoeAttn, [
                  p("qkv_proj.weight", ["(Hq+2*Hkv)*D/tp", "H"]),
                  p("o_proj.weight", ["H", "Hq*D/tp"]),
                  p("q_norm.weight", ["D"]),
                  p("k_norm.weight", ["D"]),
                ]),
              },
              addSub("residual_1"),
              normSub("post_attention_layernorm", `${lp}.post_attention_layernorm`),
              {
                id: "mlp",
                role: "ffn",
                component: {
                  kind: "moe",
                  experts: 128,
                  top_k: 8,
                  shared_experts: 0,
                  router: { scoring: "softmax", norm_topk: true, groups: null },
                  expert: { variant: "swiglu", intermediate: 768 },
                  shared: null,
                },
                io: io([hidden()], [hidden()]),
                module: mod(
                  `${lp}.mlp`,
                  "Qwen3MoeSparseMoeBlock",
                  SRC.qwen3MoeBlock,
                  [
                    p("gate.weight", ["E", "H"]),
                    p("experts.w13_weight", ["E", "2*I_moe/tp", "H"]),
                    p("experts.w2_weight", ["E", "H", "I_moe/tp"]),
                  ],
                  "ascend_override",
                  { baseline_class: "FusedMoE", ascend_class: "AscendFusedMoE" },
                ),
              },
              addSub("residual_2"),
            ],
          }),
        ],
      },
      finalNorm(),
      lmHead(151936, 2048),
    ],
    notes: [
      {
        severity: "info",
        target: "moe/mlp",
        text: "Router: softmax over 128 experts, top-8, weights renormalised (norm_topk_prob). AscendFusedMoE uses MC2 dispatch when ep > 1 (annotated).",
        triage_kind: null,
      },
    ],
  };
}

// -------------------------------------------------------------------------- DeepSeek-V3

function deepseek_v3() {
  const mla = (L) => ({
    id: "self_attn",
    role: "mixer",
    component: {
      kind: "attention",
      variant: "mla",
      heads_q: 128,
      heads_kv: 128,
      head_dim: 192,
      rope: {
        kind: "yarn",
        theta: 10000,
        scaling: { factor: 40, original_max_position_embeddings: 4096, beta_fast: 32, beta_slow: 1 },
      },
      qk_norm: false,
      extras: {
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
      },
    },
    io: io([t("positions", ["T"], "int64"), hidden()], [hidden()]),
    module: mod(
      `model.layers.${L}.self_attn`,
      "DeepseekV2MLAAttention",
      SRC.dsMla,
      [
        p("fused_qkv_a_proj.weight", ["Rq+Rkv+Dr", "H"]),
        p("q_a_layernorm.weight", ["Rq"]),
        p("q_b_proj.weight", ["Hq*(Dn+Dr)/tp", "Rq"]),
        p("kv_a_layernorm.weight", ["Rkv"]),
        p("kv_b_proj.weight", ["Hq*(Dn+Dv)/tp", "Rkv"]),
        p("o_proj.weight", ["H", "Hq*Dv/tp"]),
      ],
      "ascend_override",
      {
        baseline_class: "MultiHeadLatentAttentionWrapper",
        ascend_class: "AscendMultiHeadLatentAttention",
      },
    ),
  });
  const block = (L, ffn) => ({
    kind: "decoder",
    residual: "pre_norm",
    sublayers: [
      normSub("input_layernorm", `model.layers.${L}.input_layernorm`),
      mla(L),
      addSub("residual_1"),
      normSub("post_attention_layernorm", `model.layers.${L}.post_attention_layernorm`),
      ffn,
      addSub("residual_2"),
    ],
  });
  const moe = {
    id: "mlp",
    role: "ffn",
    component: {
      kind: "moe",
      experts: 256,
      top_k: 8,
      shared_experts: 1,
      router: { scoring: "sigmoid", norm_topk: true, groups: { n_group: 8, topk_group: 4 } },
      expert: { variant: "swiglu", intermediate: 2048 },
      shared: { variant: "swiglu", intermediate: 2048 },
    },
    io: io([hidden()], [hidden()]),
    module: mod(
      "model.layers.3.mlp",
      "DeepseekV2MoE",
      SRC.dsMoe,
      [
        p("gate.weight", ["E", "H"]),
        p("gate.e_score_correction_bias", ["E"], "float32"),
        p("experts.w13_weight", ["E", "2*I_moe/tp", "H"]),
        p("experts.w2_weight", ["E", "H", "I_moe/tp"]),
        p("shared_experts.gate_up_proj.weight", ["2*I_moe/tp", "H"]),
        p("shared_experts.down_proj.weight", ["H", "I_moe/tp"]),
      ],
      "ascend_override",
      { baseline_class: "SharedFusedMoE", ascend_class: "AscendSharedFusedMoE" },
    ),
  };
  return {
    schema_version: 1,
    model: {
      id: "deepseek-v3",
      hf_id: "deepseek-ai/DeepSeek-V3",
      architecture: "DeepseekV3ForCausalLM",
      family: "mla",
    },
    device: "A2",
    symbols: {
      Dn: sym(128, "qk_nope_head_dim"),
      Dr: sym(64, "qk_rope_head_dim"),
      Dv: sym(128, "v_head_dim"),
      E: sym(256, "routed experts"),
      H: sym(7168, "hidden size"),
      Hq: sym(128, "attention heads"),
      I: sym(18432, "dense MLP intermediate size"),
      I_moe: sym(2048, "expert intermediate size"),
      L: sym(61, "decoder layers"),
      Rkv: sym(512, "kv_lora_rank"),
      Rq: sym(1536, "q_lora_rank"),
      T: tSym,
      V: sym(129280, "vocabulary size"),
      topk: sym(8, "experts per token"),
      tp: tpSym,
    },
    input: t("input_ids", ["T"], "int64"),
    output: t("logits", ["T", "V"]),
    stages: [
      embedStage(129280, 7168),
      {
        id: "layers",
        kind: "stack",
        label: "Decoder layers",
        bands: [
          band(
            "dense",
            [0, 2],
            block(
              0,
              swigluSub("model.layers.0.mlp", "DeepseekV2MLP", SRC.dsMlp, {
                sym: "I",
                value: 18432,
              }),
            ),
          ),
          band("moe", [3, 60], block(3, moe)),
        ],
      },
      finalNorm(),
      lmHead(129280, 7168),
      {
        id: "mtp",
        kind: "other",
        placement: "aside",
        label: "Multi-token prediction",
        component: {
          kind: "generic",
          class_name: qualify("DeepSeekMultiTokenPredictorLayer", SRC.dsMtp),
          children: [
            { name: "enorm", class_name: "RMSNorm" },
            { name: "hnorm", class_name: "RMSNorm" },
            { name: "eh_proj", class_name: "ReplicatedLinear" },
            { name: "mtp_block", class_name: "DeepseekV2DecoderLayer" },
            { name: "shared_head", class_name: "SharedHead" },
          ],
        },
        module: mod(
          "model.layers.61",
          "DeepSeekMultiTokenPredictorLayer",
          SRC.dsMtp,
          [
            p("enorm.weight", ["H"]),
            p("hnorm.weight", ["H"]),
            p("eh_proj.weight", ["H", "2*H"]),
            p("shared_head.norm.weight", ["H"]),
          ],
          "generic",
        ),
        io: io([t("input_ids", ["T"], "int64"), hidden("previous_hidden_states")], [hidden()]),
      },
    ],
    notes: [
      {
        severity: "warn",
        target: "mtp",
        text: "DeepSeekMultiTokenPredictorLayer has no lowering rule; drawn as a generic component (used only for speculative decoding).",
        triage_kind: "unknown_layer_class",
      },
      {
        severity: "warn",
        target: "moe/mlp",
        text: "AscendSharedFusedMoE dispatch depends on ep size (MC2 vs all-gather); the override is conditional.",
        triage_kind: "conditional_override",
      },
      {
        severity: "info",
        target: "layers",
        text: "first_k_dense_replace = 3: layers 0–2 use a dense SwiGLU MLP, layers 3–60 use MoE.",
        triage_kind: null,
      },
      {
        severity: "info",
        target: "moe/mlp",
        text: "Group-limited routing: sigmoid scores + e_score_correction_bias, 4 of 8 groups, top-8, routed_scaling_factor 2.5.",
        triage_kind: null,
      },
    ],
  };
}

// --------------------------------------------------------------------- Qwen3-Next-80B-A3B

function qwen3_next() {
  const PATTERN = "3× linear + 1× full";
  const norm = (id, L) =>
    normSub(id, `model.layers.${L}.${id}`, { cls: "GemmaRMSNorm", source: SRC.gemmaRms });
  const moe = (L) => ({
    id: "mlp",
    role: "ffn",
    component: {
      kind: "moe",
      experts: 512,
      top_k: 10,
      shared_experts: 1,
      router: { scoring: "softmax", norm_topk: true, groups: null },
      expert: { variant: "swiglu", intermediate: 512 },
      shared: { variant: "swiglu", intermediate: 512 },
    },
    io: io([hidden()], [hidden()]),
    module: mod(
      `model.layers.${L}.mlp`,
      "Qwen3NextSparseMoeBlock",
      SRC.nextMoe,
      [
        p("gate.weight", ["E", "H"]),
        p("shared_expert_gate.weight", ["1", "H"]),
        p("experts.w13_weight", ["E", "2*I_moe/tp", "H"]),
        p("experts.w2_weight", ["E", "H", "I_moe/tp"]),
        p("shared_expert.gate_up_proj.weight", ["2*I_sh/tp", "H"]),
        p("shared_expert.down_proj.weight", ["H", "I_sh/tp"]),
      ],
      "ascend_override",
      { baseline_class: "SharedFusedMoE", ascend_class: "AscendSharedFusedMoE" },
    ),
  });
  const linear = (L) => ({
    kind: "decoder",
    residual: "pre_norm",
    sublayers: [
      norm("input_layernorm", L),
      {
        id: "linear_attn",
        role: "mixer",
        component: {
          kind: "attention",
          variant: "linear",
          heads_q: 16,
          heads_kv: 16,
          head_dim: 128,
          rope: null,
          qk_norm: false,
          extras: {
            kind: "gated_deltanet",
            num_k_heads: 16,
            num_v_heads: 32,
            head_k_dim: 128,
            head_v_dim: 128,
            conv_kernel: 4,
            output_gate: true,
          },
        },
        io: io([hidden()], [hidden()]),
        module: mod(
          `model.layers.${L}.linear_attn`,
          "QwenGatedDeltaNetAttention",
          SRC.nextGdn,
          [
            p("in_proj_qkvz.weight", ["(2*Nk*Dk+2*Nv*Dv)/tp", "H"]),
            p("in_proj_ba.weight", ["2*Nv/tp", "H"]),
            p("conv1d.weight", ["(2*Nk*Dk+Nv*Dv)/tp", "1", "K"]),
            p("A_log", ["Nv/tp"], "float32"),
            p("dt_bias", ["Nv/tp"]),
            p("norm.weight", ["Dv"]),
            p("out_proj.weight", ["H", "Nv*Dv/tp"]),
          ],
          "unannotated_override",
          null,
        ),
      },
      addSub("residual_1"),
      norm("post_attention_layernorm", L),
      moe(L),
      addSub("residual_2"),
    ],
  });
  const full = (L) => ({
    kind: "decoder",
    residual: "pre_norm",
    sublayers: [
      norm("input_layernorm", L),
      {
        id: "self_attn",
        role: "mixer",
        component: {
          kind: "attention",
          variant: "gqa",
          heads_q: 16,
          heads_kv: 2,
          head_dim: 256,
          rope: { kind: "rope", theta: 10000000, scaling: null, partial_rotary_factor: 0.25 },
          qk_norm: true,
          extras: { output_gate: true },
        },
        io: io([t("positions", ["T"], "int64"), hidden()], [hidden()]),
        module: mod(`model.layers.${L}.self_attn`, "Qwen3NextAttention", SRC.nextAttn, [
          p("qkv_proj.weight", ["(2*Hq*D+2*Hkv*D)/tp", "H"]),
          p("o_proj.weight", ["H", "Hq*D/tp"]),
          p("q_norm.weight", ["D"]),
          p("k_norm.weight", ["D"]),
        ]),
      },
      addSub("residual_1"),
      norm("post_attention_layernorm", L),
      moe(L),
      addSub("residual_2"),
    ],
  });
  const bands = [];
  for (let g = 0; g < 12; g++) {
    const s = g * 4;
    bands.push(band(`linear_${s}_${s + 2}`, [s, s + 2], linear(s), PATTERN));
    bands.push(band(`full_${s + 3}`, [s + 3, s + 3], full(s + 3), PATTERN));
  }
  return {
    schema_version: 1,
    model: {
      id: "qwen3-next-80b-a3b",
      hf_id: "Qwen/Qwen3-Next-80B-A3B-Instruct",
      architecture: "Qwen3NextForCausalLM",
      family: "hybrid",
    },
    device: "A2",
    symbols: {
      D: sym(256, "full-attention head dim"),
      Dk: sym(128, "linear-attention key head dim"),
      Dv: sym(128, "linear-attention value head dim"),
      E: sym(512, "routed experts"),
      H: sym(2048, "hidden size"),
      Hkv: sym(2, "full-attention key/value heads"),
      Hq: sym(16, "full-attention query heads"),
      I_moe: sym(512, "expert intermediate size"),
      I_sh: sym(512, "shared expert intermediate size"),
      K: sym(4, "linear-attention conv kernel"),
      L: sym(48, "decoder layers"),
      Nk: sym(16, "linear-attention key heads"),
      Nv: sym(32, "linear-attention value heads"),
      T: tSym,
      V: sym(151936, "vocabulary size"),
      topk: sym(10, "experts per token"),
      tp: tpSym,
    },
    input: t("input_ids", ["T"], "int64"),
    output: t("logits", ["T", "V"]),
    stages: [
      embedStage(151936, 2048),
      { id: "layers", kind: "stack", label: "Decoder layers", bands },
      { ...finalNorm("GemmaRMSNorm", SRC.gemmaRms) },
      lmHead(151936, 2048),
    ],
    notes: [
      {
        severity: "info",
        target: "layers",
        text: "Hybrid layout from layer_types: every 4th layer is full (gated) attention, the rest Gated DeltaNet (full_attention_interval = 4).",
        triage_kind: null,
      },
      {
        severity: "warn",
        target: "layers",
        text: "QwenGatedDeltaNetAttention: vllm-ascend patches the Gated DeltaNet chunk kernels (triton fla ops); no reviewed annotation yet (36 layers).",
        triage_kind: "unannotated_override",
      },
      {
        severity: "info",
        target: "layers",
        text: "SharedFusedMoE → AscendSharedFusedMoE (annotated).",
        triage_kind: "conditional_override",
      },
    ],
  };
}

// ---------------------------------------------------------------------- DeepSeek-V4-Flash

/** config.compress_ratios[0..42] (entry 43 is the MTP layer). */
const V4_RATIOS = [0, 0, ...Array.from({ length: 41 }, (_, i) => (i % 2 === 0 ? 4 : 128))];
const V4_HASH_LAYERS = 3;

function deepseek_v4_flash() {
  const F32 = "float32";
  // v1.3 §3: mHC ops are generic components (class + source kept), role "other".
  const hcSub = (id, L, post) => {
    const cls = post ? "MHCPostOp" : "MHCPreOp";
    const source = post ? SRC.mhcPost : SRC.mhcPre;
    const which = id.endsWith("attn") ? "attn" : "ffn";
    return {
      id,
      role: "other",
      component: { kind: "generic", class_name: qualify(cls, source), children: [] },
      io: post
        ? {
            ...io(
              [hidden(), t("residual_streams", ["T", "HC", "H"])],
              [t("residual_streams", ["T", "HC", "H"])],
            ),
            internals: [],
          }
        : {
            ...io([t("residual_streams", ["T", "HC", "H"])], [hidden()]),
            internals: [
              t("post_mix", ["T", "HC"], F32),
              t("res_mix", ["T", "HC", "HC"], F32),
            ],
          },
      module: mod(
        `model.layers.${L}.${post ? "mhc_post" : "mhc_pre"}_${which}`,
        cls,
        source,
        post
          ? []
          : [
              p(`hc_${which}_fn`, ["(2+HC)*HC", "HC*H"], F32),
              p(`hc_${which}_base`, ["(2+HC)*HC"], F32),
              p(`hc_${which}_scale`, ["3"], F32),
            ],
        "generic",
      ),
    };
  };
  const attn = (L) => {
    const ratio = V4_RATIOS[L];
    const indexer = ratio === 4;
    const params = [
      p("fused_wqa_wkv.weight", ["Rq+D", "H"]),
      p("q_norm.weight", ["Rq"]),
      p("wq_b.weight", ["Hq*D/tp", "Rq"]),
      p("kv_norm.weight", ["D"]),
      p("wo_a.weight", ["G*Ro/tp", "Hq*D/G"]),
      p("wo_b.weight", ["H", "G*Ro/tp"]),
      p("attn_sink", ["Hq/tp"], F32),
    ];
    if (ratio > 1) params.push(p("compressor.fused_wkv_wgate.weight", ["2*D", "H"]));
    if (indexer)
      params.push(
        p("indexer.wq_b.weight", ["Ni*Di", "Rq"]),
        p("indexer.wk.weight", ["Di", "H"]),
        p("indexer.weights_proj.weight", ["Ni", "H"]),
      );
    const internals = [
      t("q_latent", ["T", "Rq"]),
      t("q", ["T", "Hq*D/tp"]),
      t("k", ["T", "D"]),
      t("k_rope", ["T", "Dr"]),
    ];
    if (ratio > 1) internals.push(t("kv_compressed", [`cdiv(T, ${ratio})`, "D"]));
    if (indexer) internals.push(t("index_scores", ["T", "cdiv(T, 4)"], F32));
    internals.push(t("o", ["T", "Hq*D/tp"]));
    return {
      id: "attn",
      role: "mixer",
      component: {
        kind: "attention",
        // v1.3 §5: head-sharing variant; locality/sparsity in extras.
        variant: "mqa",
        heads_q: 64,
        heads_kv: 1,
        head_dim: 512,
        rope: {
          kind: "yarn",
          theta: 10000,
          scaling: { factor: 16, original_max_position_embeddings: 65536 },
        },
        qk_norm: false,
        extras: {
          q_lora_rank: 1024,
          o_lora_rank: 1024,
          o_groups: 8,
          qk_rope_head_dim: 64,
          ...(ratio > 1 ? { compress_ratio: ratio } : { sliding_window: 128 }),
          ...(indexer
            ? { indexer: true, index_topk: 512, index_n_heads: 64, index_head_dim: 128 }
            : {}),
        },
      },
      io: { ...io([t("positions", ["T"], "int64"), hidden()], [hidden()]), internals },
      module: mod(`model.layers.${L}.attn`, "DeepseekV4Attention", SRC.v4Attn, params),
    };
  };
  const moe = (L) => {
    const hash = L < V4_HASH_LAYERS;
    return {
      id: "ffn",
      role: "ffn",
      component: {
        kind: "moe",
        experts: 256,
        top_k: 6,
        shared_experts: 1,
        router: { scoring: hash ? "hash" : "sqrtsoftplus", norm_topk: true, groups: null },
        expert: { variant: "swiglu", intermediate: 2048 },
        shared: { variant: "swiglu", intermediate: 2048 },
      },
      io: {
        ...io([hidden()], [hidden()]),
        internals: [
          ...(hash ? [] : [t("router_logits", ["T", "E"], F32)]),
          t("topk_ids", ["T", "topk"], "int32"),
          t("topk_weights", ["T", "topk"], F32),
          t("expert_input", ["T*topk", "H"]),
          t("expert_gate_up", ["T*topk", "2*I_moe/tp"]),
          t("expert_output", ["T*topk", "H"]),
          t("shared_output", ["T", "H"]),
        ],
      },
      module: mod(
        `model.layers.${L}.ffn`,
        "DeepseekV4MoE",
        SRC.v4Moe,
        [
          ...(hash
            ? [p("gate.tid2eid", ["V", "topk"], "int32")]
            : [p("gate.weight", ["E", "H"]), p("gate.e_score_correction_bias", ["E"], F32)]),
          p("experts.w13_weight", ["E", "2*I_moe/tp", "H"], "fp4"),
          p("experts.w2_weight", ["E", "H", "I_moe/tp"], "fp4"),
          p("shared_experts.gate_up_proj.weight", ["2*I_moe/tp", "H"]),
          p("shared_experts.down_proj.weight", ["H", "I_moe/tp"]),
        ],
        "ascend_override",
        { baseline_class: "FusedMoE", ascend_class: "AscendFusedMoE" },
      ),
    };
  };
  const block = (L) => ({
    kind: "decoder",
    residual: "hyper_connection",
    sublayers: [
      hcSub("mhc_pre_attn", L, false),
      normSub("attn_norm", `model.layers.${L}.attn_norm`),
      attn(L),
      hcSub("mhc_post_attn", L, true),
      hcSub("mhc_pre_ffn", L, false),
      normSub("ffn_norm", `model.layers.${L}.ffn_norm`),
      moe(L),
      hcSub("mhc_post_ffn", L, true),
    ],
  });
  const PATTERN = "1× MQA ÷128 + 1× MQA ÷4 + indexer";
  const bands = [
    band("swa_hash", [0, 1], block(0)),
    band("c4_hash", [2, 2], block(2)),
  ];
  for (let L = 3; L <= 42; L++)
    bands.push(band(V4_RATIOS[L] === 128 ? `c128_${L}` : `c4_${L}`, [L, L], block(L), PATTERN));
  return {
    schema_version: 1,
    model: {
      id: "deepseek-v4-flash",
      hf_id: "deepseek-ai/DeepSeek-V4-Flash",
      architecture: "DeepseekV4ForCausalLM",
      family: "mla",
    },
    device: "A2",
    symbols: {
      D: sym(512, "head dim (shared single KV head)"),
      Dr: sym(64, "qk_rope_head_dim"),
      Di: sym(128, "index_head_dim"),
      E: sym(256, "routed experts"),
      G: sym(8, "o_groups"),
      H: sym(4096, "hidden size"),
      Hq: sym(64, "attention heads"),
      I_moe: sym(2048, "expert intermediate size"),
      K_idx: sym(512, "index_topk"),
      L: sym(43, "decoder layers"),
      Ni: sym(64, "index_n_heads"),
      Ro: sym(1024, "o_lora_rank"),
      Rq: sym(1024, "q_lora_rank"),
      T: tSym,
      V: sym(129280, "vocabulary size"),
      HC: sym(4, "hyper-connection streams (hc_mult)"),
      topk: sym(6, "experts per token"),
      tp: tpSym,
    },
    input: t("input_ids", ["T"], "int64"),
    output: t("logits", ["T", "V"]),
    stages: [
      embedStage(129280, 4096),
      { id: "layers", kind: "stack", label: "Decoder layers", bands },
      // v1.3 §2: array order is dataflow order — the HC head merges the streams before the norm.
      {
        id: "hc_head_op",
        kind: "other",
        placement: "spine",
        label: "HC head (merge streams)",
        component: { kind: "generic", class_name: qualify("HCHeadOp", SRC.hcHead), children: [] },
        module: mod(
          "model.hc_head_op",
          "HCHeadOp",
          SRC.hcHead,
          [
            p("hc_head_fn", ["HC", "HC*H"], "float32"),
            p("hc_head_base", ["HC"], "float32"),
            p("hc_head_scale", ["1"], "float32"),
          ],
          "generic",
        ),
        io: {
          ...io([t("residual_streams", ["T", "HC", "H"])], [hidden()]),
          internals: [t("head_mix", ["T", "HC"], "float32")],
        },
      },
      {
        id: "final_norm",
        kind: "norm",
        label: "Final norm",
        component: { kind: "norm", variant: "rmsnorm", eps: 1e-6 },
        module: mod("model.norm", "RMSNorm", SRC.rmsnorm, [p("weight", ["H"])]),
        io: { ...io([hidden()], [hidden()]), internals: [] },
      },
      lmHead(129280, 4096),
      {
        id: "mtp",
        kind: "other",
        placement: "aside",
        label: "Multi-token prediction",
        component: {
          kind: "generic",
          class_name: qualify("DeepSeekV4MultiTokenPredictorLayer", SRC.v4Mtp),
          children: [
            { name: "enorm", class_name: "RMSNorm" },
            { name: "hnorm", class_name: "RMSNorm" },
            { name: "eh_proj", class_name: "ReplicatedLinear" },
            { name: "mtp_block", class_name: "DeepseekV4DecoderLayer" },
          ],
        },
        module: mod(
          "model.layers.43",
          "DeepSeekV4MultiTokenPredictorLayer",
          SRC.v4Mtp,
          [p("eh_proj.weight", ["H", "2*H"])],
          "generic",
        ),
        io: io([t("input_ids", ["T"], "int64"), hidden("previous_hidden_states")], [hidden()]),
      },
    ],
    notes: [
      {
        severity: "info",
        target: null,
        text: "Hyper-connections (mHC): the residual is 4 parallel streams; H_pre mixes them into the sublayer input and H_post / H_res (Sinkhorn-normalised) write back. The HC head merges the streams before the final norm.",
        triage_kind: null,
      },
      {
        severity: "info",
        target: "layers",
        text: "compress_ratios: layers 0–1 sliding window only; then KV ÷4 (with lightning indexer, top-512) and KV ÷128 alternate. Layers 0–2 route experts by token hash (num_hash_layers = 3).",
        triage_kind: null,
      },
      {
        severity: "warn",
        target: "hc_head_op",
        text: "No architecture rule describes HCHeadOp; shown as a generic component.",
        triage_kind: "unknown_layer_class",
      },
      // Targeted mHC warnings (v1.3 §3: badges like any sublayer), one per drawn period band.
      ...["c128_3", "c4_4"].map((b) => ({
        severity: "warn",
        target: `${b}/mhc_pre_attn`,
        text: "MHCPreOp is drawn from observed module boundaries (no lowering rule).",
        triage_kind: "unknown_layer_class",
      })),
      {
        severity: "warn",
        target: "mtp",
        text: "DeepSeekV4MultiTokenPredictorLayer has no lowering rule; drawn as a generic component.",
        triage_kind: "unknown_layer_class",
      },
      {
        severity: "info",
        target: "layers",
        text: "FusedMoE → AscendFusedMoE (annotated); fp4 expert weights are dequantised on Ascend.",
        triage_kind: "conditional_override",
      },
    ],
  };
}

// --------------------------------------------------------------------------- gpt-oss-120b

function gpt_oss_120b() {
  const F32 = "float32";
  const attn = (L, sliding) => ({
    id: "attn",
    role: "mixer",
    component: {
      kind: "attention",
      variant: "gqa",
      heads_q: 64,
      heads_kv: 8,
      head_dim: 64,
      rope: {
        kind: "yarn",
        theta: 150000,
        scaling: { factor: 32, original_max_position_embeddings: 4096 },
      },
      qk_norm: false,
      extras: { ...(sliding ? { sliding_window: 128 } : {}), attention_sinks: true },
    },
    io: {
      ...io([t("positions", ["T"], "int64"), hidden()], [hidden()]),
      internals: [
        t("q", ["T", "Hq*D/tp"]),
        t("k", ["T", "Hkv*D/tp"]),
        t("v", ["T", "Hkv*D/tp"]),
        t("o", ["T", "Hq*D/tp"]),
      ],
    },
    module: mod(`model.layers.${L}.attn`, "OAIAttention", SRC.ossAttn, [
      p("qkv_proj.weight", ["(Hq+2*Hkv)*D/tp", "H"]),
      p("qkv_proj.bias", ["(Hq+2*Hkv)*D/tp"]),
      p("o_proj.weight", ["H", "Hq*D/tp"]),
      p("o_proj.bias", ["H"]),
      p("sinks", ["Hq/tp"], F32),
    ]),
  });
  const mlp = (L) => ({
    id: "mlp",
    role: "ffn",
    component: {
      kind: "moe",
      experts: 128,
      top_k: 4,
      shared_experts: 0,
      router: { scoring: "softmax", norm_topk: true, groups: null },
      expert: { variant: "swiglu", intermediate: 2880 },
      shared: null,
    },
    io: {
      ...io([hidden()], [hidden()]),
      internals: [
        t("router_logits", ["T", "E"], F32),
        t("topk_ids", ["T", "topk"], "int32"),
        t("topk_weights", ["T", "topk"], F32),
        t("expert_input", ["T*topk", "H"]),
        t("expert_gate_up", ["T*topk", "2*I/tp"]),
        t("expert_output", ["T*topk", "H"]),
      ],
    },
    module: mod(`model.layers.${L}.mlp`, "MLPBlock", SRC.ossMlp, [
      p("router.weight", ["E", "H"]),
      p("router.bias", ["E"]),
      p("experts.w13_weight", ["E", "2*I/tp", "H"], "mxfp4"),
      p("experts.w2_weight", ["E", "H", "I/tp"], "mxfp4"),
    ]),
  });
  const block = (L) => ({
    kind: "decoder",
    residual: "pre_norm",
    sublayers: [
      normSub("input_layernorm", `model.layers.${L}.input_layernorm`),
      attn(L, L % 2 === 0),
      addSub("residual_1"),
      normSub("post_attention_layernorm", `model.layers.${L}.post_attention_layernorm`),
      mlp(L),
      addSub("residual_2"),
    ],
  });
  const PATTERN = "1× sliding + 1× full";
  const bands = Array.from({ length: 36 }, (_, L) =>
    band(L % 2 === 0 ? `sliding_${L}` : `full_${L}`, [L, L], block(L), PATTERN),
  );
  return {
    schema_version: 1,
    model: {
      id: "gpt-oss-120b",
      hf_id: "openai/gpt-oss-120b",
      architecture: "GptOssForCausalLM",
      family: "moe",
    },
    device: "A2",
    symbols: {
      D: sym(64, "head dim"),
      E: sym(128, "local experts"),
      H: sym(2880, "hidden size"),
      Hkv: sym(8, "key/value heads"),
      Hq: sym(64, "query heads"),
      I: sym(2880, "expert intermediate size"),
      L: sym(36, "decoder layers"),
      T: tSym,
      V: sym(201088, "vocabulary size"),
      topk: sym(4, "experts per token"),
      tp: tpSym,
    },
    input: t("input_ids", ["T"], "int64"),
    output: t("logits", ["T", "V"]),
    stages: [
      embedStage(201088, 2880),
      { id: "layers", kind: "stack", label: "Decoder layers", bands },
      finalNorm(),
      lmHead(201088, 2880),
    ],
    notes: [
      {
        severity: "info",
        target: "layers",
        text: "layer_types alternate sliding_attention (window 128) and full_attention; every attention layer has learned per-head attention sinks.",
        triage_kind: null,
      },
      {
        severity: "info",
        target: null,
        text: "Experts are mxfp4-quantised (quantization_config); attention, router, embeddings and lm_head stay bf16.",
        triage_kind: null,
      },
    ],
  };
}

// ----------------------------------------------------------- internals for the v1.1 fixtures

/** Add v1.2 `io.internals` by component kind (explicit internals are kept). */
function withInternals(arch) {
  const has = (k) => k in arch.symbols;
  const F32 = "float32";
  const forComponent = (c) => {
    if (!c) return [];
    if (c.kind === "attention") {
      if (c.variant === "mla")
        return [
          t("q_latent", ["T", "Rq"]),
          t("kv_latent", ["T", "Rkv"]),
          t("k_rope", ["T", "Dr"]),
          t("q", ["T", "Hq*(Dn+Dr)/tp"]),
          t("o", ["T", "Hq*Dv/tp"]),
        ];
      if (c.variant === "linear")
        return [
          t("qkvz", ["T", "(2*Nk*Dk+2*Nv*Dv)/tp"]),
          t("ba", ["T", "2*Nv/tp"]),
          t("conv_out", ["T", "(2*Nk*Dk+Nv*Dv)/tp"]),
          t("recurrent_state", ["Nv/tp", "Dk", "Dv"], F32),
        ];
      return [
        t("q", ["T", c.extras.output_gate ? "2*Hq*D/tp" : "Hq*D/tp"]),
        t("k", ["T", "Hkv*D/tp"]),
        t("v", ["T", "Hkv*D/tp"]),
        t("o", ["T", "Hq*D/tp"]),
      ];
    }
    if (c.kind === "moe")
      return [
        t("router_logits", ["T", "E"], F32),
        t("topk_ids", ["T", "topk"], "int32"),
        t("topk_weights", ["T", "topk"], F32),
        t("expert_input", ["T*topk", "H"]),
        t("expert_gate_up", ["T*topk", "2*I_moe/tp"]),
        t("expert_output", ["T*topk", "H"]),
        ...(c.shared_experts ? [t("shared_output", ["T", "H"])] : []),
      ];
    if (c.kind === "mlp") return [t("gate_up", ["T", "2*I/tp"]), t("act", ["T", "I/tp"])];
    return [];
  };
  const fix = (x) => {
    if (x.io && !x.io.internals) x.io.internals = forComponent(x.component ?? null);
    x.internals = undefined;
  };
  for (const s of arch.stages) {
    if (s.kind === "stack") for (const b of s.bands) for (const sub of b.block.sublayers) fix(sub);
    else fix(s);
  }
  if (!has("T")) throw new Error(`${arch.model.id}: missing T`);
  return arch;
}

for (const arch of [
  qwen3_8b(),
  qwen3_30b(),
  deepseek_v3(),
  qwen3_next(),
  deepseek_v4_flash(),
  gpt_oss_120b(),
].map(withInternals)) {
  const out = join(here, arch.model.id, "arch.json");
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, `${JSON.stringify(arch, null, 2)}\n`);
  console.log(`wrote ${out}`);
}
