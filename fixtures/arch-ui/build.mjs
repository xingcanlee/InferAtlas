#!/usr/bin/env node
/**
 * Hand-authored arch.json fixtures for the U1 architecture UI (docs/arch-contract.md, v1).
 * NOT generator output: written by hand from the public HF configs (catalog.yaml where present)
 * and vLLM 544f182 class locations, so the UI can be built before the S1 emitter lands.
 *
 *   node fixtures/arch-ui/build.mjs        # rewrites fixtures/arch-ui/<model>/arch.json
 *
 * Shapes are global (tp is a symbol, default 1) expression strings in the expr DSL.
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
      ...bands
        .filter((b) => b.id.startsWith("linear"))
        .map((b) => ({
          severity: "warn",
          target: `${b.id}/linear_attn`,
          text: "vllm-ascend patches the Gated DeltaNet chunk kernels (triton fla ops); no reviewed annotation yet.",
          triage_kind: "unannotated_override",
        })),
    ],
  };
}

for (const arch of [qwen3_8b(), qwen3_30b(), deepseek_v3(), qwen3_next()]) {
  const out = join(here, arch.model.id, "arch.json");
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, `${JSON.stringify(arch, null, 2)}\n`);
  console.log(`wrote ${out}`);
}
