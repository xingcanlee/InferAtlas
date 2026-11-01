/**
 * TEMPORARY mirror of the `arch.json` contract (docs/arch-contract.md, schema_version 1 with
 * the "Contract decisions v1.1" and "v1.2").
 *
 * TO BE REPLACED by the generated `Arch*` types from `@inferatlas/contracts` once S1 lands:
 * delete this file and re-export from the contract, e.g.
 *   export type { Arch, ArchStage, ArchBand, … } from "@inferatlas/contracts";
 * Every UI module imports these names from here only, so the swap is one file.
 *
 * v1.1 decisions mirrored here:
 * - `residual_add` sublayers have `component: null` and `module: null`.
 * - Sublayer ids are band-relative; `module.path` is the band's first layer (UI substitutes).
 * - `Note.target` is `<band id>/<sublayer id>`, a stage id, or `null` (model-level).
 * - Periodic runs share a `pattern` label; the UI also folds by structural signature.
 * - `other` stages carry `placement: "spine" | "aside"`.
 * - `class_name` is fully qualified (figures show the last segment).
 * v1.2: `IO.internals` (tensors produced inside a component), de-duplicated notes, attention
 * extras `compress_ratio` / `indexer`, residual `"hyper_connection"` with `mhc_*` sublayers
 * (role `"other"`), canonical stage order.
 */

/** Expression string in the expr DSL (docs/expr-dsl.md). */
export type Expr = string;

export interface ArchTensor {
  name: string;
  shape: Expr[];
  dtype: string;
}

export interface ArchIO {
  inputs: ArchTensor[];
  /** Only what leaves the sublayer (usually `[T, H]`). */
  outputs: ArchTensor[];
  /** v1.2: q/k/v/o, MLA latents, router logits, top-k ids/weights, per-expert tensors… */
  internals?: ArchTensor[];
}

export type ArchModuleStatus = "ruled" | "generic" | "ascend_override" | "unannotated_override";

export interface ArchSource {
  file: string;
  line: number;
  url: string;
}

export interface ArchParam {
  name: string;
  shape: Expr[];
  dtype: string;
}

export interface ArchModule {
  path: string;
  class_name: string;
  source: ArchSource | null;
  status: ArchModuleStatus;
  replaced: { baseline_class: string; ascend_class: string } | null;
  params: ArchParam[];
}

export interface ArchNormComponent {
  kind: "norm";
  variant: "rmsnorm" | "layernorm" | "other";
  eps?: number | null;
}

export interface ArchRope {
  kind: string;
  theta?: number | null;
  scaling?: Record<string, unknown> | null;
  partial_rotary_factor?: number | null;
}

/**
 * Optional attention extras (v1.1 §7): MLA `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`,
 * `qk_rope_head_dim`, `v_head_dim`; sparse `index_topk`; sliding `sliding_window`; linear
 * `kind`, `num_k_heads`, `num_v_heads`, `head_k_dim`, `head_v_dim`, `conv_kernel`, `output_gate`;
 * v1.2 compressed attention `compress_ratio` (number) and `indexer` (bool).
 */
export type ArchAttentionExtras = Record<string, unknown>;

export interface ArchAttentionComponent {
  kind: "attention";
  variant: "mha" | "gqa" | "mqa" | "mla" | "linear" | "sliding_window" | "sparse" | "other";
  heads_q: number;
  heads_kv: number;
  head_dim: number;
  rope: ArchRope | null;
  qk_norm: boolean;
  extras: ArchAttentionExtras;
}

export type ArchMlpVariant = "swiglu" | "geglu" | "gelu" | "other";

export interface ArchMlpComponent {
  kind: "mlp";
  variant: ArchMlpVariant;
  intermediate: number;
}

export interface ArchMoeComponent {
  kind: "moe";
  experts: number;
  top_k: number;
  shared_experts: number;
  router: {
    scoring: string;
    norm_topk?: boolean | null;
    groups?: { n_group: number; topk_group: number } | null;
  };
  expert: { variant: ArchMlpVariant; intermediate: number };
  shared: { variant: ArchMlpVariant; intermediate: number } | null;
}

export interface ArchGenericComponent {
  kind: "generic";
  class_name: string;
  children: { name: string; class_name: string }[];
}

export interface ArchEmbeddingComponent {
  kind: "embedding" | "lm_head";
  vocab: number;
  hidden: number;
  tied: boolean;
}

export type ArchComponent =
  | ArchNormComponent
  | ArchAttentionComponent
  | ArchMlpComponent
  | ArchMoeComponent
  | ArchGenericComponent
  | ArchEmbeddingComponent;

export type ArchSublayerRole = "norm" | "mixer" | "ffn" | "residual_add" | "other";

export interface ArchSublayer {
  id: string;
  role: ArchSublayerRole;
  component: ArchComponent | null;
  io: ArchIO;
  module: ArchModule | null;
}

export type ArchResidual = "pre_norm" | "post_norm" | "sandwich" | "hyper_connection" | "unknown";

export interface ArchBlock {
  kind: "decoder";
  residual: ArchResidual;
  sublayers: ArchSublayer[];
}

export interface ArchBand {
  id: string;
  /** Inclusive layer range. */
  layers: [number, number];
  count: number;
  pattern: string | null;
  block: ArchBlock;
}

export type ArchStageKind = "embedding" | "stack" | "norm" | "lm_head" | "other";

export interface ArchModuleStage {
  id: string;
  kind: Exclude<ArchStageKind, "stack">;
  label?: string | null;
  module: ArchModule | null;
  io: ArchIO;
  component?: ArchComponent | null;
  /** `other` stages only: drawn on the main spine or as a side branch (MTP / draft heads). */
  placement?: "spine" | "aside" | null;
}

export interface ArchStackStage {
  id: string;
  kind: "stack";
  label?: string | null;
  bands: ArchBand[];
}

export type ArchStage = ArchModuleStage | ArchStackStage;

export interface ArchNote {
  severity: "info" | "warn";
  target: string | null;
  text: string;
  triage_kind: string | null;
}

export interface ArchSymbol {
  value: number;
  description?: string | null;
}

export interface Arch {
  schema_version: 1;
  model: { id: string; hf_id: string; architecture: string; family: string };
  device: string;
  symbols: Record<string, ArchSymbol>;
  input: ArchTensor;
  output: ArchTensor;
  stages: ArchStage[];
  notes: ArchNote[];
}
