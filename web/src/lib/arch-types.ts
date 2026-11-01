/**
 * TEMPORARY mirror of the `arch.json` contract (docs/arch-contract.md, schema_version 1).
 *
 * TO BE REPLACED by the generated `Arch*` types from `@inferatlas/contracts` once S1 lands:
 * delete this file and re-export from the contract, e.g.
 *   export type { Arch, ArchStage, ArchBand, … } from "@inferatlas/contracts";
 * Every UI module imports these names from here only, so the swap is one file.
 *
 * Interpretations the contract leaves open (recorded in docs/requests/U1.md):
 * - `Sublayer.component` may be `null` for `residual_add` sublayers (no Component kind exists).
 * - A periodic pattern is a run of consecutive bands sharing the same `pattern` label.
 * - `Note.target` names a stage id, a band id, or `<band id>/<sublayer id>`.
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
  outputs: ArchTensor[];
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
}

export interface ArchAttentionComponent {
  kind: "attention";
  variant: "mha" | "gqa" | "mqa" | "mla" | "linear" | "sliding_window" | "sparse" | "other";
  heads_q: number;
  heads_kv: number;
  head_dim: number;
  rope: ArchRope | null;
  qk_norm: boolean;
  extras: Record<string, unknown>;
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

export type ArchResidual = "pre_norm" | "post_norm" | "sandwich" | "unknown";

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
