import type * as React from "react";
import { cn } from "../cn";
import { VariantBadge } from "./badges";
import { Chip, type ComponentNodeProps, FigureNode } from "./node";

const fmt = (n: number) => n.toLocaleString("en-US");

// ------------------------------------------------------------------------------------- Norm

export type NormVariant = "rmsnorm" | "layernorm" | "other";

const NORM_LABEL: Record<NormVariant, string> = {
  rmsnorm: "RMSNorm",
  layernorm: "LayerNorm",
  other: "Norm",
};

export interface NormProps extends ComponentNodeProps {
  variant?: NormVariant;
  /** Overrides the variant name (e.g. a class name for `other`). */
  name?: string | undefined;
}

export function Norm({ variant = "rmsnorm", name, ...props }: NormProps) {
  return (
    <FigureNode
      kind="norm"
      size="sm"
      title={name ?? NORM_LABEL[variant]}
      data-component="norm"
      {...props}
    />
  );
}

// ------------------------------------------------------------------------------- Embedding

export interface EmbeddingProps extends Omit<ComponentNodeProps, "hidden" | "vocab"> {
  vocab?: number | undefined;
  /** Hidden size (shadows the HTML `hidden` attribute, which figure nodes never use). */
  hidden?: number | undefined;
  tied?: boolean | undefined;
  name?: string | undefined;
}

export function Embedding({ vocab, hidden, tied, name, ...props }: EmbeddingProps) {
  return (
    <FigureNode
      kind="embedding"
      title={name ?? "Token embedding"}
      badge={tied ? <VariantBadge>tied</VariantBadge> : undefined}
      subtitle={
        vocab !== undefined && hidden !== undefined
          ? `vocab ${fmt(vocab)} → hidden ${fmt(hidden)}`
          : undefined
      }
      data-component="embedding"
      {...props}
    />
  );
}

export function LMHead({ vocab, hidden, tied, name, ...props }: EmbeddingProps) {
  return (
    <FigureNode
      kind="head"
      title={name ?? "LM head"}
      badge={tied ? <VariantBadge>tied to embedding</VariantBadge> : undefined}
      subtitle={
        vocab !== undefined && hidden !== undefined
          ? `hidden ${fmt(hidden)} → vocab ${fmt(vocab)}`
          : undefined
      }
      data-component="lm_head"
      {...props}
    />
  );
}

// --------------------------------------------------------------------------------- Generic

export interface GenericProps extends ComponentNodeProps {
  /** Python class name of the module. */
  moduleClass: string;
  /** Direct children `{name, class_name}`; the first few are drawn as chips. */
  members?: readonly { name: string; class_name: string }[] | undefined;
  /** Caption under the class name. */
  caption?: React.ReactNode;
  maxChildren?: number;
}

/** Fallback for classes without a lowering rule: class name, children, dashed outline. */
export function Generic({
  moduleClass,
  members: kids = [],
  caption,
  maxChildren = 6,
  ...props
}: GenericProps) {
  const shown = kids.slice(0, maxChildren);
  const more = kids.length - shown.length;
  return (
    <FigureNode
      kind="generic"
      title={<span className="font-mono text-[11px]">{moduleClass}</span>}
      subtitle={caption ?? "no lowering rule · generic"}
      data-component="generic"
      {...props}
    >
      {shown.length ? (
        <span className="mt-0.5 flex max-w-full flex-wrap justify-center gap-1">
          {shown.map((c) => (
            <Chip key={c.name} tone="generic" title={c.class_name}>
              {c.name}
            </Chip>
          ))}
          {more > 0 ? <Chip tone="generic">+{more}</Chip> : null}
        </span>
      ) : null}
    </FigureNode>
  );
}

// ---------------------------------------------------------------------------- ResidualAdd

export interface ResidualAddProps extends React.HTMLAttributes<HTMLElement> {
  onSelect?: (() => void) | undefined;
  selected?: boolean | undefined;
}

/** The ⊕ of a residual connection. */
export function ResidualAdd({ onSelect, selected, className, ...props }: ResidualAddProps) {
  const glyph = (
    <svg viewBox="0 0 20 20" className="size-5" aria-hidden="true">
      <circle
        cx="10"
        cy="10"
        r="9"
        className="fill-background stroke-arch-wire"
        strokeWidth="1.25"
      />
      <path d="M10 5.5 V14.5 M5.5 10 H14.5" className="stroke-foreground/80" strokeWidth="1.4" />
    </svg>
  );
  const cls = cn(
    "relative z-[1] flex size-5 shrink-0 items-center justify-center rounded-full outline-none",
    selected && "ring-2 ring-focus ring-offset-1 ring-offset-background",
    onSelect && "cursor-pointer focus-visible:ring-2 focus-visible:ring-ring/70",
    className,
  );
  if (onSelect)
    return (
      <button
        type="button"
        aria-label="Residual add"
        aria-pressed={!!selected}
        data-slot="residual-add"
        className={cls}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        {...(props as React.ButtonHTMLAttributes<HTMLButtonElement>)}
      >
        {glyph}
      </button>
    );
  return (
    <div role="img" aria-label="Residual add" data-slot="residual-add" className={cls} {...props}>
      {glyph}
    </div>
  );
}

// ------------------------------------------------------------------------------- Attention

export type AttentionVariant =
  | "mha"
  | "gqa"
  | "mqa"
  | "mla"
  | "linear"
  | "sliding_window"
  | "sparse"
  | "other";

const ATTN_BADGE: Record<AttentionVariant, string> = {
  mha: "MHA",
  gqa: "GQA",
  mqa: "MQA",
  mla: "MLA",
  linear: "Linear",
  sliding_window: "Sliding",
  sparse: "Sparse",
  other: "Attn",
};

export const attentionBadge = (v: AttentionVariant): string => ATTN_BADGE[v];

/** MLA low-rank compression path (DeepSeek-V2/V3). */
export interface LatentPath {
  /** Query compression rank; `null`/undefined when queries are not compressed. */
  qRank?: number | null | undefined;
  kvRank: number;
  ropeDim?: number | undefined;
}

export interface AttentionProps extends ComponentNodeProps {
  variant: AttentionVariant;
  name?: string | undefined;
  headsQ?: number | undefined;
  headsKv?: number | undefined;
  headDim?: number | undefined;
  /** Short positional-encoding tag, e.g. `RoPE θ 1e6`, `YaRN ×40`. */
  rope?: string | null | undefined;
  qkNorm?: boolean | undefined;
  /** Sigmoid output gate (gated attention). */
  gated?: boolean | undefined;
  latent?: LatentPath | undefined;
  /** Extra one-line description (e.g. `Gated DeltaNet · conv 4`). */
  detail?: React.ReactNode;
}

function Lane({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <span className="contents">
      <span className="text-right font-mono text-[9.5px] text-muted-foreground">{label}</span>
      <span className="flex items-center gap-1">{children}</span>
    </span>
  );
}

const To = () => (
  <span aria-hidden="true" className="text-[10px] text-arch-wire">
    →
  </span>
);

/** The MLA latent compression diagram: down-projection → cached latent → up-projection. */
export function LatentPathDiagram({ qRank, kvRank, ropeDim }: LatentPath) {
  return (
    <span
      data-slot="latent-path"
      className="mt-1 grid grid-cols-[auto_auto] items-center justify-center gap-x-1.5 gap-y-1"
    >
      {qRank ? (
        <Lane label="q">
          <Chip tone="attention">W_DQ</Chip>
          <To />
          <Chip>c_Q {fmt(qRank)}</Chip>
          <To />
          <Chip tone="attention">W_UQ</Chip>
        </Lane>
      ) : null}
      <Lane label="kv">
        <Chip tone="attention">W_DKV</Chip>
        <To />
        <Chip tone="accent" title="cached in the KV cache">
          c_KV {fmt(kvRank)}
        </Chip>
        <To />
        <Chip tone="attention">W_UK·W_UV</Chip>
      </Lane>
      {ropeDim ? (
        <Lane label="rope">
          <Chip tone="attention">W_KR</Chip>
          <To />
          <Chip tone="accent" title="cached in the KV cache">
            k_R {fmt(ropeDim)}
          </Chip>
        </Lane>
      ) : null}
    </span>
  );
}

export function Attention({
  variant,
  name,
  headsQ,
  headsKv,
  headDim,
  rope,
  qkNorm,
  gated,
  latent,
  detail,
  ...props
}: AttentionProps) {
  const heads =
    headsQ !== undefined
      ? [
          `${headsQ} q`,
          headsKv !== undefined && headsKv !== headsQ ? `${headsKv} kv` : undefined,
          headDim !== undefined ? `d ${headDim}` : undefined,
        ]
          .filter(Boolean)
          .join(" · ")
      : undefined;
  const tags = [rope, qkNorm ? "QK-Norm" : undefined, gated ? "out gate" : undefined].filter(
    Boolean,
  ) as string[];
  return (
    <FigureNode
      kind="attention"
      title={name ?? (variant === "linear" ? "Linear attention" : "Attention")}
      badge={<VariantBadge tone="attention">{ATTN_BADGE[variant]}</VariantBadge>}
      subtitle={heads ? `heads ${heads}` : undefined}
      data-component="attention"
      data-variant={variant}
      {...props}
    >
      {detail ? <span className="text-[10.5px] text-muted-foreground">{detail}</span> : null}
      {variant === "mla" && latent ? <LatentPathDiagram {...latent} /> : null}
      {tags.length ? (
        <span className="mt-0.5 flex flex-wrap justify-center gap-1">
          {tags.map((t) => (
            <Chip key={t}>{t}</Chip>
          ))}
        </span>
      ) : null}
    </FigureNode>
  );
}

// ------------------------------------------------------------------------------------- MLP

export type MlpVariant = "swiglu" | "geglu" | "gelu" | "other";

const MLP_BADGE: Record<MlpVariant, string> = {
  swiglu: "SwiGLU",
  geglu: "GeGLU",
  gelu: "GELU",
  other: "MLP",
};

export const mlpBadge = (v: MlpVariant): string => MLP_BADGE[v];

export interface MLPProps extends ComponentNodeProps {
  variant?: MlpVariant;
  intermediate?: number | undefined;
  name?: string | undefined;
  /** Draw the projection chips (gate ⊙ up → down, or up → down). Default true. */
  showProjections?: boolean;
}

/** Gated MLP: gate & up projections, activation product, down projection. */
export function MLP({
  variant = "swiglu",
  intermediate,
  name,
  showProjections,
  ...props
}: MLPProps) {
  const gated = variant === "swiglu" || variant === "geglu";
  const act = variant === "geglu" ? "GELU" : variant === "swiglu" ? "SiLU" : "GELU";
  return (
    <FigureNode
      kind="mlp"
      title={name ?? "MLP"}
      badge={<VariantBadge tone="ffn">{MLP_BADGE[variant]}</VariantBadge>}
      subtitle={intermediate !== undefined ? `intermediate ${fmt(intermediate)}` : undefined}
      data-component="mlp"
      data-variant={variant}
      {...props}
    >
      {(showProjections ?? true) ? (
        <span data-slot="mlp-projections" className="mt-0.5 flex items-center gap-1">
          {gated ? (
            <>
              <Chip tone="mlp">gate</Chip>
              <span className="font-mono text-[9.5px] text-muted-foreground">{act}</span>
              <span aria-hidden="true" className="text-[11px] text-foreground/70">
                ⊙
              </span>
              <Chip tone="mlp">up</Chip>
            </>
          ) : (
            <>
              <Chip tone="mlp">up</Chip>
              <span className="font-mono text-[9.5px] text-muted-foreground">{act}</span>
            </>
          )}
          <To />
          <Chip tone="mlp">down</Chip>
        </span>
      ) : null}
    </FigureNode>
  );
}

// ------------------------------------------------------------------------------------- MoE

export interface MoEProps extends ComponentNodeProps {
  experts: number;
  topK: number;
  sharedExperts?: number | undefined;
  /** Router scoring, e.g. `softmax`, `sigmoid`. */
  scoring?: string | undefined;
  /** Group-limited routing `{groups, topkGroups}` (DeepSeek-V3 `n_group` / `topk_group`). */
  groups?: { groups: number; topkGroups?: number | undefined } | undefined;
  normTopk?: boolean | undefined;
  expert?: { variant: MlpVariant; intermediate?: number | undefined } | undefined;
  shared?: { variant: MlpVariant; intermediate?: number | undefined } | null | undefined;
  name?: string | undefined;
  /** How many routed expert boxes to draw before the ellipsis (plus the last one). */
  visibleExperts?: number;
}

/**
 * Mixture of experts as in the Switch / Mixtral / DeepSeekMoE figures: router, a row of expert
 * boxes with the selected ones highlighted, optional shared expert(s) beside them.
 */
export function MoE({
  experts,
  topK,
  sharedExperts = 0,
  scoring,
  groups,
  normTopk,
  expert,
  shared,
  name,
  visibleExperts = 5,
  ...props
}: MoEProps) {
  const head = Math.min(visibleExperts, experts);
  const ids = Array.from({ length: head }, (_, i) => i + 1);
  const tail = experts > head ? experts : undefined;
  // Illustrative routing: highlight every other drawn box, up to top-k.
  const active = new Set(ids.filter((_, i) => i % 2 === 1).slice(0, topK));
  const router = [
    scoring,
    `top-${topK}`,
    groups ? `${groups.topkGroups ?? "?"}/${groups.groups} groups` : undefined,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <FigureNode
      kind="moe"
      title={name ?? "MoE"}
      badge={<VariantBadge tone="ffn">{`${experts}E · top-${topK}`}</VariantBadge>}
      data-component="moe"
      {...props}
    >
      <span
        data-slot="moe-router"
        className="mt-0.5 inline-flex min-h-5 max-w-full flex-wrap items-center justify-center gap-x-1 rounded-[3px] border border-arch-router-border bg-arch-router px-1.5 py-0.5 text-[10px] leading-tight"
        title={normTopk ? "top-k weights renormalised" : undefined}
      >
        <span className="font-semibold text-arch-router-ink">Router</span>
        <span className="font-mono text-[9.5px] text-muted-foreground">{router}</span>
      </span>
      <span aria-hidden="true" className="-my-0.5 text-[10px] leading-none text-arch-wire">
        ↓
      </span>
      <span data-slot="moe-experts" className="flex items-center gap-1">
        {sharedExperts > 0 ? (
          <>
            <Chip tone="shared" title={`${sharedExperts} shared expert(s), always active`}>
              S{sharedExperts > 1 ? `×${sharedExperts}` : ""}
            </Chip>
            <span aria-hidden="true" className="h-4 w-px bg-border" />
          </>
        ) : null}
        {ids.map((i) => (
          <Chip
            key={i}
            tone={active.has(i) ? "accent" : "mlp"}
            className="min-w-[18px]"
            data-active={active.has(i) || undefined}
          >
            {i}
          </Chip>
        ))}
        {tail ? (
          <>
            <span aria-hidden="true" className="text-[10px] text-muted-foreground">
              …
            </span>
            <Chip tone="mlp">{tail}</Chip>
          </>
        ) : null}
      </span>
      <span className="text-[10px] text-muted-foreground">
        {[
          `${fmt(experts)} routed`,
          sharedExperts ? `${sharedExperts} shared` : undefined,
          expert
            ? `${MLP_BADGE[expert.variant]}${expert.intermediate ? ` · ${fmt(expert.intermediate)}` : ""}`
            : undefined,
        ]
          .filter(Boolean)
          .join(" · ")}
        {shared &&
        sharedExperts &&
        shared.intermediate &&
        shared.intermediate !== expert?.intermediate
          ? ` · shared ${fmt(shared.intermediate)}`
          : ""}
      </span>
    </FigureNode>
  );
}
