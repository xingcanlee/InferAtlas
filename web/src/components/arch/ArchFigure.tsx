/**
 * arch.json → @inferatlas/arch-ui. The only place that maps contract data onto figure
 * primitives; it is generic over the contract (no per-model branches).
 */
import {
  Attention,
  Band,
  BandCard,
  BandList,
  BlockStack,
  BlockStep,
  DecoderBlock,
  Embedding,
  FigureBranch,
  FigureTerminal,
  FlowArrow,
  Generic,
  LMHead,
  MLP,
  ModelFigure,
  MoE,
  type NodeMarkers,
  Norm,
  PatternGroup,
  ResidualBranch,
  TensorEdge,
} from "@inferatlas/arch-ui";
import type { ReactNode } from "react";
import {
  bandSummary,
  blockSegments,
  layerRange,
  MAX_STACK_COLUMNS,
  noteMarker,
  notesFor,
  represented,
  type StackItem,
  shortClass,
  stackColumns,
  stackItems,
  subTarget,
} from "@/lib/arch";
import type {
  Arch,
  ArchBand,
  ArchComponent,
  ArchModuleStage,
  ArchRope,
  ArchStage,
  ArchSublayer,
  ArchTensor,
} from "@/lib/arch-types";

export interface ArchSelection {
  /** Drawn (representative) band that is expanded in the detail panel. */
  band?: ArchBand | undefined;
  sub?: string | undefined;
  stage?: string | undefined;
}

export interface ArchFigureProps {
  arch: Arch;
  evaluate: (expr: string) => number | undefined;
  selection: ArchSelection;
  onSelectBand: (band: ArchBand) => void;
  onSelectSub: (band: ArchBand, sub: string) => void;
  onSelectStage: (stage: string) => void;
  /** id of the detail panel the band headers control. */
  detailId?: string;
}

const num = (v: unknown): number | undefined => (typeof v === "number" ? v : undefined);

const sci = (n: number) =>
  n >= 1e4 && /^10*$/.test(String(n)) ? `1e${String(n).length - 1}` : String(n);

export function ropeTag(rope: ArchRope | null): string | undefined {
  if (!rope) return undefined;
  const s = rope.scaling ?? {};
  if (rope.kind === "yarn") return `YaRN ×${String(s.factor ?? "?")}`;
  // v1.1 §8: top-level partial_rotary_factor (scaling.partial_rotary_factor kept as fallback).
  const factor = rope.partial_rotary_factor ?? num(s.partial_rotary_factor);
  const partial = typeof factor === "number" && factor < 1 ? ` · ${Math.round(factor * 100)}%` : "";
  return `RoPE${rope.theta ? ` θ ${sci(rope.theta)}` : ""}${partial}`;
}

type GlyphProps = NodeMarkers & {
  onSelect?: () => void;
  selected?: boolean;
  label?: string;
  "data-testid"?: string;
};

/** One contract Component → one arch-ui component. */
export function ComponentGlyph({
  component,
  fallbackClass,
  ...common
}: { component: ArchComponent | null; fallbackClass?: string | undefined } & GlyphProps) {
  if (!component)
    return <Generic moduleClass={shortClass(fallbackClass) || "module"} {...common} />;
  switch (component.kind) {
    case "norm":
      return <Norm variant={component.variant} {...common} />;
    case "attention": {
      const x = component.extras;
      const mla = component.variant === "mla";
      const linear = component.variant === "linear";
      const gdn = x.kind === "gated_deltanet";
      // Linear-attention fields (v1.1 §7): describe the linear mixer itself.
      const linearDetail = linear
        ? [
            num(x.num_k_heads) !== undefined ? `${num(x.num_k_heads)} k-heads` : undefined,
            num(x.num_v_heads) !== undefined ? `${num(x.num_v_heads)} v-heads` : undefined,
            num(x.head_k_dim) !== undefined && num(x.head_v_dim) !== undefined
              ? `d ${num(x.head_k_dim)}/${num(x.head_v_dim)}`
              : undefined,
            num(x.conv_kernel) !== undefined ? `conv ${num(x.conv_kernel)}` : undefined,
          ]
            .filter(Boolean)
            .join(" · ")
        : "";
      return (
        <Attention
          variant={component.variant}
          headsQ={component.heads_q}
          headsKv={component.heads_kv}
          headDim={component.head_dim}
          rope={ropeTag(component.rope)}
          qkNorm={component.qk_norm}
          gated={x.output_gate === true}
          latent={
            mla && num(x.kv_lora_rank)
              ? {
                  qRank: num(x.q_lora_rank) ?? null,
                  kvRank: num(x.kv_lora_rank) as number,
                  ropeDim: num(x.qk_rope_head_dim),
                }
              : undefined
          }
          window={num(x.sliding_window)}
          indexTopk={num(x.index_topk)}
          name={gdn ? "Gated DeltaNet" : undefined}
          detail={linearDetail || undefined}
          {...common}
        />
      );
    }
    case "mlp":
      return <MLP variant={component.variant} intermediate={component.intermediate} {...common} />;
    case "moe":
      return (
        <MoE
          experts={component.experts}
          topK={component.top_k}
          sharedExperts={component.shared_experts}
          scoring={component.router.scoring}
          normTopk={component.router.norm_topk ?? undefined}
          groups={
            component.router.groups
              ? {
                  groups: component.router.groups.n_group,
                  topkGroups: component.router.groups.topk_group,
                }
              : undefined
          }
          expert={component.expert}
          shared={component.shared}
          {...common}
        />
      );
    case "generic":
      return (
        <Generic
          moduleClass={shortClass(component.class_name)}
          members={component.children.map((c) => ({ ...c, class_name: shortClass(c.class_name) }))}
          {...common}
        />
      );
    case "embedding":
      return (
        <Embedding
          vocab={component.vocab}
          hidden={component.hidden}
          tied={component.tied}
          {...common}
        />
      );
    case "lm_head":
      return (
        <LMHead
          vocab={component.vocab}
          hidden={component.hidden}
          tied={component.tied}
          {...common}
        />
      );
  }
}

const firstOut = (s: ArchStage): ArchTensor | undefined =>
  s.kind === "stack"
    ? s.bands[s.bands.length - 1]?.block.sublayers.at(-1)?.io.outputs[0]
    : s.io.outputs[0];

export function ArchFigure({
  arch,
  evaluate,
  selection,
  onSelectBand,
  onSelectSub,
  onSelectStage,
  detailId,
}: ArchFigureProps) {
  // v1.1 §5: `other` stages go on the spine unless `placement: "aside"` (legacy: aside).
  const isAside = (s: ArchStage) => s.kind === "other" && (s.placement ?? "aside") === "aside";
  const main = arch.stages.filter((s) => !isAside(s));
  const others = arch.stages.filter((s): s is ArchModuleStage => isAside(s));
  const branchAt = [...main].reverse().find((s) => s.kind !== "stack")?.id;

  const stageGlyph = (s: ArchModuleStage) => (
    <ComponentGlyph
      key={s.id}
      component={s.component ?? null}
      fallbackClass={s.module?.class_name ?? s.label ?? s.id}
      status={s.module?.status}
      notes={noteMarker(notesFor(arch, [s.id]))}
      selected={selection.stage === s.id}
      onSelect={() => onSelectStage(s.id)}
      label={`${s.label ?? s.id}: show details`}
      data-testid={`stage-${s.id}`}
    />
  );

  const subGlyph = (items: StackItem[], band: ArchBand, sub: ArchSublayer) => {
    const targets = represented(items, band).map((b) => subTarget(b.id, sub.id));
    return (
      <ComponentGlyph
        key={sub.id}
        component={sub.component}
        fallbackClass={sub.module?.class_name ?? sub.role}
        status={sub.module?.status}
        notes={noteMarker(notesFor(arch, targets))}
        selected={selection.band === band && selection.sub === sub.id}
        onSelect={() => onSelectSub(band, sub.id)}
        label={`${sub.id} (${layerRange(band.layers)}): show details`}
        data-testid={`sub-${band.id}-${sub.id}`}
      />
    );
  };

  const blockView = (items: StackItem[], band: ArchBand) => (
    <DecoderBlock residual={band.block.residual}>
      {blockSegments(band.block).map((seg, i) =>
        seg.kind === "branch" ? (
          <ResidualBranch
            // biome-ignore lint/suspicious/noArrayIndexKey: segments are positional
            key={i}
            add={
              seg.add
                ? {
                    onSelect: () => onSelectSub(band, (seg.add as ArchSublayer).id),
                    selected: selection.band === band && selection.sub === seg.add.id,
                  }
                : undefined
            }
          >
            {seg.body.map((s) => subGlyph(items, band, s))}
          </ResidualBranch>
        ) : (
          // biome-ignore lint/suspicious/noArrayIndexKey: segments are positional
          <BlockStep key={i}>{subGlyph(items, band, seg.sublayer)}</BlockStep>
        ),
      )}
    </DecoderBlock>
  );

  const bandView = (items: StackItem[], band: ArchBand, inPattern: boolean) => (
    <Band
      key={band.id}
      title={inPattern ? bandSummary(band) : layerRange(band.layers)}
      subtitle={inPattern ? undefined : bandSummary(band)}
      count={band.count}
      selected={selection.band === band}
      onSelect={() => onSelectBand(band)}
      controls={detailId}
      data-testid={`band-${band.id}`}
    >
      {blockView(items, band)}
    </Band>
  );

  const stackView = (stage: Extract<ArchStage, { kind: "stack" }>) => {
    const items = stackItems(stage.bands);
    const total = stage.bands.reduce((n, b) => n + b.count, 0);
    if (stackColumns(items) > MAX_STACK_COLUMNS) {
      // Too heterogeneous to draw side by side: a wrapping band list + the selected block.
      const shown =
        selection.band && stage.bands.includes(selection.band)
          ? selection.band
          : (stage.bands[0] as ArchBand);
      return (
        <BlockStack
          key={stage.id}
          title={stage.label ?? "Decoder layers"}
          layers={total}
          layout="column"
          data-testid={`stage-${stage.id}`}
        >
          <BandList data-testid="band-list">
            {stage.bands.map((b) => (
              <BandCard
                key={b.id}
                title={layerRange(b.layers)}
                subtitle={bandSummary(b)}
                count={b.count}
                selected={selection.band === b}
                onClick={() => onSelectBand(b)}
                data-testid={`band-card-${b.id}`}
              />
            ))}
          </BandList>
          {bandView(items, shown, false)}
        </BlockStack>
      );
    }
    return (
      <BlockStack
        key={stage.id}
        title={stage.label ?? "Decoder layers"}
        layers={total}
        data-testid={`stage-${stage.id}`}
      >
        {items.map((it) =>
          it.kind === "band" ? (
            bandView(items, it.band, false)
          ) : (
            <PatternGroup
              key={it.members[0]?.id}
              title={layerRange(it.layers)}
              subtitle={it.label}
              repeats={it.repeats}
              data-testid="pattern-group"
            >
              {it.period.map((b) => bandView(items, b, true))}
            </PatternGroup>
          ),
        )}
      </BlockStack>
    );
  };

  const spine: ReactNode[] = [
    <FigureTerminal
      key="input"
      name={arch.input.name}
      shape={arch.input.shape}
      dtype={arch.input.dtype}
    />,
  ];
  main.forEach((s, i) => {
    const prev = main[i - 1];
    const t = prev ? firstOut(prev) : undefined;
    spine.push(
      t ? <TensorEdge key={`e-${s.id}`} shape={t.shape} name={t.name} /> : null,
      s.kind === "stack" ? (
        stackView(s)
      ) : s.id === branchAt && others.length ? (
        <FigureBranch
          key={`branch-${s.id}`}
          node={stageGlyph(s)}
          label={others.map((o) => o.label ?? o.id).join(" · ")}
          branch={others.map(stageGlyph)}
          data-testid="figure-branch"
        />
      ) : (
        stageGlyph(s)
      ),
    );
  });
  spine.push(
    <FlowArrow key="e-out" length="lg" />,
    <FigureTerminal
      key="output"
      name={arch.output.name}
      shape={arch.output.shape}
      dtype={arch.output.dtype}
    />,
  );

  return (
    <ModelFigure
      evaluate={evaluate}
      data-testid="arch-figure"
      aria-label={`${arch.model.hf_id} architecture`}
    >
      {spine}
    </ModelFigure>
  );
}
