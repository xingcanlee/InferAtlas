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
  FigureMerge,
  FigureTerminal,
  Flow,
  FlowArrow,
  Generic,
  HyperBranch,
  LMHead,
  MhcBox,
  MLP,
  ModelFigure,
  MoE,
  type NodeMarkers,
  Norm,
  PatternGroup,
  ResidualBranch,
  TensorEdge,
  VariantBadge,
} from "@inferatlas/arch-ui";
import type {
  Arch,
  ArchBand,
  ArchComponent,
  ArchModuleStage,
  ArchRope,
  ArchStage,
  ArchSublayer,
  ArchTensor,
} from "@inferatlas/contracts";
import { type ReactNode, useState } from "react";
import {
  bandSummary,
  blockSegments,
  compactFocus,
  componentSummary,
  encoderStages,
  encoderTitle,
  isAsideStage,
  isConfigOnly,
  itemLayers,
  keyInternals,
  layerRange,
  MAX_STACK_COLUMNS,
  mhcStageLabel,
  modelInputs,
  noteMarker,
  notesAt,
  ownsModule,
  primaryStack,
  represented,
  type StackItem,
  shortClass,
  spineStages,
  stackColumns,
  stackItems,
  stageLabel,
  subNotes,
  tokenEmbedding,
} from "@/lib/arch";

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
  /** Draw key internal tensors (q/k/v, latents, routing) as chips in Attention / MoE. */
  showInternals?: boolean;
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
  /** Key internal tensors shown as chips inside Attention / MoE (optional, v1.2 §1). */
  tensors?: readonly ArchTensor[] | undefined;
  onSelect?: () => void;
  selected?: boolean;
  label?: string;
  "data-testid"?: string;
};

/** One contract Component → one arch-ui component. */
/**
 * How a generic / component-less stage is named in the figure (v1.4): its label ("Stream merge
 * (HC head)", "Multi-token prediction × 1") before the class, the class only when the stage is
 * that module (synthetic ops and config-only branches carry their containing model's class as
 * provenance, which would read as the node's own class), and a tag: `config only` for branches
 * the config declares but the model does not build, none for a ruled tensor op.
 */
interface StageNaming {
  name?: string | undefined;
  moduleClass?: string | undefined;
  tag?: string | null | undefined;
  /** Second line (config-only branches). */
  caption?: string | undefined;
  /** mHC stream expansion / merge: a one-line op on the wire (mHC-box height). */
  compact?: boolean | undefined;
}

function genericStage(arch: Arch, s: ArchModuleStage): StageNaming | undefined {
  if (s.component && s.component.kind !== "generic") return undefined;
  const label = s.label?.trim();
  const named = !!label && label !== s.id;
  const own = ownsModule(s);
  const configOnly = isConfigOnly(s);
  const mhc = s.placement !== "aside" && mhcStageLabel(s.id) !== undefined;
  if (!named && own && !configOnly && !mhc) return undefined;
  return {
    name: named ? stageLabel(arch, s) : undefined,
    moduleClass: own || !named ? shortClass(s.module?.class_name) : undefined,
    tag: configOnly || mhc || (!s.component && s.module?.status !== "generic") ? null : undefined,
    caption: configOnly ? "config only · not built by vLLM" : undefined,
    compact: mhc,
  };
}

export function ComponentGlyph({
  component,
  fallbackClass,
  stage,
  tensors,
  ...common
}: {
  component: ArchComponent | null;
  fallbackClass?: string | undefined;
  /** Stage naming for generic / component-less stages (`genericStage`). */
  stage?: StageNaming | undefined;
} & GlyphProps) {
  const tag =
    stage?.tag === undefined ? undefined : stage.tag === null ? null : (
      <VariantBadge>{stage.tag}</VariantBadge>
    );
  if (!component)
    return (
      <Generic
        moduleClass={stage ? (stage.moduleClass ?? "") : shortClass(fallbackClass) || "module"}
        name={stage?.name}
        tag={tag}
        caption={stage?.caption}
        size={stage?.compact ? "xs" : undefined}
        {...common}
      />
    );
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
          window={num(x.sliding_window) ?? (x.sliding_window === true ? true : undefined)}
          compressRatio={num(x.compress_ratio)}
          indexer={x.indexer === true}
          sinks={x.sinks === true}
          tensors={tensors}
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
          // A single routing group (n_group 1) is plain top-k routing: no group chip.
          groups={
            component.router.groups && component.router.groups.n_group > 1
              ? {
                  groups: component.router.groups.n_group,
                  topkGroups: component.router.groups.topk_group,
                }
              : undefined
          }
          expert={component.expert}
          shared={component.shared}
          tensors={tensors}
          {...common}
        />
      );
    case "generic":
      return (
        <Generic
          moduleClass={stage ? (stage.moduleClass ?? "") : shortClass(component.class_name)}
          name={stage?.name}
          tag={tag}
          caption={stage?.caption}
          size={stage?.compact ? "xs" : undefined}
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

/** Residual stream count of an mHC block, from the `mhc_*` input shape `[T, n_hc, H]`. */
function hyperStreams(band: ArchBand, evaluate: (e: string) => number | undefined): number {
  const pre = band.block.sublayers.find((s) => /^mhc/i.test(s.id));
  const shape = pre?.io.inputs[0]?.shape;
  const n = shape && shape.length === 3 ? evaluate(shape[1] as string) : undefined;
  return n && Number.isInteger(n) && n > 1 && n <= 16 ? n : 4;
}

/**
 * Input pill of an encoder branch: the model's own extra input (v1.4 §3 `inputs`, e.g.
 * `pixel_values`), else the first stage's declared input, else a muted placeholder that says it
 * is not in the data (never invent a shape).
 */
function encoderInput(
  arch: Arch,
  stages: ArchStage[],
): { name: string; shape?: string[]; dtype?: string; muted?: boolean; hint?: string } {
  const first = stages[0];
  const t =
    first?.kind === "stack"
      ? first.bands[0]?.block.sublayers[0]?.io.inputs[0]
      : first?.io.inputs[0];
  // v1.4 lists text + image inputs: the one the encoder's first stage consumes, else the first
  // non-text input (MiniMax-M3: pixel_values, not grid_thw).
  const extras = modelInputs(arch).filter((x) => x.name !== arch.input.name);
  const extra = extras.find((x) => x.name === t?.name) ?? extras[0];
  if (extra) return extra;
  if (t) return t;
  return {
    name: encoderTitle(stages) === "Vision encoder" ? "image" : "input",
    muted: true,
    hint: "(not in arch.json)",
  };
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
  showInternals = false,
}: ArchFigureProps) {
  // v1.3 §2: array order is dataflow order; `other` stages on the spine unless aside.
  const main = spineStages(arch);
  const others = arch.stages.filter((s): s is ArchModuleStage => isAsideStage(s));
  const primary = primaryStack(arch);
  // Secondary stacks (vision / encoder) start collapsed unless they hold the selection.
  const [openStacks, setOpenStacks] = useState<Set<string>>(() => new Set());
  const branchAt = [...main].reverse().find((s) => s.kind !== "stack")?.id;

  const stageGlyph = (s: ArchModuleStage) => (
    <ComponentGlyph
      key={s.id}
      component={s.component ?? null}
      fallbackClass={s.module?.class_name ?? stageLabel(arch, s)}
      stage={genericStage(arch, s)}
      status={s.module?.status}
      notes={noteMarker(notesAt(arch, [s.id]))}
      selected={selection.stage === s.id}
      onSelect={() => onSelectStage(s.id)}
      label={`${stageLabel(arch, s)}: show details`}
      data-testid={`stage-${s.id}`}
    />
  );

  const subGlyph = (items: StackItem[], band: ArchBand, sub: ArchSublayer) => {
    return (
      <ComponentGlyph
        key={sub.id}
        component={sub.component}
        fallbackClass={sub.module?.class_name ?? sub.role}
        status={sub.module?.status}
        notes={noteMarker(subNotes(arch, represented(items, band), sub.id))}
        tensors={showInternals ? keyInternals(sub) : undefined}
        selected={selection.band === band && selection.sub === sub.id}
        onSelect={() => onSelectSub(band, sub.id)}
        label={`${sub.id} (${layerRange(band.layers)}): show details`}
        data-testid={`sub-${band.id}-${sub.id}`}
      />
    );
  };

  const mhcBox = (
    items: StackItem[],
    band: ArchBand,
    sub: ArchSublayer | undefined,
    post: boolean,
    streams: number,
  ) =>
    sub ? (
      <MhcBox
        key={sub.id}
        post={post}
        streams={streams}
        status={sub.module?.status}
        notes={noteMarker(subNotes(arch, represented(items, band), sub.id))}
        onSelect={() => onSelectSub(band, sub.id)}
        selected={selection.band === band && selection.sub === sub.id}
        label={`${sub.id} (${layerRange(band.layers)}): show details`}
        data-testid={`sub-${band.id}-${sub.id}`}
      />
    ) : null;

  const blockView = (items: StackItem[], band: ArchBand) => (
    <DecoderBlock residual={band.block.residual} streams={hyperStreams(band, evaluate)}>
      {blockSegments(band.block).map((seg, i) =>
        seg.kind === "hyper" ? (
          <HyperBranch
            key={`hyper-${seg.pre?.id ?? seg.body[0]?.id ?? seg.post?.id}`}
            pre={mhcBox(items, band, seg.pre, false, hyperStreams(band, evaluate))}
            post={mhcBox(items, band, seg.post, true, hyperStreams(band, evaluate))}
          >
            {seg.body.map((s) => subGlyph(items, band, s))}
          </HyperBranch>
        ) : seg.kind === "branch" ? (
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
      // Inside a pattern group a single-layer member needs no "1×".
      hideCount={inPattern && band.count === 1}
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
    if (stage !== primary) {
      const holdsSelection = !!selection.band && stage.bands.includes(selection.band);
      const open = holdsSelection || openStacks.has(stage.id);
      const toggle = (next: boolean) =>
        setOpenStacks((prev) => {
          const s = new Set(prev);
          if (next) s.add(stage.id);
          else s.delete(stage.id);
          return s;
        });
      if (!open)
        return (
          <BlockStack
            key={stage.id}
            title={stageLabel(arch, stage)}
            layers={total}
            collapsible
            open={false}
            onOpenChange={toggle}
            summary={[...new Set(stage.bands.map(bandSummary))].join(" / ")}
            data-testid={`stage-${stage.id}`}
          />
        );
      return (
        <BlockStack
          key={stage.id}
          title={stageLabel(arch, stage)}
          layers={total}
          collapsible
          open
          onOpenChange={toggle}
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
              >
                {it.period.map((b) => bandView(items, b, true))}
              </PatternGroup>
            ),
          )}
        </BlockStack>
      );
    }
    const itemView = (it: StackItem, labelled = true) =>
      it.kind === "band" ? (
        bandView(items, it.band, false)
      ) : (
        <PatternGroup
          key={it.members[0]?.id}
          title={labelled ? layerRange(it.layers) : undefined}
          subtitle={labelled ? it.label : undefined}
          repeats={it.repeats}
          data-testid="pattern-group"
        >
          {it.period.map((b) => bandView(items, b, true))}
        </PatternGroup>
      );
    if (stackColumns(items) > MAX_STACK_COLUMNS) {
      // Too wide to draw side by side: one card per item + the selected (or largest) item.
      const focus = compactFocus(items, selection.band);
      return (
        <BlockStack
          key={stage.id}
          title={stageLabel(arch, stage)}
          layers={total}
          layout="column"
          data-testid={`stage-${stage.id}`}
        >
          <BandList data-testid="band-list">
            {items.map((it) => {
              const first = (it.kind === "band" ? it.band : it.period[0]) as ArchBand;
              return (
                <BandCard
                  key={first.id}
                  title={layerRange(itemLayers(it))}
                  subtitle={it.kind === "band" ? bandSummary(it.band) : it.label}
                  count={it.kind === "band" ? it.band.count : it.repeats}
                  selected={it === focus}
                  onClick={() => onSelectBand(first)}
                  data-testid={`band-card-${first.id}`}
                />
              );
            })}
          </BandList>
          {/* The selected card already labels the drawn item. */}
          {focus ? itemView(focus, false) : null}
        </BlockStack>
      );
    }
    return (
      <BlockStack
        key={stage.id}
        title={stageLabel(arch, stage)}
        layers={total}
        data-testid={`stage-${stage.id}`}
      >
        {items.map((it) => itemView(it))}
      </BlockStack>
    );
  };

  // The encoder branch starts collapsed (one summary frame) unless it holds the selection.
  const [encoderOpen, setEncoderOpen] = useState(false);
  const encoderView = (stages: ArchStage[]) => {
    const stackIds = stages.filter((x) => x.kind === "stack");
    const holds =
      stages.some((x) => x.id === selection.stage) ||
      (!!selection.band &&
        stackIds.some((x) => x.kind === "stack" && x.bands.includes(selection.band as ArchBand)));
    const open = holds || encoderOpen;
    const layers = stackIds.reduce(
      (n, x) => n + (x.kind === "stack" ? x.bands.reduce((m, b) => m + b.count, 0) : 0),
      0,
    );
    // v1.4 labels encoder stages ("Patch embedding", "Vision adapter"): the summary uses them;
    // older data falls back to the component summary / class name.
    const summary = stages
      .map((x) =>
        x.kind === "stack"
          ? `${x.bands.reduce((m, b) => m + b.count, 0)}× ${[...new Set(x.bands.map(bandSummary))].join(" / ")}`
          : x.label && stageLabel(arch, x) === x.label.trim() && x.label !== x.id
            ? x.label.trim()
            : x.component && x.component.kind !== "generic"
              ? componentSummary(x.component, x.kind)
              : shortClass(x.module.class_name),
      )
      .join(" → ");
    return (
      <>
        <FigureTerminal {...encoderInput(arch, stages)} data-testid="encoder-input" />
        <FlowArrow length="md" />
        <BlockStack
          title={encoderTitle(stages)}
          layers={layers || undefined}
          layout="column"
          collapsible
          open={open}
          onOpenChange={setEncoderOpen}
          summary={<span className="block max-w-[var(--arch-col)] text-center">{summary}</span>}
          data-testid="encoder-branch"
        >
          <Flow className="*:data-[slot=figure-node]:w-[var(--arch-col)]">
            {chain(stages, undefined, true)}
          </Flow>
        </BlockStack>
      </>
    );
  };

  /** A module stage drawn with corner badges (status other than ruled, or warn notes). */
  const badged = (s: ArchStage) =>
    s.kind !== "stack" &&
    ((!!s.module?.status && s.module.status !== "ruled") || !!noteMarker(notesAt(arch, [s.id])));

  /**
   * Stages joined by arrows; an edge is annotated with the previous stage's output tensor. The
   * label never meets the corner badges of the node it points into: on the spine it moves to the
   * left of the wire; in the narrow encoder column (full-width nodes) the edge gets longer.
   */
  const chain = (stages: ArchStage[], from?: ArchStage, column = false): ReactNode[] =>
    stages.flatMap((s, i) => {
      const prev = i ? stages[i - 1] : from;
      const t = prev ? firstOut(prev) : undefined;
      const hasBadges = badged(s);
      // A collapsed secondary stack is narrow: a longer edge keeps its label off the legend.
      const len = (s.kind === "stack" && s !== primary) || (column && hasBadges) ? "xl" : "md";
      return [
        t ? (
          <TensorEdge
            key={`e-${s.id}`}
            shape={t.shape}
            name={t.name}
            length={len}
            labelSide={hasBadges && !column ? "left" : "right"}
          />
        ) : null,
        s.kind === "stack" ? (
          stackView(s)
        ) : s.id === branchAt && others.length ? (
          <FigureBranch
            key={`branch-${s.id}`}
            node={stageGlyph(s)}
            // A named aside box carries its own label; the edge then names the tensor it reads.
            label={others
              .map((o) =>
                genericStage(arch, o)?.name
                  ? (o.io.inputs[0]?.name ?? stageLabel(arch, o))
                  : stageLabel(arch, o),
              )
              .join(" · ")}
            branch={others.map(stageGlyph)}
            data-testid="figure-branch"
          />
        ) : (
          stageGlyph(s)
        ),
      ];
    });

  const input = (
    <FigureTerminal
      key="input"
      name={arch.input.name}
      shape={arch.input.shape}
      dtype={arch.input.dtype}
    />
  );
  // VLMs: stages before the token embedding (vision tower, adapter, projector) are a second
  // input branch drawn left of the spine, merging into the token embeddings.
  const encoder = encoderStages(arch);
  const tok = encoder.length ? tokenEmbedding(arch) : undefined;
  const spine: ReactNode[] = [];
  if (tok) {
    const rest = main.slice(main.indexOf(tok) + 1);
    const out = firstOut(encoder[encoder.length - 1] as ArchStage);
    spine.push(
      <FigureMerge
        key="merge"
        top={input}
        node={stageGlyph(tok)}
        side={encoderView(encoder)}
        label={out ? `${out.name}` : "features"}
        data-testid="figure-merge"
      />,
      ...chain(rest, tok),
    );
  } else {
    spine.push(input, ...chain(main));
  }
  spine.push(
    <FlowArrow key="e-out" length="md" />,
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
