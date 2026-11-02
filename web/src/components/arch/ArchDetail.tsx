import {
  evaluateShape,
  formatNumeric,
  formatSymbolic,
  NoteBadge,
  numel,
  StatusBadge,
} from "@inferatlas/arch-ui";
import type {
  Arch,
  ArchBand,
  ArchModule,
  ArchModuleStage,
  ArchNote,
  ArchSublayer,
  ArchTensor,
} from "@inferatlas/contracts";
import { ChevronLeft, ChevronRight, ExternalLink, Info, TriangleAlert } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  bandSummary,
  componentSummary,
  encoderStages,
  encoderTitle,
  groupNotes,
  isConfigOnly,
  layerCount,
  layerPath,
  layerRange,
  notesAt,
  notesFor,
  ownsModule,
  RUNTIME_SYMBOLS,
  shortClass,
  stackItems,
  stageLabel,
  subNotes,
  uniqueNotes,
} from "@/lib/arch";
import { fmtBytes } from "@/lib/format";
import { cn } from "@/lib/utils";

type Evaluate = (expr: string) => number | undefined;

const DTYPE_BYTES: Record<string, number> = {
  float32: 4,
  bfloat16: 2,
  float16: 2,
  int64: 8,
  int32: 4,
  int8: 1,
  float8: 1,
};

function Section({
  title,
  children,
  className,
  aside,
}: {
  title: ReactNode;
  children: ReactNode;
  className?: string;
  aside?: ReactNode;
}) {
  return (
    <section className={cn("border-b px-4 py-3", className)}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
          {title}
        </h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

/**
 * Fully qualified names (`vllm.model_executor.layers.mhc.MHCPreOp`) wrap after a dot, not in the
 * middle of a class name; `overflow-wrap:anywhere` is only the fallback for a single long segment.
 */
export function QualifiedName({ name, className }: { name: string; className?: string }) {
  const parts = name.split(".");
  return (
    <span className={cn("[overflow-wrap:anywhere]", className)} title={name}>
      {parts.map((p, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: segments are positional
        <span key={i}>
          {p}
          {i < parts.length - 1 ? (
            <>
              .<wbr />
            </>
          ) : null}
        </span>
      ))}
    </span>
  );
}

/** Note prose: wrap qualified class names inside it at dots, never overflow the panel. */
function NoteText({ text }: { text: string }) {
  const parts = text.split(/(\b[\w]+(?:\.[\w]+){2,}\b)/);
  return (
    <>
      {parts.map((p, i) =>
        i % 2 ? (
          // biome-ignore lint/suspicious/noArrayIndexKey: segments are positional
          <QualifiedName key={i} name={p} className="font-mono text-[11px]" />
        ) : (
          p
        ),
      )}
    </>
  );
}

// ----------------------------------------------------------------------------- tensors

export function TensorTable({
  tensors,
  evaluate,
  kind,
}: {
  tensors: ArchTensor[];
  evaluate: Evaluate;
  kind: "io" | "param";
}) {
  if (!tensors.length)
    return (
      <p className="text-[12px] text-muted-foreground" data-testid="tensors-empty">
        {kind === "io" ? "not derived (unknown for this component)" : "none"}
      </p>
    );
  return (
    <ul className="flex flex-col">
      {tensors.map((t) => {
        const values = evaluateShape(t.shape, evaluate) ?? [];
        const n = numel(values);
        const bytes = kind === "param" && n !== undefined ? n * (DTYPE_BYTES[t.dtype] ?? 0) : 0;
        return (
          <li
            key={t.name}
            data-testid={kind === "io" ? "tensor-row" : "param-row"}
            className="border-t border-border/60 py-1 text-[11.5px] first:border-t-0"
          >
            <div className="flex items-baseline justify-between gap-2">
              <span className="min-w-0 truncate font-mono text-foreground" title={t.name}>
                {t.name}
              </span>
              <span className="shrink-0 font-mono text-[10.5px] text-muted-foreground">
                {t.dtype}
                {bytes ? <span className="tabular"> · {fmtBytes(bytes)}</span> : null}
              </span>
            </div>
            <div className="flex flex-wrap items-baseline gap-x-1.5 font-mono">
              <span data-testid="shape-sym" className="text-muted-foreground">
                {formatSymbolic(t.shape)}
              </span>
              <span aria-hidden="true" className="text-muted-foreground/60">
                =
              </span>
              <span data-testid="shape-num" className="tabular text-foreground">
                {formatNumeric(values)}
              </span>
            </div>
          </li>
        );
      })}
    </ul>
  );
}

// ------------------------------------------------------------------------------- notes

/**
 * Model-wide notes grouped by `triage_kind` with counts. Warn notes are listed; info notes are
 * collapsed behind a disclosure so they never crowd out the structure.
 */
export function NoteGroups({ notes }: { notes: ArchNote[] }) {
  const groups = groupNotes(notes);
  return (
    <div className="flex flex-col gap-2.5" data-testid="note-groups">
      {groups.map((g) => {
        const warn = g.notes.filter((n) => n.severity === "warn");
        const info = g.notes.filter((n) => n.severity !== "warn");
        return (
          <section
            key={g.kind ?? "general"}
            data-testid={`note-group-${g.kind ?? "general"}`}
            className="flex flex-col gap-1"
          >
            <header className="flex items-center gap-2 text-[11.5px]">
              <span className="font-mono">{g.kind ?? "general"}</span>
              {warn.length ? <NoteBadge severity="warn" count={warn.length} size="md" /> : null}
              {info.length ? <NoteBadge severity="info" count={info.length} size="md" /> : null}
            </header>
            <NoteList notes={warn} withTarget />
            {info.length ? (
              <details className="group/info text-[12px]" data-testid="info-notes">
                <summary className="cursor-pointer text-muted-foreground select-none hover:text-foreground">
                  {info.length} info note{info.length === 1 ? "" : "s"}
                </summary>
                <div className="mt-1">
                  <NoteList notes={info} withTarget />
                </div>
              </details>
            ) : null}
          </section>
        );
      })}
    </div>
  );
}

export function NoteList({ notes, withTarget }: { notes: ArchNote[]; withTarget?: boolean }) {
  const list = uniqueNotes(notes);
  if (!list.length) return null;
  return (
    <ul className="flex flex-col gap-1.5" data-testid="note-list">
      {list.map((n) => (
        <li
          key={`${n.target}-${n.text}`}
          data-severity={n.severity}
          className={cn(
            "flex gap-2 rounded-md border px-2 py-1.5 text-[12px]",
            n.severity === "warn" ? "border-warn/35 bg-warn/5" : "bg-muted/40",
          )}
        >
          {n.severity === "warn" ? (
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-warn" aria-label="warning" />
          ) : (
            <Info className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" aria-label="info" />
          )}
          <span className="min-w-0 [overflow-wrap:anywhere]">
            {withTarget && n.target ? (
              <span className="mr-1.5 font-mono text-[10.5px] text-muted-foreground">
                {n.target}
              </span>
            ) : null}
            <NoteText text={n.text} />
            {!withTarget && n.triage_kind ? (
              <span className="ml-1.5 rounded border px-1 font-mono text-[11px] text-muted-foreground">
                {n.triage_kind}
              </span>
            ) : null}
          </span>
        </li>
      ))}
    </ul>
  );
}

// ------------------------------------------------------------------------------ module

function ModuleCard({
  title,
  subtitle,
  module,
  path,
  io,
  notes,
  evaluate,
  selected,
  testId,
  id,
  provenance,
}: {
  title: string;
  subtitle?: string | undefined;
  module: ArchModule | null;
  /** The module fields are provenance (synthetic / config-only stage): shown instead of them. */
  provenance?: string | undefined;
  path?: string | undefined;
  io: { inputs: ArchTensor[]; outputs: ArchTensor[]; internals?: ArchTensor[] | undefined };
  notes: ArchNote[];
  evaluate: Evaluate;
  selected?: boolean;
  testId: string;
  id?: string;
}) {
  return (
    <article
      id={id}
      data-testid={testId}
      data-selected={selected || undefined}
      className={cn(
        "rounded-lg border bg-card",
        selected && "border-focus/60 ring-2 ring-focus/20",
      )}
    >
      <header className="flex items-start justify-between gap-2 border-b px-3 py-2">
        <div className="min-w-0">
          <div className="font-medium">{title}</div>
          {subtitle ? <div className="text-[11px] text-muted-foreground">{subtitle}</div> : null}
        </div>
        {module ? <StatusBadge status={module.status} size="md" /> : null}
      </header>
      <div className="flex flex-col gap-2.5 px-3 py-2">
        {provenance ? (
          <p className="text-[11.5px] text-muted-foreground" data-testid="stage-provenance">
            {provenance}
          </p>
        ) : null}
        {module && !provenance ? (
          <div className="flex flex-col gap-0.5 text-[11.5px]">
            <div className="font-mono" data-testid="module-path">
              <QualifiedName name={path ?? module.path} />
            </div>
            <div className="flex flex-wrap items-center gap-x-2 text-muted-foreground">
              <span className="w-full font-mono" data-testid="class-name">
                <QualifiedName name={module.class_name} />
              </span>
              {module.source ? (
                <a
                  href={module.source.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-0.5 hover:text-foreground"
                  data-testid="source-link"
                >
                  {module.source.file.split("/").at(-1)}:{module.source.line}
                  <ExternalLink className="size-3" />
                </a>
              ) : null}
            </div>
            {module.replaced ? (
              <div className="text-[11px] text-overlay">
                Ascend:{" "}
                <QualifiedName name={module.replaced.baseline_class} className="font-mono" /> →{" "}
                <QualifiedName name={module.replaced.ascend_class} className="font-mono" />
              </div>
            ) : null}
          </div>
        ) : null}
        <div>
          <div className="mb-0.5 text-[10.5px] font-medium text-muted-foreground uppercase">
            Inputs
          </div>
          <TensorTable tensors={io.inputs} evaluate={evaluate} kind="io" />
        </div>
        <div>
          <div className="mb-0.5 text-[10.5px] font-medium text-muted-foreground uppercase">
            Outputs
          </div>
          <TensorTable tensors={io.outputs} evaluate={evaluate} kind="io" />
        </div>
        {io.internals?.length ? (
          <div data-testid="internals">
            <div className="mb-0.5 text-[10.5px] font-medium text-muted-foreground uppercase">
              Internal tensors · {io.internals.length}
            </div>
            <TensorTable tensors={io.internals} evaluate={evaluate} kind="io" />
          </div>
        ) : null}
        {module?.params.length ? (
          <div>
            <div className="mb-0.5 text-[10.5px] font-medium text-muted-foreground uppercase">
              Parameters · {module.params.length}
            </div>
            <TensorTable tensors={module.params} evaluate={evaluate} kind="param" />
          </div>
        ) : null}
        <NoteList notes={notes} />
      </div>
    </article>
  );
}

// ------------------------------------------------------------------------ layer select

/**
 * Jump to any decoder layer. `layer` is the ordinal over all stacks (== layer index for one
 * stack); with several stacks the current stack's label is shown. Hidden without a stack.
 */
export function LayerSelector({
  total,
  layer,
  onLayer,
  stackLabel,
  touch = false,
}: {
  total: number;
  layer: number | undefined;
  onLayer: (layer: number | undefined) => void;
  stackLabel?: string | null | undefined;
  /** Touch layout: 40px targets, short total ("/ 43"). */
  touch?: boolean;
}) {
  const [text, setText] = useState(layer === undefined ? "" : String(layer));
  useEffect(() => setText(layer === undefined ? "" : String(layer)), [layer]);
  const commit = (raw: string) => {
    if (raw.trim() === "") return onLayer(undefined);
    const n = Number(raw);
    if (Number.isInteger(n)) onLayer(Math.min(Math.max(n, 0), total - 1));
  };
  const step = (d: number) =>
    onLayer(Math.min(Math.max((layer ?? (d > 0 ? -1 : total)) + d, 0), total - 1));
  if (total === 0)
    return (
      <span className="text-[12px] text-muted-foreground" data-testid="layer-selector">
        No decoder layers
      </span>
    );
  const btn = touch ? "size-10" : "size-7";
  return (
    <div className="flex min-w-0 items-center gap-1" data-testid="layer-selector">
      <label
        htmlFor="arch-layer"
        className={cn("mr-1 text-[12px] text-muted-foreground", touch && "sr-only")}
      >
        Layer
      </label>
      <Button
        variant="outline"
        size="icon-sm"
        className={btn}
        aria-label="Previous layer"
        onClick={() => step(-1)}
        disabled={layer === 0}
      >
        <ChevronLeft className="size-3.5" />
      </Button>
      <Input
        id="arch-layer"
        data-testid="layer-input"
        inputMode="numeric"
        className={cn(
          "px-2 text-center font-mono",
          // ≥ 16px on touch so iOS does not zoom the page when the field takes focus.
          touch ? "h-10 w-14 text-[16px]" : "h-7 w-14 text-[12px]",
        )}
        placeholder="—"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit((e.target as HTMLInputElement).value);
          else if (e.key === "ArrowUp") {
            e.preventDefault();
            step(1);
          } else if (e.key === "ArrowDown") {
            e.preventDefault();
            step(-1);
          }
        }}
      />
      <Button
        variant="outline"
        size="icon-sm"
        className={btn}
        aria-label="Next layer"
        onClick={() => step(1)}
        disabled={layer === total - 1}
      >
        <ChevronRight className="size-3.5" />
      </Button>
      <span
        className="ml-1 min-w-0 truncate text-[11px] whitespace-nowrap text-muted-foreground"
        data-testid="layer-total"
      >
        {touch ? `/ ${total}` : `of ${total} layers`}
        {stackLabel && !touch ? ` · ${stackLabel}` : ""}
      </span>
    </div>
  );
}

// ----------------------------------------------------------------------------- symbols

export function SymbolsEditor({
  arch,
  env,
  overridden,
  onSymbol,
}: {
  arch: Arch;
  env: Record<string, unknown>;
  overridden: Set<string>;
  onSymbol: (name: string, value: number | undefined) => void;
}) {
  /** Config-derived dimensions are shown read-only; only runtime symbols are editable. */
  const fixed = Object.entries(arch.symbols).filter(([k]) => !RUNTIME_SYMBOLS.has(k));
  return (
    <div className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-1" data-testid="symbols">
      {Object.entries(arch.symbols)
        .filter(([k]) => RUNTIME_SYMBOLS.has(k))
        .map(([k, s]) => (
          <SymbolRow
            key={k}
            name={k}
            description={s.description ?? undefined}
            value={Number(env[k])}
            dflt={s.value}
            changed={overridden.has(k)}
            onChange={(v) => onSymbol(k, v === s.value ? undefined : v)}
          />
        ))}
      {fixed.length ? (
        <>
          <div className="col-span-2 mt-2 text-[10.5px] font-medium text-muted-foreground uppercase">
            Model dimensions
          </div>
          <dl
            className="col-span-2 grid grid-cols-[auto_1fr_auto] gap-x-3 gap-y-0.5 text-[12px]"
            data-testid="model-dims"
          >
            {fixed.map(([k, s]) => (
              <div key={k} className="contents">
                <dt className="font-mono">{k}</dt>
                <dd
                  className="truncate text-[11px] text-muted-foreground"
                  title={s.description ?? ""}
                >
                  {s.description}
                </dd>
                <dd className="text-right font-mono tabular">{s.value.toLocaleString("en-US")}</dd>
              </div>
            ))}
          </dl>
        </>
      ) : null}
    </div>
  );
}

function SymbolRow({
  name,
  description,
  value,
  dflt,
  changed,
  onChange,
}: {
  name: string;
  description: string | undefined;
  value: number;
  dflt: number;
  changed: boolean;
  onChange: (v: number | undefined) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const n = Number(text);
    if (Number.isFinite(n) && n > 0) onChange(n);
    else setText(String(value));
  };
  return (
    <>
      <label
        htmlFor={`sym-${name}`}
        className="flex min-w-0 flex-col leading-tight"
        title={description}
      >
        <span className={cn("font-mono text-[12px]", changed && "text-focus")}>{name}</span>
        {description ? (
          <span className="truncate text-[10.5px] text-muted-foreground">{description}</span>
        ) : null}
      </label>
      <div className="flex items-center gap-1">
        <Input
          id={`sym-${name}`}
          data-testid={`sym-${name}`}
          inputMode="decimal"
          className="h-7 px-2 text-right font-mono text-[12px]"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => e.key === "Enter" && commit()}
        />
        {changed ? (
          <button
            type="button"
            className="text-[10.5px] text-muted-foreground underline-offset-2 hover:underline"
            onClick={() => onChange(undefined)}
            title={`Reset to ${dflt}`}
          >
            reset
          </button>
        ) : null}
      </div>
    </>
  );
}

// ------------------------------------------------------------------------------ panels

export function LayerDetail({
  arch,
  band,
  layer,
  stackLabel,
  sub,
  evaluate,
}: {
  arch: Arch;
  band: ArchBand;
  layer: number;
  stackLabel?: string | undefined;
  sub: string | undefined;
  evaluate: Evaluate;
}) {
  useEffect(() => {
    if (sub) document.getElementById(`sub-card-${sub}`)?.scrollIntoView({ block: "nearest" });
  }, [sub]);
  return (
    <div data-testid="layer-detail">
      <div className="border-b px-4 py-3">
        <div className="text-[15px] font-semibold tracking-tight" data-testid="layer-title">
          {stackLabel ? `${stackLabel} · ` : ""}Layer {layer}
        </div>
        <div className="text-[12px] text-muted-foreground">
          {layerRange(band.layers)} · {band.count}× · {bandSummary(band)} ·{" "}
          {band.block.residual.replace("_", "-")} residual
          {band.pattern ? ` · ${band.pattern}` : ""}
        </div>
        <div className="mt-2">
          <NoteList notes={notesFor(arch, [band.id])} />
        </div>
      </div>
      <ol className="flex flex-col gap-2 p-3">
        {band.block.sublayers.map((s: ArchSublayer, i) => (
          <li key={s.id}>
            <ModuleCard
              id={`sub-card-${s.id}`}
              testId={`sub-card-${s.id}`}
              title={`${i + 1}. ${componentSummary(s.component, s.role, s.id)}`}
              subtitle={`${s.id} · ${s.role.replace("_", " ")}`}
              module={s.module}
              path={s.module ? layerPath(s.module, band, layer) : undefined}
              io={s.io}
              notes={subNotes(arch, [band], s.id)}
              evaluate={evaluate}
              selected={sub === s.id}
            />
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Why a v1.4 synthetic stage's module fields describe another module (`ownsModule`). */
function stageProvenance(stage: ArchModuleStage): string | undefined {
  if (isConfigOnly(stage))
    return `Config only: declared by the model config but not constructed by vLLM (no parameters extracted). Declared by ${shortClass(stage.module.class_name)}.`;
  if (!ownsModule(stage))
    return `Tensor operation inside ${stage.module.path || "the model"} (${shortClass(stage.module.class_name)}); it has no module of its own.`;
  return undefined;
}

export function StageDetail({
  arch,
  stage,
  evaluate,
}: {
  arch: Arch;
  stage: ArchModuleStage;
  evaluate: Evaluate;
}) {
  return (
    <div className="p-3" data-testid="stage-detail">
      <ModuleCard
        testId={`stage-card-${stage.id}`}
        title={stageLabel(arch, stage) || componentSummary(stage.component ?? null, stage.kind)}
        subtitle={`${stage.id} · ${stage.kind.replace("_", " ")}`}
        module={stage.module}
        provenance={stageProvenance(stage)}
        io={stage.io}
        notes={notesAt(arch, [stage.id])}
        evaluate={evaluate}
      />
    </div>
  );
}

export function Overview({
  arch,
  onBand,
  onStage,
}: {
  arch: Arch;
  onBand: (band: ArchBand) => void;
  onStage: (id: string) => void;
}) {
  const warn = uniqueNotes(arch.notes).filter((n) => n.severity === "warn").length;
  const encoder = encoderStages(arch);
  const stageItem = (s: Arch["stages"][number]) => (
    <li key={s.id}>
      {s.kind === "stack" ? (
        <div>
          <span className="font-medium">{stageLabel(arch, s)}</span>
          <ul className="mt-0.5 ml-3 flex flex-col">
            {stackItems(s.bands).map((it) => {
              const first = (it.kind === "band" ? it.band : it.period[0]) as ArchBand;
              return (
                <li key={first.id}>
                  <button
                    type="button"
                    className="text-left hover:underline"
                    onClick={() => onBand(first)}
                  >
                    {layerRange(it.kind === "band" ? it.band.layers : it.layers)}{" "}
                    <span className="text-muted-foreground">
                      ·{" "}
                      {it.kind === "band" ? bandSummary(it.band) : `(${it.label}) × ${it.repeats}`}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ) : (
        <button
          type="button"
          className="text-left [overflow-wrap:anywhere] hover:underline"
          onClick={() => onStage(s.id)}
        >
          <span className="font-medium">{stageLabel(arch, s)}</span>{" "}
          {isConfigOnly(s) ? (
            <span className="text-[11px] text-muted-foreground">config only</span>
          ) : ownsModule(s) ? (
            <span className="font-mono text-[11px] text-muted-foreground">
              <span title={s.module.class_name}>{shortClass(s.module.class_name)}</span>
            </span>
          ) : null}
        </button>
      )}
    </li>
  );
  return (
    <div data-testid="arch-overview">
      <Section title="Model">
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[12px]">
          <dt className="text-muted-foreground">Architecture</dt>
          <dd className="font-mono">{arch.model.architecture}</dd>
          <dt className="text-muted-foreground">Family</dt>
          <dd>{arch.model.family}</dd>
          <dt className="text-muted-foreground">Layers</dt>
          <dd className="tabular">{layerCount(arch)}</dd>
          <dt className="text-muted-foreground">Device</dt>
          <dd className="font-mono">{arch.device}</dd>
        </dl>
      </Section>
      <Section title="Dataflow">
        <ol className="flex flex-col gap-1 text-[12px]" data-testid="dataflow">
          {encoder.length ? (
            <li key="encoder" data-testid="dataflow-encoder">
              <span className="font-medium">{encoderTitle(encoder)}</span>
              <ol className="mt-0.5 ml-3 flex flex-col gap-0.5">{encoder.map(stageItem)}</ol>
            </li>
          ) : null}
          {arch.stages.filter((s) => !encoder.includes(s)).map(stageItem)}
        </ol>
      </Section>
      <Section
        title="Notes"
        aside={warn ? <NoteBadge severity="warn" count={warn} size="md" /> : undefined}
      >
        {arch.notes.length ? (
          <NoteGroups notes={arch.notes} />
        ) : (
          <p className="text-[12px] text-muted-foreground">No notes.</p>
        )}
      </Section>
    </div>
  );
}

export { Section };
