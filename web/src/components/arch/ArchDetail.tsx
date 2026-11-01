import {
  evaluateShape,
  formatNumeric,
  formatSymbolic,
  NoteBadge,
  numel,
  StatusBadge,
} from "@inferatlas/arch-ui";
import { ChevronLeft, ChevronRight, ExternalLink, Info, TriangleAlert } from "lucide-react";
import { type ReactNode, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  bandSummary,
  componentSummary,
  layerCount,
  layerPath,
  layerRange,
  notesFor,
  RUNTIME_SYMBOLS,
  shortClass,
  subTarget,
  uniqueNotes,
} from "@/lib/arch";
import type {
  Arch,
  ArchBand,
  ArchModule,
  ArchModuleStage,
  ArchNote,
  ArchParam,
  ArchSublayer,
  ArchTensor,
} from "@/lib/arch-types";
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

// ----------------------------------------------------------------------------- tensors

export function TensorTable({
  tensors,
  evaluate,
  kind,
}: {
  tensors: (ArchTensor | ArchParam)[];
  evaluate: Evaluate;
  kind: "io" | "param";
}) {
  if (!tensors.length) return <p className="text-[12px] text-muted-foreground">none</p>;
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

export function NoteList({ notes }: { notes: ArchNote[] }) {
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
          <span>
            {n.text}
            {n.triage_kind ? (
              <span className="ml-1.5 rounded border px-1 font-mono text-[10px] text-muted-foreground">
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
}: {
  title: string;
  subtitle?: string | undefined;
  module: ArchModule | null;
  path?: string | undefined;
  io: { inputs: ArchTensor[]; outputs: ArchTensor[] };
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
        {module ? (
          <div className="flex flex-col gap-0.5 text-[11.5px]">
            <div className="font-mono break-all" data-testid="module-path">
              {path ?? module.path}
            </div>
            <div className="flex flex-wrap items-center gap-x-2 text-muted-foreground">
              <span className="w-full font-mono break-all" data-testid="class-name">
                {module.class_name}
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
                Ascend: <span className="font-mono">{module.replaced.baseline_class}</span> →{" "}
                <span className="font-mono">{module.replaced.ascend_class}</span>
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
}: {
  total: number;
  layer: number | undefined;
  onLayer: (layer: number | undefined) => void;
  stackLabel?: string | null | undefined;
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
  return (
    <div className="flex items-center gap-1" data-testid="layer-selector">
      <label htmlFor="arch-layer" className="mr-1 text-[12px] text-muted-foreground">
        Layer
      </label>
      <Button
        variant="outline"
        size="icon-sm"
        className="size-7"
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
        className="h-7 w-14 px-2 text-center font-mono text-[12px]"
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
        className="size-7"
        aria-label="Next layer"
        onClick={() => step(1)}
        disabled={layer === total - 1}
      >
        <ChevronRight className="size-3.5" />
      </Button>
      <span
        className="ml-1 text-[11px] whitespace-nowrap text-muted-foreground"
        data-testid="layer-total"
      >
        of {total} layers{stackLabel ? ` · ${stackLabel}` : ""}
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
              title={`${i + 1}. ${componentSummary(s.component, s.role)}`}
              subtitle={`${s.id} · ${s.role.replace("_", " ")}`}
              module={s.module}
              path={s.module ? layerPath(s.module, band, layer) : undefined}
              io={s.io}
              notes={notesFor(arch, [subTarget(band.id, s.id)])}
              evaluate={evaluate}
              selected={sub === s.id}
            />
          </li>
        ))}
      </ol>
    </div>
  );
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
        title={stage.label ?? componentSummary(stage.component ?? null, stage.kind)}
        subtitle={`${stage.id} · ${stage.kind.replace("_", " ")}`}
        module={stage.module}
        io={stage.io}
        notes={notesFor(arch, [stage.id])}
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
        <ol className="flex flex-col gap-1 text-[12px]">
          {arch.stages.map((s) => (
            <li key={s.id}>
              {s.kind === "stack" ? (
                <div>
                  <span className="font-medium">{s.label ?? "Decoder layers"}</span>
                  <ul className="mt-0.5 ml-3 flex flex-col">
                    {s.bands.length > 8 ? (
                      <li className="text-muted-foreground">
                        {s.bands.length} bands
                        {s.bands[0]?.pattern ? ` · ${s.bands[0].pattern}` : ""}
                      </li>
                    ) : (
                      s.bands.map((b) => (
                        <li key={b.id}>
                          <button
                            type="button"
                            className="text-left hover:underline"
                            onClick={() => onBand(b)}
                          >
                            {layerRange(b.layers)}{" "}
                            <span className="text-muted-foreground">· {bandSummary(b)}</span>
                          </button>
                        </li>
                      ))
                    )}
                  </ul>
                </div>
              ) : (
                <button
                  type="button"
                  className="text-left hover:underline"
                  onClick={() => onStage(s.id)}
                >
                  <span className="font-medium">{s.label ?? s.id}</span>{" "}
                  <span className="font-mono text-[11px] text-muted-foreground">
                    <span title={s.module?.class_name}>{shortClass(s.module?.class_name)}</span>
                  </span>
                </button>
              )}
            </li>
          ))}
        </ol>
      </Section>
      <Section
        title="Notes"
        aside={warn ? <NoteBadge severity="warn" count={warn} size="md" /> : undefined}
      >
        {arch.notes.length ? (
          <NoteList notes={arch.notes} />
        ) : (
          <p className="text-[12px] text-muted-foreground">No notes.</p>
        )}
      </Section>
    </div>
  );
}

export { Section };
