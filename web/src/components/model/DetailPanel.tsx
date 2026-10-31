import type { IRNode, TensorSpec } from "@inferatlas/contracts";
import { ArrowDownRight, ChevronDown, ExternalLink, Maximize2, X } from "lucide-react";
import type { ReactNode } from "react";
import { shapeText } from "@/components/graph/build";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { srcUrl } from "@/lib/data";
import { tryEvaluate } from "@/lib/expr";
import { fmtBytes, fmtFlops, fmtNum, fmtSI, qText, shortClass } from "@/lib/format";
import { nodeMetrics, type Phase, type Q, unshardedEnv } from "@/lib/model";
import { cn } from "@/lib/utils";
import { drillTarget, inheritedParallel, irNode, viewForNode } from "@/lib/views";
import { useModel } from "@/pages/model/context";

export function Section({
  title,
  children,
  aside,
}: {
  title: string;
  children: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <section className="border-b px-4 py-3">
      <div className="mb-2 flex items-center">
        <h3 className="text-[11px] font-semibold tracking-wide text-muted-foreground uppercase">
          {title}
        </h3>
        {aside ? <div className="ml-auto">{aside}</div> : null}
      </div>
      {children}
    </section>
  );
}

export function Row({
  k,
  v,
  title,
  strong,
}: {
  k: ReactNode;
  v: ReactNode;
  title?: string | undefined;
  strong?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5">
      <span className="shrink-0 text-muted-foreground">{k}</span>
      <span
        className={cn(
          "min-w-0 truncate text-right font-mono text-[12px] tabular",
          strong && "font-medium",
        )}
        title={title}
      >
        {v}
      </span>
    </div>
  );
}

function SrcLink({ src, label }: { src: IRNode["src"]; label?: string }) {
  return (
    <a
      href={srcUrl(src)}
      target="_blank"
      rel="noreferrer"
      className="group/src inline-flex max-w-full items-center gap-1 font-mono text-[11px] text-muted-foreground hover:text-foreground"
      title={`${src.repo}@${src.commit}`}
    >
      <span className="truncate">
        {label ?? `${src.repo}/${src.file}`}:{src.line}
      </span>
      <ExternalLink className="size-3 shrink-0 opacity-60 group-hover/src:opacity-100" />
    </a>
  );
}

function Tensor({ t }: { t: TensorSpec }) {
  const m = useModel();
  const sym = shapeText(t.shape, m.envState.env, m.phase, "sym");
  const num = shapeText(t.shape, m.envState.env, m.phase, "num");
  const dtype = t.dtype ? (m.envState.env[t.dtype] ?? t.dtype) : undefined;
  return (
    <div className="flex items-baseline gap-2 py-0.5">
      <span className="w-20 shrink-0 truncate text-muted-foreground">{t.name}</span>
      <span
        className="min-w-0 flex-1 font-mono text-[12px] break-all"
        title={m.shapes === "num" ? sym : num}
        data-testid="tensor-shape"
      >
        {m.shapes === "num" ? num : sym}
      </span>
      {dtype ? (
        <span className="font-mono text-[11px] text-muted-foreground">{String(dtype)}</span>
      ) : null}
    </div>
  );
}

function PhasePair({
  label,
  get,
  fmt,
}: {
  label: string;
  get: (p: Phase) => Q;
  fmt: (x: number) => string;
}) {
  const m = useModel();
  return (
    <div className="grid grid-cols-[1fr_auto_auto] items-baseline gap-x-3 py-0.5">
      <span className="text-muted-foreground">{label}</span>
      {(["prefill", "decode"] as const).map((p) => {
        const q = get(p);
        return (
          <span
            key={p}
            className={cn(
              "text-right font-mono text-[12px] tabular",
              p === m.phase ? "text-foreground" : "text-muted-foreground",
            )}
            title={q.e ?? `${p}`}
          >
            {qText(q, fmt)}
          </span>
        );
      })}
    </div>
  );
}

function IntensityBar({ ai }: { ai: number }) {
  const m = useModel();
  const hw = m.hw;
  const dtype = String(m.envState.env.dtype_a ?? "bf16");
  const peak = hw?.peak_tflops[dtype] ?? null;
  const bw = hw?.bandwidth_gbps ?? null;
  if (!peak || !bw) {
    return (
      <div className="text-[11px] text-muted-foreground">
        No {dtype} peak / bandwidth for {hw?.label ?? "this device"} — roofline position hidden.
      </div>
    );
  }
  const ridge = (peak * 1e12) / (bw * 1e9);
  const lo = 0.1;
  const hi = 10000;
  const pos = (x: number) =>
    (Math.log10(Math.min(Math.max(x, lo), hi)) - Math.log10(lo)) / Math.log10(hi / lo);
  const bound = ai < ridge ? "memory-bound" : "compute-bound";
  const attainable = Math.min(peak * 1e12, ai * bw * 1e9);
  return (
    <div className="space-y-1.5">
      <div className="relative h-5">
        <div className="absolute inset-x-0 top-2 h-px bg-border" />
        <div
          className="absolute top-0.5 h-4 w-px bg-muted-foreground/60"
          style={{ left: `${pos(ridge) * 100}%` }}
          title={`ridge ${fmtNum(ridge)} FLOP/B`}
        />
        <div
          className="absolute top-[3px] size-2.5 -translate-x-1/2 rounded-full border-2 border-background bg-focus"
          style={{ left: `${pos(ai) * 100}%` }}
        />
      </div>
      <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
        <span>0.1</span>
        <span>ridge {fmtNum(Number(ridge.toFixed(0)))}</span>
        <span>10⁴ FLOP/B</span>
      </div>
      <Row k="Bound" v={bound} strong />
      <Row k="Attainable" v={`${fmtSI(attainable, "FLOP/s")}`} />
    </div>
  );
}

function AscendCard({ node }: { node: IRNode }) {
  const m = useModel();
  const a = node.ascend;
  if (!a) return null;
  if (!m.overlay) {
    return (
      <Section title="Ascend">
        <button
          type="button"
          className="text-left text-[12px] text-muted-foreground hover:text-foreground"
          onClick={() => m.patch({ overlay: true })}
        >
          <span className="font-mono text-overlay">{a.tag.join(" ")}</span> — turn on the Ascend
          overlay to compare.
        </button>
      </Section>
    );
  }
  return (
    <Section title="Ascend difference">
      <div
        className="rounded-md border border-dashed border-overlay/60 p-3"
        data-testid="ascend-card"
      >
        <div className="mb-2 flex flex-wrap items-center gap-1.5">
          {a.tag.map((t) => (
            <span
              key={t}
              className="rounded-[3px] border border-overlay/60 px-1 font-mono text-[10px] font-semibold text-overlay"
            >
              {t}
            </span>
          ))}
          <span className="rounded-[3px] bg-muted px-1 font-mono text-[10px] text-muted-foreground">
            {a.level}
          </span>
          <span className="ml-auto font-mono text-[10px] text-muted-foreground">
            {a.annotation_id}
          </span>
        </div>
        <div className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 text-[12px]">
          <span className="text-muted-foreground">vLLM</span>
          <div className="min-w-0">
            <div className="truncate font-mono text-[11.5px]">{shortClass(node.class)}</div>
            <SrcLink src={node.src} />
          </div>
          {a.src.map((s) => (
            <FragmentRow key={`${s.file}:${s.line}`} label="Ascend" src={s} />
          ))}
        </div>
        <p className="mt-2 text-[12px] leading-relaxed">{a.note}</p>
      </div>
    </Section>
  );
}

function FragmentRow({ label, src }: { label: string; src: IRNode["src"] }) {
  return (
    <>
      <span className="text-muted-foreground">{label}</span>
      <div className="min-w-0">
        <div className="truncate font-mono text-[11.5px]">{shortClass(src.symbol)}</div>
        <SrcLink src={src} />
      </div>
    </>
  );
}

function Formulas({ node }: { node: IRNode }) {
  const rows: [string, string][] = [
    ["weight_bytes", node.weight_bytes],
    ["flops.prefill", node.flops.prefill],
    ["flops.decode", node.flops.decode],
    ["bytes_moved.prefill", node.bytes_moved.prefill],
    ["bytes_moved.decode", node.bytes_moved.decode],
  ];
  if (node.comm) {
    if (node.comm.when) rows.push(["comm.when", node.comm.when]);
    rows.push(["comm.logical.prefill", node.comm.logical_bytes.prefill]);
    rows.push(["comm.logical.decode", node.comm.logical_bytes.decode]);
    rows.push(["comm.bus.prefill", node.comm.bus_bytes.prefill]);
    rows.push(["comm.bus.decode", node.comm.bus_bytes.decode]);
  }
  return (
    <Collapsible>
      <Section
        title="Rule & formulas"
        aside={
          <CollapsibleTrigger className="group/c flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground">
            source text{" "}
            <ChevronDown className="size-3 transition-transform group-data-[state=open]/c:rotate-180" />
          </CollapsibleTrigger>
        }
      >
        <Row k="Rule" v={node.rule} />
        <CollapsibleContent>
          <div className="mt-2 space-y-2">
            {rows.map(([k, src]) => (
              <div key={k}>
                <div className="font-mono text-[10.5px] text-muted-foreground">{k}</div>
                <code className="block rounded bg-muted px-2 py-1 font-mono text-[11px] leading-relaxed break-all">
                  {src}
                </code>
              </div>
            ))}
          </div>
        </CollapsibleContent>
      </Section>
    </Collapsible>
  );
}

function NodeDetail({ node }: { node: IRNode }) {
  const m = useModel();
  const env = m.envState.env;
  const metrics = m.metrics.get(node.id);
  const both = (p: Phase) => nodeMetrics(node, env, p);
  const unsharded = tryEvaluate(node.weight_bytes, unshardedEnv(env));
  const drill = drillTarget(m.ir, m.view, node.id);
  const where = viewForNode(m.ir, m.view, node.id);
  const parent = node.parent ? irNode(m.ir, node.parent) : undefined;
  const inherited = inheritedParallel(m.ir, node);
  const parallel = Object.entries(inherited?.spec ?? {});
  const hasWork = node.kind !== "comm" && node.kind !== "io";
  return (
    <div data-testid="node-detail">
      <div className="border-b px-4 pt-3 pb-3">
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5">
              <h2 className="truncate text-[15px] font-semibold tracking-tight">{node.label}</h2>
              <span className="rounded bg-muted px-1 font-mono text-[10px] text-muted-foreground">
                {node.kind}
              </span>
            </div>
            <div className="mt-0.5 font-mono text-[11px] text-muted-foreground">
              {parent ? `${parent.id} › ` : ""}
              {node.id}
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Close details"
            onClick={() => m.select(undefined)}
          >
            <X />
          </Button>
        </div>
        <div className="mt-2 font-mono text-[12px] break-all">{node.class}</div>
        <div className="mt-1">
          <SrcLink src={node.src} />
        </div>
        <div className="mt-2 flex gap-1.5">
          {drill ? (
            <Button
              variant="outline"
              size="xs"
              onClick={() => m.goView(drill)}
              data-testid="drill-button"
            >
              <ArrowDownRight /> Open {drill.split(":")[0]}
            </Button>
          ) : null}
          {where && where.viewId !== m.viewId ? (
            <Button variant="outline" size="xs" onClick={() => m.goView(where.viewId, node.id)}>
              <Maximize2 /> Show in graph
            </Button>
          ) : null}
        </div>
      </div>

      <Section
        title="Tensors"
        aside={<span className="font-mono text-[10.5px] text-muted-foreground">{m.phase}</span>}
      >
        {node.in.length ? (
          <div className="mb-1 text-[10.5px] text-muted-foreground uppercase">in</div>
        ) : null}
        {node.in.map((t) => (
          <Tensor key={`in-${t.name}`} t={t} />
        ))}
        {node.out.length ? (
          <div className="mt-1.5 mb-1 text-[10.5px] text-muted-foreground uppercase">out</div>
        ) : null}
        {node.out.map((t) => (
          <Tensor key={`out-${t.name}`} t={t} />
        ))}
      </Section>

      {node.kind !== "comm" ? (
        <Section title="Parameters">
          <Row
            k="Resident on this rank"
            v={qText(metrics?.weight ?? { e: "?" }, fmtBytes)}
            strong
          />
          <Row k="Unsharded" v={unsharded.ok ? fmtBytes(unsharded.value) : "—"} />
          {parallel.length ? (
            <Row
              k={
                inherited && inherited.from !== node.id
                  ? `Sharding (via ${inherited.from})`
                  : "Sharding"
              }
              v={parallel.map(([axis, p]) => `${axis}: ${p.dim}`).join(" · ")}
            />
          ) : (
            <Row k="Sharding" v="replicated" />
          )}
          {node.kind === "group" ? (
            <p className="mt-1 text-[11px] text-muted-foreground">
              Group totals summarize descendants.
            </p>
          ) : null}
        </Section>
      ) : null}

      {hasWork ? (
        <Section
          title="Work"
          aside={
            <span className="grid grid-cols-2 gap-3 font-mono text-[10px] text-muted-foreground">
              <span>prefill</span>
              <span>decode</span>
            </span>
          }
        >
          <PhasePair label="FLOPs" get={(p) => both(p).flops} fmt={fmtFlops} />
          <PhasePair label="HBM bytes" get={(p) => both(p).bytes} fmt={fmtBytes} />
          <PhasePair
            label="Intensity"
            get={(p) => {
              const x = both(p);
              return x.intensity !== undefined ? { v: x.intensity } : { e: "n/a" };
            }}
            fmt={(x) => `${fmtNum(Number(x.toPrecision(3)))}`}
          />
        </Section>
      ) : null}

      {hasWork && metrics?.intensity !== undefined ? (
        <Section title={`Roofline · ${m.phase}`}>
          <IntensityBar ai={metrics.intensity} />
        </Section>
      ) : null}

      {node.comm ? (
        <Section title="Communication">
          <Row k="Collective" v={`${node.comm.op} over ${node.comm.group}`} strong />
          <Row
            k="Active"
            v={metrics?.comm?.active ? "yes" : "no — contributes zero traffic"}
            title={node.comm.when ?? undefined}
          />
          <div className="mt-1">
            <PhasePair
              label="Logical"
              get={(p) => both(p).comm?.logical ?? { e: "?" }}
              fmt={fmtBytes}
            />
            <PhasePair label="Bus" get={(p) => both(p).comm?.bus ?? { e: "?" }} fmt={fmtBytes} />
          </div>
          {node.comm.notes?.map((n) => (
            <p key={n} className="mt-1.5 text-[11px] text-muted-foreground">
              {n}
            </p>
          ))}
        </Section>
      ) : null}

      <AscendCard node={node} />
      <Formulas node={node} />

      {node.notes?.length ? (
        <Section title="Notes">
          <ul className="space-y-1 text-[12px] text-muted-foreground">
            {node.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </Section>
      ) : null}
    </div>
  );
}

function ModelSummary() {
  const m = useModel();
  const l = m.ledger;
  const ir = m.ir;
  const ctx = m.envState.env.context;
  const seqs =
    l.kvTokens.v !== undefined && typeof ctx === "number"
      ? Math.floor(l.kvTokens.v / Math.max(1, ctx + 1))
      : undefined;
  return (
    <div data-testid="model-summary">
      <div className="border-b px-4 py-3">
        <h2 className="text-[15px] font-semibold tracking-tight">{ir.snapshot.model.hf_id}</h2>
        <p className="mt-0.5 text-[12px] text-muted-foreground">
          {ir.views.family} family · rank-local values · select a node for details
        </p>
      </div>
      <Section title="Per-rank memory">
        <Row k="Weights" v={qText(l.weights, fmtBytes)} strong />
        <Row k="Activation peak" v={qText(l.activation, fmtBytes)} />
        <Row k="KV per token" v={qText(l.kvPerToken, fmtBytes)} />
        <Row
          k="KV capacity"
          v={l.kvTokens.v !== undefined ? `${fmtNum(l.kvTokens.v)} tok` : "—"}
          strong
        />
        {seqs !== undefined ? (
          <Row k={`Seqs @ ${fmtNum(ctx as number)} ctx`} v={fmtNum(seqs)} />
        ) : null}
      </Section>
      <Section title="Config symbols">
        <div className="grid grid-cols-3 gap-x-4">
          {Object.entries(m.envState.symbols).map(([k, v]) => (
            <Row key={k} k={<span className="font-mono">{k}</span>} v={fmtNum(v)} />
          ))}
        </div>
      </Section>
      {ir.memory.unmodelled.length ? (
        <Section title="Not modelled">
          <ul className="flex flex-wrap gap-1">
            {ir.memory.unmodelled.map((u) => (
              <li key={u} className="rounded border px-1.5 py-px text-[11px] text-muted-foreground">
                {u}
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
      {ir.notes?.length ? (
        <Section title="Accounting notes">
          <ul className="list-disc space-y-1 pl-4 text-[12px] text-muted-foreground">
            {ir.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </Section>
      ) : null}
    </div>
  );
}

export function DetailPanel({ className }: { className?: string }) {
  const m = useModel();
  const node = m.selected ? irNode(m.ir, m.selected) : undefined;
  return (
    <aside className={cn("text-[12.5px]", className)} data-testid="detail-panel">
      {node ? <NodeDetail node={node} /> : <ModelSummary />}
    </aside>
  );
}
