import type { IRNode } from "@inferatlas/contracts";
import { AlertTriangle } from "lucide-react";
import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  LabelList,
  Line,
  ReferenceLine,
  Scatter,
  XAxis,
  YAxis,
} from "recharts";
import { EmptyState } from "@/components/AppShell";
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { fmtBytes, fmtFlops, fmtNum, fmtSI } from "@/lib/format";
import { kvSweep } from "@/lib/model";
import { SIDE_TABS, type SideTab } from "@/lib/search";
import { cn } from "@/lib/utils";
import { descendants, inheritedParallel, irNode } from "@/lib/views";
import { useModel } from "@/pages/model/context";

const TAB_LABEL: Record<SideTab, string> = {
  memory: "Memory waterfall",
  kv: "KV capacity",
  roofline: "Roofline",
  ranks: "Rank lanes",
};

function Stat({ label, value, sub }: { label: string; value: string; sub?: string | undefined }) {
  return (
    <div className="min-w-0">
      <div className="text-[11px] text-muted-foreground">{label}</div>
      <div className="font-mono text-[15px] font-medium tabular">{value}</div>
      {sub ? <div className="text-[10.5px] text-muted-foreground">{sub}</div> : null}
    </div>
  );
}

// ------------------------------------------------------------------ memory waterfall

const memConfig = {
  weights: { label: "Weights", color: "var(--chart-1)" },
  activation: { label: "Activation peak", color: "var(--chart-2)" },
  kv: { label: "KV cache", color: "var(--chart-3)" },
  other: { label: "Non-torch / reserved", color: "var(--chart-4)" },
  slack: { label: "Unused", color: "var(--chart-neutral)" },
  total: { label: "Total", color: "var(--muted-foreground)" },
} satisfies ChartConfig;

type MemKey = keyof typeof memConfig;

function MemoryWaterfall() {
  const m = useModel();
  const l = m.ledger;
  const w = l.weights.v ?? 0;
  const a = l.activation.v ?? 0;
  const o = (l.nonTorch.v ?? 0) + (l.reserved.v ?? 0);
  const kv = l.kvBytes.v ?? 0;
  const budget = l.budget.v;
  const hbm = l.hbm.v;
  if (budget === undefined || hbm === undefined) {
    return (
      <EmptyState title="Memory budget unavailable">This IR has no hbm_bytes parameter.</EmptyState>
    );
  }
  const slack = Math.max(0, budget - w - a - o - kv);
  const over = w + a + o - budget;
  const steps: { name: string; key: MemKey; base: number; value: number }[] = [];
  let acc = 0;
  const push = (name: string, key: MemKey, value: number) => {
    steps.push({ name, key, base: acc, value });
    acc += value;
  };
  push("Weights", "weights", w);
  push("Activation", "activation", a);
  if (o > 0) push("Non-torch", "other", o);
  push("KV cache", "kv", kv);
  if (slack > 0) push("Rounding", "slack", slack);
  steps.push({ name: "Budget", key: "total", base: 0, value: budget });
  steps.push({ name: "Headroom", key: "slack", base: budget, value: Math.max(0, hbm - budget) });
  steps.push({ name: "Device", key: "total", base: 0, value: hbm });
  const data = steps.map((s) => ({ ...s, label: fmtBytes(s.value) }));
  const util = m.envState.env.gpu_memory_utilization;
  return (
    <div className="grid h-full grid-cols-[minmax(0,1fr)_220px] gap-4">
      <div className="flex min-h-0 flex-col">
        {over > 0 ? (
          <div className="mb-2 flex items-center gap-2 rounded-md border border-bad/40 px-2 py-1 text-[12px]">
            <AlertTriangle className="size-3.5 text-bad" /> Weights + activation exceed the budget
            by <span className="font-mono">{fmtBytes(over)}</span>; no KV cache fits.
          </div>
        ) : null}
        <ChartContainer
          config={memConfig}
          className="aspect-auto min-h-0 flex-1"
          data-testid="memory-chart"
        >
          <BarChart
            data={data}
            margin={{ top: 20, right: 8, left: 8, bottom: 0 }}
            barCategoryGap={10}
          >
            <CartesianGrid vertical={false} strokeDasharray="2 4" />
            <XAxis dataKey="name" tickLine={false} axisLine={false} tickMargin={6} fontSize={11} />
            <YAxis
              tickLine={false}
              axisLine={false}
              width={64}
              fontSize={10.5}
              tickFormatter={(v: number) => fmtBytes(v)}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  hideIndicator
                  formatter={(_v, _n, item) => {
                    const p = item.payload as (typeof data)[number];
                    return (
                      <div className="flex w-full justify-between gap-4">
                        <span>{memConfig[p.key].label}</span>
                        <span className="font-mono">{fmtBytes(p.value)}</span>
                      </div>
                    );
                  }}
                />
              }
            />
            <Bar dataKey="base" stackId="w" fill="transparent" isAnimationActive={false} />
            <Bar dataKey="value" stackId="w" radius={[3, 3, 3, 3]} isAnimationActive={false}>
              {data.map((d) => (
                <Cell
                  key={d.name}
                  fill={d.key === "total" ? "var(--muted-foreground)" : `var(--color-${d.key})`}
                  fillOpacity={d.key === "total" ? 0.35 : 1}
                />
              ))}
              <LabelList
                dataKey="label"
                position="top"
                fontSize={10.5}
                className="fill-muted-foreground"
              />
            </Bar>
          </BarChart>
        </ChartContainer>
      </div>
      <div className="space-y-3 overflow-y-auto">
        <Stat
          label="Budget"
          value={fmtBytes(budget)}
          sub={`${fmtBytes(hbm)} × ${typeof util === "number" ? util : "?"} utilization`}
        />
        <Stat
          label="KV capacity"
          value={l.kvTokens.v !== undefined ? `${fmtNum(l.kvTokens.v)} tok` : "—"}
          sub={`${qOr(l.kvPerToken.v, fmtBytes)} per token`}
        />
        <div>
          <div className="mb-1 text-[11px] text-muted-foreground">Not modelled</div>
          <ul className="space-y-0.5 text-[11px] text-muted-foreground">
            {m.ir.memory.unmodelled.map((u) => (
              <li key={u}>· {u}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

const qOr = (v: number | undefined, f: (x: number) => string) => (v === undefined ? "—" : f(v));

// ------------------------------------------------------------------ KV grouped stacked bar

const kvConfig = {
  weights: { label: "Weights", color: "var(--chart-1)" },
  activation: { label: "Activation", color: "var(--chart-2)" },
  kv: { label: "KV cache", color: "var(--chart-3)" },
  other: { label: "Other", color: "var(--chart-4)" },
} satisfies ChartConfig;

function KvCapacity() {
  const m = useModel();
  const cur = m.envState.params.tp;
  // Memoised per parameter state: one ledger per TP choice.
  const data = useMemo(
    () => kvSweep(m.ir, m.envState).map((r) => ({ ...r, name: `tp=${r.tp}` })),
    [m.ir, m.envState],
  );
  const current = data.find((d) => d.tp === cur);
  const ctx = m.envState.env.context;
  const S = m.envState.env.S;
  const block = Number(m.envState.env.block_size ?? 0);
  const occupancy =
    current && typeof S === "number" && typeof ctx === "number" && current.tokens > 0
      ? (S * (ctx + 1)) / current.tokens
      : undefined;
  return (
    <div className="grid h-full grid-cols-[minmax(0,1fr)_220px] gap-4">
      <ChartContainer
        config={kvConfig}
        className="aspect-auto h-full min-h-0"
        data-testid="kv-chart"
      >
        <BarChart
          data={data}
          margin={{ top: 20, right: 8, left: 8, bottom: 0 }}
          barCategoryGap="28%"
        >
          <CartesianGrid vertical={false} strokeDasharray="2 4" />
          <XAxis dataKey="name" tickLine={false} axisLine={false} tickMargin={6} fontSize={11} />
          <YAxis
            tickLine={false}
            axisLine={false}
            width={64}
            fontSize={10.5}
            tickFormatter={(v: number) => fmtBytes(v)}
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                formatter={(v, n) => <TooltipRow name={String(n)} value={Number(v)} />}
              />
            }
          />
          {(["weights", "activation", "other", "kv"] as const).map((k, i, arr) => (
            <Bar
              key={k}
              dataKey={k}
              stackId="mem"
              fill={`var(--color-${k})`}
              isAnimationActive={false}
              stroke="var(--background)"
              strokeWidth={1}
              radius={i === arr.length - 1 ? [3, 3, 0, 0] : 0}
            >
              {data.map((d) => (
                <Cell key={d.name} fillOpacity={d.tp === cur ? 1 : 0.45} />
              ))}
              {k === "kv" ? (
                <LabelList
                  dataKey="tokens"
                  position="top"
                  fontSize={10.5}
                  className="fill-muted-foreground"
                  formatter={(v: unknown) => `${fmtSI(Number(v))} tok`}
                />
              ) : null}
            </Bar>
          ))}
        </BarChart>
      </ChartContainer>
      <div className="space-y-3">
        <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
          {Object.entries(kvConfig).map(([k, c]) => (
            <span key={k} className="inline-flex items-center gap-1">
              <span className="size-2 rounded-[2px]" style={{ background: c.color }} />
              {c.label}
            </span>
          ))}
        </div>
        <Stat
          label={`KV capacity at tp=${String(cur)}`}
          value={current ? `${fmtNum(current.tokens)} tok` : "—"}
          sub={
            block
              ? `${fmtNum(Math.floor((current?.tokens ?? 0) / block))} blocks of ${block}`
              : undefined
          }
        />
        <Stat
          label={`Sequences @ ${fmtNum(typeof ctx === "number" ? ctx : undefined)} context`}
          value={current ? fmtNum(current.seqs) : "—"}
        />
        {occupancy !== undefined ? (
          <Stat
            label={`Decode batch S=${String(S)} occupies`}
            value={`${(occupancy * 100).toFixed(1)}%`}
          />
        ) : null}
      </div>
    </div>
  );
}

function TooltipRow({ name, value }: { name: string; value: number }) {
  const label = (kvConfig as Record<string, { label: string }>)[name]?.label ?? name;
  return (
    <div className="flex w-full justify-between gap-4">
      <span>{label}</span>
      <span className="font-mono">{fmtBytes(value)}</span>
    </div>
  );
}

// ------------------------------------------------------------------ roofline

function leafCompute(nodes: IRNode[]): IRNode[] {
  const parents = new Set(nodes.map((n) => n.parent).filter(Boolean));
  return nodes.filter((n) => n.kind === "compute" && !parents.has(n.id));
}

const roofConfig = {
  roof: { label: "Roof", color: "var(--muted-foreground)" },
  ops: { label: "Operators", color: "var(--chart-1)" },
} satisfies ChartConfig;

function Roofline() {
  const m = useModel();
  const hw = m.hw;
  const dtype = String(m.envState.env.dtype_a ?? "bf16");
  const peak = hw?.peak_tflops[dtype] ?? null;
  const bw = hw?.bandwidth_gbps ?? null;
  if (!hw || !peak || !bw) {
    return (
      <EmptyState title="Roofline hidden">
        {hw ? hw.label : "No hardware preset"} has no confirmed {dtype} peak and memory bandwidth.
        Pick another preset in the parameter panel; unknown vendor figures are never guessed.
      </EmptyState>
    );
  }
  const P = peak * 1e12;
  const B = bw * 1e9;
  const ridge = P / B;
  const xs = [0.1, 0.3, 1, 3, 10, 30, 100, ridge, 300, 1000, 3000, 10000].sort((a, b) => a - b);
  const roof = xs.map((x) => ({ x, roof: Math.min(P, x * B) }));
  const points = leafCompute(m.ir.nodes)
    .map((n) => ({ n, mt: m.metrics.get(n.id) }))
    .filter(({ mt }) => mt?.intensity !== undefined && (mt.flops.v ?? 0) > 0)
    .map(({ n, mt }) => {
      const x = Math.min(Math.max(mt?.intensity ?? 0.1, 0.1), 10000);
      return { x, y: Math.min(P, x * B), id: n.id, label: n.label, flops: mt?.flops.v ?? 0 };
    });
  return (
    <div className="grid h-full grid-cols-[minmax(0,1fr)_220px] gap-4">
      <ChartContainer
        config={roofConfig}
        className="aspect-auto h-full min-h-0"
        data-testid="roofline-chart"
      >
        <ComposedChart margin={{ top: 12, right: 16, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="2 4" />
          <XAxis
            dataKey="x"
            type="number"
            scale="log"
            domain={[0.1, 10000]}
            ticks={[0.1, 1, 10, 100, 1000, 10000]}
            tickFormatter={(v: number) => fmtNum(v)}
            allowDataOverflow
            fontSize={10.5}
            tickLine={false}
            label={{
              value: "FLOP / byte",
              position: "insideBottomRight",
              offset: -2,
              fontSize: 10.5,
            }}
          />
          <YAxis
            type="number"
            scale="log"
            domain={[1e10, P * 2]}
            ticks={[1e10, 1e11, 1e12, 1e13, 1e14, 1e15].filter((t) => t <= P * 2)}
            allowDataOverflow
            tickFormatter={(v: number) => fmtSI(v, "FLOP/s")}
            width={84}
            fontSize={10.5}
            tickLine={false}
          />
          <ReferenceLine x={ridge} stroke="var(--edge-soft)" strokeDasharray="3 3" />
          <Line
            data={roof}
            dataKey="roof"
            stroke="var(--muted-foreground)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            type="linear"
          />
          <ChartTooltip
            cursor={false}
            content={
              <ChartTooltipContent
                hideLabel
                formatter={(_v, _n, item) => {
                  const p = item.payload as (typeof points)[number] & { roof?: number };
                  if (!p.id)
                    return <span className="font-mono">roof {fmtSI(p.roof, "FLOP/s")}</span>;
                  return (
                    <div className="space-y-0.5">
                      <div className="font-medium">{p.label}</div>
                      <div className="font-mono">
                        AI {fmtNum(Number(p.x.toPrecision(3)))} FLOP/B
                      </div>
                      <div className="font-mono">{fmtFlops(p.flops)}</div>
                    </div>
                  );
                }}
              />
            }
          />
          <Scatter
            data={points}
            dataKey="y"
            isAnimationActive={false}
            onClick={(p: { payload?: { id?: string } }) => p.payload?.id && m.select(p.payload.id)}
            shape={(props: { cx?: number; cy?: number; payload?: { id: string } }) => {
              const sel = props.payload?.id === m.selected;
              return (
                <circle
                  cx={props.cx}
                  cy={props.cy}
                  r={sel ? 6 : 4.5}
                  fill={sel ? "var(--focus)" : "var(--chart-1)"}
                  fillOpacity={sel ? 1 : 0.75}
                  stroke="var(--background)"
                  strokeWidth={2}
                  style={{ cursor: "pointer" }}
                />
              );
            }}
          />
        </ComposedChart>
      </ChartContainer>
      <div className="space-y-3">
        <Stat label="Device" value={hw.device} sub={hw.label} />
        <Stat
          label={`Peak ${dtype}`}
          value={fmtSI(P, "FLOP/s")}
          sub={`${fmtSI(B, "B/s")} memory bandwidth`}
        />
        <Stat
          label="Ridge point"
          value={`${fmtNum(Number(ridge.toFixed(0)))} FLOP/B`}
          sub={`${m.phase}; operators sit on the roof — bounds, not measurements`}
        />
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ rank lanes

function RankLanes() {
  const m = useModel();
  const env = m.envState.env;
  const tp = Number(env.tp ?? 1);
  const ep = Number(env.ep ?? 1);
  const dp = Number(env.dp ?? 1);
  const E = Number(env.E ?? 0);
  const axis = ep > 1 ? "ep" : "tp";
  const n = axis === "ep" ? ep : tp;
  const shown = Math.min(n, 8);
  if (!m.graph) return <EmptyState title="No view to slice" />;
  const cols = [...m.graph.nodes].sort((a, b) => a.x - b.x || a.y - b.y);
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="rank-lanes">
      <div className="mb-2 flex gap-4 text-[11px] text-muted-foreground">
        <span>
          lanes = <span className="font-mono text-foreground">{axis}</span> ranks ({n})
        </span>
        <span>
          tp <span className="font-mono text-foreground">{tp}</span> · dp{" "}
          <span className="font-mono text-foreground">{dp}</span> · ep{" "}
          <span className="font-mono text-foreground">{ep}</span>
        </span>
        <span>columns = nodes in this view; cells = this rank's slice and resident bytes</span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        <div
          className="grid gap-px rounded-md border bg-border"
          style={{ gridTemplateColumns: `48px repeat(${cols.length}, minmax(112px, 1fr))` }}
        >
          <div className="bg-muted/60" />
          {cols.map((c) => (
            <button
              type="button"
              key={c.id}
              onClick={() => m.select(c.ir_node)}
              className={cn(
                "truncate bg-muted/60 px-2 py-1 text-left text-[11px] font-medium hover:bg-accent",
                m.selected === c.ir_node && "text-focus",
              )}
            >
              {c.label}
            </button>
          ))}
          {Array.from({ length: shown }, (_, r) => (
            <RankRow
              m={m}
              key={`r${String(r)}`}
              r={r}
              n={n}
              axis={axis}
              cols={cols.map((c) => c.ir_node)}
              E={E}
            />
          ))}
        </div>
        {n > shown ? (
          <div className="mt-1 text-[11px] text-muted-foreground">
            … {n - shown} more ranks with identical slices
          </div>
        ) : null}
      </div>
    </div>
  );
}

/** Expert range owned by rank r of n (integer bounds; remainder spread over the first ranks). */
export function expertRange(r: number, n: number, E: number): [number, number] {
  const base = Math.floor(E / n);
  const extra = E % n;
  const lo = r * base + Math.min(r, extra);
  return [lo, lo + base + (r < extra ? 1 : 0) - 1];
}

function RankRow({
  m,
  r,
  n,
  axis,
  cols,
  E,
}: {
  m: ReturnType<typeof useModel>;
  r: number;
  n: number;
  axis: string;
  cols: string[];
  E: number;
}) {
  return (
    <>
      <div className="flex items-center bg-background px-2 font-mono text-[11px] text-muted-foreground">
        r{r}
      </div>
      {cols.map((id) => {
        const node = irNode(m.ir, id);
        if (!node) return <div key={id} className="bg-background" />;
        const mt = m.metrics.get(id);
        if (node.kind === "comm") {
          const active = mt?.comm?.active;
          return (
            <div
              key={id}
              className={cn(
                "relative flex items-center justify-center bg-background px-2 py-1.5 font-mono text-[10.5px]",
                active ? "text-foreground" : "text-muted-foreground",
              )}
            >
              <span className="absolute inset-y-0 left-1/2 border-l border-dashed border-edge" />
              <span className="relative bg-background px-1">
                {r === 0 ? `${node.comm?.op} ${active ? fmtBytes(mt?.comm?.bus.v) : "off"}` : "│"}
              </span>
            </div>
          );
        }
        const par = inheritedParallel(m.ir, node)?.spec ?? {};
        const ax = par[axis];
        let slice = "full copy";
        if (ax && ax.dim !== "replicated" && n > 1) {
          slice =
            ax.dim === "experts" && E
              ? `E ${expertRange(r, n, E).join("–")}`
              : `${ax.dim} ${r + 1}/${n}`;
        } else if (n === 1) {
          slice = "whole";
        } else if (node.kind === "group") {
          const parts = descendants(m.ir, node.id).filter((d) => {
            const dim = d.parallel?.[axis]?.dim;
            return dim && dim !== "replicated";
          });
          if (parts.length) slice = `${parts.length} sharded inside`;
        }
        return (
          <div key={id} className="bg-background px-2 py-1.5">
            <div
              className={cn(
                "font-mono text-[10.5px]",
                slice === "full copy" && "text-muted-foreground",
              )}
            >
              {slice}
            </div>
            <div className="font-mono text-[10px] text-muted-foreground">
              {fmtBytes(mt?.weight.v)}
            </div>
          </div>
        );
      })}
    </>
  );
}

// ------------------------------------------------------------------ tabs

export function AnalysisTabs({ className }: { className?: string }) {
  const m = useModel();
  return (
    <Tabs
      value={m.side}
      onValueChange={(v) => m.patch({ side: v === "memory" ? undefined : (v as SideTab) }, true)}
      className={cn("flex min-h-0 flex-col gap-0", className)}
    >
      <div className="flex h-10 shrink-0 items-center border-b px-3">
        <TabsList className="h-7">
          {SIDE_TABS.map((t) => (
            <TabsTrigger
              key={t}
              value={t}
              className="h-6 px-2.5 text-[12px]"
              data-testid={`tab-${t}`}
            >
              {TAB_LABEL[t]}
            </TabsTrigger>
          ))}
        </TabsList>
        <span className="ml-auto text-[11px] text-muted-foreground">per rank · {m.phase}</span>
      </div>
      <div className="min-h-0 flex-1 p-3">
        <TabsContent value="memory" className="h-full">
          <MemoryWaterfall />
        </TabsContent>
        <TabsContent value="kv" className="h-full">
          <KvCapacity />
        </TabsContent>
        <TabsContent value="roofline" className="h-full">
          <Roofline />
        </TabsContent>
        <TabsContent value="ranks" className="h-full">
          <RankLanes />
        </TabsContent>
      </div>
    </Tabs>
  );
}
