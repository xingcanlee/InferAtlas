import { Handle, type NodeProps, Position } from "@xyflow/react";
import { ChevronsDownUp } from "lucide-react";
import type { ReactNode } from "react";
import { fmtBytes, shortClass } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { GraphNode, GraphNodeData } from "./types";

function Handles() {
  return (
    <>
      <Handle type="target" position={Position.Top} isConnectable={false} />
      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </>
  );
}

function Badges({ d }: { d: GraphNodeData }) {
  const tags = d.overlay ? (d.ir?.ascend?.tag ?? []) : [];
  const diff = d.diff;
  if (!tags.length && !(d.overlay && d.hiddenTagged) && !diff) return null;
  return (
    <div className="pointer-events-none absolute -top-2 right-2 flex gap-0.5">
      {diff ? (
        <span
          className="rounded-[3px] border border-focus bg-background px-1 font-mono text-[9px] leading-[13px] font-semibold text-focus uppercase"
          data-testid="diff-badge"
        >
          {diff}
        </span>
      ) : null}
      {tags.map((t) => (
        <span
          key={t}
          className={cn(
            "rounded-[3px] border border-overlay/60 bg-background px-1 font-mono text-[9px] leading-[13px] font-semibold text-overlay",
            d.ir?.ascend?.level === "KERNEL" && "border-overlay/35",
          )}
          data-testid="ascend-badge"
        >
          {t}
        </span>
      ))}
      {d.overlay && d.hiddenTagged ? (
        <span
          className="rounded-[3px] border border-dashed border-overlay/60 bg-background px-1 font-mono text-[9px] leading-[13px] text-overlay"
          title={`${d.hiddenTagged} Ascend-tagged node(s) inside`}
        >
          +{d.hiddenTagged}
        </span>
      ) : null}
    </div>
  );
}

/** Thin share-of-view bar pinned to the bottom edge (design §3.3 "heat"). */
function HeatBar({ d }: { d: GraphNodeData }) {
  if (d.share === undefined) return null;
  return (
    <div className="absolute inset-x-2.5 bottom-[3px] h-[2px] overflow-hidden rounded-full bg-node-muted">
      <div
        className={cn("h-full rounded-full", d.selected ? "bg-focus" : "bg-heat")}
        style={{ width: `${Math.max(d.share > 0 ? 2 : 0, d.share * 100)}%` }}
      />
    </div>
  );
}

/** KV cache feeds this node (collapsed views); the explicit KV loop appears in detail views. */
function KvChip({ text }: { text: string }) {
  return (
    <span
      className="shrink-0 rounded-[3px] border border-dashed border-edge px-1 font-mono text-[9px] leading-[12px] text-muted-foreground"
      title={`KV cache ${text}`}
    >
      KV
    </span>
  );
}

function Frame({ d, children }: { d: GraphNodeData; children: ReactNode }) {
  const tagged = d.overlay && ((d.ir?.ascend?.tag?.length ?? 0) > 0 || d.hiddenTagged > 0);
  return (
    <div
      className={cn(
        "group/node relative h-full w-full",
        tagged && "rounded-[9px] outline-1 outline-offset-[3px] outline-overlay/70 outline-dashed",
        d.diff === "removed" && "opacity-60",
      )}
      data-testid={`node-${d.vn.ir_node}`}
      data-selected={d.selected || undefined}
    >
      {children}
      <Badges d={d} />
      <Handles />
    </div>
  );
}

function borderClass(d: GraphNodeData) {
  if (d.selected || d.diff) return "border-focus";
  if (d.related) return "border-focus/50";
  return "border-node-border group-hover/node:border-foreground/40";
}

/** Layout labels may carry extra lines (e.g. "Layers 0…L-1", band strips). */
function Lines({
  d,
  right,
  sub,
}: {
  d: GraphNodeData;
  right?: ReactNode;
  /** Appended to the first extra line (e.g. evaluated "×48" after "Layers 0…L-1"). */
  sub?: ReactNode;
}) {
  const [head, ...rest] = d.vn.label.split("\n");
  return (
    <>
      <div className="flex min-w-0 items-center gap-1">
        <span className="truncate text-[12.5px] leading-4 font-semibold">{head}</span>
        {right}
      </div>
      {rest.map((l, i) => (
        <div
          key={l}
          className="flex min-w-0 gap-1 font-mono text-[10px] leading-[13px] text-muted-foreground"
        >
          <span className="truncate">{l}</span>
          {i === 0 ? sub : null}
        </div>
      ))}
    </>
  );
}

export function ComputeNode({ data: d }: NodeProps<GraphNode>) {
  const isGroup = d.ir?.kind === "group";
  const extra = d.vn.label.split("\n").length - 1;
  const showClass = d.vn.h >= 40 + extra * 13;
  return (
    <Frame d={d}>
      {isGroup ? (
        <div
          className={cn(
            "absolute inset-0 translate-x-[3px] translate-y-[3px] rounded-[6px] border bg-node",
            borderClass(d),
          )}
        />
      ) : null}
      <div
        className={cn(
          "relative flex h-full w-full flex-col justify-center rounded-[6px] border bg-node px-2.5 pb-0.5 transition-colors",
          borderClass(d),
          (d.selected || d.diff) && "border-[1.5px] bg-focus-soft",
        )}
        title={d.ir ? `${d.ir.class}${d.heatText ? ` · ${d.heatText} of view` : ""}` : undefined}
      >
        <Lines
          d={d}
          right={
            d.repeat && !extra ? (
              <span className="ml-auto shrink-0 font-mono text-[10px] text-muted-foreground">
                {d.repeat}
              </span>
            ) : d.drillable ? (
              <ChevronsDownUp
                className="ml-auto size-3 shrink-0 text-muted-foreground"
                aria-label="has sub-graph"
              />
            ) : null
          }
          sub={d.repeat ? <span className="ml-auto shrink-0">{d.repeat}</span> : null}
        />
        {showClass ? (
          <div className="flex min-w-0 items-baseline gap-1.5 font-mono text-[10px] leading-[13px] text-muted-foreground">
            <span className="truncate">{d.ir ? shortClass(d.ir.class) : d.vn.ir_node}</span>
            {d.kv ? (
              <span className="ml-auto">
                <KvChip text={d.kv} />
              </span>
            ) : null}
          </div>
        ) : null}
        <HeatBar d={d} />
      </div>
    </Frame>
  );
}

export function IoNode({ data: d }: NodeProps<GraphNode>) {
  return (
    <Frame d={d}>
      <div
        className={cn(
          "flex h-full w-full flex-col justify-center rounded-[32px] border bg-node-muted px-4",
          borderClass(d),
          d.selected && "border-[1.5px]",
        )}
      >
        <Lines d={d} />
        <span className="truncate font-mono text-[10px] leading-[13px] text-muted-foreground">
          {d.ir ? shortClass(d.ir.class) : "io"}
        </span>
      </div>
    </Frame>
  );
}

/** Collective: hexagon crossing a dashed horizontal rank-boundary line. */
export function CommNode({ data: d, width, height }: NodeProps<GraphNode>) {
  const w = width ?? d.vn.w;
  const h = height ?? d.vn.h;
  const c = Math.min(12, h / 3);
  const pad = 14;
  const active = d.metrics?.comm?.active ?? true;
  const hot = d.selected || !!d.diff;
  const stroke = hot || d.related ? "var(--focus)" : "var(--node-border)";
  const comm = d.ir?.comm;
  return (
    <Frame d={d}>
      <svg
        className="pointer-events-none absolute top-0 overflow-visible"
        style={{ left: -pad }}
        width={w + 2 * pad}
        height={h}
        aria-hidden="true"
      >
        <line
          x1={0}
          x2={w + 2 * pad}
          y1={h / 2}
          y2={h / 2}
          stroke="var(--edge-soft)"
          strokeDasharray="3 3"
        />
        <polygon
          points={`${pad + c},0.5 ${pad + w - c},0.5 ${pad + w - 0.5},${h / 2} ${pad + w - c},${h - 0.5} ${pad + c},${h - 0.5} ${pad + 0.5},${h / 2}`}
          fill={hot ? "color-mix(in oklch, var(--focus) 10%, var(--node))" : "var(--node)"}
          stroke={stroke}
          strokeOpacity={d.related && !d.selected ? 0.6 : 1}
          strokeWidth={d.selected ? 1.5 : 1}
          strokeDasharray={active ? undefined : "4 3"}
          data-testid={active ? undefined : "comm-inactive"}
        />
      </svg>
      {d.lane ? (
        <span
          className="absolute -top-3 font-mono text-[9px] leading-none text-muted-foreground"
          style={{ right: -pad }}
        >
          {d.lane}
        </span>
      ) : null}
      <div className="relative flex h-full flex-col justify-center px-3 text-center">
        <span
          className={cn(
            "truncate text-[12px] leading-4 font-semibold",
            !active && "text-muted-foreground",
          )}
        >
          {d.vn.label.split("\n")[0]}
        </span>
        <span className="truncate font-mono text-[10px] leading-[13px] text-muted-foreground tabular">
          {comm ? `${comm.op} · ` : ""}
          {active ? fmtBytes(d.metrics?.comm?.bus.v) : "inactive"}
        </span>
      </div>
    </Frame>
  );
}

export const nodeTypes = { compute: ComputeNode, comm: CommNode, io: IoNode };
