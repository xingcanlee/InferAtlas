import { Handle, type NodeProps, Position } from "@xyflow/react";
import { ChevronsDownUp } from "lucide-react";
import type { ReactNode } from "react";
import { fmtBytes, shortClass } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { GraphNode, GraphNodeData } from "./types";

function Handles() {
  return (
    <>
      <Handle type="target" position={Position.Left} isConnectable={false} />
      <Handle type="source" position={Position.Right} isConnectable={false} />
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

function HeatBar({ d }: { d: GraphNodeData }) {
  if (d.share === undefined) return null;
  return (
    <div
      className="absolute inset-x-2.5 bottom-2 flex items-center gap-1.5"
      title="Share of this view"
    >
      <div className="h-[3px] flex-1 overflow-hidden rounded-full bg-node-muted">
        <div
          className={cn("h-full rounded-full", d.selected ? "bg-focus" : "bg-heat")}
          style={{ width: `${Math.max(d.share > 0 ? 2 : 0, d.share * 100)}%` }}
        />
      </div>
      <span className="font-mono text-[9.5px] leading-none text-muted-foreground tabular">
        {d.heatText}
      </span>
    </div>
  );
}

/** Dashed KV-cache stub entering from above (design §3.3 "KV cache flow"). */
function KvStub({ text }: { text: string }) {
  return (
    <div className="pointer-events-none absolute -top-[54px] left-1/2 flex -translate-x-1/2 flex-col items-center">
      <span className="rounded-[3px] border border-dashed border-edge bg-canvas px-1 font-mono text-[9.5px] leading-[14px] whitespace-nowrap text-muted-foreground">
        KV {text}
      </span>
      <span className="h-[38px] border-l border-dashed border-edge" />
    </div>
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
      {d.kv ? <KvStub text={d.kv} /> : null}
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

export function ComputeNode({ data: d }: NodeProps<GraphNode>) {
  const isGroup = d.ir?.kind === "group";
  return (
    <Frame d={d}>
      {isGroup ? (
        <div
          className={cn(
            "absolute inset-0 translate-x-[4px] translate-y-[4px] rounded-[6px] border bg-node",
            borderClass(d),
          )}
        />
      ) : null}
      <div
        className={cn(
          "relative h-full w-full rounded-[6px] border bg-node px-2.5 pt-2 transition-colors",
          borderClass(d),
          (d.selected || d.diff) && "border-[1.5px] bg-focus-soft",
        )}
      >
        <div className="flex items-center gap-1">
          <span className="truncate text-[12.5px] leading-4 font-semibold">{d.vn.label}</span>
          {d.repeat ? (
            <span className="ml-auto shrink-0 font-mono text-[10px] text-muted-foreground">
              {d.repeat}
            </span>
          ) : null}
          {d.drillable && !d.repeat ? (
            <ChevronsDownUp
              className="ml-auto size-3 shrink-0 text-muted-foreground"
              aria-label="has sub-graph"
            />
          ) : null}
        </div>
        <div className="mt-0.5 truncate font-mono text-[10.5px] leading-[14px] text-muted-foreground">
          {d.ir ? shortClass(d.ir.class) : d.vn.ir_node}
        </div>
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
        <span className="truncate text-[12.5px] leading-4 font-semibold">{d.vn.label}</span>
        <span className="truncate font-mono text-[10.5px] leading-[14px] text-muted-foreground">
          {d.ir ? shortClass(d.ir.class) : "io"}
        </span>
      </div>
    </Frame>
  );
}

/** Collective: hexagon crossing a dashed rank-boundary line. */
export function CommNode({ data: d, width, height }: NodeProps<GraphNode>) {
  const w = width ?? d.vn.w;
  const h = height ?? d.vn.h;
  const c = 12;
  const active = d.metrics?.comm?.active ?? true;
  const stroke =
    d.selected || d.diff ? "var(--focus)" : d.related ? "var(--focus)" : "var(--node-border)";
  const comm = d.ir?.comm;
  return (
    <Frame d={d}>
      <svg
        className="pointer-events-none absolute -top-4 left-0 overflow-visible"
        width={w}
        height={h + 32}
        aria-hidden="true"
      >
        <line
          x1={w / 2}
          x2={w / 2}
          y1={0}
          y2={h + 32}
          stroke="var(--edge-soft)"
          strokeDasharray="3 3"
          strokeWidth={1}
        />
        <polygon
          points={`${c},16 ${w - c},16 ${w},${16 + h / 2} ${w - c},${16 + h} ${c},${16 + h} 0,${16 + h / 2}`}
          fill={
            d.selected || d.diff
              ? "color-mix(in oklch, var(--focus) 10%, var(--node))"
              : "var(--node)"
          }
          stroke={stroke}
          strokeOpacity={d.related && !d.selected ? 0.6 : 1}
          strokeWidth={d.selected ? 1.5 : 1}
          strokeDasharray={active ? undefined : "4 3"}
        />
      </svg>
      {d.lane ? (
        <span className="absolute -top-[18px] left-1/2 ml-1.5 font-mono text-[9px] leading-none text-muted-foreground">
          {d.lane}
        </span>
      ) : null}
      <div
        className={cn("relative flex h-full flex-col justify-center px-4", !active && "opacity-60")}
      >
        <span className="truncate text-center text-[12px] leading-4 font-semibold">
          {d.vn.label}
        </span>
        <span className="truncate text-center font-mono text-[10px] leading-[14px] text-muted-foreground">
          {comm ? `${comm.op} · ${comm.group}` : "comm"}
        </span>
        <span className="truncate text-center font-mono text-[9.5px] leading-[12px] text-muted-foreground tabular">
          {active ? `bus ${fmtBytes(d.metrics?.comm?.bus.v)}` : "inactive"}
        </span>
      </div>
    </Frame>
  );
}

export const nodeTypes = { compute: ComputeNode, comm: CommNode, io: IoNode };
