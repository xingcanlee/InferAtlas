import type { ReactNode } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Heat } from "@/lib/search";

function Item({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      {icon}
      {children}
    </span>
  );
}

const HEAT_TEXT: Record<Heat, string> = {
  flops: "FLOPs",
  weight: "weights",
  bytes: "HBM bytes",
};

export function Legend({
  heat,
  overlay,
  collapsed,
}: {
  heat: Heat;
  overlay: boolean;
  collapsed: string[];
}) {
  return (
    <div
      className="flex min-h-9 flex-wrap items-center gap-x-4 gap-y-1 border-t bg-background px-3 py-1.5 text-[11px] text-muted-foreground"
      data-testid="legend"
    >
      <Item icon={<span className="h-3 w-4 rounded-[3px] border border-node-border bg-node" />}>
        compute
      </Item>
      <Item
        icon={
          <span className="relative h-3 w-4">
            <span className="absolute inset-0 translate-x-[2px] translate-y-[2px] rounded-[3px] border border-node-border bg-node" />
            <span className="absolute inset-0 rounded-[3px] border border-node-border bg-node" />
          </span>
        }
      >
        group · drill in
      </Item>
      <Item
        icon={
          <svg width="18" height="12" aria-hidden="true">
            <line x1="9" x2="9" y1="0" y2="12" stroke="var(--edge-soft)" strokeDasharray="2 2" />
            <polygon
              points="4,1 14,1 17,6 14,11 4,11 1,6"
              fill="var(--node)"
              stroke="var(--node-border)"
            />
          </svg>
        }
      >
        collective · rank boundary
      </Item>
      <Item
        icon={<span className="h-3 w-4 rounded-full border border-node-border bg-node-muted" />}
      >
        io
      </Item>
      <span className="h-3 w-px bg-border" />
      <Item
        icon={
          <svg width="22" height="8" aria-hidden="true">
            <line x1="0" x2="22" y1="4" y2="4" stroke="var(--edge)" strokeWidth="1.25" />
          </svg>
        }
      >
        tensor <span className="font-mono">[shape]</span>
      </Item>
      <Item
        icon={
          <svg width="22" height="8" aria-hidden="true">
            <line
              x1="0"
              x2="22"
              y1="4"
              y2="4"
              stroke="var(--edge-soft)"
              strokeWidth="1.25"
              strokeDasharray="2 3"
            />
          </svg>
        }
      >
        residual
      </Item>
      <Item
        icon={
          <svg width="22" height="8" aria-hidden="true">
            <line
              x1="0"
              x2="22"
              y1="4"
              y2="4"
              stroke="var(--edge)"
              strokeWidth="1.25"
              strokeDasharray="5 4"
            />
          </svg>
        }
      >
        KV cache
      </Item>
      <Item
        icon={
          <span className="h-[3px] w-5 overflow-hidden rounded-full bg-node-muted">
            <span className="block h-full w-2/5 bg-heat" />
          </span>
        }
      >
        share of view {HEAT_TEXT[heat]}
      </Item>
      {overlay ? (
        <Item
          icon={
            <span className="rounded-[3px] border border-overlay/60 px-1 font-mono text-[9px] font-semibold text-overlay">
              OOT
            </span>
          }
        >
          Ascend replaced / patched
        </Item>
      ) : null}
      {collapsed.length ? (
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="ml-auto cursor-help underline decoration-dotted underline-offset-2">
              {collapsed.length} merged into this view
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">
            <div className="mb-1 font-medium">Fidelity ledger — not drawn as separate nodes:</div>
            <div className="font-mono text-[11px]">{collapsed.join(", ")}</div>
          </TooltipContent>
        </Tooltip>
      ) : null}
    </div>
  );
}
