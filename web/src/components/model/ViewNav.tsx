import { Fragment } from "react";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Kbd } from "@/components/ui/kbd";
import { tryEvaluate } from "@/lib/expr";
import { cn } from "@/lib/utils";
import { breadcrumb, getView, irNode, viewTitle } from "@/lib/views";
import { useModel } from "@/pages/model/context";

function BandStrip() {
  const m = useModel();
  const bands = m.ir.groups.map((g) => {
    const [a, b] = g.layers as [string, string];
    const lo = tryEvaluate(String(a), m.envState.env);
    const hi = tryEvaluate(String(b), m.envState.env);
    const n = lo.ok && hi.ok ? hi.value - lo.value + 1 : undefined;
    return { g, lo: lo.ok ? lo.value : undefined, hi: hi.ok ? hi.value : undefined, n };
  });
  if (!bands.length) return null;
  const total = bands.reduce((s, b) => s + (b.n ?? 1), 0);
  return (
    <div className="flex items-center gap-2" data-testid="band-strip">
      <span className="text-[11px] text-muted-foreground">Layers</span>
      <div className="flex h-6 w-72 overflow-hidden rounded-md border">
        {bands.map(({ g, lo, hi, n }) => {
          const id = `L1:${g.id}`;
          const active = m.viewId === id;
          const has = !!getView(m.view, id);
          const label = irNode(m.ir, g.template)?.label.replace(/\s*×\s*\w+$/, "") ?? g.id;
          return (
            <button
              type="button"
              key={g.id}
              disabled={!has}
              onClick={() => m.goView(id)}
              className={cn(
                "flex min-w-0 items-center justify-between gap-2 border-r px-2 text-[11px] last:border-r-0 transition-colors hover:bg-accent disabled:cursor-default",
                active && "bg-focus-soft text-focus",
              )}
              style={{ flexGrow: (n ?? 1) / total, flexBasis: 0 }}
              title={`${label}: layers ${lo}–${hi} (representative layer ${g.representative_layer})`}
            >
              <span className="truncate">{label}</span>
              <span className="font-mono text-muted-foreground">
                {lo}–{hi} ×{n}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function ViewNav() {
  const m = useModel();
  const chain = breadcrumb(m.ir, m.view, m.viewId);
  return (
    <div className="flex h-10 shrink-0 items-center gap-4 border-b px-3">
      <Breadcrumb>
        <BreadcrumbList className="gap-1 text-[12.5px] sm:gap-1">
          {chain.map((id, i) => {
            const last = i === chain.length - 1;
            const level = id.split(":")[0];
            return (
              <Fragment key={id}>
                {i > 0 ? <BreadcrumbSeparator /> : null}
                <BreadcrumbItem>
                  {last ? (
                    <BreadcrumbPage
                      className="flex items-center gap-1.5 font-medium"
                      data-testid="current-view"
                    >
                      <span className="rounded bg-muted px-1 font-mono text-[10px] text-muted-foreground">
                        {level}
                      </span>
                      {viewTitle(m.ir, id)}
                    </BreadcrumbPage>
                  ) : (
                    <button
                      type="button"
                      className="flex items-center gap-1.5 hover:text-foreground"
                      onClick={() => m.goView(id)}
                    >
                      <span className="rounded bg-muted px-1 font-mono text-[10px]">{level}</span>
                      {viewTitle(m.ir, id)}
                    </button>
                  )}
                </BreadcrumbItem>
              </Fragment>
            );
          })}
        </BreadcrumbList>
      </Breadcrumb>
      <div className="ml-auto flex items-center gap-4">
        <BandStrip />
        <span className="hidden items-center gap-1 text-[11px] text-muted-foreground xl:flex">
          double-click to open · <Kbd>Esc</Kbd> up · <Kbd>/</Kbd> find
        </span>
      </div>
    </div>
  );
}
