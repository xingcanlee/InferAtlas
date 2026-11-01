import type { IndexCell } from "@inferatlas/contracts";
import { ArrowUpRight, FileCode2, Moon, Sun } from "lucide-react";
import { useEffect } from "react";
import { Logo } from "@/components/Logo";
import { StatusBadge } from "@/components/StatusBadge";
import { StatusCounts, StatusLegend } from "@/components/StatusSummary";
import { Button } from "@/components/ui/button";
import type { InlineData } from "@/lib/data";
import { shortSha } from "@/lib/format";
import { hubLinks, linkFor } from "@/lib/hub";
import { cellStatusText, devicesOf } from "@/lib/status";
import { applyTheme, useUi } from "@/lib/store";
import { cn } from "@/lib/utils";

function ThemeToggle() {
  const theme = useUi((s) => s.theme);
  const setTheme = useUi((s) => s.setTheme);
  useEffect(() => applyTheme(theme), [theme]);
  const next = theme === "dark" ? "light" : "dark";
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      className="size-7"
      aria-label="Toggle theme"
      title={`${next === "dark" ? "Dark" : "Light"} theme`}
      onClick={() => setTheme(next)}
      data-testid="theme-toggle"
    >
      {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </Button>
  );
}

function HubCell({ cell, href }: { cell: IndexCell | undefined; href: string | undefined }) {
  if (!cell) return <span className="text-[12px] text-muted-foreground">—</span>;
  const badge = <StatusBadge status={cell.status} count={cell.triage_count ?? undefined} />;
  const id = `hub-${cell.version}-${cell.model}-${cell.device}`;
  if (href) {
    // A plain relative link to the sibling export: works from file:// with no fetch.
    return (
      <a
        href={href}
        className="group/cell -m-2 flex flex-col gap-1.5 rounded-md p-2 transition-colors hover:bg-accent"
        data-testid={id}
      >
        <span className="flex items-center justify-between gap-2">
          {badge}
          <span className="inline-flex items-center gap-0.5 text-[12px] font-medium text-foreground">
            Open
            <ArrowUpRight className="size-3.5 text-muted-foreground transition-colors group-hover/cell:text-foreground" />
          </span>
        </span>
        <span className="flex min-w-0 items-center gap-1 font-mono text-[10.5px] text-muted-foreground">
          <FileCode2 className="size-3 shrink-0" />
          <span className="truncate" title={href.slice(2)}>
            {href.slice(2)}
          </span>
        </span>
      </a>
    );
  }
  const detail = cellStatusText(cell).detail;
  if (cell.status === "missing") {
    // The legend explains "missing"; repeating it in every cell is noise.
    return (
      <div title={detail} data-testid={id}>
        {badge}
      </div>
    );
  }
  return (
    <div className="flex flex-col items-start gap-1.5" data-testid={id}>
      {badge}
      <span className="text-[11px] leading-snug text-muted-foreground">
        {cell.data_path ? "No offline export in this release." : detail}
      </span>
    </div>
  );
}

/** Landing page for a release `dist/`: every model × device, linking to its single-file export. */
export function HubPage({ data }: { data: InlineData | undefined }) {
  if (!data) {
    return (
      <div className="p-8 text-center text-muted-foreground">
        This hub page was built without data.
      </div>
    );
  }
  const { index, hardware = [] } = data;
  const links = hubLinks(data.hub);
  const devices = devicesOf(index);
  const hwLabel = (d: string) => hardware.find((h) => h.device === d)?.label;
  const exported = index.cells.filter((c) => linkFor(links, c)).length;
  const cellOf = (v: string, m: string, d: string) =>
    index.cells.find((c) => c.version === v && c.model === m && c.device === d);
  // Versions with offline pages first, then those with any results, then the empty ones.
  const rank = (v: string) => {
    const cells = index.cells.filter((c) => c.version === v);
    if (cells.some((c) => linkFor(links, c))) return 0;
    return cells.some((c) => c.status !== "missing") ? 1 : 2;
  };
  const versions = [...index.versions].sort((a, b) => rank(a.id) - rank(b.id));

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex h-11 shrink-0 items-center gap-3 border-b px-3">
        <span className="flex items-center gap-2 font-semibold tracking-tight">
          <Logo />
          <span className="text-[14px]">InferAtlas</span>
        </span>
        <span className="h-4 w-px bg-border" />
        <span className="text-[13px] text-muted-foreground">Offline exports</span>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </header>
      <main className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto max-w-6xl px-6 py-8">
          <h1 className="text-[20px] font-semibold tracking-tight">All models</h1>
          <p className="mt-1 max-w-2xl text-muted-foreground">
            Each available cell is a self-contained HTML page that works offline: the rank-local
            graph, memory ledger, KV cache, roofline and Ascend differences for one model on one
            device. Cells without a page show why there is none.
          </p>
          <div className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-1">
            <StatusCounts cells={index.cells} />
            <span className="text-[12px] text-muted-foreground">
              <span className="font-mono text-foreground tabular">{exported}</span> offline pages
            </span>
          </div>

          {versions.map((v) => (
            <section key={v.id} className="mt-6" data-testid={`hub-version-${v.id}`}>
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
                <h2 className="font-medium">{v.label}</h2>
                <span className="font-mono text-[11px] text-muted-foreground">
                  vllm {shortSha(v.vllm)} · ascend {shortSha(v.ascend)}
                </span>
              </div>
              {rank(v.id) === 2 ? (
                <p className="mt-2 flex items-center gap-2 rounded-lg border border-dashed px-4 py-3 text-[12px] text-muted-foreground">
                  <StatusBadge status="missing" />
                  Nothing has been generated for this version pair yet.
                </p>
              ) : (
                <div className="mt-2 overflow-x-auto rounded-lg border">
                  <table
                    className="w-full min-w-[720px] table-fixed border-collapse text-left"
                    data-testid="hub-matrix"
                  >
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="w-56 px-4 py-2.5 align-bottom text-[12px] font-medium text-muted-foreground">
                          Model
                        </th>
                        {devices.map((d) => (
                          <th key={d} className="border-l px-4 py-2.5 align-bottom font-normal">
                            <div className="font-mono text-[12px] font-medium">{d}</div>
                            {hwLabel(d) ? (
                              <div
                                className="mt-0.5 truncate text-[11px] text-muted-foreground"
                                title={hwLabel(d)}
                              >
                                {hwLabel(d)}
                              </div>
                            ) : null}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {index.models.map((m, i) => (
                        <tr key={m.id} className={cn(i > 0 && "border-t")}>
                          <td className="px-4 py-3 align-top">
                            <div className="flex items-center gap-2">
                              <span className="font-semibold">{m.label}</span>
                              <span className="rounded border px-1 font-mono text-[10px] text-muted-foreground uppercase">
                                {m.family}
                              </span>
                            </div>
                            <div className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                              {m.hf_id}
                            </div>
                          </td>
                          {devices.map((d) => {
                            const cell = cellOf(v.id, m.id, d);
                            return (
                              <td key={d} className="border-l px-4 py-3 align-top">
                                <HubCell cell={cell} href={cell && linkFor(links, cell)} />
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          ))}
          <StatusLegend className="mt-4 text-[11px] text-muted-foreground" />
        </div>
      </main>
    </div>
  );
}
