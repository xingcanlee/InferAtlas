import type { Index, IndexCell } from "@inferatlas/contracts";
import { getRouteApi, Link } from "@tanstack/react-router";
import { ArrowRight, GitCompareArrows } from "lucide-react";
import { EmptyState } from "@/components/AppShell";
import { StatusBadge } from "@/components/StatusBadge";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { shortSha } from "@/lib/format";
import { cn } from "@/lib/utils";

const route = getRouteApi("/");

export function devicesOf(index: Index): string[] {
  return [...new Set(index.cells.map((c) => c.device))].sort();
}

function Cell({ cell }: { cell: IndexCell | undefined }) {
  if (!cell) {
    return <span className="text-[12px] text-muted-foreground">—</span>;
  }
  const badge = <StatusBadge status={cell.status} count={cell.triage_count ?? undefined} />;
  if (!cell.data_path) return badge;
  return (
    <Link
      to="/m/$version/$device/$model"
      params={{ version: cell.version, device: cell.device, model: cell.model }}
      className="group/cell -m-2 flex items-center justify-between gap-2 rounded-md p-2 transition-colors hover:bg-accent"
      data-testid={`cell-${cell.version}-${cell.model}`}
    >
      {badge}
      <ArrowRight className="size-3.5 text-muted-foreground opacity-0 transition-opacity group-hover/cell:opacity-100" />
    </Link>
  );
}

export function HomePage() {
  const index = route.useLoaderData();
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const devices = devicesOf(index);
  const device = search.device && devices.includes(search.device) ? search.device : devices[0];

  if (!index.models.length || !index.versions.length) {
    return (
      <EmptyState title="No snapshots published yet">
        The pipeline has not produced any version × model cells. Run <code>just build</code> to
        generate one.
      </EmptyState>
    );
  }

  const cellOf = (v: string, m: string) =>
    index.cells.find((c) => c.version === v && c.model === m && c.device === device);
  const counts = { ok: 0, triage: 0, missing: 0 };
  for (const c of index.cells) if (c.device === device) counts[c.status]++;

  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-6xl px-6 py-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-[20px] font-semibold tracking-tight">Version × model matrix</h1>
            <p className="mt-1 max-w-2xl text-muted-foreground">
              Every published snapshot pairs a vLLM commit with its vllm-ascend counterpart. Open a
              cell to explore the rank-local graph, memory ledger and Ascend differences.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[12px] text-muted-foreground">Device</span>
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={device}
              onValueChange={(v) =>
                v && navigate({ search: { device: v === devices[0] ? undefined : v } })
              }
            >
              {devices.map((d) => (
                <ToggleGroupItem key={d} value={d} className="h-7 px-3 font-mono text-[12px]">
                  {d}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>
        </div>

        <div className="mt-5 flex gap-4 text-[12px] text-muted-foreground">
          <span>
            <span className="font-mono text-foreground tabular">{counts.ok}</span> publishable
          </span>
          <span>
            <span className="font-mono text-foreground tabular">{counts.triage}</span> need triage
          </span>
          <span>
            <span className="font-mono text-foreground tabular">{counts.missing}</span> missing
          </span>
        </div>

        <div className="mt-3 overflow-x-auto rounded-lg border">
          <table className="w-full border-collapse text-left" data-testid="matrix">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="w-72 px-4 py-3 align-bottom text-[12px] font-medium text-muted-foreground">
                  Version pair
                </th>
                {index.models.map((m) => {
                  const withData = index.cells.filter(
                    (c) => c.model === m.id && c.device === device && c.data_path,
                  );
                  return (
                    <th key={m.id} className="min-w-52 border-l px-4 py-3 align-bottom font-normal">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">{m.label}</span>
                        <span className="rounded border px-1 font-mono text-[10px] text-muted-foreground uppercase">
                          {m.family}
                        </span>
                      </div>
                      <div className="mt-0.5 flex items-center justify-between gap-2">
                        <span className="font-mono text-[11px] text-muted-foreground">
                          {m.hf_id}
                        </span>
                        {withData.length >= 2 ? (
                          <Link
                            to="/diff"
                            search={{
                              model: m.id,
                              device,
                              a: withData[0]?.version,
                              b: withData[withData.length - 1]?.version,
                            }}
                            className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
                            title="Compare versions"
                          >
                            <GitCompareArrows className="size-3" /> diff
                          </Link>
                        ) : null}
                      </div>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {index.versions.map((v, i) => (
                <tr key={v.id} className={cn(i > 0 && "border-t")}>
                  <td className="px-4 py-3 align-top">
                    <div className="font-medium">{v.label}</div>
                    <div className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                      vllm {shortSha(v.vllm)} · ascend {shortSha(v.ascend)}
                    </div>
                  </td>
                  {index.models.map((m) => (
                    <td key={m.id} className="border-l px-4 py-3 align-middle">
                      <Cell cell={cellOf(v.id, m.id)} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-[11px] text-muted-foreground">
          <span className="font-medium">ok</span> passed verification and is publishable ·{" "}
          <span className="font-medium">triage</span> has unresolved items (count) ·{" "}
          <span className="font-medium">missing</span> has no artifacts for this pair.
        </p>
      </div>
    </div>
  );
}
