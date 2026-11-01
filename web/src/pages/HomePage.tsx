import type { IndexCell, IndexModel } from "@inferatlas/contracts";
import { getRouteApi, Link } from "@tanstack/react-router";
import { ArrowRight, GitCompareArrows, Search } from "lucide-react";
import { EmptyState } from "@/components/AppShell";
import { StatusBadge } from "@/components/StatusBadge";
import { StatusCounts, StatusLegend } from "@/components/StatusSummary";
import { Input } from "@/components/ui/input";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { shortSha } from "@/lib/format";
import { cellStatusText, devicesOf } from "@/lib/status";
import { cn } from "@/lib/utils";

const route = getRouteApi("/");

/** Family display order and labels; unknown families follow alphabetically. */
const FAMILY_ORDER = ["dense", "moe", "mla", "hybrid", "vlm"];
const FAMILY_LABEL: Record<string, string> = {
  dense: "Dense",
  moe: "MoE",
  mla: "MLA",
  hybrid: "Hybrid",
  vlm: "Vision-language",
};
const familyLabel = (f: string) => FAMILY_LABEL[f] ?? f;
const familyRank = (f: string) => {
  const i = FAMILY_ORDER.indexOf(f);
  return i < 0 ? FAMILY_ORDER.length : i;
};

function Cell({ cell }: { cell: IndexCell | undefined }) {
  if (!cell) {
    return <span className="text-[12px] text-muted-foreground">—</span>;
  }
  const badge = <StatusBadge status={cell.status} count={cell.triage_count ?? undefined} />;
  if (!cell.data_path) {
    return (
      <span title={cellStatusText(cell).detail} data-testid={`cell-${cell.version}-${cell.model}`}>
        {badge}
      </span>
    );
  }
  // Structure first: every cell with data links to its model page, whatever its triage status.
  return (
    <Link
      to="/m/$version/$device/$model"
      params={{ version: cell.version, device: cell.device, model: cell.model }}
      className="group/cell -m-1.5 inline-flex items-center gap-1.5 rounded-md p-1.5 transition-colors hover:bg-accent"
      data-testid={`cell-${cell.version}-${cell.model}`}
    >
      {badge}
      <ArrowRight className="size-3.5 text-muted-foreground opacity-0 transition-opacity group-hover/cell:opacity-100" />
    </Link>
  );
}

const matches = (m: IndexModel, q: string) =>
  !q ||
  [m.id, m.label, m.hf_id, m.family, familyLabel(m.family)].some((s) =>
    s.toLowerCase().includes(q),
  );

export function HomePage() {
  const index = route.useLoaderData();
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const devices = devicesOf(index);
  const device =
    search.device && devices.includes(search.device)
      ? search.device
      : (index.cells.find((cell) => cell.status === "ok" && cell.data_path)?.device ?? devices[0]);

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
  /** The row's primary target: first version with data (index order prefers newest pairs). */
  const primaryCell = (m: string) =>
    index.versions.map((v) => cellOf(v.id, m)).find((c) => c?.data_path) as IndexCell | undefined;

  const q = (search.q ?? "").trim().toLowerCase();
  const families = [...new Set(index.models.map((m): string => m.family))].sort(
    (a, b) => familyRank(a) - familyRank(b) || a.localeCompare(b),
  );
  const family = search.family && families.includes(search.family) ? search.family : undefined;
  const shown = index.models.filter((m) => matches(m, q) && (!family || m.family === family));
  const groups = families
    .map((f) => ({ family: f, models: shown.filter((m) => m.family === f) }))
    .filter((g) => g.models.length);
  const setSearch = (patch: { q?: string | undefined; family?: string | undefined }) =>
    navigate({ search: (prev) => ({ ...prev, ...patch }), replace: true });
  const cols = index.versions.length + 2;

  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-6xl px-6 py-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-[20px] font-semibold tracking-tight">Models</h1>
            <p className="mt-1 max-w-2xl text-muted-foreground">
              Open a model to see its architecture: whole-model dataflow and every layer's tensor
              inputs and outputs. Each column pairs a vLLM commit with its vllm-ascend counterpart.
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
                v &&
                navigate({
                  search: (prev) => ({ ...prev, device: v === devices[0] ? undefined : v }),
                })
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

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <div className="relative">
            <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              aria-label="Filter models"
              placeholder="Filter models…"
              className="h-7 w-56 pl-7 text-[12px]"
              value={search.q ?? ""}
              onChange={(e) => setSearch({ q: e.target.value || undefined })}
              data-testid="model-filter"
            />
          </div>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            value={family ?? "all"}
            onValueChange={(v) => v && setSearch({ family: v === "all" ? undefined : v })}
            aria-label="Model family"
            data-testid="family-filter"
          >
            <ToggleGroupItem value="all" className="h-7 px-2.5 text-[12px]">
              All
            </ToggleGroupItem>
            {families.map((f) => (
              <ToggleGroupItem key={f} value={f} className="h-7 px-2.5 text-[12px]">
                {familyLabel(f)}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <div className="ml-auto">
            <StatusCounts cells={index.cells.filter((c) => c.device === device)} />
          </div>
        </div>

        <div className="mt-3 overflow-x-auto rounded-lg border">
          <table className="w-full border-collapse text-left" data-testid="matrix">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="px-4 py-2.5 align-bottom text-[12px] font-medium text-muted-foreground">
                  Model
                </th>
                {index.versions.map((v) => (
                  <th key={v.id} className="border-l px-4 py-2.5 align-bottom font-normal">
                    <div className="text-[12.5px] font-medium">{v.label}</div>
                    <div className="mt-0.5 font-mono text-[10.5px] text-muted-foreground">
                      vllm {shortSha(v.vllm)} · ascend {shortSha(v.ascend)}
                    </div>
                  </th>
                ))}
                <th className="w-16 border-l px-2 py-2.5" aria-label="Compare" />
              </tr>
            </thead>
            {groups.map((g) => (
              <tbody key={g.family} data-testid={`family-${g.family}`}>
                <tr className="border-b bg-muted/25">
                  <th
                    colSpan={cols}
                    scope="colgroup"
                    className="px-4 py-1.5 text-[11px] font-medium tracking-wide text-muted-foreground uppercase"
                  >
                    {familyLabel(g.family)}{" "}
                    <span className="font-mono normal-case">· {g.models.length}</span>
                  </th>
                </tr>
                {g.models.map((m, i) => {
                  const primary = primaryCell(m.id);
                  const withData = index.cells.filter(
                    (c) => c.model === m.id && c.device === device && c.data_path,
                  );
                  const name = (
                    <>
                      <span className="font-semibold">{m.label}</span>
                      <span className="ml-2 font-mono text-[11px] text-muted-foreground">
                        {m.hf_id}
                      </span>
                    </>
                  );
                  return (
                    <tr
                      key={m.id}
                      className={cn("group/row hover:bg-accent/40", i > 0 && "border-t")}
                      data-testid={`model-row-${m.id}`}
                    >
                      <td className="px-4 py-2 align-middle">
                        {primary ? (
                          <Link
                            to="/m/$version/$device/$model"
                            params={{
                              version: primary.version,
                              device: primary.device,
                              model: primary.model,
                            }}
                            className="inline-flex items-baseline rounded-sm hover:underline focus-visible:ring-2 focus-visible:ring-ring/70 focus-visible:outline-none"
                            data-testid={`model-link-${m.id}`}
                          >
                            {name}
                          </Link>
                        ) : (
                          <span className="text-muted-foreground">{name}</span>
                        )}
                      </td>
                      {index.versions.map((v) => (
                        <td key={v.id} className="border-l px-4 py-2 align-middle">
                          <Cell cell={cellOf(v.id, m.id)} />
                        </td>
                      ))}
                      <td className="border-l px-2 py-2 text-center align-middle">
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
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            ))}
          </table>
          {groups.length ? null : (
            <p
              className="px-4 py-6 text-center text-[12px] text-muted-foreground"
              data-testid="no-models"
            >
              No models match the filter.
            </p>
          )}
        </div>
        <StatusLegend className="mt-3 text-[11px] text-muted-foreground" />
      </div>
    </div>
  );
}
