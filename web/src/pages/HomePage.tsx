import type { IndexCell, IndexModel } from "@inferatlas/contracts";
import { getRouteApi, Link } from "@tanstack/react-router";
import { ArrowRight, GitCompareArrows, Search } from "lucide-react";
import { EmptyState } from "@/components/AppShell";
import { CoverageBadge, VerificationMark, WarnCount } from "@/components/CoverageBadge";
import { StatusBadge } from "@/components/StatusBadge";
import { Input } from "@/components/ui/input";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cellCoverage } from "@/lib/coverage";
import { shortSha } from "@/lib/format";
import { useIsNarrow, useIsPhone } from "@/lib/responsive";
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

function Cell({
  cell,
  touch = false,
  wrap = false,
}: {
  cell: IndexCell | undefined;
  /** ≥ 40px tall tap target. */
  touch?: boolean;
  /** Card layout: badges may wrap. */
  wrap?: boolean;
}) {
  if (!cell) {
    return <span className="text-[12px] text-muted-foreground">—</span>;
  }
  if (!cell.data_path) {
    return (
      <span
        title={cellStatusText(cell).detail}
        className={cn(touch && "inline-flex min-h-10 items-center")}
        data-testid={`cell-${cell.version}-${cell.model}`}
      >
        <StatusBadge status={cell.status} count={cell.triage_count ?? undefined} />
      </span>
    );
  }
  // Structure first (v1.4 §1): the cell leads with structure coverage; numeric / annotation
  // verification is a secondary mark. Every cell with data links to its model page.
  const coverage = cellCoverage(cell);
  return (
    <Link
      to="/m/$version/$device/$model"
      params={{ version: cell.version, device: cell.device, model: cell.model }}
      className={cn(
        "group/cell -m-1.5 inline-flex items-center gap-x-2 gap-y-1 rounded-md p-1.5 whitespace-nowrap transition-colors hover:bg-accent",
        touch && "min-h-10",
        wrap && "flex-wrap whitespace-normal",
      )}
      data-testid={`cell-${cell.version}-${cell.model}`}
    >
      <CoverageBadge coverage={coverage} />
      {coverage ? <WarnCount count={coverage.notes_warn} /> : null}
      <VerificationMark cell={cell} />
      <ArrowRight className="size-3.5 text-muted-foreground opacity-0 transition-opacity group-hover/cell:opacity-100 max-md:opacity-60" />
    </Link>
  );
}

/** "N with structure · N numerically verified · N missing" above the matrix. */
function CoverageCounts({ cells }: { cells: IndexCell[] }) {
  const structure = cells.filter((c) => c.data_path && cellCoverage(c)?.arch !== false).length;
  const verified = cells.filter((c) => c.status === "ok" && c.data_path).length;
  const missing = cells.filter((c) => !c.data_path).length;
  const item = (n: number, text: string) => (
    <span className="whitespace-nowrap">
      <span className="font-mono text-foreground tabular">{n}</span> {text}
    </span>
  );
  return (
    <div
      className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-muted-foreground"
      data-testid="status-counts"
    >
      {item(structure, "with structure")}
      {item(verified, "numerically verified")}
      {item(missing, "missing")}
    </div>
  );
}

function Legend() {
  return (
    <p className="mt-3 text-[11px] text-muted-foreground" data-testid="coverage-legend">
      <span className="font-medium">structure ✓ · R/N ruled</span> every sublayer has an
      architecture rule · <span className="font-medium">G generic</span> sublayers drawn as generic
      components · <span className="font-medium">⚠ n</span> warning notes ·{" "}
      <span className="font-medium">verified / unverified</span> numeric memory &amp; annotation
      checks (secondary; never hides structure).
    </p>
  );
}

const matches = (m: IndexModel, q: string) =>
  !q ||
  [m.id, m.label, m.hf_id, m.family, familyLabel(m.family)].some((s) =>
    s.toLowerCase().includes(q),
  );

export function HomePage() {
  const index = route.useLoaderData();
  const phone = useIsPhone();
  // Touch-size controls below `lg` (tablets); card layout only below `md` (phones).
  const touch = useIsNarrow();
  const search = route.useSearch();
  const navigate = route.useNavigate();
  // Devices with only `missing` placeholder cells (nothing generated yet) are not offered.
  const all = devicesOf(index);
  const builtDevices = all.filter((d) =>
    index.cells.some((c) => c.device === d && c.status !== "missing"),
  );
  const devices = builtDevices.length ? builtDevices : all;
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
  // v1.4 §1: a version pair without any built cell on this device is not a column.
  const built = index.versions.filter((v) =>
    index.cells.some((c) => c.version === v.id && c.device === device && c.data_path),
  );
  const versions = built.length ? built : index.versions;
  /** The row's primary target: first version with data (index order prefers newest pairs). */
  const primaryCell = (m: string) =>
    versions.map((v) => cellOf(v.id, m)).find((c) => c?.data_path) as IndexCell | undefined;

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
  const withDataOf = (m: string) =>
    index.cells.filter((c) => c.model === m && c.device === device && c.data_path);
  // The compare column only when some model has two snapshots with data on this device.
  const canDiff = index.models.some((m) => withDataOf(m.id).length >= 2);
  const cols = versions.length + (canDiff ? 2 : 1);

  const modelName = (m: IndexModel) => (
    <>
      <span className="font-semibold">{m.label}</span>
      <span className="ml-2 font-mono text-[11px] text-muted-foreground max-md:ml-0 max-md:block max-md:[overflow-wrap:anywhere]">
        {m.hf_id}
      </span>
    </>
  );
  const modelLink = (m: IndexModel) => {
    const primary = primaryCell(m.id);
    return primary ? (
      <Link
        to="/m/$version/$device/$model"
        params={{ version: primary.version, device: primary.device, model: primary.model }}
        className="inline-flex items-baseline rounded-sm hover:underline focus-visible:ring-2 focus-visible:ring-ring/70 focus-visible:outline-none max-md:block max-md:min-h-10 max-md:py-0.5"
        data-testid={`model-link-${m.id}`}
      >
        {modelName(m)}
      </Link>
    ) : (
      <span className="text-muted-foreground max-md:block">{modelName(m)}</span>
    );
  };
  const diffLink = (m: IndexModel) => {
    const withData = withDataOf(m.id);
    return withData.length >= 2 ? (
      <Link
        to="/diff"
        search={{
          model: m.id,
          device,
          a: withData[0]?.version,
          b: withData[withData.length - 1]?.version,
        }}
        className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground max-md:min-h-10"
        title="Compare versions"
      >
        <GitCompareArrows className="size-3" /> diff
      </Link>
    ) : null;
  };
  const toggleItem = cn("h-7 px-2.5 text-[12px]", touch && "h-10 px-3 text-[13px]");

  /** Phones: one card per model, versions stacked inside (no sideways scrolling). */
  const cards = (
    <div className="mt-3 flex flex-col gap-4" data-testid="matrix-cards">
      {groups.map((g) => (
        <section key={g.family} data-testid={`family-${g.family}`}>
          <h2 className="mb-1.5 text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
            {familyLabel(g.family)}{" "}
            <span className="font-mono normal-case">· {g.models.length}</span>
          </h2>
          <ul className="flex flex-col gap-2">
            {g.models.map((m) => (
              <li
                key={m.id}
                className="rounded-lg border bg-card px-3 py-2"
                data-testid={`model-row-${m.id}`}
              >
                {modelLink(m)}
                <ul className="mt-1 flex flex-col">
                  {versions.map((v) => (
                    <li
                      key={v.id}
                      className="flex flex-col gap-0.5 border-t py-1.5 first:border-t-0"
                    >
                      {versions.length > 1 ? (
                        <span className="font-mono text-[11px] text-muted-foreground">
                          {v.label}
                        </span>
                      ) : null}
                      <span>
                        <Cell cell={cellOf(v.id, m.id)} touch wrap />
                      </span>
                    </li>
                  ))}
                </ul>
                {diffLink(m)}
              </li>
            ))}
          </ul>
        </section>
      ))}
      {groups.length ? null : (
        <p className="py-6 text-center text-[12px] text-muted-foreground" data-testid="no-models">
          No models match the filter.
        </p>
      )}
    </div>
  );

  const table = (
    <div className="mt-3 overflow-x-auto rounded-lg border">
      <table className="w-full border-collapse text-left" data-testid="matrix">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="px-4 py-2.5 align-bottom text-[12px] font-medium text-muted-foreground">
              Model
            </th>
            {versions.map((v) => (
              <th key={v.id} className="border-l px-4 py-2.5 align-bottom font-normal">
                <div className="text-[12.5px] font-medium">{v.label}</div>
                <div className="mt-0.5 font-mono text-[10.5px] text-muted-foreground">
                  vllm {shortSha(v.vllm)} · ascend {shortSha(v.ascend)}
                </div>
              </th>
            ))}
            {canDiff ? <th className="w-16 border-l px-2 py-2.5" aria-label="Compare" /> : null}
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
            {g.models.map((m, i) => (
              <tr
                key={m.id}
                className={cn("group/row hover:bg-accent/40", i > 0 && "border-t")}
                data-testid={`model-row-${m.id}`}
              >
                <td className="px-4 py-2 align-middle">{modelLink(m)}</td>
                {versions.map((v) => (
                  <td key={v.id} className="border-l px-4 py-2 align-middle">
                    <Cell cell={cellOf(v.id, m.id)} touch={touch} />
                  </td>
                ))}
                {canDiff ? (
                  <td className="border-l px-2 py-2 text-center align-middle">{diffLink(m)}</td>
                ) : null}
              </tr>
            ))}
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
  );

  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-6xl px-6 py-8 max-md:pr-[max(1rem,env(safe-area-inset-right))] max-md:pl-[max(1rem,env(safe-area-inset-left))] max-md:pt-5 max-md:pb-[max(2rem,env(safe-area-inset-bottom))]">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-[20px] font-semibold tracking-tight">Models</h1>
            <p className="mt-1 max-w-2xl text-muted-foreground">
              Open a model to see its architecture: whole-model dataflow and every layer's tensor
              inputs and outputs. Each column pairs a vLLM commit with its vllm-ascend counterpart.
            </p>
          </div>
          <div className="flex max-w-full flex-wrap items-center gap-2">
            <span className="text-[12px] text-muted-foreground">Device</span>
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={device}
              aria-label="Device"
              className="max-w-full"
              onValueChange={(v) =>
                v &&
                navigate({
                  search: (prev) => ({ ...prev, device: v === devices[0] ? undefined : v }),
                })
              }
              data-testid="device-switch"
            >
              {devices.map((d) => (
                <ToggleGroupItem
                  key={d}
                  value={d}
                  className={cn("h-7 px-3 font-mono text-[12px]", touch && "h-10 min-w-12")}
                >
                  {d}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <div className="relative max-md:w-full">
            <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              aria-label="Filter models"
              placeholder="Filter models…"
              className={cn(
                "h-7 w-56 pl-7 text-[12px]",
                // ≥ 16px on phones so iOS does not zoom the page on focus.
                touch && "h-10 text-[16px]",
                phone && "w-full",
              )}
              value={search.q ?? ""}
              onChange={(e) => setSearch({ q: e.target.value || undefined })}
              data-testid="model-filter"
            />
          </div>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            spacing={phone ? 1 : 0}
            value={family ?? "all"}
            onValueChange={(v) => v && setSearch({ family: v === "all" ? undefined : v })}
            aria-label="Model family"
            className={cn(phone && "w-full flex-wrap")}
            data-testid="family-filter"
          >
            <ToggleGroupItem value="all" className={toggleItem}>
              All
            </ToggleGroupItem>
            {families.map((f) => (
              <ToggleGroupItem key={f} value={f} className={toggleItem}>
                {familyLabel(f)}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <div className="ml-auto max-md:ml-0">
            <CoverageCounts
              cells={index.cells.filter(
                (c) => c.device === device && versions.some((v) => v.id === c.version),
              )}
            />
          </div>
        </div>

        {phone ? cards : table}
        <Legend />
      </div>
    </div>
  );
}
