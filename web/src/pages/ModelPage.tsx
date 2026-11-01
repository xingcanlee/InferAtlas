import { getRouteApi } from "@tanstack/react-router";
import { ChevronRight, Network, Workflow } from "lucide-react";
import { EmptyState } from "@/components/AppShell";
import { HomeLink } from "@/components/HomeLink";
import { StatusBadge } from "@/components/StatusBadge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { type ModelTab, modelTab } from "@/lib/search";
import { ArchitectureView } from "./model/ArchitectureView";
import { GraphWorkspace } from "./model/GraphWorkspace";

const route = getRouteApi("/m/$version/$device/$model");

/**
 * Model page: the Architecture tab (structure first: whole-model dataflow + per-layer tensor
 * I/O) is the default; the graph / analysis workspace is the secondary tab. Triage status is
 * shown as a badge and never hides either.
 */
export function ModelPage() {
  const { bundle, arch, archError, cell, index, hardware } = route.useLoaderData();
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const tab = modelTab(search);
  const hfId =
    arch?.model.hf_id ??
    bundle?.ir.snapshot.model.hf_id ??
    index.models.find((m) => m.id === cell.model)?.hf_id ??
    cell.model;
  const version = index.versions.find((v) => v.id === cell.version);

  const setTab = (t: ModelTab) =>
    navigate({
      search: (prev) => ({
        ...prev,
        tab: t === modelTab({ ...prev, tab: undefined }) ? undefined : t,
      }),
    });

  return (
    <Tabs
      value={tab}
      onValueChange={(v) => setTab(v as ModelTab)}
      className="flex h-full min-h-0 flex-col gap-0"
    >
      <div className="flex h-10 shrink-0 items-center gap-3 border-b px-3">
        <div className="flex min-w-0 items-center gap-2">
          <HomeLink className="text-muted-foreground hover:text-foreground" />
          <ChevronRight className="size-3.5 text-muted-foreground" />
          <h1
            className="truncate text-[15px] font-semibold tracking-tight"
            data-testid="model-title"
          >
            {hfId}
          </h1>
          <span className="rounded border px-1.5 font-mono text-[11px] text-muted-foreground">
            {cell.device}
          </span>
          {version ? (
            <span className="hidden truncate text-[12px] text-muted-foreground lg:inline">
              {version.label}
            </span>
          ) : null}
          {cell.status === "triage" ? (
            <StatusBadge status="triage" count={cell.triage_count ?? undefined} />
          ) : null}
        </div>
        <TabsList variant="line" className="ml-auto h-8" aria-label="Model views">
          <TabsTrigger value="arch" className="px-2.5 text-[12.5px]" data-testid="tab-arch">
            <Workflow className="size-3.5" />
            Architecture
          </TabsTrigger>
          <TabsTrigger value="graph" className="px-2.5 text-[12.5px]" data-testid="tab-graph">
            <Network className="size-3.5" />
            Graph &amp; analysis
          </TabsTrigger>
        </TabsList>
      </div>
      <TabsContent value="arch" className="min-h-0 flex-1">
        {arch ? (
          <ArchitectureView arch={arch} />
        ) : archError ? (
          <EmptyState title="Could not read the architecture data" icon={<Workflow />}>
            <p data-testid="arch-error" className="font-mono">
              {archError}
            </p>
            <p className="mt-3">
              <button
                type="button"
                className="underline underline-offset-2"
                onClick={() => setTab("graph")}
              >
                Open graph &amp; analysis
              </button>
            </p>
          </EmptyState>
        ) : (
          <EmptyState title="No architecture data for this snapshot yet" icon={<Workflow />}>
            <p data-testid="arch-empty">
              This snapshot was published before the generator emitted <code>arch.json</code>, so
              the structure view is not available. The graph and analysis tab still works from{" "}
              <code>ir.json</code>.
            </p>
            <p className="mt-3">
              <button
                type="button"
                className="underline underline-offset-2"
                onClick={() => setTab("graph")}
              >
                Open graph &amp; analysis
              </button>
            </p>
          </EmptyState>
        )}
      </TabsContent>
      <TabsContent value="graph" className="min-h-0 flex-1">
        {bundle ? (
          <GraphWorkspace bundle={bundle} cell={cell} hardware={hardware} />
        ) : (
          <EmptyState title="No graph data for this snapshot" icon={<Network />}>
            <p data-testid="graph-empty">
              Only the structure (<code>arch.json</code>) was published for this cell; the numeric
              graph, memory and roofline need <code>ir.json</code>.
            </p>
          </EmptyState>
        )}
      </TabsContent>
    </Tabs>
  );
}
