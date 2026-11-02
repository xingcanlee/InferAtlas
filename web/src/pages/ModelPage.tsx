import { getRouteApi } from "@tanstack/react-router";
import { ChevronLeft, ChevronRight, Monitor, Network, Workflow } from "lucide-react";
import { useState } from "react";
import { EmptyState } from "@/components/AppShell";
import { VerificationMark } from "@/components/CoverageBadge";
import { HomeLink } from "@/components/HomeLink";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useIsPhone } from "@/lib/responsive";
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
  const phone = useIsPhone();
  // The graph workspace needs a wide screen; on phones it opens only on request.
  const [graphAnyway, setGraphAnyway] = useState(false);

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
      <div className="flex h-10 shrink-0 items-center gap-3 border-b px-3 max-lg:h-12 max-md:gap-1 max-md:pr-[max(0.25rem,env(safe-area-inset-right))] max-md:pl-[max(0.25rem,env(safe-area-inset-left))]">
        <div className="flex min-w-0 items-center gap-2 max-md:gap-1.5">
          {phone ? (
            <HomeLink className="inline-flex size-10 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground">
              <ChevronLeft className="size-4" />
              <span className="sr-only">Back to the matrix</span>
            </HomeLink>
          ) : (
            <>
              <HomeLink className="text-muted-foreground hover:text-foreground" />
              <ChevronRight className="size-3.5 text-muted-foreground" />
            </>
          )}
          <h1
            className="truncate text-[15px] font-semibold tracking-tight max-md:text-[14px]"
            data-testid="model-title"
            title={hfId}
          >
            {hfId}
          </h1>
          <span className="shrink-0 rounded border px-1.5 font-mono text-[11px] text-muted-foreground">
            {cell.device}
          </span>
          {version ? (
            <span className="hidden truncate text-[12px] text-muted-foreground lg:inline">
              {version.label}
            </span>
          ) : null}
          <VerificationMark cell={cell} className="max-md:hidden" />
        </div>
        <TabsList
          variant="line"
          className="ml-auto h-8 shrink-0 max-lg:h-10"
          aria-label="Model views"
        >
          <TabsTrigger
            value="arch"
            className="px-2.5 text-[12.5px] max-lg:h-10 max-md:min-w-10 max-md:px-2"
            data-testid="tab-arch"
            aria-label={phone ? "Architecture" : undefined}
          >
            <Workflow className="size-3.5 max-md:size-4" />
            <span className="max-md:sr-only">Architecture</span>
          </TabsTrigger>
          <TabsTrigger
            value="graph"
            className="px-2.5 text-[12.5px] max-lg:h-10 max-md:min-w-10 max-md:px-2"
            data-testid="tab-graph"
            aria-label={phone ? "Graph & analysis" : undefined}
          >
            <Network className="size-3.5 max-md:size-4" />
            <span className="max-md:sr-only">Graph &amp; analysis</span>
          </TabsTrigger>
        </TabsList>
      </div>
      <TabsContent value="arch" className="min-h-0 flex-1">
        {arch ? (
          <ArchitectureView key={`${cell.version}/${cell.device}/${cell.model}`} arch={arch} />
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
        {bundle && phone && !graphAnyway ? (
          <EmptyState title="Graph & analysis needs a wider screen" icon={<Monitor />}>
            <p data-testid="graph-narrow">
              The node graph, memory ledger and roofline are laid out for screens at least 768 px
              wide. Rotate to landscape or use a tablet or desktop; the Architecture tab works on
              any screen.
            </p>
            <p className="mt-3 flex flex-wrap justify-center gap-2">
              <Button variant="outline" className="h-10" onClick={() => setTab("arch")}>
                <Workflow className="size-4" /> Architecture
              </Button>
              <Button
                variant="ghost"
                className="h-10"
                onClick={() => setGraphAnyway(true)}
                data-testid="graph-anyway"
              >
                Open anyway
              </Button>
            </p>
          </EmptyState>
        ) : bundle ? (
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
