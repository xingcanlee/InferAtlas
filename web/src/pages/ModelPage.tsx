import { getRouteApi } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo } from "react";
import { EmptyState } from "@/components/AppShell";
import { GraphCanvas } from "@/components/graph/GraphCanvas";
import { Legend } from "@/components/graph/Legend";
import { AnalysisTabs } from "@/components/model/AnalysisTabs";
import { DetailPanel } from "@/components/model/DetailPanel";
import { ModelToolbar } from "@/components/model/ModelToolbar";
import { NodeSearch } from "@/components/model/NodeSearch";
import { ParamsPanel } from "@/components/model/ParamsPanel";
import { ViewNav } from "@/components/model/ViewNav";
import {
  buildEnv,
  computeAll,
  memoryLedger,
  type ParamValue,
  runtimeAnnotations,
} from "@/lib/model";
import { paramOverrides } from "@/lib/search";
import { useUi } from "@/lib/store";
import { bandAnchor, descendants, drillTarget, getView, L0, parentView } from "@/lib/views";
import { ModelContext, type ModelCtx } from "./model/context";

const route = getRouteApi("/m/$version/$device/$model");

function isTyping(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return !!t && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));
}

export function ModelPage() {
  const { bundle, cell, hardware } = route.useLoaderData();
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const { ir: sourceIr, view, report } = bundle;
  const searchOpen = useUi((s) => s.searchOpen);
  const setSearchOpen = useUi((s) => s.setSearchOpen);

  const overrides = paramOverrides(search);
  const overridesKey = JSON.stringify(overrides);
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed by serialized overrides
  const envState = useMemo(() => buildEnv(sourceIr, overrides), [sourceIr, overridesKey]);
  const ir = useMemo(() => runtimeAnnotations(sourceIr, envState.env), [sourceIr, envState]);
  const phase = search.phase ?? "prefill";
  const metrics = useMemo(() => computeAll(ir, envState.env, phase), [ir, envState, phase]);
  const ledger = useMemo(() => memoryLedger(ir, envState.env), [ir, envState]);

  const viewId = search.view && getView(view, search.view) ? search.view : L0;
  const graph = getView(view, viewId);
  const selected =
    search.node && ir.nodes.some((n) => n.id === search.node) ? search.node : undefined;
  const hw =
    hardware.find((h) => h.id === search.hw) ??
    hardware.find((h) => h.device === ir.snapshot.device);

  const patch = useCallback<ModelCtx["patch"]>(
    (p, replace = false) => navigate({ search: (prev) => ({ ...prev, ...p }), replace }),
    [navigate],
  );
  const setParam = useCallback<ModelCtx["setParam"]>(
    (name, value) => {
      const norm = (k: string, x: ParamValue | undefined) =>
        x === envState.defaults[k] ? undefined : x;
      const next: Record<string, ParamValue | undefined> = { [name]: norm(name, value) };
      // Keep enabled EP on the flattened tp×dp group when tp/dp change (vLLM rule).
      const ep = envState.params.ep;
      if ((name === "tp" || name === "dp") && typeof ep === "number" && ep > 1) {
        const tp =
          name === "tp" ? Number(value ?? envState.defaults.tp) : Number(envState.params.tp);
        const dp =
          name === "dp" ? Number(value ?? envState.defaults.dp) : Number(envState.params.dp);
        const choices = ir.params.ep?.choices;
        const want = tp * dp;
        next.ep = norm("ep", !choices || choices.includes(String(want)) ? want : 1);
      }
      patch(next, true);
    },
    [envState.defaults, envState.params, ir.params, patch],
  );
  const select = useCallback((id: string | undefined) => patch({ node: id }), [patch]);
  const goView = useCallback(
    (v: string, node?: string) => patch({ view: v === L0 ? undefined : v, node }),
    [patch],
  );

  // Fidelity ledger: represented scope minus what is drawn.
  const merged = useMemo(() => {
    const drawn = new Set(graph?.nodes.map((n) => n.ir_node));
    return (graph?.collapsed_of ?? []).filter((id) => !drawn.has(id));
  }, [graph]);

  // Highlight the node itself, or the visible node that contains it.
  const highlight = useMemo(() => {
    if (!selected || !graph) return undefined;
    const visible = new Set(graph.nodes.map((n) => n.ir_node));
    if (visible.has(selected)) return selected;
    return [...visible].find((v) => descendants(ir, v).some((d) => d.id === selected));
  }, [graph, ir, selected]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (searchOpen || isTyping(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "/") {
        e.preventDefault();
        setSearchOpen(true);
      } else if (e.key === "Escape") {
        const up = parentView(ir, view, viewId);
        // Going up keeps focus on the node we came out of.
        const from = viewId.startsWith("L2:")
          ? viewId.slice(3)
          : viewId.startsWith("L1:")
            ? bandAnchor(ir, view, viewId.slice(3))
            : undefined;
        if (up) goView(up, from);
        else if (selected) select(undefined);
      } else if (e.key === "Enter" && selected) {
        const t = drillTarget(ir, view, selected);
        if (t) goView(t);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goView, ir, searchOpen, select, selected, setSearchOpen, view, viewId]);

  const overlay = search.overlay ?? false;
  const shapes = search.shapes ?? "num";
  const heat = search.heat ?? "flops";
  const side = search.side ?? "memory";
  const ctx = useMemo<ModelCtx>(
    () => ({
      ir,
      view,
      report,
      cell,
      hardware,
      hw,
      envState,
      metrics,
      ledger,
      phase,
      viewId,
      graph,
      selected,
      overlay,
      shapes,
      heat,
      side,
      search,
      patch,
      setParam,
      select,
      goView,
    }),
    [
      ir,
      view,
      report,
      cell,
      hardware,
      hw,
      envState,
      metrics,
      ledger,
      phase,
      viewId,
      graph,
      selected,
      overlay,
      shapes,
      heat,
      side,
      search,
      patch,
      setParam,
      select,
      goView,
    ],
  );

  return (
    <ModelContext.Provider value={ctx}>
      <div className="grid h-full min-h-0 grid-cols-[264px_minmax(0,1fr)_352px] grid-rows-[auto_minmax(0,1fr)]">
        <ModelToolbar className="col-span-3" />
        <ParamsPanel className="min-h-0 overflow-y-auto border-r" />
        <section className="flex min-h-0 min-w-0 flex-col">
          <ViewNav />
          <div className="min-h-[280px] flex-1 bg-canvas">
            {graph ? (
              <GraphCanvas
                ir={ir}
                view={view}
                graph={graph}
                metrics={metrics}
                env={envState.env}
                phase={phase}
                shapes={ctx.shapes}
                heat={ctx.heat}
                overlay={ctx.overlay}
                selected={highlight}
                onSelect={select}
                onDrill={(id) => {
                  const t = drillTarget(ir, view, id);
                  if (t) goView(t);
                }}
              />
            ) : (
              <EmptyState title="No diagram for this snapshot">
                <code>view.json</code> is missing, so there is no precomputed layout to draw.
                Metrics, memory and the parameter panel still work from <code>ir.json</code>.
              </EmptyState>
            )}
          </div>
          <Legend heat={heat} overlay={overlay} collapsed={merged} />
          <AnalysisTabs className="h-[300px] shrink-0 border-t" />
        </section>
        <DetailPanel className="min-h-0 overflow-y-auto border-l" />
      </div>
      <NodeSearch />
    </ModelContext.Provider>
  );
}
