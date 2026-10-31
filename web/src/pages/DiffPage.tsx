import type { Index } from "@inferatlas/contracts";
import { getRouteApi } from "@tanstack/react-router";
import { ArrowLeftRight, GitCompareArrows, Loader2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { EmptyState } from "@/components/AppShell";
import { GraphCanvas } from "@/components/graph/GraphCanvas";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { type Bundle, loadBundle } from "@/lib/data";
import { diffIR, type IrDiff, sideMarks, viewChangeCount } from "@/lib/diff";
import { buildEnv, computeAll, type Phase } from "@/lib/model";
import { cn } from "@/lib/utils";
import { getView, L0, viewTitle } from "@/lib/views";

const route = getRouteApi("/diff");

function useBundle(dataPath: string | null | undefined) {
  const [state, setState] = useState<{ path?: string; bundle?: Bundle; error?: string }>({});
  useEffect(() => {
    if (!dataPath) return;
    let live = true;
    loadBundle(dataPath).then(
      (bundle) => live && setState({ path: dataPath, bundle }),
      (e: Error) => live && setState({ path: dataPath, error: e.message }),
    );
    return () => {
      live = false;
    };
  }, [dataPath]);
  return state.path === dataPath ? state : {};
}

function Picker({
  label,
  value,
  items,
  onChange,
  testId,
}: {
  label: string;
  value: string | undefined;
  items: [string, string][];
  onChange: (v: string) => void;
  testId?: string;
}) {
  return (
    <div className="flex items-center gap-2 text-[12px] text-muted-foreground">
      {label}
      <Select value={value ?? ""} onValueChange={onChange}>
        <SelectTrigger
          size="sm"
          className="h-7 min-w-36 text-[12px] text-foreground"
          data-testid={testId}
        >
          <SelectValue placeholder="—" />
        </SelectTrigger>
        <SelectContent>
          {items.map(([v, t]) => (
            <SelectItem key={v} value={v} className="text-[12px]">
              {t}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

function cellsFor(index: Index, model: string | undefined, device: string | undefined) {
  return index.cells.filter((c) => c.model === model && c.device === device && c.data_path);
}

export function DiffPage() {
  const index = route.useLoaderData();
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const patch = (p: Partial<typeof search>) => navigate({ search: (prev) => ({ ...prev, ...p }) });

  // Default to the first model that has two versions with data.
  const candidates = index.models.filter((m) =>
    index.cells.some(
      (c) => c.model === m.id && c.data_path && cellsFor(index, m.id, c.device).length >= 2,
    ),
  );
  const model = search.model ?? candidates[0]?.id ?? index.models[0]?.id;
  const devices = [
    ...new Set(index.cells.filter((c) => c.model === model && c.data_path).map((c) => c.device)),
  ];
  const device = search.device ?? devices[0];
  const cells = cellsFor(index, model, device);
  const versionLabel = (id: string) => index.versions.find((v) => v.id === id)?.label ?? id;
  const a = search.a ?? cells[0]?.version;
  const b = search.b ?? cells[cells.length - 1]?.version;
  const cellA = cells.find((c) => c.version === a);
  const cellB = cells.find((c) => c.version === b);
  const A = useBundle(cellA?.data_path);
  const B = useBundle(cellB?.data_path);
  const phase: Phase = search.phase ?? "prefill";

  const header = (
    <div className="flex flex-wrap items-center gap-3 border-b px-3 py-2">
      <h1 className="mr-2 flex items-center gap-2 text-[15px] font-semibold tracking-tight">
        <GitCompareArrows className="size-4" /> Version diff
      </h1>
      <Picker
        label="Model"
        value={model}
        items={index.models.map((m) => [m.id, m.label])}
        onChange={(v) =>
          patch({ model: v, a: undefined, b: undefined, view: undefined, node: undefined })
        }
      />
      <Picker
        label="Device"
        value={device}
        items={devices.map((d) => [d, d])}
        onChange={(v) => patch({ device: v })}
      />
      <Picker
        label="A"
        testId="diff-a"
        value={a}
        items={cells.map((c) => [c.version, versionLabel(c.version)])}
        onChange={(v) => patch({ a: v })}
      />
      <Button
        variant="ghost"
        size="icon-sm"
        className="size-7"
        aria-label="Swap"
        onClick={() => patch({ a: b, b: a })}
      >
        <ArrowLeftRight className="size-3.5" />
      </Button>
      <Picker
        label="B"
        testId="diff-b"
        value={b}
        items={cells.map((c) => [c.version, versionLabel(c.version)])}
        onChange={(v) => patch({ b: v })}
      />
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        className="ml-auto"
        value={phase}
        onValueChange={(v) => v && patch({ phase: v === "prefill" ? undefined : (v as Phase) })}
      >
        <ToggleGroupItem value="prefill" className="h-7 px-2.5 text-[12px]">
          Prefill
        </ToggleGroupItem>
        <ToggleGroupItem value="decode" className="h-7 px-2.5 text-[12px]">
          Decode
        </ToggleGroupItem>
      </ToggleGroup>
    </div>
  );

  let body: React.ReactNode;
  if (cells.length < 2) {
    body = (
      <EmptyState title="Nothing to compare" icon={<GitCompareArrows className="size-5" />}>
        This model has fewer than two published snapshots on {device ?? "any device"}.
      </EmptyState>
    );
  } else if (A.error || B.error) {
    body = <EmptyState title="Could not load a snapshot">{A.error ?? B.error}</EmptyState>;
  } else if (!A.bundle || !B.bundle) {
    body = (
      <div className="flex h-full items-center justify-center gap-2 text-muted-foreground">
        <Loader2 className="size-4 animate-spin" /> Loading snapshots…
      </div>
    );
  } else {
    body = (
      <DiffBody
        a={A.bundle}
        b={B.bundle}
        labelA={versionLabel(a ?? "")}
        labelB={versionLabel(b ?? "")}
        phase={phase}
        viewId={search.view}
        selected={search.node}
        onView={(v) => patch({ view: v === L0 ? undefined : v })}
        onSelect={(n) => patch({ node: n })}
      />
    );
  }
  return (
    <div className="flex h-full min-h-0 flex-col">
      {header}
      <div className="min-h-0 flex-1">{body}</div>
    </div>
  );
}

function DiffBody({
  a,
  b,
  labelA,
  labelB,
  phase,
  viewId,
  selected,
  onView,
  onSelect,
}: {
  a: Bundle;
  b: Bundle;
  labelA: string;
  labelB: string;
  phase: Phase;
  viewId: string | undefined;
  selected: string | undefined;
  onView: (v: string) => void;
  onSelect: (n: string | undefined) => void;
}) {
  const diff = useMemo(() => diffIR(a.ir, b.ir), [a.ir, b.ir]);
  const envA = useMemo(() => buildEnv(a.ir).env, [a.ir]);
  const envB = useMemo(() => buildEnv(b.ir).env, [b.ir]);
  const mA = useMemo(() => computeAll(a.ir, envA, phase), [a.ir, envA, phase]);
  const mB = useMemo(() => computeAll(b.ir, envB, phase), [b.ir, envB, phase]);
  const viewIds = [
    ...new Set([...(a.view?.views ?? []), ...(b.view?.views ?? [])].map((v) => v.id)),
  ].sort();
  const vid = viewId && viewIds.includes(viewId) ? viewId : L0;
  const ga = getView(a.view, vid);
  const gb = getView(b.view, vid);
  const marksA = useMemo(() => sideMarks(diff, ga, "a"), [diff, ga]);
  const marksB = useMemo(() => sideMarks(diff, gb, "b"), [diff, gb]);

  const pane = (bundle: Bundle, g: typeof ga, label: string, side: "a" | "b") => (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex h-8 shrink-0 items-center gap-2 border-b px-3 text-[12px]">
        <span className="rounded bg-muted px-1 font-mono text-[10px] text-muted-foreground uppercase">
          {side}
        </span>
        <span className="font-medium">{label}</span>
        <span className="font-mono text-[11px] text-muted-foreground">
          vllm {bundle.ir.snapshot.vllm.tag}@{bundle.ir.snapshot.vllm.commit.slice(0, 7)}
        </span>
      </div>
      <div className="min-h-0 flex-1 bg-canvas">
        {g ? (
          <GraphCanvas
            testId={`diff-graph-${side}`}
            ir={bundle.ir}
            view={bundle.view}
            graph={g}
            metrics={side === "a" ? mA : mB}
            env={side === "a" ? envA : envB}
            phase={phase}
            shapes="num"
            minFitZoom={0.7}
            heat="flops"
            overlay={true}
            selected={selected}
            nodeDiff={(side === "a" ? marksA : marksB).nodes}
            changedEdges={(side === "a" ? marksA : marksB).edges}
            onSelect={onSelect}
            onDrill={(id) => {
              const target = bundle.ir.groups.find((x) => x.template === id);
              onView(target ? `L1:${target.id}` : `L2:${id}`);
            }}
          />
        ) : (
          <EmptyState title="View absent in this version" />
        )}
      </div>
    </div>
  );

  return (
    <div className="grid h-full min-h-0 grid-rows-[auto_minmax(0,1fr)_minmax(0,0.9fr)]">
      <div className="flex h-10 items-center gap-1 border-b px-3">
        {viewIds.map((id) => {
          const n = viewChangeCount(diff, getView(a.view, id), getView(b.view, id));
          return (
            <button
              type="button"
              key={id}
              onClick={() => onView(id)}
              className={cn(
                "inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-[12px] text-muted-foreground hover:bg-accent hover:text-foreground",
                id === vid && "bg-accent text-foreground",
              )}
              data-testid={`diff-view-${id}`}
            >
              <span className="font-mono text-[10px]">{id.split(":")[0]}</span>
              {viewTitle(b.ir, id)}
              {n ? (
                <span className="rounded-full border border-focus/40 px-1.5 font-mono text-[10px] text-focus">
                  {n}
                </span>
              ) : null}
            </button>
          );
        })}
        <span className="ml-auto text-[11px] text-muted-foreground">
          {diff.nodes.size} node · {diff.edges.size} edge · {diff.model.length} model-level changes
        </span>
      </div>
      <div className="flex min-h-0 divide-x border-b">
        {pane(a, ga, labelA, "a")}
        {pane(b, gb, labelB, "b")}
      </div>
      <ChangeList diff={diff} selected={selected} onSelect={onSelect} a={a} b={b} />
    </div>
  );
}

const MARK_CLS = {
  added: "text-ok border-ok/40",
  removed: "text-bad border-bad/40",
  changed: "text-focus border-focus/40",
} as const;

function ChangeList({
  diff,
  selected,
  onSelect,
  a,
  b,
}: {
  diff: IrDiff;
  selected: string | undefined;
  onSelect: (id: string | undefined) => void;
  a: Bundle;
  b: Bundle;
}) {
  const items = [...diff.nodes.values()].sort((x, y) => x.id.localeCompare(y.id));
  const label = (id: string) =>
    b.ir.nodes.find((n) => n.id === id)?.label ?? a.ir.nodes.find((n) => n.id === id)?.label ?? id;
  if (!items.length && !diff.model.length && !diff.edges.size) {
    return (
      <EmptyState title="No semantic differences">
        Only provenance (commits/lines) changed.
      </EmptyState>
    );
  }
  return (
    <div className="min-h-0 overflow-y-auto" data-testid="change-list">
      <table className="w-full text-[12px]">
        <thead className="sticky top-0 bg-background">
          <tr className="border-b text-left text-[11px] text-muted-foreground">
            <th className="w-56 px-3 py-1.5 font-medium">Node</th>
            <th className="w-20 py-1.5 font-medium">Change</th>
            <th className="py-1.5 font-medium">Fields (A → B)</th>
          </tr>
        </thead>
        <tbody>
          {items.map((c) => (
            <tr
              key={c.id}
              className={cn(
                "cursor-pointer border-b align-top hover:bg-accent/60",
                selected === c.id && "bg-focus-soft",
              )}
              onClick={() => onSelect(selected === c.id ? undefined : c.id)}
            >
              <td className="px-3 py-1.5">
                <div className="font-medium">{label(c.id)}</div>
                <div className="font-mono text-[11px] text-muted-foreground">{c.id}</div>
              </td>
              <td className="py-1.5">
                <span
                  className={cn(
                    "rounded border px-1 font-mono text-[10px] uppercase",
                    MARK_CLS[c.mark],
                  )}
                >
                  {c.mark}
                </span>
              </td>
              <td className="py-1.5 pr-3">
                {c.mark === "changed" ? (
                  <div className="space-y-0.5">
                    {c.fields.slice(0, 6).map((f) => (
                      <div
                        key={f.path}
                        className="grid grid-cols-[180px_1fr] gap-2 font-mono text-[11px]"
                      >
                        <span className="truncate text-muted-foreground">{f.path}</span>
                        <span className="break-all">
                          <span className="text-bad line-through decoration-bad/40">
                            {f.a ?? "∅"}
                          </span>
                          {" → "}
                          <span>{f.b ?? "∅"}</span>
                        </span>
                      </div>
                    ))}
                    {c.fields.length > 6 ? (
                      <div className="text-[11px] text-muted-foreground">
                        +{c.fields.length - 6} more fields
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <span className="text-muted-foreground">
                    {c.mark === "added" ? "new in B" : "only in A"}
                  </span>
                )}
              </td>
            </tr>
          ))}
          {diff.model.map((f) => (
            <tr key={f.path} className="border-b align-top">
              <td className="px-3 py-1.5 font-mono text-[11px]">{f.path}</td>
              <td className="py-1.5">
                <span
                  className={cn(
                    "rounded border px-1 font-mono text-[10px] uppercase",
                    MARK_CLS.changed,
                  )}
                >
                  model
                </span>
              </td>
              <td className="py-1.5 pr-3 font-mono text-[11px] break-all">
                {f.a ?? "∅"} → {f.b ?? "∅"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
