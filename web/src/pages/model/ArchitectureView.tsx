import { getRouteApi } from "@tanstack/react-router";
import { ArrowLeft, ChevronDown, Maximize2, Minimize2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  LayerDetail,
  LayerSelector,
  Overview,
  StageDetail,
  SymbolsEditor,
} from "@/components/arch/ArchDetail";
import { ArchFigure } from "@/components/arch/ArchFigure";
import { FitBox } from "@/components/arch/FitBox";
import { Button } from "@/components/ui/button";
import {
  archEnv,
  bandOrdinal,
  bandSummary,
  drawnBand,
  evaluator,
  layerCount,
  layerRange,
  moduleStages,
  slotAt,
  stacks,
} from "@/lib/arch";
import type { Arch, ArchBand } from "@/lib/arch-types";
import { type ModelSearch, paramOverrides } from "@/lib/search";
import { useUi } from "@/lib/store";
import { cn } from "@/lib/utils";

const route = getRouteApi("/m/$version/$device/$model");
const DETAIL_ID = "arch-detail";
const OVERLAY_SELECTOR = '[role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"]';

function isTyping(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return !!t && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));
}

/** Architecture tab: whole-model figure (left) + per-layer / per-stage tensor I/O (right). */
export function ArchitectureView({ arch }: { arch: Arch }) {
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const patch = useCallback(
    (p: Partial<ModelSearch>, replace = false) =>
      navigate({ search: (prev) => ({ ...prev, ...p }), replace }),
    [navigate],
  );

  const overrides = paramOverrides(search);
  const overridesKey = JSON.stringify(overrides);
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed by serialized overrides
  const { env, overridden } = useMemo(() => archEnv(arch, overrides), [arch, overridesKey]);
  const evaluate = useMemo(() => evaluator(env), [env]);

  // `layer` in the URL is the ordinal over all stacks (== layer index for one stack).
  const total = layerCount(arch);
  const multiStack = stacks(arch).length > 1;
  const layer = search.layer !== undefined && search.layer < total ? search.layer : undefined;
  const slot = layer !== undefined ? slotAt(arch, layer) : undefined;
  const band = slot?.band;
  const drawn = band ? drawnBand(arch, band) : undefined;
  const stage =
    !band && search.stage ? moduleStages(arch).find((s) => s.id === search.stage) : undefined;
  const sub =
    band && search.sub && band.block.sublayers.some((s) => s.id === search.sub)
      ? search.sub
      : undefined;

  const clear = useCallback(
    () => patch({ layer: undefined, sub: undefined, stage: undefined }),
    [patch],
  );
  const onSelectBand = (b: ArchBand) =>
    drawn === b && sub === undefined
      ? clear()
      : patch({
          layer: drawn === b ? layer : bandOrdinal(arch, b),
          sub: undefined,
          stage: undefined,
        });
  const onSelectSub = (b: ArchBand, id: string) =>
    patch({ layer: drawn === b ? layer : bandOrdinal(arch, b), sub: id, stage: undefined });
  const onSelectStage = (id: string) =>
    stage?.id === id ? clear() : patch({ stage: id, layer: undefined, sub: undefined });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented || isTyping(e)) return;
      // Leave Esc to the command palette / dialogs / menus while they are open.
      if (useUi.getState().searchOpen || document.querySelector(OVERLAY_SELECTOR)) return;
      clear();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clear]);

  const [symbolsOpen, setSymbolsOpen] = useState(false);
  const [fit, setFit] = useState(true);
  const [internals, setInternals] = useState(false);
  const [scale, setScale] = useState(1);

  return (
    <div className="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_360px] min-[1440px]:grid-cols-[minmax(0,1fr)_400px]">
      <div className="flex min-h-0 min-w-0 flex-col">
        <div className="flex h-10 shrink-0 items-center gap-3 border-b px-3">
          <LayerSelector
            total={total}
            layer={layer}
            stackLabel={multiStack ? slot?.stack.label : undefined}
            onLayer={(n) => patch({ layer: n, stage: undefined, sub: undefined })}
          />
          <span className="truncate text-[12px] text-muted-foreground" data-testid="layer-band">
            {band
              ? `→ ${multiStack ? `${slot?.stack.label ?? slot?.stack.id} · ` : ""}${layerRange(band.layers)} · ${bandSummary(band)}`
              : total
                ? "Click a block or pick a layer to see its tensors"
                : "No decoder stack in this model"}
          </span>
          <span className="ml-auto hidden shrink-0 text-[11px] text-muted-foreground xl:inline">
            Hover or focus a shape for its value
          </span>
          <Button
            variant="outline"
            size="sm"
            className={cn("h-7 shrink-0 px-2 text-[12px]", internals && "bg-accent")}
            aria-pressed={internals}
            onClick={() => setInternals((v) => !v)}
            title="Show key internal tensors (q/k/v, latents, routing) inside the figure"
            data-testid="internals-toggle"
          >
            Internals
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-7 shrink-0 gap-1.5 px-2 text-[12px]"
            aria-pressed={fit}
            onClick={() => setFit((f) => !f)}
            title={fit ? "Show at 100%" : "Fit the figure to the window"}
            data-testid="fit-toggle"
          >
            {fit ? <Maximize2 className="size-3.5" /> : <Minimize2 className="size-3.5" />}
            {fit ? `Fit · ${Math.round(scale * 100)}%` : "100%"}
          </Button>
        </div>
        <FitBox fit={fit} onScale={setScale} className="flex-1 bg-canvas" data-testid="arch-canvas">
          <ArchFigure
            arch={arch}
            evaluate={evaluate}
            selection={{ band: drawn, sub, stage: stage?.id }}
            onSelectBand={onSelectBand}
            onSelectSub={onSelectSub}
            onSelectStage={onSelectStage}
            detailId={DETAIL_ID}
            showInternals={internals}
          />
        </FitBox>
      </div>
      <aside
        id={DETAIL_ID}
        aria-label="Architecture details"
        className="flex min-h-0 flex-col overflow-y-auto border-l bg-background"
      >
        {band || stage ? (
          <div className="flex h-10 shrink-0 items-center border-b px-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 gap-1 px-2 text-[12px]"
              onClick={clear}
              data-testid="arch-back"
            >
              <ArrowLeft className="size-3.5" /> Overview
            </Button>
          </div>
        ) : null}
        {band && slot ? (
          <LayerDetail
            arch={arch}
            band={band}
            layer={slot.layer}
            stackLabel={multiStack ? (slot.stack.label ?? slot.stack.id) : undefined}
            sub={sub}
            evaluate={evaluate}
          />
        ) : stage ? (
          <StageDetail arch={arch} stage={stage} evaluate={evaluate} />
        ) : (
          <Overview
            arch={arch}
            onBand={(b) => patch({ layer: bandOrdinal(arch, b) })}
            onStage={(id) => patch({ stage: id })}
          />
        )}
        <section className="border-t px-4 py-3">
          <button
            type="button"
            className="flex w-full items-center justify-between text-[11px] font-medium tracking-wide text-muted-foreground uppercase"
            aria-expanded={symbolsOpen || overridden.size > 0}
            onClick={() => setSymbolsOpen((o) => !o)}
            data-testid="symbols-toggle"
          >
            <span>
              Symbols
              {overridden.size ? (
                <span className="ml-1 text-focus normal-case">· {overridden.size} changed</span>
              ) : null}
            </span>
            <ChevronDown
              className={cn(
                "size-3.5 transition-transform",
                (symbolsOpen || overridden.size > 0) && "rotate-180",
              )}
            />
          </button>
          {symbolsOpen || overridden.size > 0 ? (
            <div className="mt-2">
              <SymbolsEditor
                arch={arch}
                env={env}
                overridden={overridden}
                onSymbol={(k, v) => patch({ [k]: v }, true)}
              />
            </div>
          ) : null}
        </section>
      </aside>
    </div>
  );
}
