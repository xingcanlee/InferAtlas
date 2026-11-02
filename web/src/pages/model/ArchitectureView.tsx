import type { Arch, ArchBand } from "@inferatlas/contracts";
import { getRouteApi } from "@tanstack/react-router";
import {
  ArrowLeft,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Maximize2,
  Minimize2,
  Minus,
  MoreVertical,
  PanelBottomOpen,
  Plus,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  LayerDetail,
  LayerSelector,
  Overview,
  StageDetail,
  SymbolsEditor,
} from "@/components/arch/ArchDetail";
import { ArchFigure } from "@/components/arch/ArchFigure";
import { FitBox, type Zoom } from "@/components/arch/FitBox";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import {
  archEnv,
  bandOrdinal,
  bandSummary,
  drawnBand,
  evaluator,
  layerCount,
  layerRange,
  moduleStages,
  normalizeArch,
  slotAt,
  stacks,
  stageLabel,
} from "@/lib/arch";
import { stepZoom, useIsNarrow, ZOOM_MAX, ZOOM_MIN } from "@/lib/responsive";
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
export function ArchitectureView({ arch: raw }: { arch: Arch }) {
  // Older arch.json (deprecated attention variants) is viewed through the v1.3 shape.
  const arch = useMemo(() => normalizeArch(raw), [raw]);
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

  // Keep the selected block in view when the figure is taller than the canvas (layer selector,
  // panel links): scroll only as far as needed.
  useEffect(() => {
    if (!drawn) return;
    const el = document.querySelector(`[data-testid="band-${drawn.id.replace(/["\\]/g, "")}"]`);
    el?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
  }, [drawn]);

  const narrow = useIsNarrow();
  const [symbolsOpen, setSymbolsOpen] = useState(false);
  const [zoom, setZoom] = useState<Zoom>("fit");
  const [internals, setInternals] = useState(false);
  const [scale, setScale] = useState(1);
  // Narrow screens: the detail panel is a bottom sheet, opened by a selection (or a deep link
  // that carries one) and by the Details button; closing it keeps the selection.
  const [sheetOpen, setSheetOpen] = useState(() => narrow && !!(band || stage));
  const reveal = () => narrow && setSheetOpen(true);
  // A selection arriving through the URL (deep link, back/forward) opens the sheet too.
  const selKey = band || stage ? `${layer}/${sub ?? ""}/${stage?.id ?? ""}` : "";
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed by the selection only
  useEffect(() => {
    if (narrow && selKey) setSheetOpen(true);
  }, [selKey]);
  const selectBand = (b: ArchBand) => {
    onSelectBand(b);
    reveal();
  };
  const selectSub = (b: ArchBand, id: string) => {
    onSelectSub(b, id);
    reveal();
  };
  const selectStage = (id: string) => {
    onSelectStage(id);
    reveal();
  };
  const onLayer = (n: number | undefined) => patch({ layer: n, stage: undefined, sub: undefined });

  const status = band
    ? `${multiStack ? `${slot?.stack.label ?? slot?.stack.id} · ` : ""}${layerRange(band.layers)} · ${bandSummary(band)}`
    : undefined;
  const pct = `${Math.round(scale * 100)}%`;

  const detail =
    band && slot ? (
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
    );

  const symbols = (
    <section className="border-t px-4 py-3">
      <button
        type="button"
        className="flex min-h-6 w-full items-center justify-between text-[11px] font-medium tracking-wide text-muted-foreground uppercase max-lg:min-h-10"
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
  );

  const figure = (
    <ArchFigure
      arch={arch}
      evaluate={evaluate}
      selection={{ band: drawn, sub, stage: stage?.id }}
      onSelectBand={selectBand}
      onSelectSub={selectSub}
      onSelectStage={selectStage}
      detailId={DETAIL_ID}
      showInternals={internals}
    />
  );

  if (narrow)
    return (
      <div className="flex h-full min-h-0 flex-col" data-layout="narrow">
        <div className="flex h-12 shrink-0 items-center gap-2 border-b safe-px">
          <LayerSelector
            total={total}
            layer={layer}
            stackLabel={multiStack ? slot?.stack.label : undefined}
            onLayer={(n) => {
              onLayer(n);
              if (n !== undefined) reveal();
            }}
            touch
          />
          <div className="ml-auto flex shrink-0 items-center gap-1">
            <Button
              variant="outline"
              className="h-10 gap-1.5 px-3 text-[13px]"
              onClick={() => setSheetOpen(true)}
              aria-controls={DETAIL_ID}
              aria-expanded={sheetOpen}
              data-testid="open-details"
            >
              <PanelBottomOpen className="size-4" />
              Details
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-lg"
                  aria-label="Figure options"
                  data-testid="arch-menu"
                >
                  <MoreVertical className="size-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuCheckboxItem
                  checked={internals}
                  onCheckedChange={(v) => setInternals(v === true)}
                  data-testid="internals-toggle"
                >
                  Show internal tensors
                </DropdownMenuCheckboxItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={() => setZoom("fit")}>
                  <Minimize2 /> Fit to width
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => setZoom(1)}>
                  <Maximize2 /> Actual size (100%)
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        {status ? (
          <button
            type="button"
            className="flex h-8 shrink-0 items-center gap-1 truncate border-b text-left text-[12px] text-muted-foreground safe-px"
            onClick={() => setSheetOpen(true)}
            data-testid="layer-band"
          >
            <span className="truncate">→ {status}</span>
          </button>
        ) : null}
        <div className="relative min-h-0 flex-1">
          <FitBox
            zoom={zoom}
            mode="width"
            onScale={setScale}
            onZoom={setZoom}
            className="h-full bg-canvas"
            data-testid="arch-canvas"
          >
            {figure}
          </FitBox>
          <ZoomControls zoom={zoom} scale={scale} pct={pct} onZoom={setZoom} />
        </div>
        <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
          <SheetContent
            side="bottom"
            id={DETAIL_ID}
            aria-label="Architecture details"
            aria-describedby={undefined}
            className="max-h-[min(80dvh,calc(100dvh-5rem))] md:max-h-[65dvh]"
            data-testid="detail-sheet"
            // Focus the sheet itself, not its first button (no stray focus ring on open).
            onOpenAutoFocus={(e) => {
              e.preventDefault();
              (e.currentTarget as HTMLElement | null)?.focus();
            }}
          >
            <div className="flex min-h-12 shrink-0 items-center gap-1 border-b py-1 pr-12 safe-pl">
              <span
                aria-hidden="true"
                className="absolute top-1.5 left-1/2 h-1 w-10 -translate-x-1/2 rounded-full bg-border"
              />
              {band || stage ? (
                <Button
                  variant="ghost"
                  className="h-10 gap-1 px-2 text-[13px]"
                  onClick={clear}
                  data-testid="arch-back"
                >
                  <ArrowLeft className="size-4" /> Overview
                </Button>
              ) : null}
              <SheetTitle className="min-w-0 truncate px-1 text-[14px]">
                {band && slot
                  ? `${multiStack ? `${slot.stack.label ?? slot.stack.id} · ` : ""}Layer ${slot.layer}`
                  : stage
                    ? stageLabel(arch, stage)
                    : "Overview"}
              </SheetTitle>
              {band ? (
                <div className="ml-auto flex shrink-0 items-center gap-1">
                  <Button
                    variant="outline"
                    size="icon-lg"
                    aria-label="Previous layer"
                    disabled={layer === 0}
                    onClick={() => onLayer(Math.max(0, (layer ?? 0) - 1))}
                    data-testid="sheet-prev-layer"
                  >
                    <ChevronLeft className="size-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon-lg"
                    aria-label="Next layer"
                    disabled={layer === total - 1}
                    onClick={() => onLayer(Math.min(total - 1, (layer ?? -1) + 1))}
                    data-testid="sheet-next-layer"
                  >
                    <ChevronRight className="size-4" />
                  </Button>
                </div>
              ) : null}
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain text-[13px]">
              {detail}
              {symbols}
            </div>
          </SheetContent>
        </Sheet>
      </div>
    );

  return (
    <div
      className="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_360px] min-[1440px]:grid-cols-[minmax(0,1fr)_400px]"
      data-layout="wide"
    >
      <div className="flex min-h-0 min-w-0 flex-col">
        <div className="flex h-10 shrink-0 items-center gap-3 border-b px-3">
          <LayerSelector
            total={total}
            layer={layer}
            stackLabel={multiStack ? slot?.stack.label : undefined}
            onLayer={onLayer}
          />
          <span className="truncate text-[12px] text-muted-foreground" data-testid="layer-band">
            {status
              ? `→ ${status}`
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
            aria-pressed={zoom === "fit"}
            onClick={() => setZoom((z) => (z === "fit" ? 1 : "fit"))}
            title={zoom === "fit" ? "Show at 100%" : "Fit the figure to the window"}
            data-testid="fit-toggle"
          >
            {zoom === "fit" ? (
              <Maximize2 className="size-3.5" />
            ) : (
              <Minimize2 className="size-3.5" />
            )}
            {zoom === "fit" ? `Fit · ${pct}` : pct}
          </Button>
        </div>
        <FitBox
          zoom={zoom}
          onScale={setScale}
          onZoom={setZoom}
          className="flex-1 bg-canvas"
          data-testid="arch-canvas"
        >
          {figure}
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
        {detail}
        {symbols}
      </aside>
    </div>
  );
}

/** Thumb-reachable zoom buttons over the canvas (touch layouts). */
function ZoomControls({
  zoom,
  scale,
  pct,
  onZoom,
}: {
  zoom: Zoom;
  scale: number;
  pct: string;
  onZoom: (z: Zoom) => void;
}) {
  return (
    <fieldset
      className="absolute right-[max(0.75rem,env(safe-area-inset-right))] bottom-[max(0.75rem,env(safe-area-inset-bottom))] m-0 flex min-w-0 items-center overflow-hidden rounded-lg border bg-background/95 p-0 backdrop-blur"
      data-testid="zoom-controls"
    >
      <legend className="sr-only">Zoom</legend>
      <Button
        variant="ghost"
        size="icon-lg"
        className="rounded-none"
        aria-label="Zoom out"
        disabled={scale <= ZOOM_MIN}
        onClick={() => onZoom(stepZoom(scale, -1))}
        data-testid="zoom-out"
      >
        <Minus className="size-4" />
      </Button>
      <Button
        variant="ghost"
        className="h-10 min-w-16 rounded-none border-x px-2 font-mono text-[12px] tabular"
        aria-label={zoom === "fit" ? `Fitted at ${pct}; show at 100%` : `Zoom ${pct}; fit to width`}
        aria-pressed={zoom === "fit"}
        onClick={() => onZoom(zoom === "fit" ? 1 : "fit")}
        data-testid="zoom-fit"
      >
        {zoom === "fit" ? `Fit ${pct}` : pct}
      </Button>
      <Button
        variant="ghost"
        size="icon-lg"
        className="rounded-none"
        aria-label="Zoom in"
        disabled={scale >= ZOOM_MAX}
        onClick={() => onZoom(stepZoom(scale, 1))}
        data-testid="zoom-in"
      >
        <Plus className="size-4" />
      </Button>
    </fieldset>
  );
}
