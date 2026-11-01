import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Layout, LayoutChangedMeta, PanelSize } from "react-resizable-panels";
import { usePanelRef } from "react-resizable-panels";
import { AnalysisTabs } from "@/components/model/AnalysisTabs";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";

/** localStorage key for the user's graph / analysis split (UI preference, not shareable state). */
export const SPLIT_KEY = "inferatlas-model-split";
/** Collapsed analysis panel: just its 40px tab strip plus the top border. */
const BAR_PX = 41;
/** Below this the analysis charts are unusable, so dragging further collapses the panel. */
const ANALYSIS_MIN_PX = 180;
/** Open size when expanding without a remembered size. */
const ANALYSIS_OPEN = "28%";
/**
 * Short windows show the analysis panel collapsed so the whole graph fits readably (≥ 0.6 zoom);
 * taller windows show it open. Applies until the user resizes or toggles the panel.
 * 900px + the 40px model tab header (Architecture | Graph & analysis) above the workspace.
 */
const OPEN_BY_DEFAULT_MIN_HEIGHT = 940;

/** Re-open size; percentages need the unit, a bare number would be pixels in v4. */
const openSize = (pct: number | undefined) => (pct === undefined ? ANALYSIS_OPEN : `${pct}%`);

function savedLayout(): Layout | undefined {
  try {
    const v = JSON.parse(localStorage.getItem(SPLIT_KEY) ?? "null") as unknown;
    if (
      v &&
      typeof v === "object" &&
      typeof (v as Layout)["graph-pane"] === "number" &&
      typeof (v as Layout)["analysis-pane"] === "number"
    )
      return v as Layout;
  } catch {
    // Unavailable or corrupt storage falls back to the viewport default.
  }
  return undefined;
}

/**
 * Vertical split of the model page's main column: graph (+ legend) above, analysis tabs below.
 * The analysis panel is resizable (drag or arrow keys on the separator) and collapsible to its
 * tab strip (toggle button, Enter on the separator, or dragging past its minimum); the user's
 * choice persists in localStorage.
 */
export function MainSplit({ graph }: { graph: ReactNode }) {
  const panel = usePanelRef();
  const saved = useMemo(savedLayout, []);
  const initialOpen = useMemo(() => window.innerHeight >= OPEN_BY_DEFAULT_MIN_HEIGHT, []);
  const [collapsed, setCollapsed] = useState(
    saved ? false : !initialOpen, // corrected by the first onResize when a layout is restored
  );
  /** Last open size as a percentage of the group (v4 `resize(number)` would mean pixels). */
  const lastOpen = useRef<number | undefined>(undefined);
  const userToggle = useRef(false);

  const onResize = useCallback((size: PanelSize) => {
    const isCollapsed = size.inPixels <= BAR_PX + 1;
    setCollapsed(isCollapsed);
    if (!isCollapsed) lastOpen.current = size.asPercentage;
  }, []);

  const onLayoutChanged = useCallback((layout: Layout, meta: LayoutChangedMeta) => {
    if (!meta.isUserInteraction && !userToggle.current) return;
    userToggle.current = false;
    try {
      localStorage.setItem(SPLIT_KEY, JSON.stringify(meta.requestedLayout ?? layout));
    } catch {
      // Persisting is best-effort.
    }
  }, []);

  const setAnalysisCollapsed = useCallback(
    (next: boolean) => {
      const p = panel.current;
      if (!p || p.isCollapsed() === next) return;
      // Only a real change produces an onLayoutChanged that should be persisted.
      userToggle.current = true;
      if (next) p.collapse();
      else p.resize(openSize(lastOpen.current));
    },
    [panel],
  );

  // Until the user picks a split, the default follows the window height (debounced).
  useEffect(() => {
    let t: number | undefined;
    const onResize = () => {
      window.clearTimeout(t);
      t = window.setTimeout(() => {
        const p = panel.current;
        if (!p || savedLayout()) return;
        const open = window.innerHeight >= OPEN_BY_DEFAULT_MIN_HEIGHT;
        if (open && p.isCollapsed()) p.resize(openSize(lastOpen.current));
        else if (!open && !p.isCollapsed()) p.collapse();
      }, 120);
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      window.clearTimeout(t);
    };
  }, [panel]);

  return (
    <ResizablePanelGroup
      orientation="vertical"
      id="model-split"
      className="min-h-0 flex-1"
      defaultLayout={saved}
      onLayoutChanged={onLayoutChanged}
    >
      <ResizablePanel id="graph-pane" minSize="30%" className="flex flex-col">
        {graph}
      </ResizablePanel>
      <ResizableHandle
        withHandle
        aria-label="Resize analysis panel"
        data-testid="analysis-resize"
        onKeyDown={(e) => {
          // Enter toggles the analysis panel; arrows/Home/End keep their resize behaviour.
          if (e.key !== "Enter") return;
          e.preventDefault();
          setAnalysisCollapsed(!panel.current?.isCollapsed());
        }}
      />
      <ResizablePanel
        id="analysis-pane"
        panelRef={panel}
        collapsible
        collapsedSize={BAR_PX}
        minSize={ANALYSIS_MIN_PX}
        defaultSize={initialOpen ? ANALYSIS_OPEN : BAR_PX}
        onResize={onResize}
      >
        <AnalysisTabs
          className="h-full"
          collapsed={collapsed}
          onCollapsedChange={setAnalysisCollapsed}
          contentId="analysis-content"
        />
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
