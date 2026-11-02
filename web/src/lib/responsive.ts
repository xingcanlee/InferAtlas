/**
 * Responsive helpers: media-query hooks (Tailwind breakpoints) and the pure zoom maths behind the
 * architecture canvas (fit, pinch, zoom steps around a focal point). The maths is pure so it is
 * unit-tested without a browser.
 */
import { useSyncExternalStore } from "react";

/** Tailwind `lg` (1024px): below it the detail panel becomes a bottom sheet. */
export const NARROW_QUERY = "(max-width: 1023.98px)";
/** Tailwind `md` (768px): below it tables become cards and the graph workspace is gated. */
export const PHONE_QUERY = "(max-width: 767.98px)";

function mql(query: string): MediaQueryList | undefined {
  return typeof window !== "undefined" && typeof window.matchMedia === "function"
    ? window.matchMedia(query)
    : undefined;
}

/**
 * Live `matchMedia` state. Read synchronously on first render (client-only app), so the right
 * layout is chosen before paint and nothing shifts. Without `matchMedia` (jsdom) it is `false`,
 * i.e. the desktop layout.
 */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (cb) => {
      const m = mql(query);
      if (!m) return () => {};
      m.addEventListener?.("change", cb);
      return () => m.removeEventListener?.("change", cb);
    },
    () => mql(query)?.matches ?? false,
    () => false,
  );
}

/** Below `lg`: bottom-sheet detail panel, compact toolbars, touch-size controls. */
export const useIsNarrow = () => useMediaQuery(NARROW_QUERY);
/** Below `md`: card layouts. */
export const useIsPhone = () => useMediaQuery(PHONE_QUERY);

// ----------------------------------------------------------------------------- zoom maths

export const ZOOM_MIN = 0.2;
export const ZOOM_MAX = 2.5;
/** Desktop fit floor: the smallest figure text (10px) stays ≥ 9px effective. */
export const DESKTOP_FIT_MIN = 0.9;

export const clampZoom = (z: number, min = ZOOM_MIN, max = ZOOM_MAX): number =>
  Math.min(max, Math.max(min, Number.isFinite(z) ? z : 1));

const round3 = (n: number) => Math.round(n * 1000) / 1000;

export interface FitInput {
  /** Natural content size. */
  contentW: number;
  contentH: number;
  /** Available box (already minus padding). */
  boxW: number;
  boxH: number;
  /**
   * `contain`: fit both axes, never below `min` (desktop: readable text, scroll beyond).
   * `width`: fit the width only (touch: overview, scroll down, pinch to read).
   */
  mode: "contain" | "width";
  min?: number;
}

/** Scale that fits the content into the box; never scales up. */
export function fitScale({ contentW, contentH, boxW, boxH, mode, min }: FitInput): number {
  if (!contentW || !contentH || boxW <= 0 || boxH <= 0) return 1;
  const sw = boxW / contentW;
  const s = mode === "width" ? Math.min(1, sw) : Math.min(1, sw, boxH / contentH);
  return round3(Math.max(min ?? (mode === "width" ? ZOOM_MIN : DESKTOP_FIT_MIN), s));
}

/** Next zoom level for the +/− buttons (multiplicative steps, snapped to 100%). */
export function stepZoom(z: number, dir: 1 | -1, factor = 1.25): number {
  const next = dir > 0 ? z * factor : z / factor;
  // Snap to 100% when a step crosses it, so "actual size" is always reachable.
  if ((z < 1 && next > 1) || (z > 1 && next < 1)) return 1;
  return round3(clampZoom(next));
}

export interface Point {
  x: number;
  y: number;
}

export const distance = (a: Point, b: Point): number => Math.hypot(a.x - b.x, a.y - b.y);
export const midpoint = (a: Point, b: Point): Point => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

/** Zoom after a pinch that started at `startDist` with zoom `startZoom` and is now `dist`. */
export const pinchZoom = (startZoom: number, startDist: number, dist: number): number =>
  round3(clampZoom(startDist > 0 ? (startZoom * dist) / startDist : startZoom));

/**
 * Scroll offsets that keep the content point under `focus` (coordinates relative to the scroll
 * box's visible area) fixed while the zoom changes from `from` to `to`. `offsetFrom`/`offsetTo`
 * are the scaled content's position inside the scrolled area before/after the change (padding,
 * centring of content narrower than the box), which do not scale with the content.
 */
export function zoomScroll(
  scroll: Point,
  focus: Point,
  from: number,
  to: number,
  offsetFrom: Point = { x: 0, y: 0 },
  offsetTo: Point = offsetFrom,
): Point {
  const cx = (scroll.x + focus.x - offsetFrom.x) / from;
  const cy = (scroll.y + focus.y - offsetFrom.y) / from;
  return {
    x: Math.max(0, cx * to + offsetTo.x - focus.x),
    y: Math.max(0, cy * to + offsetTo.y - focus.y),
  };
}

/** True when the page scrolls sideways (a layout bug on phones). */
export const hasHorizontalOverflow = (scrollWidth: number, innerWidth: number): boolean =>
  scrollWidth > innerWidth + 0.5;
