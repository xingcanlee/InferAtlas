import { type ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  clampZoom,
  DESKTOP_FIT_MIN,
  distance,
  fitScale,
  midpoint,
  type Point,
  pinchZoom,
  zoomScroll,
} from "@/lib/responsive";
import { cn } from "@/lib/utils";

/**
 * Below this the figure text gets too small to read on a desktop; scroll instead. The smallest
 * figure text is 10px (chips, badges), so 0.9 keeps every label at ≥ 9px effective.
 */
export const MIN_FIT_SCALE = DESKTOP_FIT_MIN;
const PAD = 24;

export type Zoom = number | "fit";

/**
 * Scrollable, zoomable canvas. `zoom="fit"` scales the content down (never up) to the box:
 * `contain` (desktop: both axes, floor 0.9, scroll beyond) or `width` (touch: fit the width,
 * scroll down, pinch to read). A number is an explicit zoom. The content keeps its natural
 * layout; only a transform is applied and the wrapper takes the scaled size, so native scrolling
 * (one-finger drag with momentum on touch) pans. Two-finger pinch and Ctrl/⌘ + wheel zoom around
 * the gesture's focal point via `onZoom`; the page itself never zooms.
 */
export function FitBox({
  fit,
  zoom,
  mode = "contain",
  children,
  className,
  onScale,
  onZoom,
  ...props
}: {
  /** Shorthand for `zoom={fit ? "fit" : 1}` (kept for callers without zoom state). */
  fit?: boolean;
  zoom?: Zoom;
  mode?: "contain" | "width";
  children: ReactNode;
  className?: string;
  onScale?: (scale: number) => void;
  /**
   * Gesture zoom (pinch / Ctrl + wheel); the caller stores it and passes it back as `zoom`. Omit
   * it to leave those gestures to the browser (desktop page zoom).
   */
  onZoom?: (zoom: number) => void;
  "data-testid"?: string;
}) {
  const z: Zoom = zoom ?? (fit === false ? 1 : "fit");
  const outer = useRef<HTMLDivElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const inner = useRef<HTMLDivElement>(null);
  const [state, setState] = useState({ scale: 1, w: 0, h: 0 });
  /** Focal point of the next zoom change (box coordinates); default: the box centre. */
  const focus = useRef<Point | undefined>(undefined);
  /** Last laid-out scale and content offset inside the scrolled area. */
  const last = useRef<{ scale: number; offset: Point } | undefined>(undefined);

  useLayoutEffect(() => {
    const o = outer.current;
    const i = inner.current;
    if (!o || !i) return;
    const measure = () => {
      // Layout size is unaffected by the transform.
      const w = i.offsetWidth;
      const h = i.offsetHeight;
      const s =
        z === "fit"
          ? fitScale({
              contentW: w,
              contentH: h,
              boxW: o.clientWidth - PAD,
              boxH: o.clientHeight - PAD,
              mode,
            })
          : clampZoom(z);
      setState((p) => (p.scale === s && p.w === w && p.h === h ? p : { scale: s, w, h }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(o);
    ro.observe(i);
    return () => ro.disconnect();
  }, [z, mode]);

  // Keep the focal point fixed across a zoom change (pinch midpoint, wheel cursor, box centre).
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs once per committed scale
  useLayoutEffect(() => {
    const o = outer.current;
    const wr = wrap.current;
    if (!o || !wr) return;
    const box = o.getBoundingClientRect();
    const r = wr.getBoundingClientRect();
    const offset = { x: r.left - box.left + o.scrollLeft, y: r.top - box.top + o.scrollTop };
    const prev = last.current;
    // Fit re-measures (resize, new content) keep the scroll position; explicit zooms keep the focus.
    if (prev && prev.scale !== state.scale && (typeof z === "number" || focus.current)) {
      const f = focus.current ?? { x: o.clientWidth / 2, y: o.clientHeight / 2 };
      const next = zoomScroll(
        { x: o.scrollLeft, y: o.scrollTop },
        f,
        prev.scale,
        state.scale,
        prev.offset,
        offset,
      );
      o.scrollLeft = next.x;
      o.scrollTop = next.y;
    }
    focus.current = undefined;
    last.current = { scale: state.scale, offset };
  }, [state.scale]);

  useLayoutEffect(() => onScale?.(state.scale), [onScale, state.scale]);

  // Touch fit (`width`) may be wider than the box (fit floor): keep it top-aligned and centred on
  // the spine while the fitted size settles (fonts, data), until the user pans or zooms; the Fit
  // button re-centres.
  const moved = useRef(false);
  useLayoutEffect(() => {
    if (z !== "fit") moved.current = true;
    else moved.current = false;
  }, [z]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-run when the fitted size settles
  useLayoutEffect(() => {
    const o = outer.current;
    if (!o || z !== "fit" || mode !== "width" || !state.w || moved.current) return;
    o.scrollLeft = Math.max(0, (o.scrollWidth - o.clientWidth) / 2);
    o.scrollTop = 0;
  }, [z, mode, state.w, state.h, state.scale]);
  useEffect(() => {
    const o = outer.current;
    if (!o) return;
    const touched = () => {
      moved.current = true;
    };
    o.addEventListener("pointerdown", touched, { passive: true });
    o.addEventListener("touchstart", touched, { passive: true });
    o.addEventListener("wheel", touched, { passive: true });
    return () => {
      o.removeEventListener("pointerdown", touched);
      o.removeEventListener("touchstart", touched);
      o.removeEventListener("wheel", touched);
    };
  }, []);

  // Pinch (two fingers) and Ctrl/⌘ + wheel (trackpad pinch), only when the caller handles zoom
  // (`onZoom`, touch layouts): then the gesture zooms the figure instead of the page. Without
  // `onZoom` (desktop) nothing is prevented and the browser's own page zoom keeps working.
  const scaleRef = useRef(state.scale);
  scaleRef.current = state.scale;
  const onZoomRef = useRef(onZoom);
  onZoomRef.current = onZoom;
  useEffect(() => {
    const o = outer.current;
    if (!o) return;
    let pinch: { dist: number; zoom: number } | undefined;
    const pts = (e: TouchEvent): [Point, Point] | undefined => {
      const a = e.touches[0];
      const b = e.touches[1];
      if (!a || !b) return undefined;
      const box = o.getBoundingClientRect();
      return [
        { x: a.clientX - box.left, y: a.clientY - box.top },
        { x: b.clientX - box.left, y: b.clientY - box.top },
      ];
    };
    // touchmove is non-passive only while two fingers are down, so one-finger pans stay fully
    // passive (native scrolling is never blocked waiting for script).
    const start = (e: TouchEvent) => {
      const p = pts(e);
      if (!p || !onZoomRef.current) return;
      if (!pinch) o.addEventListener("touchmove", move, { passive: false });
      pinch = { dist: distance(p[0], p[1]), zoom: scaleRef.current };
    };
    const move = (e: TouchEvent) => {
      const p = pts(e);
      if (!pinch || !p) return;
      e.preventDefault();
      const next = pinchZoom(pinch.zoom, pinch.dist, distance(p[0], p[1]));
      if (next === scaleRef.current) return;
      focus.current = midpoint(p[0], p[1]);
      onZoomRef.current?.(next);
    };
    const end = (e: TouchEvent) => {
      if (e.touches.length >= 2 || !pinch) return;
      pinch = undefined;
      o.removeEventListener("touchmove", move);
    };
    const wheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey) || !onZoomRef.current) return;
      e.preventDefault();
      const box = o.getBoundingClientRect();
      focus.current = { x: e.clientX - box.left, y: e.clientY - box.top };
      const next = clampZoom(scaleRef.current * Math.exp(-e.deltaY / 300));
      onZoomRef.current(Math.round(next * 1000) / 1000);
    };
    o.addEventListener("touchstart", start, { passive: true });
    o.addEventListener("touchend", end, { passive: true });
    o.addEventListener("touchcancel", end, { passive: true });
    o.addEventListener("wheel", wheel, { passive: false });
    return () => {
      o.removeEventListener("touchstart", start);
      o.removeEventListener("touchmove", move);
      o.removeEventListener("touchend", end);
      o.removeEventListener("touchcancel", end);
      o.removeEventListener("wheel", wheel);
    };
  }, []);

  return (
    <div
      ref={outer}
      className={cn("min-h-0 touch-pan-x touch-pan-y overflow-auto overscroll-contain", className)}
      data-zoom={z}
      {...props}
    >
      <div className="flex min-h-full min-w-fit items-start justify-center p-3">
        <div
          ref={wrap}
          style={
            state.w ? { width: state.w * state.scale, height: state.h * state.scale } : undefined
          }
        >
          <div
            ref={inner}
            data-scale={state.scale}
            className="w-max origin-top-left"
            style={state.scale === 1 ? undefined : { transform: `scale(${state.scale})` }}
          >
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}
