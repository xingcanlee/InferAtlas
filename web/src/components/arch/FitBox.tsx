import { type ReactNode, useLayoutEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

/** Below this the figure text gets too small to read; scroll instead. */
export const MIN_FIT_SCALE = 0.6;
const PAD = 32;

/**
 * Scrollable canvas that scales its content down (never up) to fit the available box when
 * `fit` is on. The content keeps its natural layout; only a transform is applied, and the
 * wrapper takes the scaled size so scrolling still works when the minimum scale is reached.
 */
export function FitBox({
  fit,
  children,
  className,
  onScale,
  ...props
}: {
  fit: boolean;
  children: ReactNode;
  className?: string;
  onScale?: (scale: number) => void;
  "data-testid"?: string;
}) {
  const outer = useRef<HTMLDivElement>(null);
  const inner = useRef<HTMLDivElement>(null);
  const [state, setState] = useState({ scale: 1, w: 0, h: 0 });

  useLayoutEffect(() => {
    const o = outer.current;
    const i = inner.current;
    if (!o || !i) return;
    const measure = () => {
      // Layout size is unaffected by the transform.
      const w = i.offsetWidth;
      const h = i.offsetHeight;
      const s =
        fit && w && h
          ? Math.max(
              MIN_FIT_SCALE,
              Math.min(1, (o.clientWidth - PAD) / w, (o.clientHeight - PAD) / h),
            )
          : 1;
      const scale = Math.round(s * 1000) / 1000;
      setState((p) => (p.scale === scale && p.w === w && p.h === h ? p : { scale, w, h }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(o);
    ro.observe(i);
    return () => ro.disconnect();
  }, [fit]);

  useLayoutEffect(() => onScale?.(state.scale), [onScale, state.scale]);

  return (
    <div ref={outer} className={cn("min-h-0 overflow-auto", className)} {...props}>
      <div className="flex min-h-full min-w-fit items-start justify-center p-4">
        <div
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
