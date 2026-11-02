/** Responsive helpers: media-query hook and the pure zoom maths of the architecture canvas. */
import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clampZoom,
  DESKTOP_FIT_MIN,
  distance,
  fitScale,
  hasHorizontalOverflow,
  midpoint,
  NARROW_QUERY,
  pinchZoom,
  stepZoom,
  TOUCH_FIT_MIN,
  useMediaQuery,
  ZOOM_MAX,
  ZOOM_MIN,
  zoomScroll,
} from "@/lib/responsive";

describe("fitScale", () => {
  const base = { contentW: 1000, contentH: 2000, boxW: 400, boxH: 800 };
  it("fits the width on touch layouts, but not below the readable touch floor", () => {
    expect(fitScale({ contentW: 500, contentH: 2000, boxW: 400, boxH: 800, mode: "width" })).toBe(
      0.8,
    );
    // 0.4 would fit the width but wastes the screen height: floor, scroll sideways instead.
    expect(fitScale({ ...base, mode: "width" })).toBe(TOUCH_FIT_MIN);
    expect(fitScale({ ...base, mode: "width", min: 0.3 })).toBe(0.4);
  });
  it("fits both axes on desktop but never below the readable floor", () => {
    expect(fitScale({ ...base, mode: "contain" })).toBe(DESKTOP_FIT_MIN);
    expect(fitScale({ contentW: 500, contentH: 500, boxW: 470, boxH: 900, mode: "contain" })).toBe(
      0.94,
    );
  });
  it("never scales up and survives empty measurements", () => {
    expect(fitScale({ contentW: 100, contentH: 100, boxW: 400, boxH: 400, mode: "width" })).toBe(1);
    expect(fitScale({ contentW: 0, contentH: 0, boxW: 400, boxH: 400, mode: "width" })).toBe(1);
    expect(fitScale({ contentW: 100, contentH: 100, boxW: -5, boxH: 400, mode: "width" })).toBe(1);
  });
  it("clamps very wide content to the floor", () => {
    expect(
      fitScale({
        contentW: 10_000,
        contentH: 10,
        boxW: 300,
        boxH: 300,
        mode: "width",
        min: ZOOM_MIN,
      }),
    ).toBe(ZOOM_MIN);
  });
});

describe("zoom steps and pinch", () => {
  it("steps multiplicatively and snaps to 100% when crossing it", () => {
    expect(stepZoom(0.5, 1)).toBe(0.625);
    expect(stepZoom(0.9, 1)).toBe(1);
    expect(stepZoom(1.1, -1)).toBe(1);
    expect(stepZoom(ZOOM_MAX, 1)).toBe(ZOOM_MAX);
    expect(stepZoom(ZOOM_MIN, -1)).toBe(ZOOM_MIN);
  });
  it("clamps arbitrary values", () => {
    expect(clampZoom(10)).toBe(ZOOM_MAX);
    expect(clampZoom(0)).toBe(ZOOM_MIN);
    expect(clampZoom(Number.NaN)).toBe(1);
  });
  it("scales with the finger distance", () => {
    expect(distance({ x: 0, y: 0 }, { x: 3, y: 4 })).toBe(5);
    expect(midpoint({ x: 0, y: 0 }, { x: 10, y: 20 })).toEqual({ x: 5, y: 10 });
    expect(pinchZoom(0.4, 100, 200)).toBe(0.8);
    expect(pinchZoom(1, 100, 10)).toBe(ZOOM_MIN);
    expect(pinchZoom(1, 0, 10)).toBe(1);
  });
  it("keeps the focal point fixed while zooming", () => {
    // Content point under the focus: (100 + 50 - 0) / 0.5 = 300 → at 1.0 it sits at 300.
    expect(zoomScroll({ x: 100, y: 0 }, { x: 50, y: 0 }, 0.5, 1)).toEqual({ x: 250, y: 0 });
    // Unscaled offsets (padding / centring) are taken out before and put back after.
    expect(
      zoomScroll({ x: 0, y: 0 }, { x: 200, y: 100 }, 0.5, 1, { x: 100, y: 12 }, { x: 12, y: 12 }),
    ).toEqual({ x: 12, y: 88 });
    // Never a negative scroll offset.
    expect(zoomScroll({ x: 0, y: 0 }, { x: 10, y: 10 }, 1, 0.5)).toEqual({ x: 0, y: 0 });
  });
  it("flags horizontal page overflow", () => {
    expect(hasHorizontalOverflow(390, 390)).toBe(false);
    expect(hasHorizontalOverflow(441, 390)).toBe(true);
  });
});

describe("useMediaQuery", () => {
  const original = window.matchMedia;
  afterEach(() => {
    window.matchMedia = original;
  });

  it("is false without a match (jsdom default → desktop layout)", () => {
    function Probe() {
      return <span data-testid="probe">{String(useMediaQuery(NARROW_QUERY))}</span>;
    }
    render(<Probe />);
    expect(screen.getByTestId("probe")).toHaveTextContent("false");
  });

  it("follows matchMedia changes", () => {
    let matches = true;
    const listeners = new Set<() => void>();
    window.matchMedia = vi.fn((q: string) => ({
      get matches() {
        return matches;
      },
      media: q,
      onchange: null,
      addListener() {},
      removeListener() {},
      addEventListener: (_: string, cb: () => void) => listeners.add(cb),
      removeEventListener: (_: string, cb: () => void) => listeners.delete(cb),
      dispatchEvent: () => false,
    })) as unknown as typeof window.matchMedia;
    function Probe() {
      return <span data-testid="probe">{String(useMediaQuery(NARROW_QUERY))}</span>;
    }
    render(<Probe />);
    expect(screen.getByTestId("probe")).toHaveTextContent("true");
    act(() => {
      matches = false;
      for (const l of listeners) l();
    });
    expect(screen.getByTestId("probe")).toHaveTextContent("false");
  });
});
