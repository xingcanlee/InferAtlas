import type { PhaseShape } from "@inferatlas/contracts";

// Frozen 14px sans advances; unknown glyphs get a conservative full-em width.
const advances = new Map<string, number>();
for (const [characters, width] of [
  [" ilI.,:;!'|", 4],
  ["frt()[]{}", 5],
  ["abcdeghknopqsuvxyz0123456789_+-/=×", 8],
  ["ABCDEFGHJKLNOPQRSTUVXYZ", 10],
  ["mwMW@%", 12],
] as const) {
  for (const character of characters) advances.set(character, width);
}

export function textWidth(text: string, font: "sans" | "mono" = "sans"): number {
  return Math.max(
    0,
    ...text
      .split("\n")
      .map((line) =>
        [...line].reduce(
          (width, character) => width + (font === "mono" ? 8 : (advances.get(character) ?? 14)),
          0,
        ),
      ),
  );
}

export const snap = (value: number): number => Math.round(value / 4) * 4;
export const ceilGrid = (value: number): number => Math.ceil(value / 4) * 4;

export function nodeSize(label: string): { w: number; h: number } {
  return {
    w: Math.max(96, ceilGrid(textWidth(label) + 40)),
    h: ceilGrid(label.split("\n").length * 20 + 24),
  };
}

export function shapeText(shape: string[] | PhaseShape): string {
  const dimensions = (values: string[]): string => {
    const inline = `[${values.join(", ")}]`;
    return inline.length <= 24 ? inline : `[${values.join(",\n ")}]`;
  };
  return Array.isArray(shape)
    ? dimensions(shape)
    : `prefill ${dimensions(shape.prefill)}\ndecode ${dimensions(shape.decode)}`;
}
