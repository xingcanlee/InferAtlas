import { z } from "zod";

const paramValue = z.union([z.number(), z.string(), z.boolean()]);

export const SIDE_TABS = ["memory", "kv", "roofline", "ranks"] as const;
export type SideTab = (typeof SIDE_TABS)[number];
export const HEAT = ["flops", "weight", "bytes"] as const;
export type Heat = (typeof HEAT)[number];

/**
 * Model page URL state. Known UI keys are validated; every other key is a runtime parameter
 * override (validated later against IR.params; unknown keys are ignored).
 */
export const MODEL_TABS = ["arch", "graph"] as const;
export type ModelTab = (typeof MODEL_TABS)[number];

export const modelSearchSchema = z
  .object({
    /** Model page tab; default "arch" unless a graph-only key is present (old deep links). */
    tab: z.enum(MODEL_TABS).optional().catch(undefined),
    /** Architecture tab: expanded layer index, selected sublayer, selected stage. */
    layer: z.number().int().nonnegative().optional().catch(undefined),
    sub: z.string().optional().catch(undefined),
    stage: z.string().optional().catch(undefined),
    view: z.string().optional().catch(undefined),
    node: z.string().optional().catch(undefined),
    phase: z.enum(["prefill", "decode"]).optional().catch(undefined),
    overlay: z.boolean().optional().catch(undefined),
    side: z.enum(SIDE_TABS).optional().catch(undefined),
    shapes: z.enum(["sym", "num"]).optional().catch(undefined),
    heat: z.enum(HEAT).optional().catch(undefined),
    hw: z.string().optional().catch(undefined),
  })
  .catchall(paramValue.optional().catch(undefined));

export type ModelSearch = z.infer<typeof modelSearchSchema>;

export const UI_KEYS = new Set([
  "tab",
  "layer",
  "sub",
  "stage",
  "view",
  "node",
  "phase",
  "overlay",
  "side",
  "shapes",
  "heat",
  "hw",
]);

/** Graph-only keys: a link carrying any of them opens the graph tab by default. */
const GRAPH_KEYS = ["view", "node", "phase", "overlay", "side", "heat", "hw"] as const;

export function modelTab(search: ModelSearch): ModelTab {
  if (search.tab) return search.tab;
  return GRAPH_KEYS.some((k) => search[k] !== undefined) ? "graph" : "arch";
}

export function paramOverrides(
  search: Record<string, unknown>,
): Record<string, string | number | boolean> {
  const out: Record<string, string | number | boolean> = {};
  for (const [k, v] of Object.entries(search)) {
    if (UI_KEYS.has(k) || v === undefined) continue;
    if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") out[k] = v;
  }
  return out;
}

export const homeSearchSchema = z.object({
  device: z.string().optional().catch(undefined),
});

export const diffSearchSchema = z.object({
  model: z.string().optional().catch(undefined),
  device: z.string().optional().catch(undefined),
  a: z.string().optional().catch(undefined),
  b: z.string().optional().catch(undefined),
  view: z.string().optional().catch(undefined),
  node: z.string().optional().catch(undefined),
  phase: z.enum(["prefill", "decode"]).optional().catch(undefined),
});
export type DiffSearch = z.infer<typeof diffSearchSchema>;
