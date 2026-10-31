import type { GraphView, IndexCell, IR, Report, View } from "@inferatlas/contracts";
import { createContext, useContext } from "react";
import type { Hardware } from "@/lib/data";
import type { EnvState, MemoryLedger, NodeMetrics, ParamValue, Phase } from "@/lib/model";
import type { Heat, ModelSearch, SideTab } from "@/lib/search";

/** Everything the model page panels share; derived once per URL change in ModelPage. */
export interface ModelCtx {
  ir: IR;
  view: View | undefined;
  report: Report | undefined;
  cell: IndexCell;
  hardware: Hardware[];
  hw: Hardware | undefined;
  envState: EnvState;
  metrics: Map<string, NodeMetrics>;
  ledger: MemoryLedger;
  phase: Phase;
  viewId: string;
  graph: GraphView | undefined;
  selected: string | undefined;
  overlay: boolean;
  shapes: "sym" | "num";
  heat: Heat;
  side: SideTab;
  search: ModelSearch;
  /** Patch URL search state. `replace` avoids history spam for continuous controls. */
  patch: (p: Partial<ModelSearch>, replace?: boolean) => void;
  setParam: (name: string, value: ParamValue | undefined) => void;
  select: (id: string | undefined) => void;
  goView: (viewId: string, node?: string) => void;
}

export const ModelContext = createContext<ModelCtx | null>(null);

export function useModel(): ModelCtx {
  const c = useContext(ModelContext);
  if (!c) throw new Error("useModel outside ModelPage");
  return c;
}
