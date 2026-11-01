import { createContext, type ReactNode, useContext, useMemo } from "react";

/**
 * Figure-wide services. arch-ui never parses expressions itself: the host supplies an evaluator
 * (InferAtlas passes `@inferatlas/expr` bound to the current symbol values).
 */
export interface ArchFigureContextValue {
  /** Evaluate one shape expression; `undefined` when it cannot be evaluated. */
  evaluate?: ((expr: string) => number | undefined) | undefined;
}

const Ctx = createContext<ArchFigureContextValue>({});

export function ArchFigureProvider({
  evaluate,
  children,
}: ArchFigureContextValue & { children: ReactNode }) {
  const value = useMemo(() => ({ evaluate }), [evaluate]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useArchFigure(): ArchFigureContextValue {
  return useContext(Ctx);
}
