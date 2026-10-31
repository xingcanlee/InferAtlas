export const BUDGET = Object.freeze({ nodes: 9, edges: 12, accents: 2 });

export function checkBudget(
  view: { id: string; nodes: readonly unknown[]; edges: readonly unknown[] },
  accents = 0,
): string[] {
  return (Object.keys(BUDGET) as (keyof typeof BUDGET)[]).flatMap((key) => {
    const actual = key === "accents" ? accents : view[key].length;
    return actual > BUDGET[key]
      ? [`${view.id}: ${key} ${actual} exceeds budget ${BUDGET[key]}`]
      : [];
  });
}

export function assertBudget(view: Parameters<typeof checkBudget>[0], accents = 0): void {
  const issues = checkBudget(view, accents);
  if (issues.length) throw new Error(issues.join("; "));
}
