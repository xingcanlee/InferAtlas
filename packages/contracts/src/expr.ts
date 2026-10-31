/** Stable AST contract shared by the expression implementation and its consumers. */
export type Ast =
  | { type: "number"; value: number }
  | { type: "boolean"; value: boolean }
  | { type: "identifier"; name: string }
  | { type: "unary"; op: "+" | "-" | "!"; argument: Ast }
  | {
      type: "binary";
      op: "+" | "-" | "*" | "/" | "%" | "**" | "<" | "<=" | ">" | ">=" | "==" | "!=" | "&&" | "||";
      left: Ast;
      right: Ast;
    }
  | { type: "conditional"; test: Ast; consequent: Ast; alternate: Ast }
  | {
      type: "call";
      name: "max" | "min" | "ceil" | "floor" | "cdiv" | "round_up" | "bytes";
      args: Ast[];
    };

export type ExprEnv = Record<string, number | string | boolean>;

/** Synchronous, pure evaluator. All successful numeric results must be finite. */
export interface ExprEvaluator {
  parse(src: string): Ast;
  evaluate(src: string | Ast, env: ExprEnv): number;
  /** Sorted unique unbound identifiers; builtin booleans and dtype names are excluded. */
  free(src: string): string[];
  /** Symbolic mode canonicalizes; numeric mode evaluates with env and prints a finite number. */
  format(src: string, mode: "symbolic" | "numeric", env?: ExprEnv): string;
}
