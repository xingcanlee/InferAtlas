import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

try {
  const url = process.argv[2]
    ? pathToFileURL(process.argv[2])
    : new URL("../../../packages/expr/dist/index.js", import.meta.url);
  const { evaluate } = await import(url.href);
  const { expressions, env } = JSON.parse(readFileSync(0, "utf8"));
  const values = expressions.map((expression) => evaluate(expression, env));
  process.stdout.write(`${JSON.stringify({ values })}\n`);
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exitCode = 1;
}
