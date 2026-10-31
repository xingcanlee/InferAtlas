import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { compile } from "../dist/index.js";

const vectors = JSON.parse(
  readFileSync(new URL("../test-vectors.json", import.meta.url), "utf8"),
).filter((vector) => vector.fixture);
// Compilation/parsing is intentionally outside the timed UI recomputation loop.
const expressions = Array.from({ length: 10_000 }, (_, i) => {
  const vector = vectors[i % vectors.length];
  return { run: compile(`${i} >= 0 ? (${vector.expr}) : 0`), env: vector.env, value: vector.value };
});
function batch() {
  let checksum = 0;
  for (const { run, env } of expressions) checksum += run(env);
  return checksum;
}
const expected = expressions.reduce((sum, item) => sum + item.value, 0);
for (let i = 0; i < 3; i++) assert.equal(batch(), expected);
const times = [];
for (let i = 0; i < 7; i++) {
  const start = performance.now();
  const checksum = batch();
  times.push(performance.now() - start);
  assert.equal(checksum, expected);
}
times.sort((a, b) => a - b);
const medianMs = times[3];
assert.ok(medianMs < 50, `10,000 compiled evaluations took ${medianMs}ms (limit 50ms)`);
console.log(JSON.stringify({ expressions: expressions.length, medianMs }));
