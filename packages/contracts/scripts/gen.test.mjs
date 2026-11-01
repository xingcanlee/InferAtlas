import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";

const sourceDir = new URL("../src/", import.meta.url);
async function generated() {
  const files = (await readdir(sourceDir)).filter((name) => name.endsWith(".generated.ts")).sort();
  return Object.fromEntries(
    await Promise.all(
      files.map(async (name) => [name, await readFile(new URL(name, sourceDir), "utf8")]),
    ),
  );
}

test("committed generated types match schemas and regeneration is deterministic", async () => {
  const before = await generated();
  assert.equal(Object.keys(before).length, 14);
  execFileSync(process.execPath, [new URL("./gen.mjs", import.meta.url).pathname]);
  assert.deepEqual(await generated(), before);
  execFileSync(process.execPath, [new URL("./gen.mjs", import.meta.url).pathname]);
  assert.deepEqual(await generated(), before);
});

test("public contract barrel and ExprEvaluator typecheck in strict TypeScript 5", () => {
  execFileSync("pnpm", ["exec", "tsc", "-p", "tsconfig.json", "--noEmit"], {
    cwd: new URL("../", import.meta.url),
    stdio: "pipe",
  });
});
