import { spawnSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "vite";
import { expect, it } from "vitest";

it("release builds exclude stray public data and reject overlapping input", async () => {
  const scratch = mkdtempSync(join(tmpdir(), "inferatlas-sync-"));
  const web = join(scratch, "web");
  const source = join(scratch, "release");
  const target = join(web, "public", "data");
  const repoWeb = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  try {
    mkdirSync(join(web, "scripts"), { recursive: true });
    cpSync(join(repoWeb, "scripts", "sync-data.mjs"), join(web, "scripts", "sync-data.mjs"));
    symlinkSync(join(repoWeb, "node_modules"), join(web, "node_modules"), "dir");
    for (const stale of ["fixture-prev-A2", "main-verified-A2"]) {
      const dir = join(target, stale, "qwen3-30b-a3b");
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, "ir.json"), '{"fixture":true}');
    }
    writeFileSync(join(target, "stray.json"), "{}");
    mkdirSync(source);
    writeFileSync(join(source, "index.json"), '{"cells":[]}');
    writeFileSync(join(source, "hardware.json"), "[]");
    const sync = (input: string) =>
      spawnSync(process.execPath, [join(web, "scripts", "sync-data.mjs")], {
        cwd: web,
        env: { ...process.env, INFERATLAS_DATA_DIR: input },
        encoding: "utf8",
      });
    const result = sync(source);
    expect(result.status, result.stderr).toBe(0);
    writeFileSync(join(web, "index.html"), "<!doctype html><title>Release</title>");
    await build({ configFile: false, root: web, logLevel: "silent" });
    for (const stale of ["stray.json", "fixture-prev-A2", "main-verified-A2"]) {
      expect(existsSync(join(web, "dist", "data", stale))).toBe(false);
    }
    expect(readFileSync(join(web, "dist", "data", "index.json"), "utf8")).toBe('{"cells":[]}');
    for (const overlapping of [target, join(target, "nested"), join(web, "public")]) {
      mkdirSync(overlapping, { recursive: true });
      writeFileSync(join(overlapping, "index.json"), "{}");
      const rejected = sync(overlapping);
      expect(rejected.status).toBe(1);
      expect(rejected.stderr).toContain("release input must be separate");
    }
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
});
