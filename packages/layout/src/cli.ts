#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { layoutIR, serializeView } from "./index.js";

try {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      output: { type: "string", short: "o" },
      help: { type: "boolean", short: "h" },
    },
  });
  if (values.help) {
    console.log("Usage: inferatlas-layout <ir.json> -o <view.json>");
  } else {
    const input = positionals[0];
    if (!input || positionals.length !== 1 || !values.output) {
      throw new Error("Usage: inferatlas-layout <ir.json> -o <view.json>");
    }
    if (resolve(input) === resolve(values.output)) throw new Error("Output must differ from input");
    const view = await layoutIR(JSON.parse(await readFile(input, "utf8")), {
      onWarning: (message) => console.error(`inferatlas-layout: warning: ${message}`),
    });
    await writeFile(values.output, serializeView(view), "utf8");
  }
} catch (error) {
  console.error(`inferatlas-layout: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
