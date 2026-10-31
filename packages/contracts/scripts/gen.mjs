import { readdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { compile } from "json-schema-to-typescript";

const schemaDir = new URL("../../../schemas/", import.meta.url);
const outDir = new URL("../src/", import.meta.url);
const names = (await readdir(schemaDir)).filter((name) => name.endsWith(".schema.json")).sort();
for (const name of names) {
  const schema = JSON.parse(await readFile(new URL(name, schemaDir), "utf8"));
  const output = await compile(schema, schema.title, {
    cwd: fileURLToPath(schemaDir),
    bannerComment: "/* Generated from schemas; edit generator/inferatlas/schema instead. */",
    additionalProperties: false,
    format: true,
    style: { singleQuote: false, tabWidth: 2, printWidth: 100 },
  });
  await writeFile(new URL(name.replace(".schema.json", ".generated.ts"), outDir), output);
}
