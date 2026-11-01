/** `hub.json`, written by the generator next to index.json (contract with G1). */
export interface HubExport {
  version: string;
  model: string;
  device: string;
  /** Sibling file name of the single-file export, e.g. `inferatlas-qwen3-8b-a2.html`. */
  file: string;
}
export interface HubManifest {
  exports: HubExport[];
}

type CellKey = { version: string; model: string; device: string };

const key = (c: CellKey) => `${c.version}\0${c.model}\0${c.device}`;

/**
 * Plain relative href to a sibling export, or undefined when the name is not a bare `.html` file
 * name (no directories, no traversal, no scheme), so the hub can only link next to itself.
 */
export function exportHref(file: string): string | undefined {
  if (!/^[\w][\w.-]*\.html$/i.test(file) || file.includes("..")) return undefined;
  return `./${file}`;
}

/** Map version/model/device → relative href of its single-file export. */
export function hubLinks(manifest: HubManifest | undefined): Map<string, string> {
  const out = new Map<string, string>();
  for (const e of manifest?.exports ?? []) {
    const href = exportHref(e.file);
    if (href && !out.has(key(e))) out.set(key(e), href);
  }
  return out;
}

export function linkFor(links: Map<string, string>, cell: CellKey): string | undefined {
  return links.get(key(cell));
}
