import type { Q } from "./model";

const trim = (x: number, digits = 3): string => {
  const s = x.toPrecision(digits);
  return s.includes("e") ? x.toExponential(1) : String(Number(s));
};

/** IEC bytes (GiB etc.), matching vLLM's memory logs. */
export function fmtBytes(b: number | undefined): string {
  if (b === undefined || Number.isNaN(b)) return "—";
  if (b === 0) return "0 B";
  const sign = b < 0 ? "−" : "";
  let x = Math.abs(b);
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
  let i = 0;
  while (x >= 1024 && i < units.length - 1) {
    x /= 1024;
    i++;
  }
  return `${sign}${trim(x)} ${units[i]}`;
}

/** Decimal SI with a unit suffix: FLOP, B/s… */
export function fmtSI(x: number | undefined, unit = ""): string {
  if (x === undefined || Number.isNaN(x)) return "—";
  if (x === 0) return `0${unit ? ` ${unit}` : ""}`;
  const sign = x < 0 ? "−" : "";
  let v = Math.abs(x);
  const prefixes = ["", "K", "M", "G", "T", "P", "E"];
  let i = 0;
  while (v >= 1000 && i < prefixes.length - 1) {
    v /= 1000;
    i++;
  }
  return `${sign}${trim(v)} ${prefixes[i]}${unit}`.trim();
}

export const fmtFlops = (x: number | undefined) => fmtSI(x, "FLOP");

/** Plain integers get thousands separators; non-integers keep 4 significant digits. */
export function fmtNum(x: number | undefined): string {
  if (x === undefined || Number.isNaN(x)) return "—";
  if (Number.isInteger(x)) return x.toLocaleString("en-US");
  return trim(x, 4);
}

/** Compact dimension for shapes: 4096, 151.9K… keeps exact integers up to 6 digits. */
export function fmtDim(x: number | undefined): string {
  if (x === undefined) return "?";
  if (Number.isInteger(x) && Math.abs(x) < 1e6) return String(x);
  if (!Number.isInteger(x)) return trim(x, 4);
  return fmtSI(x);
}

export function qText(v: Q, f: (x: number) => string = fmtNum): string {
  return v.v === undefined ? "err" : f(v.v);
}

export function shortClass(cls: string): string {
  const i = cls.lastIndexOf(".");
  return i >= 0 ? cls.slice(i + 1) : cls;
}

export function shortSha(sha: string): string {
  return /^[0-9a-f]{40}$/.test(sha) ? sha.slice(0, 7) : sha;
}

export function pct(x: number): string {
  if (!Number.isFinite(x)) return "—";
  if (x > 0 && x < 0.001) return "<0.1%";
  return `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
}
