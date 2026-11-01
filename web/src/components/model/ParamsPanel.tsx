import type { ExprEnv, ParamSpec } from "@inferatlas/contracts";
import { AlertTriangle, Info, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { tryEvaluate } from "@/lib/expr";
import { fmtNum } from "@/lib/format";
import { type ParamValue, paramBounds } from "@/lib/model";
import { cn } from "@/lib/utils";
import { useModel } from "@/pages/model/context";

const GROUPS: [string, string[]][] = [
  ["Parallelism", ["tp", "pp", "dp", "ep"]],
  ["Workload", ["T", "S", "batch", "context"]],
  ["Dtypes", ["dtype_w", "dtype_a", "dtype_kv", "dtype_ids"]],
  [
    "Memory & scheduler",
    [
      "hbm_bytes",
      "gpu_memory_utilization",
      "max_num_batched_tokens",
      "max_num_seqs",
      "block_size",
      "non_torch_bytes",
      "reserved_bytes",
    ],
  ],
];

const LABELS: Record<string, string> = {
  tp: "Tensor parallel",
  pp: "Pipeline parallel",
  dp: "Data parallel",
  ep: "Expert parallel",
  T: "Prefill tokens",
  S: "Decode seqs",
  batch: "Prefill batch",
  context: "Context length",
  dtype_w: "Weights",
  dtype_a: "Activations",
  dtype_kv: "KV cache",
  dtype_ids: "Token ids",
  hbm_bytes: "Device memory",
  gpu_memory_utilization: "Memory utilization",
  max_num_batched_tokens: "Max batched tokens",
  max_num_seqs: "Max seqs",
  block_size: "KV block size",
  non_torch_bytes: "Non-torch bytes",
  reserved_bytes: "Reserved bytes",
};

/** Log-scale mapping for wide integer ranges (e.g. 1..40960 tokens). */
function useScale(min: number, max: number, integer: boolean) {
  const log = max / Math.max(min, 1) >= 64;
  const to = (v: number) =>
    log ? Math.log(v - min + 1) / Math.log(max - min + 1) : (v - min) / (max - min);
  const from = (p: number) => {
    const v = log ? min - 1 + (max - min + 1) ** p : min + p * (max - min);
    return integer ? Math.round(v) : v;
  };
  return { to, from };
}

function NumberControl({
  name,
  spec,
  value,
  onChange,
}: {
  name: string;
  spec: ParamSpec;
  value: number;
  onChange: (v: number) => void;
}) {
  const m = useModel();
  const b = paramBounds(spec, m.envState.symbols);
  const integer = spec.kind === "integer";
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  const commit = () => {
    const n = Number(draft);
    if (draft.trim() !== "" && Number.isFinite(n)) onChange(n);
    else setDraft(String(value));
  };
  const hasRange = b.min !== undefined && b.max !== undefined && b.max > b.min;
  const scale = useScale(b.min ?? 0, b.max ?? 1, integer);
  return (
    <div className="flex items-center gap-2">
      {hasRange ? (
        <Slider
          className="flex-1"
          min={0}
          max={1000}
          step={1}
          value={[Math.round(scale.to(Math.min(Math.max(value, b.min ?? 0), b.max ?? 1)) * 1000)]}
          onValueChange={([p]) => {
            let v = scale.from((p ?? 0) / 1000);
            if (!integer && b.step) v = Math.round(v / b.step) * b.step;
            onChange(Number(v.toFixed(6)));
          }}
          aria-label={name}
        />
      ) : null}
      <Input
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
          if (e.key === "Escape") {
            setDraft(String(value));
            (e.target as HTMLInputElement).blur();
          }
        }}
        inputMode="decimal"
        className={cn("h-7 px-2 font-mono text-[12px] tabular", hasRange ? "w-20" : "w-full")}
        aria-label={`${name} value`}
        data-testid={`param-${name}`}
      />
    </div>
  );
}

/**
 * Why a choice is not allowed at the current parameters, or undefined if it is.
 * Uses an IR-declared `constraint` expression when present (T3), otherwise the vLLM rule
 * that EP is either disabled (1) or spans the flattened tp×dp group.
 */
function choiceBlocked(
  name: string,
  spec: ParamSpec,
  choice: string,
  env: ExprEnv,
): string | undefined {
  const constraint = spec.constraint;
  if (constraint) {
    const r = tryEvaluate(constraint, { ...env, [name]: Number(choice) });
    return r.ok && r.value === 0 ? `violates ${constraint}` : undefined;
  }
  if (name === "ep" && typeof env.tp === "number" && typeof env.dp === "number") {
    const v = Number(choice);
    const group = env.tp * env.dp;
    if (v !== 1 && v !== group) return `EP is 1 (off) or tp×dp = ${group}`;
  }
  return undefined;
}

function ChoiceControl({
  name,
  spec,
  value,
  onChange,
}: {
  name: string;
  spec: ParamSpec;
  value: ParamValue;
  onChange: (v: ParamValue) => void;
}) {
  const m = useModel();
  const choices = spec.choices ?? [];
  const blocked = (c: string) => choiceBlocked(name, spec, c, m.envState.env);
  const numeric = spec.kind === "integer" || spec.kind === "number";
  const parse = (s: string): ParamValue => (numeric ? Number(s) : s);
  if (choices.length <= 6) {
    return (
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        className="w-full"
        value={String(value)}
        onValueChange={(v) => v && onChange(parse(v))}
        aria-label={name}
        data-testid={`param-${name}`}
      >
        {choices.map((c) => (
          <ToggleGroupItem
            key={c}
            value={c}
            disabled={!!blocked(c) && String(value) !== c}
            title={blocked(c)}
            className="h-7 flex-1 px-1 font-mono text-[11.5px]"
          >
            {c}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    );
  }
  return (
    <Select value={String(value)} onValueChange={(v) => onChange(parse(v))}>
      <SelectTrigger
        size="sm"
        className="h-7 w-full font-mono text-[12px]"
        data-testid={`param-${name}`}
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {choices.map((c) => (
          <SelectItem
            key={c}
            value={c}
            disabled={!!blocked(c) && String(value) !== c}
            className="font-mono text-[12px]"
            data-testid={`param-${name}-${c}`}
          >
            {c}
            {blocked(c) ? (
              <span className="font-sans text-[10.5px] text-muted-foreground">{blocked(c)}</span>
            ) : null}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function ParamRow({ name, spec }: { name: string; spec: ParamSpec }) {
  const m = useModel();
  const value = m.envState.params[name];
  const def = m.envState.defaults[name];
  const error = m.envState.errors[name];
  const raw = m.search[name];
  const modified = raw !== undefined && !error;
  const warn = m.envState.warnings.some((w) => w.params.includes(name));
  if (value === undefined) return null;
  return (
    <div className="space-y-1.5 py-2" data-modified={modified || undefined}>
      <div className="flex items-center gap-1.5">
        <span className="text-[12px] font-medium">{LABELS[name] ?? name}</span>
        <span className="font-mono text-[10.5px] text-muted-foreground">{name}</span>
        <Tooltip>
          <TooltipTrigger asChild>
            <Info className="size-3 text-muted-foreground/70" aria-label={`About ${name}`} />
          </TooltipTrigger>
          <TooltipContent side="right" className="max-w-64">
            {spec.description}
            <div className="mt-1 font-mono opacity-70">default {String(def)}</div>
          </TooltipContent>
        </Tooltip>
        {warn ? <AlertTriangle className="size-3 text-warn" aria-label="inconsistent" /> : null}
        {modified ? (
          <button
            type="button"
            className="ml-auto flex items-center gap-1 text-[10.5px] text-muted-foreground hover:text-foreground"
            onClick={() => m.setParam(name, undefined)}
            title={`Reset to ${String(def)}`}
          >
            <span className="size-1.5 rounded-full bg-focus" />
            reset
          </button>
        ) : null}
      </div>
      {spec.kind === "boolean" ? (
        <Switch checked={value === true} onCheckedChange={(v) => m.setParam(name, v)} />
      ) : spec.choices?.length ? (
        <ChoiceControl
          name={name}
          spec={spec}
          value={value}
          onChange={(v) => m.setParam(name, v)}
        />
      ) : (
        <NumberControl
          name={name}
          spec={spec}
          value={value as number}
          onChange={(v) => m.setParam(name, v)}
        />
      )}
      {name === "hbm_bytes" && typeof value === "number" ? (
        <div className="font-mono text-[10.5px] text-muted-foreground">
          = {fmtNum(value / 1e9)} GB · {fmtNum(Number((value / 2 ** 30).toFixed(2)))} GiB
        </div>
      ) : null}
      {error ? (
        <div className="text-[11px] text-bad">
          {String(raw)} rejected: {error}
        </div>
      ) : null}
    </div>
  );
}

function HardwareSelect() {
  const m = useModel();
  if (!m.hardware.length) return null;
  return (
    <div className="space-y-1.5 py-2">
      <div className="text-[12px] font-medium">Hardware preset</div>
      <Select
        value={m.hw?.id ?? ""}
        onValueChange={(id) => {
          const h = m.hardware.find((x) => x.id === id);
          const patch: Record<string, string | number | boolean | undefined> = {
            hw: h && h.device === m.ir.snapshot.device ? undefined : id,
          };
          if (h?.hbm_gb && "hbm_bytes" in m.ir.params) {
            const bytes = h.hbm_gb * 1e9;
            patch.hbm_bytes = bytes === m.envState.defaults.hbm_bytes ? undefined : bytes;
          }
          for (const name of ["non_torch_bytes", "reserved_bytes"] as const) {
            if (h && name in m.ir.params) {
              const value = h[name] ?? 0;
              patch[name] = value === m.envState.defaults[name] ? undefined : value;
            }
          }
          m.patch(patch, true);
        }}
      >
        <SelectTrigger size="sm" className="h-7 w-full text-[12px]" data-testid="hw-select">
          <SelectValue placeholder="Select hardware" />
        </SelectTrigger>
        <SelectContent>
          {m.hardware.map((h) => (
            <SelectItem key={h.id} value={h.id} className="text-[12px]">
              <span className="font-mono">{h.device}</span>
              <span className="text-muted-foreground">{h.label}</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <div className="text-[10.5px] text-muted-foreground">
        Sets device memory, estimated overheads and the roofline ceilings.
      </div>
    </div>
  );
}

export function ParamsPanel({ className }: { className?: string }) {
  const m = useModel();
  const params = m.ir.params;
  const known = new Set(GROUPS.flatMap(([, ks]) => ks));
  const other = Object.keys(params).filter((k) => !known.has(k));
  const groups: [string, string[]][] = [...GROUPS, ["Other", other]]
    .map(([t, ks]) => [t, (ks as string[]).filter((k) => k in params)] as [string, string[]])
    .filter(([, ks]) => ks.length > 0);
  const anyModified = Object.keys(params).some((k) => m.search[k] !== undefined);
  return (
    <aside className={cn("px-3 pb-6", className)} data-testid="params-panel">
      <div className="sticky top-0 z-10 -mx-3 flex h-10 items-center border-b bg-background px-3">
        <span className="text-[12px] font-semibold tracking-wide text-muted-foreground uppercase">
          Parameters
        </span>
        {anyModified ? (
          <Button
            variant="ghost"
            size="xs"
            className="ml-auto text-muted-foreground"
            onClick={() =>
              m.patch(Object.fromEntries(Object.keys(params).map((k) => [k, undefined])), true)
            }
          >
            <RotateCcw /> Reset all
          </Button>
        ) : null}
      </div>
      {m.envState.warnings.length ? (
        <div
          className="mt-3 space-y-1 rounded-md border border-warn/40 p-2 text-[11.5px]"
          data-testid="param-warnings"
        >
          {m.envState.warnings.map((w) => (
            <div key={w.message} className="flex gap-1.5">
              <AlertTriangle className="mt-0.5 size-3 shrink-0 text-warn" />
              {w.message}
            </div>
          ))}
        </div>
      ) : null}
      {groups.map(([title, keys]) => (
        <section key={title} className="mt-3">
          <h3 className="text-[11px] font-medium text-muted-foreground">{title}</h3>
          <div className="divide-y">
            {title === "Memory & scheduler" ? <HardwareSelect /> : null}
            {keys.map((k) => (
              <ParamRow key={k} name={k} spec={params[k] as ParamSpec} />
            ))}
          </div>
        </section>
      ))}
    </aside>
  );
}
