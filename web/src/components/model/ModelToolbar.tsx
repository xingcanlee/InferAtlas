import { Link } from "@tanstack/react-router";
import { ChevronRight, ExternalLink } from "lucide-react";
import { StatusBadge } from "@/components/StatusBadge";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { shortSha } from "@/lib/format";
import type { Heat } from "@/lib/search";
import { cn } from "@/lib/utils";
import { useModel } from "@/pages/model/context";

function Seg<T extends string>({
  value,
  onChange,
  items,
  label,
  testId,
}: {
  value: T;
  onChange: (v: T) => void;
  items: [T, string][];
  label: string;
  testId?: string;
}) {
  return (
    <ToggleGroup
      type="single"
      variant="outline"
      size="sm"
      value={value}
      onValueChange={(v) => v && onChange(v as T)}
      aria-label={label}
      data-testid={testId}
    >
      {items.map(([v, text]) => (
        <ToggleGroupItem key={v} value={v} className="h-7 px-2.5 text-[12px]">
          {text}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}

function Commit({ repo, tag, commit }: { repo: string; tag: string; commit: string }) {
  const url = `https://github.com/vllm-project/${repo}/commit/${commit}`;
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center gap-1 rounded border px-1.5 py-px font-mono text-[11px] text-muted-foreground hover:text-foreground"
      title={`${repo} ${tag} @ ${commit}`}
    >
      {repo} <span className="text-foreground">{tag}</span>@{shortSha(commit)}
      <ExternalLink className="size-2.5" />
    </a>
  );
}

export function ModelToolbar({ className }: { className?: string }) {
  const m = useModel();
  const snap = m.ir.snapshot;
  const checks = m.report?.checks ?? [];
  const failed = checks.filter((c) => c.status === "fail").length;
  const reportStatus = !m.report ? undefined : failed ? "fail" : "pass";
  return (
    <div
      className={cn("flex flex-wrap items-center gap-x-4 gap-y-2 border-b px-3 py-2", className)}
    >
      <div className="flex min-w-0 items-center gap-2">
        <Link to="/" className="text-muted-foreground hover:text-foreground">
          Matrix
        </Link>
        <ChevronRight className="size-3.5 text-muted-foreground" />
        <h1 className="truncate text-[15px] font-semibold tracking-tight" data-testid="model-title">
          {snap.model.hf_id}
        </h1>
        <span className="rounded border px-1.5 font-mono text-[11px] text-muted-foreground">
          {snap.device}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <Commit repo="vllm" tag={snap.vllm.tag} commit={snap.vllm.commit} />
        <Commit repo="vllm-ascend" tag={snap.ascend.tag} commit={snap.ascend.commit} />
        {reportStatus ? (
          <Tooltip>
            <TooltipTrigger>
              <StatusBadge status={reportStatus} count={failed || undefined} />
            </TooltipTrigger>
            <TooltipContent className="max-w-sm">
              {checks.map((c) => (
                <div key={`${c.id}-${c.node ?? ""}`}>
                  <span className="font-mono">{c.id}</span>: {c.status} — {c.message}
                </div>
              ))}
            </TooltipContent>
          </Tooltip>
        ) : null}
        {m.cell.status === "triage" ? (
          <StatusBadge status="triage" count={m.cell.triage_count ?? undefined} />
        ) : null}
      </div>
      <div className="ml-auto flex flex-wrap items-center gap-3">
        <Seg
          label="Phase"
          testId="phase-toggle"
          value={m.phase}
          onChange={(v) => m.patch({ phase: v === "prefill" ? undefined : v })}
          items={[
            ["prefill", "Prefill"],
            ["decode", "Decode"],
          ]}
        />
        <Seg
          label="Shapes"
          value={m.shapes}
          onChange={(v) => m.patch({ shapes: v === "num" ? undefined : v })}
          items={[
            ["num", "123"],
            ["sym", "T·H"],
          ]}
        />
        <Seg<Heat>
          label="Heat bar metric"
          value={m.heat}
          onChange={(v) => m.patch({ heat: v === "flops" ? undefined : v })}
          items={[
            ["flops", "FLOPs"],
            ["weight", "Weights"],
            ["bytes", "HBM"],
          ]}
        />
        <div className="flex items-center gap-2">
          <Switch
            id="overlay"
            checked={m.overlay}
            onCheckedChange={(v) => m.patch({ overlay: v || undefined })}
            data-testid="overlay-toggle"
          />
          <Label htmlFor="overlay" className="text-[12px] font-normal">
            Ascend overlay
          </Label>
        </div>
      </div>
    </div>
  );
}
