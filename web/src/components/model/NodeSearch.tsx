import type { IRNode } from "@inferatlas/contracts";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { shortClass } from "@/lib/format";
import { useUi } from "@/lib/store";
import { viewForNode } from "@/lib/views";
import { useModel } from "@/pages/model/context";

const KIND_TITLE: Record<IRNode["kind"], string> = {
  group: "Groups",
  compute: "Compute",
  comm: "Collectives",
  io: "I/O",
};

export function NodeSearch() {
  const m = useModel();
  const open = useUi((s) => s.searchOpen);
  const setOpen = useUi((s) => s.setSearchOpen);
  const kinds: IRNode["kind"][] = ["group", "compute", "comm", "io"];
  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title="Find node"
      description="Search IR nodes"
    >
      <CommandInput
        placeholder="Search nodes by name, id or class…"
        data-testid="node-search-input"
      />
      <CommandList>
        <CommandEmpty>No matching node.</CommandEmpty>
        {kinds.map((k) => {
          const nodes = m.ir.nodes.filter((n) => n.kind === k);
          if (!nodes.length) return null;
          return (
            <CommandGroup key={k} heading={KIND_TITLE[k]}>
              {nodes.map((n) => (
                <CommandItem
                  key={n.id}
                  value={`${n.label} ${n.id} ${n.class}`}
                  className="py-1.5"
                  onSelect={() => {
                    setOpen(false);
                    const where = viewForNode(m.ir, m.view, n.id);
                    if (where) m.goView(where.viewId, n.id);
                    else m.select(n.id);
                  }}
                >
                  <span className="font-medium">{n.label}</span>
                  <span className="font-mono text-[11px] text-muted-foreground">{n.id}</span>
                  <span className="ml-auto truncate font-mono text-[11px] text-muted-foreground">
                    {shortClass(n.class)}
                  </span>
                  {n.ascend?.tag?.length ? (
                    <span className="font-mono text-[10px] text-overlay">
                      {n.ascend.tag.join(" ")}
                    </span>
                  ) : null}
                </CommandItem>
              ))}
            </CommandGroup>
          );
        })}
      </CommandList>
    </CommandDialog>
  );
}
