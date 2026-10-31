#!/usr/bin/env bash
# One-line status per agent task: state, commits ahead of main, last activity.
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$ROOT"
printf "%-4s %-8s %-6s %-6s %s\n" ID STATE AHEAD FILES LAST
for wt in .worktrees/*/; do
  id=$(basename "$wt"); log=.agents/$id
  if [ -f "$log/done" ]; then st=$(sed 's/ .*//' "$log/done"); else st=running; fi
  ahead=$(git rev-list --count main..task/$id 2>/dev/null || echo -)
  files=$(git -C "$wt" status --porcelain 2>/dev/null | wc -l)
  last=""
  [ -f "$log/events.jsonl" ] && last=$(tail -n 30 "$log/events.jsonl" | python3 -c '
import sys,json
t=""
for l in sys.stdin:
  try: e=json.loads(l)
  except: continue
  it=e.get("item") or {}
  if it.get("type")=="command_execution": t="$ "+(it.get("command") or "")
  elif it.get("type") in ("agent_message","reasoning"): t=(it.get("text") or "")
print(t.replace("\n"," ")[:90])')
  printf "%-4s %-8s %-6s %-6s %s\n" "$id" "$st" "$ahead" "$files" "$last"
done
