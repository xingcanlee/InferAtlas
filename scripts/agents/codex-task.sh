#!/usr/bin/env bash
# Run one Codex task in its own git worktree/branch.
# Usage: scripts/agents/codex-task.sh <task-id> <prompt-file>
set -euo pipefail
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ID=$1; PROMPT=$(realpath "$2"); BASE=${3:-main}
WT="$ROOT/.worktrees/$ID"; LOG="$ROOT/.agents/$ID"
mkdir -p "$LOG"
if [ ! -d "$WT" ]; then
  git -C "$ROOT" worktree add -q -b "task/$ID" "$WT" "$BASE"
fi
cd "$WT"
codex exec -m gpt-6.1-sol -c model_reasoning_effort=high \
  --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
  -C "$WT" --json -o "$LOG/last.md" - < "$PROMPT" > "$LOG/events.jsonl" 2> "$LOG/stderr.log" && rc=0 || rc=$?
echo "exit=$rc branch=task/$ID head=$(git -C "$WT" rev-parse --short HEAD)" | tee "$LOG/done"
