#!/usr/bin/env bash
# Block until any codex task has finished and not yet been reported; print it.
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel); cd "$ROOT/.agents"
while :; do
  for d in */; do id=${d%/}
    if [ -f "$id/done" ] && [ ! -f "$id/notified" ]; then touch "$id/notified"; echo "$id: $(cat "$id/done")"; exit 0; fi
  done; sleep 30
done
