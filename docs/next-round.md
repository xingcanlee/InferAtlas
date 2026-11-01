# Next round (from final release review, 2026-11-01)

## UI (Claude frontend)
1. Default graph zoom crops content at 1280×800: center on first drillable group when nothing selected; allow fit down to ~0.65 for ≤12-node views; collapsible bottom panel; e2e asserting all nodes in canvas before Fit. Optional: elk.aspectRatio ~2.2 (regenerate goldens).
2. Single-file build should open directly on the embedded model, not a one-cell matrix.
3. Deep links to triage cells say "missing" — derive message from cell.status/triage_count.

## Generator / release (Codex)
4. Single-file default name must include device; drop device-less duplicates; fix smoke spec path.
5. Add a triage kind for extraction failures (DeepSeek-V3 A2 rotary cache needs NPU device on CPU mock).
6. Schema doc: IndexCell.data_path null also for triage cells; regenerate contracts.
7. release.spec.ts: assert graph content fit, record console errors.

## Gaps vs design goals
- Calibration ±10% unverified (needs real NPU/GPU serve logs; calibrate.yml self-hosted runner).
- Diff view needs a second real snapshot (v0.32.0 pair build).
- DeepSeek-V3 / MLA rules missing (M1 promised 5 golden models; have 2).
- A3 / A5 / 310P not built; "H100" = preset over CPU-extracted facts.
- Ascend annotations `reviewed_by: pending` — need human review.
- M3 automation (cron poll → build → Pages deploy, draft PRs) written but not yet run on GitHub; site not deployed.
