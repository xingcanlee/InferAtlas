# Next round (from final release review, 2026-11-01)

## UI (Claude frontend)
1. **Done (F2 + H2 release verification).** Whole graph fits on first load at 1280×800; analysis panel collapses/resizes; header items do not overlap. Browser checks cover all A2 views at four desktop sizes. No ELK/golden change was needed.
2. Single-file build should open directly on the embedded model, not a one-cell matrix.
3. Deep links to triage cells say "missing" — derive message from cell.status/triage_count.

## Generator / release (Codex)
4. **Done (H2).** Single-file defaults include the device; removed the web builder's device-less fallback copy and updated the smoke path. Release exports cover A2 and H100 with no device-less duplicates.
5. Add a triage kind for extraction failures (DeepSeek-V3 A2 rotary cache needs NPU device on CPU mock).
6. **Done (H2).** IndexCell.data_path documents null for missing cells and triage cells without numeric bundles; JSON Schema and TypeScript contracts regenerated.
7. **Done (F2 + H2).** Release checks assert graph fit and fail on unexpected console errors or page exceptions; all four standalone exports are checked offline.

## Gaps vs design goals
- Calibration ±10% unverified (needs real NPU/GPU serve logs; calibrate.yml self-hosted runner).
- Diff view needs a second real snapshot (v0.32.0 pair build).
- DeepSeek-V3 / MLA rules missing (M1 promised 5 golden models; have 2).
- A3 / A5 / 310P not built; "H100" = preset over CPU-extracted facts.
- Ascend annotations `reviewed_by: pending` — need human review.
- M3 automation (cron poll → build → Pages deploy, draft PRs) written but not yet run on GitHub; site not deployed.
