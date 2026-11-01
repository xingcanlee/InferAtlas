# Next round (from final release review, 2026-11-01)

## UI (Claude frontend)
1. **Done (F2 + H2 release verification).** Whole graph fits on first load at 1280×800; analysis panel collapses/resizes; header items do not overlap. Browser checks cover all A2 views at four desktop sizes. No ELK/golden change was needed.
2. **Done (F3 + H3).** Single files open directly on the embedded model. The offline hub links all four exports, and “All models” returns to the sibling `index.html`; the real release is checked over file:// with no network.
3. **Done (F3 + H3).** Deep links derive their message from cell.status/triage_count and say “Needs triage” with the unresolved count and no numeric data. Both site and hub keep blocked cells non-clickable.

## Generator / release (Codex)
4. **Done (H2).** Single-file defaults include the device; removed the web builder's device-less fallback copy and updated the smoke path. Release exports cover A2 and H100 with no device-less duplicates.
5. **Done (G1 + H3).** `extraction_failure` records stage, exception and normalized message. The real DeepSeek-V3 A2 release records the CPU mock's NPU rotary-cache failure; H100 extracts but remains blocked by MLA lowering/verification/layout.
6. **Done (H2).** IndexCell.data_path documents null for missing cells and triage cells without numeric bundles; JSON Schema and TypeScript contracts regenerated.
7. **Done (F2 + H2).** Release checks assert graph fit and fail on unexpected console errors or page exceptions; all four standalone exports are checked offline.
8. **Done (G1 + F3 + H3).** `just release` exports the catalog × A2/H100 and builds `dist/index.html`, an offline model/device hub with four sibling HTML links and honest DeepSeek triage cells. Export defaults choose the smallest valid TP with positive device KV capacity (MoE A2: TP=2); other defaults are preserved and the rule is recorded in IR/report provenance.

## Gaps vs design goals
- Calibration ±10% unverified (needs real NPU/GPU serve logs; calibrate.yml self-hosted runner).
- Diff view needs a second real snapshot (v0.32.0 pair build).
- DeepSeek-V3 / MLA rules missing (M1 promised 5 golden models; have 2).
- A3 / A5 / 310P not built; "H100" = preset over CPU-extracted facts.
- Ascend annotations `reviewed_by: pending` — need human review.
- M3 automation (cron poll → build → Pages deploy, draft PRs) written but not yet run on GitHub; site not deployed.
