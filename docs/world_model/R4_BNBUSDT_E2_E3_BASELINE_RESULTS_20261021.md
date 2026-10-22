# RouteB R4 BNBUSDT E2/E3 comparison results

## Outcome

The registered BNBUSDT Train/Validation comparison completed all six runs (E2 and E3 at seeds 31, 102, and 20261018). The official run report is [`r4_e2_e3_comparison_report.json`](../../.artifacts/world_model/fresh-bnbusdt-r4-e2-e3-20261021/ensemble/run-01/registered/r4_e2_e3_comparison_report.json); its SHA-256 is `0d2f885deb0c14a04f54cac1b36a707baf4ee863c01119b3c1d6f3d3e8951aa6`.

| Seed | E3 − E2 path log-RV CRPS | 95% paired block-bootstrap CI | Holm p-value | E3 return-CRPS no-harm |
| ---: | ---: | ---: | ---: | --- |
| 31 | -0.04287 | [-0.07461, -0.01100] | 0.01899 | Pass |
| 102 | -0.09665 | [-0.12166, -0.07384] | 0.00150 | Pass |
| 20261018 | +0.01751 | [-0.01272, +0.04833] | 0.28386 | Pass |

Negative primary deltas favor E3. Two seeds improved, while the third was inconclusive and had a positive point estimate. Return CRPS met the preregistered no-harm check for all seeds. Because promotion requires all three primary confidence intervals and Holm-adjusted tests to pass, **E3 was not promoted**. Keep E2 as the current default and E3 as a research candidate.

## Scope and integrity

- This is a cross-instrument system comparison, not an architecture-only ablation: E2 includes its established multi-horizon risk-head supervision.
- Each candidate completed its frozen budget: E2 2,067 updates with a 512-minute window; E3 1,000 updates with a 1,024-minute window. Train target counts were matched within 0.02%.
- All scoring used the same 60 Validation anchors and 15-minute horizon. Paired bootstrap used 12-anchor blocks and 2,000 replicates.
- The 23,040 Test rows remained timestamp-only; the report and completion receipt both state `test_payload_decoded=false`.
- The BNB interval is now consumed in the RouteB research ledger. Any replication must first select and preregister a fresh non-overlapping scope; do not rerun this Fold.
- This result is not economic certification and does not establish profitability.

## Recommended next work

1. Preserve E2 as the serving/default model. Do not treat the two favorable seeds as sufficient promotion evidence given the third seed's opposite point estimate.
2. The preregistered XRP replication is now complete; its seed-level evidence and synthesis are recorded in [`R4_XRPUSDT_E2_E3_BASELINE_RESULTS_20261021.md`](R4_XRPUSDT_E2_E3_BASELINE_RESULTS_20261021.md) and [`R4_E2_E3_CROSS_MARKET_SYNTHESIS_20261021.md`](R4_E2_E3_CROSS_MARKET_SYNTHESIS_20261021.md).
3. Before spending another full training budget, isolate the comparison confound: E2 includes multi-horizon risk-head supervision while E3 does not. A future ablation should match auxiliary objectives or clearly limit claims to system-level comparison.
4. Keep both Test payloads sealed. Do not use either Test period to explain the different Validation outcomes; a future test-stage study needs its own explicit preregistration and untouched evaluation plan.

## Frozen identities

- Protocol: `96ae10b6a7cbdd8adb6a0c1442c49f204e8790b67b09c0507bc0ca855d38062d`
- Dataset: `73051c11f09d7731bbc2954b0539ab2d27d8b84b31cb5c47398c471e5a487c8b`
- Fold: `bc94a0e87818474fd96ddd7d90564131610e1b8cc9665e0cf2bbf6bca2f26292`
- Completion receipt: [`research_completion_receipt.json`](../../.artifacts/world_model/fresh-bnbusdt-r4-e2-e3-20261021/ensemble/run-01/registered/research_completion_receipt.json)
