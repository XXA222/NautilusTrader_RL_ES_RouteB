# RouteB R4 XRPUSDT E2/E3 comparison results

## Outcome

The registered XRPUSDT Train/Validation comparison completed all six runs (E2 and E3 at seeds 31, 102, and 20261018). The official run report is [`r4_e2_e3_comparison_report.json`](../../.artifacts/world_model/fresh-xrpusdt-r4-e2-e3-20261021/ensemble/run-01/registered/r4_e2_e3_comparison_report.json); its SHA-256 is `0de65d74b3e1e3055b134bccdf9000ca592c8dd0ae1d84303b565f940b53ebe5`.

| Seed | E3 − E2 path log-RV CRPS | 95% paired block-bootstrap CI | Holm p-value | E3 return-CRPS no-harm |
| ---: | ---: | ---: | ---: | --- |
| 31 | -0.00515 | [-0.02896, +0.01918] | 0.87256 | Pass |
| 102 | +0.01049 | [-0.01633, +0.03233] | 0.87256 | Pass |
| 20261018 | -0.03294 | [-0.08018, +0.01390] | 0.51724 | Pass |

Negative primary deltas favor E3. All three confidence intervals include zero, and none of the Holm-adjusted tests passed. Return CRPS met the preregistered no-harm check for all seeds. **E3 was not promoted**; retain E2 as the current default and E3 as a research candidate.

## Cross-market reading

The BNB comparison had two favorable E3 seeds and one inconclusive seed, while XRP produced no statistically reliable primary improvement. Together these results show that E3's BNB gains did not replicate on XRP under this recipe. They support keeping E2 as the default and argue against treating the hierarchy as a generally better architecture from the current evidence. Both studies are system-level comparisons because E2 includes its established risk-head supervision.

## Scope and integrity

- All six candidates completed the frozen budgets: E2 2,067 updates with a 512-minute window; E3 1,000 updates with a 1,024-minute window.
- Scoring used the same 60 Validation anchors and 15-minute horizon. The paired bootstrap used 12-anchor blocks and 2,000 replicates.
- The 23,040 Test rows remained timestamp-only; the report and completion receipt both state `test_payload_decoded=false`.
- This XRP interval is now consumed in the RouteB research ledger. Do not rerun this Fold. A future ablation or replication requires a new non-overlapping preregistered scope.
- This result is not economic certification and does not establish profitability.

## Recommended next work

1. Keep E2 as the serving/default model; do not promote E3 based on the mixed BNB result and null XRP replication.
2. Before spending another full cross-market training budget, isolate the confound in the comparison: E2 has multi-horizon risk-head supervision while E3 does not. A future ablation should add matched auxiliary supervision or compare hierarchy variants under a clearly preregistered objective.
3. Preserve the same sealed-Test rule. Do not inspect either Fold's Test values to explain the differing Validation outcomes.

## Frozen identities

- Protocol: `1e1b95c5d93a69342e8fa6597b6ec68198c66bb3344fc1db31a6dd996e6ccd13`
- Dataset: `46cef5c531cd205fc92a79988b2b32280ab189d48f808210698e710089531929`
- Fold: `6086817db18eb8a1b9de4204997838a2d03fb437b2ee4f9d509f7c35147dca73`
- Completion receipt: [`research_completion_receipt.json`](../../.artifacts/world_model/fresh-xrpusdt-r4-e2-e3-20261021/ensemble/run-01/registered/research_completion_receipt.json)
