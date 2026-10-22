# RouteB R4 E2/E3 cross-market synthesis

## Decision

E3 did not pass the preregistered all-seed promotion gate on either BNBUSDT or XRPUSDT. Keep E2 as the default. E3 remains a research candidate; these Validation-only experiments do not justify a general hierarchy advantage or an economic claim.

| Market | E3-favoring primary seeds | Inconclusive / opposite point estimate | Return-CRPS no-harm | Promotion |
| --- | --- | --- | --- | --- |
| BNBUSDT | 31, 102; both Holm-adjusted p < 0.05 | 20261018 point estimate favors E2; CI crosses zero | Pass, all seeds | No |
| XRPUSDT | None statistically reliable | All three CIs cross zero; seed 102 point estimate favors E2 | Pass, all seeds | No |

Detailed seed metrics are in the [BNB results](R4_BNBUSDT_E2_E3_BASELINE_RESULTS_20261021.md) and [XRP results](R4_XRPUSDT_E2_E3_BASELINE_RESULTS_20261021.md). Both reports link the immutable run JSON, report hash, source identity, and ledger completion receipt.

## Interpretation

The BNB improvement did not replicate on XRP under the same frozen target budget, anchor schedule, scoring, and validation process. This is evidence of market/seed sensitivity in the current setup, not evidence that E3 is universally inferior. The all-seed gate correctly blocks promotion.

The comparison remains system-level. E2 receives multi-horizon risk-head supervision, while E3 does not. Before another full multi-market replication, match or separately ablate this auxiliary objective. Any architecture claim must be based on a design that isolates the hierarchy effect.

## Integrity limits

- Each market used its own no-overlap RouteB ledger scope and sealed-source Fold.
- Both Test periods remain timestamp-only and were not decoded.
- E2 completed 2,067 updates with a 512-minute window; E3 completed 1,000 updates with a 1,024-minute window. Train target budgets matched within 0.02%.
- Neither study is economic certification; fees, funding, execution, and instrument-rule evidence were not supplied.

## Next work

1. Keep E2 as the default and retain E3 without promotion.
2. Design an objective-matched ablation that gives E2 and E3 equivalent auxiliary risk supervision, or remove the confound under a newly preregistered comparison.
3. Reuse neither consumed market interval. Check the RouteB ledger before selecting any new scope, freeze the recipe before downloading data, and preserve timestamp-only Test sealing.
