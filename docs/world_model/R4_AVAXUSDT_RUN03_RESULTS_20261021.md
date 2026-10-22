# RouteB R4 AVAXUSDT hierarchy comparison — run 03 results

Status: completed registered Train/Validation comparison. This is one Fold of research evidence, not a production or economic certification.

## Scope and integrity

- Fold hash: `3ef84b2879ad7dac3e76ce315cb44672bfb213239ef2076fbdf15e25f5e19a4d`
- Dataset hash: `c2820b9f7a200791accf50833e1196a697d830f09c6de4839c8e20634c27880d`
- Validation: `[2025-07-28T00:01:00Z, 2025-08-12T00:01:00Z)`; 60 common anchors.
- All three variants completed 1,000 optimizer updates for each of seeds 31, 102, and 20261018 on `cuda:0`.
- Every anchor had 32 accepted paths from the common ordered pool of 128, decoded in batches of eight. Rejected candidate counts across 60 anchors were E3 11, E3b 13, and E4 11. No path was clipped.
- The research completion receipt is `completed`. Test/Holdout payloads decoded: `false`.
- Report SHA-256: `6625d54282413a8ec2f97f3e006584fb86bb9879bed7e40db07dcf178dfa0a32`.

## Primary metric: 15-minute path log-RV CRPS

Lower CRPS is better. Differences are E4 minus the comparator; the per-seed p-values are Holm-adjusted across the six primary tests.

| Comparator | Seed | Mean difference | 95% paired block-bootstrap interval | Holm p |
|---|---:|---:|---:|---:|
| E3b | 31 | -0.003253 | [-0.007820, +0.001957] | 0.417791 |
| E3b | 102 | -0.036804 | [-0.047677, -0.025691] | 0.002999 |
| E3b | 20261018 | -0.032628 | [-0.045442, -0.020362] | 0.002999 |
| E3 | 31 | -0.015508 | [-0.026596, -0.004381] | 0.023988 |
| E3 | 102 | +0.042585 | [+0.023412, +0.059706] | 0.002999 |
| E3 | 20261018 | +0.001369 | [-0.009509, +0.015390] | 0.846077 |

Pooled E4-minus-E3b was `-0.024228` (95% interval `[-0.031687, -0.016605]`). Pooled E4-minus-E3 was `+0.009482` (95% interval `[-0.002168, +0.020232]`). The preregistered all-seed E4 gate failed: E4 did not improve on E3b for seed 31 and was significantly worse than E3 for seed 102. The selected research path is therefore **no hierarchy promotion; prefer E3 pending further evidence**.

The return-CRPS guardrail showed no significant E4 harm against either comparator. E4 was better than E3b on the pooled return score (`-0.00004794`, 95% interval `[-0.00005848, -0.00003657]`) and inconclusive against E3 (`+0.00001608`, interval `[-0.00004357, +0.00007384]`).

## Interpretation and next work

The conditional hierarchy has mixed seed-dependent Validation results and does not justify added complexity over E3. Keep E4/E3b as research candidates only. Next compare E3 with the existing RouteB recurrent baseline over additional preregistered, disjoint folds and instruments, inspect calibration and volatility-clustering diagnostics, then decide whether to allocate Test access. Do not use this result to claim profitability or production readiness.
