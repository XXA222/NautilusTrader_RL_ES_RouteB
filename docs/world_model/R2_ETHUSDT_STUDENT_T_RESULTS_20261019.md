# R2 ETHUSDT variance-matched Student-t results

Date: 2026-10-19
Status: completed; primary promotion gate not met

## Frozen experiment

This one-shot run tested the preregistered E2 causal prior observation head using a fixed Student-t distribution (`nu=5`) with variance-matched marginal standard deviation. E1, E0 statistical baselines, RSSM risk heads and loss weights, optimizer budget, and forecast protocol were frozen as recorded in [`R2_ETHUSDT_STUDENT_T_PRE_REGISTRATION_20261019.md`](R2_ETHUSDT_STUDENT_T_PRE_REGISTRATION_20261019.md).

- Source: ETHUSDT USD-M 1-minute Binance klines, 2022-10-01 through 2023-01-01; 93 daily archives, 133,920 continuous bars.
- Registry scope: `routeb-fold-ethusdt-binance-601515-20221001-v1`; Train/Validation read `[2022-10-01T00:01:00Z, 2022-12-17T00:01:00Z)`.
- Approved 60/15/15 Fold; 180 Validation anchors, 32 paths per seed, 96 pooled paths per candidate, 2,000 paired bootstrap replicates, 12-anchor blocks.
- Seeds 31, 102 and 20261018; 1,000 updates per model and seed.
- The report records `test_payload_decoded=false` and the registered receipt records `includes_test=false`. No model selection or economic certification was performed.

## Primary result

The strongest statistical 15-minute baseline was E0 S1 (EWMA volatility with Train innovations), CRPS `0.0009550564`. E2 Student-t CRPS was `0.0009921716`, so E2 was worse by `+0.0000371152`. The paired 95% block-bootstrap interval was `[+0.0000051702, +0.0000640337]`; raw p was `0.01449`, but Holm-adjusted p was `0.23188`. The primary gate fails. E2 versus E1 was inconclusive: difference `-0.0000122693`, 95% CI `[-0.0000497481, +0.0000195201]`, Holm p `1.0`.

R2 is not promoted. R3/R4/R5/R6 remain closed under the v3.0 gate.

## Secondary risk evidence

The native risk head's 30- and 60-minute log-realized-volatility CRPS comparisons favored E2 in every seed after the registered family-wise adjustment:

| Seed | Paired E2 native-risk comparison |
|---:|---|
| 102 | 30m: Δ=-0.1218, CI [-0.1638, -0.0752], Holm p=0.0050; 60m: Δ=-0.2195, CI [-0.2744, -0.1524], Holm p=0.0050 |
| 20261018 | 30m: Δ=-0.1309, CI [-0.1760, -0.0776], Holm p=0.0050; 60m: Δ=-0.2232, CI [-0.2952, -0.1762], Holm p=0.0050 |
| 31 | 30m: Δ=-0.1244, CI [-0.1604, -0.0790], Holm p=0.0050; 60m: Δ=-0.2221, CI [-0.2715, -0.1569], Holm p=0.0050 |

This is evidence for conditional long-horizon risk prediction, not a substitute for the primary return-path gate. F1 free rollouts remained weak in volatility persistence. Realized absolute-return lag-1 autocorrelation was `0.11310`; E2 generated `0.00862` (E1 `0.04397`). Realized rolling-volatility mean/p95 were `0.00048405` / `0.00097378`; E2 generated `0.00046747` / `0.00086145`. E2 brought the mean close but still understated the p95 and clustering.

## Integrity and outputs

- Dataset SHA-256: `fad492bb174357be17cab66970b8e8f0e815ba13d80a417e5ac77a8fa530ed5f`.
- Source marker SHA-256: `eea1f54f83bf8199daedff79171b011ed98a9430e5ecc95222f1aa25b1b197ab`.
- Parquet SHA-256: `6651b2df555df887415117589dc83525ce016241f7760f8e1410d94f51fea21a`.
- Fold SHA-256: `2c9e1c745b0b89a12b742017e2f9dd668bff1eca5cf55d5d251c9cbd1da91743`.
- Aggregate report: `.artifacts/world_model/fresh_ethusdt_20221001_student_t_ensemble_20261019/multi_seed_fold_experiment_report.json`.
- Embedded canonical digest: `76a6d633f148b31b7c45cc96ccb64b9649ec602434fb2197857007c7b60731d0`; raw report SHA-256: `952a302df7bd8c8058704509e5f0fa8b80c3e7af8acb3d00e80788b26a933b97`.
- Aggregate and seed reports for 31, 102, 20261018 verify with RouteB's canonical verifier and UTF-8 decoding. Both pooled forecast files match the hashes in the report.
- Test payload remains undecoded; no market economics were inferred.

## Decision

Retain the Student-t model as an engineering and secondary-risk result, not as the selected E2 forecast. The primary predictive gate did not pass, so no higher research stage is authorized. Continue R2 with one preregistered two-component Gaussian-mixture prior head on a new, never-consumed scope, as suggested by the v3.0 design's distribution-head sequence. Do not tune the Student-t degrees of freedom on this scope.
