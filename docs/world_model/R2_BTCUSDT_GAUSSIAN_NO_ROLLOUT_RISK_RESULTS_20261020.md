# R2 BTCUSDT Gaussian rollout-conditioning ablation results

Status: the registered three-seed Validation experiment completed on 2026-10-20. The primary path-derived 15-minute log-RV gate is inconclusive in all seeds. The rollout ablation reduced the large upper-tail expansion seen in the Student-t run, but paths remain under-dispersed and this recipe is not promoted. Test/Holdout were not decoded; no economic claim is made.

## Scope and integrity

- Binance USD-M BTCUSDT one-minute source `[2023-01-25T00:00:00Z, 2023-04-29T00:00:00Z)`: 94 daily archives; all official checksum sidecars and ZIP integrity checks passed; 135,360 bars curated.
- Registered reader accessed Train+Validation only, `[2023-01-25T00:01:00Z, 2023-04-14T00:01:00Z)`, 112,320 bars. The Test score interval `[2023-04-14T00:01:00Z, 2023-04-29T00:01:00Z)` remains unread.
- E2 retained Gaussian prior and native risk loss scale `0.5`, with rollout risk conditioning disabled; 1,000 updates and batch size 8 for seeds 31, 102, 20261018.
- 180 Validation anchors, 32 paths per seed, 96 pooled paths; arrays are `[180,96,60,5]`. All generated paths are valid.
- Dataset hash `d00cd88db1b3b623ce38cde6ab6a97a8dc4f6acee08e5f7bb788f8d40ee2e95a`; marker SHA-256 `ca05a6a393cb9f9f33994e491458a2b7d147dd69789385934d474996c122c4d7`; Parquet SHA-256 `2cba49ceb75ad86f162dd2c24bcbb50163ba74f40579a0206dd0602b93a9b65b`.
- Semantic Fold hash `63eb0971017b97eef3622e466e27b7cd72106be2e09f22050138d5e30575dc35`; aggregate report SHA-256 `787f2ce05c2dfe38b937c854c7e49f7b9ed5bdf7bf3f5dbee7899d3ca726a75e`.
- Independent verification passed for aggregate and predictive report digests, all three seed report digests, all per-seed forecast/risk files and both pooled forecasts. All flags show `includes_test=false`, `test_payload_decoded=false`.
- Immutable outputs: `.artifacts/world_model/fresh-btcusdt-20230125-gaussian-no-rollout-risk-20261020/experiment/`.

## Primary path-derived 15-minute log-RV CRPS

The strongest Validation-selected E0 was `E0_S1_EWMA_VOL_TRAIN_INNOVATIONS` for all seeds. Lower CRPS is better; positive E2-minus-E0 is worse.

| Seed | E2 minus E0 | Paired 95% CI | Raw p | Holm p |
|---:|---:|---:|---:|---:|
| 31 | +0.030780 | [-0.027740, +0.086327] | 0.282859 | 1.000000 |
| 102 | -0.013814 | [-0.057909, +0.030714] | 0.559720 | 1.000000 |
| 20261018 | +0.001799 | [-0.044009, +0.046355] | 0.932034 | 1.000000 |
| Overall | +0.004771 | [-0.045653, +0.052999] | 0.850075 | — |

Every interval crosses zero and every corrected p-value is 1.0. The registered gate fails. E2 mean path log-RV CRPS was `0.295556` versus E0's `0.290785` and E1's `0.312697`.

## Diagnostics and disposition

E2 generated rolling-RV mean/p95 were `0.000426`/`0.000809`, below realized `0.000499`/`0.001089`. Absolute-return lag-1 autocorrelation was `0.0090` against realized `0.1221`. Disabling rollout scaling removed the strong high-volatility expansion seen in the preceding Student-t experiment, but did not restore clustered volatility or improve the primary score. Native risk scores were mixed; none overrides the registered path gate.

Conclusion: rollout risk scaling contributed to over-expansion under the tested Student-t setup, while unconditioned Gaussian rollouts remain under-dispersed. No R2 promotion. Continue R2 with a single frozen low-strength Gaussian conditioning candidate on a new ledger-clear scope; R3/R4/R5/R6 remain gated. See the preregistration for frozen identities.