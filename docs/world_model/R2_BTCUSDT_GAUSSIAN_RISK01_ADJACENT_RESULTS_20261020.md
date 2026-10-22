# R2 BTCUSDT Gaussian rollout-risk strength-0.1 adjacent transfer results

Status: the registered three-seed Validation run completed on 2026-10-20. The candidate improved path-derived 15-minute log-RV CRPS against the Validation-selected E0 in all three seeds, but seed 31 missed the registered Holm-adjusted threshold. The all-seed promotion gate therefore failed. Test/Holdout were not read; no economic claim or R2 promotion follows.

## Scope and integrity

- Binance USD-M BTCUSDT 1-minute source `[2023-11-03T00:00:00Z, 2024-02-05T00:00:00Z)`: 94 official daily archives; all `.CHECKSUM` sidecars and ZIP members passed; 135,360 bars.
- Registered reader used Train+Validation `[2023-11-03T00:01:00Z, 2024-01-21T00:01:00Z)` only. Test `[2024-01-21T00:01:00Z, 2024-02-05T00:01:00Z)` stayed unread (`includes_test=false`, `test_payload_decoded=false`).
- E2: Gaussian prior, native risk prediction loss scale `0.5`, rollout risk-conditioning strength `0.1`, 1,000 updates, batch size 8, seeds 31/102/20261018. E1 used the matched fixed GRU budget.
- 180 Validation anchors, 32 paths/seed, 96 pooled paths, horizons `[1,5,15,30,60]`; E1/E2 paths were valid at every anchor.
- Dataset hash `bae840314107789817f66ae416498e6378b820084574d6ad7a203ce3ab9adf60`; marker SHA-256 `479f36c0d824c3958a95939b0169c0f25219d62ac29e9e84a4d95b2825e529ae`; Parquet SHA-256 `1ce97fbad648aa3611ef70363c5862c98f8a4914246b5e893e94d79e4d23ce89`.
- Semantic Fold hash `b284f278c94e713c5e686698a1d6ef9ab11eb79a0acc4705112763b0077d5607`; aggregate report digest `2f05d47b295cb3cc6cd3ec22de8c45106cc164d881fa0c4d0b0c99515e1287e8`; aggregate file SHA-256 `98a01e0a869a28c596dcdef69193fdc583e9ffb3c861788082c5e76411cff7f3`.
- Independent verification passed for the aggregate and predictive report digests, all three seed report digests, every per-seed E1/E2 forecast and E2 native-risk artifact, and both pooled forecast hashes.
- Immutable output: `.artifacts/world_model/fresh-btcusdt-20231103-gaussian-risk01-adjacent-20261020/experiment-retry-01/`.

## Primary path-derived 15-minute log-RV CRPS

The Validation-selected E0 was `E0_S1_EWMA_VOL_TRAIN_INNOVATIONS`. Lower CRPS is better; negative E2-minus-E0 favors E2.

| Seed | E2 CRPS | E0 CRPS | E2 minus E0 | Paired 95% CI | Holm p |
|---:|---:|---:|---:|---:|---:|
| 31 | 0.246353 | 0.282637 | -0.036284 | [-0.065680, -0.003867] | 0.087956 |
| 102 | 0.201524 | 0.284934 | -0.083410 | [-0.110312, -0.059312] | 0.005497 |
| 20261018 | 0.199643 | 0.280659 | -0.081016 | [-0.116003, -0.054555] | 0.005497 |
| Pooled | 0.204828 | 0.277485 | -0.072657 | [-0.100277, -0.048890] | 0.000500 |

The CI was below zero for each seed, but the registered gate also requires every seed's Holm-adjusted p-value to be below `0.05`. Seed 31 failed that condition, so this transfer does not promote the candidate. Pooled E2 CRPS was lower than E1 (`0.214075`), while E2 beat E1 in two of three seed-specific point estimates; seed variation remains material.

## Diagnostics and disposition

E2 generated rolling-RV mean/p95 were `0.000524`/`0.001047`, below realized `0.000706`/`0.001608`. Generated absolute-return lag-1 autocorrelation was `0.0079` versus realized `0.0642`. Volatility clustering and upper-tail dispersion remain understated despite the lower path score. All generated E2 paths were structurally valid.

The adjacent run confirms a promising aggregate path-score gain but fails the preregistered all-seed rule and does not match realized volatility persistence. Keep R2 unpromoted; R3/R4/R5/R6 remain gated. Stop treating rollout-risk scaling as a promotion candidate and prioritize a training-time path-consistency objective for the next separately registered experiment. No parameter changes or Test access are authorized by this report.
