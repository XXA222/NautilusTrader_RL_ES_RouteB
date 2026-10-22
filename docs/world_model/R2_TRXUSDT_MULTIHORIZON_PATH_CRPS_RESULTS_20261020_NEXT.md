# R2 TRXUSDT multi-horizon Train path log-RV CRPS results — 2023-07-08 scope

Status: completed registered Train/Validation experiment. Primary promotion gate **NOT PASSED**. Test/Holdout were not decoded. No economics or R2 promotion claim.

## Frozen run and integrity

- Source: Binance USD-M TRXUSDT 1m, 94 daily archives, 135,360 closed bars, 2023-07-08 through 2023-10-09. Dataset hash `46ce6923d4053637f18cdd907d8dc4073f4a8a807433f7658f504f2e4b90fe3a`; source marker `8c47d4854a553c489bf9366c19ee6d0e3d69758acc91ffdcfab43ae2beab18a0`.
- Split and model recipe match the preregistration: 60/15/15 days, 1h purge, 23h embargo, three seeds `[31,102,20261018]`, E1/E2 each 1,000 optimizer updates, 180 Validation anchors, 32 paths per seed, 96 pooled paths.
- All six candidate fits used `cuda:0`; JAX/JAXlib 0.11.2. Test access and Test query-intent counts are both zero. Train+Validation has one reserved access and no formal query intent.
- Independently verified aggregate and each seed's canonical report digest, all checkpoint/config/train-sequence/forecast/risk artifact hashes, pooled forecast hashes and pooled OHLCV candle geometry. Pooled shape for E1 and E2 is `[180,96,60,5]`; generated bars and paths are 100% valid.
- Aggregate file SHA-256: `dca25db0965fe1fbc3333a0e5d8ac3f59ce0edb12efd5b5265120b1e8979a2ef`; canonical report SHA-256: `941fb7c9f14a5cb6a4ab0bbb07be601f8e593edb2efccd71f6b4d70cd88ae014`.

## Primary 15-minute path log-RV CRPS

The strongest Validation statistical baseline was E0 S2 stationary block bootstrap (mean CRPS `0.257915`); E1 scored `0.212183`, and E2 scored `0.235569`.

| Seed | E2 − E0 S2 mean | Paired 95% CI | Holm p | Seed gate |
|---|---:|---:|---:|---|
| 31 | `+0.001743` | `[-0.023012,+0.033626]` | `1.0000` | Fail |
| 102 | `-0.036625` | `[-0.061300,-0.010901]` | `0.02399` | Pass |
| 20261018 | `-0.034168` | `[-0.057267,-0.006440]` | `0.04398` | Pass |
| Pooled 96 paths | `-0.022346` | `[-0.042868,+0.002877]` | `0.05847` | Inconclusive |

The preregistered gate requires all three seed intervals below zero and all three Holm p-values below 0.05. Seed 31 fails and the pooled interval crosses zero, so E2 is not promoted. E1 remains the lower-CRPS Validation candidate in this scope.

## Guardrails and diagnostics

- E2-minus-E0 S2 15m return CRPS: `-0.00001453`, 95% CI `[-0.00003162,+0.00000569]`, Holm p `1.0`; there is no statistically significant corrected harm.
- E2 generated 15m rolling realized-volatility mean/p95 were `0.0002697`/`0.0004339`; realized values were `0.0003387`/`0.0006201`. Generated absolute-return lag-1 autocorrelation was `0.00836` versus realized `0.06156`. Free rollouts remain under-dispersed and reproduce volatility clustering weakly.
- E2 native 30m/60m log-RV head gains appear in seeds 102 and 20261018 but not seed 31; this does not override the failed path-level gate.

## Verified artifacts

- Preregistration: [`R2_TRXUSDT_MULTIHORIZON_PATH_CRPS_PRE_REGISTRATION_20261020_NEXT.md`](R2_TRXUSDT_MULTIHORIZON_PATH_CRPS_PRE_REGISTRATION_20261020_NEXT.md)
- Aggregate report: `../../.artifacts/world_model/fresh-trxusdt-20230708-multihorizon-path-crps-20261020/ensemble/multi_seed_fold_experiment_report.json`
- Pooled E1/E2 Validation paths: `pooled_e1_validation_forecasts.npz`, `pooled_e2_validation_forecasts.npz` in the same ensemble directory.

R2 remains open; R3/R4/R5/R6 remain gated. This experiment only supports further frozen research on a fresh scope.
