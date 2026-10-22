# R2 BTCUSDT Gaussian rollout-risk strength-0.1 transfer results

Status: the registered three-seed Validation experiment completed on 2026-10-20. Strength `0.1` improved the primary path-derived 15-minute log-RV score against the strongest E0 in two seeds, but seed 20261018 was inconclusive, so the all-seed promotion gate failed. This is promising transfer evidence, not promotion. Test/Holdout remained unread; no economic claim is made.

## Scope and integrity

- Binance USD-M BTCUSDT 1-minute source `[2023-08-01T00:00:00Z, 2023-11-03T00:00:00Z)`: 94 archives; all official `.CHECKSUM` sidecars and ZIP members passed; 135,360 bars.
- Registered reader accessed Train+Validation `[2023-08-01T00:01:00Z, 2023-10-19T00:01:00Z)` only: 112,320 bars. Test `[2023-10-19T00:01:00Z, 2023-11-03T00:01:00Z)` was not decoded.
- E2: Gaussian prior, native risk loss scale `0.5`, rollout conditioning strength `0.1`, 1,000 updates, batch size 8, seeds 31, 102, 20261018.
- 180 Validation anchors, 32 paths per seed, 96 pooled paths, shape `[180,96,60,5]`; all generated paths valid.
- Dataset hash `edb8996498f9636a74f9d18b8d4d1794053dea88a9a7e901bd2186f29b2f63ea`; marker SHA-256 `07027ea80189516165063051b24c60d9af4d24697a9181ad935e0edc90697235`; Parquet SHA-256 `e6df1054917cbd83a47e6b4f34e645b4cffb01bc80d7354d36c8e64cca13f665`.
- Semantic Fold hash `2fa5e4d9adc00f003296beb4220297fb8f8f67ca65cadbefa341157c5cb38286`; aggregate report SHA-256 `1ebfd273379220c14a88ee5a7a947e72ccb84fad82d05f564bf9d435f9f794de`.
- Independent aggregate/predictive digest, all seed reports, all per-seed forecast and risk artifacts, and both pooled file hashes verify. Test flags are false throughout.
- Immutable output: `.artifacts/world_model/fresh-btcusdt-20230801-gaussian-risk01-20261020/experiment/`.

## Primary path-derived 15-minute log-RV CRPS

Strongest Validation-selected E0 was `E0_S2_STATIONARY_BLOCK_BOOTSTRAP`. Lower CRPS is better; negative E2-minus-E0 favors E2.

| Seed | E2 minus E0 | Paired 95% CI | Raw p | Holm p |
|---:|---:|---:|---:|---:|
| 31 | -0.107479 | [-0.160911, -0.057549] | 0.000500 | 0.005497 |
| 102 | -0.087662 | [-0.139144, -0.037144] | 0.001000 | 0.005497 |
| 20261018 | -0.045661 | [-0.105167, +0.016583] | 0.146427 | 0.146427 |
| Overall | -0.081502 | [-0.135187, -0.028448] | 0.005497 | — |

E2 mean CRPS was `0.275592`; selected E0 was `0.357094`. E1 mean CRPS was lower at `0.244568`. The all-seed rule fails because seed 20261018's interval crosses zero and adjusted p exceeds `0.05`.

## Diagnostics and disposition

E2 generated rolling-RV mean/p95 were `0.000402`/`0.000779`, versus realized `0.000358`/`0.000805`. Generated absolute-return lag-1 autocorrelation was `0.0141` versus realized `0.0728`, so path volatility clustering remains weak. Strength `0.1` improves E2 versus the selected E0 on two seeds in this higher-volatility scope, but does not beat E1's average path score and does not pass the registered all-seed gate. Keep R2 unpromoted; R3/R4/R5/R6 remain gated. Continue a single frozen adjacent transfer of strength `0.1` before deciding whether to retain the candidate.