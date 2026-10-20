# R2 BTCUSDT risk-conditioned rollout results, revision 3

Status: registered Validation experiment completed on 2026-10-20. The primary
gate **failed** in all three seeds. No Test or Holdout payload was decoded;
there is no promotion or economic claim.

## Frozen scope and integrity

- Binance USD-M BTCUSDT one-minute source interval:
  `[2022-04-18T00:00:00Z, 2022-07-21T00:00:00Z)`, 94 archives and 135,360 bars.
- The registered reader accessed Train and Validation only:
  `[2022-04-18T00:01:00Z, 2022-07-05T00:01:00Z)`. Report flags confirm
  `includes_test=false` and `test_payload_decoded=false`.
- E2 used risk prediction scale `0.5`, full risk-conditioned rollout,
  1,000 updates and batch size 8. Three seeds: 31, 102 and 20261018.
- Each seed produced 32 retained paths over 180 anchors. Forecast execution
  used four-anchor chunks with deterministic seed rule
  `request_seed + chunk_start_index`; each worker completed without OOM.
- Dataset hash:
  `8b09e1844b705f06015c0851e43933c0652d1ad15f30fd4c104cffb4121bb0c7`.
- Marker SHA-256:
  `b07f7cef4525d0fce19985433a72b505c54042f6f232bb810073402129df7900`.
- Parquet SHA-256:
  `d7220bbe0c843a396e20ae69222cbaf8c09cba1db30bdb1c649a272064f89924`.
- Fold hash:
  `516c4041bfa442c4395c21a5ed9a1e987ccbd70224ded284191337af0734b23a`.
- Aggregate report digest verified:
  `d0415be7e3613a12904264ef20940ccc6d9c0d5877e639467d62a30d6e34f2e2`.
  All three seed report digests and predictive report digests verify.
  Pooled E1/E2 forecast arrays have shape `[180,96,60,5]` and their recorded
  SHA-256 values match the files.
- Reports and forecasts are under
  `.artifacts/world_model/fresh-btcusdt-20220418-risk-conditioned-20261020/experiment/`.

## Primary path-derived 15-minute log-RV score

The Validation-selected strongest E0 baseline was
`E0_S1_EWMA_VOL_TRAIN_INNOVATIONS` in every seed. Scores are mean per-anchor
CRPS for `log(sqrt(sum(close_log_return**2)) + 1e-8)`. Positive E2-minus-E0 is
worse.

| Seed | E0 mean CRPS | E2 mean CRPS | E2 minus E0 | Paired 95% CI | Raw p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 0.270476 | 0.400370 | +0.129895 | [+0.076901, +0.194254] | 0.000500 | 0.005497 |
| 102 | 0.276001 | 0.366898 | +0.090897 | [+0.035528, +0.159251] | 0.003498 | 0.013993 |
| 20261018 | 0.275420 | 0.433180 | +0.157759 | [+0.095762, +0.228934] | 0.000500 | 0.005497 |

All confidence intervals are strictly above zero and Holm-adjusted p-values
are below 0.05 in the wrong direction. The all-seed improvement rule fails.
E2 was also worse than E1 on this mean score in all three seeds.

## Diagnostics and disposition

Generated E2 rolling realized-volatility mean was about `0.001052`, `0.001292`
and `0.001416` for seeds 102, 31 and 20261018 respectively, against realized
`0.000919`. Its pooled generated mean and p95 were `0.001292` and `0.003011`,
versus realized `0.000919` and `0.001819`. This confirms that full-strength
rollout scale replacement over-expanded volatility in this scope. The native
risk head's direct scores do not repair the path-distribution mismatch.

The report's 11-comparison Holm families include the registered path-derived
primary. No results were used to open Test. This scope is consumed and will
not be reused. The next experiment uses a new zero-overlap BTC interval and a
single predeclared geometric conditioning strength of `0.5`; it is documented
in [`R2_BTCUSDT_RISK_CONDITIONING_TEMPERED_PRE_REGISTRATION_20261020.md`](R2_BTCUSDT_RISK_CONDITIONING_TEMPERED_PRE_REGISTRATION_20261020.md).
