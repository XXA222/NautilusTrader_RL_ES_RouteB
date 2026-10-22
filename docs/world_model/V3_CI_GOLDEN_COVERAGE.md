# Route B v3 CI golden coverage

This matrix maps the 18 acceptance cases in §9.1 of the v3.0 design to
executable checks. The Route B suite is now an explicit PR/nightly/release CI
step. The pinned Python 3.12/JAX CPU job runs the five JAX-dependent unit tests
that the Python 3.14 host environment skips, followed by the synthetic prior,
gradient, resume, and rollout checks. These checks do not use market payloads.

The workspace Python 3.14 environment intentionally does not install JAX. Run
the full suite there for non-JAX coverage, then run the five skipped node IDs
through Route B's locked `packages/world_model/environment` Python 3.12 worker.
CI runs both sets explicitly; a host report with JAX skips alone is not the
complete world-model verification result.

| # | Golden | Automated evidence | State |
|---:|---|---|---|
| 1 | `causal_future_mutation` | `test_scheduler_prior_does_not_depend_on_current_or_future_bar_values`; pinned `routeb_prior_gradient_smoke.py` future mutation check | Covered in unit suite and worker smoke |
| 2 | `posterior_prior_split` | Pinned `routeb_prior_gradient_smoke.py` perturbs future observations and compares the aligned prior distribution | Covered in worker smoke |
| 3 | `prior_gradient` | Pinned `routeb_prior_gradient_smoke.py` checks finite nonzero prior-head and transition gradients | Covered in worker smoke |
| 4 | `burn_in` | `test_scoring_includes_first_post_burn_in_target`; `test_masks_include_last_target_and_end_at_train_boundary` | Covered in CI suite |
| 5 | `cross_gap` | `test_risk_targets_horizons_and_gaps_are_masked_per_horizon`; `test_gap_fails_closed_and_restart_requires_explicit_contiguous_warmup` | Covered in CI suite |
| 6 | `delayed_hour` | `test_late_available_bar_cannot_advance_aggregator_or_slow_state`; `test_delayed_hourly_bucket_visibility_no_future` | Covered in CI suite |
| 7 | `utc_day` | `test_day_state_updates_only_after_all_1440_utc_minute_bars`; `test_daily_and_hourly_states_update_together_only_after_midnight_bar` | Covered in CI suite |
| 8 | `bucket_complete` | `test_hourly_and_daily_features_require_complete_contiguous_aggregates`; `test_train_sequence_builder_rejects_incomplete_multiscale_warmup` | Covered in CI suite |
| 9 | `no_action_market` | `test_trade_action_cannot_change_market_transition_input`; RSSM adapters also assert the dummy action is zero | Covered in CI suite |
| 10 | `prior_rollout_path` | Pinned prior smoke validates finite `[2,H,5]` free rollouts at H=1,5,15,60; decoder checks legal multi-minute paths | Covered in worker smoke and CI suite |
| 11 | `F2_synthetic_only` | `test_synthetic_cross_hour_uses_only_simulated_future`; `test_synthetic_day_close_event` | Covered for the synthetic bucket builder |
| 12 | `F1_F2_separation` | `test_comparison_report_integrates_generation_and_f2_checks` keeps path diagnostics and F2 aggregation checks in separate report fields | Partial: distinct F1/F2 mode sampling remains deferred to R5 |
| 13 | `valid_ohlcv` | `test_ohlcv_target_roundtrips_and_tick_quantization_preserves_envelope`; `test_invalid_ohlc_path_rejected_even_with_positive_closes`; generation diagnostics report invalid rates | Covered in CI suite |
| 14 | `horizon_targets` | `test_risk_targets_horizons_and_gaps_are_masked_per_horizon`; `test_horizon_specific_barriers_are_applied_to_matching_targets` | Covered in CI suite |
| 15 | `R2_negative` | Pinned prior smoke reports finite constant-input loss and latent variance; no NaN is accepted | Covered in worker smoke |
| 16 | `checkpoint_resume` | Pinned B2 CPU smoke compares all checkpoint leaves, optimizer state, sampler order, and PRNG seeds after 2+2 vs 4 updates | Covered in worker smoke |
| 17 | `native_limit_fill` | Imagined-account limit/open-gap/cancel scenario tests cover its declared mechanics | Deferred: Native differential execution parity belongs to R5 and is not claimed |
| 18 | `research_scope` | `test_routeb_fresh_fold_access` and `test_research_access` reject prior-scope reuse and unauthorized reads | Covered in PR contract tests and release suite |

Cases 12 and 17 are explicitly not promoted to PASS. The v3.0 handoff defers
end-to-end F1/F2 research and Native parity to R5; the current implementation
does not expose F2 as a trained forecast mode and does not certify the imagined
account as NautilusTrader-equivalent.
