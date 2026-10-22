# R2 TRXUSDT multi-horizon path CRPS: registered attempt blocked

This is an append-only technical failure record for the frozen interval [2023-04-05, 2023-07-08). It is not predictive evidence and does not consume or inspect Test/Holdout.

## Frozen attempt

- Initial preregistered source identity: 35af314d1b39eaf276d4d770b3f4630427376ec8a5197392fb42153c55f98107.
- Runner identity: aa64f5564abebd6d63ab5fff013c138a3179d9c8651772c85a5b00e2ea6ee7b9.
- One registered Train+Validation read occurred. Registry inspection after the failure found one recorded access row and zero query_intent rows for the exact TRXUSDT interval.
- Registration receipt SHA-256: 88ed7c1b8dc07e9b45a30f2ccf5823dac6b0cd9db9bea5c0ae00a45a0428afd7.
- E1 completed its 1,000 updates and Validation forecast. E2 completed its 1,000 updates and checkpoint creation. The E2 Validation forecast worker then stopped before scoring with checkpoint schema/train/config identity mismatch.
- Runner failure artifact: .artifacts/world_model/fresh-trxusdt-20230405-multihorizon-path-crps-20261020/ensemble/runner_failure.json. Aggregate Test/Holdout decoded flag remains false.

## Root cause and correction

optimizer_recipe() converted risk_horizons and risk_horizon_weights tuples to JSON-ready lists, but did not normalize the new path_log_rv_crps_horizons tuple. The checkpoint writer serialized that tuple as a JSON list; the forecast worker rebuilt a tuple from model configuration and rejected it against the saved list during strict identity comparison. The checkpoint NPZ checksum and model/optimizer arrays were intact.

The correction adds path_log_rv_crps_horizons to the recipe's tuple-to-list normalization. Regression tests now exercise both the JSON-ready optimizer recipe and a full checkpoint save/load roundtrip using (15, 30, 60). Corrected RouteB source identity SHA-256: 593841f700ef138c34a2c49f15e6be021361c8f2770355767a319da0c61dc98c.

Validation: both focused checkpoint/recipe tests passed; Ruff passed. Since the correction changes the frozen code identity after this interval's registered read, this consumed interval will not be retried. Continue with a new preregistered, non-overlapping interval using the corrected source fingerprint.
