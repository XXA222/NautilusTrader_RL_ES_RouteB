# R2 ETHUSDT risk-scale-0.5 adjacent-period confirmation pre-registration

Date frozen: 2026-10-19, before downloading or reading this source scope. The risk-scale-0.5 July-October 2023 transfer run favored E2 by point estimate against the strongest baseline, but its paired interval crossed zero. This adjacent-period run provides one independent confirmation of the same fixed recipe. The source interval starts exactly at the previous Fold's Test score end; none of that earlier Test interval is reused. The new Train/Validation end is before the next legacy ETHUSDT access recorded in RouteB's ledger.

## Fixed candidate and hypothesis

Use the same Gaussian E2 configuration, `risk_prediction_scale=0.5`, along with unchanged E1/E0, model, data features, risk targets, optimizer and forecast protocol. No parameter changes, scale sweep, or extra candidate are permitted. Hypothesis: the prior 15-minute CRPS point improvement and native 30/60-minute log-realized-volatility gains persist in the next chronological window. The primary gate remains a paired 95% interval favoring E2 with Holm-adjusted p < 0.05 against the strongest statistical baseline selected on Validation. Secondary risk results cannot override that gate.

## Frozen scope and protocol

- Source: Binance USD-M ETHUSDT 1-minute klines, open times `[2023-10-16T00:00:00Z, 2024-01-17T00:00:00Z)`, 93 days / 133,920 expected bars. Download daily archives October 16 through January 16 inclusive.
- RouteB registry was checked read-only before freezing. Proposed Train/Validation read scope `[2023-10-16T00:01:00Z, 2024-01-01T00:01:00Z)` has no overlapping ETHUSDT perpetual-market access or query intent. It is adjacent to, and begins at, the previous test-score boundary; it does not reuse that Test. The registered one-shot reader remains authoritative.
- 60/15/15 Fold: Train warmup October 16 00:01; Train score October 17 through December 16 00:01; Validation warmup December 6 00:01; Validation score December 17 through January 1 00:01; Test warmup December 22 00:01; Test score January 2 through January 17 00:01. Preserve 1-hour purge and 23-hour embargo; Test remains undecoded.
- Three seeds `31, 102, 20261018`; 1,000 optimizer updates per seed; 32 paths per seed; 180 two-hour Validation anchors; 96 pooled paths per candidate; 2,000 paired block-bootstrap replicates with 12-anchor blocks.
- Secondary: 30/60-minute native log-realized-volatility CRPS/Brier, free-rollout absolute-return lag-1 autocorrelation, rolling-RV mean/p95, path validity. No Test/Holdout access or economic claim.

## Frozen implementation/provenance

- E1: `configs/routeb/fresh_ethusdt_e1_ensemble.json`.
- E2: `configs/routeb/fresh_ethusdt_e2_risk05_ensemble.json`, SHA-256 `e64938514b7ed10ae375b2dc42e100100772a9f406deb8fbd50b299606ca60e9`.
- RouteB HEAD `cb7a3ec9d542ed48b9f182acbaac330fc90b3177`; source identity SHA-256 `e1e08a244a8e95e3c3dfd26c1c1e15765a0ec0d6114cc322a257afd63df656db`.
- Source archive checksums, curated marker/dataset/Parquet identities, and Fold hash will be recorded here before the registered reader accesses Train/Validation. The scope is consumed after its one reservation. R3/R4/R5/R6 remain gated unless the primary rule passes.

## Frozen source identities recorded before reservation

- Download/prepare receipt: `docs/world_model` source artifact `source_preparation.json`, SHA-256 `e9387aaa3804848decc0dd4d818b619e2a4d20aec8fb330e5751db04503561b0`; all 93 archive files matched their official checksum sidecars.
- Prepared source dataset hash `68e29f58f1fe3c81af4c0e741b8838e057d531eabb693ffd32fc6754ae987d37`, marker SHA-256 `6cf4ca844440d00fc9d848cd8124347408a530033132129dcb7a2dc8e2ba7e23`, source file SHA-256 `c40d0ae4cf77505dcc013478bd56e4fc520ba50a3b8f3a921fffb132d5d62c7f`; Parquet SHA-256 `c40d0ae4cf77505dcc013478bd56e4fc520ba50a3b8f3a921fffb132d5d62c7f`; `133920` bars.
- Registered Fold config: `configs/routeb/fresh_ethusdt_20231016_risk05_adjacent_60_15_15_fold.json`, SHA-256 `f305f99f6b6c0805440e52c6616aae980d19f1ffd5433b3454e2ed84712ddc3e`.
- Read-only access and query-intent ledger preflight was repeated for `[2023-10-16T00:01:00Z, 2024-01-01T00:01:00Z)` immediately before reservation; both overlap counts were zero. The registered one-shot reader remains authoritative.
- RouteB source manifest after adding the frozen Fold config: 439 files, identity SHA-256 `e768ecf12a98a79b57e6ab97d813b562d57634267fc1c79770d2cd7dd1ba0361`; verification passed before reservation.
