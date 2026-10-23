# Route B v2.2 源码加固审查记录

## 基线与范围

- 参考仓库：`XXA222/NautilusTrader-RL-ES`，固定提交 `80194a587444bf4a11f06f6e273723e5754f2483`。
- 上游 DreamerV3 固定提交 `e01491fad6434b2245a3b8ca201dd7faedcc458c`，本包不修改上游文件。
- 本次只针对上一轮**独立增量源码包**的 `rles_world_model/routeb` 开发。未读取本机真实 Train/Validation/Test/Holdout，也未修改你的实际仓库。

## P0/P1 缺陷及修复证据

| 等级 | 定位 | 根因 | 修复和覆盖 |
|---|---|---|---|
| P0 | `sequences.py`, `jax_model.py` | 末尾目标被 mask 丢弃、预测目标与 burn-in 掩码配合错误 | 末尾纳入 score、只用目标时点 mask；测试监督目标数量 |
| P0 | `evaluation.py:return_paths` | `log(close/open)` 忽略前收盘到本开盘的 gap | 强制提供锚点，累计 close-to-close；非零 gap 金标 |
| P0 | `checkpoint.py`, `jax_train.py` | `steps` / 报告间隔进入冻结优化 recipe 导致不能从 2 步续训到 4 步 | 锁算法超参，训练目标步数可延长；smoke 加完整逐数组续训对照（待运行） |
| P1 | `checkpoint.py` | 只核 leaf 数量/形状，无法拒绝相同形状下不同的 PyTree | v2 记录具体 PyTree 结构；模拟损坏、相同形状不同键的拒绝测试 |
| P1 | `codec.py`, `sequences.py`, `jax_train.py` | 冻结 scaler 未绑定训练模型，预测时可能误用别的标准化参数 | scaler 指纹加入 Train 元数据和 checkpoint identity，并绑定预测请求；拒绝篡改测试 |
| P1 | `worker_cli.py`, `integration.py` | 上下文无观测可用时间序列，容易误配已闭合边界 | forecast-request v2 携带时间戳数组、标准化器身份及 SHA，worker 预校验 |
| P1 | `worker_client.py` | `COMPLETED` 文字可能掩盖旧路径/错误生成结果 | 校验路径、SHA、输出文件结构、OHLCV 数学约束 |
| P1 | `synthetic_feedback.py` | 缺历史日内前缀时未来日线可以被静默跳过 | 每个应更新的整点/UTC 零点必须形成完整桶，否则失败；完整数据不受影响 |
| P1 | `imagined_account.py` | 历史可用时间未绑定决策屏障，且单净仓账户可混入不同 instrument | 收盘连续性、availability 单调性、订单决策不早于最后数据可用时间、首根 K 线锁定 instrument；市场化限价用 taker 情景费率；拒绝反向重放 |
| P1 | `hierarchy_dataset.py`（新） | B3 缺直接对齐数据入口，容易手工错配日/小时未来状态 | 完整桶生成、Train-only scaler、时间可见索引及无未来验证 |
| P2 | `jax_model.py`, `evaluation.py` | NaN 的超参/负价格等输入可能通过基本检查 | 明确非有限/几何失败路径与定向测试 |

## 实际执行证据

| 检查 | 本轮实测 |
|---|---|
| RouteB F7 五组回归 | **175 passed, 18 dependency warnings, 106.36s**；Python 3.14.7 main `.venv` executable, RouteB source via `PYTHONPATH` |
| 语法 AST / compileall | PASS，本轮增量 Python 代码与测试 |
| 固定环境 B2 CPU smoke | **PASS**；Python 3.12.10, JAX/JAXLIB 0.11.2, Ninjax 3.5.1, Elements 3.19.1, Optax 0.2.8, pinned DreamerV3 commit; real optimizer/checkpoint/forecast/worker path |
| 固定环境 B3 CPU smoke | **PASS**；合成三时间尺度数据，分钟/小时/日线 RSSM 梯度均非零，有限损失，断点恢复与不中断训练检查点逐数组一致；不含真实行情/收益结论 |
| 训练中途恢复逐步等价 | **PASS on synthetic mechanism data**；2-step resume to 4 steps exactly matches all uninterrupted checkpoint arrays |
| Windows / WSL2 CUDA | Windows x64 CPU smoke passed; WSL2 CUDA smoke **not run** |
| Nautilus 原生撮合/费用/资金费差分 | **未做**；想象账户不是正式经济模拟 |
| 独立真实 Validation 和收益认证 | **未做**；现有项目数据访问门禁必须保留 |

## 下一阶段明确开发缺口

1. availability 时间戳到 F2 事件的因果映射、UTC close + 显式延迟日程、同 barrier 多桶拒绝和无事件场景已覆盖；F2 现从 `MultiScaleClock` 生成日程，并在状态可用的决策 barrier 立即影响对应分钟预测。延迟由调用方作为冻结假设提供，尚无真实 feed 的延迟校准证据。
2. B2 单尺度发行前，继续验证严格独立 Validation 读取路径和预注册 E0/E1/E2 分布预测比较；若无授权数据，保留 BLOCKED，不输出预测改善结论。
3. B3 三级梯度优化器及 F2 递归先验生成已实现并通过合成 CPU smoke；事件计划扰动测试确认未来事件变化不会改写更早预测。仍需执行相同授权 Train 数据/固定预算的 E3/E4 消融。
4. B4 轻量账户不可无差分校验直接推导费用后可交易盈利；杠杆、Hedge Mode、资金费、标记价格和极端 K 线成交时序还未受 Nautilus 原生认证。

B2 smoke exact output and environment pins are archived in `docs/audit_evidence/20261023/b2_cpu_smoke_20261023.json` and `docs/audit_evidence/20261023/b2_cpu_smoke_result_20261023.json`. The smoke uses synthetic bars and explicitly reports `real_market_validation=false` and `economic_claim=false`.

## 2026-10-23 B3 three-scale gradient training

Added `jax_hierarchy_train.py` with deterministic causal minute windows, Train-tensor/scaler/checkpoint identity binding, Adam plus gradient clipping, finite metric rejection, and exact checkpoint resume. The shared hourly/daily streams are broadcast across sampled minute windows while retaining absolute causal availability indexes; all three RSSMs receive gradients. Fixed the model's pinned-upstream import order and DreamerV3 bfloat16 network input contract.

`qa/world_model/routeb_b3_cpu_smoke.py` passed on the locked Python 3.12.10 CPU environment using 4,317 synthetic 1m bars. A two-step optimizer run split at step 1 and resumed to step 2 matched the uninterrupted step-2 checkpoint arrays exactly; all three RSSM gradient norms were nonzero (minute 285.33, hour 22.23, day 68.71). Follow-up review found and fixed a one-decision F2 lag: a due hour/day prior now conditions the forecast emitted at that same decision barrier. `MultiScaleClock` projects close-plus-delay events from stored last-observed slow close timestamps without future OHLCV. Zero-delay and one-minute-delay end-to-end runs confirm preceding forecasts are bit-identical while the first forecast at each event barrier changes; the clock regression also checks a two-minute delay. All `tests/world_model` tests pass: **128 passed, 3 JAX-only skipped**; B3's pinned JAX CPU smoke passes separately. Ruff, targeted MyPy, and compileall pass. Evidence: `docs/audit_evidence/20261023/b3_f2_clock_alignment_20261023.json`.

This closes the B3 optimizer, clock-driven synthetic F2 wiring, and decision-barrier alignment code gaps. Real-feed latency calibration, learning-quality evidence on authorized Train data, independent E3/E4 ablations, and native economic parity remain open; this synthetic smoke establishes no economics.

## 2026-10-23 B4 imagined-account availability boundary

Audit found that `ImaginedAccount.submit()` rejected decisions older than the last candle close but ignored a delayed `available_at_ns`; this could let a strategy act on a bar before the bar was actually observable. The single-net-position account also accepted bars from multiple instrument IDs. The account now tracks the last availability barrier, rejects backdated decisions, rejects availability regressions, and binds replay to the first candle's instrument. Synthetic regressions cover delayed availability through the next eligible bar open, cross-instrument input and fail-before-state-mutation behavior.

Targeted tests pass with the RouteB world-model suite; Ruff passes and the RouteB source manifest is refreshed. This is a causal-input correctness fix only. `economics_eligible` remains false; native NautilusTrader differential parity for fills, fees, account/funding, leverage, liquidation and edge-bar sequencing is still required before any economic claim.

> 本版没有建立多级哈希链。SHA 仅用于文件损坏/数据标准化身份防错，不声称抵御有意篡改研究准入数据库的对手。

## 2026-10-23 F7 online native summary and Decimal reconciliation

The NativeReportAccumulator parity test was present in the workspace while its source implementation was absent. The new reducer consumes ordered barriers by timestamp and retains only the active same-timestamp group. It incrementally aggregates account facts, deduplicated fills/funding, liquidation order state, source-identified cashflows, and optional unit-linear equity reconciliation. It rejects cashflows at duplicate or initial barriers and checks that all supplied cashflow times correspond to a sample. Batch and online reports are compared on synthetic duplicate-timestamp, cashflow, tail-metric, and same-timestamp fill/funding cases.

Summary-only `rollout()` now feeds compact native snapshots directly into `NativeReportAccumulator`; after each timestamp group is reduced, the factory retains only the final snapshot for terminal order diagnostics. An end-to-end test compares the streaming report with retained-report replay and confirms one retained snapshot. The same pass fixed Decimal context sensitivity in turnover, slippage, and linear-equity arithmetic. Exact helpers lower Emin to MIN_EMIN, clear inherited flags, trap inexact/rounded results, and fail closed when results exceed the helper's 28-digit arithmetic precision. Account return samples are now stored as packed doubles in an auto-deleted temporary file with 64 KiB batch writes. This removes the O(n) Python return list while preserving replay-based exact metrics. The five-suite historical-factory, parity, precision, metric, and liquidation selector passes **175 tests, 18 dependency warnings**; Ruff and targeted MyPy pass. Source manifest: 366 files, `5fc001946596b337cb666de41826924f00460e4d02edec852e17a7506766c08a`. Evidence: `docs/audit_evidence/20261023/m01_native_decimal_exact_arithmetic_20261023.json` and `docs/audit_evidence/20261023/f7_disk_backed_returns_20261023.json`.

F7 remains partial: return facts use O(n) temporary disk storage, and exact tail selection builds a heap proportional to the configured tail fraction, bounded by `max_return_count`; this remains O(qN) peak Python memory at result time. A 200,000-return synthetic storage microbenchmark reduced traced peak allocation from 6,424,384 to 217,154 bytes (96.62%) at 2.36x append latency. Full `series_metrics` replay returned exactly equal values and used 50.2% less additional traced allocation at 1.28x latency. These are microbenchmarks, not a full real-history Train memory/trajectory profile. Requested account rows remain materialized. No real market/account payload was read, and no installed-wheel or clean Hosted CI result was produced. Tests used the main-project `.venv` executable read-only and loaded RouteB workspace sources; this is not proof of RouteB-only environment isolation. The designated `D:\Program Files\NautilusTrader_RL_ES_RouteB` checkout was inaccessible. The verified workspace snapshot and test evidence are backed up in the separate public RouteB repository, branch `backup/routeb-20261023`.

## 2026-10-23 F7 bounded-memory tail-selection follow-up

Replaced the disk-backed return path's tail-sized `heapq.nsmallest()` allocation with fixed-size sorted runs and bounded-fan-in external merge. Each merge keeps at most the exact lowest `K` values; compaction bounds Python heap and open-file counts as the sequence grows. The in-memory list API retains its existing `heapq.nsmallest()` behavior.

Verification: 91 focused report/metric tests passed with one Nautilus `Money` integration test deselected; native liquidation/history integration could not import `nautilus_trader.model` or `nautilus_trader.backtest` in this Python 3.12 workspace. Ruff and MyPy passed. A 1,000,000-return synthetic differential probe with a 50,000-observation tail matched VaR and CVaR exactly and reduced `tracemalloc` peak from 6,500,920 bytes to 1,275,986 bytes. External merge took 2.378s versus 0.518s for heap selection while `tracemalloc` was active; actual historical Train throughput remains unprofiled. Evidence is recorded in `docs/audit_evidence/20261023/f7_bounded_tail_selection_20261023.json`.
