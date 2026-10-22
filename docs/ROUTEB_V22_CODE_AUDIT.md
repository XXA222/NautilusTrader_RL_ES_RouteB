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
| P1 | `imagined_account.py` | 历史回放顺序可错、市场化限价单也错误使用 maker 费率 | 时间单调约束；开盘市场化限价用 taker 情景费率；拒绝反向重放 |
| P1 | `hierarchy_dataset.py`（新） | B3 缺直接对齐数据入口，容易手工错配日/小时未来状态 | 完整桶生成、Train-only scaler、时间可见索引及无未来验证 |
| P2 | `jax_model.py`, `evaluation.py` | NaN 的超参/负价格等输入可能通过基本检查 | 明确非有限/几何失败路径与定向测试 |

## 实际执行证据

| 检查 | 本轮实测 |
|---|---|
| `pytest` 新旧定向测试 | **35 passed**，Python 3.13.5；命令见 README |
| 语法 AST / compileall | PASS，本轮增量 Python 代码与测试 |
| JAX 基础 pytree checkpoint | PASS（环境 JAX 0.9.0.1 CPU；**不是**项目要求的 0.11.2） |
| 固定 DreamerV3 RSSM + Ninjax 实际反向梯度 | **BLOCKED**（本容器没有 Ninjax/Elements/Optax 或固定 upstream checkout） |
| 训练中途恢复逐步等价 | smoke 脚本已补充，**待用户锁定环境验证** |
| Python 3.14.7、Windows、WSL2 CUDA | **未在本环境运行** |
| Nautilus 原生撮合/费用/资金费差分 | **未做**；想象账户不是正式经济模拟 |
| 独立真实 Validation 和收益认证 | **未做**；现有项目数据访问门禁必须保留 |

## 下一阶段明确开发缺口

1. 在固定 Python 3.12.10、JAX 0.11.2、DreamerV3 commit 的环境运行 B2 CPU smoke；若发现 Ninjax API 与实际网络张量不一致，**仅修改独立适配器**，不改 Dreamer 上游源码。
2. 在授权 Train 内实现/记录系统性梯度、损失数值和 batch/sequence 资源曲线；中断恢复要比较每个参数与 Optax 状态数组，而不仅是 loss 数值。
3. B2 单尺度发行前，补严格独立 Validation 读取路径和预注册 E0/E1/E2 分布预测比较；若无授权数据，保留 BLOCKED 而不是输出预测改善结论。
4. B3 仍需实现真正的三级梯度优化器、跨尺度生成器 F2（目前只有时钟与候选损失），并在相同训练数据/预算下进行 E3/E4 消融。
5. B4 轻量账户不可无差分校验直接推导费用后可交易盈利；杠杆、Hedge Mode、资金费、标记价格和极端 K 线成交时序还未受 Nautilus 原生认证。

> 本版没有建立多级哈希链。SHA 仅用于文件损坏/数据标准化身份防错，不声称抵御有意篡改研究准入数据库的对手。

## 2026-10-23 F7 online native summary and Decimal reconciliation

The NativeReportAccumulator parity test was present in the workspace while its source implementation was absent. The new reducer consumes ordered barriers by timestamp and retains only the active same-timestamp group. It incrementally aggregates account facts, deduplicated fills/funding, liquidation order state, source-identified cashflows, and optional unit-linear equity reconciliation. It rejects cashflows at duplicate or initial barriers and checks that all supplied cashflow times correspond to a sample. Batch and online reports are compared on synthetic duplicate-timestamp, cashflow, tail-metric, and same-timestamp fill/funding cases.

Summary-only `rollout()` now feeds compact native snapshots directly into `NativeReportAccumulator`; after each timestamp group is reduced, the factory retains only the final snapshot for terminal order diagnostics. An end-to-end test compares the streaming report with retained-report replay and confirms one retained snapshot. The same pass fixed Decimal context sensitivity in turnover, slippage, and linear-equity arithmetic. Exact helpers lower Emin to MIN_EMIN, clear inherited flags, trap inexact/rounded results, and fail closed when results exceed the helper's 28-digit arithmetic precision. Account return samples are now stored as packed doubles in an auto-deleted temporary file with 64 KiB batch writes. This removes the O(n) Python return list while preserving replay-based exact metrics. The five-suite historical-factory, parity, precision, metric, and liquidation selector passes **175 tests, 18 dependency warnings**; Ruff and targeted MyPy pass. Source manifest: 366 files, `5fc001946596b337cb666de41826924f00460e4d02edec852e17a7506766c08a`. Evidence: `docs/audit_evidence/20261023/m01_native_decimal_exact_arithmetic_20261023.json` and `docs/audit_evidence/20261023/f7_disk_backed_returns_20261023.json`.

F7 remains partial: return facts use O(n) temporary disk storage, and exact tail selection builds a heap proportional to the configured tail fraction, bounded by `max_return_count`; this remains O(qN) peak Python memory at result time. A 200,000-return synthetic storage microbenchmark reduced traced peak allocation from 6,424,384 to 217,154 bytes (96.62%) at 2.36x append latency. Full `series_metrics` replay returned exactly equal values and used 50.2% less additional traced allocation at 1.28x latency. These are microbenchmarks, not a full real-history Train memory/trajectory profile. Requested account rows remain materialized. No real market/account payload was read, and no installed-wheel or clean Hosted CI result was produced. Tests used the main-project `.venv` executable read-only and loaded RouteB workspace sources; this is not proof of RouteB-only environment isolation. The designated `D:\Program Files\NautilusTrader_RL_ES_RouteB` checkout was inaccessible. The verified workspace snapshot and test evidence are backed up in the separate public RouteB repository, branch `backup/routeb-20261023`.
