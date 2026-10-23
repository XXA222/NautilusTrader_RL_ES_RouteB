# 2026-10-17 审计整改：可连续开发目标

**Codex 活动开发目标入口**

可直接交给 Codex Goal 的完整自动执行指令和唯一续做游标见 [DEVELOPMENT_GOAL_CURRENT.md](DEVELOPMENT_GOAL_CURRENT.md)。本文件负责报告映射、任务队列、依赖与验收证据；历史执行细节见 [DEVELOPMENT_PLAN_20261017.md](DEVELOPMENT_PLAN_20261017.md)。发生摘要冲突时，以当前游标文档和仓库现状为准，不能把历史任务记录当成未完成代码缺口。

## 依据与当前基线

- 输入：用户提供的《NautilusTrader-RL-ES 源码深度审计与后续开发建议》（2026-10-17）。报告依据远程提交 `c24d1bc4a47584f91a4c4db93bf77117d4e8341c` 做静态审计。将报告建议转化为本目标中的任务、依赖、验收条件和执行顺序；Codex 后续实际执行仍须遵守下文研究数据门禁。
- 当前工作区含较多未提交开发成果。`DEVELOPMENT_PLAN_20261017.md` 记录了此前已做的 A01/A02/A03/A05/A08/A10 局部整改、142 项与 52 项定向测试，以及 gamma 配对实验未通过门禁的结果。本计划只规定**尚待完成或重新验收**的工作，不把报告中的旧基线描述直接当作当前代码事实。
- 所有历史费用、过滤规则、流动性和清算假设须保留来源与有效期。现有回放中的 scenario-only 结果不能表述为真实可交易盈利。正式 Test/Holdout 的已有访问记录继续有效；制定计划不消耗新的正式样本。
- 初次编制目标时的只读核对确认：同时间戳归并已存在；E2 两维物理中性已有单测；`FitnessSpec` 仍采用加权惩罚，硬约束筛选未闭环；项目级访问登记已存在；DSR 仍使用当前候选数并注明范围。以前记录的 **142 passed / 52 passed** 是历史定向结果，不能作为当前全量或两平台 CI 通过证据；后续执行结果均以本目标对应的开发计划与证据为准。
- 报告的 C/L 项优先核实现状；R/H 项必须保留待验证性质。不得将“可能导致亏损”“可能有性能瓶颈”直接写成已经复现的经济或性能结论。

## 总目标与完成定义

建立从数据与原生账户事实、Actor 动作、风控与执行，到 ES/RL 目标和独立验证的可追溯闭环。完成时应能回答：一次策略动作为什么产生或没有产生订单、最终成交多少、费用与 Funding 如何进入权益、各训练阶段是否提供独立的费用后改进，以及结论使用了多少真正独立的市场样本和研究尝试。

每个任务的交付物须包括必要源码、确定性回归、运行配置/数据身份、实际执行结果、失败或未认证边界。工程任务的完成门禁是相关单元与集成测试及适用范围 Ruff、MyPy 通过；平台交付和研究晋级再检查对应 Linux/Windows CI 与经济门禁。固定回放除预期经济语义变化外保持一致；不以训练 Fitness、环境步数或计划层 `SUBMIT` 替代原生费用后经济结果。涉及性能时同时报告耗时、峰值 RSS 与结果等价性。实验按冻结协议执行、产出可复核拒绝结论，也可完成该实验任务；候选未通过不能写成通过收益门，更不能靠无限调参维持任务。

## 连续任务队列与续做游标

**当前工程续做：M0.1 原生金额解析及 M0.2 分发覆盖/manifest 范围缺口已修复。** A08 现在拒绝会被旧逗号清理逻辑改写的畸形 native money token；研究指标与原生清算回归 **83 passed**。分发 smoke 覆盖当前构建中的全部 **12 个 RLES 包**，包括 `rles_world_model`；source manifest 排除嵌套 `.tmp`、`.venv*` 和 `dist*` 生成目录。2026-10-18 又修复 GitHub 工作流硬编码 11 wheels 的问题，改为按 workspace 包数核对；wheel smoke 现逐项导入 12 包声明的 7 个 console/plugin entry points 并检查归档来源。当前 Windows 默认 selector **669 passed、1 CUDA-only skipped、102 deselected、46 warnings**，Ruff 和 `mypy packages qa`（148 files）通过；12 wheels/12 sdists 构建及 wheel manifest **all_match=true**，隔离 wheel/entry-point smoke 通过。M0.1 的真实现金流真实性与完整性、M0.2 clean Hosted Linux/Windows CI、M1.2 独立 Validation、M1.3 兼容阶段输入、M2.3 真实深度、M3.2 新授权独立覆盖仍受各自外部门禁限制；继续寻找其它已具备条件的代码缺口。

**2026-10-22 当前状态刷新：** F7/A07 的 RouteB 原生环境合成 1/30/90 天报告快照保留/精简对照均逐步轨迹一致，报告状态仅为 synthetic engineering evidence；峰值 RSS retained/lean 为 79.97/0.36、254.13/15.78、618.17/136.90 MiB。R1 学习工厂现关闭训练和内部 eval 不使用的报告快照，外部经济回放保留默认快照。焦点回归 4 passed；Ruff 通过、MyPy 212 source files clean。当前 source manifest 558 files，identity `967411f88f4dc2f589c293911c0fc11c2a74fa13c9fca5b69f9835be25036459`；12 wheel + 12 sdist `all_match=true`，RouteB 独立 app venv 的 12 package / 6 entry point provenance smoke 通过。本地结果仍不替代真实长窗账本与 clean Hosted CI。

**Route B WorldModel 当前状态（2026-10-21 核验）：** 上述早期 B0/B1-only 状态已被后续开发取代，不再适用。B0–B3 的数据边界、RSSM/JAX 训练、checkpoint resume、E2 风险监督、R3 latent regularizer、R4 三尺度 E3/E3b/E4 与 F2 合成反馈均已有工程实现；RouteB 隔离 CPU/JAX 回归 **14 passed**，app 主环境组合回归中 JAX 项仍单列跳过。R2/R3 的多窗/消融已有登记 Validation 结果但候选未晋级；R4 AVAX run-03 完成后 E4 未通过全种子主门，E3 仍待与既有 recurrent baseline 做新增独立 Fold/跨标的验证。没有 Test/Holdout 解封、生产启用或经济认证。状态与剩余门槛见 `docs/world_model/V3_IMPLEMENTATION_STATUS.md`、`docs/ROUTEB_V22_CODE_AUDIT.md` 和 [当前连续开发游标](DEVELOPMENT_GOAL_CURRENT.md)。

**最新工程进展：N04 supervised GPU quota 正常释放和后代保护回归已通过。** Windows Job 的 PID 清单同时含有 Python launcher parent 和实际 owner；现在释放前先确认共享 GPU 内核锁已空闲，并确认 Job 内其它活动 PID 都是 owner 的祖先，活动应用后代会令 quota release 拒绝。`tests/test_resource_recovery.py tests/test_compute_device.py` **27 passed**，Ruff 与 Windows MyPy 通过。N04 仍不构成 VRAM 上限或跨 OS-user/平台认证；下一项仍按任务表选择依赖已满足的真实代码缺口。

**M2.2/A12 决策级费用归因补全：** 执行轨迹此前把 native fill commission 中币种后的金额直接相加；现在按币种汇总，单币种仍提供兼容 `fee` 与 `fee_currency`，多币种只提供 `fees_by_currency` 并将无单位总额置为 `null`。费用格式、非有限值及混合币种回归通过；完整 `tests/test_historical_factory.py` **59 passed**，Ruff 与 Windows MyPy 通过。没有读取市场或账户数据；本改动修复报告单位语义，不扩展或认证多标的能力。

**M2.2/A12 决策权益归因补全：** Gym 步骤和可选逐决策 trace 现记录策略 barrier 前权益、策略执行后的 barrier 权益、准确差额与币种，并检查权益变化复算一致。终端强平变化继续单列，其期初权益必须与最后一次策略 barrier 的权益一致。完整 Gym bridge 与历史工厂套件 **79 passed、18 dependency warnings**；Ruff 与 Windows MyPy 通过。只用合成原生引擎回归，未访问市场、账户或正式评价 payload，不构成收益认证。

终止结算报告也要求 drain 前后账户币种一致，合成快照不匹配回归与 Gym bridge/rollout 组合 **22 passed**，Windows MyPy 与 Ruff 通过。该防线只校验同一账户的记账单位，没有声称支持多币种账户。

**A02 最新代码缺口已修复：** Constraint-priority v1/v2 与 native ES worker 现在把有限负终值传入破产可行性筛选，拒绝原因为 `episode_insolvency_detected`；旧 Fitness 版本仍保持既有负权益校验。Worker 还验证正权益 solvent episode 的逐步对数收益能复算首末权益。Scorer/dispatcher/worker 回归 **16 passed**，完整 compute-worker pool **30 passed**，原生训练/ES worker **29 passed**，Ruff 与 Windows MyPy 通过。该项仍需独立 Validation 才能评估目标晋级，工程回归没有打开该窗口。

**数据隔离工程游标：Stage69 的真实 Train-inner 特征和 Reset profile 均已 PASS。** v0.4.1 scoped reader 与 ledger-first `read_registered_train_window()` 已接入 `application.prepare()`，训练仅允许 `evaluate=False`，market-audit/WFO/backtest 拒绝 scoped Train-only 配置。60/360 日特征 profile 结果见 [feature 证据包](audit_evidence/20261017/stage69_feature_profile_360d_v2/)。冷/热 Reset 为 60 日 **3.1787/0.0098 秒**、360 日 **21.5383/0.0838 秒**，same-seed 起始 observation/native account fingerprint 重复一致；读取前登记的物理隔离 v0.4.0 快照共 117 个 row group、1,907,766 条事件，正式区间重叠为 0，结果见 [Reset 证据包](audit_evidence/20261017/stage69_episode_reset_360d_v1/)。规则为 scenario-only，没有训练或经济收益结论。旧 M2.4 profile 使用跨正式日期的大文件整读，不作为 Stage69 Train-only 数据证据引用。下一步转到 M3.2/A13 训练覆盖报告源码复核；Stage60 日期已有复用记录，不能当作新独立市场覆盖。

M0.1 已核对空仓、开仓 mark-to-market、手续费、成交、Funding、终止平仓和原生线性清算。NautilusTrader 2.0.0rc6 的 `AccountState` 没有标准化 deposit/withdrawal 类型；Engine 中 `reported=True` 既可和初始账户状态一起出现，也可和 Funding settlement 同时出现，因此不能解释成转账。本轮新增调用方显式提供的现金流事实接口，要求事件 ID、时间、币种、方向、金额、来源 ID 与来源 SHA-256，并将 signed flow 纳入 return 与逐屏障线性权益对账；不从 `reported` 或权益变化推断转账。SHA 只固定来源标识，不证明源文件真实性。无真实类型化来源时，未知或相互抵消的现金流仍不能从原生余额推断，M0.1 保持 `PARTIAL`。旧证据见 [90 天 Train-inner](audit_evidence/20261017/m0_1_native_equity_90d.json)、[Engine 线性强平](audit_evidence/20261017/m0_1_linear_native_liquidation.json) 与[合成清算交叉案例](audit_evidence/20261017/m0_1_synthetic_linear_liquidation.json)；本轮回归见执行计划。

M0.2 将有来源/哈希的 1 秒 BookTicker Quote 用例纳入默认 PR `micro_golden` 标记。本地 Windows 上实际执行工作流相同选择器 **478 passed、60 deselected、38 warnings，115.26 秒**；Quote 用例通过。`.github/workflows/l6-ci-reusable.yml` 的 Linux/Windows 远程 job 尚未运行，须在干净交付后验收。

M1.1 已通过实现等价验收。M1.2 默认保留旧 `FitnessSpec`；新 `constraint_priority_train_inner_v1` 先筛 episode 破产、维持保证金不足和原生强平，再按有界费用后经济分数排名。保证金/强平分类只在原生快照声明能力且事实完整时启用；新协议遇到未知事实失败关闭。HardRisk 干预频率不作为违规。拒绝原因贯穿候选聚合、新协议 checkpoint 与最终 bundle lineage；真实 A4 E2 中断恢复验证了报告持久性。三种目标已在相同 Train-inner 归档、冻结 R1 actor、共同随机数场景及相同预算下分别完成 A2 搜索和同参数原生回放；三组均无不可行候选，约束优先与零干预惩罚选中相同参数，Train-inner 净收益均为 -0.1510%，legacy 为 -0.1578%。规则仅为 scenario-only，未做 Validation，因此 M1.2 保持 `PARTIAL`，不能由此推断目标优劣或策略盈利。最近定向回归 **17 passed、13 warnings**；Ruff 和 MyPy（4 个源码文件）通过。详见 [M1.1 证据](audit_evidence/20261017/m1_1_neutral_evaluation.json)、[M1.2 实现证据](audit_evidence/20261017/m1_2_constraint_priority.json)、[三目标 Train-inner 证据](audit_evidence/20261017/m1_2_three_objective_train_inner.json)。

状态定义：`PARTIAL` 表示已有实现但验收不完整；`READY` 表示当前可开展工程工作；`PENDING` 表示依赖未满足；`CONDITIONAL` 表示依赖数据、独立样本或平台证据；`DEFERRED` 表示暂缓。表中记录当前已知状态；每次实际执行后按证据更新状态、游标和执行记录。

| 顺序 / ID | 对应审计项 | 当前状态 | 依赖 / 完成证据 |
|---|---|---|---|
| 01 / M0.1 | A08 | PARTIAL，等待已认证且覆盖可证明的真实现金流采集 | 显式外部现金流输入、身份/时间/来源哈希校验、Binance USDⓈ-M `TRANSFER` JSON 导入及平仓/持仓权益核对已有回归；分页归档入口校验 inclusive 页段连续覆盖、越界事件与潜在截断，manifest 绑定每页原始 SHA；乱序 native snapshots 按时间统一排列 fill/Funding/清算生命周期；清算订单累计 `filled_qty` 拒绝回退/超量；Transfer JSON 单页/分页解码拒绝重复对象键，`income` 必须是官方 schema 要求的精确十进制字符串；native cashflow 金额拒绝浮点，source ID 拒绝首尾空白；核心金额边界不再删除逗号，畸形数值失败关闭。 instrument_rule 的 maker/taker fee validator 现接受有符号有限值以保留合法 rebate，同时 margin rate 仍非负；数据事件/归档/历史工厂源码与隔离 wheel 选择器各 167 passed，12-wheel source match 通过。 共享 Event schema 还拒绝空白/首尾空白 instrument 标识；当前数据事件/归档/hash 选择器 source 与 wheel 各 77 passed，历史规则工厂各 2 passed。当前现金流/指标/清算/历史工厂联合回归 **200 passed**；12-wheel source match 与隔离 venv provenance 通过。真实账户账本核验仍待独立类型化来源；来源 SHA 不认证账户真实性或查询完整性 |
| 02 / M0.2 | A10 | PASS（prior hosted revision）；PARTIAL（current manifest v3 locally verified, hosted CI pending） | Hosted Actions run #11 `39839975757`, commit `adf44ad58a5d7fe70d880673f6ae81b0a44ab230`: Linux and Windows locked installs, source manifest, 12 distributions, wheel/source matching, installed provenance, MyPy, PR contracts/unit, full world-model suite, and Ruff all passed. Windows selector: **855 passed, 4 skipped, 608 deselected**; Linux: **848 passed, 11 skipped, 608 deselected**; world-model on each OS: **228 passed, 14 JAX-only skipped**; dedicated JAX CPU suite: **11 passed**, including prior-causality and checkpoint/resume smokes. Current source identity: **604 files**, `59a584b0864d448535e1905a39905e2099536050a9d7550fad94d272327deb73`. Hosted runs exposed and verified fixes for worker NaN field diagnostics, Linux wheel workspace discovery, and inventory LF/CRLF checksum stability. Evidence: `docs/audit_evidence/20261023/worker_cli_anchor_nonfinite_validation_20261023.json` and `docs/audit_evidence/20261023/research_registry_linux_wheel_root_20261023.json`.
| 03 / M0.3 | A04 | PARTIAL，WFO Train/Validation/Test/Holdout 各阶段现按授权 scope 解码；登记前历史访问覆盖仍不可证明 | 除既有 ledger-first Train/worker 入口外，WFO 不再在启动时解码至 Holdout；默认候选 Train、Validation、stress Validation、Test baseline、正式 Test、最终 Holdout 和 final retrain 均使用各自窗口 reader。正式 Test/Holdout 必须先消费 EvaluationAccess，再读取其 warmup→score-end 窗口。复合 bundle identity 绑定市场 marker snapshot 与 rules SHA。WFO+application CLI+historical factory 源码选择器 **163 passed、18 warnings**；Ruff/MyPy 通过。详见 `docs/audit_evidence/20261022/wfo_scoped_payload_boundaries_20261022.json`。项目账本/真实市场未读取，M0.3 仍需解决授权启用前历史覆盖不可证明 |
| 04 / M1.1 | A01 | PASS，合成 Train-inner perpetual parity；E2 neutral and legacy restore regressions | 同一 frozen L4 actor 的基准 SoftRisk 与 `(1,1)` 逐步经济 trace、fills、Funding 和指标相同；新版 E2 中性候选的两代 checkpoint/resume 对照验证 evidence、durable ledger（5→10 requests）、预算与最终选择；pre-neutral E2 schema 3 恢复不插入额外候选（6 checkpoint tests passed） |
| 05 / M1.2 | A02 | PARTIAL | v1 保持兼容；v2 候选级 lexicographic feasibility 排序令任何不可行 episode 的选择分数为 -2，并将可行分数严格约束在 (-1,1)，lane 拒绝越界可行响应。最新修复了有限负终值被旧输入校验挡在破产筛选之前的问题；constraint-priority v1/v2 与 installed worker 均覆盖 `final_equity < 0`。Compute-worker pool **30 passed**、native training/ES worker **29 passed**，独立 Validation 证据未满足，不能晋级新目标 |
| 06 / M1.3 | A11 | PARTIAL（v2 三阶段分析器工程通过；完整原生经济报告缺输入） | v2 分析器验证 E1 genome/spec/hash/lane evidence 与 bundle lineage，要求 E1/R1/E2 报告绑定同一预注册及 market/rules/window/seed/environment/execution/hard-risk/scaler 身份，复算共同 native timestamps、成本对账和三个阶段差值；本轮新增严格的硬风控计数/决策与订单数量边界、冻结起始余额与原生初始权益相符性、`max_abs_residual` 明细/摘要一致性、报告与 bundle/actor/sidecar 路径限制及解析后目录边界校验；再修复两版分析器 JSON 重复键静默覆盖、非法 UTF-8 异常外泄、NaN 对账残差绕过、bool 数字混入及报告计数越界，两个新版及兼容分析器定向回归 37 passed、1 skipped（仅系统禁止 symlink 创建）。既有 R1-neutral→E2 配对保留，仓库暂无兼容 E1 bundle/三阶段报告，不重训或重放历史窗口制造输入 |
| 07 / M2.1 | A03 | PASS（T0/T1 全决策覆盖诊断） | 冻结的 90 天 Train-inner T0/T1 固定策略配对；90 个连续 rank-exclusive episode lane 完整覆盖 T0 的 129,597 个决策步（此前实现漏 87 步，现已修正）；T2 未开发，不属于当前验收范围；收益与成本仍是 scenario-only，T1 独立起始余额求和不构成复利收益或纯强制平仓因果效应 |
| 08 / M2.2 | A12 | PASS（工程验收） | 可选逐决策 trace 贯通 Actor normalized raw action、SoftRisk/HardRisk target、计划、订单、fill、币种费用与权益变化；`CANCEL/WAIT` 行补充相关既有订单的最终状态、累计 fill 数量与事件 ID、费用和残量原因，submitted fill 与相关订单事实分开。原生订单身份拒绝空白及首尾空格、拒绝冲突重复事实并仅折叠完全相同快照；`FILLED`/`PARTIALLY_FILLED` 状态必须与累计成交量一致；当前 NautilusTrader 的 15 个 `OrderStatus` 变体均得到显式 trace 结果。无效拒单原因退回稳定状态标签，完整历史工厂回归 **79 passed**，Ruff/Windows MyPy 通过；最新默认 Windows selector 与当前 wheel 证据见执行计划 |
| 09 / M2.3 | A06 | PASS（Bar Base/Stress 与合成部分成交压力工程回归）/ CONDITIONAL（真实 Quote/Depth） | Base 保留足量合成深度；Stress 的 `simulated_depth_fraction` 绑定配置身份并由原生 Nautilus 引擎实际部分撮合；同窗历史深度仍缺，因此不认证真实流动性 |
| 10 / M2.4 | A07 | PASS（60/360 Train-inner 性能与经济/Replay 等价；合成 F7 1/30/90d retention/trajectory engineering） | 工厂范围缓存可见批次与不可变 BarrierFrame；报告冷/热 Reset、吞吐、峰值 RSS 和缓存计数。F7 在 RouteB Native 环境对照 retaining 与 lean 报告快照，1/30/90d 的 1,437/43,197/129,597 个决策轨迹逐步等价；RSS 证据和 4 项定向回归见 2026-10-22 开发计划。本次为合成工程证据，不认证真实历史市场 Native RSS/经济账本 |
| 11 / M3.1 | A05 | PARTIAL | WFO 最终 DSR 报告现同时包含本 protocol 的候选证据和共享登记库跨 protocol 的 standalone attempt/recipe/feature/seed/lifecycle 汇总；注册 recipe/detail 将非有限或非 JSON 输入规范为 `ValueError`，seed 限定为 `[0, 2**32)`，防止超范围种子进入 SQLite。新增 Python 与 SQLite 双层生命周期迁移校验，允许 WFO 选择后的阶段恢复和 standalone failed/interrupted 恢复；直连数据库的迁移触发器回归也通过；研究访问/WFO/CLI 组合 **144 passed**，Ruff、Windows MyPy 通过。未登记的历史人工选择和失败尝试仍不可恢复，历史有效试验数保持 `not_estimated` |
| 12 / M3.2 | A13 | PARTIAL（代码诊断通过，研究覆盖待授权） | 训练 diagnostics 分列 interaction samples、唯一决策 timestamp/revisit fraction、Episode overlap pair 和固定 volatility/trend/Funding regime bins；本轮 overlap pair 统计改为 O(n log n) sweep/heap，57 项 RL/native Episode 回归通过。Stage60 v4 使用复用 Train 日期；历史 `recorded` 标签不构成新读授权，独立市场覆盖仍缺合法的新授权协议 |
| 13 / M3.3 | A14 | PASS（Train-inner 诊断）/ 未晋级 | 36 份冻结 Base/Stress 报告完成 51,732 步逐决策归因、固定状态分组和区块 bootstrap；权益对账通过，零 Validation/Test/Holdout 查询；冻结 RL 全部落入 deadband、无实际敞口/成交，未通过跨窗口基线优势门槛 |
| 14 / M3.4 | A11/A14 | CONDITIONAL | 上述研究门禁及合格独立窗口；先小预算筛查，再决定路线扩张 |
| 15 / M4.1 | A09 | PARTIAL（现行公开规则快照及三标的工程验收通过；历史规则/真实账户认证待续） | 已保存 Binance USDⓈ-M exchangeInfo 的 BTC/ETH/SOL 原始 symbol 记录，并以三标的原生引擎验收精度、数量步长、最低名义额、合成 Funding 结算和 USDT 权益单位（36 项 adapter smoke 通过）。快照仅对应 2026-10-22T06:35:32.110Z；历史生效规则、账户费率与真实账户流水仍未认证 |
| 16 / M4.2 | A09 | DEFERRED | M4.1 及独立保证金/双向持仓合同与金标准数据 |

依赖只约束依赖它的工作：远程 CI、历史规则或真实深度阻塞时，可以继续不依赖该证据的工程修复与明确标注的 Train-inner 诊断。M1 的 P0 工作不必等待真实深度或正式 Holdout。依赖未满足时不得开启正式晋级实验，也不得把可推进的工程项一起搁置。

## 里程碑 M0：经济事实与可复现基础（先决条件）

**M0.1 原生账户账本闭环（A08，P1）**。在现有同时间戳快照事件归并的基础上，定义决策快照、终止平仓快照及相同事件 ID 的次序契约；核对 fill、手续费、滑点、Funding、已实现/未实现盈亏与最终权益。加入同时间戳多成交、多 Funding、重复且相同 ID、相同 ID 冲突、终止平仓和清算的金标准样例。验收为每一经济事实恰好计入一次，冲突显式失败，原生总权益与逐事件现金流核对一致；长窗真实数据回放保留可复核差异表。现有归并实现是起点，不视为该闭环已完成。

首个工作包限定在 `packages/research/src/rles_research/native_metrics.py`、`metrics.py` 与必要的原生事实接口，优先复用 `tests/test_research_metrics.py`、`test_native_liquidation_facts.py`、`test_historical_factory.py`。先保留并核对已有 `test_native_report_merges_same_timestamp_facts_and_uses_terminal_equity`，再补真正未覆盖的情况。核对式必须说明期初权益、外部现金流、价格盈亏、手续费和带符号 Funding；滑点已经通过实际成交价进入盈亏时，只作成本归因，禁止从权益再次扣除。初始工程验收不依赖新增正式评价样本；长窗核对仅用已允许的 Train 数据，记录精度容差、规则假设、逐项差额及未解释残差。

**M0.2 CI 与样例交付（A10，P1）**。把默认基础 CI 所需的小型公开数据、来源清单和哈希纳入仓库；真实大文件测试单列数据依赖 Job，缺失时输出明确 `BLOCKED` 与所需文件，不把它统计为策略失败或成功。保留已修复的 Windows API 跨平台类型检查，并在 Linux、Windows 干净 checkout 各跑基础检查。验收为两平台基础检查独立通过；有数据的 Nightly 通过，缺数据的 Nightly 明确列出阻塞证据；Quote 样例不依赖本机被忽略目录。

**M0.3 项目级研究权限（A04，P1）**。复核现有 `ProjectResearchRegistry` 的跨输出目录作用域、半开区间重叠、数据版本变更、进程崩溃和重试幂等性。把所有正式 Test/Holdout 的读取统一接入同一个项目级登记库；记录旧运行导入的来源和无法证明的历史访问。验收为换目录、换数据哈希或换协议 ID 不能把已消费 Holdout 重新标成未用；恢复只复用已完成报告，不重复读取样本；报告分别说明“本协议未重复”与“项目级尚未使用”。

**M0 门禁**：M0.1/M0.2 的经济与可复现回归通过，M0.3 的研究访问测试通过，才能启动需要正式统计解释的实验；开发诊断可继续限定在 Train-inner。没有远程运行证据时 M0.2 的跨平台交付状态保持未验收，不用本地模拟平台类型检查代替实际 CI。

## 里程碑 M1：ES 搜索与目标一致性（P0 优先）

**M1.1 可辨识 E2 参数及精确中性候选（A01，P0）**。复核新运行默认两维空间和旧三维 Checkpoint 兼容路径；禁止新协议再同时搜索两个只靠乘积起作用的常数。把物理参数 `(1,1)` 的 R1 原样候选明确纳入候选矩阵、账本和评估预算，不能依赖有限潜变量的 `tanh` 恰好到达上界。若以后增加动态波动率控制，必须定义因果窗口、观测可用时间、TTL、上下限与缺失行为，并证明与仓位缩放参数的响应可区分。验收为中性候选的逐步订单、fill、Funding、权益完全复现 R1；各有效维度单独变化产生可解释差异；旧 Checkpoint 在冻结身份下可恢复。

**M1.2 约束优先的 ES 适应度（A02，P0）**。保留旧 `FitnessSpec` 作为版本化消融，不静默改变旧 Checkpoint。明确强平、结算失败、权益/保证金/硬风险违规等不可行条件；先拒绝不可行候选，再按预先冻结的费用后经济指标对可行候选排序。经济目标与风险权重写入配置、运行身份、报告。比较旧公式、去 HardRisk 调整惩罚、约束优先三组，预算、场景和随机种子相同。验收为盈利、空仓、亏损、违规四类确定性候选的选择及拒绝原因可解释；新目标只有在独立 Validation 的原生经济指标不恶化时才可替代旧目标，空仓始终保留基准资格。

**M1.3 阶段边际贡献（A11，P1）**。冻结相同 Actor/SoftRisk、风险预算、回放区间与成本场景，分别记录 E1 前后、R1 前后、E2 前后的原生回放。对 A0/A1/A3/A4 路线报告净收益、最大回撤、实际换手、手续费、滑点、Funding、成交量与风险干预差。验收为每个阶段有可复算的增量表；若 E2 没有稳定的独立经济改进，则保留精确中性 R1。

**M1 门禁**：M1.1 的中性完全等价、M1.2 的目标排序和配置身份、M1.3 的阶段原生增量均可复核后，才考虑扩大 ES 搜索预算。

## 里程碑 M2：训练、执行与流动性语义

**M2.1 Episode 终止时域（A03，P1）**。以相同固定策略、相同行情与执行配置配对：T0 连续账户，T1 按短 Episode 强制平仓，T2（若开发）仅切换采样边界而不断开账户。已完成五步原生合成回归和 90 天 Train-inner T0/T1 对照，报告额外平仓次数/费用、Funding、持仓时长、独立 Episode 权益和窗口覆盖差异。M2.1 当前通过范围为 T0/T1；未开发 T2，因此不作 T2 结论。报告边界明确 T1 独立起始余额不复利，情景 PnL 差同时受到强制结算、重开仓、余额重置和未覆盖决策点影响；训练与评估报告不得混用两种收益语义。

**M2.2 执行动作漏斗（A12，P1）**。在现有风险目标、数量取整和原生敞口归因上，逐决策记录 Actor 请求、Soft/Hard 调整、Planner 目标数量、NOOP/WAIT/CANCEL/SUBMIT 原因、原生订单 ID、完全/部分/未成交、最终敞口和费用后权益变化。验收为任一回放决策可沿 ID 找到结果或明确的未成交原因；计划层提交数与真实 fill 数分列；缺失覆盖时标 `unknown`。

**M2.3 流动性证据与成本压力（A06，P1）**。明确 Bar 基准是固定不利滑点加足量深度假设；可冻结 `simulated_depth_fraction` 对合成 Bar Stress 盘口限深，Base 固定为旧足量深度，深度比例参与配置/场景身份；Quote 模式拒绝合成限深并只用原生盘口。原生撮合回归已证明请求 0.500 时 0.200 成交（fill ratio 0.4），剩余订单仍可见。该结果只验证压力模型行为，不是现实流动性证据。剩余研究需对相同策略做 Base、Stress 与同窗、有来源证明的 Quote/Depth 对照；仅有 top-of-book 时限制可证明成交量，不捏造深层盘口。验收为 Fill Ratio、成交量、实际滑点、费用和净收益逐场景可比；没有真实深度证据时标“流动性未经认证”。

**M2.4 Episode Reset 性能（A07，P1，已完成）**。先测当前 `available_event_window` 切片与 Reset 重建的实际成本；只有出现显著热点时才设计工厂级只读事件索引缓存，并保持每个 Episode 新建引擎/账户。验收为固定数据与种子下观察、订单、成交、Funding、权益和 Checkpoint Replay 完全一致；在 60/360 天场景报告 Reset 延迟、总吞吐、RSS 峰值及缓存构造成本。已缓存按可用时间排序的原生市场批次、逐决策增量组及不可变 `BarrierFrame`。首个 Reset 负责冷建缓存，后续随机 Episode 只重建首帧 warmup 批次并复用其余帧。360 天 warm Reset 从 20.01 秒降到 0.070 秒，60 天从 3.62 秒降到 0.017 秒；缓存完整 rollout 的速度提升分别为 12.14% 和 15.35%，最大 RSS 增量与基线同量级（360 天约 1.22 GB）。60/360 天的 native order/fill/Funding 计数、terminal-flat 和 net-return 与未缓存基线完全相同；定向 uncached frame 与缓存 Replay 测试比较观察、边界 token 和原生费用/权益报告。收益只适用于重复使用同一冻结工厂的多 Episode 情形；一次性冷启动延迟近似原基线。

**M2 门禁**：能够区分人为终止、动作未执行、流动性/费用损失与价格信号错误，才进入更大训练实验。

## 里程碑 M3：Alpha 与统计可靠性

**M3.1 有效试验数（A05，P1）**。在项目级登记候选、特征版本、参数组合、种子、人工挑选和失败尝试；制定预先冻结的有效试验数估计政策。现有 DSR 只标明“本协议候选”，不得作为全历史校正。验收为 DSR 显示试验数来源、去重/相关性假设、历史覆盖界限与敏感性；无法估计时明确标 `not_estimated`，不输出全历史显著性结论。当前 WFO 报告已写入 `trial_count_evidence`，包括冻结候选 ID 与 SHA-256、账本观察覆盖、候选 ID 去重范围、跨折处理及独立试验 Sharpe 假设。共用项目 SQLite 现在追加记录候选配方及哈希、特征哈希、seed 和 started/selected/not_selected/failed/interrupted 生命周期事件；协议恢复按身份和事件键幂等。只读旧访问清单包含 10 个 WFO 协议库、21 条 Validation 与 1 条 Test 查询、18 个协议/候选组合、10 个候选 ID、22 条 standalone 报告索引和 67 条访问登记，范围限于两个已知根；离线盘点脚本已把 11 个匹配配置中的 10 种参数/seed/feature 配方指纹与 18 个协议/候选组合分开记录；配方指纹只是精确配置去重，不是有效独立试验数。人工筛选及未入账失败研究仍不可还原。当前协议报告已增加 effective trials=1 完全依赖端点与候选数独立端点的 DSR 场景敏感性，固定观测候选 Sharpe 离散度且不估计相关系数。历史登记前研究仍未覆盖，旧候选缺少跨候选同时间切片回报，不能进行历史相关性估计或全历史去重；历史有效试验数仍为 `not_estimated`，工作包保持 `PARTIAL`。

**M3.2 独立市场覆盖（A13，P1）**。把环境交互步数与独立市场时间点分开，报告 Episode 重叠率、唯一时间点、持仓状态、波动/趋势/Funding Regime 覆盖及各状态收益贡献。验收为重复采样只计计算预算，不计独立样本；固定新增 Train-inner 数据能否改善 Validation 的检验先于扩大训练步数。

**M3.3 费用后动作价值（A14，P1，Train-inner 诊断已完成）**。冻结的 36 份 Base/Stress 原生回放已配对比较冻结 Actor、空仓、持有及趋势/反转基线，完成固定波动/趋势/Funding 状态、执行可行性、费用后账户变化及 60 分钟区块 bootstrap 归因；51,732 个策略决策时点均完成原生权益核对，分析不重复扣除已进入权益的滑点。所有六个冻结 RL 回放的请求仓位都未越过执行 deadband，实际敞口和成交为零；结果只支持“执行可行性阻断”的机制结论，不能诊断方向信号是否有效。四个窗口为复用 Train 日期，且未通过跨窗口基线优势门槛，因此 Train-inner 诊断任务完成，但独立收益/晋级门槛未通过。最终分析与冻结修订分别见 `docs/audit_evidence/20261017/m33_action_value/analysis_final.json` 和 `preregistration_revision5.json`；不得重训、选模或把结果写成独立策略证据。

**M3.4 训练路线对照（A11/A14）**。仅在 M0–M3.3 门禁通过且有未消费的独立窗口时，按相同成本、风险预算、种子和 Validation 访问预算比较不交易/持有、简单趋势/反转、PPO、SAC、TD3、DDPG、E1、R1→E2、E1→R1→E2。主要指标预先冻结为费用后净收益及风险、费用、成交与置信区间。当前 gamma 三 seed/三窗对照未通过优势门禁，不追加同一机制训练预算。零成本都不能超过简单基线时，先回查特征、动作和环境语义。

**M3 门禁**：没有跨独立窗口、费用后且可执行的增量证据，不开启新的正式 Holdout、不宣称盈利、不扩大模型搜索。

## 里程碑 M4：能力扩展（最后实施）

**M4.1 标的元数据（A09，P2）**。将 ETH 数量单位等硬编码替换为从已验证 InstrumentRule/合约规格导出的元数据，先保持单资产 NETTING 行为；BTC/ETH/SOL 用同构小样本验证数量精度、最小名义额、Funding 与权益单位。验收为切换标的和规则不修改源码常量，原 ETH 金标准不变。

**M4.2 Hedge Mode 与杠杆认证（A09，P2）**。独立设计双向持仓账本、保证金与维持保证金、Mark 强平触发及 Funding/撮合事件的金标准回放。验收前保留 `>1` 倍杠杆限制和 NETTING 能力说明；不通过删除限制来宣称支持杠杆或独立多空。

## 跨轮次续做规则

1. 每次开始先读本文、当前目标入口和最近的执行记录，检查 `git status` 与相关源码差异。当前 HEAD 为审计基线，但大量源码修改尚未提交；保留这些修改，不 reset、不覆盖他人工作，不把 HEAD 当作当前实现的完整身份。
2. 从续做游标挑选首个依赖满足的任务；先核对已完成部分，再实施最小必要修改与有意义的回归。已通过的模块只做本次变更需要的检查，不重开全仓审计、重复造哈希体系或做无关格式化。
3. 工程修复、确定性原生样例和正式策略研究分开记账。涉及历史行情的实验先冻结代码差异、数据/规则身份、窗口、成本、种子、预算、主指标和停止线；只有项目登记允许的窗口可读。已消费数据继续保留其使用状态，换目录、协议或快照不得恢复新鲜性。
4. 必要检查通过后自动推进下一项；只有实际需要新增授权、缺少无法推断的信息或前置条件未满足时才停下相关分支。Test/Holdout 仍受既有一次性访问策略和冻结协议约束，不能因“连续开发”越过数据门禁。
5. 对外部证据不足、执行失败和候选被拒分别记录 `BLOCKED`、`FAILED`、`REJECTED`，写清依赖和可继续项。相同修复连续两轮无效时保存复现与原因，换可推进的工作包，不扩大同一失败候选预算。
6. 每完成工作包或结束一轮，在 [执行记录](DEVELOPMENT_PLAN_20261017.md) 追加结果；同步本文队列状态与 [当前入口](DEVELOPMENT_GOAL_CURRENT.md) 的游标。未完成就列出剩余动作和精确续做位置，不把计划中的命令写作已执行。

每轮记录最少包含：`任务 ID / 状态`、`修改文件与行为`、`命令与实际结果`、`证据路径`、`行情窗口及访问用途（无则写无）`、`阻塞与剩余边界`、`下一任务和起点`。任务可拆为“工程已完成 / 研究未通过 / 平台待验收”，避免一个总状态掩盖差异。大工件已有持久清单即可；新的关键小型摘要和运行器纳入 `docs/audit_evidence/`、`configs/`、`qa/`，不让唯一证据只留在 `.tmp/`。

## Codex 连续执行协议

收到继续开发指令后按以下顺序工作：

1. 阅读本文的续做游标、对应任务规格和 `DEVELOPMENT_PLAN_20261017.md` 最近记录；检查 `git status`、目标源码、相关测试和已有证据，辨别已完成修改与真正缺口。
2. 从续做游标开始，选择第一个依赖满足且仍有工程动作的任务。不得因某项策略研究、远程 CI、历史规则或真实深度阻塞而停下不依赖它的源码工作。
3. 直接实施最小必要改动，补有意义的回归；运行受影响测试及适用 Ruff/MyPy。涉及运行数据时先确认登记、冻结配置、数据/规则/模型哈希、窗口、用途和资源停止线。
4. 验收通过后，把任务状态推进并自动选择下一项；验收失败则保存真实错误和复现信息，标记 `FAILED` 或 `BLOCKED`，然后继续不依赖该项的工作。实验结果未过门仍可作为已完成的诊断结论，但不得写成收益改进。
5. 每轮在开发计划追加真实命令、实际结果、证据位置、数据访问用途、边界与下一游标；同步当前入口。不得只生成建议、重复已通过的全部检查、覆盖旧证据、虚报测试或将 Train-inner / scenario-only 结果说成正式验证。

此协议是 Codex 活动目标的执行说明。用户要求“按目标继续开发”时，应启动续做游标中的工程工作并持续推进；单纯要求编辑目标文档时，只修改目标本身。任何研究执行仍须满足已登记的数据用途与访问门禁。

## 检查命令与交付

以下是各工作包可用的检查命令参考，不代表已执行。先核实项目锁定依赖和当前源码环境；依赖未装好时单列环境问题，不以零收集测试或跳过代替通过。M0.1 优先跑前两个指标/清算测试文件，涉及历史工厂再扩展第三个；每项实际命令与结果追加至开发计划。

```powershell
uv run --locked --no-sync python -m pytest -q tests/test_research_metrics.py tests/test_native_liquidation_facts.py tests/test_historical_factory.py
uv run --locked --no-sync python -m pytest -q tests/test_es_dimension.py tests/test_es_fitness.py tests/test_es_checkpoint_resume.py
uv run --locked --no-sync python -m pytest -q tests/test_research_access.py tests/test_research_wfo.py
uv run --locked --no-sync python -m ruff check packages tests qa ci
uv run --locked --no-sync python -m mypy packages qa
git diff --check
```

按影响选择定向集成与全量测试，避免每个小修改都反复跑全部检查。平台交付对照 `.github/workflows/l6-ci-reusable.yml` 的实际 Linux/Windows Jobs；CI 尚未运行或失败就如实保留状态。收到继续开发指令后，按本文件协议实施对应源码任务并把实际结果追加到开发计划；不能以维护文档代替源码工作。正式 Test/Holdout、发布或其它未获授权的动作仍须服从各自门禁。

## 保留的交付与能力目标

原目标中的协调器/learner 资源边界、硬退出恢复、必要 GPU 路线、干净环境复现与发布候选验收继续保留，细节沿用 [资源恢复目标](DEVELOPMENT_RESOURCE_RECOVERY_20261013.md) 和 [2026-10-16 开发计划](DEVELOPMENT_PLAN_20261016.md)。R1-learning v5 现按训练块记录 CPU RSS 与 CUDA allocator 的 allocated/reserved 峰值，实际 CUDA smoke 已验证数值关系；这不构成整机 VRAM 使用测量或硬件级显存上限。N04 的 Windows Job 仍只硬限 CPU 提交内存，其他平台及 VRAM 强隔离继续保留未认证边界。

最终交付须能从干净源码与可分发小样本复现基础链路，给出当前实现、平台、研究与能力边界的独立状态。正式经济验收若因历史规则或独立样本阻塞，允许交付明确边界的工程成果，仍保持收益认证未完成。未经经济与语义门禁，不扩大模型/ES 搜索、不开放新 Holdout、不开展多币种或高杠杆主线。

**2026-10-22 RouteB risk-channel diagnostic continuation:** the E3 synthetic terminal gradient decomposition now identifies unstandardized log-realized-volatility as the dominant risk-head channel (median gradient norm `315.44`, versus at most `8.40` for the other individual components). This prompted an additive trainer diagnostic that logs each risk channel while preserving the combined risk objective formula. RouteB’s latest package state is source manifest 569 files (`23c4ca9f79a1eab579d9528305f2d084f681369ea013d55e4143979871fbc94b`), 12/12 wheels source-matched, 56 focused JAX tests passed and 204/14 skipped in the app selector. No market objective was retuned; synthetic evidence does not identify a market cause. Remaining gates retain their existing evidence requirements.


**2026-10-22 账本回执解析完整性补强：** Reconciler 所有 JSON 输入现拒绝重复对象键；目录扫描对含 R4 schema（含转义）的歧义 failure/registration 仍会登记为需拒绝的候选，避免消费过的失败 access 静默遗漏。聚焦测试 18 passed、Ruff/MyPy 通过；真实 RouteB ledger dry-run 42 accepted、0 rejected、0 unresolved，未写账或读取行情 payload。细节见 `docs/DEVELOPMENT_PLAN_20261017.md` 最新记录及 `docs/audit_evidence/20261022/reconcile_strict_json_20261022.json`。


**2026-10-22 A04 历史访问清单完整性：** 项目 ledger 启动导入现在拒绝重复 JSON 键、无效 UTF-8 与未知 inventory schema；修复前的合成测试确认重复 `unresolved` 可以被空值覆盖。安装后 RouteB wheel 的研究权限+reconciler 回归 92 passed；12 wheel/source 比对及 12 包、6 entry points 来源验证通过。该工程修复不补造无法追溯的登记前历史，也不认证旧清单来源真实性。


**2026-10-22 A04 最终本地验证：** 重新安装 RouteB 构建 wheel 后，PR marker selector **722 passed / 4 skipped**；world-model app suite **208 passed / 14 JAX-only skipped**；独立 CPU/JAX venv 的 world-model 非 CLI 部分 **207 passed**。四项 skip 的环境原因已分别覆盖：CUDA-only 仍需 GPU，JAX 项由 CPU/JAX venv 执行，CLI 在 app suite 执行。Hosted clean Linux/Windows CI 仍待交付运行。

### A04 query-intent validation boundary — 2026-10-22

The query-intent batch and verifier APIs now fail closed on scalar-string batches, boolean timestamps, invalid scope, and malformed required identifiers before lookup. Regressions were first observed failing against the prior implementation and now pass. Final installed-wheel selector: **165 passed**; 12 wheels match package sources; all 12 modules/6 entry points load from the isolated RouteB application venv; `uv pip check` confirms 69 compatible packages. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`. Next audit the remaining locally actionable A01–A14 inputs and authorization boundaries; externally hosted CI and external research evidence remain open.

### A04 registry input hardening follow-up — 2026-10-22

Fixed malformed status/digest handling that leaked `TypeError`, and protocol summaries that accepted a non-string ID as a successful empty result. Added regression tests, with the three new cases observed failing before the fix. The final installed-wheel selector passed **168 tests**; direct RouteB Ruff and MyPy checks passed; all 12 wheels match source and import from the isolated app venv, whose 69 dependencies are compatible. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`. Next continue with the first remaining locally actionable audit item; Hosted CI and research certification inputs remain external.

### A04 frozen candidate-set uniqueness — 2026-10-22

The project-level formal query intent API now rejects duplicate candidate IDs itself, keeping the frozen query identity unambiguous even if a caller bypasses the WFO runner. The pre-fix regression demonstrated acceptance and now passes. Final installed-wheel selector: **169 passed**; 12 wheels source-match; 12 modules/6 entry points load from RouteB's isolated app venv; 69 dependencies compatible. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`.

### A05 trial-recipe JSON identity — 2026-10-22

Canonical project-registry serialization now rejects non-string JSON object keys recursively before candidate recipe hashing and persistence, preventing Python's integer-key coercion from aliasing two distinct inputs into the same trial identity. Top-level and nested key regressions pass. Installed-wheel regression selector: **171 passed**; 12 wheels match source, the RouteB app dependency check and package provenance pass, Ruff/MyPy pass. Historical unregistered trials remain unrecoverable, so effective project-wide trial count stays `not_estimated`. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`.

### A04 legacy-inventory schema and import boundary — 2026-10-22

Strict parser and registry-level import now validate the exact historical access entry schema, timestamp types/range, allowed purpose, identity object, and lowercase source SHA-256 before import. This prevents malformed tracked inventory rows from escaping as raw `TypeError` or producing partial import state. Synthetic malformed-shape tests pass; the real tracked inventory parses read-only as 67 entries/0 unresolved. Final installed-wheel suite across access/WFO/CLI/fresh-fold/reconciler: **197 passed**. 12 wheels source-match; RouteB app dependencies/provenance, Ruff and MyPy pass. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`.

### A04 paired Test component validation — 2026-10-22

Malformed `component` JSON values in the paired Test identity path now return a normal reservation conflict instead of leaking `TypeError`. Regression was confirmed against the previous implementation. Final installed-wheel access/WFO/CLI/fresh-fold/reconciler suite: **198 passed**; 12 wheels source-match, import provenance and 69-package dependency checks pass, Ruff/MyPy pass. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`.

### A04 registry read/verify/consume entry points — 2026-10-22

`consume`, `verify_report`, and `development_result` now validate parameter types and SHA shape before SQLite. Previously malformed list arguments leaked `ProgrammingError`, while malformed report SHA values looked like ordinary receipt mismatches. Seven new negative cases cover these boundaries. Latest installed-wheel selector: **205 passed**; source/wheel match, package provenance, 69 dependencies, Ruff/MyPy pass. A proposed candidate-ID-wide identity rule was not retained because RouteB R4 legitimately scopes the same label across folds/seeds; attempt identity remains immutable. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`.

### A05 deterministic trial attempt ordering — 2026-10-22

The registry trial summary now uses an explicit `candidate_id, seed, attempt_key` order. A reverse-seed insertion regression showed the old output followed insertion order; the summary is now stable. Final installed-wheel access/WFO/CLI/fresh-fold/reconciler suite: **206 passed**; 12 source-matched wheels, app provenance/dependency checks, Ruff and MyPy pass. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`.


### A04 duplicate legacy inventory access keys — 2026-10-22

Repeated identical access keys in the strict legacy inventory were silently collapsed by idempotent registry insertion. The parser and direct registry import now both reject duplicate keys before import; malformed non-string keys fail with the registry's stable validation error. Regression first reproduced against the previous installed wheels. Final RouteB installed-wheel access/WFO/CLI/fresh-fold/reconciler selector: **209 passed**; Ruff/MyPy pass; 12 wheels/12 sdists build, all wheels match package sources, 12-package/6-entry-point provenance and 69-dependency checks pass. Source identity `b4b71e568a20e7d715e2c02693bb5755f656231d9d6d0efbe3e5b3b40d0268d4`. Evidence: `docs/audit_evidence/20261022/legacy_inventory_duplicate_access_keys_20261022.json`. No project ledger or market/evaluation data was accessed.

### A10 world-model artifact JSON ambiguity — 2026-10-22

World-model CLI requests, the repository base-commit marker, bundle manifests, JAX checkpoint sidecars, and research ArtifactStore JSON records reject duplicate keys recursively. Focused checks: **32 passed / 2 JAX-only skipped**; full app `tests/world_model`: **211 passed / 14 JAX-only skipped**; the exact dedicated CPU/JAX workflow selector **11 passed**; research artifact source and installed-wheel selectors **2 passed each**. Final wheel imports, Ruff/MyPy, and 571-file source manifest verification pass. Evidence: `docs/audit_evidence/20261022/world_model_duplicate_json_keys_20261022.json`. A10 remains `PARTIAL` until clean Hosted Linux/Windows CI is available.

### A10 orchestrator and WFO JSON inputs — 2026-10-22

Application config, resume state, WFO bundle/report/receipt/selection/baseline/Holdout readers now use the same recursive duplicate-key rejection policy. Source and installed-wheel application CLI/WFO selectors each pass **78 tests**; combined world-model + CLI + WFO source selector passes **289 tests / 14 app-environment JAX skips**; dedicated CPU/JAX selector passes 11. Ruff/MyPy pass; 571-file source manifest verifies. Evidence: `docs/audit_evidence/20261022/orchestrator_wfo_duplicate_json_keys_20261022.json`.

### A10 SB3 checkpoint and frozen actor JSON — 2026-10-22

SB3 checkpoint manifest/state JSON and L0 actor integrity/metadata now reject duplicate object keys recursively. Synthetic boundary regressions **2 passed**, restore parity **2 passed**, actor export/load parity **1 passed**; final RL wheel installed in RouteB app venv, 69 dependencies compatible, Ruff/MyPy pass. Source manifest: 573 files, `4c68bdf12bf3f90ca5ce88eb818a5e352da29681735c48b4b1af0aa1640c9f67`. Evidence: `docs/audit_evidence/20261022/sb3_checkpoint_actor_duplicate_json_20261022.json`.

### A10 curated data and Binance archive JSON — 2026-10-22

Curated markers/rows and Binance cached receipts/REST funding rows now reject duplicate JSON keys recursively. Source data selector **36 passed**; installed-wheel data + CLI + WFO selector **114 passed**; combined current-source world-model/CLI/WFO/SB3 JSON selector **291 passed, 14 app-JAX skipped**; exact CPU/JAX CI selector **11 passed**; Ruff/MyPy pass. 574-file manifest verifies at `aa24bc738e2a7db3a9cf1bd71e555910f404c6fca6e5492a608d3ec9f2cda8a5`. Evidence: `docs/audit_evidence/20261022/data_marker_archive_duplicate_json_20261022.json`.


### A04 canonical research-scope namespace — 2026-10-22

Registry overlap protection previously compared raw market/family strings, allowing an overlapping same-market development reservation under a padded market alias. Scope keys/owners/markets/families now reject surrounding whitespace; market identifiers must be uppercase and family identifiers lowercase. The strict inventory parser and direct registry import enforce identical normalization. Six alias regressions failed against the previous installed wheels. Final installed-wheel access/WFO/CLI/fresh-fold/reconciler suite: **215 passed**; Ruff/MyPy pass; 12 wheels/12 sdists build and all wheels match package source; provenance checks verify 12 packages/6 entry points and 69 dependencies. Source identity `1df7998039047073208d001fd97fa8e10f04ec9dfe1ac35e5677b546634a45b6`. Evidence: `docs/audit_evidence/20261022/registry_scope_namespace_canonicalization_20261022.json`. No project ledger or market/evaluation payload was accessed.


### A04/A05 formal query and trial identifier contracts — 2026-10-22

Formal candidate IDs, project trial IDs, lifecycle event keys, and protocol summary IDs now reject whitespace-only or surrounding-whitespace aliases before registration/lookup. Five regressions failed against the prior installed wheels. Final installed-wheel access/WFO/CLI/fresh-fold/reconciler selector: **220 passed**; Ruff/MyPy pass; 12 wheels/12 sdists match source, updated research wheel installed in RouteB's isolated app venv, provenance verifies 12 packages/6 entry points, and 69 dependencies are compatible. Source identity `978cbf8723d42aa28c0496f869d4e8119aa0237bdd51229d8ca4de81b6893d16`. Evidence: `docs/audit_evidence/20261022/research_identifier_contracts_20261022.json`. Synthetic identifiers and temporary SQLite only; no project ledger or market/evaluation data was accessed.


### A04 duplicate legacy inventory access keys — 2026-10-22

Repeated identical access keys in the strict legacy inventory were silently collapsed by idempotent registry insertion. The parser and direct registry import now both reject duplicate keys before import; malformed non-string keys fail with the registry's stable validation error. Regression first reproduced against the previous installed wheels. Final RouteB installed-wheel access/WFO/CLI/fresh-fold/reconciler selector: **209 passed**; Ruff/MyPy pass; 12 wheels/12 sdists build, all wheels match package sources, 12-package/6-entry-point provenance and 69-dependency checks pass. Source identity `b4b71e568a20e7d715e2c02693bb5755f656231d9d6d0efbe3e5b3b40d0268d4`. Evidence: `docs/audit_evidence/20261022/legacy_inventory_duplicate_access_keys_20261022.json`. No project ledger or market/evaluation data was accessed.


## 2026-10-22 — A09/M0.1 positionless fill multiplier guard

**Finding:** the account-row path could see fills without any position snapshots, then silently assume multiplier 1. For supported linear contracts with a multiplier above 1, turnover and slippage were understated. The main native report often caught this later through equity reconciliation, but direct account-row reports and compact reports without reconciliation could return wrong values.

**Change:** account rows, native reports, and equity reconciliation accept the frozen rule multiplier. Production application, the zero-target probe, and F7 profile pass the multiplier from the historical factory. If fills exist and no position facts establish a multiplier, omitting the frozen multiplier now raises MetricsError; when both sources exist they must agree. Added multiplier-2 positionless round-trip coverage and a conflicting-rule regression.

**Verification:** source and installed-wheel metrics/liquidation selectors each pass **96 tests**. The preceding cross-package source and installed-wheel selector each passed **246 tests** (18 upstream warnings). Ruff passed; MyPy passed packages qa (**224 source files**); RouteB .venv-win-app reports **69 compatible packages**. The Research and Orchestrator wheel modules match source byte-for-byte. Source manifest verifies **594 files**, identity 0f02c4421cb6da5db27a662bc6872c545b7f7a1ff36b5b53678c650858d6b670. Evidence: docs/audit_evidence/20261022/a09_m01_positionless_fill_multiplier_20261022.json.

Only synthetic data was used. No market, account, ledger, Validation, Test, or Holdout payload was accessed. Authenticated rules, complete cashflow provenance, and hosted CI remain open. Continue the local A01–A14 audit.

## 2026-10-22 — A09/M2 Bar Base/Stress multiplier-aware notional

The Train-inner comparison's `filled_quote_notional` omitted the frozen contract multiplier. Its report now applies and records the multiplier supplied by `HistoricalEnvironmentFactory`; the combined synthetic/frozen-plan regression passes (**9 passed**), as do Ruff, py_compile, targeted MyPy, and the verified source manifest. No study data was accessed. Evidence: `docs/audit_evidence/20261022/a09_bar_cost_multiplier_20261022.json`.

## 2026-10-22 — A08 aggTrades finite numeric boundary

USD-M aggTrades now reject NaN/Infinity in price and quantity before Decimal comparisons or aggregation. Four synthetic ZIP regressions passed after reproducing the prior parser behavior. Archive + data-event tests: **34 passed** from source and **34 passed** against the isolated installed wheel. Ruff, py_compile, targeted MyPy, 12-wheel source matching, installed-module SHA equality, and the 69-package dependency check pass. No market archive was accessed. Evidence: `docs/audit_evidence/20261022/a08_aggtrade_nonfinite_values_20261022.json`.

## 2026-10-22 — A08 archive CSV UTF-8 normalization

Malformed UTF-8 in aggTrades, bookTicker, and kline CSVs now fails with the data package's stable `DataError`; streaming behavior is preserved. Synthetic source and installed-wheel selectors each pass **37 tests**. Ruff, py_compile, MyPy, 12-wheel source matching, installed module hash equality, and 69-package dependency check pass. No exchange archives were accessed. Evidence: `docs/audit_evidence/20261022/a08_archive_csv_utf8_errors_20261022.json`.

## 2026-10-22 — A08 archive integer-field errors

Malformed integer IDs/timestamps in bookTicker, fundingRate, and kline rows now fail with stable `DataError`. Synthetic archive/data-event selector: **40 source passed** and **40 installed-wheel passed**. Ruff, py_compile, MyPy, 12-wheel source match, installed module SHA equality, and 69-package dependency check pass. Only temporary synthetic ZIPs were used. Evidence: `docs/audit_evidence/20261022/a08_archive_integer_fields_20261022.json`.

## 2026-10-22 — A08 aggTrades strict decimal lexemes

aggTrades price/quantity now accept only ASCII unsigned fixed-point notation; underscores and Unicode digits can no longer normalize into apparently valid trade values. The archive/data-event selector passes **44 source tests** and **44 installed-wheel tests**. Ruff, MyPy, 12-wheel source matching, installed parser hash equality, and the 69-package dependency check pass. No exchange archives were read. Evidence: `docs/audit_evidence/20261022/a08_aggtrade_ascii_decimal_20261022.json`.

## 2026-10-22 — A08 archive ASCII integer lexemes

Archive IDs/timestamps now require unsigned ASCII digits; underscores, Unicode digits, and explicit plus signs fail with `DataError`. The archive/data-event selector passes **49 source tests** and **49 installed-wheel tests**. Ruff, py_compile, MyPy, 12-wheel source matching, installed parser hash equality, and the 69-package dependency check pass. Synthetic ZIPs only. Evidence: `docs/audit_evidence/20261022/a08_archive_ascii_integer_lexemes_20261022.json`.

## 2026-10-22 — A08 archive ASCII decimal lexemes

Archive market numeric CSV fields now reject underscore and Unicode-digit spellings that Python float() accepts. Source and installed-wheel selectors pass 53 tests each; all 12 wheels match current source, and the RouteB venv dependency check passes. The change limits malformed external tokens before training/backtest ingestion; it does not certify the correctness or provenance of exchange archives. Evidence: docs/audit_evidence/20261022/a08_archive_ascii_decimal_lexemes_20261022.json.

## 2026-10-22 — A08 shared Event ASCII decimal lexemes

Extended strict decimal-token checks from archive readers to the shared Event schema, closing equivalent malformed-string acceptance through other data ingestion paths. Source and installed-wheel selectors each pass 75 tests; all 12 wheels match current package source and RouteB's isolated venv checks clean. Evidence: docs/audit_evidence/20261022/a08_shared_event_ascii_decimal_lexemes_20261022.json.

## 2026-10-22 — F7 compact report position-fact reuse

Compact report snapshots share immutable position facts across barriers when unchanged. One-, 30-, and 90-day synthetic full native runs preserve trajectory and metrics; the 90-day retained RSS peak delta improved by 74 MB (15.68%). Current evidence is saved in the F7 audit JSON and paired profile reports. Snapshot retention remains linear, so F7 is still partial.

## 2026-10-22 F7 online native return metrics

The native report path with account-row output disabled no longer stores the complete return array. It now accumulates compounding, drawdown, dispersion, downside statistics, and exact lower-tail VaR/CVaR online. The tail heap uses a snapshot-derived upper bound and is trimmed on a copy, preserving exact final quantiles and safe intermediate result calls.

Source and installed-wheel history/metric regression each pass **188 tests**; Ruff, py_compile, MyPy, all-12-wheel source matching, 12-module/6-entry-point import smoke, and the 69-package dependency check pass. The final synthetic 90-day native run covers 129,597 steps and has trajectory and native-metric hashes identical to the pre-change profile. Its RSS measurement is recorded as a single observation because repeated-run absolute peaks varied. Evidence: docs/audit_evidence/20261022/f7_online_native_metrics_20261022.json and f7_synthetic_native_growth_final_20261022.json.

F7 remains **partial**: compact per-barrier snapshots are still retained in an O(n) sequence, and include_account_rows=true intentionally materializes report rows. A full real-history Train RSS/trajectory profile remains outstanding.

### M0.1 online accumulator failure closure — 2026-10-22

A malformed or conflicting snapshot could raise after partially mutating `NativeReportAccumulator`, but the same object still accepted later snapshots or finalization. Two regression cases reproduced both unsafe continuations against the old implementation. The accumulator now stores the original failure and refuses subsequent `add_snapshot()` and `finish()` calls, preserving the failure cause. The four-file M0.1 source selector and installed RouteB wheel selector each pass **197 tests**; Ruff, targeted MyPy, py_compile and diff checks pass. The rebuilt `rles-research` wheel is installed only in `.venv-win-app`; its module SHA matches source, all **12 wheels** match package sources, and **69 dependencies** are compatible. Source identity: 600 files, `348f837f699634f23d3d48891c6225c60ba4b0bafe9e6e9ec95912f8241403e3`. Evidence: `docs/audit_evidence/20261022/m01_online_accumulator_fail_closed_20261022.json`. Synthetic facts only. Real authenticated cashflow provenance remains an M0.1 external gate.

### M0.2 reusable CI lane fail-closed validation — 2026-10-22

The reusable workflow previously accepted arbitrary lane strings while every real job was conditionally restricted to `pr`, `nightly`, `gpu`, or `release`; a caller typo could therefore make all jobs skip. The new unconditional `validate-lane` job rejects unknown values, and CPU, JAX, GPU, and release jobs all depend on it. The dispatch options and validator's allowed set are regression-checked together. The new contract suite passes **12 tests**; Ruff, targeted MyPy, YAML parsing and diff checks pass. The built `rles-l6-ci-release` wheel contains the exact validator source; the current 12 RLES workspace wheels match their package sources. Source manifest: **602 files**, `73470b5f5c08dc4a744c70849320a47c61d967462f4fac2d526b8e061513c934`. Evidence: `docs/audit_evidence/20261022/m02_ci_lane_fail_closed_20261022.json`. This is local engineering evidence only; clean hosted Linux/Windows CI remains unrun.


### M0.3 registry SQLite foreign-key enforcement — 2026-10-22

`ProjectResearchRegistry` declared foreign keys for development-result receipts and trial-attempt events, but its per-transaction SQLite connections left enforcement disabled. A new raw-transaction regression proved an orphan receipt could be inserted before the fix. `_tx()` now enables `PRAGMA foreign_keys=ON` before opening each write transaction. Access/WFO/fresh-fold source and installed-wheel selectors each pass **167 tests**; Ruff, targeted MyPy, py_compile and diff checks pass. The existing RouteB registry was checked read-only and has **0 current foreign-key violations**. The rebuilt research wheel matches source, all **12 workspace wheels** match, and **69 app dependencies** are compatible. Source identity: 602 files, `9e4730fd2a0130f2c041a7a28fe50723b27b766eff5a80414b23682dbe6b181e`. Evidence: `docs/audit_evidence/20261022/m03_sqlite_foreign_key_enforcement_20261022.json`. This improves future registry integrity but cannot reconstruct unrecorded pre-registry access history; M0.3 remains partial.


**A08 / 2026-10-22 archive importer instrument identity:** both Binance importer APIs reject a supplied empty or whitespace-padded instrument before output creation/fetch. The synthetic no-side-effect regression is included in the 97-test source and installed-wheel selector. All 12 rebuilt wheels match package source; installed provenance and 69-package compatibility pass. This closes only this ingress defect; the broader A08/M0.1 authenticated cashflow evidence remains partial.


**A08 / 2026-10-22 archive ingress boundaries:** both Binance importer APIs reject supplied empty or whitespace-padded instrument IDs before output creation/fetch, and non-ASCII remote `.CHECKSUM` bytes map to contextual `DataError`. Both fixes are covered by the 98-test source and installed-wheel selector. All 12 rebuilt wheels match package source; installed module hash and provenance pass, as does the 69-package dependency check. This closes two importer error-boundary defects; authenticated cashflow evidence remains partial.


**A08 / 2026-10-22 archive decode error boundaries:** malformed instrument IDs now fail before archive side effects; non-ASCII checksum content and invalid ZIP containers now become contextual `DataError`. The combined synthetic archive/data-event selector passes **100 tests against source and installed wheel**. All 12 rebuilt wheels match source; module hash, provenance, and 69-package dependency checks pass. These engineering fixes do not close authenticated account cashflow evidence.


**A08 / 2026-10-22 archive cache size boundary:** an existing archive without a source receipt could bypass the size cap and be read fully before verification. The importer now checks the shared stream-specific limit via file metadata before reading. A synthetic regression proves rejection before `read_bytes`; the combined source and installed-wheel selector passes **100 tests**. The same wheelhouse has all 12 package wheels matching source.


**M0.1 / 2026-10-22 canonical transfer identity inputs:** the paged and single-response importer now reject padded account scope and currency IDs before parsing, avoiding silent source-ID aliasing. Twelve regressions cover both APIs and surrounding/blank whitespace. The cashflow/metrics/liquidation source and installed-wheel selectors each pass **140 tests**; all 12 rebuilt wheels match source. This validates input semantics only; authenticated account export provenance remains outstanding.

**M0.1 / 2026-10-22 exact cashflow amount strings:** batch native reporting and the online accumulator now use one parser for exact `int`/`Decimal` values and unsigned ASCII fixed-point strings. Separators, Unicode digits, `+`, surrounding whitespace, and exponent strings fail before accounting. Source and installed research-wheel combined selectors each pass **245 tests, 18 dependency warnings**; Ruff, targeted MyPy, py_compile, 12-wheel source matching, 12-module/6-entry-point provenance, and the 69-package dependency check pass. Evidence: `docs/audit_evidence/20261022/m01_native_cashflow_amount_lexemes_20261022.json`. M0.1 remains `PARTIAL`: parser correctness cannot authenticate the source account or prove the completeness of real transfer history.


**2026-10-23 M0.1 续审：** 批量与在线路径现以精确 Decimal 累加同屏障现金流，低环境精度不再舍入；提款符号翻转不受环境精度影响；超出 4096 位对齐上限或 Decimal 算术范围时明确失败。源码与安装 wheel 联合选择器各 **249 passed / 18 warnings**，12-wheel manifest、安装 provenance 和 69 项依赖检查均通过。该工程缺口已修；M0.1 仍为 PARTIAL，等待已认证账户来源及可证明完整的真实转账历史。见 `docs/audit_evidence/20261023/m01_native_cashflow_sum_overflow_20261023.json`。


**2026-10-23 F7 rollout 接口续审：** PR selector 首次发现普通报告分支不应读取 `factory.rows`，该访问已限制在无账户行在线摘要路径；mock factory 同时补齐必要的 `contract_multiplier`。Windows PR selector 现为 **855 passed / 4 skipped / 597 deselected**，RouteB 已安装 wheel 的 CLI/Native accumulator 定向选择器 **57 passed**。全仓 Ruff 与 MyPy（225 个源码文件）、12 wheel 源码匹配、导入 provenance、依赖检查通过。见 `docs/audit_evidence/20261023/f7_rollout_factory_interface_20261023.json`。Hosted clean CI 仍为 M0.2 外部验证门。

**2026-10-23 M0.1 Decimal arithmetic follow-up:** the new precision-3 regression found native batch and online fill summary arithmetic could still round turnover/slippage. Both paths now use exact bounded Decimal operations. The 16-test native accumulator parity selector passes against source and a freshly rebuilt rles-research wheel; source and installed module SHA-256 match (966287f6cc96d7b3677894acbaf2cfe9d21083db5c316fc362aa18762aadd2a2). Source manifest: **604 files**, identity 6d723b79436a79535f10ef6e8efe326e7f300b662781d756cda1a4c35ddf5607. The broad Windows selector has not been rerun after this focused change. Evidence: docs/audit_evidence/20261023/m01_native_fill_decimal_context_20261023.json.


**2026-10-23 M0.1 reconciliation follow-up:** low caller Decimal precision also rounded native position quantity and linear-equity reconciliation. Batch and online reconciliation now use exact bounded operations throughout the ledger and valuation. The combined Native/accounting/history selector passes **251 tests in source and 251 against the rebuilt research wheel**, with 18 upstream warnings in each run; targeted Ruff/MyPy pass and installed/source module hashes match. Source manifest: **604 files**, identity `eae1a07e7bdb04d4c9ae7fed5c90d1a22cb925bb4e109b683d645e24b7992b24`. Evidence: docs/audit_evidence/20261023/m01_native_equity_reconciliation_decimal_context_20261023.json. Real authenticated transfer provenance and complete account history remain external M0.1 gates.


**2026-10-23 M0.1 exponent-boundary follow-up:** exact Decimal helpers now avoid underflowing representable values and clear inherited rounding flags; any unavoidable rounding fails explicitly. The five-file Native/accounting/history selector passes **252 tests against source and 252 against the rebuilt research wheel**, with 18 upstream warnings each. Ruff, target MyPy and diff checks pass; source/wheel module SHA-256 matches. Source manifest: **604 files**, identity `b9d91a34cbc084e995e42ddd6d5aa93135e93b0b73c757051787f93d05030292`. Evidence: docs/audit_evidence/20261023/m01_native_decimal_range_boundaries_20261023.json. Authenticated account source and transfer completeness remain external M0.1 gates.

### A10 current-revision update — 2026-10-23

Hosted run 39905239046 failed the Linux and Windows workspace MyPy step on stale reset-profiler reads of removed factory-cache attributes; its JAX CPU job passed. Those three metric reads are now removed from both profiler scripts, local MyPy passes 226 source files, and fresh Hosted CI for the RouteB-only follow-up is pending. Evidence: docs/audit_evidence/20261023/a10_stale_reset_profiler_mypy_fix_20261023.json.