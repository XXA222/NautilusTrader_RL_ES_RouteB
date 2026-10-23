Warning: truncated output (original token count: 130299)
Total output lines: 2101

# 2026-10-17 源码审计整改与开发门禁

**当前目标与续做入口：** [可连续开发目标](DEVELOPMENT_GOALS_REVIEW_20261017.md)。本文件保留既有整改和实验结果，后续工作追加记录；下文“下一步”是当时记录，当前顺序与任务状态以最新目标为准。

依据用户提供的 `NautilusTrader-RL-ES_源码深度审计与后续开发建议_20261017.md` 核对当前工作树。该报告审查的是 GitHub 提交 `c24d1bc`；本工作树还包含 2026-10-16 开发中的未提交修改。以下状态以当前源码和本地测试为准。报告中的收益、吞吐及真实流动性推测只作为待验证假设。

## 本轮已落实

| 审计项 | 当前修改和证据 | 边界 |
|---|---|---|
| A08 | 同时间戳归并、事件 ID 幂等/冲突拒绝、平仓核对、单一 quote-settled 单位乘数线性持仓估值及原生线性强平逐事件权益核对均已实现。真实 8 小时 ETHUSDT golden 与冻结 Train-inner 90 天 long exposure 的已标记持仓屏障残差为 0；Engine 级线性强平小样例也残差为 0。 | 不含外部现金流；强平压力规则/费用为合成测试参数，不认证历史成本；逆向/多资产账户不适用。 |
| A01 | 新应用 E2 默认只搜索仓位缩放和回撤节流两维，避免两个常数相乘造成的冗余；旧三维 RunSpec 指纹与 Checkpoint 仍可恢复。精确 `(1,1)` 候选已经纳入真实评估、预算、选择和恢复；冻结 L4 actor + 合成 perpetual Funding trace parity 通过。 | 只认证实现等价，不认证市场表现。旧三维显式 schema 仅为恢复兼容。 |
| A02 | 应用配置新增版本选择。`legacy_weighted_v1` 仍为默认；`constraint_priority_train_inner_v1` 按原生 insolvency、已认证维持保证金和强平事实筛选，按 `tanh` 有界费用后经济分数排名；拒绝原因进入代际报告、检查点和 bundle lineage。三目标同 actor/场景/预算的 A2 搜索及 Train-inner 原生回放已完成。 | 实验未观察到不可行候选，候选筛选经验效果未在搜索中触发；scenario-only 规则且无 Validation 证据，不能宣称目标改进或策略盈利。 |
| A03 | 合成五步回归与 90 天 Train-inner T0/T1 原生回放均完成；R3 修复后，T1 90 个每日独立账户 lane 完整覆盖 129,597 个决策点，终止强平及费用、Funding、持仓时长和权益差均有报告。 | 固定目标、scenario-only 规则；T1 PnL 为独立账户和，不构成复利收益或强制平仓纯因果归因；T2 尚未开发。旧 r2 的 87 步缺口已由 R3 修正，历史数值仅保留作审计轨迹。 |
| A10 | Windows Job/Recovery 与本轮 QA 脚本使用运行时 Windows API 解析；Linux 平台 MyPy 对 121 个源码文件及 Windows 两模块 `--platform win32` 均通过。将 2024-01-01 的 1 秒采样 bookTicker 测试 Parquet、完成标记和公开来源摘要放入 `tests/fixtures/`；Quote 用例纳入默认 PR `micro_golden`。最近 Windows PR 选择 **478 passed、60 deselected、38 warnings**。 | 远程 Linux/Windows CI 尚未运行；长窗真实 golden 仍留在 nightly；历史规则仍是 scenario-only。 |
| A12 | M2.2 可选逐决策 trace 已把 decision/plan/idempotency IDs 关联到订单、fills、费用、终态与未成交原因；执行结果报告默认只给摘要，CLI `--execution-trace` 开启明细。 | 逐订单 trace 仅 opt-in；成本与流动性仍受当前 Bar 模型假设限制。 |
| A06 | 90 天 Train-inner 固定敞口 Bar Base/Stress 对照已报告订单量、成交量、订单/数量 fill ratio、滑点、费用、Funding、净收益和回撤。 | Bar 模拟足量深度；Quote/Depth 深度与真实历史流动性仍未认证。 |
| A05 | WFO DSR 输出新增试验数范围：当前冻结协议候选；项目历史试验数标为未估计。 | 不能把该 DSR 当作全历史多重检验校正。 |

先前工作树已有项目级 Train/正式区间访问登记（A04）、Base/Stress 成本情景（A06）、动作到数量取整/原生敞口的漏斗诊断（A12）和独立市场覆盖报告（A13）。这些功能的存在不构成 Holdout 尚未污染、真实盘口深度或费用后 Alpha 认证。

## C 阶段预注册 gamma 配对实验的结果

`configs/discount_half_life_probe_20261017.json` 冻结了同一安全 Train 区间、3 个相同 seed、每臂每 seed 32,768 步的控制组 `gamma=0.99` 与六小时半衰期组 `gamma=0.9980764435756287`。6 个训练运行完成。首次诊断窗口预检为 DQ-D，未执行模型诊断；随后在查看任何模型表现前冻结 `configs/discount_half_life_evaluation_20261017.json`，改用三个 DQ-B Train-inner 开发窗口。36 份原生模型报告完成，Base/Stress 中每个 seed 的三窗平均配对收益差均为 0；两个组在多数窗口空仓。预注册门禁 `FAIL_NO_CROSS_WINDOW_MECHANISM_ADVANTAGE`，`candidate_promotion=false`、`budget_expansion_allowed=false`。这只否定当前固定设置下的预算扩张，不证明其它动作契约或数据没有 Alpha。原始结果位于 `.artifacts/audit-development-20261016/gamma-half-life-eval-v1/summary.json`，六个训练包位于 `gamma-half-life-v5/`；两份计划及模型清单都有 SHA sidecar。

## 目标建立时记录的计划顺序（历史；当前续做见活动目标）

1. **经济目标门禁（A02/A11）**：在相同 Train-inner 场景和预算下，比较旧 Fitness、去 HardRisk 惩罚、先筛硬约束再按经济指标排序。加入原生亏损/空仓/盈利/违规四类确定性测试，并记录 E1→R1→E2 每阶段的原生成交、费用、Funding、收益和回撤差。现阶段不增加 ES 搜索预算。
2. **中性 E2 门禁（A01）**：把精确 R1 中性候选纳入有账本记录的候选矩阵和评价预算；逐步比对相同 Actor 的订单、fill、权益和终态。只有改进可在固定验证情景复现时才采纳 E2 参数。
3. **Episode 时域门禁（A03）**：同一固定策略做完整连续账户与等长强制清仓窗口配对，报告人为终止成交、费用、Funding 持有时长和权益差。采样边界不产生现金流的连续账户方案另立实验。
4. **执行与数据门禁（A06/A07/A10/A12）**：继续扩展 Actor→Soft→Hard→Planner→NativeFill 的逐决策归因；Quote/Depth 情景仅在有历史来源时启用。先基准 60/360 天 Reset，再考虑事件索引缓存。把可分发小样本与外部真实大文件测试分开，Linux/Windows CI 各自验证。
5. **研究统计与能力边界（A04/A05/A09/A14）**：项目级登记历史候选/seed/特征版本并冻结试验数估计政策；新独立样本准备就绪前，不重用已消费 Test/Holdout。BTC/SOL、Hedge Mode、>1 倍杠杆保持未认证；先完成当前 ETH NETTING 原生经济事实及简单基线的配对反事实。

每项实验须先冻结源码、数据快照、规则来源、参数、Seed、成本假设、主指标、停机线和可访问的数据分区。已有历史规则缺口仍使收益只代表明示情景，不能报告为真实可交易收益。

本轮验证：ES/应用/历史工厂/研究账本/Compute 定向回归 **142 passed**；在加入 Episode 与 Quote 夹具后，指标/历史工厂/真实黄金样例再回归 **52 passed**。Ruff `packages tests qa` 通过，Linux 平台 MyPy `packages qa` **121 source files** 通过；Windows API 两模块 Windows 平台 MyPy 通过。未将这些定向结果称作全量 CI 或真实策略盈利认证。最终源码快照 `.artifacts/audit-development-20261016/source-snapshot-v14.zip` 包含此前尚未提交工作及本轮可交付 Quote 夹具。

## 2026-10-17 连续开发目标编辑

- 范围：根据用户指定的下载目录审计报告，编辑连续开发目标；只读核对当前 HEAD、工作区差异、相关源码、已有测试与 CI 定义。
- 文档：将原待审核稿改为带状态/依赖/验收/停止条件的任务队列；更新当前入口；保留已有局部修复、失败实验和历史证据，不重置状态为全部未开始。
- 覆盖：A01–A14 全部映射到 M0–M4；补跨轮次游标与记录规则，区分工程完成、研究拒绝和平台未验收；资源恢复/GPU/干净交付目标继续保留。
- 本轮未执行：源码修改、pytest/Ruff/MyPy、训练、回测、正式样本读取、远程 CI、提交和发布。历史 **142/52 passed** 仅保留为此前运行记录。
- 文档检查：三份目标/入口/记录文件的 UTF-8 编码和本地 Markdown 文档链接通过；A01–A14 覆盖及 16 个任务行检查通过；限定三份文件的 `git diff --check` 无报错。三份文档当前为已有未跟踪文件，编码/链接/覆盖检查直接读取工作区文件，不将 Git 差异检查当作未跟踪内容的完整校验。
- 当时的游标从 M2.2 执行漏斗 ID 关联继续；该任务现已完成。随后 M2.3 Base/Stress 诊断也已完成，当前游标以 [DEVELOPMENT_GOAL_CURRENT.md](DEVELOPMENT_GOAL_CURRENT.md) 为准，现为 M2.4 Episode Reset 性能。M0.1 外部现金流分类仍受 Nautilus `AccountState` 接口证据边界限制。

## 活动目标首轮实现：M0 账本、CI 基础与研究登记

用户随后明确要求创建 Codex 可自动实施的活动目标，已启用并授权连续开发，取代此前文档中的“等待启动”。本轮直接修改实现，没有新增正式评价或访问用户的 Test/Holdout；测试使用隔离研究登记。

### 账本修改与证据

- `native_metrics.py`：统一验证事件 ID、快照时间/币种、成交 BUY/SELL、正且有限的数量/价格及参考价；同 ID 的完全相同事实只计一次，矛盾记录明确失败，包含 Funding 跨时间戳冲突。
- 新增 `reconcile_linear_equity`，独立累积原生成交的带符号价格现金流、手续费与 Funding，在每个平仓屏障核对 `期初权益 + 成交价格盈亏 - 手续费 + Funding`。同时间戳决策与终止屏障分别核对；实际成交价已含滑点，不再次扣减。容差按显式每事件舍入上界乘以已观察事件数，不隐藏残差。
- 应用 `rollout` 接入该核对，完整报表保留逐屏障差额，摘要模式只保留统计。适用范围为无外部现金流、报价币结算、单位乘数的线性账户；**持仓屏障未做估值认证**，报告单列 `unchecked_open_barriers`；逆向合约、多资产和出入金不适用。
- 回归补了手算多/空两个 round trip、多 Funding、重复累计快照、同时间戳终止与篡改权益；净收益/手续费/Funding/滑点与独立金额核对。真实 ETHUSDT 场景黄金测试继续精确匹配既有原生终态权益 `9998.21338803`，并通过含 Funding 的终态独立核对。

### CI 基础缺口

- 新增 `.gitattributes`，对 `tests/fixtures/**` 禁用文本换行转换。当前仓库 `core.autocrlf=true`；本地 Git checkout-filter 对照证明：没有属性时规则来源 JSON 的 23 行变为 CRLF，SHA 为 `912b788c045c7a1ec0c28fff759537da9724330938c1f3a9be92dc9749a96f45`，与声明不符；有属性时恢复声明的 `b73028b1fca697c6dda6196a1b7c9e89d7aed0815ccb7d0facef900b3b67152f`。证据为 [m0_fixture_checkout.json](audit_evidence/20261017/m0_fixture_checkout.json)。这是本地过滤器复现，不是远程 CI 成功声明。
- `test_native_liquidation_facts.py` 移除 `runpy` 加载被忽略的本机上游 `examples/backtest/liquidation_demo.py`，改为使用已安装原生公开 API 的自包含报价/策略夹具。真实原生强平标记和 compact Position 字段等价测试通过；该 inverse fixture 只认证原生清算事实读取，不认证线性或高杠杆账户经济。
- 既有源码清单与打包器纳入 `.gitattributes`，打包器不再强行依赖上游示例。尚未提交/推送，未触发远程 Linux/Windows CI；Quote 夹具仍为已有未跟踪交付内容，后续源码交付必须纳入。

### 研究登记修复

- 原 SQL 对同 owner 豁免所有重叠，允许换 key 将已开发区间改称 Holdout，或重开原 Holdout。现在涉及 Holdout 的重叠即使 owner 相同也拒绝；完全相同 key/身份仍幂等，访问状态继续阻止重复 consume。
- 既有协议内 Test/development 的 owner 规则保留，以兼容顺序 WFO；不得将这一规则称为对任意调用者的安全隔离。旧历史记录允许记录已经存在的污染，但以 `recorded` 状态导入，不产生新评价授权。
- 历史清单的所有条目与导入完成标记在同一 SQLite 事务提交；中途非法条目会整体回滚。新增测试覆盖同 owner 的五种 Holdout 重标/重读组合、历史重叠的保守导入、导入幂等与部分失败回滚。

### 实际检查

运行前设置 `PYTHONPATH` 为所有工作区 `packages/*/src`，并确认 `rles_research.native_metrics.__file__` 指向当前源码，避免误测 `.venv` 内旧 wheel。测试解释器为 `.venv/Scripts/python.exe`（Python 3.14.7）。

```powershell
$env:PYTHONPATH = ((Get-ChildItem -LiteralPath packages -Directory | ForEach-Object { Join-Path $_.FullName 'src' }) -join [IO.Path]::PathSeparator)
.venv/Scripts/python.exe -m pytest -q tests/test_research_metrics.py tests/test_native_liquidation_facts.py tests/test_historical_factory.py tests/test_real_golden_ethusdt.py tests/test_application_cli.py -p no:cacheprovider --basetemp .tmp/pytest-goal-accounting-v1-20261017 --tb=short
.venv/Scripts/python.exe -m pytest -q tests/test_native_liquidation_facts.py tests/test_research_access.py tests/test_research_wfo.py -p no:cacheprovider --basetemp .tmp/pytest-goal-registry-v1-20261017 --tb=short
.venv/Scripts/python.exe -m ruff check packages tests qa ci
.venv/Scripts/python.exe -m mypy --platform linux packages qa
```

结果：账本/工厂/真实行情/应用定向 **94 passed、18 warnings、71.37 秒**；研究登记/WFO/自包含强平夹具 **63 passed、4.46 秒**；全项目 Ruff 通过，Linux 平台 MyPy **121 source files** 通过。初次只指定两个文件跑 MyPy 时导入解析指向未带类型标记的已装 wheel，出现 20 个 import-untyped；改为明确检查工作区 `packages qa` 后通过，未通过关闭诊断来规避。

完整 CPU 回归结果：**521 passed、4 GPU deselected、46 warnings、281.27 秒**，无失败。JUnit 位于 `.artifacts/audit-development-20261017/m0-cpu.xml`；已解析 XML 核实 521 个用例、0 failures/0 errors/0 skipped，小型证据为 [m0_verification.json](audit_evidence/20261017/m0_verification.json)。本轮新增的实现和测试保持冻结至全量检查结束；没有在运行中更改实现。该结果认证当前 Windows 源码环境的 CPU 回归，不等于远程 Linux/Windows CI、GPU 或收益认证。

```powershell
.venv/Scripts/python.exe -m pytest -q tests -m 'not gpu' -p no:cacheprovider --basetemp .tmp/pytest-goal-m0-cpu-20261017 --tb=short --junitxml .artifacts/audit-development-20261017/m0-cpu.xml
```

当时 M1.1 已通过实现等价门禁，M1.2 目标实现和匹配预算 Train-inner 搜索已完成，独立 Validation 仍未完成；该段记录的游标为 M0.1。此后本计划追加了 M2.1、M2.2 和 M2.3 的实际结果；当前续做游标以 [当前开发目标](DEVELOPMENT_GOAL_CURRENT.md) 为准：M2.4（A07）。A01–A14 尚未全部闭环。

## 2026-10-17 M1.1 精确中性 E2 候选进入评估协议

- 目标：把物理 `1.0` 全维 R1 中性参数作为独立候选，与每代 CMA-ES 候选在相同 Train-inner 场景种子下真实运行；让请求预算、L6 登记、选择结果及检查点恢复都看见该候选。
- 实现：`GenerationLaneEvaluator` 对新 E2 协议额外派发 `neutral-r1`，为每代持久化得分与证据，并将中性物理向量和协议版本纳入新检查点签名；禁用该功能时保留既有 schema 3 签名，恢复旧 Checkpoint 不漂移。`ESStageExecutor` 对优化器冠军与中性候选按既有最高 fitness 规则比较，同分优先返回精确全 `1.0` 中性参数。`EvolutionResult` 和阶段使用量计入每代额外候选；应用新 E2 RunSpec 的 evaluation/lane cap 与每代多出的请求一致，并在 `run.json` 记录协议版本。旧 manifest 恢复旧 E2 上限与协议，避免给旧检查点补记未执行的评估。
- 回归：A4 真实 ES/RL/ES + Native Engine + SQLite L6 流程现在实际评估中性候选，并核对 lane/evaluation 使用量；另有同种子 CRN、预算/检查点恢复、精确中性选择及旧 E2 应用恢复回归。相关文件集回归 **99 passed、29 warnings、198.40 秒**；签名兼容修正后的 ES checkpoint/真实应用恢复专项 **17 passed、17 warnings、151.62 秒**；参数选择、评估器恢复与 A4 路由回归 **8 passed、14 warnings、14.30 秒**。新增固定 actor 的同种子合成原生 spot 回放，对比基准 SoftRisk 与精确 `(1,1)`：请求/调整目标、计划动作、实际敞口、逐步权益/收益/回撤、归一化 fills 与终态权益完全相同（**1 passed，1.26 秒**）。该夹具无 Funding，使用 canonical fixture actor，不能代替导出 L4 R1 actor + perpetual Funding 场景。全新 A4 恢复用例与新增中性选择 helper 已分别在专项覆盖。Ruff 通过；显式设置 `PYTHONPATH`/`MYPYPATH` 指向工作区后，MyPy 对 ES 核心与 Orchestrator ES/application 共 **4 source files** 通过。M0 时全工作区 Linux MyPy **121 files** 通过证据仍有效。
- 结果边界：本轮只使用合成 Train-inner perpetual episode，未读取市场样本或访问 Validation/Test/Holdout。先在同一 episode 训练并导出 PPO frozen L4 actor，再以相同 seed 分别使用基准 SoftRisk 和中性乘子 `(1,1)` 原生回放；逐决策风险目标、计划、执行暴露、权益/经济收益/回撤、归一化成交、两笔 Funding、账户行和最终指标完全一致。预算、真实 A4 账本计数与恢复回归一并通过，M1.1 标记 `PASS`（仅限实现等价，不含市场表现）。
- 下一步：进入 M1.2 约束优先适应度。原生回放目前可观察且可靠的候选级硬约束是终态破产（最终权益归零）；HardRisk 调整次数是干预诊断，不能充当违规计数；结算/清算证据缺失继续作为无效运行拒绝。新版本需有有界且确定性的经济排序、可审计拒绝原因和冻结配置身份，同时不改变旧检查点身份。

## 2026-10-17 M1.2 约束优先适应度与匹配预算 Train-inner 对照

- 目标：旧加权 `FitnessSpec` 作为可复现基线保留；新协议先筛硬约束，再按配置的费用后净回报、最差尾损和最大回撤排名，并在 Run 的配置/ES evaluator 身份中明确版本。
- 实现：新增 `ConstraintPriorityFitnessSpec` / `FitnessDecision`，协议名为 `constraint_priority_train_inner_v1`。`NautilusBarrierEnv` 每步暴露 solvency 及认证的原生 margin/liquidation 事实；L6 candidate session 跨步累计 episode insolvency、maintenance margin breach、native liquidation，并把事实加入证据摘要。约束候选分数固定为 `-2`，可行候选分数为配置化 `tanh` 费用后经济聚合值（严格位于 `(-1,1)`），HardRisk intervention rate 不进入新目标。破产、维持保证金不足、原生强平分别保留结构化拒绝原因。新协议遇到 margin 事实未知时失败关闭；结算/terminal drain 无法证明仍作为无效运行中止。配置 `es_fitness.protocol` 显式选协议，legacy 仍为默认；legacy evaluator identity 和调用参数序列不变，新协议版本隔离 checkpoint。
- 回归：公式级利润/空仓/亏损/破产、分数有界、HardRisk 干预中立、margin breach、原生强平和未知 margin 事实拒绝均有确定性覆盖；候选原因聚合、E2 检查点恢复及 A4 新协议恢复后的 bundle lineage 均通过。最新 M1.2 定向回归 **17 passed、13 warnings、31.26 秒**；Ruff touched source/tests 通过，MyPy 4 个核心源码文件通过，`git diff --check` 通过。
- 三目标 Train-inner 对照：固定 checksum-verified ETHUSDT 2024-01-04 归档夹具，使用同一个 frozen R1 actor、`seed=31`、共同 `scenario_fold_id=a02-three-fitness-crn-20261017`、相同两代 CMA-ES 配置（population 4、8 optimizer evaluations、每臂 10 lane requests，含每代 R1 neutral 候选）。legacy、零 HardRisk 干预惩罚、constraint-priority 三臂均先以 `train(evaluate=False)` 搜索，再只对同臂选中参数执行相同 Train-inner native replay。legacy 净收益 **-0.1578%**，费用/资金费/滑点分别 **12.2536 / 0.0160 / 2.5314**，换手率 **2.4528**；零惩罚与约束优先选中同一参数，二者净收益均 **-0.1510%**，成本分别 **11.4319 / 0.0147 / 2.3616**，换手率 **2.2882**。三臂均为 **0 infeasible candidates**，所以本次搜索没有经验触发约束筛选；单测覆盖了 margin/强平分类分支。证据：[m1_2_three_objective_train_inner.json](audit_evidence/20261017/m1_2_three_objective_train_inner.json)，SHA-256 `D4036715857AAF9BD97070E48C28B8FFBE97E5A3DF22BC1A0F7AC821CF0E363F`；复现脚本 [compare_es_fitness_objectives_20261017.py](../qa/compare_es_fitness_objectives_20261017.py)，完整运行产物位于 `.artifacts/audit-development-20261017/m12-three-objective/`。
- 边界与状态：这是一个 scenario-only 规则日窗、Train-inner 诊断，不读取 Validation/Test/Holdout。三目标分值使用不同尺度，不能跨协议比较 raw fitness；原生回放均亏损，不存在收益晋级结论。匹配预算搜索和候选级 margin/liquidation 事实分类已完成，M1.2 保持 `PARTIAL`，因为尚无冻结且可用的独立 Validation 证据。工程证据见 [m1_2_constraint_priority.json](audit_evidence/20261017/m1_2_constraint_priority.json)。
- 下一步：保留 M1.2 当前游标，等待/核实冻结协议允许的独立 Validation 和有日期证据的经济规则；此前可继续不依赖新样本的工程整改，但不得把这次 scenario-only 搜索解释为目标收益改善，不访问未获准的 Test/Holdout。

本次确切复核命令（PowerShell，仓库根目录）：

```powershell
$workspacePythonPath = ((Get-ChildItem packages -Directory | ForEach-Object { Join-Path $_.FullName 'src' }) -join ';')
$env:PYTHONPATH = "$workspacePythonPath;tests"
$env:MYPYPATH = $workspacePythonPath
.venv\Scripts\python.exe -m pytest -q --basetemp=.artifacts/audit-development-20261017/pytest-m12-margin tests/test_es_fitness.py tests/test_l2_gym_bridge.py::test_native_margin_and_liquidation_facts_are_classified_only_when_certified tests/test_compute_worker_pool.py::test_worker_dispatcher_transmits_and_reconciles_account_context tests/test_compute_worker_pool.py::test_constraint_rejections_are_reported_and_checkpointed_with_e2_neutral_protocol tests/test_real_application_flow.py::test_real_archive_a4_recovers_from_durable_stage_checkpoint[E2] tests/test_native_training_episode.py::test_a4_provider_backed_workflow_runs_real_es_rl_es_and_l6_ledger
.venv\Scripts\ruff.exe check packages\es_evotorch\src\rles_es_evotorch\fitness.py packages\es_evotorch\src\rles_es_evotorch\evaluation.py packages\orchestrator\src\rles_orchestrator\native_es_worker.py packages\gym_bridge\src\rles_gym_bridge\env.py tests\test_es_fitness.py tests\test_l2_gym_bridge.py tests\test_compute_worker_pool.py tests\test_native_training_episode.py
.venv\Scripts\mypy.exe packages\es_evotorch\src\rles_es_evotorch\fitness.py packages\es_evotorch\src\rles_es_evotorch\evaluation.py packages\orchestrator\src\rles_orchestrator\native_es_worker.py packages\gym_bridge\src\rles_gym_bridge\env.py
git diff --check
```

## 2026-10-17 M2.2 执行 trace 完成与 M2.1 T0/T1 Train-inner 对照

- M2.2 实现：`rollout()` 增加默认关闭的 `include_execution_trace`。打开时逐决策保留 `decision_id`、时间戳、计划/幂等 ID、Actor/Soft/Hard 目标、取整目标、数量变化、计划原因和原生敞口；用客户端订单 ID 关联原生订单终态与 fill 数量/价格/费用，并为未成交报告 `REJECTED`、`CANCELED`、`EXPIRED` 或 `unknown`。非提交动作记录提交状态及相关挂单结果。默认 report 仅输出漏斗汇总；公开 `backtest` 和 `diagnose-train` CLI 使用 `--execution-trace` 才输出明细。紧凑历史快照改为保留每个已观察原生订单的最后状态，避免订单完成后 Nautilus 缩减缓存导致 trace 无状态；同时将 `position_id` 纳入紧凑 Position 字段。
- M2.2 验证：历史工厂、原生训练、应用 CLI 相关集 **94 passed**；随后修复 `_snapshot` 类型检查后定向回归 **5 passed**；最新完整默认 Windows PR 选择 **478 passed、60 deselected、38 warnings、115.26 秒**。Ruff、MyPy 对变更源码通过。trace 集成回归验证真实合成原生 Engine 的 decision→order→fills 连接、成交费用及订单终态；CLI 测试确认 flag 正确透传。证据在 `.artifacts/audit-development-20261017/m2-2-pr-windows-final.xml` 与测试/源码。
- M2.1 输入冻结：复用 `.artifacts/audit-development-20261016/f7-growth-v2/plan.json` 引用的 Train-inner 市场与规则，bundle SHA-256 `ea2ea516780f25756c7dae35636ee59eca7e3443a24259c6b58526b2536fec68`，市场 SHA-256 `06323e909997909dee35b4abf1543444ae230f30cbc1f672ded033b3bc41fe70`，规则 SHA-256 `690292b508ce722d6ba2f5b9fb745671f56fc396d2026943fd36668e9028c93b`。窗口为 2024-01-15 00:00 至 2024-04-14 00:00 UTC，seed 102，执行配置及软风险参数来自同一冻结 bundle；策略为固定 target exposure 0.25。Base Bar 成本为情景假设，非历史规则认证。Train-inner 开发用途登记，未访问 Validation/Test/Holdout。启动时自由物理内存超过 10 GiB；每个 T1 Episode 前检查 2 GiB 停止线。
- M2.1 结果：T0 为一个连续账户，129,597 决策，权益从 10,000 到 10,052.48862672 USDT，PnL `+52.48862672`、费用 `0.28397867`、Funding `-14.65498461`、3 fills、1 个 Position 身份、持仓 `7,775,820` 秒，终态平仓。T1 为 90 个互斥每日时间 lane，每个 1,439 步；所有账户均以 10,000 USDT 独立起始、每日强制平仓，90/90 平仓。其独立账户 PnL 和 `+21.44852187`（不复利）、费用 `22.40042008`、Funding `-13.91367805`、180 fills、90 个 Position 身份、持仓 `7,770,600` 秒；90 次终止平仓 fill 的专属费用合计 `11.21465070`。两臂 PnL 差 `-31.04010485`、费用差 `+22.11644141`、Funding 差 `+0.74130656`、fill 差 `+177`。90 个 episode 窗口无重叠；相对于 T0 共有 87 个市场决策点未进入 T1 的 90 个 1,439 步 Episode。故 PnL 差混合了强制结算/再开仓成本、余额重置、lane 的未覆盖点和仓位复位影响，不能当作强制平仓的纯因果成本或策略收益结论。每个 episode 的 summary、汇总和窗口无重叠验证通过。
- 资源和证据：实验最低可用物理内存 `12,778,246,144` 字节，大于冻结 `2,147,483,648` 字节停止线。摘要 [m2_1_episode_horizon_train_inner.json](audit_evidence/20261017/m2_1_episode_horizon_train_inner.json)，SHA-256 `136922fcaa90aaecdafb0b18cf6cb2bd7f9d992048c4dac8c00e1b904c05a16b`；运行清单 [m2_1_episode_horizon_execution_manifest.json](audit_evidence/20261017/m2_1_episode_horizon_execution_manifest.json)，SHA-256 `ba921e097535c1d3362c6c0aa525ab90ca95e9dd59e483852127316f02723a8c`。清单绑定 10 个执行相关源码/测试文件哈希、输入计划/bundle/行情/规则哈希及两条项目级 `development` 访问登记；完整产物 `.artifacts/audit-development-20261017/m2-1-t0-t1-daily-r2/`，首次 seed 缺失及 coverage schedule 越界错误分别保留于 `.tmp/m2-1-seed-missing.log`、`.tmp/m2-1-t0-t1-run.log`。首次失败后修复 QA runner：从冻结源计划继承缺失的 seed；按回放窗口调用现有 `coverage_for_window()`，避免原 bundle 全范围 Funding schedule 被误判在窗口外。无业务源码调整。
- M2.1 定向验收：`test_short_episode_boundary_has_extra_native_flatten_cost_vs_continuous_account` 与 `test_rollout_execution_trace_joins_decisions_to_native_orders_and_fills` **2 passed、2.69 秒**；QA runner Ruff、MyPy 通过；90 天 T0/T1 实际回放及 summary/哈希/资源检查通过。M2.1 标记 `PASS`（仅完成 T0/T1 情景机制诊断，未开发 T2）。
- 下一游标：M2.3（A06）。盘点已有 Bar Base/Stress 报告是否提供 Fill Ratio、真实成交量、实际滑点、费用和费用后收益；复用已允许 Train-inner 固定策略比较 Base/Stress，补齐缺失的可解释字段。没有完整历史深度证据时 Quote/Depth 及真实流动性认证仍保持 `CONDITIONAL`。

复现命令（仓库根目录 PowerShell）：

```powershell
$workspacePythonPath = ((Get-ChildItem packages -Directory | ForEach-Object { Join-Path $_.FullName 'src' }) -join ';')
$env:PYTHONPATH = "$workspacePythonPath;qa"
.venv\Scripts\python.exe qa\compare_episode_horizon_20261017.py --plan .artifacts\audit-development-20261016\f7-growth-v2\plan.json --output .artifacts\audit-development-20261017\m2-1-t0-t1-daily-r2
```

## 2026-10-17 M0.1 现金流接口边界、M0.2 默认 CI Quote 与 M2.2 决策 ID

- M0.1 接口核查：锁定环境 NautilusTrader `2.0.0rc6` 的 `AccountState` 公共型定义提供 `balances`、`margins`、`reported`、`event_id`、`info`、`ts_event` 等字段，但无标准化的 deposit/withdrawal 类型或原因字段。对 Engine 原生 ETHUSDT 8 小时回放的只读诊断观察到初始 `reported=True` 账户状态，和 08:00 Funding settlement 对齐的新 reported 状态；两者 `info={}`。因此 `reported=True` 不能当作入金/出金分类依据。尝试性判定会错误拒绝正常 Funding，已撤回，不把错误启发式留在代码中。当前权益残差可拒绝净未解释现金变动，但无法证明彼此抵消的转账与漏记…115299 tokens truncated…ile pass. All **12 wheels + 12 sdists** build; wheel manifest **all_match=true**; import provenance covers 12 packages / 6 entry points; 69 app dependencies are compatible. Installed `native_metrics.py` SHA matches source (`398ba6e5e413ad11f5bb800f1a087968d4e80b7556e787c22d2de71adda4104f`). Source manifest verifies **604 files**, identity `9469a741ea8de5eaabb202bcc2428d0a34016b618bc86da256316d4aff153326`. Evidence: `docs/audit_evidence/20261022/m01_native_cashflow_amount_lexemes_20261022.json`. Only synthetic cashflow facts were used; authenticated account origin and transfer completeness remain external M0.1 gates. Continue the A01–A14 local source audit.


## 2026-10-23 — M0.1 exact sourced cashflow aggregation

**Finding:** same-barrier transfers could silently lose low-order value when the ambient Decimal precision was small; withdrawal unary negation could round a value before accumulation; and the earlier overflow guard did not catch finite inexact results. Excessive exponent alignment also needed a bounded resource contract.

**Fix:** batch reporting and `NativeReportAccumulator` share an exact Decimal adder. It derives precision from both finite operands in a local context, rejects sums requiring more than 4096 digits of aligned precision, uses `copy_negate()` for withdrawals, and maps Decimal arithmetic/range failures to stable `MetricsError`. Regressions exercise `1E+28 + 1` and `-12345` under precision 3, both report paths, the alignment bound, and arithmetic overflow.

**Verification:** final combined source selector and final installed RouteB research-wheel selector each pass **249 tests, 18 dependency warnings**. Ruff, targeted MyPy, py_compile and `git diff --check` pass. All **12 wheels + 12 sdists** build; the final wheel manifest matches all 12; installed provenance covers 12 packages / 6 entry points; `uv pip check` confirms **69 compatible packages**. Source/installed module SHA-256: `28af40fe9d2c4e18eaaf920897ed7522f5beca8be6011e4f067ab2b8bf292acc`. Source manifest: **604 files**, identity `b0a2241bb3d1197ff29c51a28822113895fa546acf01f4186d6a9f8173466c69`. Evidence: `docs/audit_evidence/20261023/m01_native_cashflow_sum_overflow_20261023.json`. Synthetic flows only; authenticated account origin and transfer completeness remain external M0.1 gates. Continue the local A01–A14 source audit.


## 2026-10-23 — F7 rollout factory interface guard

**Finding:** the PR selector exposed four failures in `rollout()`: it accessed `factory.rows` before deciding whether the online summary path was active. The ordinary account-row path and generic environment factories do not require that attribute. Once row access was made lazy, the same fixtures exposed that they omitted the explicit `contract_multiplier` required by current native accounting.

**Fix:** only derive the step bound from `factory.rows` for the summary-only, non-spooled path. The standard report path no longer reads it. Test factories now state their unit multiplier explicitly, matching the public accounting contract.

**Verification:** the first full selector reproduced the four errors; after the fix the Windows PR marker selector passes **855, skips 4, deselects 597**, with 46 third-party warnings. The installed-wheel `test_application_cli.py` + native accumulator selector passes **57**. Full Ruff (`packages tests qa ci`), full MyPy (`packages qa`, 225 source files), targeted py_compile and diff check pass. All **12 wheels + 12 sdists** build; final wheel manifest all-match; all 12 RouteB wheels were installed in `.venv-win-app`, import provenance verifies 12 modules/6 entry points, and 69 dependencies are compatible. Source manifest: **604 files**, identity `44ad07a1e5fac4b6deac0eb82302f391c305ae354ca715a7651a03696d542ace`. Evidence: `docs/audit_evidence/20261023/f7_rollout_factory_interface_20261023.json`. Synthetic unit fixtures only. Hosted clean Linux/Windows CI and data-backed research gates remain open. Continue the remaining code-ready A01–A14 audit work.


### M0.1 native linear-equity Decimal precision  2026-10-23

Batch and online reconciliation now use bounded exact Decimal operations throughout fill quantity/cash ledgers, fees, funding, external flows, marked notional, equity, residual and tolerance calculations. A precision-3 open/close regression fails on the prior code and passes after the fix. Source and rebuilt installed-wheel selectors: 251 passed each, 18 dependency warnings; targeted Ruff/MyPy pass. Source/installed native_metrics.py SHA-256: 9b3291ecb387bd28d5580fa11a9e73574f63de61c1827c52954ee21331b496cc. Source identity: 604 files, eae1a07e7bdb04d4c9ae7fed5c90d1a22cb925bb4e109b683d645e24b7992b24. Evidence: docs/audit_evidence/20261023/m01_native_equity_reconciliation_decimal_context_20261023.json. Synthetic facts only; authenticated cashflow provenance and history completeness remain open external M0.1 gates.


### M0.1 Decimal exponent-boundary and reconciliation follow-up  2026-10-23

Expanded exact Decimal helpers below the default Emin while retaining the default Emax. They clear inherited status flags before operation and map unavoidable Rounded/Inexact signals to MetricsError. Low-precision tiny-add/tiny-multiply tests, the open/close equity reconciliation case, and existing cashflow/overflow cases pass. Source and rebuilt research-wheel five-file selectors each pass 252 tests with 18 dependency warnings. Ruff, targeted MyPy, diff check, source manifest, and installed/source native_metrics.py SHA parity pass. Manifest: 604 files, b9d91a34cbc084e995e42ddd6d5aa93135e93b0b73c757051787f93d05030292. Evidence: docs/audit_evidence/20261023/m01_native_decimal_range_boundaries_20261023.json. Authenticated account transfer evidence remains an external gate.

## 2026-10-23 A04 reconciler directory-scan fail-closed boundary

**Finding:** recursive report discovery had no explicit traversal-error handling, omitted symlink directories, and dropped unreadable failure/registration receipts. Recovery paths skipped malformed identity JSON, while the report reader accepted valid non-object identity JSON. Raw SQLite URI interpolation also mishandled valid registry paths containing URI-reserved characters such as `#`.

**Fix:** `_report_paths()` now uses deterministic `os.walk(..., followlinks=False, onerror=...)`. If any directory cannot be scanned or a symlink directory would be skipped, it raises a path-qualified `ValueError` before parsing reports or opening the ledger. Unreadable or invalid failure/registration receipts are surfaced for rejection. Both development-access readers fail closed on malformed/non-object identity instead of skipping or passing that row forward. All five read-only SQLite connections now percent-encode path characters in their URIs.

**Verification:** injected permission-error, omitted-symlink, unreadable-failure/registration-receipt, corrupted identity, non-object identity, and `#`-path SQLite regressions pass. The full reconciler suite passes **26 tests**; Ruff passes for the two changed Python files; project MyPy passes for **225 source files**. Source manifest verifies **604 files**, identity `c080242137afcfa8ad4bcab0356345432f0d54ee27985123fa52c920abe362cc`. Evidence: `docs/audit_evidence/20261023/reconcile_directory_scan_fail_closed_20261023.json`. Temporary synthetic paths only; no project ledger or market/evaluation payload was accessed.

**Next cursor:** continue the A01–A14 code-ready audit from current source. External authenticated cashflows, clean Hosted CI, independent Validation, new market-scope authorization, authentic depth, historical trial coverage, and M4.2 remain separate gates.

## 2026-10-23 — A04 legacy R4 reconciler malformed ledger entries

**Finding:** `_matching_failed_legacy_r4()` skipped malformed trial-attempt recipe JSON and malformed event-detail JSON. A corrupt sibling ledger row could therefore be omitted while the remaining attempt was still treated as the unique match for a legacy R4 failure receipt.

**Fix:** malformed JSON and valid non-object values in any candidate recipe or its event detail now raise contextual `ValueError`; reconciliation cannot silently establish uniqueness from a partial ledger view. Added regression coverage by corrupting a temporary sibling attempt recipe and then its failure-event detail, verifying each is rejected, and restoring the synthetic rows to confirm the legitimate read-only reconciliation still works.

**Verification:** `tests/world_model/test_reconcile_routeb_research_ledger.py` — **26 passed**; Ruff passed for the changed source and test; py_compile passed; project MyPy passed for **225 source files**. Source manifest verifies **604 files**, identity `48815532670160883cc0f2f1c37a82aeab8603e6bd9a9a52ea5101bf0c911350`. Evidence: `docs/audit_evidence/20261023/reconcile_directory_scan_fail_closed_20261023.json`. Only temporary synthetic SQLite records and report receipts were used; no project ledger, market, account, Validation, Test, or Holdout data was accessed.

**Next cursor:** continue the dependency-satisfied A01–A14 source audit. External authenticated cashflows, clean Hosted CI, independent Validation, new market-scope authorization, authentic depth, historical trial coverage, and M4.2 remain separate gates.

## 2026-10-23 — M0.2 worker CLI finite-field diagnostics from Hosted CI

**Finding:** Hosted workflow run `39834856764` failed both Windows and Linux CPU world-model jobs on `test_worker_rejects_invalid_anchor_or_instrument_before_forward[anchor_close-nan-anchor close]`. Strict JSON decoding rejected `NaN` before the worker's anchor field validator, so the result was correctly `BLOCKED` but its detail lacked the specific `anchor close` diagnostic required by the contract.

**Fix:** `loads_unique_json()` remains strict by default and continues rejecting duplicate keys. The worker request boundary retries only non-standard numeric constants with duplicate-key protection intact, then recursively rejects non-finite values; the dedicated anchor check now reports its specific finite-positive requirement. Added coverage that a non-finite `seed` still fails at its own field path.

**Verification:** the focused worker request/duplicate-key selector passes; the full RouteB app `tests/world_model` suite passes (**228 passed, 14 JAX-only skipped**; the earlier Hosted JAX CPU job passed). Full Ruff passes; MyPy passes for **225 source files**; changed modules compile. The source manifest verifies **604 files**, identity `c0bbd1ed76eab14292776aceb96b86478274b2494f59a2e6e89dfe7fc70637ab`. Evidence: `docs/audit_evidence/20261023/worker_cli_anchor_nonfinite_validation_20261023.json`. The hosted failure is from `d1defb7298314ce6231178842dac204f23c75619`; validation on the repaired revision remains pending.

**Next cursor:** obtain a clean Hosted Linux/Windows run on the repaired branch revision, then continue dependency-satisfied A01–A14 code audit. External cashflows, independent Validation, authorized market scope, authentic depth, historical trial coverage, and M4.2 remain separate gates.

## 2026-10-23 — M0.2 Linux installed-wheel registry root discovery

**Finding:** Hosted run `39836917121` exposed that `project_registry()` used `Path(__file__).parents[4]` as the checkout root. Linux wheel paths include an extra `lib/pythonX` level compared with the Windows venv layout, so the heuristic selected `.venv` and could not locate the project registry inventory when the test changed cwd.

**Fix:** discover the workspace root by scanning resolved package-file ancestors for the workspace marker. The regression fixture simulates `.venv/lib/python3.14/site-packages/rles_orchestrator/research_access.py`, changes cwd to a temporary path, and still proves the registry resolves to the checkout's `.artifacts` directory.

**Verification:** `tests/test_research_access.py` — **115 passed**. Full Ruff and MyPy (`packages qa`, **225 source files**) pass; modified modules compile; source manifest verifies **604 files**, identity `0fcdb77a15f788b828df2ee7e360a2908a3fc5f8f03b18e82493751609372fe0`. Evidence: `docs/audit_evidence/20261023/research_registry_linux_wheel_root_20261023.json`. No project registry or formal data was opened. Hosted Linux/Windows/JAX rerun remains pending after this correction.

**Next cursor:** rerun Hosted PR CI on the corrected revision, then continue the dependency-satisfied A01–A14 source audit. Keep the external evidence gates unchanged.

### 2026-10-23 M0.2 Hosted CI closure

Hosted PR run `39839975757` on `adf44ad58a5d7fe70d880673f6ae81b0a44ab230` passed end to end: Linux CPU, Windows CPU, and dedicated world-model JAX CPU jobs. Both OS jobs passed clean locked install, source manifest, all distribution builds, wheel/source match, installed-wheel provenance, MyPy, PR contracts/unit, the complete Route B world-model suite, and Ruff. The JAX lane passed JAX-specific unit tests and prior-causality/checkpoint-resume smokes. This closes M0.2 Hosted CI for that revision; evidence is recorded in `docs/audit_evidence/20261023/research_registry_linux_wheel_root_20261023.json` and `docs/audit_evidence/20261023/worker_cli_anchor_nonfinite_validation_20261023.json`.

**Next cursor:** continue the dependency-satisfied A01–A14 source audit. M0.1 authenticated cashflows, M0.3 historical pre-registration coverage, independent Validation/market authorization, authentic depth, matching E1/R1/E2 artifacts, historical trial completeness, and M4.2 remain separate evidence or input gates.

## 2026-10-23 — F7 Native event-history storage follow-up

The F7 audit found duplicate retained state in online reporting. Unique fills and Funding were copied into both reporting and reconciliation indexes; liquidation order stable facts were copied into a separate identity map alongside the latest full order facts. Reconciliation now receives each newly deduplicated fill/Funding exactly once and shares the report indexes. Liquidation identity comparison derives stable fields from the latest order snapshot, removing the duplicate immutable-facts index while retaining the cumulative filled-quantity regression guard. Exact report output and reconciliation semantics are covered by the Native accumulator, liquidation, research metrics, and historical-factory source selectors. This reduces duplicate structures; it does not remove the required unique event detail or prove constant-memory behavior. Evidence: `docs/audit_evidence/20261023/f7_native_event_fact_storage_20261023.json`.

**Next cursor:** continue source auditing other locally actionable A01–A14 items; preserve the existing external account provenance, historical access, independent sample, and market-depth gates.

## 2026-10-23 — M0.1 AccountFactsAccumulator failed-update consistency

**Finding:** `_StreamingSeriesMetrics.add()` can update running statistics before detecting a compounded-path overflow. `AccountFactsAccumulator.add()` previously exposed the exception but left the accumulator reusable, so a caller could continue from partially changed state and potentially emit misleading metrics.

**Fix:** store the first ingestion failure in `AccountFactsAccumulator`; reject later `add()` and `result()` calls while chaining the original exception. A successful interim `result()` remains non-finalizing and callers may continue normal incremental accumulation.

**Verification:** a deterministic sequence of finite equities doubles from `1e-100` until it reaches the path-overflow branch after streaming-statistic mutation. The four-file Native/metrics/historical-factory source selector passes **210 tests, 18 dependency warnings**. The source and test also pass Ruff; targeted MyPy for `metrics.py`, in-memory compile, and `git diff --check` pass. Offline `rles-research` wheel built and was installed into RouteB `.venv-win-app`; the research-metrics + native-accumulator parity wheel selector passes **114 tests** with `PYTHONPATH` unset. Source and installed `metrics.py` SHA-256 both equal `944dfa3b521dadcbe7b371958080fd02f08eba0043d1223ecd01854e94fb6b17`. Evidence: `docs/audit_evidence/20261023/m01_account_metrics_fail_closed_20261023.json`.

**Boundary / next cursor:** synthetic numerical facts only; no market, account, registry, Validation, Test, or Holdout payload was accessed. Continue dependency-satisfied A01–A14 source audit; M0.1 provenance and history-completeness, M0.3 pre-registration history, independent Validation, new market-scope authorization, authentic depth, historic attempt coverage, and M4.2 remain separate external/input gates.

## 2026-10-23 — M0.1 finite-input aggregate overflow

**Finding:** individually finite account-row costs/flows could overflow cumulative fee, funding, slippage, turnover, or external-flow totals. Finite realized trade PnLs and annualization factors could also produce non-finite derived report metrics.

**Fix:** validate running aggregate increments and totals before committing an account fact. Use checked `math.fsum` for gross-return and realized trade-PnL sums, reject non-finite profit factors and annualized Sharpe/Sortino, and retain fail-closed behavior after failed ingestion.

**Verification:** new regressions cover two finite `1e308` fee rows overflowing the accumulated fee and finite realized trade PnLs overflowing aggregate profit. The Native/metrics/historical-factory source selector passes **212 tests, 18 dependency warnings**; the freshly rebuilt RouteB research wheel metrics + accumulator parity selector passes **116 tests** with `PYTHONPATH` unset. Full Ruff and MyPy (`packages qa`, **225 files**) pass; source/installed `metrics.py` SHA-256 matches at `a124b4c5c4896646933e375342539303930cebf9aaadb808a7c9031d081e60bf`. The 604-file source manifest verifies identity `2ba0e7a63a5e99fb527b863ec2d2f783c581f2bf09ca8b381f9f1595401dd569`; compile and diff checks pass. Evidence: `docs/audit_evidence/20261023/m01_metric_aggregation_overflow_20261023.json`.

**Boundary / next cursor:** synthetic finite values only; no market, account, registry, Validation, Test, or Holdout payload was accessed. Continue dependency-satisfied A01–A14 source audit; authenticated cashflows, pre-registration history, independent Validation, new market authorization, authentic depth, historic trial coverage, and M4.2 remain external/input gates.

## 2026-10-23 — M0.1 initial liquidation-event guard

**Finding:** batch and online linear-equity reconciliation checked that the initial barrier had no positions, regular orders, fills, or Funding, but omitted native liquidation orders. A pre-existing liquidation order could be accepted as the flat, event-free baseline.

**Fix:** include visible liquidation orders in both initial-barrier guards; batch reconciliation also validates a provided liquidation-order history as a sequence.

**Verification:** the same synthetic baseline with a submitted, zero-filled liquidation order is rejected by both batch and online paths. The four-file Native/metrics/historical-factory selector passes **214 tests, 18 dependency warnings**; the RouteB app-venv wheel parity selector passes **118 tests** with `PYTHONPATH` unset. The offline `rles-research` wheel build/install succeeds and the installed/source `native_metrics.py` SHA-256 matches at `bf72b60d7141c72b01981933cb706e77458422ec0aa7cfed440b47ee2c03d53c`. Ruff and targeted MyPy pass; the 604-file source manifest verifies identity `137a3fee290d06711a0bf487521a79933ca0ba718461364647182b30835687f1`; compile and diff checks pass. Evidence: `docs/audit_evidence/20261023/m01_initial_liquidation_event_guard_20261023.json`.

**Boundary / next cursor:** synthetic snapshots only; no market, account, registry, Validation, Test, or Holdout payload was accessed. Continue dependency-satisfied A01–A14 audit; authenticated cashflows, historical access coverage, independent Validation, new market authorization, authentic depth, historical trial coverage, and M4.2 remain separate external/input gates.

## 2026-10-23 — M0.1/A08 signed native fee rebate accounting

**Finding:** instrument rules and the Native fee model retain negative maker/taker rates as exchange rebates; commission parsing also allowed signed Money amounts. `AccountFactsAccumulator` nevertheless rejected every negative row fee, so direct facts and batch/online native reports failed on valid rebates.

**Fix:** accept finite signed fee rows while retaining nonnegative turnover, positive equity, and existing slippage constraints. The signed fee remains present in `total_fee` and gross-return normalization removes it consistently.

**Verification:** a direct account-facts rebate test and batch/online Native report parity test failed before the fix and pass after it. The Native/metrics/historical-factory source selector passes **216 tests, 18 dependency warnings**; the RouteB app-venv wheel parity selector passes **120 tests** with `PYTHONPATH` unset. Offline research wheel build/install succeeds, and source/installed `metrics.py` SHA-256 matches at `dcb22bafd1249274ed3a096375605849cae96532744813892be15b3c0b9d4985`. Full Ruff and MyPy (`packages qa`, **225 files**) pass. The 604-file source manifest verifies identity `a2fd164e9c63a2bc13989bba8d38d4ed6c2ed894da9a1c85d94ead126809d69f`; compile and diff checks pass. Evidence: `docs/audit_evidence/20261023/m01_signed_native_fee_rebate_20261023.json`.

**Boundary / next cursor:** synthetic fee/equity facts only; no market, account, registry, Validation, Test, or Holdout payload was accessed. Continue dependency-satisfied A01–A14 audit; external cashflow provenance, pre-registration history, independent Validation, new market authorization, authentic depth, historic trial coverage, and M4.2 remain separate gates.

### 2026-10-23 M0.2 follow-up — canonical inventory checksum bytes

Hosted run `39837931130` confirmed the Linux wheel root fix, then failed when reaching the inventory: the Git blob was LF (`2950db3e…`), while its marker matched only Windows CRLF working-copy bytes (`9fb401e…`). Added exact-byte `-text` attributes for both inventory and marker, normalized the tracked file to LF, and rebound the marker to the LF SHA-256. A further package-root audit found the same fixed-depth assumption in WFO source hashing and world-model CLI code identity; both now discover the workspace marker through package-file ancestors. The research-access + world-model CLI suites pass **131 tests**, WFO passes **36**, full Ruff/MyPy (**225 source files**) pass, and the 604-file source identity is `59a584b0864d448535e1905a39905e2099536050a9d7550fad94d272327deb73`. Updated evidence: `docs/audit_evidence/20261023/research_registry_linux_wheel_root_20261023.json`. Hosted verification of the combined fixes remains pending.

## 2026-10-23 F7 fixed-memory tail selection

- **Finding:** The streaming series metric retained ceil(max_count * tail_quantile) Python floats in its worst-tail heap; peak tail memory therefore grew with authorized account-history length.
- **Change:** Cap the in-memory lower-tail heap at 8,192 observations. When the configured exact tail exceeds that cap, retain returns in an auto-deleted disk spool and use NumPy in-place memmap.partition to select the exact VaR boundary. CVaR is computed over the selected memory-mapped view. The spool uses native float64 encoding, remains appendable after an interim result(), and can be partitioned again as more barriers arrive.
- **Regression / verification:** Small-tail and forced large-tail regressions pass, including signed-zero boundary values; the large-tail case asserts the heap is absent and checks exact VaR/CVaR against the sorted reference. The four-suite RouteB isolated selector (native report parity, research metrics, historical factory, native liquidation facts) passes **218 tests, 18 dependency warnings**. The 122-test focused metrics/parity selector, Ruff, targeted MyPy, and diff checks pass. On 1,000,000 synthetic returns with K=50,000, all streaming metrics matched the legacy heap reference; tracemalloc peak was 217,746 bytes with memmap selection versus 2,046,312 bytes with the old tail heap (89.4% lower), and measured time was 5.800s versus 6.216s while tracing.
- **Boundary / next cursor:** The memory figures cover Python allocations under tracemalloc, not process RSS or a full history workload. Inputs were synthetic only; no account, market, Validation, Test, or Holdout payload was read. Continue remaining M0.1/A01–A14 implementation and separate externally gated requirements.

Evidence: docs/audit_evidence/20261023/f7_bounded_tail_memory_cap_20261023.json.

## 2026-10-23 F7 exact-tail RSS correction

Process-RSS profiling found that whole-history `memmap.partition` lowered Python-traced allocations but increased resident working set: on 5,000,000 synthetic returns it peaked 40,513,536 bytes over baseline, versus 10,199,040 bytes for the exact heap. Replaced it with bounded 65,536-value, eight-pass radix selection plus a stable bounded scan for exact CVaR. The latest comparison matches all metrics and reduces sampled RSS peak delta from 10,129,408 to 4,018,176 bytes (60.3%); runtime rises from 5.370s to 5.956s (10.9%). Source and installed-wheel four-suite selectors each pass 224 tests with 18 dependency warnings; installed/source module SHA, 72-package compatibility, Ruff, and 226-file MyPy checks pass. Runner/evidence: `qa/profile_f7_tail_rss.py` and `docs/audit_evidence/20261023/f7_tail_process_rss_20261023.json`.

**Remaining F7 gate:** this is synthetic metric-stream evidence, not a complete Native historical Train RSS/trajectory profile. Keep that gate open; preserve the remaining authenticated cashflow, historical access, independent Validation, market scope/depth, historical trial coverage, and M4.2 gates.

### F7 continuation — accurately labeled current-source long-window profile

The profile is schema v2 and compares online summary reporting through a snapshot sink against lean execution; neither mode retains a per-barrier snapshot sequence. The 90-day current-source run processed 129,597 steps with identical full trajectory hashes. RSS deltas were 510,554,112 and 510,062,592 bytes; this single pair shows no stable memory gain, while online summary reporting took 13.2% longer. Evidence: `docs/audit_evidence/20261023/f7_synthetic_native_growth_current_20261023.json`.

Keep F7 open. Next code work must measure and reduce O(n) retained report/event state without changing economic report or reconciliation output. The synthetic profile does not replace a separately authorized real-history Train RSS/trajectory run.

### F7 continuation — lazy frames, slotted FeatureRows and direct timestamp bisect

The 90-day installed-wheel replay processes 129,597 decisions with unchanged trajectory SHA-256. Lazy Fold frames and compact metadata keep online/lean replay RSS delta near 193 / 193 MB versus the original 510 / 510 MB baseline. Slotted FeatureRows retain hash and pickle behavior and save 88 bytes per row by object-size accounting. Removing three duplicate timestamp tuples from CuratedStore reduced 90-day source-generation peak RSS by 10,440,704 bytes and post-generation RSS by 8,245,248 bytes; point-in-time query tests pass unchanged. Source and installed-wheel selectors each pass 172 tests. Evidence: `docs/audit_evidence/20261023/f7_lazy_barrier_frames_20261023.json`.

Next code cursor: audit duplicate event-reference indexes and Native session/cache/report history. Preserve late-arriving-event visibility and exact report parity. Authenticated real-history Train, external cashflow provenance, historical-access completeness, independent Validation, market scope/depth, historical trial coverage and M4.2 remain open.

## 2026-10-23 F7 duplicate event-reference follow-up

**Finding and fix:** `CuratedStore` kept separate per-kind tuples for availability-time and event-time lookup even when the event ordering was identical. It now aliases the event-time view to the availability view for chronological histories and keeps a separate sorted tuple only when late arrivals make the orders differ. Removed the unused `HistoricalEnvironmentFactory._bars` Fold-wide OHLCV reference tuple. `NativeReportAccumulator` no longer stores a per-cashflow-timestamp barrier-count dictionary: duplicate barriers are rejected from the chronological previous timestamp, and one scalar count verifies that every cashflow timestamp matched exactly one native barrier.

**Verification:** source selector passed **160 tests** across layer invariants, Native report parity, research metrics, and the lazy-frame historical factory regression. A broader data and historical factory selector passed **148 tests** before removal of `_bars`; the focused historical factory regression was rerun in the 160-test selector. Ruff passed, and MyPy passed for the three changed source modules. On the 389,258-event synthetic store, all 389,258 visible event references shared per-kind tuples for the five event kinds. This avoids **3,114,064 bytes** of duplicate tuple pointer storage; removing the unused 129,662-row `_bars` tuple avoids another **1,037,296 bytes** of pointer slots by tuple-size accounting. These are structural measurements, not process-RSS claims. Evidence: `docs/audit_evidence/20261023/f7_duplicate_event_indexes_20261023.json`.

**Status / next cursor:** these changes reduce duplicate indexes while preserving late-arriving visibility and exact batch/online cashflow-report parity. F7 remains open for the full-history Native RSS/trajectory profile and representative growth of unique fill, Funding, order, and cashflow detail. Continue source-auditing Native session/cache/report retention without dropping report-contract facts. Authenticated cashflow provenance, historical-access completeness, independent Validation, market authorization/depth, historic trial coverage, M4.2, and a separately approved real-history Train profile remain open gates.

