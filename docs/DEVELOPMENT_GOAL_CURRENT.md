# RouteB 审计整改：当前连续开发目标

## 2026-10-23 最新续做：B4 Native/Imagined 限价触发对齐

多空对照发现，想象账户原先会因 OHLC 盘中高低点触价而成交；固定 Nautilus bar engine 在收盘仍不满足限价条件时保持挂单。现将默认触发改为按 bar close 判断，并保留原 OHLC touch 近似为显式 `limit_fill_policy="ohlc_touch"` 场景。多空限价收盘成交及盘中触价未收盘穿价对照加入回归。聚焦选择器 **11 passed**；适配器与 WorldModel 三文件 **136 passed、5 个 JAX-only 跳过**；Ruff、MyPy 235 文件、语法编译、差异检查通过。source manifest：606 files，identity `9d6d7433a98162ae95fcda9d80a680a7a4a3e108b83e984b7ad22c6217527ac3`。证据：`docs/audit_evidence/20261023/b4_imagined_native_limit_close_parity_20261023.json`。仅使用合成 BTC/USD.SIM OHLC；Hosted 验证待提交后运行，盘中撮合次序、队列位置、部分成交和自定义滑点 fill model 仍未覆盖。继续其它本地可解决审计项。

## 2026-10-23 最新续做：M0.1 接受 Native maker fee 返佣

审计发现数据规则和原生费用模型已支持负 maker/taker 费率返佣，Native commission 解析也保留有符号金额，但账户指标累加器仍拒绝负手续费，致使返佣在直接、批处理和在线经济报告中报错。现允许有限的有符号手续费；成交额仍非负、权益必须为正，滑点约束不变。新增直接账户事实与 batch/online Native parity 回归。四个源码套件 **216 passed、18 warnings**；RouteB 独立 wheel 两套件 **120 passed**。离线 wheel 已安装到 RouteB app venv，源/安装模块 SHA-256 一致；全仓 Ruff 与 MyPy 225 文件通过；604 文件 source manifest 验证通过。证据：`docs/audit_evidence/20261023/m01_signed_native_fee_rebate_20261023.json`。仅合成费用事实；未访问市场、账户、登记库或正式评价数据。继续剩余可本地实施审计项。

## 2026-10-23 最新续做：线性权益初始屏障拒绝预存清算订单

复核 M0.1 对账不变量发现，批处理与在线线性权益对账的初始屏障均漏检 `liquidation_orders`，预存强平订单可能被接受为“无事件”基线。现在两条路径都拒绝初始屏障中的可见清算订单；批处理入口也校验提供的清算订单历史类型。新增同一合成输入对批处理/在线路径的双参数回归。四个 Native/metrics/历史工厂源码套件 **214 passed、18 warnings**；RouteB 独立 wheel 两套件 **118 passed**。Wheel 离线构建并安装成功，源与安装模块 SHA-256 一致；Ruff、目标 MyPy、语法编译、清单验证和差异检查通过。Source manifest：604 files，identity `137a3fee290d06711a0bf487521a79933ca0ba718461364647182b30835687f1`。证据：`docs/audit_evidence/20261023/m01_initial_liquidation_event_guard_20261023.json`。仅合成快照；未访问行情、账户、登记库或 Validation/Test/Holdout。继续其余可本地解决审计项。

## 2026-10-23 最新续做：有限输入的累计指标溢出保护

继续 M0.1 数值审计发现，单行手续费、Funding、滑点、现金流或换手率增量都有限，跨行累计仍可能溢出为 `inf`；已实现交易 PnL 的求和以及年化 Sharpe/Sortino 也存在有限输入产生非有限输出的路径。现对每次累计增量和累计值做有限性校验，用 `math.fsum` 并捕获溢出计算总毛收益与闭合交易 PnL，且拒绝非有限 profit factor/年化比率。新增高量级有限费用与盈亏累加回归。四个源码回归套件 **212 passed、18 warnings**；RouteB 独立 wheel 两套件 **116 passed**，wheel 与源码哈希一致；全仓 Ruff、MyPy 225 文件通过；604 文件 source manifest 已验证。证据：`docs/audit_evidence/20261023/m01_metric_aggregation_overflow_20261023.json`。仅合成数值；无市场、账户、登记库或正式评价数据访问。继续其余可本地完成的 A01–A14；外部现金流真实性、历史登记覆盖、Validation、新授权样本和真实盘口深度仍保持各自独立门槛。

## 2026-10-23 最新续做：AccountFactsAccumulator 增量失败关闭

审计发现 M0.1 账户指标累加器可能在流式指标部分更新后才遇到复利溢出，原实现允许调用方捕获异常后继续使用该对象，可能产出部分状态的报告。现在累加器保存首次摄入失败，并拒绝后续 `add()` 与 `result()`，错误保留原始异常链；成功的中间 `result()` 仍可继续累积。新增有限权益按 2 倍增长至路径溢出的回归，确认失败后不可恢复使用。四个 Native/metrics/历史工厂源码选择器 **210 passed、18 warnings**；RouteB 独立 wheel 下指标与 Native parity 选择器 **114 passed**。离线研究 wheel 构建并安装成功，源码与隔离 venv 模块 SHA-256 相同；Ruff、目标 MyPy、语法编译和 `git diff --check` 通过。证据：`docs/audit_evidence/20261023/m01_account_metrics_fail_closed_20261023.json`。仅合成数值；没有访问行情、账户、项目登记库或 Validation/Test/Holdout。继续其余可本地实施审计项；外部现金流真实性、登记前历史覆盖、独立 Validation、新授权样本及真实盘口深度仍是各自独立门槛。

## 2026-10-23 最新续做：减少 Native 在线报告重复事件事实

审计 F7 在线摘要发现：启用线性权益对账时，累加器把每个 unique fill/Funding 字典同时复制进报告索引和独立对账索引；liquidation 订单还另存 stable-facts 字典。现改为 fill/Funding 一次去重并共用唯一事实索引，liquidation stable-field 比对从最新订单状态即时重建；累计成交量回退检查仍保留。指标、明细和权益对账输出保持一致；unique 事件事实仍按报告合同保留，因此不声称常数内存。四个定向源码套件（Native accumulator parity、研究指标、历史工厂、liquidation）共 **209 passed，18 dependency warnings**；RouteB 独立 app venv 中的 wheel 回归 **20 passed**。离线构建 wheel 成功，安装模块与源码 SHA-256 相同；69 个依赖兼容。Ruff、MyPy 与 `git diff --check` 通过。Source manifest：604 files，identity `11936595d1a5ac6b6fd1552cbdd6c4c37a21612407dc5e4794cdf7a16da8f00d`。证据：`docs/audit_evidence/20261023/f7_native_event_fact_storage_20261023.json`。仅合成/临时测试，无行情、账户、项目账本或正式评价数据访问。继续审计 A01–A14 中其余可本地完成项；真实长窗和外部认证门槛仍按原状态处理。

## 2026-10-23 最新续做：修复 Linux wheel 下项目登记库定位

Hosted CI #9 的 Ubuntu 基础测试暴露 `project_registry()` 固定取 `__file__.parents[4]`，Linux 的 `lib/pythonX/site-packages` 目录更深，导致将 `.venv` 错当仓库根。现改为从包文件祖先中查找 workspace marker，并用模拟 Linux wheel 路径和临时 cwd 验收。研究访问回归 **115 passed**，全仓 Ruff、MyPy（225 files）通过；604-file source manifest identity `0fcdb77a15f788b828df2ee7e360a2908a3fc5f8f03b18e82493751609372fe0`。证据：`docs/audit_evidence/20261023/research_registry_linux_wheel_root_20261023.json`。Hosted 修复版重跑待提交后触发。

### 2026-10-23 最新复核：修正历史 inventory 的跨平台哈希

Hosted CI #10 已越过路径发现，但发现历史 inventory 的 marker 只匹配 Windows CRLF 工作副本；Git blob/LF 的 SHA 是 `2950db3e…`。现对 inventory 和 marker 增加 `-text` 精确字节属性，inventory 规范化为 LF，marker 改为 LF 内容 SHA。研究访问回归 **115 passed**；Ruff、MyPy（225 files）通过；604-file source identity `b68138cd0fe7ba378f03c4edd6743937ee796c9da28659984f23bf39b5fdca2c`。修正后的 Hosted CI 验证待提交。

复核发现另外两个同类固定深度根路径：WFO source-tree hash 与 world-model CLI repository identity。它们现共用 workspace marker 的祖先查找逻辑。研究访问 + world-model CLI **131 passed**，WFO **36 passed**；全仓 Ruff、MyPy（225 files）通过；当前 604-file source manifest identity `59a584b0864d448535e1905a39905e2099536050a9d7550fad94d272327deb73`。三个入口及 inventory byte-stability 一并等待远端验证。

### M0.2 Hosted gate 已通过（2026-10-23）

Hosted Actions run [#11](https://github.com/XXA222/NautilusTrader_RL_ES_RouteB/actions/runs/39839975757)，commit `adf44ad58a5d7fe70d880673f6ae81b0a44ab230`：Linux CPU、Windows CPU、world-model JAX CPU 全部通过。两平台的锁定安装、604-file source manifest、12-wheel/sdist build/source match、installed provenance、MyPy、PR contract/unit、world-model suite、Ruff 全绿；JAX 专项单测与因果性/checkpoint-resume smoke 通过。M0.2 CI 代码门禁已闭环，继续 A01–A14 其它依赖已满足的源码审计。其它数据与研究认证门槛仍按任务索引独立处理。

## 2026-10-23 最新续做：修复 Hosted CI worker 字段诊断

读取 GitHub Actions #8 作业日志发现 Windows/Linux 的 world-model selector 因 `anchor_close=NaN` 报通用 JSON 错误而失败（219 passed、14 skipped、1 failed；JAX CPU 作业通过）。现保留 JSON 默认严格解析和重复键保护，仅让 worker request 在非标准数值进入字段校验前做安全的唯一键解析；递归拒绝其它非有限字段，anchor_close 返回字段专用拒绝。修改后 app venv 的完整 `tests/world_model` 回归 **228 passed、14 个 JAX-only skipped**，全仓 Ruff、MyPy（225 files）通过；604-file source manifest identity `c0bbd1ed76eab14292776aceb96b86478274b2494f59a2e6e89dfe7fc70637ab`。Hosted 修复版重跑仍待确认。证据：`docs/audit_evidence/20261023/worker_cli_anchor_nonfinite_validation_20261023.json`。此前 A04 legacy R4 损坏账目拒绝修复亦已完成。

## 2026-10-23 最新续做：A04 legacy R4 账本损坏记录

复核旧版 R4 失败报告的唯一 attempt 匹配时发现，账本中畸形 recipe/event-detail JSON 曾被静默跳过。现改为失败关闭并拒绝不完整账本视图；临时合成 SQLite 回归覆盖损坏配方和非对象事件 detail，合法单匹配恢复仍通过。对账器回归 **26 passed**，Ruff、py_compile、全仓 MyPy（225 files）通过；604-file source manifest identity `48815532670160883cc0f2f1c37a82aeab8603e6bd9a9a52ea5101bf0c911350`。证据位于 `docs/audit_evidence/20261023/reconcile_directory_scan_fail_closed_20261023.json`。无项目账本、市场、账户或正式评估数据访问。继续 A01–A14 代码审计。

持续在独立工作区 `D:\Program Files\NautilusTrader_RL_ES_RouteB` 落实 [审计任务索引](DEVELOPMENT_GOALS_REVIEW_20261017.md) 中仍可实施的 A01–A14 工程整改。每轮从实际源码和开发执行记录继续，直接实现并完成必要回归；不能以审计、计划或文档更新替代代码工作。

## 执行约束

- 只修改 RouteB；保持运行库、环境、配置和账本独立，不触碰主项目或 RouteA。
- 保留现有未提交改动；不 reset、clean，也不改 NautilusTrader 上游源码。
- 正式研究只能使用项目账本授权的未消费窗口；不得重跑已消费 Fold，不解封 Test/Holdout，不用来源哈希代替真实性认证。
- 按任务风险运行有针对性的测试、Ruff 与 MyPy，并记录结果、数据用途及未认证边界。
- 只有依赖已满足的工程项均已完成，且剩余项确实需要外部证据时才暂停；这不代表策略或研究目标完成。

## 当前续做位置（2026-10-22）

### 2026-10-22 后续游标 — A08 instrument 标识边界

共享 Event schema 原先只检查 instrument 是 truthy 字符串，空格、首尾空白和 tab 前缀会通过。现要求标识为非空且首尾无空白；4 个红测复现后已修。数据事件、Binance 归档、JSON 边界和 hash identity 当前源码及隔离 wheel 各 **77 passed**；历史规则清单相关用例源码与 wheel 各 **2 passed**。Ruff、py_compile、目标 MyPy 通过；12 wheels/12 sdists 构建且 wheel 内容 12/12 匹配；schema 源码/已安装 wheel SHA-256 均为 460978f67bb8724c4beb027bd66582117148d5ebcef8f8e3b1a2d57d98c7de51。source manifest 604 files、identity 5af7918e196a72520c0799df1a5a782044ec4809e93a85564359b4469fc03af3。证据：docs/audit_evidence/20261022/a08_event_instrument_whitespace_20261022.json。继续 A01–A14 本地审计。

### 2026-10-22 后续游标 — A08 signed instrument fee rebates

统一 instrument_rule 校验器原先把 maker_fee/taker_fee 都按非负数验证，但 Nautilus perpetual 费用逻辑会保留负 rebate，导致有效 rebate 规则在数据入口被拒。已按字段约束将两项 fee 改为接受有符号有限值，保证金率仍为非负；先前红测复现 maker/taker 两项失败。data event guards、archive、hash identity、historical factory 的源码与隔离 wheel 联合回归各 **167 passed、18 warnings**；单独 guards **19 passed**，费用合同 **3 passed**。Ruff、py_compile、目标 MyPy 通过；12 wheels/12 sdists 构建，wheel source match 12/12；source manifest 604 files、identity e51add80b68e69a1498b56627e984aaeca941a28098c2ff681dacccb0b3c78f1。证据：docs/audit_evidence/20261022/a08_signed_fee_rebates_20261022.json。继续 A01–A14 本地审计。

### 2026-10-22 后续游标 — A09/M4.1 三标的现行规则验收

从 Binance USDⓈ-M 官方 GET /fapi/v1/exchangeInfo 保存 BTCUSDT、ETHUSDT、SOLUSDT 三项原始 symbol 规则，serverTime 为 2026-10-22T06:35:32.110Z，fixture SHA-256 4d400184cb16c8adb9987c9f12c1c3a389a682bb4ec479decbe1f452f57772a4。新增参数化原生引擎验收，从快照逐标的装载 tick/step/min-notional，执行合约数量与 Funding 结算并核对权益始终以 USDT 计价；3 个新用例通过，完整 Nautilus adapter smoke 为 36 passed。Funding 输入为机械单位检查用的合成事件、手续费设为零、乘数 1 是线性基础数量约定；这些结果不证明历史生效规则、个体手续费层级或真实账户流水。证据：docs/audit_evidence/20261022/a09_current_usdm_rules_20261022.json。M4.1 当前规格的工程验收已补；历史规则与真实经济核对仍需独立认证。

最新工程续做位置：F7 在线 Native 摘要累积器已接入 `rollout()` 的无账户行路径，直接接收引擎屏障快照并增量汇总指标/对账；磁盘 snapshot spool 保持显式 opt-in，因为 90 天合成轨迹虽与基线逐步相同、峰值 RSS 少约 8.2 MB，但单次耗时多约 42 秒。变化点列式容器基准节省 52,688,220 bytes。F7 仍为 PARTIAL：真实长窗 Train RSS/轨迹证据缺失，保留账户行的完整报告仍需物化。继续 A01–A14 源码审计，不把合成性能观察写成市场结论。

### 2026-10-23 后续游标 — F7 rollout 工厂接口边界

完整 Windows PR 标记选择器发现 `rollout()` 在普通报告路径也访问 `factory.rows`，与通用环境工厂接口不符。现在仅在无账户行、非 spool 的在线摘要路径读取该属性；测试工厂显式提供原生报告合同要求的单位 `contract_multiplier`。修后 Windows selector **855 passed、4 skipped、597 deselected**；RouteB 安装 wheel 的 CLI 与 Native accumulator selector **57 passed**。全仓 Ruff 与 MyPy（225 个源码文件）、py_compile、`git diff --check` 通过；12 wheels/12 sdists 全部构建且 all-match，12 个 RouteB wheel 安装到独立 `.venv-win-app`，12 modules/6 entry points provenance 与 69 包兼容检查通过。604-file source identity：`44ad07a1e5fac4b6deac0eb82302f391c305ae354ca715a7651a03696d542ace`。证据：`docs/audit_evidence/20261023/f7_rollout_factory_interface_20261023.json`。没有访问行情、账户、登记库或 Validation/Test/Holdout 数据。Hosted CI 和需要外部认证的研究门槛仍开放；继续 A01–A14 其余可实施工程项。

### 2026-10-22 后续游标 — Bar Base/Stress 乘数

继续审计发现 Train-inner Bar Base/Stress 诊断的 `filled_quote_notional` 仍漏乘冻结规则中的 `contract_multiplier`。现已要求订单汇总显式传入正有限 Decimal 乘数，并由每个历史工厂传入，报告同时记录该值。合成回归与冻结计划选择器共 **9 passed**，Ruff、py_compile、目标 MyPy 及 595-file source manifest 均通过，identity `28127481828a9f40b96de186643190c07e741fc8dfc6662e852338e5edcfc859`。证据：`docs/audit_evidence/20261022/a09_bar_cost_multiplier_20261022.json`。没有打开或重跑研究窗口。继续审计其他 A01–A14 本地工程路径。

### 2026-10-22 后续游标 — Binance aggTrades 非有限值

`aggTrades` CSV 输入此前没有在 Decimal 比较前检查有限性：Infinity 会拖到 Event 校验才失败，NaN 则泄漏 `decimal.InvalidOperation`。现在解析边界先拒绝非有限价格/数量；四个合成 ZIP 回归复现旧行为后全部通过。归档 + data-event 选择器源码和隔离 wheel 各 **34 passed**，Ruff、py_compile、目标 MyPy 通过。12 wheels/12 sdists 构建且全部 12 wheels 与源码匹配，安装模块 SHA 相同；595-file source identity `42726740515c92e3a5244c6980ae8c046f7cedafe3217a37fd0d056475f6ad33`，依赖检查 69 compatible。证据：`docs/audit_evidence/20261022/a08_aggtrade_nonfinite_values_20261022.json`。没有下载或打开行情归档，继续本地 A01–A14 审计。

### 2026-10-22 后续游标 — Binance archive CSV 解码

三类 CSV 流的损坏 UTF-8 此前会泄漏 `UnicodeDecodeError`。共享流式 reader 现统一映射 Unicode/CSV 语法错误到 `DataError`。三项 ZIP 回归复现后通过；归档/data-event 选择器源码与隔离 wheel 各 **37 passed**，Ruff、py_compile、MyPy 通过。12 wheels/12 sdists 构建并全数 source-match，安装模块 SHA 等于源码；595-file source identity `105dd6cc2afabd851182e1a3b0484134467b88fa264142cebff53081a315eb03`，69 依赖兼容。证据：`docs/audit_evidence/20261022/a08_archive_csv_utf8_errors_20261022.json`。没有下载或打开行情归档，继续 A01–A14 本地审计。

### 2026-10-22 后续游标 — Binance archive 整数字段

bookTicker、Funding、K 线归档中的损坏整数字段此前泄漏原始 `ValueError`。新增字段级转换后，归档/data-event 选择器源码与隔离 wheel 各 **40 passed**，Ruff、py_compile、MyPy 通过。12 wheels/12 sdists 构建且全数 source-match，安装模块 SHA 等于源码；595-file source identity `e4355283b62ee3113b6b1eaeb8bfa4ed3807025e0a5082d94cb1197f0a73a882`，69 个依赖兼容。证据：`docs/audit_evidence/20261022/a08_archive_integer_fields_20261022.json`。只用合成 ZIP，无行情归档访问；继续 A01–A14 本地审计。

### 2026-10-22 后续游标 — aggTrades Decimal 词法

USD-M aggTrades 的 Decimal 原先会接受下划线和 Unicode 数字并静默规范化。价格/数量现严格要求 ASCII 无符号定点格式；四项 malformed-lexeme 回归复现后通过。归档/data-event 选择器源码与隔离 wheel 各 **44 passed**，Ruff、MyPy 通过。12 wheels/12 sdists 构建且全数 source-match，安装模块 SHA 等于源码；595-file source identity `8c7b7cfa420dfbece8e2bc6828d605e1fc6d40bc157158d429204d2db2dc7d2c`，69 依赖兼容。证据：`docs/audit_evidence/20261022/a08_aggtrade_ascii_decimal_20261022.json`。只用合成 ZIP，继续 A01–A14 审计。

### 2026-10-22 后续游标 — 归档整数词法

`int()` 原先会把下划线、Unicode 数字和正号接受为合法 ID/时间。归档整数解析现只接受 ASCII 数字；5 个合成 ZIP 用例复现旧行为后通过。归档/data-event 选择器源码与独立 wheel 各 **49 passed**，Ruff、py_compile、MyPy 通过。12 wheels/12 sdists 构建且全数 source-match，安装模块 SHA 等于源码；595-file source identity `246deaf2f0296dd2ceafef840db955fc96db87c6bcb183061d91fb427081680a`，69 依赖兼容。证据：`docs/audit_evidence/20261022/a08_archive_ascii_integer_lexemes_20261022.json`。只用合成 ZIP；继续 A01–A14 审计。

最近执行记录见开发计划末尾 **CUDA long-run diagnosis and failure-context reporting**。同一故障 seed 的 synthetic 特征幅度 4/16/32/48× CUDA 500-step 均通过；60× CUDA 500-step 也通过。60× CUDA 2,067-step 曾两次约 400 updates 报 `CUDA_ERROR_ILLEGAL_ADDRESS`，之后一次同配置完整通过；CPU 同幅度 2,067-step 完成。故障有间歇性，根因仍未知。训练器现在在 update/指标同步失败时把 optimizer step、抽样索引摘要与 Train 输入范围加入异常上下文；41 个质量回归测试通过、3 个因 Windows app venv 缺 JAX 跳过，Ruff 与 MyPy（211 files）通过。当前 557-file source manifest identity 记录在开发计划。

最新续做为 **F7 synthetic native long-window RSS and trajectory parity**：最终 RouteB 源码下，实际 Native 环境在纯合成 1/30/90 天分别运行 1,437/43,197/129,597 个决策；保留/关闭报告快照的逐步轨迹一致，RSS 峰值增量为 79.97/0.36、254.13/15.78、618.17/136.90 MiB。依据实测结果，R1 学习专用的共享/重建环境显式关闭未使用的报告快照保留；外部 `rollout()` 经济报告仍保留快照能力。4 项焦点回归通过（含最终 wheel 安装状态）、Ruff 通过、MyPy 212 files clean。最终 558-file source manifest identity `967411f88f4dc2f589c293911c0fc11c2a74fa13c9fca5b69f9835be25036459`；12 wheels/12 sdists 构建且 `all_match=true`，12/12 modules 和 6/6 entry points provenance smoke 通过。最终 wheelhouse 在 workspace `routeb-wheels-20261022-f7-r1-final`。证据保存在 RouteB `docs/audit_evidence/20261022/`，不认证真实市场长窗，也未访问研究账本。继续沿审计索引处理下一个可实施缺口。

RouteB 本机分发验收已完成：12 wheels/12 sdists 构建成功，wheel manifest all_match，12 modules/6 entry points 已在独立 app venv 验证。下一步继续审计 A01–A14 中可本地实施的工程问题。不得重跑 SOL 已消费窗口，任何新研究需重新预注册和登记独立未消费窗口。

## 外部门槛与停止条件

仍未由源码工程替代的证据包括：经认证且覆盖完整的真实账户现金流、干净 Hosted Linux/Windows CI、独立 Validation、兼容的 E1/R1/E2 阶段工件、真实盘口深度、新授权独立市场窗口、覆盖账本登记前历史的有效试验数，以及 A09 标的/持仓/杠杆扩展所需输入与认证。它们各自只阻止依赖它们的验收；不能阻止其它独立工程工作，也不能把 RouteB 项目整体标记为完成。


**2026-10-22 R4 matched-objective continuation:** on a ledger-cleared DOGEUSDT Fold, the pinned E2/E3 runner completed all six seed/candidate runs in RouteB's isolated WSL CUDA environment. E3 significantly improved primary path log-RV CRPS in seeds 31 and 20261018 but significantly worsened seed 102; return-CRPS no-harm passed, so the strict all-seed gate failed and E3 was not promoted. Test OHLCV stayed sealed. Exploratory review of saved outputs found E3 worse on 44/60 paired anchors for seed 102, versus 17/60 for each other seed; this has no verified cause and changes no gate. Full evidence: `docs/world_model/R4_DOGEUSDT_E2_E3_RESULTS_20261022.md`. Do not retune or rerun this consumed Fold; a further comparison needs a new frozen hypothesis and independently cleared scope.

**2026-10-22 R4 run-02 temporal replication:** the same frozen protocol completed on the disjoint earlier DOGEUSDT scope `[2023-07-30, 2024-08-11)`. Again E3 significantly improved the primary metric on seeds 31 and 20261018 but significantly worsened seed 102 (E3−E2 `+0.03938`, 95% CI `[+0.00444,+0.07450]`, Holm `p=0.027486`). This replicates seed sensitivity across two periods without identifying its mechanism. Both all-seed gates fail; no promotion. Full evidence: `docs/world_model/R4_DOGEUSDT_E2_E3_RESULTS_RUN02_20261022.md`. Both windows are consumed and remain closed to retuning.

**隔离修复：** 检查发现 RouteB `.git/objects/info/alternates` 仍只读引用 `NautilusTrader_RL_ES` 的对象库。现已在 RouteB 本地 repack 所有 refs 可达对象并移除 alternate；RouteB `git status` 可用，`git fsck --full --connectivity-only` 无缺失对象。原 alternate 路径保存在 `docs/audit_evidence/20261022/`；未清理或重置工作区。

**2026-10-22 最新工程修复：JAX seed 身份域。** RouteB 训练/checkpoint/forecast/F2/worker 现在拒绝 32-bit JAX PRNG 域外及运行中派生 key 溢出，层级 batch keys 用 uint32。合成与定向 JAX regression 61 passed，JAX 集成子集 84 passed，完整 world-model app selector 200 passed/11 JAX-only skipped；Ruff、MyPy 7 个源文件通过。最终 566-file manifest `cb88bacc4d88ec1fbd2f5de9d3f50d36d00a9885392e2ef72eb57b158a8d44d0`，12 wheels/12 sdists all-match，12 modules/6 entry-point wheel provenance smoke 通过。该修复加强种子可区分性，不解释 DOGE seed 102 敏感性；不重用已消费 Fold。继续寻找下一项可实施工程缺口。

**2026-10-22 最终状态复验：** 在最后的 F2 高位 seed 用例加入后重新运行：JAX 0.11.2 RouteB 集成子集 **85 passed**；完整 `tests/world_model` app selector **200 passed、12 skipped**，跳过项均因 app venv 未安装 JAX，JAX 专项在隔离 CPU venv 单独全通过。该复验覆盖最终源码状态，取代上一段的 84-pass/11-skip 次数。

**2026-10-22 当前最新游标：** E2/E3 的可选 `SeedStreams` 已完成合成因子诊断。显式旧默认 RNG 根逐数组兼容，checkpoint resume 精确；聚焦 JAX tests **63 passed**，完整 app world-model selector **202 passed/12 JAX-only skipped**，Ruff、MyPy 3 源文件、diff check 通过。12 wheels/12 sdists 构建通过、all-match，最终 567-file source identity `8e840af0bc03b52384fd5fa8a48d7e6951a1a2a99fd227cb2fd3da60efd4bc8b`；RouteB app venv 12 modules/6 entry points 均来自本轮本地 wheel。六个 E3 初始化 seed 的短程合成 loss 范围 `57.50–87.02`、SD `10.24`；seed 102 最高但非唯一高波动，不能解释市场验证 seed 102 反向。证据位于 `.artifacts/world_model/routeb-seed-stream-factorial-20261022-run03/report.json` 和 `docs/audit_evidence/20261022/seed_stream_factorial_wheel_{manifest,imports}.json`。下一步做训练 loss/gradient 分解和因素交互分析；DOGE Fold/Test/Holdout 均保持封存。

## 2026-10-22 superseding continuation cursor

The latest work completes the full E2/E3 2×2×2 synthetic seed-stream design and fixes explicit-seed checkpoint forecast loading. E3 initialization main effects were +29.82 total loss, +30.37 risk auxiliary loss, and +457.77 global gradient norm; synthetic-only, with no market causal claim. Resume identity remains strict, while forecast accepts only the validated `seed_streams` metadata extension. JAX-focused tests: 81 passed; final installed-wheel app selector: 204 passed, 13 skipped because that app environment has no JAX; Ruff/MyPy/diff checks pass. Final source manifest: 568 files, identity `15b29b61584dbe45be852999d5acc45c5f988ffa24715bf507e42455272f184f`; 12 wheels/12 sdists built, all-match, 12 modules/6 entry points verified. Evidence: `.artifacts/world_model/routeb-seed-stream-factorial-20261022-run07/report.json` and `docs/audit_evidence/20261022/seed_stream_final_wheel_manifest.json`, `seed_stream_final_wheel_imports.json`. Next: decompose E3 objective-component gradient contributions, then resume the A01–A14 code-ready audit queue. Consumed DOGE folds and Test/Holdout remain sealed.

## 2026-10-22 current cursor — A10 world-model JSON reader integrity

The CLI request parser, repository base-commit marker, model bundle manifest, JAX checkpoint sidecar, and research ArtifactStore index/completion/L1 manifest now reject duplicate object keys recursively. Full app `tests/world_model`: **211 passed, 14 JAX-only skips**; the exact dedicated CPU/JAX workflow selector: **11 passed**; research artifact source and installed-wheel selectors each **2 passed**. Ruff, targeted MyPy, final wheel imports, 69-package compatibility, and source manifest checks pass. The 570-file source identity is `8238f38a9e8a66f302b312219975e77e5eede393854a5662144a395e809bfc53`. PyArrow 25.0.1 was already present in the isolated app venv and passed a Parquet roundtrip. The JAX training env intentionally excludes app-only Gymnasium, so full app module collection is not run there. Evidence: `docs/audit_evidence/20261022/world_model_duplicate_json_keys_20261022.json`. Continue the remaining local A01–A14 source audit; Hosted CI, real cashflow, independent Validation, new authorized market scope, real depth, historical effective-trial coverage, and A09 certification remain open.

### 2026-10-22 A10 orchestrator/WFO JSON continuation

All `json.loads` calls in application and WFO modules now route through recursive duplicate-key rejection, covering configs, bundle/report/receipt files, resume manifests, baselines, and Holdout reports. Combined current-source world-model + CLI + WFO suites pass **289 tests with 14 JAX-only app-environment skips**; the exact separate CPU/JAX CI selector passes 11. Source and installed-wheel CLI/WFO selectors each pass **78 tests**; Ruff/MyPy pass. The 571-file source identity verifies as `641e814dd9e8ceb2db547db19a72fc196e83819d2a0ea22cf4d9e17dd05cdce5`. Evidence: `docs/audit_evidence/20261022/orchestrator_wfo_duplicate_json_keys_20261022.json`. Continue the remaining local A01–A14 source audit; external certification and Hosted CI gates remain open.

### 2026-10-22 A10 L4 checkpoint/L0 actor continuation

SB3 checkpoint and frozen actor JSON readers now reject duplicate keys across manifest, recovery state, integrity ledger, and actor metadata. Synthetic duplicate-key regressions **2 passed**; SB3 continuous/resume parity **2 passed**; actor export/load/native fixture parity **1 passed**, including installed-wheel checks. Ruff/MyPy and 69-package app compatibility pass. The latest 573-file source identity is `4c68bdf12bf3f90ca5ce88eb818a5e352da29681735c48b4b1af0aa1640c9f67`. Evidence: `docs/audit_evidence/20261022/sb3_checkpoint_actor_duplicate_json_20261022.json`. Continue the remaining local A01–A14 audit queue.

### 2026-10-22 A10 curated data and Binance archive continuation

Curated COMPLETE markers, scoped/full readers, row payloads and Binance archive/funding receipts and rows now use recursive duplicate-key rejection. Data source selector: **36 passed**; installed rles-data + application CLI/WFO selector: **114 passed**, PYTHONPATH unset. The current-source world-model + CLI + WFO + SB3 JSON selector passes **291 tests, 14 app-environment JAX skips**; the exact separate CPU/JAX workflow selector passes **11**. Ruff/MyPy and 69-package compatibility pass. The verified 574-file manifest identity is `aa24bc738e2a7db3a9cf1bd71e555910f404c6fca6e5492a608d3ec9f2cda8a5`; evidence is `docs/audit_evidence/20261022/data_marker_archive_duplicate_json_20261022.json`. Continue the A01–A14 source audit; external research/Hosted CI gates remain open.

**2026-10-22 continuation — E3 gradient decomposition:** RouteB’s saved eight-arm synthetic factorial now has a terminal objective-gradient report. The risk auxiliary gradient cosine with the total gradient is `0.9830–0.9996` across arms; median risk-gradient norm is `317.17` versus `25.45` for minute prediction. This establishes dominance of the risk term on these short synthetic checkpoints only, not a cause of either DOGE Validation result and not a basis for changing frozen market protocols. Diagnostic script now derives and validates each checkpoint’s training seed, including the legacy seed-102 arm; 8/8 runs, Python compile, no-cache Ruff, and diff check pass. Evidence: `docs/audit_evidence/20261022/e3_gradient_component_diagnostic.json`. Next: audit the remaining locally actionable A01–A14 source items; external cashflow, hosted CI, independent Validation, fresh authorized scope, authentic depth, and A09 certification remain open.

Distribution identity after adding the diagnostic is 569 files, `07d52275149cd4a2df72e8c0ac774b05bf0696f6282a82f40978a7dfbda899a3`, verified immediately. All 12 existing wheels still match every package source file (`all_match=true`); the added/updated content is QA/docs only. See `docs/audit_evidence/20261022/e3_gradient_wheel_manifest.json`.

**E3 risk-head channel follow-up:** Training diagnostics now expose each continuous risk channel while preserving the combined loss formula. Across the same eight synthetic terminal arms, the log-realized-volatility channel had median gradient norm `315.44`; the next-largest single risk component was maximum-high-return at `8.40`. On the synthetic Train labels, `log(RV)` remains in raw logarithmic units (means about `-8.19` to `-5.79`). This identifies a scale-dominant channel in that fixture, but does not establish a defect or market cause. No objective weights changed. JAX-focused 56 tests and Ruff/MyPy/compile checks pass. Persistent report: `docs/audit_evidence/20261022/e3_gradient_component_diagnostic_channelwise.json`.

**Latest final checks:** the channelwise diagnostic and objective-preserving training-metric addition pass 56 JAX CPU tests; full app world-model suite passes 204 with 14 JAX-only skips; all 8 synthetic gradient cases pass. Maximum old/new terminal objective difference is `8.6e-6`, consistent with floating-point reduction ordering. Source identity `23c4ca9f79a1eab579d9528305f2d084f681369ea013d55e4143979871fbc94b` (569 files) verifies, and 12 rebuilt wheels match source and import from RouteB's isolated app venv (12 packages/6 entry points). Resume the next source-level A01–A14 issue; external evidence gates listed above remain open.

**A10 local CI equivalent — latest:** PR selector **721 passed / 4 skipped / 313 deselected**, full Ruff and MyPy (**217 files**) pass. Selector exposed and led to fixing the diagnostic’s duplicate QA module import; MyPy then exposed and led to fixing tuple-key typing in the seed-factorial helper. Post-fix factorial test: **2 passed**. Manifest 569 files, identity `5ad5b5b3b0c62b32b5dbc072daef786e76af0474711d9e47891ded57a4007fa0`; rebuilt package wheels all match source; RouteB venv wheel provenance 12 packages/6 entry points. Hosted CI remains an external gate. Continue audit queue; do not mark full project complete.


## 2026-10-22 superseding source-audit cursor

最近解决的本地缺口是研究账本 reconciler 的重复 JSON 键歧义：当前所有解析路径严格拒绝重复键；损坏的 R4 failure/registration 回执只要包含已知 schema（包括 JSON 转义写法）就会被发现并列入拒绝，apply 批次 fail-closed。18 项聚焦测试、Ruff、单文件 MyPy 通过；实际 ledger metadata-only dry-run 为 42/42 accepted、0 rejected、0 unresolved。证据：`docs/audit_evidence/20261022/reconcile_strict_json_20261022.json`。

RouteB CPU/JAX 环境在 `packages/world_model/environment/.venv-win-cpu`，Python 3.12.10 / JAX 0.11.2 / PyArrow 25.0.1，67 个包的 `uv pip check` 通过。该依赖已存在，无需重复安装。下一步按 A01–A14 队列继续找可独立修复的源码缺口；已消费窗口、Test/Holdout、账本边界保持不变。


## 2026-10-22 superseding cursor — A04 inventory integrity

项目级 legacy research inventory 现在拒绝重复 JSON keys、非 UTF-8 和未知 schema，避免重复 `unresolved` 字段掩盖已知历史访问。缺陷曾由合成清单复现；当前安装 wheel 下 `tests/test_research_access.py` + reconciler tests **92 passed**，重建的 12-wheel/12-sdist source comparison 与 12/6 import provenance smoke 通过。证据：`docs/audit_evidence/20261022/legacy_inventory_strict_json_20261022.json`。继续从 A01–A14 选择依赖满足的本地源码问题；不重开已消费窗口或 Test/Holdout。


**最终本地回归（2026-10-22 A04）：** workflow PR selector **722 passed / 4 skipped / 317 deselected**；app `tests/world_model` **208 passed / 14 JAX-only skipped**；隔离 Python 3.12/JAX venv 对其余 world-model 目录 **207 passed**（CLI 文件在 app suite 覆盖）。Source identity 569 files `2a27b3b332721b53bc3b5a19804d2bdbbbb37fce5b8f2bcd55393e8098b10d38`，12 wheels all-match，app venv 69 packages compatible。

## 2026-10-22 current continuation cursor — A04 query-intent input validation

Fixed two reproducible ProjectResearchRegistry API boundary defects: `start_query_intents` no longer treats a scalar string as an iterable of intent IDs, and `verify_query_intent` no longer accepts `bool` as a timestamp; scope and required identifier contracts are checked before database lookup. Final installed-wheel regression set: **165 passed**. Rebuilt 12 wheels match package source, import provenance confirms 12 modules/6 entry points in RouteB's independent `.venv-win-app`, and all 69 installed dependencies are compatible. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`. Continue A01–A14 source-level audit and fixes; external hosted CI and external evidence gates remain open. No market-data or ledger authorization changed.

## 2026-10-22 superseding cursor — A04 research registry input contracts

In addition to query-intent validation, fixed three registry API defects found in the adjacent audit: unhashable result status and non-string completion digest raised low-level `TypeError`; non-string protocol IDs generated successful empty summaries. The boundary now rejects these inputs explicitly. On final installed RouteB wheels, 168 relevant tests passed; Ruff and MyPy passed; 12/12 wheels match source and import provenance verifies 12 modules/6 entry points; 69 app dependencies are compatible. Source manifest: 569 files, identity `c6ffa2d5f1b3eca06962abe749587bd2c6217d0ef5d5f5b0f01cd1f85f3ddd6a`. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`. Continue A01–A14 local audit; Hosted CI and independent research evidence gates remain open. No market data or ledger was accessed.

## 2026-10-22 latest A04 cursor — frozen candidate-set uniqueness

The formal query intent registry itself now rejects duplicate candidate IDs; this prevents duplicate entries from becoming part of the pre-read immutable Test/Holdout query identity even outside the WFO caller. Final installed-wheel regression count is **169 passed**; 12 wheels match package source; 12 modules and 6 entry points resolve from the isolated RouteB app venv; all 69 dependencies are compatible. Ruff/MyPy pass. Source manifest: `d98f258a3ddddb2cda6fc0932b318879edf384023973e95e0f763ff26103a3b0`. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`. Continue the A01–A14 source-level audit; no market data or ledger access occurred.

## 2026-10-22 latest cursor — A05 strict trial-recipe identity

Canonical research-registry serialization now recursively rejects non-string JSON object keys. This closes a real trial-count integrity gap: Python otherwise aliases integer-key recipes to string-key JSON identities, causing distinct inputs to share a candidate digest. Direct regressions cover top-level and nested keys. Final installed-wheel selector: **171 passed**; 12 wheels source-match and import from RouteB's isolated app environment; 69 dependencies compatible; Ruff/MyPy pass. Source manifest: 569 files, `f271cda807873531d803f0869080d8f51df7d291db3060e9f99bedf2ce56212b`. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`. M3.1 historical effective trial count remains `not_estimated` due unregistered history. Continue the local source audit; no market data or ledger was accessed.

## 2026-10-22 latest cursor — A04 legacy inventory import contract

Both layers now validate the legacy access inventory: parser checks exact row shape/types and maps malformed source rows to `DataError`; registry import checks the exact key set and lowercase SHA-256 before opening a transaction. Invalid rows cannot leave partial imports. The tracked inventory passes read-only validation (67 entries; 0 unresolved). On final installed RouteB wheels, access/WFO/CLI/fresh-fold/reconciler tests: **197 passed**; all 12 wheels source-match and import from RouteB `.venv-win-app`; all 69 dependencies compatible; Ruff/MyPy pass. Source identity: `e74f7153cec266e51ab45bb92cc22cb6488875e05f9d04ce75685d6b46049feb`. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`. Continue the source audit; external Hosted CI and research evidence gates remain open. No project ledger mutation or market data access occurred.

## 2026-10-22 latest cursor — A04/A05 registry and historical inventory integrity

The project registry now enforces strict serialized trial identity, unique frozen candidate sets, stable invalid-input errors, exact legacy inventory row contracts before import, and string-only labels in paired Test component comparison. Malformed-row tests verify rollback/no partial import; malformed component regression was observed failing before the guard. Final installed-wheel suite across access, WFO, CLI, fresh-fold, and reconciler: **198 passed**. All 12 wheels source-match; 12 module/6 entry point provenance, 69 app dependencies, Ruff/MyPy pass. Tracked legacy inventory read-only validation: 67 entries/0 unresolved. Source manifest: `ba699924f5ba4386a15604d473e683cf077b10575bd5f868668d2decbf69208b`. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`. Continue remaining A01–A14 audit work; hosted CI and external study gates stay open; no market data or ledger was accessed.

## 2026-10-22 latest cursor — A04 registry operation contracts

Registry `consume`, `verify_report`, and `development_result` now fail early on malformed key/token/identity/digest inputs rather than leaking SQLite `ProgrammingError` or reporting an invalid digest as a normal completion mismatch. Seven new negative regressions pass. Final installed-wheel test set across access/WFO/CLI/fresh-fold/reconciler: **205 passed**; 12 wheels match source, 12 modules/6 entry points import from RouteB `.venv-win-app`, all 69 app dependencies compatible, Ruff/MyPy pass. A wider same-candidate recipe lock was evaluated and not applied because existing R4 attempt identity is fold/seed scoped. Source identity: `b605d4c1acb1f1d209b0c4cee89a06849f0cda4a0487a9f4810f9bc787d45ddd`. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`. Continue the remaining audit queue; no market or ledger access occurred.

## 2026-10-22 latest cursor — A05 deterministic trial attempt summaries

Registry trial summaries now order attempts by `candidate_id`, `seed`, and `attempt_key`; a reverse-insertion test confirmed the previous candidate-only ordering returned seeds in insertion order. Final installed-wheel access/WFO/CLI/fresh-fold/reconciler tests: **206 passed**; all 12 wheels match source and import from RouteB's isolated app venv, 69 dependencies compatible; Ruff/MyPy pass. Source identity: `a4bc06b9d8ce7e76f91675d01ed146cc3a4fda1294e9cb47db088d08515626f2`. Evidence: `docs/audit_evidence/20261022/query_intent_input_contract_20261022.json`. External hosted CI and historical trial coverage remain open; no market or ledger was accessed.


## 2026-10-22 latest cursor — M1.3 native report integrity

M1.3 v2 now rejects absent, boolean, fractional, string, or negative `risk_hard_interventions` values instead of silently coercing them, and enforces hard-intervention ≤ decision, submit ≤ decision, and native-order ≤ submit count bounds; it binds the frozen starting balance to native initial equity. It also verifies the reported `max_abs_residual` against the maximum event-level residual before summarizing. Report-mode paths are constrained to one component; bundle, actor artifact, and SHA receipt paths are bundle-local and resolved report/artifact targets must remain in their registered directories. The focused synthetic analyzer suite passes **26 tests, 1 skipped** (the host denies native symlink creation; a platform-independent resolved-path escape regression passes); targeted Ruff, MyPy (1 file), Python compile, and `git diff --check` pass. Source manifest: 569 files, identity `c5ccc5b8a551669888a5d3544cfe1a5436904b1de29bc5df7cda611809812f8f`. Evidence: `docs/audit_evidence/20261022/m13_stage_report_integrity_20261022.json`. Full M1.3 remains partial because no identity-compatible E1 bundle and frozen E1/R1/E2 report set is available; no replay or market/ledger access was performed. Continue A01–A14 audit at the next locally actionable source gap; hosted CI and external evidence gates remain open.


## 2026-10-22 latest cursor — M0.1 exact sourced-cashflow inputs

The Binance USDⓈ-M transfer importer now requires `income` as the exact decimal string specified by the official response schema, preventing JSON floats from being accepted after precision loss. Native sourced cashflow facts reject float/bool amounts and source IDs with surrounding whitespace. Installed RouteB wheel regression across cashflow, metrics, native liquidation, and historical factory tests: **200 passed, 18 warnings**. Ruff, MyPy (2 modules), compile, wheel provenance (12 packages/6 entry points), and dependency check (69 compatible) pass. Rebuilt 12 wheels/12 sdists, all source matched; source identity: 569 files, `727f2d7cd6da859aa99dd499bab631df9e55cb27e119eebf05350916a578a199`. Evidence: `docs/audit_evidence/20261022/m0_1_cashflow_decimal_contract_20261022.json`. No ledger, market, account, Validation, Test, or Holdout payload was read. Real cashflow provenance and completeness remain open; continue the next locally actionable A01–A14 issue.


## 2026-10-22 superseding cursor — A04 duplicate legacy access keys

The legacy inventory parser and `ProjectResearchRegistry.import_inventory` now reject duplicate access keys before import; the registry also validates key type/non-emptiness before uniqueness checks. The bug was reproduced against the prior installed wheels. Final installed-wheel access/WFO/CLI/fresh-fold/reconciler selector: **209 passed**. Ruff and MyPy pass; 12 wheels/12 sdists built, all package source comparisons match, updated orchestrator/research wheels installed into RouteB's isolated app venv, provenance smoke verifies 12 packages/6 entry points, and 69 dependencies are compatible. Source manifest: 569 files, identity `b4b71e568a20e7d715e2c02693bb5755f656231d9d6d0efbe3e5b3b40d0268d4`, verified. Evidence: `docs/audit_evidence/20261022/legacy_inventory_duplicate_access_keys_20261022.json`. Only synthetic inventory and temporary SQLite were used; no project ledger or market/evaluation payload was accessed. Continue A01–A14 source audit; external research evidence and Hosted CI remain open.


## 2026-10-22 superseding cursor — A04 canonical scope namespaces

Project interval authorization now rejects noncanonical identifiers that could alias the same market/family namespace: identifiers cannot contain surrounding whitespace, market symbols must be uppercase, and family names lowercase. The strict legacy inventory parser and direct import API enforce the same rule. The previous installed wheel reproduced six failures; final installed-wheel registry/WFO/CLI/fresh-fold/reconciler selector: **215 passed**. Ruff/MyPy pass; 12 wheels/12 sdists built and source-match, updated wheels installed in RouteB's isolated venv, provenance verifies 12 packages/6 entry points and 69 dependencies are compatible. Source identity: 569 files, `1df7998039047073208d001fd97fa8e10f04ec9dfe1ac35e5677b546634a45b6`, verified. Evidence: `docs/audit_evidence/20261022/registry_scope_namespace_canonicalization_20261022.json`. Synthetic identifiers/temporary SQLite only; no project ledger or market/evaluation payload read. Continue A01–A14 code audit; Hosted CI and external research-evidence gates remain open.


## 2026-10-22 superseding cursor — A04/A05 canonical research identifiers

Frozen Test candidate IDs and trial registry identities now reject blank or padded aliases before counting, lookup, or SQLite insertion. Five new regressions failed against the previous installed wheel. The final installed-wheel access/WFO/CLI/fresh-fold/reconciler selector passes **220 tests**; Ruff/MyPy pass; 12 wheels/12 sdists build and match source, the rebuilt research wheel is installed in RouteB's isolated app venv, provenance verifies 12 packages/6 entry points, and 69 dependencies are compatible. Source manifest: 569 files, `978cbf8723d42aa28c0496f869d4e8119aa0237bdd51229d8ca4de81b6893d16`, verified. Evidence: `docs/audit_evidence/20261022/research_identifier_contracts_20261022.json`. Synthetic IDs and temporary SQLite only; no project ledger or evaluation payload was accessed. Continue A01–A14 source audit; Hosted CI and external research evidence remain open.

## 2026-10-22 cashflow text-boundary continuation

Binance USDⓈ-M cashflow import now converts an unencodable Python text payload (for example, an unpaired surrogate) into the stable `MetricsError` contract instead of leaking `UnicodeEncodeError`. Focused `tests/test_binance_cashflows.py`: **26 passed**; Ruff on the changed source and test files passes. Pytest cache was disabled and its temporary directory redirected to the authorized workspace after the default `D:\Program Files` and sandbox temp locations denied writes. Source manifest refreshed: 569 files, identity `5518967361639c18b67c156deb414d9914767a0cb15eaac358170b03e1811325`. This hardens failure handling only; it does not authenticate Binance provenance or establish account-history completeness. Continue the A01–A14 source audit; external evidence gates remain open.

## 2026-10-22 cashflow decimal-lexeme follow-up

The USDⓈ-M transfer importer now validates an ordinary ASCII decimal lexeme before calling `Decimal`, rejecting parser extensions such as underscore separators and Unicode decimal digits that are not valid exchange-format amounts. `tests/test_binance_cashflows.py`: **28 passed** against both source and the freshly installed RouteB research wheel; installed module SHA matches source. Ruff and MyPy passed. All 12 workspace wheels and 12 sdists rebuilt to `C:\Users\QX\Documents\ChatGPT\RouteB\routeb-wheels-20261022-cashflow-lexical`; wheel manifest reports **12/12 source matches**. Binance documents `income` as a string and shows a signed decimal amount in its [USDⓈ-M Get Income History response](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account); this input check does not authenticate response origin or completeness. Continue the A01–A14 audit; real cashflow proof and other external gates remain open.

## 2026-10-22 cashflow paged Unicode serialization follow-up

Paged Binance import now serializes intermediate merged rows with ASCII escaping. This preserves the exact original page digests while avoiding a raw `UnicodeEncodeError` when an ignored metadata field contains a JSON-escaped unpaired surrogate. The new regression verifies successful transfer normalization and unchanged exact page SHA binding. Installed RouteB wheel `tests/test_binance_cashflows.py`: **29 passed**; the imported module SHA matches source; Ruff, MyPy, and `uv pip check` (69 compatible packages) pass. All 12 wheels/12 sdists rebuilt; wheel manifest reports 12/12 source matches. Source manifest verifies 569 files, identity `ffe2db1ce9e815563b00149ced3c360deda5c6666617a942cca7914e1035179b`. Artifacts: `C:\Users\QX\Documents\ChatGPT\RouteB\routeb-wheels-20261022-cashflow-unicode` and `cashflow-unicode-wheel-manifest.json`. This does not weaken amount validation or authenticate source account/history completeness; continue the remaining A01–A14 audit and keep external gates open.

## 2026-10-22 A13 episode-window diagnostics integrity

The SB3 Train coverage callback now requires the exact episode-window schema, positive ordered inclusive timestamp bounds, a positive decision-step count, and containment of every observed decision timestamp. It rejects a window whose bounds change under the same `(worker, episode_index, start_offset)` identity instead of silently overwriting prior facts. The `TrainingDiagnostics` comment was also corrected: volatility bins use fixed predeclared absolute thresholds, not empirical per-call quartiles. Full `tests/test_native_training_episode.py`: **30 passed, 16 warnings** against both source and the installed RouteB wheel. Ruff and MyPy pass; all 12 wheels/12 sdists rebuilt, wheel manifest 12/12 source-matched, installed provider file SHA matches source, and all 69 app dependencies are compatible. Source manifest verifies 569 files, identity `3947a3de975b4499c23a767dd6cf7bd6b8d910b0f8140b81f5c94a24b015db65`. Wheelhouse and manifest: `C:\Users\QX\Documents\ChatGPT\RouteB\routeb-wheels-20261022-m33-window` and `m33-window-wheel-manifest.json`. No project ledger, market, Validation, Test, or Holdout data was accessed.

## 2026-10-22 A02 constraint-priority explicit native facts

The public `ConstraintPriorityFitnessSpec.evaluate_episode()` now rejects missing/non-Boolean insolvency, maintenance-margin, or native-liquidation facts, and requires the explicit `margin_facts_known=True` completeness marker. This prevents omitted facts or placeholder `False` values from being treated as “no violation.” The old installed wheel reproduced the defect by returning `feasible=True` when all facts were omitted; direct score API callers bypassed the dispatcher's existing completeness check. Source and installed-wheel fitness/worker selector: **49 passed, 8 warnings**; the installed native ES worker insolvency path: **1 passed**. Ruff and MyPy pass. All 12 wheels/12 sdists were rebuilt and match source; installed fitness SHA matches source, and all 69 app dependencies are compatible. Source manifest verifies 569 files, identity `ad2edeb98ce51a5396194e77405dc73da3ca3c2b65c6a8736984a7b1c07f8e5a`. Wheelhouse and manifest: `C:\Users\QX\Documents\ChatGPT\RouteB\routeb-wheels-20261022-a02-known-marker` and `a02-known-marker-wheel-manifest.json`. The dedicated CPU/JAX venv already imports PyArrow 25.0.1, so no dependency change was needed. No market, account, or formal evaluation payload was accessed.


## 2026-10-22 A11 M1.3 strict JSON reader

Both the current M1.3 stage-contribution analyzer and its still-callable revision-2 compatibility entry now reject duplicate JSON object keys recursively and normalize invalid UTF-8 into a stable `ValueError` boundary. The compatibility entry also rejects non-finite/bool numeric values, inconsistent reconciliation summaries, and execution/risk counts beyond decision counts. They cover bundle, actor, preregistration, E1/R1/E2 and older R1/E2 reports. Combined regression suites pass **37 tests with 1 symlink-permission skip**; Ruff passes, and targeted MyPy passes with only the installed untyped-package diagnostic disabled. The 569-file source manifest verifies as `96edcde1206a50e362cd77cce732cc51cc7ad95306995b05f661c7c3d177dda7`. Evidence: `docs/audit_evidence/20261022/m13_stage_analyzer_strict_json_20261022.json`. Only synthetic temporary fixtures were read; M1.3 remains partial pending compatible native stage artifacts. Continue the remaining local audit queue.

## 2026-10-22 latest cursor — A05 registered R4 control JSON integrity

Both registered R4 hierarchy and E2/E3 comparison runners, all five R4 Fold freezers, and the ADA/DOGE/SOL archive downloaders now reject duplicate JSON object keys in their frozen controls and receipts. Malformed UTF-8, JSON syntax and IO failures in registered runners normalize to `ValueError`. Synthetic nested-duplicate and invalid-byte regression included. Focused R4 tests: **11 passed**; Ruff, Python compilation, and repository MyPy (`packages qa`, 224 source files) pass with installed untyped-wheel diagnostics disabled. Source manifest verifies 579 files, identity `a4aa67a4ad0216f0333488b2ebd0bfee64100bd37cac93cab016d1a5853ad28c`. Evidence: `docs/audit_evidence/20261022/r4_registered_controls_strict_json_20261022.json`. Synthetic temporary JSON only; archive tools were not run; no market or ledger payload was read. Continue local A01–A14 audit; external CI and independent research gates remain open.

## 2026-10-22 latest cursor — registered world-model QA JSON boundaries

The R3 registered ablation and multi-seed ensemble now use the strict world-model JSON reader for forecast request, E2 report, and source-preparation identity. CLI worker smoke, E3 gradient diagnostic, and seed-stream factorial use it for their generated reports and metadata. R4 registered/freezer/downloader coverage is recorded in `r4_registered_controls_strict_json_20261022.json`. Combined registered-report, CLI, randomness, and R4 tests: **39 passed, 2 JAX-only skipped**; changed QA scripts compile, Ruff passes, MyPy `packages qa` reports no issues across 224 source files. Current 580-file source manifest: `13e896181c83d6d6011a2270a49fb6dadaf67811b8772c30cb039b42726d22c9`. Evidence: `docs/audit_evidence/20261022/world_model_qa_json_integrity_20261022.json`. Remaining direct parser calls in this QA subtree are duplicate-protected document parsing or scalar-string extraction. No market, ledger, or evaluation data was read. Continue the A01–A14 source audit; Hosted CI and independent research gates remain open.

## 2026-10-22 legacy research inventory input integrity

The read-only A04/A05 historical access and trial inventory tools now recursively reject duplicate JSON keys, malformed UTF-8/JSON, IO failures, and non-object roots in inventory manifests, configs, and report summaries. Synthetic regression covers duplicate fields, invalid UTF-8, and wrong root type through both readers. Focused tests: **6 passed**; Ruff and `mypy packages qa` (**224 files**) pass. The 581-file source manifest verifies as `e7004296bba56cca8fa1f69e6899e7c47203c8709c7830e5f28b89e88865dcd0`. Evidence: `docs/audit_evidence/20261022/legacy_inventory_json_integrity_20261022.json`. No actual historical ledger or market data was read. Continue the remaining A01–A14 local audit.

## 2026-10-22 latest cursor — A11 M1.3 legacy replay-generator reader

The still-callable M1.3 replay generator now rejects duplicate JSON keys and invalid/non-object bundle, fixture manifest, completion receipt, and frozen preregistration inputs. Combined runner/analyzer regression: **40 passed, 1 symlink-capability skip**; Ruff and MyPy `packages qa` (224 files) pass. Current 582-file source manifest verifies as `a384d87ec586b2596fc3664e1dbf035229e1489a30f8e252e83efcee39f99521`. Evidence: `docs/audit_evidence/20261022/m13_runner_duplicate_json_20261022.json`. Synthetic JSON only; no bundle or market payload was opened. The runner source hash changed, so any future authorized replay requires a newly frozen preregistration bound to the current runner. Continue A01–A14 local audit; external stage-artifact, Hosted CI, and research certification gates remain open.

## 2026-10-22 latest cursor — A13/A14 M3 evidence analyzer parsing

Stage58/59/60 coverage analyzers and the M3.3 action-value analyzer now reject duplicate JSON keys, malformed UTF-8/JSON/IO, and non-object roots across preregistration, config, learning reports, completion receipts, and action-value evidence. Temporary-file regressions exercise every reader. Combined M3/M1.3/inventory regression selection: **58 passed, 1 symlink-capability skip**; Ruff, py_compile, and MyPy `packages qa` (224 source files) pass. Current 583-file source manifest verifies as `4606afd3a67f9f48030bde02c57f448de7b71947762a19ab10b1aed17f659a4e`. Evidence: `docs/audit_evidence/20261022/m3_evidence_analyzers_strict_json_20261022.json`. Only synthetic JSON fixtures were read. Continue the remaining A01–A14 local audit; external evidence and Hosted CI gates remain open.

## 2026-10-22 latest cursor — Train-only preregistration and source-marker readers

The Stage59/60 and M3.3 preregistration tools now use recursive duplicate-key JSON parsing; Stage69's Train-only episode-reset runner applies it to preregistration, config, and source completion markers before registering or reading data. Regression across these readers plus prior M3/M1.3/inventory checks: **70 passed, 1 symlink-capability skip**. Ruff, compilation, and MyPy `packages qa` (224 files) pass. Current 584-file manifest verifies as `f865bb5383a928035253886dc4894fbd20a02c40f64b8ae594e43f9f640c4250`. Evidence: `docs/audit_evidence/20261022/stage_prereg_and_scope_reader_integrity_20261022.json`. Synthetic JSON only; no prior frozen data was opened. Previous frozen source hashes are intentionally no longer current; consumed Train windows must not be replayed. Continue the A01–A14 local source audit.


## 2026-10-22 A10/M0.2 manifest-input hardening

The source-manifest verifier and wheel-manifest reader now reject duplicate JSON keys and invalid UTF-8. The wheel checker rejects duplicate ZIP paths, repeated METADATA Name/Version headers, corrupt archives, stale project versions, and symlink wheel/package/source roots, and source traversal excludes symlinks before file stat. Focused tests: **12 passed** (one expected `zipfile` warning from the duplicate-member fixture); Ruff/MyPy passed. Mocked path-order tests verify symlink rejection before file stat and wheelhouse-directory rejection; native symlink/junction integration remains untested. The regenerated 569-file source identity `43df4edfc9a9c4a55d83b7d906167a3a033e4b145eee3d4eff71706112de9c11` verifies, and all 12 existing wheels match package source. Evidence: `docs/audit_evidence/20261022/a10_manifest_input_hardening_20261022.json`. Hosted clean Linux/Windows CI remains outstanding; continue the local audit queue.

## 2026-10-22 latest cursor — A04/M0.3 WFO split-scoped market reads

Removed WFO's startup decode from `data_start_ns` through `holdout_end`. The default Train and final-retrain paths now use their registered Train windows; Validation and stress Validation read only their fold warmup through Validation end; Test and its baselines read only that Test fold after its access token is consumed; final Holdout reads its own warmup-to-end scope only after one-shot ledger opening and token consumption. A composite bundle/report dataset identity binds the market snapshot and rules-file SHA, while the Holdout ledger retains the separate market-only one-shot identity. WFO + application CLI + full historical factory source selectors: **163 passed, 18 warnings**. Ruff and targeted MyPy pass. Evidence: `docs/audit_evidence/20261022/wfo_scoped_payload_boundaries_20261022.json`. Only synthetic/temporary test fixtures were used; no project registry or real market/evaluation payload was opened. M0.3 remains partial because pre-registration historical access cannot be reconstructed; continue the local A01–A14 audit.


## 2026-10-22 A10 RouteB worker JSON integrity

世界模型 worker 配置、预测请求、Train NPZ provenance sidecar 与子进程 JSON 响应现在递归拒绝重复键；worker 客户端把解析失败统一包装为 `WorkerFailure` 并保留冲突字段信息。新增配置、请求、sidecar、worker 响应回归。源码定向测试 **59 passed、5 个 app venv JAX 条件跳过**；当前源码 world-model + CLI + WFO + SB3 JSON 选择器 **294 passed、14 个 JAX 条件跳过**。新 world-model wheel 在 `PYTHONPATH` 未设置时安装版定向测试 **59 passed、5 个 JAX 条件跳过**；Ruff/MyPy 通过，RouteB app venv 69 个包兼容，三个安装模块 SHA 与源码吻合。574-file source manifest identity `c8195fc140a2e91e61fbc293489b7719bdd781fb1a5ad4148e0d333e6003a46e` verified。证据：`docs/audit_evidence/20261022/world_model_worker_json_integrity_20261022.json`。仅临时合成数据与 mock worker 响应；未读取正式市场、账户、Validation、Test 或 Holdout payload。继续 A01–A14 本地审计队列，外部 hosted CI 与真实研究证据门槛仍待完成。


## 2026-10-22 A10 R1 resume evidence JSON integrity

R1 断点恢复中的 `R1-transfer.json` 与 `R1-learning.json` 现在递归拒绝重复 JSON 字段；新增迁移证据与带有效 SHA 标记的训练诊断合成回归。安装版 R1 + application CLI + WFO 选择器 **80 passed**（未设置 PYTHONPATH）；当前源码 world-model + CLI + WFO + SB3 JSON + R1 选择器 **296 passed、14 个 app venv JAX 专项跳过**。Ruff、MyPy 通过，RouteB app venv 的 69 个依赖兼容；新 orchestrator wheel 安装模块与源码 hash 一致。575-file source manifest identity `de47121ae6661f454fb68b527935c5285688aa3322059dff2cf0312cae68f650` verified。证据：`docs/audit_evidence/20261022/r1_resume_evidence_duplicate_json_20261022.json`。仅使用合成临时 R1 证据，没有读取登记市场或正式评价 payload；继续本地 A01–A14 审计，外部 hosted CI 和真实研究证据门槛仍开放。


## 2026-10-22 A10 EvoTorch checkpoint JSON integrity

E1/E2 的 EvoTorch checkpoint manifest 与原生优化器状态 NPZ 内嵌元数据现在递归拒绝重复 JSON 字段；checkpoint manifest 错误归一为完整性失败，状态恢复前即拒绝歧义元数据。源码和安装 wheel 的 duplicate-key + optimizer resume selector 均 **8 passed**，包含 PGPE/CMA-ES 连续与恢复候选流一致性；Ruff/MyPy 通过，RouteB app venv 69 个依赖兼容，安装 checkpoint/provider 文件 hash 与源码一致。577-file source manifest identity `0b655feeb2e80887bef956bd91ecbd747abd26e8429308c9b2f6dd44b8d624da` verified。证据：`docs/audit_evidence/20261022/es_checkpoint_duplicate_json_20261022.json`。仅合成 checkpoint 与状态文件；未运行真实优化或读取市场/正式评价数据。继续 A01–A14 本地审计；外部 Hosted CI 与真实研究验证门槛仍未关闭。


## 2026-10-22 A10 historical execution rules JSON integrity

Historical `rules.json` manifests now use recursive duplicate-key rejection and map malformed/ambiguous roots to the stable `DataError` boundary before execution rules are consumed. Full `tests/test_historical_factory.py` passes **88 tests** both against source and with `PYTHONPATH` unset against the newly installed gym-bridge wheel; Ruff/MyPy pass, all 69 RouteB app dependencies are compatible, and installed loader SHA matches source. The 577-file manifest `b1a1bd89ba8125da22ecbe573755b743f494e92c12d9892de5b43e847e2b9115` verifies. Evidence: `docs/audit_evidence/20261022/historical_rules_duplicate_json_20261022.json`. Only synthetic historical rules and fixtures were used. Continue A01–A14 local source audit; external Hosted CI and real-data evidence gates remain open.


## 2026-10-22 A10 compute lease/recovery JSON integrity

Compute runtime 持久化的 lease owner 锁文件、Windows recovery enrollment/evidence 与 SQLite recovery receipts 现在递归拒绝重复 JSON 键；损坏状态会转为 `ComputeError` 并保持资源占用，不会按歧义身份释放或恢复。device + resource-recovery 源码与安装 wheel 回归均 **30 passed**；Ruff/MyPy 通过，69 个 app 包兼容，resources/recovery 安装 SHA 与源码吻合。578-file source manifest identity `d2a8e8d5db05708206eec26cc3d35eb0a8af216181e42dd140086cbddd997cb2` verified。证据：`docs/audit_evidence/20261022/compute_lease_duplicate_json_20261022.json`。没有启动进程树或执行真实 GPU 回收；只测试合成锁元数据和临时 SQLite。继续 A01–A14 本地源码审计；外部 CI/研究证据门槛尚未关闭。


## 2026-10-22 A10 research registry persisted identity JSON integrity

Research registry 的 persisted access identity、query intent、development receipt 与 trial recipe/event 现在递归拒绝重复 JSON 字段。修正了一个授权失败开放路径：历史 formal identity 解析错误此前会被跳过；现在损坏身份抛出 `ResearchLockError`，阻止 frozen query ID 复用。`tests/test_research_access.py` 源码回归 **113 passed**，Ruff/MyPy 通过，新增源清单已刷新并验证。wheel 暂未构建：本机及 RouteB 两个 venv 没有 Hatchling，uv 隔离构建从 PyPI 获取构建后端时遇到 TLS 握手失败；本轮没有安装或替换 app wheel。证据：`docs/audit_evidence/20261022/research_registry_duplicate_json_20261022.json`。只用临时合成 SQLite，无真实市场或评价账本访问；本地 A01–A14 审计继续，外部 CI/研究数据证据仍开放。


### A10 research registry wheel validation superseding build note — 2026-10-22

The initial Hatchling fetch failed because PyPI TLS was unavailable, but RouteB's existing local UV build cache contained the backend; the research wheel was subsequently built offline and installed only into `.venv-win-app`. The installed-wheel registry/WFO/application CLI/fresh-fold/reconciler selector passes **225 tests** with `PYTHONPATH` unset. The installed registry module SHA matches RouteB source, all 69 app dependencies are compatible, and the 579-file source manifest `57a2e2bb4ba046e00adf204fe4d8e9c25325b3f3cbd86c27fcb1463b3ac14065` verifies. Updated evidence: `docs/audit_evidence/20261022/research_registry_duplicate_json_20261022.json`.

## 2026-10-22 consolidated parser regression closeout

One consolidated selector spanning all strict JSON readers changed in this audit, the M1.3 analyzers, R3/R4 report readers, CLI, and randomness contracts passes **109 tests, 3 skipped** (2 JAX unavailable in app venv; 1 host symlink capability). Current 584-file manifest verifies as `f865bb5383a928035253886dc4894fbd20a02c40f64b8ae594e43f9f640c4250`; all 9 related evidence JSON records parse successfully. This validates input integrity only; no market, ledger, or frozen study payload was read. Continue the A01–A14 audit.

## 2026-10-22 latest cursor — M3.3 Train baseline input chain

M3.3 baseline preparation and Train-only comparison now use strict duplicate-key JSON parsing for the prior run record, frozen plan, source bundle, and scenario-rule manifest before interpreting paths or registering windows. Targeted strict-reader selection: **39 passed**; Ruff, py_compile, and `mypy packages qa` (224 source files) pass. Current 585-file source manifest verifies as `5c117f9ed160c76c0ab9f39004b75c3fe7461739e81d03fd77ef66ab6c2cf9b2`. Evidence: `docs/audit_evidence/20261022/m33_baseline_source_json_integrity_20261022.json`. Only synthetic JSON was read; no baseline run, market, or ledger was opened. Continue the local A01–A14 audit.

## 2026-10-22 latest cursor — M2 frozen comparison plans

The Train-inner episode-horizon and Bar Base/Stress runners now strictly parse their frozen plan, source plan, and policy bundle before interpreting window or cost controls. Synthetic strict-reader selector: **36 passed**; Ruff, py_compile, and MyPy `packages qa` (224 source files) pass. Current 586-file manifest verifies as `849c557ff6bf8d02417b83a21c25db039d5a0ed012bcbe9d4be76c6d28ea7bc6`. Evidence: `docs/audit_evidence/20261022/m2_frozen_diagnostic_plan_json_integrity_20261022.json`. No Train replay or market data was accessed. Continue local A01–A14 audit.

## 2026-10-22 latest cursor — A07 profile input and report readers

Native-history and Episode Reset profiling now reject duplicate keys and malformed/non-object JSON in Train plans, bundles, profile case outputs, failure receipts, and summaries. A07 parser + project-isolation + M2/M3 adjacent selector: **28 passed**; Ruff, py_compile, and MyPy `packages qa` (224 files) pass. Current 587-file source manifest verifies as `a46f18270df200d56aea4cde7d98db9cf9a15017630279f531b5d2a8d0867672`. Evidence: `docs/audit_evidence/20261022/a07_profile_input_json_integrity_20261022.json`. Synthetic JSON only; no archived Train data or profile was rerun. Continue local A01–A14 audit.

## 2026-10-22 latest cursor — A02 fitness comparison evidence

The matched-seed A02 objective comparison now strictly parses the fixture manifest, baseline config, bundle/run identity, and embedded candidate-feasibility/spec JSON. Targeted selector: **27 passed** across strict input, scorer, and checkpoint regressions; Ruff, py_compile, and MyPy `packages qa` (224 files) pass. Current 588-file manifest verifies as `6cba8db77b7871171119de6256d9b64b6699c7c86a05264fa2c6f2ee67b34285`. Evidence: `docs/audit_evidence/20261022/a02_objective_comparison_json_integrity_20261022.json`. Synthetic JSON only; no A02 training or data replay was run. Continue local A01–A14 audit.

## 2026-10-22 latest cursor — A08 long-window equity plan

The long-window native-equity reconciler now strictly parses its frozen reconciliation plan, underlying Train plan, and bundle before source validation or development registration. Synthetic A08 + A07 isolation + M2/M3/A02 selector: **31 passed**; Ruff, py_compile, and MyPy `packages qa` (224 files) pass. Current 589-file manifest verifies as `8500966ec203456cb3e620195be2046730db2fe426192f6e6fa4362575eb30c4`. Evidence: `docs/audit_evidence/20261022/a08_long_equity_plan_json_integrity_20261022.json`. The historical 90-day reconcile was not rerun; continue A01–A14 local audit.

## 2026-10-22 consolidated A01–A14 local parser audit

Final combined selector covering all newly hardened R4/R3/M3.3/M2/A07/A08/A02 readers, M1.3 analyzers/runner, legacy research inventories, CLI, randomness and RouteB project isolation: **146 passed, 3 skipped** (2 JAX unavailable in app venv; 1 host symlink capability). Ruff passed across all touched RouteB QA/world-model areas. The 589-file source manifest verifies as `8500966ec203456cb3e620195be2046730db2fe426192f6e6fa4362575eb30c4`; all 11 linked evidence records parse. This closes only the local JSON-boundary findings handled in this batch. A01–A14 remain open at external gates: cashflow provenance/completeness, clean Hosted CI, independent Validation/new Train authorization, compatible native stage artifacts, historical liquidity, and historical trial coverage. Do not replay consumed windows or access Test/Holdout.

## 2026-10-22 latest cursor — Discount and legacy diagnostic input integrity

The paired discount half-life runner/evaluator and low-capacity/zero-target plan generators now recursively reject duplicate JSON fields, malformed UTF-8/JSON/IO, and non-object persisted inputs. The shared legacy reader regression covers six entry points. Combined A01–A14 JSON integrity, isolation, R3/R4/CLI/randomness selector: **112 passed, 2 skipped** because JAX is absent from .venv-win-app. Ruff, py_compile, and MyPy packages qa (224 source files) pass. The verified source manifest has 590 files and identity 80790a907b55e068def6756ac9af7d5fef358bb1645960ccfabf98b993977264. Evidence: docs/audit_evidence/20261022/discount_and_legacy_diagnostic_json_integrity_20261022.json. Synthetic JSON only; no market, ledger, frozen study, or Train payload was opened. Continue local A01–A14 audit; independent evidence, cashflow authentication, and Hosted CI gates remain open.

Full local verification update: the recorded project PR selector collected 1,241 tests and completed with **782 passed, 4 skipped, 455 deselected, 46 warnings** in 273.92 seconds. The skips are one unavailable CUDA learner test and three JAX-only cases in the application venv; the RouteB dedicated JAX environment remains separate. Project-wide Ruff passed; whole-workspace MyPy passed 224 source files. This local result does not close Hosted CI or data-backed research gates.

M0.2 wheel provenance follow-up: the installed-wheel smoke now rejects duplicate or malformed direct_url.json metadata before accepting a package as an archived wheel. Its synthetic reader and wheel-manifest tests pass (17 passed); the actual isolated RouteB app venv smoke confirms 12/12 distribution modules and 6/6 entry points import from .venv-win-app site-packages. Ruff, py_compile and MyPy packages qa (224 files) pass. Current verified source manifest: 590 files, 60e1cf8cd3505790e95196ee036314d38909171db3cac755aaf7928b98625740. Evidence: docs/audit_evidence/20261022/wheel_import_provenance_strict_json_20261022.json. Hosted CI remains outstanding.

Distribution artifact verification update: the most recent RouteB wheelhouse contains 12 wheels and the local wheel manifest reports all_match=True against current package source. This verifies local package content and import origin; the Hosted Linux/Windows jobs still have not run on this uncommitted source state.

A04/A05 legacy inventory follow-up: removed the access inventory's fallback read of holdout-report.json account rows. When the Holdout end is unknown, the inventory now quarantines the interval through MAX_NS without opening the report. Both legacy inventory tools reject symlinked or escaping input paths; the trial inventory accepts only its frozen roots and database paths within them. Regression across inventory, research registry, fresh-fold, and ledger reconciliation: 160 passed; Ruff, py_compile, and MyPy packages qa (224 source files) pass. Source manifest verifies 590 files at 6639d1a434059f5a4c8a0b87bada949af77167f9364eb3e87e5e469e1e117b50. Evidence: docs/audit_evidence/20261022/legacy_inventory_path_and_holdout_isolation_20261022.json. Synthetic temporary JSON/SQLite only; no project ledger or market/account/Test/Holdout data was opened. Historical trial completeness remains unresolved.

Legacy inventory state recheck: the read-only RouteB access inventory now reports 0 databases, 0 entries, and 3 unresolved conditions (both frozen .tmp roots missing; manifest discovery exited 2). The trial inventory fails closed with a clear missing-root ValueError. It did not read or modify a ledger or load market/account/formal-split payloads. This confirms legacy access and trial completeness cannot be reconstructed from the current checkout; the frozen snapshot is metadata only. The inventory code now blocks empty-result imports while this source evidence is unavailable. Evidence: docs/audit_evidence/20261022/legacy_inventory_path_and_holdout_isolation_20261022.json.

A04/A05 missing-root fail-closed follow-up: historical access inventory now records missing frozen scan roots and discovery errors as unresolved, so an empty scan cannot be imported as proof of zero prior access. The trial inventory emits a clear missing-root error. The current read-only RouteB probe returned 0 databases, 0 entries, 3 unresolved (two missing .tmp roots plus rg discovery exit code 2); no project DB was opened. Combined inventory/registry/fresh-fold/reconciler tests: 162 passed; Ruff, compilation and MyPy packages qa (224 files) pass. Source manifest verifies 590 files at a85efab192b2c5c22343485ac6696bdedcc40247cb63d7f0c8d428b8196327bd. Evidence: docs/audit_evidence/20261022/legacy_inventory_path_and_holdout_isolation_20261022.json. Historical access/trial completeness remains unresolved because source roots are absent.

## 2026-10-22 A09/M4.1 linear perpetual instrument metadata

The native replay path no longer hard-codes ETH/USDT, ETH order quantity units, or planner multiplier 1. Instrument rules now carry canonical `base_currency`, `quote_currency`, `quantity_unit`, and positive `contract_multiplier`; the linear contract requires `quantity_unit == base_currency`. The Nautilus `CryptoPerpetual`, quote/settlement currency, native order submitter, and target planner consume that same frozen rule. The existing ETH fixture remains multiplier 1; a synthetic BTC/USDT case with multiplier 2 verifies quantity scaling and BTC order units. Source and installed-wheel selectors each pass **201 tests**; all 12 wheels match the 590-file source identity `2ffb2892cbad117fb0540de300d6aead65364e9ca7c59e4b1f08adaf6ef8683a`; RouteB app venv has 69 compatible packages and wheel smoke sees 12 package roots / 6 entry points. Ruff and MyPy (224 files) pass. Evidence: `docs/audit_evidence/20261022/a09_instrument_metadata_20261022.json`. This is synthetic engineering evidence only: M4.1 remains partial until sourced BTC/ETH/SOL historical rules and cross-asset Funding/equity-unit checks are available; no market, account, ledger, Validation, Test, or Holdout payload was accessed.

**A09 consistency follow-up:** the native adapter now rejects a contract identifier whose symbol conflicts with the frozen base/quote currencies. The full Nautilus adapter module plus the BTC multiplier-2 factory test pass **34 source tests** and **34 installed-wheel tests** after this guard. The rebuilt 12-wheel manifest matches current source identity `b6166319348c179cfe2c39a7081e54be6a9f73074987efc602f62b6691fbfa4b`; evidence above supersedes the earlier source identity.

**A09 feature-unit follow-up — 2026-10-22:** historical feature specs now require explicit quantity/quote units; BTC volume and positions are tagged BTC while equity/cash/PnL are tagged USDT. WFO loads frozen instrument rules before building candidate feature specs, so candidate feature hashes include those units. Source and installed-wheel historical factory + WFO + CLI selectors each pass **167 tests**. The rebuilt 12-wheel manifest matches source identity 21cd85bbaf214e86afaa5f398e78a00f8adecc25b889b2e5448af5b31a31507d; evidence updated in docs/audit_evidence/20261022/a09_instrument_metadata_20261022.json. M4.1 remains partial pending authenticated cross-asset rule and Funding/equity reconciliation.


## 2026-10-22 A10 L6 release attestation JSON integrity

L6 release gate reports and the trusted-producer registry now reject duplicate keys recursively, NaN/Infinity constants, and non-object gate-report roots. Synthetic malformed-input and valid-round-trip regression: **5 passed**; Ruff passes and MyPy reports success for the changed attestation module. Source manifest verifies 591 files at `3fcf7d0605bb6b0f14fc4cc544e4f91eedd9acf7878bae02e80c1d309c1d8fee`. Evidence: `docs/audit_evidence/20261022/l6_release_attestation_json_integrity_20261022.json`. Only synthetic malformed JSON was used; no gate was approved and no market or evaluation payload was read. Continue the local A01–A14 audit; Hosted Linux/Windows CI and external evidence remain open.


## 2026-10-22 A10 RouteB package JSON non-finite constants

Seven package-level duplicate-key JSON readers now also reject non-standard `NaN`, `Infinity`, and `-Infinity` values by default; the data helper preserves its explicit override API. Source and installed-wheel focused selectors each pass **81 tests, 1 JAX-only skip, 238 deselected**. Full Ruff passes; MyPy passes for **224 source files**; RouteB app venv dependency check is **69 compatible packages**. Twelve rebuilt wheels match the 592-file source identity `68c36f96d7175cc2865495c6db4842a45c82c64ae2d84e53af52d2111c954275`, and wheel provenance smoke confirms **12 modules / 6 entry points**. Evidence: `docs/audit_evidence/20261022/package_json_nonfinite_constants_20261022.json`. Synthetic JSON only; continue local audit. Hosted CI and data-authentication gates remain open.


## 2026-10-22 A10 Research ArtifactStore non-finite JSON

The integrity index, completion marker, and L1 manifest readers now reject NaN/Infinity constants as well as duplicate fields. Tests re-hash malformed index/manifest content and their markers to confirm rejection is independent of the outer checksums. Source and installed-wheel selectors each pass **3 new cases**; the artifact_restore compatibility selector passes **2 cases**. Ruff and MyPy (`packages qa`, 224 files) pass; the RouteB app venv has 69 compatible packages. The rebuilt 12-wheel set matches the 593-file source identity `7944048b1493902b099caeed87f27073027d807ed4030851b0a959e7cc478406`; installed wheel provenance confirms 12 modules and 6 entry points. Evidence: `docs/audit_evidence/20261022/research_artifact_nonfinite_json_20261022.json`. Synthetic artifacts only; external CI and research evidence gates remain open.


## 2026-10-22 A10 persisted research JSON non-finite rejection

The Research ArtifactStore's index, completion marker, and L1 manifest loaders and the registered-ledger reconciler now reject `NaN`/`Infinity` constants as well as duplicate keys. Synthetic tests include tampered artifacts with recomputed outer checksums and malformed fold reports. Combined source and installed-wheel regression: **68 passed** each; full Ruff passed and MyPy passed **224 source files**. The 12-wheel manifest matches the verified 593-file source identity `ca1dabb25c6d1a879c4821716d96b2688cc6436643344e083ebfbaf6b3c72d1f`; RouteB app venv shows 69 compatible packages and provenance for 12 modules / 6 entry points. Evidence: `docs/audit_evidence/20261022/persisted_research_json_nonfinite_rejection_20261022.json`. Only synthetic temporary files were used; external CI and market/economic evidence gates remain open.


## 2026-10-22 A10 persisted research JSON non-finite rejection — expanded

The Research ArtifactStore, legacy project access inventory, and registered-fold ledger reconciler now reject `NaN`/`Infinity` constants as well as duplicate keys. Artifact tests recompute enclosing hashes to prove malformed inner values remain rejected. Combined source and installed-wheel selectors each pass **182 tests**. Full Ruff passes and MyPy passes **224 source files**. The 12-wheel manifest matches the verified 593-file source identity `08e500cd08f2ec9d6baa6a6b1961d32f72c50ab9d728d6c9be057cc44b403a13`; RouteB app venv reports 69 compatible packages and provenance for 12 modules / 6 entry points. Updated evidence: `docs/audit_evidence/20261022/persisted_research_json_nonfinite_rejection_20261022.json`. Synthetic temporary fixtures only; external gates remain open.


## 2026-10-22 A10 cashflow, manifest, and wheel provenance strict JSON

Binance transfer payloads, source/wheel identity manifests, and installed-wheel `direct_url.json` readers now reject non-standard `NaN`/`Infinity` constants in addition to duplicate keys. Source and installed-wheel regression selectors each pass **237 tests** (one expected warning from the duplicate-ZIP-member fixture). Full Ruff passes; MyPy passes **224 source files**; RouteB app venv has 69 compatible packages. Twelve wheels match the 593-file source identity `ee99c214d58820978f8e1e8650d0868530fd140472c8a0c208e40fd05f6725a1`; wheel provenance smoke verifies 12 modules and 6 entry points. Evidence: `docs/audit_evidence/20261022/strict_json_manifest_cashflow_provenance_20261022.json`. Only synthetic payloads were used; Hosted CI and authenticated account/market evidence remain open.

## 2026-10-22 M1.3 analyzer non-finite JSON rejection

Both independent M1.3 report analyzers now reject `NaN`, `Infinity`, and `-Infinity` in every JSON field, including extension fields outside their metric-specific finite-number checks. The M1.3 analyzer/runner regression passes **46 tests, 1 symlink-capability skip**; Ruff passes. The 594-file source manifest verifies as `a8b56b74d59ad12d2d9414650c777723f5d5b52bcde5a9f98b94925b1f51ae5e`. Evidence: `docs/audit_evidence/20261022/m13_analyzer_nonfinite_json_20261022.json`. Compatible native stage artifacts and external A01-A14 research gates remain open.

## 2026-10-22 latest cursor — M4.1 contract multiplier exposure consistency

The next source audit found that observations, pending exposure, hard-risk position valuation, and executed exposure still multiplied native contract quantity by mark without the now-supported `contract_multiplier`. The historical factory uses the frozen multiplier consistently for notional and reports effective base quantity, while planning remains in native contract units. Source and installed-wheel selectors each pass **151 tests**; Ruff and MyPy (155 package files) pass. The RouteB Gym bridge wheel is installed in `.venv-win-app`, its installed source hash matches, and 69 dependencies are compatible. Source identity: `195cd507685a1711f9b5e36834612657d531f39308e9c8002e021a67837a5e64`; evidence: `docs/audit_evidence/20261022/a09_contract_multiplier_exposure_20261022.json`. Next continue the audit for other locally actionable A01–A14 engineering gaps. Authenticated cross-asset specs/Funding, Hosted CI, independent-validation inputs, and unconsumed data scope remain external gates.

## 2026-10-22 superseding cursor — multiplier-aware account accounting

The M0.1 research report path also failed to apply native contract multipliers. Turnover, slippage, realized fill cash flows, and marked open PnL now use one explicit stable linear multiplier across the report; inverse or unknown position capability, missing/nonpositive multipliers, and multiplier changes fail closed. The Gym bridge and Research source/wheel suites pass (245 combined source cases; 151 Gym installed-wheel cases; 94 Research installed-wheel cases), MyPy passes 155 package files, and the 594-file source identity is `d0d500139720741e4bbd40e19668b2d6a13b8c54aa52233474e19fc92dc22e06`. Evidence: `docs/audit_evidence/20261022/a09_m01_multiplier_accounting_20261022.json`. Only synthetic BTC×2 facts were used; authenticated exchange rules, real funding/equity reconciliation, hosted CI, and independent research gates remain open. Next audit the remaining locally actionable A01–A14 engineering paths.


## 2026-10-22 — A09/M0.1 positionless fill multiplier guard

**Finding:** the account-row path could see fills without any position snapshots, then silently assume multiplier 1. For supported linear contracts with a multiplier above 1, turnover and slippage were understated. The main native report often caught this later through equity reconciliation, but direct account-row reports and compact reports without reconciliation could return wrong values.

**Change:** account rows, native reports, and equity reconciliation accept the frozen rule multiplier. Production application, the zero-target probe, and F7 profile pass the multiplier from the historical factory. If fills exist and no position facts establish a multiplier, omitting the frozen multiplier now raises MetricsError; when both sources exist they must agree. Added multiplier-2 positionless round-trip coverage and a conflicting-rule regression.

**Verification:** source and installed-wheel metrics/liquidation selectors each pass **96 tests**. The preceding cross-package source and installed-wheel selector each passed **246 tests** (18 upstream warnings). Ruff passed; MyPy passed packages qa (**224 source files**); RouteB .venv-win-app reports **69 compatible packages**. The Research and Orchestrator wheel modules match source byte-for-byte. Source manifest verifies **594 files**, identity 0f02c4421cb6da5db27a662bc6872c545b7f7a1ff36b5b53678c650858d6b670. Evidence: docs/audit_evidence/20261022/a09_m01_positionless_fill_multiplier_20261022.json.

Only synthetic data was used. No market, account, ledger, Validation, Test, or Holdout payload was accessed. Authenticated rules, complete cashflow provenance, and hosted CI remain open. Continue the local A01–A14 audit.

## 2026-10-22 latest cursor — A08 archive ASCII decimal lexemes

Binance archive numeric fields in bookTicker, fundingRate, klines, and mark/index-price klines now require canonical ASCII fixed-point tokens, preventing Python float() from silently normalizing underscores or Unicode digits. Source and installed-wheel archive/data-event selectors each pass **53 tests**. Ruff, py_compile, targeted MyPy, 12-wheel/12-sdist source matching, installed parser hash equality, 12-module/6-entry-point smoke, and the 69-package dependency check pass. Source manifest verifies 595 files at d68b5f7c5a00f8e6cd5fb60154334826930910afbed71b7cd30e4dcacd66d21e. Evidence: docs/audit_evidence/20261022/a08_archive_ascii_decimal_lexemes_20261022.json. Synthetic ZIP fixtures only; continue local A01–A14 audit.

## 2026-10-22 latest cursor — A08 shared Event ASCII decimal lexemes

The shared Event schema now rejects malformed numeric strings (underscores, Unicode digits, plus signs, whitespace, and exponent notation) across all ingestion paths, not only Binance archives. Source and installed-wheel data/archive/feature/dataset selectors each pass **75 tests**. Ruff, py_compile, MyPy, all-12-wheel source matching, installed import provenance (12 modules/6 entry points), and the 69-package dependency check pass. Source identity: 595 files, 66bf9cbff78d9f56e091f32c213562092b1371b657d8524e69a879486432a5f5. Evidence: docs/audit_evidence/20261022/a08_shared_event_ascii_decimal_lexemes_20261022.json. Synthetic inputs only; continue local A01–A14 audit.

## 2026-10-22 latest cursor — F7 compact snapshot position-fact reuse

Compact native report snapshots now reuse immutable position facts when they have not changed. Synthetic 1/30/90-day native runs preserve full trajectory and native metric hashes; 90-day retained RSS peak delta dropped 74,108,928 bytes (15.68%) to 398,458,880 bytes. Source selectors (186 prior cases plus the new immutability regression) and installed-wheel selectors (2 targeted plus 95 snapshot/metrics cases) pass; Ruff, py_compile, MyPy, 12-wheel source match, import smoke and dependency check pass. Current 595-file source identity: 336d37fdb1fe904756b8c5cb9532df8a60b88185e5e14192f13647a182418415. Evidence: docs/audit_evidence/20261022/f7_native_growth_position_reuse_20261022.json. F7 remains partial because per-barrier snapshots are still O(n); continue the source audit and streaming improvements.

## 2026-10-22 latest cursor — F7 online native return metrics

Native reports with account rows omitted now compute return/drawdown/dispersion and exact tail metrics online; the native report path no longer retains a full return array. Exact metrics survive an interim result followed by further accumulation. Source and installed-wheel combined regressions each pass **188 tests**; Ruff, py_compile, MyPy, all-12-wheel source matching, 12/6 import provenance, and the 69-package dependency check pass. The final 90-day synthetic native replay completed 129,597 steps with trajectory and metric hashes equal to the pre-change profile. Source identity: 595 files, c84e6bc3b712fb36394b52bb11b7a110c89254eb53fcf5b22717c597aebe1e27. Evidence: docs/audit_evidence/20261022/f7_online_native_metrics_20261022.json. F7 remains partial because compact per-barrier snapshots still use O(n) memory and full report rows are intentionally materialized when requested.

## 2026-10-22 latest cursor — F7 online Native summary reducer

The summary-only online Native reducer consumes each barrier once without retaining the compact per-barrier snapshot series. Sourced cashflow at a repeated timestamp now fails before equity reconciliation with the same explicit duplicate-barrier reason as the batch report. The F7 profile now exercises online reduction.

- The focused source and installed-wheel history/metrics/liquidation/rollout selector passes **194 tests** each; Ruff, py_compile, and targeted MyPy pass.
- The 90-day synthetic replay covered **129,597 decisions / 129,600 barriers**. Online and lean trajectory SHA both equal `4954c0cea6125814c08ee7b3e3aacd0397d24f6b84da7e306f410bb2eea5420f`; native metric SHA matches the baseline `0c9bb2fbdbd241a24d05c4c9b6a3e864bf37db3ea5e78dc71eb9785ea9e50a45`.
- One RSS observation: online +509,599,744 bytes / 83.38 s; lean +510,210,048 bytes / 76.99 s. Peak RSS is effectively the same and does not support a stable reduction claim; online avoids retaining O(n) compact snapshots and is substantially faster than the prior disk-spool prototype.
- The 597-file source identity `d7771d4ad8683e5a493c9c48545aeeaf9d6108b8d47c938a964823022d12fed2` verifies. Twelve wheels and twelve sdists built; all wheel modules match source; installed smoke covers 12 packages and 6 entry points; all 69 RouteB app dependencies are compatible.

Evidence: `docs/audit_evidence/20261022/f7_online_native_summary_stream_20261022.json` and `docs/audit_evidence/20261022/f7_online_native_summary_validation_20261022.json`. Continue the local A01–A14 source audit from current RouteB code. Authenticated cashflow provenance, authenticated market rules, independent research windows, and clean Hosted CI remain external gates.



## 2026-10-22 latest cursor — A08 archive date input normalization

`import_day` and `import_range` now reject invalid or noncanonical date inputs with `DataError` before archive fetch. Source and installed-wheel archive/data-event selectors each pass **66 tests**; Ruff, MyPy, py_compile, all-12-wheel source matching, installed module SHA parity, and 69-package dependency compatibility pass. Source identity: 598 files, `d76d042ab8932adb60a6cf8eed3f94f2d25058c745ae36f49aae6fc86f20de09`. Evidence: `docs/audit_evidence/20261022/a08_archive_date_input_20261022.json`. Continue the local A01–A14 audit from RouteB source.


## 2026-10-22 latest cursor — A08 archive ingress validation before network access

Date inputs now use canonical `YYYY-MM-DD` validation, and `import_day`/`import_range` validate positive integer `delay_ns` before output creation or archive fetch. Source and installed-wheel selectors each pass **74 tests**; Ruff, MyPy, py_compile, 12-wheel source match, installed data-module SHA parity, source manifest, and 69-package dependency checks pass. Source identity: 598 files, `4947b39a4e02482a684e32d9033c9b6f559a86050002c843bbd29e67f9b76cd9`. Evidence: `docs/audit_evidence/20261022/a08_archive_date_input_20261022.json`. Continue local A01–A14 engineering audit; authenticated account/market evidence and clean Hosted CI remain external gates.


## 2026-10-22 latest cursor — A08 strict data ingress

Archive date and delay inputs now fail with stable `DataError` before output or fetch; the shared UTC timestamp parser rejects Unicode digits instead of silently normalizing them. Source and installed-wheel focused selectors each pass **78 tests**; Ruff, MyPy, py_compile, 12-wheel source match, installed module SHA parity, source manifest verification, and 69-package dependency compatibility pass. Source identity: 598 files, `9f59a570101cf63fa610cf828e2a8cf824cb42c170c8250c8cf1ebf86535ee3f`. Evidence: `docs/audit_evidence/20261022/a08_archive_date_input_20261022.json`. Continue local A01–A14 review; real authenticated market/account evidence and clean Hosted CI remain external gates.


## 2026-10-22 latest cursor — A08 archive public inputs

Both Binance archive import APIs validate symbol/date/instrument/delay inputs before output creation or fetch; the shared ISO timestamp parser rejects Unicode digits. Source and installed-wheel focused selectors each pass **86 tests**. Ruff, MyPy, py_compile, all-12-wheel source match, installed `rles_data` module SHA parity, manifest verification, and 69-package compatibility pass. Source identity: 599 files, `4794b2bad7d1ebfe582716c00de727d5eca8abf9df89554f622a7ba1907c8e37`. Evidence: `docs/audit_evidence/20261022/a08_archive_date_input_20261022.json`. Continue local A01–A14 source audit; authenticated data evidence and Hosted CI remain external gates.


## 2026-10-22 latest cursor — A08 archive URL boundary

`fetch_archive` rejects unsafe relative URL syntax and paths before cache or network effects. Public imports validate symbol/date/instrument/delay, and shared UTC timestamps reject Unicode digits. Source and installed-wheel selectors each pass **96 tests**; Ruff, MyPy, py_compile, 12-wheel source matching, installed module SHA parity, source manifest, and 69-package check pass. Source identity: 599 files, `e41896d707a8d954efb303e755d5e76db2452b44070231fd1b7834f62d11ce0b`. Evidence: `docs/audit_evidence/20261022/a08_archive_date_input_20261022.json`. Continue local A01–A14 code audit; authenticated evidence and Hosted CI remain external.


## 2026-10-22 latest cursor — A10 curated JSON error boundary

Curated marker and event-row readers now convert strict JSON parse failures (duplicate keys, non-standard constants, invalid UTF-8/types) into contextual `DataError`. Source and installed-wheel combined selector each pass **118 tests**; Ruff, targeted MyPy, py_compile, all-12-wheel match, installed data module SHA parity, source manifest, and 69-package check pass. Source identity: 600 files, `432806e0ecfa9598e278a06967838cc811234e13fbd6a96e9b9d5f89e30bd9e9`. Evidence: `docs/audit_evidence/20261022/data_json_error_boundaries_20261022.json`. Continue local A01–A14 source audit; authenticated market/account evidence and clean Hosted CI remain external gates.


## 2026-10-22 latest cursor — A10 data JSON boundaries

Curated data readers and Binance archive/Funding response/receipt readers now convert strict JSON parser failures to contextual `DataError`. Combined source and installed-wheel selector each pass **121 tests**; Ruff, MyPy, py_compile, 12-wheel match, installed module SHA parity, source manifest and 69-package checks pass. Source identity: 600 files, `5b957ddcde5607f25cb30d8d7ab5da83c5e8322e71d913d76cce94de838f3918`. Evidence: `docs/audit_evidence/20261022/data_json_error_boundaries_20261022.json`. Continue local A01–A14 source audit; authenticated external data evidence and clean Hosted CI remain open.


## 2026-10-22 latest cursor — A10 curated and archive reader errors

Curated JSON markers/events, ASCII marker receipts, and Binance archive/Funding response/cache receipts now convert malformed inputs to contextual `DataError`. Source and installed-wheel combined selectors each pass **122 tests**. Ruff, MyPy, py_compile, 12-wheel match, installed module hash parity, source manifest and 69-package check pass. Source identity: 600 files, `2c8edff0b680f3a6f3e98f1e8ed15ff800f351ce9797b91391602ff334bb8f1f`. Evidence: `docs/audit_evidence/20261022/data_json_error_boundaries_20261022.json`. Continue local A01–A14 review; authenticated market/account evidence and clean Hosted CI remain external gates.

## 2026-10-22 latest cursor — M0.1 Funding identity collision parity

Added an explicit same-timestamp conflicting Funding identity regression. The online accumulator rejects the changed event fact as it arrives; the batch report rejects the same ledger. This keeps the accounting paths aligned and prevents silent acceptance of a reused exchange event ID with a changed PnL. Source and installed-wheel selectors each pass **195 tests**; Ruff and `git diff --check` pass. The current manifest has 600 source files, identity `c83496706f98ce34f5f9cadeb4a8b1c9677e88159267a65a9aac93b32fec2ec1`, and all 12 wheels match package source. Evidence: `docs/audit_evidence/20261022/m01_funding_duplicate_identity_20261022.json`. Continue the local M0.1/A01–A14 audit; authenticated real-account/market evidence and clean Hosted CI remain external gates.

## 2026-10-22 latest cursor — M0.1 online accumulator fail-closed state

Reproduced that `NativeReportAccumulator` remained reusable after a snapshot conflict despite partial state mutation. It now records ingestion/finalization failures and rejects subsequent report operations, chained to the original exception. New tests first failed on the previous implementation and now pass in source and the installed RouteB wheel. Both four-file selectors: **197 passed**; Ruff/MyPy/py_compile/diff checks pass; rebuilt research module hash matches source; all 12 wheels match and 69 dependencies are compatible. Source identity: `348f837f699634f23d3d48891c6225c60ba4b0bafe9e6e9ec95912f8241403e3`. Evidence: `docs/audit_evidence/20261022/m01_online_accumulator_fail_closed_20261022.json`. Continue M0.1/A01–A14 local review; authenticated real-account cashflows, registration history completeness and clean Hosted CI remain open gates.

## 2026-10-22 latest cursor — M0.2 CI lane fail-closed

Fixed a reusable-workflow false-green path: invalid lane strings could skip every conditional CI job. A required validation job now rejects unsupported lanes; all real jobs depend on it, and manual dispatch choices are checked against the validator set. The 12-test workflow contract suite, Ruff, targeted MyPy, YAML parse, wheel package inclusion and source parity checks pass. Twelve workspace wheels match package source. Current manifest: 602 files, `73470b5f5c08dc4a744c70849320a47c61d967462f4fac2d526b8e061513c934`. Evidence: `docs/audit_evidence/20261022/m02_ci_lane_fail_closed_20261022.json`. Hosted Linux/Windows CI has not run; keep M0.2 open until the actual clean jobs pass.

## 2026-10-22 latest cursor — M0.3 registry SQLite foreign keys

Enabled SQLite foreign-key enforcement on every `ProjectResearchRegistry` transaction. The regression first reproduced an accepted orphan development-result receipt with enforcement off; after the fix, the 167-test registry/WFO/fresh-fold source and installed-wheel selectors pass. Ruff, targeted MyPy, py_compile, source manifest and all-12-wheel checks pass. The existing RouteB registry was inspected read-only and has zero foreign-key violations. Source identity: 602 files, `9e4730fd2a0130f2c041a7a28fe50723b27b766eff5a80414b23682dbe6b181e`. Evidence: `docs/audit_evidence/20261022/m03_sqlite_foreign_key_enforcement_20261022.json`. Historical access before project-wide registration still cannot be proven; continue the M0.3 audit and remaining local A01–A14 work.


## 2026-10-22 latest cursor — A08 archive ingress and cache boundaries

Archive ingress accepted malformed instrument IDs, non-ASCII `.CHECKSUM` bytes and invalid ZIP bytes leaked decoder exceptions, and an unreceipted existing cache bypassed the file-size cap before a full read. The importer now rejects invalid IDs before side effects, normalizes checksum/ZIP errors to `DataError`, and checks cache file size before reading bytes. Source and installed-wheel archive/data-event selectors each pass **100 tests**. Ruff, MyPy (2 changed files), py_compile, and diff checks pass. Built **12 wheels + 12 sdists**; all 12 wheels match source, the installed `rles_data.binance_archive` hash matches source, provenance smoke verifies 12/12 packages and 6/6 entry points, and all 69 app dependencies are compatible. Source identity: 604 files, `91715791405993ba1d4a89954356a9e48443cf4a8450a79819ff3169a0826099`. Evidence: `docs/audit_evidence/20261022/a08_archive_instrument_identity_20261022.json`. No live archive or research data accessed; continue the local A01–A14 audit.


## 2026-10-22 latest cursor — M0.1 cashflow identity canonicalization

The direct and paged Binance USDⓈ-M transfer importers accepted padded account scope and currency strings; the account scope was silently stripped into event/source IDs, allowing distinct inputs to collapse onto one identity. Both entry points now require nonempty trimmed identifiers before parsing. Twelve synthetic direct/paged cases reproduced the old acceptance. Cashflow/metrics/liquidation source and installed-wheel selectors each pass **140 tests**; Ruff, MyPy on the changed source, and py_compile pass. Built **12 wheels + 12 sdists**; all 12 wheels match package source, the installed module hash matches source, provenance smoke covers 12 packages / 6 entry points, and all 69 app dependencies are compatible. Source identity: 604 files, `092cebb4581e871856e9b46d1621f8aa480f9a230c1770bd97fd34e70e4717b9`. Evidence: `docs/audit_evidence/20261022/m01_cashflow_identity_canonicalization_20261022.json`. Synthetic payloads only; this fixes identity normalization but does not authenticate a real account export. Continue M0.1/A01–A14 local audit.


### 2026-10-23 follow-up  M0.1 Native equity reconciliation precision

Low Decimal precision rounded native position quantity and equity checks. Batch and online paths now use bounded exact operations for quantity, cash, fees, funding, valuation, equity, residuals and tolerances. The five-file source and installed-wheel selectors each pass 251 tests; Ruff and targeted MyPy pass. Source/wheel module SHA-256 matches. Source manifest: 604 files, identity eae1a07e7bdb04d4c9ae7fed5c90d1a22cb925bb4e109b683d645e24b7992b24. Evidence: docs/audit_evidence/20261023/m01_native_equity_reconciliation_decimal_context_20261023.json. No real account data was accessed; M0.1 source authentication remains open.


### 2026-10-23 M0.1 continuation  Decimal exponent bounds

Extended the exact Decimal fix to representable low-exponent sums and products, and made arithmetic independent of inherited Decimal status flags. If a result must round, the helper raises a stable metrics error. The five-file source and installed-wheel selectors each pass 252 tests; Ruff and targeted MyPy pass. Module SHA-256 matches between source and RouteB app venv; 604-file source manifest identity: b9d91a34cbc084e995e42ddd6d5aa93135e93b0b73c757051787f93d05030292. Evidence: docs/audit_evidence/20261023/m01_native_decimal_range_boundaries_20261023.json. No authenticated data was used; continue remaining local audit work.

## 2026-10-23 最新续做游标 — A04 reconciler 扫描失败隔离

`qa/world_model/reconcile_routeb_research_ledger.py` 的递归扫描现使用确定顺序且不跟随目录符号链接的 `os.walk`；任一子目录扫描错误、将被跳过的符号链接目录或不可读/无效 failure、registration receipt，均会被显式拒绝。两个 development-access reader 都拒绝损坏或非对象 identity；五处只读 SQLite 连接已修复 URI 路径转义，包含 `#` 的注册库路径回归通过。reconciler 全文件 **26 passed**，Ruff 通过，项目 MyPy **225 files clean**。604-file source manifest identity：`c080242137afcfa8ad4bcab0356345432f0d54ee27985123fa52c920abe362cc`。证据：`docs/audit_evidence/20261023/reconcile_directory_scan_fail_closed_20261023.json`。继续 A01–A14 其余本地可实施问题；外部认证门槛保持独立。

## 当前续做游标 — 2026-10-23 F7 Native 报告生命周期

继续在 `D:\Program Files\NautilusTrader_RL_ES_RouteB` 的当前源码上工作。已在 Native 报告成功结束时释放去重/现金流对齐索引及精确尾部 heap/spool；快照摄取或报告最终化失败时也立即释放 spool；编排层复制终端 `all_orders` 后释放完整最终快照。测试分别达到 131、132、134 项聚焦通过；跨 Native、CLI、真实应用流程和 orchestrator workflow 的完整选择器 **197 passed / 13 third-party warnings**，异常安全改动后 **186 passed**；Ruff 与 MyPy 通过。合成 10,000 条现金流显示对齐 dict 从 294,992 B 到 64 B；真实应用流程使用仓库内归档 fixture 和临时隔离 registry，不读取用户项目 registry 或账户数据。详见 `docs/audit_evidence/20261023/f7_*release*20261023.json` 与执行计划。

下一步：继续按 `DEVELOPMENT_GOALS_REVIEW_20261017.md` 审核未完成但不依赖外部授权的 A01–A14 源码项；先检查报告/研究访问 reader 与安装分发边界中的新缺口，再做红测、修复及适用的回归。F7 仍未关闭：报告所需唯一 fills/Funding/orders/cashflows 仍按输出合同增长，完整长历史 RSS/轨迹以及获批的真实历史 Train 证据尚缺。真实现金流来源认证、历史访问完整性、独立 Validation、新市场 scope/深度、历史试验覆盖、M4.2 均保持独立门槛，不解封 Test/Holdout。

### A10 当前续做记录 — WorldModel bundle scaler manifest

继续检查发现 `save_bundle()` 会将 scaler manifest 和 learner dataclass 配置中的非字符串 map 键隐式改写。现已递归拒绝此类键，并对 scaler 序列化结果执行重复键严格解析。RouteB 隔离 JAX CPU WorldModel additive **21 passed**；Ruff/MyPy 通过。仅用合成非法输入。证据：`docs/audit_evidence/20261023/a10_bundle_scaler_string_keys_20261023.json`。接续从 A01–A14 当前源代码审计中选取下一个无需外部授权的缺口；F7 长窗 RSS、真实现金流、历史覆盖、独立 Validation、新市场授权/深度、历史有效试验数与 M4.2 仍开放。

### A10 当前续做记录 — ResearchArtifactStore manifest key

继续对照 JSON 序列化入口发现，ResearchArtifactStore 的 L1 manifest canonicalizer 会把嵌套整数键转成字符串，根整数键则泄漏原始排序异常。现已递归拒绝非字符串键，并将 manifest 编码失败统一为 `ArtifactError`。临时合成 artifact selector **7 passed**，Ruff/MyPy 通过。证据：`docs/audit_evidence/20261023/a10_research_artifact_string_keys_20261023.json`。继续审计剩余序列化/读取边界与 A01–A14 本地可实施项；F7 全长时序与真实历史、现金流真实性、历史访问覆盖、独立 Validation、新市场 scope/深度、历史试验覆盖及 M4.2 仍未关闭。

### A04 当前续做记录 — registry 循环 JSON 拒绝

`ProjectResearchRegistry` 共用 JSON 校验器对循环 list/dict 会无限递归至 `RecursionError`；RouteB 七个 strict JSON reader 对 20,000 层输入也会泄漏解码器递归异常。序列化遍历现改为迭代并检测循环，七个 reader 均将超深错误归一为 `ValueError`。显式从源码导入后，registry 与七个 reader 的选择器 **146 passed**；Ruff 与 MyPy 通过。仅使用临时 registry 和合成输入。证据：`docs/audit_evidence/20261023/a04_registry_circular_json_rejection_20261023.json`。继续扫描 A01–A14 其余可在本地落地的 reader/serializer 缺口；外部认证与独立 Validation 等门槛保持未关闭。

### A10 当前续做记录 — worker payload 循环引用

Compute worker `small_json()` 的递归预检对循环 dict/list 会先于 JSON 编码触发 `RecursionError`。现改为迭代遍历，循环对象明确失败为 `ComputeProtocolError`，编码器递归错误也统一转成协议错误。与 strict JSON/registry 回归合计 **164 passed**；Ruff/MyPy 通过。仅合成内存 payload。证据：`docs/audit_evidence/20261023/a10_worker_payload_cycle_rejection_20261023.json`。继续 A01–A14 其余源码项；任何外部数据认证和研究门槛仍开放。

### A10 当前续做记录 — contracts JSON 递归深度

canonical DTO reader 与 CLI error-envelope reader 直接调用 `json.loads`，深层输入会泄漏 `RecursionError`；canonical DTO 的递归对象解码阶段也覆盖了 JSON decoder 尚未溢出的层级。现 parse、DTO 转换和 canonical re-encode 均返回边界明确的 `ValueError`。contracts reader + config selector **39 passed**；追加递归解码级回归 **2 passed**，Ruff/MyPy 通过。证据：`docs/audit_evidence/20261023/a10_contract_json_depth_boundary_20261023.json`。继续检查其它直接解析入口和 A01–A14 可在本地完成的事项。

### A10 当前续做记录 — direct ingress/readers JSON 递归深度

Binance transfer 单页/分页导入、legacy research inventory 与 ResearchArtifactStore 的 index/manifest readers 也会泄漏 `RecursionError`。现分别统一映射为各自的 MetricsError、DataError 与 ArtifactError。三个测试文件 **169 passed**；Ruff 通过，三个源文件 MyPy 通过（忽略 venv 对无 `py.typed` 模块的报告）。仅使用深层合成 payload 和临时文件。证据：`docs/audit_evidence/20261023/a10_direct_json_reader_depth_boundaries_20261023.json`。继续处理剩余本地 A01–A14 项。
### A10 当前续做记录 — JSON reader CLI 边界复核

追加 rles-data `--manifest-json` 深层输入检查后，CLI reader 也改为把 `RecursionError` 转成 `DataError`；合并后的 JSON reader/transport/contracts 选择器 **258 passed**，Ruff 与针对性 MyPy 通过。证据：`docs/audit_evidence/20261023/a10_direct_json_reader_depth_boundaries_20261023.json`。继续审计其余直接 JSON decode 点与本地 A01–A14 项。
### A10 当前续做记录 — distribution provenance JSON 深度

source identity verifier、wheel manifest reader 与 installed-wheel `direct_url.json` provenance parser 对超深 JSON 会泄漏 `RecursionError`。现统一映射到各自无效 manifest 错误。Source/wheel manifest selectors **27 passed / 1 expected duplicate-ZIP warning**；Ruff/MyPy 通过。仅合成文件与 metadata。证据：`docs/audit_evidence/20261023/a10_distribution_manifest_depth_rejection_20261023.json`。继续核对剩余 distribution/reader 边界以及 A01–A14 本地项。
### A10 当前续做记录 — QA analysis/reconciler JSON 深度

M13 两版阶段分析器、fold-ledger reconciler 与 distribution manifest readers 对深层 JSON 的异常未完全包住。现都将 `RecursionError` 收敛到既有报告/metadata error contract。五个 QA 测试文件 **93 passed / 1 symlink skip / 1 expected ZIP warning**；Ruff 与针对性 MyPy 通过。证据：`docs/audit_evidence/20261023/a10_qa_reader_depth_normalization_20261023.json`。继续从最新审计剩余本地可行动项中推进。
### A10 当前续做记录 — current source 12-wheel parity

本轮包级 reader 修复后重新生成 605-file manifest（identity `f1a5ed8b47894570e18ccbc516d090ece79530a6d8088128bb191d0a0eb9c256`），构建 12 wheels + 12 sdists；wheel manifest `all_match=true`。12 个包、6 个 entry point 在一次性隔离 venv 中从安装 wheel 导入，所有路径均落在该 venv。JSON/registry selector 源码和安装 wheel 各 **258 passed**；QA reader 选择器 **93 passed / 1 symlink skip / 1 expected warning**。Hosted clean Linux/Windows CI 仍需在可推送 revision 上运行。证据：`docs/audit_evidence/20261023/a10_current_source_12_wheel_provenance_20261023.json`。

### 2026-10-23 current continuation cursor — A10 source manifest v3

Actual RouteB source manifest is v3: Git text line-ending normalization, exact-byte -text preservation, runtime/upstream untracked-path redaction, and fail-closed old-schema verification. RouteB dedicated .venv-win-app tests: 9 passed; Ruff/MyPy passed; 605-file manifest identity cbd98ba52db1e2ffcd3e564060aefdcab2e75f0b5ca30090c7e5d2477f0b3a82 self-verifies. Evidence: docs/audit_evidence/20261023/a10_source_manifest_platform_stable_v3_20261023.json. Hosted Linux/Windows/JAX CI for this revision remains pending. Continue implementation-ready A01-A14 engineering work.
### 2026-10-23 current continuation cursor — A10 compute transport decoder

Fixed the remaining small_json round-trip decoder RecursionError leak as ComputeProtocolError. Actual RouteB source selector: 19 passed; Ruff/MyPy passed. Current 605-file source identity: dedf0097168391c29a321bcd54863b14de650c57f21d84f195115a91788fdfaa. Evidence: docs/audit_evidence/20261023/a10_compute_transport_decode_recursion_20261023.json. New Hosted clean CI is pending; continue other implementation-ready A01-A14 engineering issues.
### 2026-10-23 current continuation cursor — B4 native market fee/PnL slice

A deterministic long/short BTC/USD.SIM round trip now compares B4 ImaginedAccount with the actual Nautilus engine at matched 0.1% taker fees. Adapter suite: 38 passed; both signs match entry and terminal equity and finish flat. Current 605-file source identity ba7f46696b03e7ce62a7d889a94aec65dddae547fb39aa567e7778cd3f919e28. Evidence: docs/audit_evidence/20261023/b4_imagined_account_native_market_parity_20261023.json. This is a scoped synthetic parity result only. Next: add a deterministic native parity edge case while preserving the explicit B4 non-economic certification boundary.
### 2026-10-23 current continuation cursor — A10 Hosted MyPy repair

Hosted run 39905239046 failed only at workspace MyPy on Linux and Windows because two stale reset profiler scripts read removed factory-cache attributes; JAX CPU passed. Removed those obsolete output fields without changing historical Stage69 evidence. Actual RouteB workspace MyPy now passes 226 source files and Ruff passes. Current source identity: 95c14df7b49ae57d5f953a921f8c083f85c3def5710b473903acfb5604a5eba9 (605 files). Evidence: docs/audit_evidence/20261023/a10_stale_reset_profiler_mypy_fix_20261023.json. Fresh Hosted CI pending; continue local audit work.
## 2026-10-23 — A10 Linux MyPy ctypes portability

Hosted Linux CI for commit `19f633038b06b9bd3ed17e33a0fabca0d7f45213` found that `qa/profile_f7_tail_rss.py` directly referenced Windows-only `ctypes.WinDLL`, `WinError`, and `get_last_error`, which Linux MyPy stubs do not expose. The profiler now resolves those platform-only attributes with `getattr` after its existing Windows runtime guard, preserving its Windows behavior.

Local RouteB verification: MyPy passes for **226 source files**; Ruff and `py_compile` pass; a 20-observation Windows profiler smoke passes; the 605-file source manifest regenerates and verifies at `bfcba666b09f2978a607665b82650016e78aa3169cd1cf457b1e0035bedacfc7`. Evidence: `docs/audit_evidence/20261023/a10_linux_mypy_ctypes_portability_20261023.json`.

Next: push this fix to the RouteB-only backup and require a fresh Linux/Windows/JAX CI run. Continue the local A01–A14 audit while hosted jobs run; external research authorization/provenance and independent-validation gates remain open.
## 2026-10-23 — B4 signed fee and taker rebate parity

**Finding:** instrument-rule and native account paths permit signed maker/taker fees, but `ImaginedAccount` rejected every negative rate. Its simulated economics therefore diverged for exchange fee rebates even on the supported bar market-order path.

**Fix:** accept finite maker/taker rates in `(-1, 1)`. The native Nautilus long/short differential now covers both `+0.1%` taker fees and `-0.1%` taker rebates. A separate synthetic resting-limit regression confirms the maker rebate is credited; no native limit-maker parity is claimed.

**Verification:** B4 differential **4 passed**; full native adapter suite **40 passed**; adapter + WorldModel source selectors **104 passed, 5 JAX-only skipped** (the app venv has no JAX); MyPy passes **226 source files**, Ruff and `py_compile` pass. The 605-file source manifest verifies at `36352e9e5eae6c115465f15a605b530d560068c3eef7a847f924312735be132f`. Evidence: `docs/audit_evidence/20261023/b4_imagined_account_native_market_parity_20261023.json`.

**Boundary / next cursor:** synthetic BTC/USD.SIM bar fills only. Limit maker behavior is covered only by the simplified ImaginedAccount model; quote/depth, partial fills, funding, margin/leverage, liquidation, fee tiers and intrabar sequencing remain open. Continue B4 native parity and the locally actionable A01–A14 audit; external authorization/provenance and independent Validation gates remain open.
### B4 native market-position reversal — 2026-10-23

Added a second long/short native differential: after opening one unit, a two-unit opposing market order reverses the NETTING position, then a final market order flattens it. At each stage, `ImaginedAccount` and Nautilus match on net quantity and equity after realized PnL and 0.1% taker fees; the terminal native snapshot is flat. The test accounts for the adapter returning cumulative fill history and asserts the latest reversal fill side and quantity. The reversal selector passes **2 tests**; full adapter suite passes **42 tests**; adapter + WorldModel source selectors pass **106 tests with 5 JAX-only skips**. MyPy passes 226 files, Ruff passes, and the current 605-file source identity is `b7323e51f4c30d82c3c4c2f937c2070bae884b8c1490f7c6356a21b3e042ff64`. Evidence remains in `docs/audit_evidence/20261023/b4_imagined_account_native_market_parity_20261023.json`.

**Boundary:** synthetic single-instrument bar-market execution only. This does not establish limit-fill, quote/depth, partial-fill, funding, margin/leverage, liquidation, or historical exchange-fee parity.
### B4 quote-precision fee accounting — 2026-10-23

**Finding:** the USD 2-decimal native account reports a sub-cent `$0.00001` fill commission as `$0.00`; `ImaginedAccount` previously deducted `$0.00001` from its balance. Existing high-notional tests could not expose this because their equity tolerance scaled with a $100,000 balance.

**Fix:** `ImaginedAccount` now accepts `equity_precision` (default 8), validates it, rounds each maker/taker fee and reported equity to that currency precision, and retains internal realized cash at full float precision. B4 differential equity assertions now use absolute `1e-8` tolerance with zero relative tolerance. New long/short micro-notional cases verify native fee strings and accumulated cash.

**Verification:** small-notional selector **2 passed**; full adapter suite **44 passed**; adapter + WorldModel selectors **108 passed, 5 JAX-only skipped**; MyPy **226 files**, Ruff pass. The 605-file source identity verifies at `c13c45aa01d472340b8207691b89eb1ce9b56f2cb9472bd0e3eece02b610fb82`. Evidence: `docs/audit_evidence/20261023/b4_imagined_account_native_market_parity_20261023.json`.

**Boundary:** synthetic BTC/USD with USD precision 2 and bar-market execution. Callers must supply the correct equity currency precision when it differs from the default. Stablecoin and other currency precision should receive independent native parity coverage; depth, partial fill, funding, margin/leverage, liquidation and intrabar sequencing remain open.

### B4 continuation — exact currency tie rounding — 2026-10-23

**Finding:** `ImaginedAccount` computed fees in binary float and then used Python `round`. On USD half-cent values this disagreed with Nautilus `Money`: for example, a `$0.015` fee became `$0.01` in the imagined account but `$0.02` natively; a `$0.005` fee also incorrectly became `$0.01` instead of `$0.00`.

**Fix:** compute fee notional and signed fee from decimal string representations, then quantize per fill to the configured currency precision using `ROUND_HALF_EVEN`. Regression cases cover `$0.005`, `$0.015`, `$0.025`, `$0.035`, long/short, and positive charges/negative rebates.

**Verification:** new native differential selector **16 passed**; adapter + WorldModel selectors **124 passed, 5 JAX-only skipped**; Ruff passed; MyPy passed for **226 files**; the **605-file** source manifest self-verifies as `ff193c572db6f43345c8dc0103a8ffb1067e63f95e93b69617e37c8beaf74bc2`. Detailed evidence: `docs/audit_evidence/20261023/b4_imagined_account_native_market_parity_20261023.json`.

**Next:** continue exact native parity at supported execution boundaries. B4 quote/depth, partial fills, funding, margin/leverage, liquidation, fee tiers, intrabar sequencing and independent economic certification remain open; external A01–A14 provenance/authorization/Validation gates remain unchanged.

### B4 continuation — partial reduction ledger parity — 2026-10-23

Added a Native-versus-Imagined long/short scenario that enters one unit at 100, reduces half at 105, and flattens the remaining half at 95. At the reduction barrier, both paths agree on the remaining signed quantity and equity including realized PnL, residual mark-to-market PnL, and fees. Both finish flat at initial equity less `$0.20` in per-fill USD fees. Focused selector: **2 passed**; adapter + WorldModel selectors: **126 passed, 5 JAX-only skipped**; Ruff and MyPy (226 files) pass; 605-file manifest identity `233c1576ad8cbfe1ebdc2c037e120775ffaa32f3405ac3b3aa1325f94ddc1398`. Evidence remains in `docs/audit_evidence/20261023/b4_imagined_account_native_market_parity_20261023.json`.

Next: continue auditing B4 supported account paths for multiple fills within one bar, order cancellation/expiry, and exact native fee classifications, alongside remaining implementation-ready A01–A14 work. Do not widen economic claims beyond synthetic Native scenarios.

### B4 continuation — enforce the single-instrument account boundary — 2026-10-23

**Finding:** the `MarketSequence` contract rejects mixed instruments, but direct `ImaginedAccount.on_bar()` calls did not. A pending order could be executed against a different instrument's bar and contaminate position PnL.

**Fix:** lock each account to the first accepted candle instrument and fail before any clock, order, fill, or equity mutation if a later candle changes instrument. A regression confirms the error leaves the pending order and prior ledger state intact.

**Verification:** regression **1 passed**; adapter + WorldModel selectors **127 passed, 5 JAX-only skipped**; Ruff and MyPy (226 files) pass; 605-file source manifest self-verifies as `d4f99c1e0c86cc4578f4a7b582c7e3ff5df6b045925360decffcb0e009dee143`. Evidence: `docs/audit_evidence/20261023/b4_imagined_account_single_instrument_20261023.json`.

**Boundary:** multi-asset account economics remain out of scope until there is an explicit multi-position and currency ledger.

### A10 continuation — cross-version JSON nesting guard — 2026-10-23

Hosted Linux PR run `39909996795` exposed five deep-JSON failures in Binance single/page imports, the data manifest CLI, and research inventory/registry readers. They relied on `json.loads()` raising `RecursionError`, but parser behavior varied by Python runtime and some inputs reached a generic schema path. Added a shared 512-level structural scanner in `rles_contracts.json_utils`; it respects quoted/escaped strings and runs before parsing in all seven package strict readers plus the four affected reader boundaries.

**Verification:** affected reader files **197 passed**; the full PR selector **903 passed, 4 skipped, 644 deselected**; MyPy passes **227 files**; Ruff passes; the 606-file source identity verifies as `95d8113385895d4e5dc5100f63958e645e2493e55a98d22b897f03bf9729bcb5`. Machine evidence: `docs/audit_evidence/20261023/a10_cross_version_json_nesting_guard_20261023.json`.

**Next:** push the fix to the RouteB-only snapshot and verify fresh Hosted Linux, Windows and JAX CPU jobs. Continue auditing the remaining serialization/reader boundaries; external research gates remain independent.

### 2026-10-23 current continuation cursor — A10 reader depth coverage follow-on

The 512-level JSON nesting scanner now also guards Research ArtifactStore integrity/completion/L1 manifests, contract error/canonical-byte readers, and compute transport serialization/decode. Verification was repeated on the resulting source: full PR selector **903 passed, 4 skipped, 644 deselected**; WorldModel suite **234 passed, 14 JAX-only skipped**; focused boundary selector **53 passed**. MyPy (227 files) and Ruff pass. The current 606-file source identity is `75a811a6bd0aa366acdc0077149b2124d3e3bc5ad7b05f25b8ab46079492bb64`. Evidence: `docs/audit_evidence/20261023/a10_cross_version_json_nesting_guard_20261023.json`.

Next: publish this follow-on source to the RouteB-only snapshot branch, inspect the fresh Hosted Linux/Windows/JAX CPU jobs, and continue the source audit. Hosted confirmation and independent provenance/research gates remain pending.

### 2026-10-23 A10 hosted Linux follow-on correction

Fresh run `39920709883` confirmed Linux PR tests **896 passed, 11 skipped**; Windows CPU and isolated JAX CPU jobs passed. Linux WorldModel then exposed one additional deep-JSON difference in `qa/world_model/reconcile_routeb_research_ledger.py`: Python continued to object validation rather than raising `RecursionError`. The reconciler now uses the shared depth preflight before decoding canonical reports. After the fix: focused selector **32 passed**, full PR selector **903 passed, 4 skipped, 644 deselected**, WorldModel suite **234 passed, 14 JAX-only skipped**, MyPy **227 files**, Ruff pass. Updated 606-file source identity: `2a62c2b1df6ff198bcd2283fddd5d488fb04882a359771ec2c3cef86a95e3313`. Evidence file contains both the hosted finding and local repair verification.

Next: publish the repair to the RouteB snapshot branch and verify a fresh full Hosted run.

### 2026-10-23 A10 follow-on — QA manifest and report readers

A further audit found that `qa/source_manifest.py`, `qa/wheel_manifest.py`, installed-wheel provenance parsing, and both M13 report analyzers only translated `RecursionError`; on Linux they could continue into generic schema checks. These readers now call the shared 512-level scanner before `json.loads`. The preceding reconciler repair has fresh Hosted confirmation: run `39924080730` passed Linux CPU, Windows CPU, and isolated JAX CPU. Expanded QA reader selector: **96 passed, 1 symlink-capability skip**, plus one expected duplicate-ZIP fixture warning. MyPy (227 files), Ruff, and the updated 606-file manifest pass; identity `c5d261fa08c2940c83476dcd8df19a69c5c7d0b195284d3a25388e01d00a87a8`.

Next: commit and push the expanded QA guard to the RouteB-only snapshot, then verify its Hosted Linux/Windows/JAX CPU run.

### 2026-10-23 B4 Native/Imagined pending-limit cancellation parity

Added a long/short native differential with a non-marketable limit order. Both paths retain the order without a fill for one bar; after cancellation, Native reports `CANCELED`, `ImaginedAccount` has no pending order, and both remain flat at unchanged equity. Focused case: **2 passed**; complete adapter smoke file: **64 passed**; Ruff pass. Updated 606-file manifest: `916c92fc521dd8f931b9f1581fe9e7254c1d945cc48bd21a69b0b66406538596`. Evidence: `docs/audit_evidence/20261023/b4_imagined_limit_cancel_native_parity_20261023.json`. Expanded QA JSON guard CI previously passed all hosted Linux/Windows/JAX CPU jobs in run `39926884607`; this B4 commit needs a fresh run.

Next: continue locally testable B4 lifecycle parity. OHLC-only inputs cannot establish exchange queue priority or exact intrabar sequencing.

### 2026-10-23 A10 L6 attestation report depth guard

The L6 release attestation loader strictly rejects duplicate keys and non-standard constants but did not preflight nesting on external gate reports or the trusted-producer registry. It now uses the shared 512-level scanner before decoding. An expanded MyPy check also found an untyped L6 `_reports` state; its dictionary type is now explicit. Joint L6/adapter regression: **70 passed**; MyPy `packages ci qa`: **235 files pass**; Ruff passes; 606-file manifest identity `1e71cd34c7f82a90deb092c86c6fb138b0dac8cb3d557114fcbb85c5f1ef2c8f`. Evidence: `docs/audit_evidence/20261023/a10_l6_attestation_json_depth_20261023.json`.

The preceding B4 cancellation snapshot has full Hosted confirmation: run `39928798741` passed Linux CPU, Windows CPU, and isolated JAX CPU. The L6 guard is the current pending CI change.

### 2026-10-23 A10 L6 standalone import correction

Run `39930562257` stopped in `validate-lane` before tests because L6's top-level import of the shared JSON scanner required RouteB packages before `uv sync`. The scanner is now imported only when strict JSON decoding runs; the early lane module imports successfully with no RouteB package installed. L6 JSON selector: **6 passed**; expanded MyPy `packages ci qa`: **235 files pass**; Ruff pass; manifest identity `e02e160f776ad42af694c3d50364ba8dffc793bfe05ee4461d39b5e2acdf959f`. Evidence records the initial pre-test CI failure and repair. New hosted validation remains pending.

**2026-10-23 RouteB 当前续做：** contracts 新增共享严格 JSON 解码器；data/research/orchestrator/compute_runtime/es_evotorch/rl_sb3/world_model 的包级解码 API 已转为薄封装，保留数据包自定义错误处理及世界模型显式非标准常量模式。实仓回归 **243 passed、4 skipped**，Ruff 全量通过，MyPy **227 files passed**；source manifest v3 重建并验证。下一步继续审查直接 `json.loads` 的恢复入口和 B4 生命周期等可本地解决项。当前变更只在 `D:\Program Files\NautilusTrader_RL_ES_RouteB` 独立仓库，未推送；Hosted 复验待后续提交。外部现金流认证、独立 Validation、真实盘口深度和独立新市场窗口仍需相应证据。

**2026-10-23 续做更新：** RouteB 全仓持久化 JSON 入口现统一委托 contracts 共享严格解析器，涵盖 A04 历史清单、ArtifactStore marker/index/manifest、Binance income 页、L2 manifest、canonical DTO、schema 默认值及 error envelope；保留 data/world-model 常量诊断、WFO `JSONDecodeError` 与 worker 错误文本兼容。相关组合回归 **389 passed、4 skipped**（CUDA-only 与当前解释器未安装 JAX），Ruff 全仓通过，MyPy 227 files 通过，`git diff --check` 通过；source manifest v3 已重建并自验。下一步继续 A01–A14 可本地修复项及 B4 Native/Imagined 生命周期对齐，并单独等待 Hosted Linux/Windows/JAX 验收；当前修改尚未提交/推送。真实账户现金流、独立 Validation/市场样本、历史盘口深度与登记前访问覆盖仍需对应外部证据。

**2026-10-23 B4 增量：**已补 Native/Imagined 有限深度多 bar 差分，long/short 都在三根连续 bar 后成交 0.2、保留 0.3 并维持部分成交状态；之后撤单清除剩余订单、保留已成交 0.2 仓位，Native 报 `CANCELED`；相关回归 99 passed、2 个 JAX 环境跳过；默认 PR 选择器 911 passed、4 skipped、662 deselected；Ruff 全量通过，MyPy 227 files 通过。结构化证据在 `docs/audit_evidence/20261023/b4_imagined_partial_depth_native_parity_20261023.json`。这里只确认固定 Native bar 配置下的合成成交能力，不代表真实盘口深度；当前 revision 的 hosted Linux/Windows/JAX CI 尚未验证。
**2026-10-23 JAX 验证补齐：** RouteB 独立 CPU learner venv（Python 3.12.10 / JAX 0.11.2）上的六个 JAX 专项测试文件通过 **98 项、0 跳过**。完整 WorldModel suite 收集到 `test_cli.py` 时因此专用 learner env 不含 Gymnasium 而停止；该缺项只影响 CLI 混合收集，没有把依赖加进隔离 learner venv。证据：`docs/audit_evidence/20261023/routeb_jax_isolated_cpu_regression_20261023.json`。CUDA 和当前源码的 Hosted job 仍需验收。
**2026-10-23 B4 adverse-bps 补齐：**修正 Imagined 市价单基准价（Native bar close）与方向性滑点/向外 tick 取整。非平盘 long/short Native differential 的 2 bps 成交价及权益匹配（1002.21 / 1001.79）。默认 PR selector **913 passed、4 skipped、666 deselected**；RouteB 隔离 JAX CPU **102 passed**；Ruff 全量、MyPy 227 files 通过。证据：`docs/audit_evidence/20261023/b4_imagined_adverse_bps_native_parity_20261023.json`。真实 spread/depth/queue、Hosted CI 及其余外部审计门槛仍待验证。
**2026-10-23 构建分发层复验：** 当前 607-file source identity `2d73d2d118be32099b121cd51ba0cd5e8d71f0100408df236e9e3b9fda1770b9` 通过 manifest 自验；从该源构建的 12 wheels + 12 sdists 与每个包的 Python 源逐项哈希一致。全新隔离 Python 3.14.7 环境安装 12 wheels 后，全部模块及 6 个入口均从 wheel 环境导入。wheel 环境定向 strict-JSON/B4 回归 **114 passed、2 skipped**；默认 PR 选择器 **913 passed、4 skipped、666 deselected**。第三方依赖经临时 `.pth` 由 RouteB app venv 提供，故验证的是本地 wheel 内容、来源和功能，不是离线完整依赖安装。证据：`docs/audit_evidence/20261023/a10_current_wheels_strict_json_b4_20261023.json`。下一步继续 A01–A14 源码审计与 B4 生命周期对齐；当前本地源仍未提交/推送，Hosted Linux/Windows/JAX 复验及外部研究/验证门槛尚未关闭。
**2026-10-23 B4/F7 续做：** ImaginedAccount 现于入口拒绝布尔数值配置、目标仓位和限价，防止 `True` 被当作 1 或延迟到成交阶段抛 `Decimal.InvalidOperation`；新回归 8 passed，适配器+WorldModel 回归 113 passed、2 个 JAX-only skipped，Ruff/MyPy 通过。当前 607-file source identity `a949604a39cfe03ced052ce93b665d2123c15b22146270d910e88eac3fe86def` 的 12 wheels 全部源码匹配，隔离安装 smoke 覆盖 12 包/6 入口，默认 wheel PR selector 913 passed、4 skipped、674 deselected。当前 90 日合成 Native 回放 129,597 决策步 online-summary/lean 轨迹一致，RSS 峰值增量约 191.2/192.6 MB。证据：`docs/audit_evidence/20261023/b4_bool_numeric_and_f7_20261023.json` 与 `f7_current_synthetic_90d_profile_20261023.json`。下一步审查 Native 实际支持的 GTC 市价生命周期并继续 A01–A14 本地可处理项；GTD/真实限价排队不属于当前 Native adapter 支持合同。Hosted Linux/Windows/JAX 与外部数据、独立 Validation、市场授权/深度及历史试验等门槛仍未关闭。
**2026-10-23 A04 数据读入口修复：**低层 `read_registered_train_window()` 原先接受带 `test`/`holdout` 字段的配置，并对畸形 `evaluation`（包括布尔时间戳）跳过重叠验证。3 个红测复现后，已改为在 ledger reservation 和 payload decode 前拒绝正式范围字段、畸形边界及 Train/evaluation 重叠。研究访问/WFO 源码和 wheel 回归各 **159 passed**；默认源码 PR selector **913 passed、4 skipped、677 deselected**；12 wheel 全部源码匹配，12 包/6 入口 provenance、Ruff、MyPy 通过。当前 607-file identity `8ac14ea4f433ea58d30eacabc5907b7c7f332540314d1ff576100f2f39269b3a`。证据：`docs/audit_evidence/20261023/a04_train_scope_fail_closed_20261023.json`。全程只用临时路径与 mocked reader，没有读取项目 registry 或任何市场、Train、Validation、Test/Holdout payload。下一步继续 A01–A14 本地审计与 Native 支持的 B4 lifecycle；真实历史访问完整性、外部授权/验证及 Hosted CI 仍待完成。
**JAX 专项复验：**RouteB 隔离 CPU learner（Python 3.12.10/JAX 0.11.2）六个模型专项文件通过 **110 passed、0 skipped**；app Python 3.14 的 4 个 skip 仍分别是 1 项 CUDA-only 与 3 项 JAX-only，已由对应环境覆盖。RouteB JAX 证据已更新。下一步提交当前本地证据后，只向独立 RouteB GitHub 仓库发布分支并请求 Hosted Linux/Windows/JAX 验收；不触碰主项目备份。
**2026-10-23 A04/A09 市场身份隔离：**WorldModel registered Train replay 原先允许调用参数的 `instrument` 与 `config.environment.instrument` 不一致，导致已登记的一个标的可被用于投影另一标的。现于调用 reader 前精确校验标的身份；红测确认 mismatch 在任何 reader 调用前拒绝。CLI/ResearchAccess/WFO 源码和 wheel 联合回归均 **217 passed、2 JAX-only skipped**，12 wheels 全匹配当前 607-file identity `09bc47996ad1e8f36e2e675f6f46238b0f37723e9efa48b45f868af47808a070`，provenance 12 包/6 入口、Ruff 和 MyPy 通过。前一提交 `9eb379b` 的 Hosted run `39958369168` Linux/Windows/JAX CPU 全通过；此新修复待新 CI。证据：`docs/audit_evidence/20261023/a04_train_instrument_identity_20261023.json`。当前下一步：推送该修复后查验 Hosted CI，再继续 A01–A14 余项；所有数据均为合成测试，无正式研究窗口访问。