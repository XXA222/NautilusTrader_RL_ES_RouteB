Warning: truncated output (original token count: 25880)
Total output lines: 508

# RouteB 审计整改：当前连续开发目标

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

继续审计发现 Train-inner Bar Base/Stress 诊断的 `filled_quote_notional` 仍漏乘冻结规则中的 `contract_multiplier`。现已要求订单汇总显式传入正有限 Decimal 乘数，并由每个历史工厂传入，报告同时记录该值。合成回归与冻结计划选择器共 **9 passed**，Ruff、py_compile、目标 MyPy 及 595-file source manifest 均通过，identity `28127481828a9f40b96de186643190c07e741fc8dfc6662e852338e5edcfc859`。证据：`docs/audit_evidence…20880 tokens truncated…remain external.


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