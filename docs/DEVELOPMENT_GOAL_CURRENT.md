# Codex 连续开发目标

本文件是可直接用于 Codex Goal 的当前执行目标。审计问题、依赖、状态和验收条件以[任务索引](DEVELOPMENT_GOALS_REVIEW_20261017.md)为准；真实结果及最近续做位置以[开发执行记录](DEVELOPMENT_PLAN_20261017.md)末尾为准。每轮开始重新核实现状，不把历史计划或审计基线当作当前代码状态。

**执行方式：** Codex 每轮自动选择一个或多个当前可实施的审计整改，直接修改源码并完成必要回归；验收后接着处理下一项，不等待逐项确认。只有当所有可实施工程项均已完成，或剩余项确实需要尚未满足的外部条件时，才暂停，并记录每项的阻塞原因、恢复条件和准确续做位置。文档更新用于记录实施结果，不能替代源码实现。

## 可直接交给 Codex 的目标

```text
持续在独立 RouteB 工作区 `D:\Program Files\NautilusTrader_RL_ES_RouteB` 中落实审计报告 `C:\Users\QX\Downloads\NautilusTrader-RL-ES_源码深度审计与后续开发建议_20261017.md` 的未完成整改（A01–A14）。工作区分支 `codex/routeb-ci-integrity` 从 RouteB 快照 `cb7a3ec9d542ed48b9f182acbaac330fc90b3177` 建立，不携带 RouteA 的 IL 包/worker；原混合工作区保持原样。这是一个连续实施目标：每轮都要从当前续做位置检查并完成可做的源码/工程工作，不能把“检查完”“提出计划”或“更新文档”当作完成。审计基线 `c24d1bc4a47584f91a4c4db93bf77117d4e8341c` 仅供历史对照。

每次启动以及每个工作包验收后都继续执行以下循环，直到没有依赖已满足且可独立实施的工程任务：

1. 先检查 `git status`、相关 diff 和当前源码；阅读本文件、任务索引及执行记录末尾。保留已有未提交工作，不 reset/clean、不覆盖无关改动。对索引中已完成或已有证据的事项，只核验本次工作所需部分，不重复实现或重跑无关检查。
2. 按任务索引优先级和依赖，选择最高优先、当前仍有明确实施动作的任务；先确认缺口确实存在，避免重复已通过工作。把需求落实为具体代码入口、预期行为和验收测试。真实现金流、独立 Validation/Train 授权、历史深度、阶段工件或 Hosted CI 等外部依赖，只阻塞依赖它们的验收；记录恢复条件后立即转做其它可独立实施的工程任务。
3. 直接完成最小必要源码修改和有意义的确定性回归；运行受影响测试及适用 Ruff/MyPy，改动范围需要时再运行集成检查。记录实际命令和结果。失败时保存可复现错误，区分 `FAILED`、`BLOCKED`、`REJECTED`；不要把未执行命令、部分运行或本机结果记作通过。
4. 涉及行情或研究数据时，读取/解码前核对登记、窗口、用途和冻结身份。只在已授权范围工作；不得访问未授权 Validation/Test/Holdout，不得通过更换目录、协议或快照重用已消费样本。不得推断缺失现金流、伪造深度、降低风险/清算门槛，或把计划订单、Train-inner、scenario-only、合成盘口、奖励/Fitness 当作真实成交或独立盈利证据。
5. 每完成一个工作包，追加到 `docs/DEVELOPMENT_PLAN_20261017.md`：缺口、修改文件与行为、真实验证命令/结果、证据路径、数据用途、未认证边界和下一游标；按需更新 `docs/DEVELOPMENT_GOALS_REVIEW_20261017.md` 状态，并让本文件保持为短入口而非复制历史日志。完成一项后自动开始下一项，不逐项停下来询问。
6. 不修改 NautilusTrader 上游源码。只有任务索引中所有当前可实施的工程工作均已完成必要实现/验证，且其余任务的状态、阻塞依赖、恢复条件和准确续做位置均已记录后，才结束目标。策略盈利、独立经济认证或外部 CI 不是工程工作的替代物，也不能在证据缺失时宣称完成。
```

## 续做位置

执行记录最新条目为 **2026-10-18 RouteB WorldModel B0 fail-closed guards**。WorldModel 源窗口投影要求 1m 对齐且与完整时间栅格逐点一致，并对缺失/迟到的 K 线可用时间 fail-closed；S0/S1/S2 基线要求 replay 与冻结 Fold 的 dataset hash 一致，并仅在 Train、Validation warmup、Validation score 各段内检查连续性（允许折间 purge/embargo 空档）。相关测试 **17 passed**，Ruff 与定向 MyPy 通过。B0 真实数据和 Validation 证据仍阻断，因此不启动 B2。稍早 M0.2 已修复 source manifest/CI/wheel coverage：manifest 覆盖整个 `qa`（排除生成报告）并规范 UTF-8 文本换行为 LF；12 wheels `all_match=true`，全部包与 7 个入口 smoke 通过。Windows 完整套件此前 **767 passed、5 CUDA-only skipped、53 warnings**；本轮 Linux 完整锁定同步因大体积 CUDA 依赖下载停滞且 WSL 返回 `Wsl/Service/E_ACCESSDENIED`，已停止这次隔离同步、未产生验证结果。M0.1 真实现金流、M0.2 clean Hosted Linux/Windows CI、M1.2 独立 Validation、M1.3 兼容阶段工件、M2.3 真实深度、M3.2 新授权独立市场窗口与 A09 仍有各自外部门槛；其余依赖已满足的工程项继续从任务索引核验。

外部门禁按执行记录保留：真实账户现金流来源与长窗对账、独立 Validation、兼容的 E1/R1/E2 阶段工件、真实历史深度、新授权独立市场窗口、干净交付 Hosted CI，以及 A09 标的/持仓/杠杆扩展。它们只限制各自依赖的验收，不得成为停止其它代码工作的理由。开始新一轮时必须复核这些状态是否变化。
