# RouteB 1m/1h/1d 时间轴真值表

本文把审计 RB-02、RB-04 与 M1 要求的时间坐标冻结为可审阅契约。所有时间戳均为 UTC Unix 纳秒；`close_ts` 是 K 线结束边界，`available_at` 是整根 K 线实际可参与决策的最早时刻，两者不得互换。该文档说明的是当前实现的合成/协议语义，不是市场数据校准结论。

## 时间轴真值表

| 时钟/步骤 | 输入区间与完整条件 | 可见/决策屏障 | 状态语义 | 影响的预测目标 |
|---|---|---|---|---|
| 1m 观测 K 线 | UTC 半开区间 `(close_ts-1m, close_ts]`；OHLCV 在 `available_at` 才可读取 | 仅当 `available_at <= decision_ts` 才可进观测前缀；前缀最后一根必须正好闭合在 anchor | Minute RSSM 的 posterior `z_t^post` 已吸收 anchor 上的真实已知特征 | `z_t^post` 上的 next-minute head 预测下一根 1m K 线 `x_(t+1)`，不预测已经观察的 `x_t` |
| 1h 聚合柱 | UTC 小时桶必须恰有 60 根连续 1m 柱；桶首/尾 `close_ts` 分别为 `hour_close-59m` 与 `hour_close` | 仅在所有组成分钟柱均 `available_at <= decision_ts` 后可见；否则不产生该小时特征，index 为 `-1` 或维持上一个可见柱 | Hour RSSM 后验只在完整柱可见时更新；当前可见索引必须指向 `available_at <= decision_ts` 的最新柱 | 在事件屏障更新后的慢状态只影响屏障后的 minute 预测，不回写/重算屏障前的预测 |
| 1d 聚合柱 | UTC 日桶必须恰有 1,440 根连续 1m 柱；首/尾时间边界与整日桶一致 | 仅在全日最后一根及所有组成分钟柱均可用后可见；不能以未闭合日柱填充 | Day RSSM 后验遵循同一完整性和可用性规则；没有完整日柱时不得伪造先验观测 | 新日状态只能影响它的 availability barrier 之后的预测 |
| 训练一步监督 | 在时间 `t` 输入分钟特征 `x_t`，`dyn.loss` 返回由该输入构成的后验特征 `z_t^post` | 慢尺度索引由每根分钟柱的 `available_at` 因果投影；Train scaler/score mask 冻结 | `next_minute_mean/scale(z_t^post)` 是一步条件分布参数 | 标签严格为下一分钟编码 `x_(t+1)`；损失按目标分钟的 Train score mask 计入 |
| rollout 首步 | 使用 anchor 时刻已观察的 minute/hour/day posterior 初始化状态 | 所有 observed prefix 必须 `available_at <= anchor`；未来柱值不进入初始化 | 首个输出使用 anchor 的 posterior state；推进一次 RSSM 得到的 prior state 单独用于下一次输出 | 第 1 个生成分钟对应 `x_(anchor+1m)`；首步 head 必须与训练的一步 posterior head 一致 |
| rollout 后续步 | 先为本分钟决策输出分布并采样/解码一根合法 OHLCV；下一步再消费已采样路径反馈 | 决策时刻严格按分钟递增；拒样必须计数且受 per-bar/总预算约束 | 已生成分钟进入 minute state；完整小时/日聚合仅在预注册 UTC event 且 availability delay 到期后更新 slow state | 当前新生成分钟不能通过自身聚合反过来影响其自身预测；新慢状态影响下一次及之后的预测 |
| E3 / E4 对照 | 两臂共用同一 Train checkpoint、同 anchors、seed、路径数、horizon、预算与上下文 | 共享冻结的注册 Fold reader receipt；仅允许已授权 Train+Validation replay，绝不读 Test/Holdout | E3 固定 observed slow posterior；E4 在 event 屏障后按该路径生成的完整慢柱推进 slow posterior | 唯一处理差异是慢状态是否在未来事件后按路径反馈；E4 不得提前看到未来慢柱 OHLCV |

## 事件顺序示例

设 anchor 为 UTC 12:00 的分钟 close，hour/day 聚合分别完整闭合在 12:00，且 `available_at` 等于 close。anchor 的 posterior 预测 12:01–12:02 分钟桶；预测 12:01 时只能使用 anchor 已完整可用的慢柱。若之后生成的 12:01–13:00 分钟路径闭合出 13:00 小时柱，hour posterior 最早在 13:00（或固定延迟后的第一可用 decision）更新，此更新只能影响其后的 minute head。整日柱同理，只有 1,440 分钟路径完成并到达其可用屏障才进入 day posterior。若实际 feed 延迟 1 或 2 分钟，事件逐分钟顺延；在 delay 到期前的输出必须与无该事件的输出逐数组相同。

## 当前实现与验证锚点

- UTC 分桶、完整柱条件、实际 availability 与未来事件日历：`packages/world_model/src/rles_world_model/routeb/clock.py` (`MultiScaleClock`).
- Train score-only scaler、slow index 因果投影和 anchor 裁切：`packages/world_model/src/rles_world_model/routeb/hierarchy_dataset.py` (`build_train_hierarchy`, `build_inference_hierarchy`).
- 训练 posterior head 对齐 `x_(t+1)`：`packages/world_model/src/rles_world_model/routeb/jax_hierarchy.py` (`loss_fn`, `initialize_rollout_f2`, `rollout_f2_step`).
- 按 sampled minute OHLCV 逐步反馈并核对完成慢柱：`packages/world_model/src/rles_world_model/routeb/hierarchical_paths.py` (`generate_recursive_hierarchical_f2_paths`).
- 数值验证：`tests/world_model/test_routeb_hierarchical_paths.py` 中 availability barrier/延迟和反馈回归；`qa/world_model/routeb_b3_cpu_smoke.py` 检查首步 posterior head、advanced prior 区分、0/1/2 分钟延迟屏障、递归 OHLCV 聚合守恒与 E3/E4 事件后分歧。最新 smoke 结果在 `docs/audit_evidence/20261023/routeb_b3_posterior_prior_event_delay_20261025.json`。

## 边界

该真值表不证明真实市场上 prior/posterior 误差可接受、open-loop 分布校准、路径的经济价值或执行撮合 parity。相关 RB-10/RB-12/RB-11 门禁仍分别要求新鲜授权 Train+Validation、真实分布校准及完整 Nautilus 原生经济差分；Test/Holdout 仍严格禁止。
