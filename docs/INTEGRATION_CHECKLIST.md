# Codex 增量接入工作清单

目标：以 `80194a587444bf4a11f06f6e273723e5754f2483` 为基线，仅合入本 ZIP 的新增 Route B 代码；禁止覆盖现有 `rles_world_model` 核心实现或修改 NautilusTrader / DreamerV3 上游代码。

1. 检查 Git 当前提交和未提交变更；先建自己的开发分支，保留现有 worktree 内容。
2. 增量放入 `rles_world_model/routeb`、1 个定向测试、1 个 JAX CPU smoke 和 Route B 样例配置。
3. 验证现有 `tests/world_model/` 与新增测试，确保主 Python 3.14.7、L2 数据接口不退化。
4. 在锁定的 Python 3.12.10 / JAX 0.11.2 里运行新的 B2 CPU 专项训练；对潜在报错最小修复 `jax_model.py`/`jax_train.py`，禁止为了绕过错误修改 DreamerV3 上游。
5. 增加实际中途 checkpoint 中断恢复后的逐参数和下一更新数值等价测试；当前只有叶片序列化测试通过。
6. 验证 JAX F1 预测重复种子一致、动作独立、禁止真实未来与 Train-only 数据身份限制。
7. 采用项目现有 ResearchRegistry/`read_registered_train_window`，禁止另写可绕过授权的 Validation/Test/Holdout 读取器。
8. 预注册真实 Train/Validation 和 E0/E1/E2 比较后再开启算法效果验证；B3/B4 实现和生效严格从下一工作包开始。
9. 最终返回实际改动文件表、定向测试和结果、没跑通的项目和错误证据；不要给未测试或未具备经济身份的能力写 PASS。
