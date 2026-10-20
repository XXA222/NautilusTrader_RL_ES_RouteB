# Route B 世界模型源码 v2.2 — 第二轮工程加固（2026-10-18）

> 此发行是**增量源码包**，对应 `XXA222/NautilusTrader-RL-ES` 的固定快照
> `80194a587444bf4a11f06f6e273723e5754f2483`。**不包含任何原项目已有代码**，
> 不包含 NautilusTrader 或 DreamerV3 上游源码、训练数据、账户 API 或训练权重。
>
> 本版会覆盖 **上一版增量源码包**中的 `rles_world_model/routeb/` 及其新增测试，
> 但不覆盖仓库已有 `rles_world_model/{cli,upstream,baselines,contracts}.py` 等文件。

## v2.2 的关键修复

1. **B2 训练目标时序**：改为以 `mask[t+1]` 判定 next-bar NLL，保留 burn-in 后第一个目标及最后一个真实目标；不再误用 `mask[t]` 造成有效样本丢失。
2. **正确的累计收益**：`evaluation.return_paths()` 必须传入可见锚点 `previous_close`，逐根计算 close-to-close 回报，计入开盘跳空，并校验 OHLCV 几何。
3. **可靠的中断恢复接口**：checkpoint v2 保存 JAX params、Optax optimizer、step/seed、训练数据与冻结 scaler 身份，同时验证 PyTree 结构/数组结构。允许续训增长总 `steps`，但拒绝改变算法超参或随机种子。崩溃时两文件发布不构成事务回滚，只会 fail-closed。
4. **训练/推断身份闭环**：Train NPZ 元数据与 checkpoint 绑定标准化器指纹。预测请求 v2 同时记录上下文 SHA、原始完成时间与可用时间，worker 在推断前复查。
5. **双 Python worker 防错**：worker 返回预测产物 SHA，client 检查产物存在、SHA、张量形状和 OHLCV 几何；对失败结果不静默继续。
6. **三级因果样本接口**：新增 `hierarchy_dataset.py`，形成 1m/1h/1d 的训练坐标与慢状态可见索引，分钟编码与 B2 的首 bar 锚点语义保持一致。
7. **F2 边界与想象执行**：缺历史前缀导致整点/日线无法组成完整合成 bar 时拒绝。账户禁止反向回放/倒填动作；限价单在下一 bar 开盘即市价可成交的情况下按 taker 场景费率计费，触价在 bar 内仅按简化 maker 假设。
8. **验证**：本运行环境 Python 3.13.5 + numpy/pytest + JAX 0.9.0.1 的 **35 项确定性测试全部通过**，涵盖参数/Optax树形式序列化（使用轻量树）、NPZ篡改、worker 请求等；Python 语法检查通过。

**重要限制**：本机已恢复固定 DreamerV3 checkout，并使用锁定的 Python 3.12.10 + JAX 0.11.2 + Ninjax/Optax 环境通过了 B2 合成数据训练、checkpoint 中断恢复、预测及双 Python worker 分布导出的端到端 CPU smoke。该运行只证明工程机制，不代表真实行情拟合或 E2 预测技能。三尺度 B3 JAX 优化器仍未通过运行验证。升级后的 QA CPU smoke 位于 `qa/world_model/routeb_b2_cpu_smoke.py`，包含“训练 2 步＋续训到 4 步”与“从头训练 4 步”的**逐个参数和优化器数组完全一致**检查。

### 升级影响（必须注意）

- `routeb.forecast-request.v1` → **v2**（新增 scaler/context SHA 和 K 线可用时间序列）；**不兼容旧 v1 请求**，需调用本版 `prepare_forecast_context()` 重新生成。
- `routeb.jax-checkpoint.v1` → **v2**（PyTree 结构与 scaler 元数据绑定）；旧研究检查点不能冒充新模型，应重新训练或显式实现经过验证的迁移工具。
- 预报上下文与模型权重不能跨不一致的 scaler 使用；任何 Validation/Test/Holdout 读取授权仍由原项目的 ResearchRegistry 负责，worker **不提供**独立数据访问 API。
- `return_paths` 不再允许省略 `previous_close`。这是一项故意的 fail-closed API 变化。
- 独立 `hierarchy_dataset.py` 是 **B3 张量构造入口**，不是已经训练成功的 B3；`jax_hierarchy.py` 仍为未完成 JAX 运行审计的研究候选。

## 测试

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'packages\world_model\src')
python -m pytest tests\world_model\test_routeb_additive.py tests\world_model\test_routeb_quality_v2.py -q
python -m compileall -q packages\world_model\src\rles_world_model\routeb
```

在指定的 JAX 隔离环境运行 **合成数据 CPU 深度验证**（不访问真实行情）：

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'packages\world_model\src')
$env:JAX_PLATFORMS = 'cpu'
& '.tmp\world-model-cpu-venv\Scripts\python.exe' qa\world_model\routeb_b2_cpu_smoke.py `
    --checkout .tmp\dreamerv3-route-b --output-dir .tmp\routeb-b2-v22-cpu-smoke
```

## 模块明细与状态

| 子系统 | 主文件 | 状态 |
|---|---|---|
| B2 数据和编码 | `market.py`、`codec.py`、`sequences.py` | 本地确定性测试 PASS |
| B2 概率预测评价 | `evaluation.py` | 本地确定性测试 PASS |
| B2 JAX 训练 | `jax_model.py`、`jax_train.py` | 锁定 JAX CPU 环境下合成训练、恢复、forecast smoke PASS；真实 Train/Validation 研究验证受 B0 门禁阻塞 |
| B2 IPC + checkpoint | `worker_cli.py`、`worker_client.py`、`checkpoint.py`、`integration.py` | 真实双 Python 子进程和 float32 NPZ 分布工件端到端 smoke PASS；仅合成未来标签 |
| B3 多尺度数据 | `clock.py`、`hierarchy.py`、`hierarchy_dataset.py` | 因果张量单测 PASS |
| B3 RSSM 候选 | `jax_hierarchy.py` | 仅研究性模型，未在固定 JAX 环境运行 |
| F2 未来周期事件 | `synthetic_feedback.py` | 确定性边界测试 PASS，不含真实 B3 state feedback |
| B4 想象账户 | `imagined_account.py` | 简化情景单测 PASS，经济结果绝不可替代 NautilusTrader |

详细缺陷、验证条件与待办见 `docs/ROUTEB_V22_CODE_AUDIT.md`。

---

# Route B — World Model Additive Source Pack (2026-10-18)

## 交付对象与版本

- **基线仓库**：`XXA222/NautilusTrader-RL-ES`（用户指定分支 `codex/archive-20261018-development-snapshot`）。
- **固定基线 SHA**：`80194a587444bf4a11f06f6e273723e5754f2483`。
- **DreamerV3 上游固定 SHA**：`e01491fad6434b2245a3b8ca201dd7faedcc458c`。
- **不包含** NautilusTrader 源码、用户 RLES 旧源码、DreamerV3 代码、数据集、训练权重、API 密钥、任何已消费的验证样本。
- **新增文件不会覆盖** `rles_world_model/cli.py`、`upstream.py`、`baselines.py` 等现有实现。将本 ZIP 的 `packages/`、`tests/`、`qa/`、`configs/` **按目录合并**到目标开发快照即可。

## 组件完成度（请勿把预研代码当成实盘完成）

| 组件 | 文件 | 完成范围 | 本次验证 |
|---|---|---|---|
| B2 因果数据与 1m/1h/1d 事件 | `market.py`, `clock.py`, `hierarchy.py` | 完整桶、延迟可见、索引与缺口拒绝 | 本地确定性测试通过 |
| B2 价格表示、Train-only 标准化 | `codec.py`, `sequences.py` | OHLCV 几何、反变换、训练片段、NPZ 格式 | 本地测试通过 |
| B2 概率预测评价 | `evaluation.py` | empirical CRPS、区间覆盖、尾部 Brier | 本地测试通过 |
| B2 JAX RSSM learner | `jax_model.py`, `jax_train.py` | 固定 DreamerV3 RSSM + 独立金融预测头、KL、训练/预测算法 | **仅静态/语法检查；缺固定 JAX 环境的集成实测** |
| B2 参数、优化器完整恢复 | `checkpoint.py` | 参数/optimizer leaf + step/seed + 配置/数据身份、无 pickle | JAX 树序列化测试通过；中断训练数值等价待验 |
| 双 Python 环境通信 | `worker_client.py`, `worker_cli.py`, `integration.py` | 3.14 host → 3.12 worker；Train 已授权 NPZ | 静态/接口测试；真实子进程回归待验 |
| B3 三级状态模型 | `jax_hierarchy.py` | **研究候选网络损失**，无正式 B3 optimizer / F2 的训练集成 | **未通过 JAX 运行验证；默认不得启用** |
| B3/F2 合成时钟 | `synthetic_feedback.py` | 合成未来产生完整 1h/1d bar，禁止真实未来混入 | 本地测试通过；未接三层 RSSM inference |
| B4 想象账户原型 | `imagined_account.py` | 单净仓、market/limit、下根 1m 生效、撤单、费用/MTM | 本地单元测试通过；**非原生交易事实** |
| 验证 | `tests/world_model/test_routeb_additive.py` | 数据、时间、数值、检查点、账户验证 | 35 passed |
| JAX 专项 smoke | `qa/world_model/routeb_b2_cpu_smoke.py` | 真正执行两步梯度更新、模型恢复及短预测 | 用户固定隔离环境待执行 |

**特意不实现为“已完成”的事项**：三级 RSSM 的正式训练/消融、B3 真实 F2 世界状态联动、Actor-Critic 策略训练、挂单/原生账户逐事件 parity、历史手续费/资金费/清算认证、独立 Validation/Test/Holdout 授权，以及盈利结论。它们按 v2.1 设计文档后续阶段完成。

## 安装方式（增量代码）

在用户机器上：

```powershell
cd 'D:\Program Files\NautilusTrader_RL_ES'
git status --short
git rev-parse HEAD
# 确认 SHA 后，把 ZIP 内各目录按相同相对路径合入开发分支；
# 本 ZIP 不包含 NautilusTrader-RL-ES 整仓，亦不覆盖上游文件。
```

主环境 **Python 3.14.7**，世界模型独立环境 **Python 3.12.10 / JAX 0.11.2**。不要用主环境覆盖单独 JAX 锁。检查已有独立环境的版本与 `.tmp` 下 DreamerV3 固定 checkout；参考基线 `docs/world_model/README_STAGE.md`。

本 ZIP 的子模块无需修改根 `pyproject.toml`：它们落在已存在的 `rles_world_model` 包下面。`WorkerClient` 自动将包的 `src` 路径注入 JAX worker 的 `PYTHONPATH`；工作者不尝试在 3.12 中安装需要 3.14.7 的原项目分发包。

## 可复现的本地测试

本包基础单测仅需 Python、numpy、pytest（部分 checkpoint 测试还需要可导入的 JAX）；无需联网、行情或交易所凭据。

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'packages\world_model\src')
python -m pytest tests\world_model\test_routeb_additive.py -q
python -m compileall -q packages\world_model\src\rles_world_model\routeb
```

JAX 真训练 smoke：在**已安装并锁定**的 Python 3.12.10 worker 环境、**先确认 DreamerV3 checkout 的 SHA**以后：

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'packages\world_model\src')
$env:JAX_PLATFORMS = 'cpu'
$env:XLA_PYTHON_CLIENT_PREALLOCATE = 'false'
$wmPython = '.tmp\world-model-cpu-venv\Scripts\python.exe'
& $wmPython qa\world_model\routeb_b2_cpu_smoke.py `
  --checkout .tmp\dreamerv3-route-b `
  --output-dir .tmp\routeb-b2-cpu-smoke
```

该命令仅使用人工合成的数据来验证梯度/参数更新、checkpoint 和短期生成，**绝不**产生真实市场性能声明。严格 Windows 及 WSL 的入口位置依本机实际安装决定，不存在时先按 `docs/world_model/README_STAGE.md` 完成既有环境安装。

## 真实 Train 训练调用关系

**数据读取授权仍然由原项目负责**。不要用测试 JSON 回放代替注册的 Train 数据。建议主程序使用已存在的 `project_registered_train_replay()`：

```python
from pathlib import Path
from rles_world_model.routeb.integration import prepare_registered_train
from rles_world_model.routeb.worker_client import WorkerClient, WorkerConfig

# rles_config / window / marker_sha256 / label / instrument 必须是
# 项目研究登记器已经授权、冻结的 Train 范围，不得由示例数据生成。
train_npz, frozen_scaler, summary = prepare_registered_train(
    rles_config=rles_config, window=window, marker_sha256=marker_sha256,
    label=label, instrument=instrument,
    target_npz=Path('.tmp/routeb-authorized-train.npz'),
)
client = WorkerClient(WorkerConfig(
    python_executable=Path('.tmp/world-model-cpu-venv/Scripts/python.exe'),
    upstream_checkout=Path('.tmp/dreamerv3-route-b'),
))
report=client.train(
    sequences=train_npz, scope=label,
    checkpoint=Path('.tmp/routeb-b2-checkpoint.npz'),
    configuration=Path('configs/routeb/b2_cpu_demo.json'),
)
```

以上只是**调用结构示例**，`rles_config` 等来自真实研究注册系统；不能直接复制即声称已获授权。首次训练完成后必须在固定但未消费的预注册 Validation 上进行预测校准。数据源检查点、实际预测/经济门按 Route B v2.1 方案执行。

## 优先修复与研究顺序

1. 先在**已固定的 JAX 环境**运行上述 B2 CPU 专项 smoke，解决上游 API/依赖差异，并固定真实 optimizer checkpoint 中途恢复等价性；通过前不可在路线 B 上声称“B2 已验收”。
2. 在正式授权数据上训练 E2 单尺度模型，完成单步、多步 open-loop CRPS/覆盖率/极端风险评价，和已有 S0/S1/S2 及概率 GRU 比较；不得直接跳过 E1。
3. E2 有增量价值后再建立 `jax_hierarchy.py` 的正式训练、索引不泄漏证明与 E3/E4 消融；`synthetic_feedback.py` 仅提供 F2 时钟，不等于完整高尺度 imagination。
4. 与 NautilusTrader 做差分执行回归。`imagined_account.py` 不是原生撮合系统，**不包含资金费、强平、保证金阶梯、价差和 bar 内价格先后顺序**，因此不能用于最终收益认证。

## 编码边界说明

- `OHLCVCodec` 在约束空间中生成高低价，有异常尾部样本时明确拒绝，绝不静默输出负价格。对极端波动可能发生拒样，必须在报告记录拒样率，不能只保留“看起来正常”的样本造成选择偏差。
- `jax_model.py` 的未来预测从最后一个可见后验出发，不注入未来真实标签或交易动作；其损失包含一步预测、当前重构和 DreamerV3 自带的 KL。当前预测头采用独立正态分布，无法保证高维金融变量协方差正确；必须由路径校准对照验证。
- `jax_hierarchy.py` 是**待 JAX 验证的 B3 候选**，不是将 MTS3 源码移植进来，也不是成熟的三级 RSSM 实盘系统。
- JAX 模型默认 `float32`；GPU/WSL2 尚需单独准入测试，绝不能以现有 PyTorch RTX 5070 CUDA PASS 代替 JAX PASS。
- 该包按用户要求不引入哈希链、重复实现交易账户或者任何未经授权的实时 API。
