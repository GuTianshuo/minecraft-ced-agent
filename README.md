# ced_starter —— P0a 骨架

对应《Minecraft CED Agent 本地训练规划 v2》的 P0a/P0b 阶段。目标只有一个：
**在没有游戏的情况下，先把"观测 → 动作 → 数据 → 训练 → 评测"整条链路跑通**，
让后续接 Craftium 时，出问题一定是环境侧的，不是训练框架侧的。

## 文件

| 文件 | 作用 | 对应仓库路径 |
| --- | --- | --- |
| `EXECUTION.md` | **D0–D10 执行手册，先看这个** | `docs/` |
| `schema_v0.py` | 观测 2568 维 / 动作 14 token 的唯一真相，含校验器 | `env/schema/` |
| `envs.py` | `DummyNavEnv` 合成环境 + `TransportEnv` 真/假服务端适配器 | `env/` |
| `transport.py` | 游戏↔Python 的文件通道契约（帧/动作/控制） | `env/` |
| `mock_game.py` | 假游戏服务端，按 20 Hz 写帧，用于在无游戏时测通契约 | `env/` |
| `docs/protocol.md` | 传输契约 v1，Lua 侧实现以此为准 | `docs/` |
| `models.py` | B0 GRU / B1 Transformer / B2 CED，共用编码器与动作头 | `models/` |
| `train.py` | 教师采集 → 行为克隆 → 贪心评测 → checkpoint → 空转自检 | `train/` |
| `runlib.py` | 建模/损失/推理的共用实现，避免三份代码漂移 | `train/` |
| `tasks.py` | T0–T6 任务注册表，未实现的层级会显式报错 | `env/tasks/` |
| `shards.py` | episode 分片写入/读取/切分/按需采样 | `data/` |
| `evalx.py` | 3 种子评测协议与门禁判定 | `train/` |
| `collect.py` | 采集 episode 到分片，打印真实磁盘账本 | `data/` |
| `bc.py` | 从分片训练 + 定期评测 + GATE 判定（P1 主入口） | `train/` |
| `tests/` | schema、环境语义、模型因果性、端到端管线、传输契约 | `tests/` |
| `scripts/bootstrap.sh` | 一次性环境准备（venv + torch + 三组测试） | `scripts/` |
| `scripts/smoke.sh` | 三档冒烟，日志自动进 `runs/` | `scripts/` |
| `configs/base.yaml` | 超参与门禁阈值基线 | `configs/` |
| `docs/decisions/0001-environment.md` | 四项决策与其代价 | `docs/decisions/` |
| `ced_obs.lua` | Luanti 侧 16×16 射线结构化观测导出（未验证） | `env/lua/ced_obs/` |

## 怎么跑

只需要 Python 3.10+、numpy、torch 2.x（CPU 也能跑）：

```bash
python train.py --model gru --steps 300 --smoke     # 约 20–40 秒
python train.py --model tf  --steps 300 --smoke     # 约 1–2 分钟
python train.py --model ced --steps 300 --smoke     # CPU 上明显更慢，有卡就用卡
```

预期看到参数规模、loss 下降、RSS 变化、贪心成功率，最后打印 `SMOKE OK`。
`--ckpt r.pt` 会额外验证存盘再加载的往返一致。

## 状态声明

**这套代码我没有在目标机器上执行过**（当前会话只能读写文件，不能运行 Python）。
它是按 schema 推敲过的骨架，第一次运行大概率还要修 1–3 处小问题——把它当成
"省掉两小时搭架子"的起点，而不是已验证的库。

已按规范设计的部分：观测布局与维度、动作 token 分解、三种主干的接口一致性、
terminated/truncated 分离、checkpoint 往返、教师只走公平观测。

尚未实现、故意留空的：Craftium 真实 reset/step、动作到键鼠的解码、Lua 侧实体的
听觉与任务字段（目前填 0）、长程记忆、RL（只有 BC）。

## 已知坑

1. **上下文长度**：35 token/帧，32 帧约 1120 token。别按"512 步"配，那会到 1.8 万 token。
   `TransformerPolicy.max_frames` 卡在 32 就是为此。
2. **CED 的评测**：`generate()` 是 14 次解码器前向的朴素自回归，够评测不够训练。
   BC 用 teacher forcing，别在训练里调 `generate()`。
3. **DummyNavEnv 的射线**只在中间一行有值**，用来验证"射线通路没被丢掉"。
   它的几何没有真实意义，别拿它的指标当环境性能。
4. **Lua 侧每 tick 写文件**在 20 Hz、8 环境下约 3 MB/s 的写放大，先接受，
   如果 P0b 实测影响 tick 时间再换 UDP 或二进制。

## 一周落地

1. 装环境，确认 `python -c "import torch; print(torch.cuda.is_available())"`。
2. 跑三个 `--smoke`，把 loss 曲线记下来当基线。
3. 装 Craftium，跑通它自带的 `Craftium/ChopTree-v0` 随机策略。
4. 把 `ced_obs.lua` 塞进 Craftium 的 world mods，确认文件在按 tick 更新。
5. 用 `StructuredObsChannel` 读一帧，`validate_obs` 必须通过。
6. 把 `CraftiumEnv` 的 reset/step 填上，让 `DummyNavEnv` 与它共用同一套训练脚本。
7. 采 5000 步随机数据，跑 B0，确认 loss 会降 —— 这时才算 P0b 结束。

## 下一步（P1）

任务从 `go_to_point` 开始，教师用 A*。门禁是：B0 在 20 万步以内 BC 成功率 < 60%
就停下来查数据管线，**不要改架构**。
