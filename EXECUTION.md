# 执行手册（D0–D10）

这份手册是给你在终端里照着敲的。每一节都写清楚：目的、命令、期望输出、失败怎么办、什么时候算完成。

**分工**：第 1 节列的三项是我在本会话里能做的（已做）；从 S0 开始的每一条都需要在你的 Ubuntu 上有 shell 权限，我这边跑不了。

---

## 0. 本会话已完成的部分

| 产物 | 位置 | 状态 |
| --- | --- | --- |
| 观测/动作 schema（2568 维 / 14 token） | `schema_v0.py` | 已写，附带校验器 |
| 合成环境 + Craftium 适配器骨架 | `envs.py` | `DummyNavEnv` 完整；`CraftiumEnv` 只有接口 |
| 三档模型（GRU / Transformer / CED） | `models.py` | 已写，参数对齐到 ±5% |
| 行为克隆训练 + 贪心评测 | `train.py` | 已写，含 `--smoke` 门禁 |
| 三组不变量测试 | `tests/` | 已写；因果性泄漏会直接报错 |
| 环境准备与冒烟脚本 | `scripts/` | 已写 |
| Luanti 结构化观测 mod | `ced_obs.lua` | 参考实现，**未验证** |
| 训练配置基线 | `configs/base.yaml` | 已写 |
| 决策记录 | `docs/decisions/` | 已写 0001 |

**未做的事，也不会假装做了**：以上 Python 代码没有在任何机器上执行过。第一次运行预计要修 1–3 处小问题。

---

## 1. 总览

| 步骤 | 内容 | 预计 | 出口判据 |
| --- | --- | --- | --- |
| S0 | 机器核对 | 20 min | 驱动、CUDA、磁盘、Java 版本全部落盘 |
| S1 | 仓库落盘 | 10 min | `ced_starter/` 内容进 git |
| S2 | bootstrap | 20 min | 三组测试全绿 |
| S3 | 三档冒烟 | 10 min | 三个 `SMOKE OK` |
| S4 | 记录基线 | 10 min | `runs/` + `docs/decisions/0002` |
| **G1** | **门禁** | — | 冒烟全部通过，且 CED/Transformer 参数比在 0.8–1.25 |
| S5 | 装 Craftium | 40 min | `Craftium/ChopTree-v0` 能随机跑 |
| S6 | 单环境节拍 | 30 min | 20 Hz 稳定，reset < 2 s |
| S7 | Lua 观测 spike | 4–6 h | 文件按 tick 更新 |
| S8 | 帧校验 | 30 min | `validate_obs` 通过 |
| S9 | 接 `CraftiumEnv` | 1–2 d | 训练脚本零改动跑真环境 |
| S10 | 采数据 + B0 | 1 d | loss 下降且成功率 > 0 |
| **G2** | **门禁** | — | 5000 步数据上 B0 能学 |
| S11 | 扩到 4 环境 | 1–2 d | 吞吐 ≥ 单环境 3 倍 |

---

## 2. 逐步骤

### S0 机器核对

```bash
lsb_release -a
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /
free -g
java -version 2>&1 | head -1
```

期望：`16376 MiB` 左右显存；磁盘至少剩 100 GB（数据配额 80 GB + 系统）。
失败：驱动缺失就先 `ubuntu-drivers autoinstall` 再重启，别在缺驱动的情况下装 PyTorch。

**完成**：把四条输出贴进 `docs/decisions/0002-baseline.md`。这张表以后每次性能结论都要引用。

---

### S1 仓库落盘

```bash
mkdir -p ~/work/minecraft-ced-agent && cd ~/work/minecraft-ced-agent
git init
# 把 ced_starter 里的文件拷到对应位置
cp -r <starter>/schema_v0.py <starter>/envs.py <starter>/models.py <starter>/train.py .
cp -r <starter>/tests <starter>/scripts <starter>/configs .
mkdir -p env/lua/ced_obs docs/decisions runs
cp <starter>/ced_obs.lua env/lua/ced_obs/init.lua
printf '.venv/\nruns/\n__pycache__/\n*.pt\n' > .gitignore
```

**完成**：`git add -A && git commit -m "P0a scaffold"`。所有后续实验都从这一刻起可追溯。

---

### S2 bootstrap

```bash
chmod +x scripts/*.sh
scripts/bootstrap.sh
```

期望：结尾打印 `== bootstrap ok ==`，前面五组测试各自 `passed`。

失败对照：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `torch.cuda.is_available() == False` | CPU 版 torch | 重装 cu126 源；别用 conda 混装 |
| `ModuleNotFoundError: schema_v0` | 在 tests/ 里直接跑而不是在根目录 | 用 `python tests/xxx.py`，脚本里已加 sys.path |
| cu126 下载 404 | 源版本变更 | 退到 cu124、cu121 |
| venv 建不了 | 缺 python3-venv | `sudo apt install python3-venv` |

---

### S3 三档冒烟

```bash
scripts/smoke.sh 300
```

期望每档：参数规模、loss 逐段下降、`SMOKE OK`。
参考量级：GRU 约 3M、Transformer 约 18M、CED 约 18.6M，CED/TF 比值 0.8–1.25。

失败对照：

| 现象 | 大概率原因 |
| --- | --- |
| 三个都不动 | schema 维度或 numpy/torch 版本 |
| GRU、TF 过，CED 挂 | CED 解码器的 `tok_pos` 切片或 mask |
| loss 不降 | 教师动作与观测不同步（检查 `rollout_teacher` 的切片） |
| 内存一直涨 | 训练循环里持有计算图，检查 `zero_grad(set_to_none=True)` |

**这不是形式检查**：三个 `SMOKE OK` 之前，不要碰 Craftium。

---

### S4 记录基线

新建 `docs/decisions/0002-baseline.md`，抄入：硬件四行、三档的 tail loss、贪心成功率、每档耗时、RSS。
以后任何"变快了/变准了"的说法，都要和这张表比。

---

### G1 门禁

- 三档冒烟全过
- CED/Transformer 参数比在 0.8–1.25（`test_param_parity` 会自动卡）
- 因果性测试通过（`test_causality`）
- 基线数字已落盘

任一不满足，停在 S4，不要往下走。

---

### S5 安装 Craftium

```bash
cd ~/work
git clone https://github.com/perceptual-ai/craftium.git
cd craftium && cat README.md      # 以仓库当前说明为准，下面只是预期路径
pip install -e .                  # 或按其 README 用 wheel / Docker
python -c "import gymnasium, craftium; print('ok')"
```

跑它自带示例（随机策略）：

```bash
python - <<'PY'
import gymnasium as gym
env = gym.make("Craftium/ChopTree-v0").unwrapped
obs, _ = env.reset()
for i in range(200):
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    if term or trunc:
        obs, _ = env.reset()
print("rollout ok", obs.shape)
PY
```

失败：注册名不对就去 `craftium/__init__.py` 里 `grep register`；渲染相关报错时确认走的是 headless。

---

### S6 单环境节拍

目标不是"能跑"，是"能按 tick 稳定跑"。

```bash
# 起一个无渲染实例，把假人放进去，测 10 分钟
python - <<'PY'
import time, gymnasium as gym
env = gym.make("Craftium/ChopTree-v0").unwrapped
env.reset()
t0, n, worst = time.time(), 0, 0.0
while time.time() - t0 < 60:
    s = time.time(); env.step(env.action_space.sample()); worst = max(worst, time.time()-s); n += 1
print(f"{n/60:.1f} steps/s  worst_step={worst*1000:.1f}ms")
PY
```

期望：单实例 ≥ 200 steps/s（远超 20 Hz），最差单步 < 20 ms。
失败：低于 100 steps/s 时先查是不是在渲染；关掉 shadow、降分辨率、确认 headless。

---

### S7 Lua 观测 spike（关键路径）

1. 把 `env/lua/ced_obs/init.lua` 放进世界目录的 `worldmods/`。
2. 用 `/player ced_bot spawn` 之类的命令生成假人（按你的 Luanti 版本调整）。
3. 观察 `worlds/<world>/ced_obs_ced_bot.txt` 是否每 tick 变大/更新：

```bash
ls -l --time-style=full-iso ~/.minetest/worlds/<world>/ced_obs_ced_bot.txt
```

期望：mtime 持续刷新，文件约 20 KB。

失败：光线投射太慢（单 tick > 5 ms）就减到 8×8 射线，或降到 10 Hz + 动作保持 2 tick。

---

### S8 帧校验

真机之前先用假服务端把契约测通（它不需要 Luanti）：

```bash
python tests/test_adapter.py
```

期望 `adapter tests passed`，耗时 1–2 秒。这一步测的是等待、超时、episode 边界、
原子替换——接真引擎时最容易出问题的四个地方。

然后校验 Lua 写出的真实帧：

```bash
python - <<'PY'
import os
from transport import ObsChannel
p = os.path.expanduser("~/.minetest/worlds/<world>/ced_obs_ced_bot.txt")
fr = ObsChannel(p, timeout=2.0).read()
print("ok", fr.tick, fr.episode_id, fr.obs.shape, "nonzero", int((fr.obs != 0).sum()))
PY
```

期望：能连续读到 tick 递增的帧，`2568` 维，非零元素几百个。
失败：字段数不对 = Lua 与 `docs/protocol.md` 不一致；维数不对 = 回到 `schema_v0.py` 对偏移量。

---

### S9 接真服务端

`TransportEnv` 已经是通用实现，只要服务端按 `docs/protocol.md` 写帧，Python 侧不用改。
剩下的是**动作通道**：Lua 每 tick 读 `act.txt`，把 14 个 token 映射成假人控制
（`core.set_player_controls` / Craftium 的控制器），并记录已应用的 tick。

判据：`python collect.py --task T0 --episodes 5` 能在真环境上产出 5 个分片，且
`read_episode` 回来 `validate_obs` 通过。

---

### S10 采数据 + B0

先用随机策略采 5000 步，确认数据能落盘、能读回、能训。
判据：B0 loss 从初值下降 ≥ 5%，成功率 > 0。

失败：loss 不动 → 查动作是否真的进了环境（打印前 10 步的动作与实际位移）。

---

### G2 门禁

5000 步数据上 B0 能学、数据可复现、环境不崩。不满足就停在 S10。

---

## 3. P1 数据管线（已提前写好，S3 通过即可用）

这套东西不依赖 Craftium：先用 T0 在合成环境里把"采集 → 分片 → 训练 → 评测"走通，
等 L1 接上以后只换任务工厂，命令不变。

```bash
python collect.py --task T0 --episodes 200 --out data
python bc.py --model gru --task T0 --data data --steps 2000 --eval-every 500
python collect.py --task T0 --out data --stats      # 随时看磁盘账本
```

判据：`bc.py` 结尾打出 `GATE PASS`。没有它就回去查数据，不要换模型。

**磁盘账本要当场读，不要沿用估算**：观测是 2568 维 × float16 = 5.1 KB/步，
比文档早期估的 1 KB 大五倍。`--stats` 会打印压缩后的真实 bytes/step，并按 8 环境 × 20 步/s
推算日增量。达到 `configs/base.yaml` 里 80 GB 配额的一半时就开始轮转旧数据。

三个新增的组件：

| 文件 | 作用 |
| --- | --- |
| `tasks.py` | T0–T6 注册表；T1–T6 在 L1 接入前调用会直接 `NotImplementedError`，不许在 dummy 上伪造 |
| `shards.py` | 一片一 episode、按 `seed % 10` 切分 train/val/test、LRU 按需加载（20 万步全进内存是 1 GB，不允许） |
| `evalx.py` | 3 种子 × N episode 的评测协议，输出成功率/步数/卡死率/非法动作率，种子方差 > 0.10 判为不可用 |

---

## 4. 记录规范

```
runs/<YYYYMMDD-HHMMSS>-<model>-<note>/   # 每次跑：日志 + ckpt + 一行 config 快照
docs/decisions/NNNN-<topic>.md           # 每个结论：背景 / 做法 / 数字 / 结论 / 是否推翻假设
```

规则：没有进 `runs/` 的实验不算数；没有数字的结论不算结论。

---

## 5. 停止条件

出现以下任一情况，**停下来写记录，不要继续调参**：

1. 三档冒烟有任一档不通过，且两小时内定位不到。
2. 教师自身到达率 < 90% —— 问题在任务定义。
3. B0 在 20 万步内 BC 成功率 < 60% —— 数据或标签问题，改架构只会掩盖它。
4. 8 环境并行时整机可用内存持续低于 8 GB —— 先降环境数。
5. 同一个 bug 修了三次还没好 —— 写下来，换最小复现。

---

## 6. 常见故障速查

| 症状 | 先查这里 |
| --- | --- |
| loss 是 NaN | lr、梯度裁剪、观测里的 inf（`validate_obs` 会拦） |
| CED 比 TF 慢很多 | `generate()` 只用于评测；训练走 teacher forcing |
| 成功率 0 但 loss 在降 | 评测用的 argmax 与训练分布不一致；看训练集的 token 准确率 |
| 显存缓慢上涨 | 缓存了整条轨迹的张量；每步 detach |
| 多环境结果串线 | 共享了 RNG 或 buffer；按 env_id 隔离 |

---

## 7. 下一步（P1 预告）

S11 之后进入任务阶段：`go_to_point` 用 A\* 教师采 20 万步，B0 先过门禁，再上 B1/B2。
任务定义与门禁在规划 v2 第 4 节，不在这里重复。

---

*本手册的命令除 S5 的安装方式外均为常规 Linux 操作；Craftium 的安装方式以其仓库当前 README 为准，我给的是预期路径。*
