# 0006 S9：14-token 动作接入 Craftium

日期：2026-11-15
状态：动作通道完成并验证；任务/奖励尚未接

## 背景

0005 把输入通道查清楚后，这里落地成代码：一个与 `DummyNavEnv` 同接口的 Craftium 环境，
让训练脚本只换任务工厂就能跑真引擎。

## 实现

`craftium_env.py::CraftiumCedEnv`

| 接口 | 行为 |
| --- | --- |
| `reset(seed)` | 硬重置 Craftium（服务端进程重启，约 1.6 s）；维护跨 episode 的逻辑 tick 偏移 |
| `step(tokens)` | 写绝对朝向 → 发按键 → 等到新观测帧；返回 `obs, reward, term, trunc, info` |
| 观测 | ced_obs 的 2568 维帧，不是像素 |
| `info` | `tick`（逻辑、跨 episode 单调）、`episode_id`、`frames`（本次决策消耗的帧数）、`yaw/pitch`（目标角度） |

动作映射（14 token → 引擎）：

| token | 通道 |
| --- | --- |
| 0–3 fwd/back/left/right | Craftium Dict 的 forward/backward/left/right |
| 4–5 jump/sneak | jump / sneak |
| 6 sprint | aux1（**实测无加速效果**，保留映射） |
| 7–9 attack/use/drop | dig / place / drop |
| 10 hotbar（9 类） | 1..9 → slot_N；0 = 不按 |
| 11–12 yaw/pitch（各 121 桶） | 绝对角度写入 `look.txt`，Lua 用 `set_look_horizontal/vertical` 应用 |
| 13 hold（4 类） | 新观测到达后额外重复该动作的帧数（0–3） |

朝向是相对量累加：每次从**上一帧观测的 self 段**读出当前 yaw/pitch，加上 token 的
±90°，pitch 夹在 ±89.4°。这样不依赖任何客户端状态，也不会漂移。

## 验证（`scripts/check_craftium_env.py`）

```
reset ok: obs (2568,), look [0. 0.] deg
yaw token 120: +0.0 -> +90.0 deg (delta +90.0, expect +90)
pitch token 0: +0.0 -> -89.4 deg (clamped)
forward x10: displacement 1.75 nodes, dir -90.0 deg, look yaw +90.0 deg
forward+backward x5: displacement 0.000 nodes
frames per decision: 3
CRAFTIUM ENV OK
```

- 一次决策消耗 **3 帧**（58 Hz 观测 / 200 Hz 步进），与 0005 的 3.4 一致。
- 前进方向 = `-yaw`（`atan2(dx,dz)` 约定），与观测 self 段一致。
- forward+backward 精确抵消为 0。

## 踩到的一个竞态

`look.txt` 是在两次服务端 tick 之间写的，但 Python 等"下一个 tick"时，那一帧可能
还没应用新朝向，于是观测和动作错配。现在的做法：拿到帧后核对 self 段的 yaw/pitch
与目标是否一致（阈值 0.05 rad），不符就再等一帧，最多两次。代价很小，只在临界时刻多等一帧。

## 还没做的

1. **任务与奖励**：`ced_obs_set_reward/termination` 还没有任何任务脚本调用；
   现在环境返回的 reward 是 Craftium 自带的（ChopTree 砍树 +1），termination 是玩家死亡。
   做 T0/T1 之前要先在 Lua 里写距离奖励与到达判定。
2. **episode 语义**：硬重置会让服务端 tick 归零，Python 用逻辑 tick 偏移修好了；
   但 `episode_id` 目前是 Python 侧自增，Lua 的帧头里仍是 0，接 Evaluation 时要统一。
3. **未实测的键**：place / drop / slot / inventory / zoom 只做了映射，没有验证效果。
4. **跳跃**：单帧即可起跳；跳跃高度曲线在 0005 里有记录，但还没做成 token 的时序约定。
