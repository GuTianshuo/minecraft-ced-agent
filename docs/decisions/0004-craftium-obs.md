# 0004 结构化观测接入 Craftium（S7 / S8）与动作通道的硬约束

日期：2026-11-15
状态：S7、S8 通过；S9 的动作通道方案需要确认

## 背景

0003 记录了 Craftium 装好、节拍达标，但原生观测是 64×64 像素。本文件记录把
`schema_v0` 的 2568 维结构化观测接进去之后的结果，以及接动作时撞到的一条引擎限制。

## S7 做法

1. `env/lua/ced_obs/`：由仓库根目录的 `ced_obs.lua` 派生。相对参考实现只改了四处：
   用 `get_connected_players()[1]` 找 agent（Craftium 只有一个真实连接玩家）、
   写到 `core.get_worldpath()/ced_obs.txt`、每个数值按 `%.4f` 格式化（默认数字转字符串
   最多 14 位有效数字，一帧会从 18 KB 涨到 36 KB）、`dist_h` 改成真正的水平距离。
   **schema 的维度、偏移量、帧头格式没有任何改动。**
2. `craftium_env.py`：把包内只读的 `craftium-envs/chop-tree` 复制到 `env/build/chop-tree`
   （已 gitignore），注入 mod，并在 `world.mt` 里加 `load_mod_ced_obs`。不改 site-packages。
3. `scripts/spike_obs.py`：可重复的验收脚本。

## S7 实测（ChopTree-v0，seed=0，headless）

```
reset 1657ms  steps 600 in 3.01s = 199.6 steps/s
frame: tick 65 -> 240 (+175 over 3.01s = 58.2 ticks/s), dim 2568,
       nonzero 1372, file 18708 bytes
steps per observed frame: 3.43
mod timing: tick 200 avg_ms 2.701 max_ms 4.366
SPIKE OK
```

- 帧通过 `validate_obs`，1372/2568 个分量非零，射线确实命中地形。
- 每帧 18.7 KB，不压缩；20 Hz 下约 374 KB/s/环境。
- mod 的 build+write 平均 2.70 ms、最大 4.37 ms，在 EXECUTION S7 的 5 ms 门禁内，
  但余量已经不多。8 环境并行时它的 CPU 占用约 8 × 58 × 2.7 ms ≈ 1.25 核。
- **8×8 射线降级还没必要**；若以后加实体/听觉段，先看这个数。

## 观测时钟：58 Hz，不是 20 Hz，也不是 200 Hz

实测 Lua `globalstep` 稳定在 **58.2–58.7 次/秒**，而 Python 侧 `env.step` 是 200 次/秒
（Craftium 的 `fps_max=200`）。也就是**每个观测帧被 3.4 个 Python 步复用**。

试过 `dedicated_server_step` = 0.005 / 0.02 / 0.05 / 默认，四档下 Lua 频率都是 58.3，
**这个参数不影响**。所以不能靠配置把观测拉到 20 Hz 或 200 Hz。

后果：S9 的决策周期必须以观测帧为准（收到新帧才决策），或者我们在 mod 里每 3 个
globalstep 才写一次帧，把决策率固定成 ~19.4 Hz。两种都能对上 `configs/base.yaml` 的
20 Hz 设计，但**不能让 Python 的 200 Hz 步进被当成决策率**——那样同一个观测会出三次动作。

## S8 帧校验

- `tests/test_adapter.py` 通过（等待、超时、episode 边界、原子替换）。
- 真机帧：`transport.ObsChannel` 连续读到 tick 递增的帧，`validate_obs` 通过。

## 动作通道的硬约束（影响 S9）

EXECUTION 的 S9 写的是"Lua 每 tick 读 `act.txt`，用 `core.set_player_controls` 施加"。
在这个构建上这条路走不通：

1. `core.set_player_controls` 在 Luanti 的 builtin 与文档里都不存在（已全量 grep）。
2. Craftium 的输入由它自己的 C 通道持有：`mt_channel.send(keys[21], mouse_x, mouse_y)`，
   Python 就是那个客户端。服务端侧再设控制会被客户端的每帧输入覆盖。

**因此 S9 改为：Python 侧把 14 个 token 映射成 Craftium 的 21 键 + 鼠标动作**，
Craftium 通道保持唯一权威。`docs/protocol.md` 的 `act.txt` 通道在 Craftium 阶段不启用，
保留给将来自建服务端（那时 Lua 侧施加控制才有意义）。

这条与 EXECUTION 的写法不一致，**需要你确认**。它不推翻 0001 的四项决策
（结构化观测、禁像素、参数上限、3 种子），只改变动作的传输路径。

## S9 之前还欠的东西

1. **鼠标灵敏度标定**：`yaw` 的 121 桶是"每个决策 ±π/2 弧度"，而 Craftium 的
   `mouse: Box(-1,1,(2,))` 是未标定的相对量。要用固定鼠标值跑几步、从
   `self` 段的 `yaw_sin/yaw_cos` 反推比例，才能把 token 映射过去。
2. **奖励与终止要由 Lua 写回**：mod 里已有 `ced_obs_set_reward/termination`，
   但还没有任何任务脚本调用它们；Craftium 自带的 `set_reward_once` 写的是它自己的通道。
3. **episode_id 语义**：Craftium 的 `reset` 默认重启 Luanti 进程，tick 会归零。
   协议的"tick 跨 episode 单调递增"需要在服务端实例上带一个序号，否则 `ObsChannel`
   会把新 episode 的帧当成旧帧丢弃。
4. **ents / audio / task 三段仍是 0**：schema 里留着，Lua 侧还没填，接任务时补。
