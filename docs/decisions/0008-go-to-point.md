# 0008 第一个 L1 任务 go_to_point、教师数据与旁观训练

日期：2026-11-15
状态：任务、采集、训练、旁观都跑通；数字见下

## 任务（Lua 侧）

`env/lua/ced_obs/init.lua` 里的 `go_to_point`：

| 项 | 值 |
| --- | --- |
| 环境 | `room`（17×13 的封闭房间，superflat） |
| 目标点 | 出生点出发，距离 `ced_goal_min`–`ced_goal_max`（默认 10–20，本轮用 5–8 格），角度由 `fixed_map_seed` 决定 |
| 奖励 | 每 tick `上一 tick 距离 − 当前距离`，到达再 +10 |
| 终止 | 水平距离 < 0.8 格 |
| task 段 | `task_id=1, dx/32, dz/32, dist/32, progress`（写进 schema 的 2568 维） |

观测、奖励、终止全部在服务端算；Python 侧不重新解释（`CraftiumCedEnv.step`
返回的是帧里的 reward/terminated）。

## 教师

`teachers.py::goto_action`：由 task 段的相对目标算出应有的绝对 yaw，转 yaw token；
夹角 < 0.45 rad 才前进，距离 < 4 格改用潜行（27 节点/s）接近，避免 0.8 格判定圈被冲过。

## 关键修复：位置刷新频率

默认配置下，**观测里的位置每 5–8 个决策才更新一次**（`dedicated_server_step`
≈0.09 s 决定客户端上报间隔），也就是 7/8 的动作是在用旧状态决策。实测房间内教师：

| dedicated_server_step | 距离连续相同的最长游程 | 教师到达用的决策 |
| --- | --- | --- |
| 默认（0.09） | 均值 5.7 | 35 |
| 0.05 | 3.0 | 13 |
| **0.01** | **1.0** | **12** |

`make_env` 现在默认 `dedicated_server_step=0.01`。注意它**不改变**观测帧率
（仍是 57–58 Hz，每决策 3 帧），只改变位置的刷新粒度。

## 物理速度

Craftium 默认 `pmul=20`（行走 80 节点/s）。本轮改为 **`pmul=1`**（4 节点/s，
接近 Minecraft）：每决策约 0.2 格，控制精细，episode 约 200 个决策。
`pmul=20` 时教师 12–13 个决策就能到，但 0.8 格的到达圈太容易冲过，BC 学不稳。

## 数据

```
python scripts/collect_craftium.py --episodes 60 --decisions 300 --out data_room \
    --env room --policy teacher --goal-min 5 --goal-max 8 --pmul 1
```

- 60 个 episode，**12578 个决策**，1.5 MiB，**125 bytes/step**
- 教师到达率：多数 episode 到达，少数（目标贴着墙/被家具挡住）在 300 决策内没到
- 分片 meta 现在额外记录 `reached` 与 `frames_per_decision`

## 训练

```
python scripts/watch_train.py --steps 4000 --eval-every 800 --eval-episodes 3 \
    --max-decisions 400 --pmul 1
```

| step | loss | val 成功率 | 平均决策 | 终点距离 |
| --- | --- | --- | --- | --- |
| 800 | 0.427 | 67% | 229 | 7.48 |
| 1600 | 0.354 | 67% | 221 | 7.48 |
| 2400 | 0.330 | 67% | 221 | 7.48 |
| 3200 | 0.314 | 67% | 224 | 7.49 |
| 4000 | 0.308 | 67% | 223 | 7.48 |

GRU 2.04M，4000 步约 2 分钟（RTX 4070 Ti SUPER）。成功率 2/3 之后就平了，
继续训练只会让 loss 更低、行为不变——下一步要加数据或换数据，不是加步数。

## 旁观训练

```
python scripts/watch_train.py --steps 12000 --eval-every 1000 --eval-episodes 3 \
    --max-decisions 400 --pmul 1 --window
```

- 训练在后台跑；每到 `--eval-every` 就用**当前模型**在真引擎里跑一遍评测，
  窗口里是 agent 的第一视角（Craftium `render_mode="human"`，64×64 放大到 720×720）。
- 窗口标题实时显示 `step / loss / success / 当前 episode / decision / 距目标`。
- 终端同步打印每次评测的一行汇总。
- 评测因渲染被限制在 30 fps，所以看起来是实时的；训练本身不受影响。

## 还没做的

1. 教师到达率不到 100%，失败的 episode 混进了训练集（`reached` 字段已记下，
   下一轮可以过滤，或在目标点选址时避开墙）。
2. 评测只有 3 个种子，且都是 val 划分；还不够判断"学会了"还是"背过了"。
3. 房间没有障碍物，T1（避障）才是真正需要射线的那一级。
4. CED/Transformer 还没在这个任务上跑过，只有 GRU。
