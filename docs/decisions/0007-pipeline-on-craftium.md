# 0007 数据管线在真引擎上跑通（S10 前半段）

日期：2026-11-15
状态：链路通过；任务与奖励未接，这次的数据不是训练素材

## 做了什么

`scripts/collect_craftium.py`：在 `CraftiumCedEnv` 上用**均匀随机策略**采 3 个 episode，
每个 150 个决策，按 `shards.py` 的格式落盘；然后用**未改动的** `bc.py` 训练 gru。

```
python scripts/collect_craftium.py --episodes 3 --decisions 150 --out data_craftium
python bc.py --model gru --task T0 --data data_craftium --steps 200 --eval-every 100
```

## 数字

| 项 | 实测 |
| --- | --- |
| 采集 | 3 个 episode，450 个决策，0.5 MiB |
| 磁盘 | **1265 bytes/step**（float16 + savez_compressed；未压缩理论上限 5164） |
| 每决策帧数 | 3–5（随机动作下略有波动，稳定动作时是 3） |
| 训练 | gru 2.04M，200 步 3 秒，loss 19.74 → 18.61 |
| 评测 | 跑通，但见下面的警告 |

## 这次数据不能当训练素材

1. **策略是随机的**，动作和任务无关，BC 学不到任何东西。
2. **奖励还没接**：`ced_obs_set_reward/termination` 没有任何任务调用，
   环境返回的仍是 Craftium 自带的 ChopTree 奖励（砍树 +1），与本轮任务无关。
3. **评测跑在 dummy 上**：`bc.py` 的评测走 `tasks.make_task("T0")`，那是合成环境。
   33.3% 的成功率是 dummy 上的数字，和 Craftium 无关，不能引用。
4. **没有 val 划分**（3 个种子的 `seed%10` 都落到 train），门禁的方差判定因此不可用。

## 它的价值

证明这条链路在真引擎上没有断：观测 2568 维 → 随机动作 → 分片 → 切分 → 训练 → 评测，
`bc.py`、`shards.py`、`evalx.py` 一行没改。剩下的问题都在环境侧，不在训练栈。

## 下一步（真正的 S10）

1. 在 Lua 里实现第一个 L1 任务（go_to_point）：目标点、距离奖励、到达判定、
   task 段（dx/dz/dist/progress）填进观测。
2. Python 侧写教师：用 task 段的夹角算 yaw token，对准后按 forward。
3. 用教师采数据，**先在 dummy 上跑通同一套命令**，再换 Craftium，确认两边指标可比。
4. 评测改用 CraftiumCedEnv，加 val 种子。
