# 0003 Craftium 接入：安装与单环境节拍（S5 / S6）

日期：2026-11-15
状态：S5、S6 通过；S7（结构化观测）未开始

## 背景

0001 决策 2 把主训练环境定为 Craftium（Luanti 分支）。本文件记录安装方式、可用环境、
单环境节拍实测，以及一个必须先说清楚的接口差：Craftium 原生观测是像素，不是我们的 schema。

## 安装（S5）

`craftium` 不在 PyPI。上游 `mikelma/craftium` 只发行预编译 wheel（v0.0.1 是唯一版本），
装在项目 venv 里：

```
.venv/bin/pip install "https://github.com/mikelma/craftium/releases/download/v0.0.1/craftium-0.0.1-cp313-cp313-manylinux_2_28_x86_64.whl"
```

wheel 328 MB，内含 Luanti 二进制，不需要系统编译依赖（cmake / SDL2 等一个都不用装）。

新增 Python 依赖：`gymnasium==0.29.1`、`pygame`、`PyYAML`（wheel 自身声明）。
这与骨架原来的"只用 numpy + torch"不同，是 Craftium 的硬性依赖，记在此处备查。

已注册 7 个环境：`Room`、`SmallRoom`、`ChopTree`、`Speleo`、`SpidersAttack`、`OpenWorld`、`ProcDungeons`。

## S5 随机策略

```
import craftium, gymnasium as gym
env = gym.make("Craftium/ChopTree-v0").unwrapped
obs, _ = env.reset()
```

- 必须先 `import craftium`：环境注册发生在它的 `__init__` 里，只 import gymnasium
  会抛 `NamespaceNotFound: Craftium`。
- `reset` 1660–1780 ms，200 步随机动作 2.79 s 总计，正常退出。

## S6 单环境节拍

20 秒连续 `step`（ChopTree-v0，headless，不渲染）：

| 指标 | 实测 | EXECUTION 门禁 |
| --- | --- | --- |
| 步数 | 3998 | — |
| 吞吐 | **199.9 steps/s** | ≥ 200（达标） |
| 单步均值 | 5.00 ms | — |
| 单步 p99 | 6.78 ms | — |
| 最差单步 | 8.59 ms | < 20 ms（达标） |
| reset | 1660 / 1675 ms | < 2 s（达标） |

199.9 这个数是 `CraftiumEnv` 的 `fps_max=200` 默认值卡出来的上限，不是 Python 侧瓶颈：
步进由 Luanti 服务端节拍驱动，Python 侧只等结果。

## 结论与必须处理的接口差

1. **节拍够用**。单环境 200 步/s 是 20 Hz 目标的 10 倍，8 环境并行的天花板在 CPU 和
   内存带宽，不在这一步。
2. **原生观测不能直接用**：`Box(0, 255, (64, 64, 3), uint8)`，64×64 像素。
   我们的 schema 是 2568 维结构化向量。0001 的"路线 A"要求结构化射线，所以
   **S7 的 Lua 观测 mod 是必经之路**，不是可选优化。
3. **原生动作也不能直接用**：`Dict`，20 个离散键（forward / backward / dig / place /
   slot_1..9 / sneak / jump …）+ `mouse: Box(-1,1,(2,))`。我们的 14 token 需要一层
   映射到这套键鼠（S9 的动作通道）。
4. **注意默认 `fps_max=200`**：以后做并行环境时，若把 fps_max 调高，Python 侧才会成为瓶颈；
   调低则吞吐直接按比例下降。这个参数要写进环境配置，不能靠默认值。

## 下一步

S7：把 `ced_obs.lua` 放进 run_dir 的 worldmods，验证每 tick 写出 2568 维结构化帧，
再用 `transport.ObsChannel` 读回并通过 `validate_obs`。这一步最耗时，也最容易卡住。
