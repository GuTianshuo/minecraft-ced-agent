# 游戏 ↔ Python 传输契约 v1

实现：`transport.py`（Python 侧）、`mock_game.py`（测试用假服务端）、
`env/lua/ced_obs/init.lua`（Luanti 侧参考实现）。

## 1. 三个文件

同一个目录下：

```
obs.txt    游戏 → Python   每 tick 原子覆写
act.txt    Python → 游戏   每个决策原子覆写
ctrl.txt   Python → 游戏   reset <seed> / stop
```

"原子覆写"= 先写 `*.tmp` 再 `rename`。读侧永远看不到半写内容，所以不需要锁。
Python 侧用 `os.replace`，Lua 侧用 `os.rename`。

## 2. 帧格式（obs.txt）

```
<schema_version> <tick> <episode_id> <obs_dim> <reward> <terminated> <truncated>
<obs_dim 个浮点数，空格分隔>
```

第一行恰好 7 个字段，第二行恰好 `obs_dim` 个数。当前 `schema_version = obs-v0.1`，
`obs_dim = 2568`。

## 3. 动作格式（act.txt）

```
<tick> <14 个整数>
```

`tick` 是这组动作**被决定时的服务端 tick**。服务端在下一次 tick 边界应用它，
并要求 `tick >= 上一次已应用的 tick`（否则丢弃，防止旧动作在乱序时覆盖新动作）。

## 4. 必须保证的语义

1. `tick` 单调递增，**跨 episode 不重置**。`episode_id` 只在 reset 时 +1。
   这样 Python 才能用 `(tick, episode_id)` 判断一帧是不是新的。
2. `reward / terminated / truncated` **由服务端计算**。Python 侧不做任何重新解释——
   奖励若依赖特权世界状态，隔离点必须在服务端，而不是在训练脚本里靠自觉。
3. 动作在服务端主线程应用；一 tick 内的顺序固定为
   **应用动作 → 世界更新 → 观测与奖励计算 → 写帧**。
4. 服务端暂停时网络/文件侧仍能接收动作，不允许控制线程与游戏线程互等。
5. 帧丢失不阻塞：Python 侧的 `ObsChannel` 用「tick 比上次新」做过滤，重复帧会被忽略。

## 5. 失败模式

| 情况 | 表现 | 处理 |
| --- | --- | --- |
| 服务端没起来 | `ObsChannel.read` 抛 `TimeoutError` | 显式失败，不静默重试到天荒地老 |
| 读到半写文件 | 字段数不足，跳过并重试 | 由原子替换 + 字段数校验双重兜底 |
| schema 版本不符 | 抛 `ValueError` 进入重试，最终超时 | 升级 schema 时必须同时改两端 |
| 动作越界 | Python 侧 `validate_actions` 直接断言失败 | 不要"帮它修"，越界说明策略或解码有 bug |
| 服务端卡死 | 所有读取超时 | 训练循环应把连续超时当致命错误，不要无限重试 |

## 6. 性能

观测 2568 个浮点文本 ≈ 5 KB/帧，20 Hz ⇒ 约 100 KB/s/环境，8 环境约 800 KB/s。
本地文件系统完全够用。真正的开销是每 tick 一次 `open/write/rename` 的系统调用
（8 环境 × 20 Hz = 160 次/秒），需要实测 tick 时间；如果超过 5 ms，
再换二进制格式或 Craftium 的 C++ 通道，但**先测再换**。

## 7. 为什么先不用 socket

Luanti 的 mod 沙箱没有 socket API；Craftium 自带的通道传的是渲染帧，
与结构化感知路线不匹配。文件通道的语义最清晰，也最容易在没有游戏的情况下用
`mock_game.py` 测通。等 S7 实测证明它拖慢 tick，再升级传输层——那时需要换的
只是 `transport.py` 一个文件。
