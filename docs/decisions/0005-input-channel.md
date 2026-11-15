# 0005 输入通道全面调查：键、鼠标、视线、节拍

日期：2026-11-15
状态：S9 的动作通道设计依据

## 背景

0004 发现动作只能走 Craftium 自己的通道。接 S9 之前，把这条通道从头到尾测了一遍：
动作怎么送、每帧生效多少、鼠标与视线怎么对应、服务端能不能看见客户端朝向。
结论直接决定 14 个 token 怎么映射。

## 1. 通道形状

Python 每帧调用一次 `CraftiumEnv.step(dict)`，字典被摊平成 21 个按键位 + 2 个 int16 鼠标量，
经 `mt_channel.send()` 用 TCP 送给引擎里的客户端（`src/client/client.cpp::pyConnStep`）。

21 个键位按 `ACTION_ORDER` 排列：forward, backward, left, right, jump, aux1, sneak, zoom,
dig, place, drop, inventory, slot_1..slot_9。

鼠标：`Dict` 里的 `mouse` 是 `Box(-1,1,(2,))`，代码按 `mouse_x = int(x*(obs_width//2))` 转成像素，
64×64 的观测下即 `int(x*32)`，所以**线上裸值范围是 ±32**，最小非零量是 1。

## 2. 每帧生效，不保持

`pyConnStep` 每帧先把所有虚拟按键清零，再按刚收到的动作重设。实测：

- 只按 1 帧 forward：当帧速度 2.825 n/s，下一帧立刻回 0，位移 0.0139 节点。
- 只给 1 帧鼠标：yaw 当帧就转完（wire=4 → -64° 视灵敏度而异），之后保持不变。
- 动作的效果**在当帧返回的 info 里就能看到**，没有额外延迟。

**所以"按住"= 在后续每一帧重复发送同一个动作**。这一点对 CED 的 hold token 是硬约束。

## 3. 物理参数与量级（Craftium 默认 `pmul=20`）

| 项 | 值 |
| --- | --- |
| 行走终速 | 80 节点/s（`movement_speed_walk = 4×pmul`） |
| 加速度 | 60 节点/s² |
| 潜行终速 | 27.0 节点/s（同时按 forward+sneak 取潜行） |
| aux1 + forward | 78.4 节点/s（**没有加速效果**，不是冲刺键） |
| forward + backward | 0（互相抵消） |
| 终速下每 Python 帧位移 | 0.388 节点 |
| reset 后到终速 | 约 30 帧 |
| 单帧跳跃 | 起跳约 6.2 节点/s，之后持续上升，40 帧（≈0.2 s）升到 0.89 节点 |

**pmul=20 意味着世界里的速度和真实 Minecraft 差一个量级**，0001 里"Luanti ≠ Minecraft"的
警告在动作层面同样成立。

## 4. 鼠标 → 视角

实测每帧转角 = `1.4 × mouse_sensitivity × wire` 度（默认 sensitivity=0.2 → 0.28°/wire）：

| sensitivity | 每 wire 每帧 | `mouse` 取满 ±1.0（±32 wire）时每帧 |
| --- | --- | --- |
| 0.2（默认） | 0.28° | ±9.0° |
| 0.6 | 0.84° | ±26.9° |

两个后果：

1. 默认灵敏度下，动作空间的鼠标**最多每帧转 9°**，一个 20 Hz 决策（≈3.4 帧）只有 ±30°，
   而我们 yaw token 的定义是每决策 ±90°。直接用鼠标，token 范围用不满。
2. 转角**不夹紧**：连续给同一个鼠标值，pitch 一路累加（实测 -1285°，客户端照单全收），
   yaw 同理。正常键盘鼠标输入有 ±89° 的俯仰限制，虚拟输入绕过了它。

## 5. 服务端看不见客户端的朝向（关键发现）

我们的射线观测跑在 Lua 里，用的是 `player:get_look_dir()`，读的是 PlayerSAO 的
`m_pitch / rotation.Y`。实测：

- 客户端鼠标转 -80.2°，服务端 `get_look_horizontal()` 仍停在旧值，pitch 完全不动
  （客户端 -48.1°，服务端 -0.01 rad）。
- 因此**只靠鼠标转动，射线观测对转头完全不敏感**——同一帧在俯角 +64° 与 -64° 下
  射线距离逐条相同。

### 绕法（已实现并验证）

Python 写 `<run_dir>/worlds/world/look.txt`，内容一行 `<yaw_rad> <pitch_rad>`；
Lua 每个 globalstep 读它，只在数值变化时调用 `player:set_look_horizontal/vertical`。实测：

| 写入 look.txt | 客户端 yaw | 帧（服务端）yaw | 客户端 pitch | 帧 pitch |
| --- | --- | --- | --- | --- |
| 1.0 rad, -0.5 rad | 57.29° | 57.29° | -28.64° | -28.64° |

射线随之改变（同一条中心射线在 yaw 0 / +90° / 180° / -90° 下距离分别为
20.54 / 8.52 / 2.51 / 23.55 节点），前进方向也跟随服务端 yaw（yaw +90° → 前进方向 -90°，
即 `atan2(dx,dz) ≈ -yaw`，中间角度有几十度的滞后，S9 要闭环验证）。

`set_look_*` 必须**只在目标变化时调用**：它每次都会给客户端发一次 MovePlayer，
每 tick 都调会让客户端 yaw 不断叠加。

## 6. dig / jump / place

- **dig 可用但慢**：俯视地面连续按 dig，约 200 帧（≈1 s 墙钟）挖掉一格
  （中心射线从 class 94 / 1.74 节点变为 class 2 / 2.90 节点）。ChopTree 只对树给奖励，
  挖土奖励为 0，这是环境定义不是 bug。
- **jump 单帧即可起跳**，之后上升约 40 帧。
- place / drop / slot / inventory 未逐一实测，S9 接任务时再验。

## 7. 对 S9 的结论

1. **朝向走 look.txt，不走鼠标**：绝对角度、无量化、无累加漂移，服务端射线与客户端镜头一致。
   鼠标通道留给"需要连续平滑转动"的场合（本轮不需要）。
2. **按键走 Craftium 的 Dict**，每决策重复发送约 3.4 帧（观测帧率 58 Hz / 步进 200 Hz）。
3. **一次决策只能改变当帧状态**：hold token 必须映射成"重复发送 N 帧"，不能指望引擎保持。
4. **动作里必须带绝对 yaw/pitch**，因为 token 是相对量、而服务端只认绝对值；由 Python 累计。
5. 直接用默认灵敏度时鼠标够不到 ±90°/决策，这也再次说明朝向不该走鼠标。

## 附：为此改动的 Lua

`env/lua/ced_obs/init.lua` 增加了 `look.txt` 的读取与应用（上面第 5 节），
并把诊断统计改为每 20 tick 写一次（含服务端 yaw/pitch/位置）。观测帧本身未改。
