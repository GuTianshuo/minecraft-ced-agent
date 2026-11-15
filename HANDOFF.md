# HANDOFF：从 Chat 转到 Code 模式

日期：2026-11-15
状态：P0a 骨架完成，**全部 Python 代码尚未在任何机器上执行过**。

---

## 1. 落盘

源目录（本机）：

```
/home/g/.config/Claude-3p/local-agent-mode-sessions/9b9c81ea/00000000/9b4f0b94/outputs/ced_starter
```

```bash
mkdir -p ~/work
cp -r /home/g/.config/Claude-3p/local-agent-mode-sessions/9b9c81ea/00000000/9b4f0b94/outputs/ced_starter \
      ~/work/minecraft-ced-agent
cd ~/work/minecraft-ced-agent
git init && git add -A && git commit -m "P0a scaffold: schema / env / models / pipeline / transport"
```

若这台机器上没有该路径（比如 code 模式在另一台机器），把这条消息附带的文件按同名放进目录即可，结构见第 3 节。

---

## 2. 代码状态

| 模块 | 文件 | 状态 |
| --- | --- | --- |
| 观测/动作 schema | `schema_v0.py` | 完整，附校验器 |
| 合成环境 | `envs.py::DummyNavEnv` | 完整 |
| 传输契约 | `transport.py` / `docs/protocol.md` | 完整，格式已定 |
| 假服务端 | `mock_game.py` | 完整，用于无游戏自测 |
| 真服务端适配器 | `envs.py::TransportEnv` | 完整；真机未验证 |
| 三档模型 | `models.py` | 完整；参数对齐 18.0M / 18.6M |
| 训练/评测 | `runlib.py` / `train.py` / `bc.py` / `evalx.py` | 完整 |
| 数据分片 | `shards.py` / `collect.py` | 完整 |
| 任务注册表 | `tasks.py` | T0 已实现；T1–T6 显式未实现 |
| Luanti 观测 mod | `ced_obs.lua` | 参考实现；动作通道未写 |
| 测试 | `tests/*.py` | 五组；**未运行过** |

已知空白：Lua 动作通道、Craftium 安装与接入、真机 tick 实测、PPO。
已知风险：首次运行可能因环境版本差异挂 1–3 处；`test_models` 的因果性断言依赖
`nn.TransformerEncoder` 的 mask 语义，若 torch 版本差异导致失败，先查 mask 而不是改断言。

---

## 3. 目录结构

```
ced_starter/
├── HANDOFF.md  README.md  EXECUTION.md
├── schema_v0.py  envs.py  transport.py  mock_game.py
├── models.py  runlib.py  train.py  collect.py  bc.py  evalx.py
├── tasks.py  shards.py  ced_obs.lua
├── configs/base.yaml
├── docs/protocol.md  docs/decisions/0001-environment.md
├── scripts/bootstrap.sh  scripts/smoke.sh
└── tests/test_schema.py  test_env.py  test_models.py
          test_pipeline.py  test_adapter.py
```

---

## 4. 给 Code 模式的提示词

复制下面整段：

```text
这是一个已有代码骨架的仓库，目标是训练一个 Minecraft 智能体。请先读文件再动手。

【背景】
硬件：i5-12600K / 48GB / RTX 4070 Ti SUPER 16GB，原生 Ubuntu。
当前阶段：P0a（训练栈空转）到 P1（数据管线）。**不接触正式版 Minecraft**，
主训练环境后续用 Craftium（Luanti 分支），现在还没装。

【先读】
HANDOFF.md、EXECUTION.md、docs/protocol.md、docs/decisions/0001-environment.md。
读完不要改任何文件。

【第一步：让测试全绿】
在仓库根目录建 venv，装 numpy 和 CUDA 版 torch，然后依次运行：
  python tests/test_schema.py
  python tests/test_env.py
  python tests/test_models.py
  python tests/test_pipeline.py
  python tests/test_adapter.py

全部通过之前不要做别的事。

【修复规则 —— 重要】
1. 测试是规格，不是障碍。若你确信某条测试本身写错了，先说明理由，
   并且把"修改测试"单独做成一次提交，禁止为了让代码过而放宽断言。
2. 不得改动 schema 的维度、偏移量、动作 token 数量。确有需要，停下来问我。
3. 不得引入新依赖，除非先说明理由并得到同意。
4. 不得用 try/except 吞异常来让测试通过；先定位根因。
5. 不要顺手重构、不要改注释风格、不要删测试、不要调整门禁阈值。

【第二步：三档冒烟】
  scripts/smoke.sh 300
三档 gru / tf / ced 都要打印 SMOKE OK。
若 CED 在 CPU 上太慢，可以用 --steps 50 先验证通路，但必须在记录里注明
这不是正式基线。

【第三步：记录】
把硬件四行（CPU/GPU/内存/磁盘）、三档的参数量、tail loss、贪心成功率、
耗时、RSS 写入 docs/decisions/0002-baseline.md。
数字必须来自实际运行，不许估算或从文档抄。

【工作方式】
- 每完成一步，给我：执行的命令、关键输出、下一步计划。
- 一次只改少数相关文件，不要批量改。
- 出错先诊断再改，不要连续试错。
- 与规划冲突时，以 docs/decisions/0001-environment.md 的四项决策为准。

【完成标准】
五个测试全过 + 三个 SMOKE OK + 0002-baseline.md 有真实数据。
到此停下汇报，不要自行进入 Craftium 安装（S5），那一步等我确认。
```

---

## 5. 交接时最容易被弄坏的三件事

1. **schema 单一真相**：`schema_v0.py` 的偏移量、`ced_obs.lua` 的打包顺序、分片里的
   `schema_version` 必须同时改。只改一处，数据会静默错位而不报错。
2. **对照实验的公平性**：三档必须共用编码器、参数对齐、训练时逐帧算 loss。
   为省时间给某一档"特调"，对照就作废了。
3. **门禁不是装饰**：`evalx.check_gate` 与 `configs/base.yaml` 的阈值是判断"该查数据还是该改模型"的
   分界线。数值不达标时先看门禁给的理由，再决定改什么。
