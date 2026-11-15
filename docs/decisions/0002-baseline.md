# 0002 P0a 空转基线

日期：2026-11-15
状态：已记录（G1 门禁通过）

## 背景

骨架首次在目标机器上执行。本文件记录 S2（五个测试）与 S3（三档冒烟）的实测结果，
作为以后任何"变快了/变准了"说法的比较基准。

所有数字来自下面的命令在 2026-11-15 的实际运行，没有估算，也没有从旧文档抄。

## 机器（实测四行）

| 项 | 实测 |
| --- | --- |
| CPU | 12th Gen Intel(R) Core(TM) i5-12600K，16 线程 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER，16376 MiB，驱动 595.99.02，最大 SM 时钟 3135 MHz |
| 内存 | 45 GiB 总量，运行时可用 39 GiB |
| 磁盘 | `/dev/nvme2n1p3`（SK hynix HFS002TEJ9X115N，1.9T NVMe）挂载 `/`，252G 中已用 184G，**剩 57G** |

软件栈：Ubuntu 26.04.1 LTS，Python 3.13.13，torch 2.14.1+cu126（CUDA 12.6，cuDNN 9.10.2），
numpy 2.5.4，venv 在仓库根 `.venv/`。

## 测试（S2）

```
python tests/test_schema.py      -> schema tests passed
python tests/test_env.py         -> env tests passed（教师到达 20/20）
python tests/test_models.py      -> model tests passed（因果性、形状、生成范围、参数对齐）
python tests/test_pipeline.py    -> pipeline tests passed
python tests/test_adapter.py     -> adapter tests passed
```

参数对齐实测：tf=17.98M，ced=18.61M，**ratio=1.04**（门禁 0.8–1.25）。

## 三档冒烟（S3）

命令（与 `scripts/smoke.sh 300` 相同的参数，外加 `/usr/bin/time -v` 取耗时与峰值 RSS）：

```
.venv/bin/python train.py --model {gru,tf,ced} --steps 300 --smoke --ckpt runs/baseline/<m>.pt
```

另单独跑过一次 `scripts/smoke.sh 300`，三档均打印 `SMOKE OK`。

训练数据：教师策略 400 episodes / 10815 个决策步，窗口 16，batch 16，lr 3e-4，无 AMP。

| 档 | 参数量 | first10 loss | tail loss（末 20 步均值） | 贪心成功率（20 eps） | 耗时 | 峰值 RSS | 训练期 RSS 增长 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gru | 2.04M | 12.5644 | 1.1732 | 5.00% | 0:06.52 | 1975 MB | −3 MB |
| tf | 17.98M | 8.1043 | 1.1610 | 0.00% | 0:46.21 | 2080 MB | 0 MB |
| ced | 18.61M | 19.1588 | 1.1730 | 0.00% | 1:13.56 | 2130 MB | 0 MB |

CED/Transformer 耗时比 1.59。CED 的评测走 14 次解码器前向的朴素自回归
（20 episodes × ≤200 步），是它耗时的主要来源；训练仍是 teacher forcing。

## 结论

1. **训练栈健康**：三档 loss 都从初值降到 1.16–1.17 量级，无 NaN，RSS 在训练中不增长，
   checkpoint 存盘→加载往返一致。G1 门禁（三档 SMOKE OK、参数比 0.8–1.25、因果性通过）满足。
2. **这一行不是学习效果基线**：300 步只验证通路。TF/CED 贪心成功率 0% 是正常的，
   不能据此判断架构优劣；真正的成功率基线在 P1 的 `bc.py` 门禁（`configs/base.yaml`）。
3. **尾部 loss 三档几乎相同**（1.161–1.173），量级差异 ≤1%，属噪声，不构成任何架构结论。

## 记录在案的问题与偏离

1. **GRU 参数量与文档不符（未改）**：实测 2.04M，而 `models.py` 头注释写 "GRU ≈ 3.7M"、
   README/EXECUTION 写 "约 3M"。改动它会改变对照实验的定义，需要你决定是修文档还是修配置。
2. **冒烟 RSS 门禁的采样点移到了第 10 步**（提交 `bcf1d07`）。原因：CUDA 上下文与
   cublas/cudnn workspace 的一次性分配（GRU 约 800 MB）发生在旧基线之后。阈值 400 MB 未改，
   门禁仍然测"训练期间是否持续增长"；三档在第 10 步之后 + 评测的增长为 66/55/2 MB（诊断时测得）。
3. **`evalx.make_env` 改为位置参数调用**（提交 `713d923`）。`tests/test_pipeline.py` 与
   `bc.py` 两处调用方都只接受一个位置参数，原关键字调用必然 TypeError。未改任何测试。
4. **磁盘余量 57G，低于 EXECUTION 期望的 100G**。`configs/base.yaml` 的 80 GB 数据配额
   在当前分区不可能满足。P1 采大批数据前必须先解决（换分区或降低配额），本阶段不受影响。
5. **llama-local 服务已停并禁用开机自启**（`systemctl --user disable --now llama-local.service`），
   显存占用从 7407 MiB 降到 1966 MiB。恢复：`systemctl --user enable --now llama-local.service`。
6. **未进入 S5**（Craftium 安装），等确认。
