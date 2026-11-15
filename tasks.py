"""任务注册表。

任务定义与门禁来自规划 v2 第 4 节。这里只登记"机器可读"的部分：
判据的自然语言描述、步数上限、所属环境层级、是否已实现。

DummyNavEnv 只承载 T0，用途是验证管线，不是研究任务本身。
T1–T6 必须在 L1（Craftium）上实现，在此之前调用会明确报错，而不是悄悄降级。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional

import envs


@dataclass(frozen=True)
class TaskSpec:
    id: str
    name: str
    env_level: str          # "dummy" | "L1"
    max_steps: int
    success: str            # 评测判据，写死，不许训练中途改
    reward_hint: str
    adds: str               # 相对上一级新增的能力维度
    implemented: bool = False
    factory: Optional[Callable] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("factory")
        return d


TASKS: dict[str, TaskSpec] = {
    "T0": TaskSpec(
        "T0", "go_to_point", "dummy", 600,
        "与目标水平距离 < 0.8 格",
        "朝目标投影距离减少 + 到达 +1；-0.001/tick",
        "基础导航", True,
        lambda seed=0: envs.DummyNavEnv(seed=seed, max_steps=200),
    ),
    "T1": TaskSpec(
        "T1", "avoid_obstacle", "L1", 600,
        "与目标水平距离 < 0.8 格，且全程未穿过实心方块",
        "同 T0，另加碰撞惩罚",
        "局部避障",
    ),
    "T2": TaskSpec(
        "T2", "follow", "L1", 1200,
        "全程平均距离 < 5 格，且无连续 3 秒静止",
        "保持 2–4 格距离的带状奖励",
        "动态目标",
    ),
    "T3": TaskSpec(
        "T3", "chop_tree", "L1", 600,
        "60 秒内背包获得 ≥ 1 个原木",
        "靠近 + 对准 + 破坏进度 + 拾取",
        "连续交互",
    ),
    "T4": TaskSpec(
        "T4", "pickup", "L1", 900,
        "60 秒内所有掉落物被拾取",
        "距离减少 + 拾取 +1",
        "精确落点",
    ),
    "T5": TaskSpec(
        "T5", "place", "L1", 900,
        "目标坐标出现指定方块，且未破坏其他方块",
        "对准误差减少 + 放置成功 +1",
        "精细放置",
    ),
    "T6": TaskSpec(
        "T6", "container", "L1", 900,
        "目标物品进入/离开箱子，且箱子未被破坏",
        "靠近 + 开箱 + 物品转移 +1",
        "状态化交互",
    ),
}

ORDER = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]


def make_task(task_id: str, seed: int = 0):
    if task_id not in TASKS:
        raise KeyError(f"unknown task {task_id}, have {ORDER}")
    spec = TASKS[task_id]
    if not spec.implemented:
        raise NotImplementedError(
            f"{task_id}({spec.name}) 需要 {spec.env_level} 环境，当前未实现。"
            f"先完成 P0b（Craftium 接入），不要在 dummy 上伪造它。")
    return spec.factory(seed=seed)


def describe_all() -> list[dict]:
    return [TASKS[t].to_dict() for t in ORDER]


def next_task(task_id: str) -> Optional[str]:
    i = ORDER.index(task_id)
    return ORDER[i + 1] if i + 1 < len(ORDER) else None
