"""教师策略：只用观测，不读特权状态。

第一个教师是 go_to_point：根据 task 段的相对目标向量转 yaw，对准后前进；
接近目标时改用潜行（27 节点/s，每决策约 0.4 格），避免以 80 节点/s 冲过 0.8 格的判定圈。

位移方向与 yaw 的约定来自 0005/0006 的实测：`atan2(dx, dz) = -yaw`。
"""

from __future__ import annotations

import math

import numpy as np

from schema_v0 import N_ACTION_TOKENS, OFF_SELF, OFF_TASK, rad_to_yaw_index

YAW_CENTER = 60
PITCH_CENTER = 60
ALIGN = 0.45          # 夹角小于此值才前进（弧度，约 26°）
SLOW_DIST = 4.0       # 小于此距离改用潜行接近


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def goto_action(obs: np.ndarray) -> np.ndarray:
    dx = float(obs[OFF_TASK + 1]) * 32.0
    dz = float(obs[OFF_TASK + 2]) * 32.0
    dist = math.hypot(dx, dz)
    yaw = math.atan2(float(obs[OFF_SELF + 3]), float(obs[OFF_SELF + 4]))
    desired_yaw = -math.atan2(dx, dz)
    delta = _wrap(desired_yaw - yaw)

    act = np.zeros(N_ACTION_TOKENS, dtype=np.int64)
    act[11] = int(rad_to_yaw_index(np.array(delta)))
    act[12] = PITCH_CENTER
    if abs(delta) < ALIGN:
        act[0] = 1                       # forward
        if dist < SLOW_DIST:
            act[5] = 1                   # sneak：降低接近速度
    return act
