"""观测/动作 schema v0 —— Python 侧唯一真相。

与 ced_obs.lua 必须逐字段一致。改任何字段都要同时改：
  1. 本文件
  2. ced_obs.lua 的打包段
  3. 数据分片里的 schema_version
约束：恒定维度，缺失填 0 并带 presence 位；禁止变长结构。
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "obs-v0.1"

# ---------------------------------------------------------------- 观测布局

RAY_GRID = 16
N_RAYS = RAY_GRID * RAY_GRID            # 256
RAY_FEATURES = [
    "block_class",   # 0=未命中；1..255 模糊方块类
    "nx", "ny", "nz",  # 命中面法向 [-1,1]
    "dist",          # 距离/32
    "dy",            # 相对眼睛高度差/32
    "dist_h",        # 水平距离/32
    "light",         # 方块光照 0..1
    "shape",         # 0 全方块 1 半砖 2 楼梯 3 栅栏 4 玻璃 5 植物 6 流体 7 其他
]
N_RAY_DIM = len(RAY_FEATURES)            # 9

SELF_FEATURES = (
    ["px", "py", "pz", "yaw_sin", "yaw_cos", "pitch_sin", "pitch_cos",
     "vx", "vy", "vz", "on_ground", "sneaking", "sprinting", "health",
     "hunger", "air", "selected_slot", "obs_light"]
    + [f"inv_{i}" for i in range(8)]
    + [f"hotbar_{i}" for i in range(8)]
    + [f"pad_{i}" for i in range(14)]
)
SELF_DIM = len(SELF_FEATURES)            # 48

N_ENT = 16
ENT_FEATURES = ["class", "dx", "dz", "dy", "dist", "vx", "vz", "light",
                "visible_tier", "has_los", "age", "pad"]
ENT_DIM = len(ENT_FEATURES)              # 12

N_AUDIO = 4
AUDIO_FEATURES = ["class", "octant", "volume"]
AUDIO_DIM = len(AUDIO_FEATURES)          # 3

TASK_FEATURES = (["task_id", "dx", "dz", "dist", "progress"]
                 + [f"pad_{i}" for i in range(7)])
TASK_DIM = len(TASK_FEATURES)            # 12

OFF_RAYS = 0
OFF_SELF = OFF_RAYS + N_RAYS * N_RAY_DIM
OFF_ENT = OFF_SELF + SELF_DIM
OFF_AUDIO = OFF_ENT + N_ENT * ENT_DIM
OFF_TASK = OFF_AUDIO + N_AUDIO * AUDIO_DIM
OBS_DIM = OFF_TASK + TASK_DIM            # 2568

# ---------------------------------------------------------------- 动作布局
# 14 个离散 token，BC 用逐 token 交叉熵，CED 解码器逐 token 生成。
ACTION_TOKENS = [
    ("fwd", 2), ("back", 2), ("left", 2), ("right", 2),
    ("jump", 2), ("sneak", 2), ("sprint", 2),
    ("attack", 2), ("use", 2), ("drop", 2),
    ("hotbar", 9), ("yaw", 121), ("pitch", 121), ("hold", 4),
]
ACTION_TOKEN_NAMES = [n for n, _ in ACTION_TOKENS]
ACTION_NCLASSES = np.array([c for _, c in ACTION_TOKENS], dtype=np.int64)
N_ACTION_TOKENS = len(ACTION_TOKENS)     # 14
MAX_CLASSES = int(ACTION_NCLASSES.max())  # 121
YAW_BUCKETS = 121
YAW_MAX_RAD = np.pi / 2                   # 单 tick 最大偏转


def yaw_index_to_rad(idx: np.ndarray) -> np.ndarray:
    """0..120 -> [-pi/2, pi/2]，中心 60 为不转。"""
    return (np.asarray(idx, dtype=np.float32) - 60.0) / 60.0 * YAW_MAX_RAD


def rad_to_yaw_index(delta: np.ndarray) -> np.ndarray:
    delta = np.clip(delta, -YAW_MAX_RAD, YAW_MAX_RAD)
    return np.clip(np.rint(delta / YAW_MAX_RAD * 60.0) + 60, 0, YAW_BUCKETS - 1).astype(np.int64)


# ---------------------------------------------------------------- 工具

def empty_obs() -> np.ndarray:
    return np.zeros(OBS_DIM, dtype=np.float32)


def unpack(obs: np.ndarray) -> dict:
    obs = np.asarray(obs, dtype=np.float32)
    assert obs.shape[-1] == OBS_DIM, f"expected {OBS_DIM}, got {obs.shape[-1]}"
    return {
        "rays": obs[..., OFF_RAYS:OFF_SELF].reshape(*obs.shape[:-1], N_RAYS, N_RAY_DIM),
        "self": obs[..., OFF_SELF:OFF_ENT],
        "ents": obs[..., OFF_ENT:OFF_AUDIO].reshape(*obs.shape[:-1], N_ENT, ENT_DIM),
        "audio": obs[..., OFF_AUDIO:OFF_TASK].reshape(*obs.shape[:-1], N_AUDIO, AUDIO_DIM),
        "task": obs[..., OFF_TASK:],
    }


def validate_obs(obs: np.ndarray) -> None:
    assert obs.dtype == np.float32, f"dtype {obs.dtype}"
    assert obs.shape[-1] == OBS_DIM, f"shape {obs.shape}"
    assert np.isfinite(obs).all(), "non-finite values in observation"
    b = unpack(obs)
    assert (b["rays"][..., 0] >= 0).all(), "block_class must be non-negative"
    assert (b["rays"][..., 0] <= 255).all()
    assert (b["self"][..., 10] <= 1).all() and (b["self"][..., 10] >= 0).all()


def validate_actions(acts: np.ndarray) -> None:
    acts = np.asarray(acts)
    assert acts.shape[-1] == N_ACTION_TOKENS, f"shape {acts.shape}"
    assert (acts >= 0).all()
    assert (acts < ACTION_NCLASSES).all(), "action token out of range"


def describe() -> dict:
    """落盘成 JSON，供 Lua 侧与数据校验器读取。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "obs_dim": OBS_DIM,
        "blocks": {
            "rays": {"offset": OFF_RAYS, "count": N_RAYS, "dim": N_RAY_DIM,
                     "features": RAY_FEATURES},
            "self": {"offset": OFF_SELF, "count": 1, "dim": SELF_DIM,
                     "features": SELF_FEATURES},
            "ents": {"offset": OFF_ENT, "count": N_ENT, "dim": ENT_DIM,
                     "features": ENT_FEATURES},
            "audio": {"offset": OFF_AUDIO, "count": N_AUDIO, "dim": AUDIO_DIM,
                      "features": AUDIO_FEATURES},
            "task": {"offset": OFF_TASK, "count": 1, "dim": TASK_DIM,
                     "features": TASK_FEATURES},
        },
        "action_tokens": [{"name": n, "n_classes": int(c)}
                          for n, c in ACTION_TOKENS],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), ensure_ascii=False, indent=2))
