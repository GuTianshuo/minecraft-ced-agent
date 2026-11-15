"""S9 验收：14-token 动作在 Craftium 上的语义。

跑法：python scripts/check_craftium_env.py

检查：
  1. 观测是 2568 维合法帧
  2. yaw / pitch token 生效且和帧里的 self 段一致
  3. forward token 真的让玩家沿朝向移动（atan2(dx,dz) ≈ -yaw）
  4. forward + backward 互相抵消
  5. 一次决策消耗多少帧（观测帧率 58 Hz / 步进 200 Hz）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import craftium  # noqa: F401,E402  —— 注册 Craftium/* 环境
import craftium_env  # noqa: E402
from schema_v0 import OFF_SELF, N_ACTION_TOKENS, validate_obs  # noqa: E402


def tokens(fwd=0, back=0, yaw=60, pitch=60, hold=0, **keys):
    a = np.zeros(N_ACTION_TOKENS, dtype=np.int64)
    a[0], a[1] = fwd, back
    a[11], a[12], a[13] = yaw, pitch, hold
    for name, idx in (("jump", 4), ("sneak", 5), ("attack", 7), ("use", 8)):
        if name in keys:
            a[idx] = keys[name]
    return a


def look_of(obs):
    return (float(np.degrees(np.arctan2(obs[OFF_SELF + 3], obs[OFF_SELF + 4]))),
            float(np.degrees(np.arctan2(obs[OFF_SELF + 5], obs[OFF_SELF + 6]))))


def pos_of(obs):
    return np.array([obs[OFF_SELF + 0], obs[OFF_SELF + 1], obs[OFF_SELF + 2]]) * 32.0


def main() -> int:
    env = craftium_env.CraftiumCedEnv(seed=0)
    obs = env.reset(seed=0)
    validate_obs(obs)
    print(f"reset ok: obs {obs.shape}, look {np.round(look_of(obs), 1)} deg")

    # 1) yaw token：+60 桶 = +90°
    y0, _ = look_of(obs)
    obs, *_ , info = env.step(tokens(yaw=120))
    y1, _ = look_of(obs)
    print(f"yaw token 120: {y0:+.1f} -> {y1:+.1f} deg (delta {y1-y0:+.1f}, expect +90)")
    yaw_ok = abs(((y1 - y0) + 180) % 360 - 180 - 90) < 8

    # 2) pitch token：-60 桶 = -90°，应被夹在 ±89.4°
    _, p0 = look_of(obs)
    obs, *_, info = env.step(tokens(yaw=60, pitch=0))
    _, p1 = look_of(obs)
    print(f"pitch token 0: {p0:+.1f} -> {p1:+.1f} deg (clamped)")
    pitch_ok = p1 < -80 and p1 > -90.0

    # 3) forward：沿 -yaw 方向移动
    env.reset(seed=0)
    obs, *_ = env.step(tokens(yaw=60, pitch=60))
    p0 = pos_of(obs)
    for _ in range(10):
        obs, *_ = env.step(tokens(fwd=1, yaw=60, pitch=60))
    p1 = pos_of(obs)
    d = p1 - p0
    yaw_now, _ = look_of(obs)
    move_dir = float(np.degrees(np.arctan2(d[0], d[2])))
    print(f"forward x10: displacement {np.linalg.norm(d):.2f} nodes, dir {move_dir:+.1f} deg, "
          f"look yaw {yaw_now:+.1f} deg")
    move_ok = np.linalg.norm(d) > 0.5 and abs(((move_dir + yaw_now) + 180) % 360 - 180) < 35

    # 4) forward + backward 抵消（先停下来，避免残余速度干扰）
    for _ in range(12):
        obs, *_ = env.step(tokens(yaw=60, pitch=60))
    p0 = pos_of(obs)
    for _ in range(5):
        obs, *_ = env.step(tokens(fwd=1, back=1, yaw=60, pitch=60))
    d = pos_of(obs) - p0
    print(f"forward+backward x5: displacement {np.linalg.norm(d):.3f} nodes")
    cancel_ok = np.linalg.norm(d) < 0.05

    print(f"frames per decision: {info['frames']} (yaw/pitch/fwd tests)")
    env.close()

    ok = yaw_ok and pitch_ok and move_ok and cancel_ok
    print("CRAFTIUM ENV OK" if ok else "CRAFTIUM ENV FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
