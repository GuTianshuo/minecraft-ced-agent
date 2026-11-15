"""S7 spike：验证 ced_obs 在 Craftium 上按 tick 写出 2568 维结构化帧。

跑法：python scripts/spike_obs.py [--steps 600]

检查四件事：
  1. 帧文件存在，且 tick 在跑动中前进（不是 reset 时写一次就停）
  2. 帧通过 schema_v0.validate_obs，并且有非零射线（说明光线投射真的命中东西）
  3. Lua globalstep 的观测更新率与 Python 步进率的关系（这决定 S9 怎么对齐）
  4. mod 单 tick 的 build+write 耗时（EXECUTION S7 的门禁是 5 ms）
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import craftium  # noqa: F401  —— 必须先 import 才会注册 Craftium/* 环境
import craftium_env  # noqa: E402
from schema_v0 import validate_obs  # noqa: E402
from transport import ObsChannel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    env = craftium_env.make_env(seed=0)
    t0 = time.time()
    env.reset()
    reset_s = time.time() - t0

    path = craftium_env.obs_frame_path(env)
    stats_path = os.path.join(os.path.dirname(path), "ced_obs_stats.txt")

    first = ObsChannel(path, timeout=5.0).read()
    validate_obs(first.obs)

    t0 = time.time()
    for _ in range(args.steps):
        env.step(env.action_space.sample())
    dt = time.time() - t0

    last = ObsChannel(path, timeout=5.0).read(after_tick=first.tick)
    validate_obs(last.obs)
    stats = open(stats_path).read().strip() if os.path.exists(stats_path) else "(no stats)"

    lua_ticks = last.tick - first.tick
    print(f"reset {reset_s*1000:.0f}ms  steps {args.steps} in {dt:.2f}s "
          f"= {args.steps/dt:.1f} steps/s")
    print(f"frame: tick {first.tick} -> {last.tick} (+{lua_ticks} over {dt:.2f}s "
          f"= {lua_ticks/dt:.1f} ticks/s), dim {last.obs.shape[0]}, "
          f"nonzero {int((last.obs != 0).sum())}, file {os.path.getsize(path)} bytes")
    print(f"steps per observed frame: {lua_ticks and args.steps/lua_ticks:.2f}")
    print(f"mod timing: {stats}")

    env.close()

    ok = lua_ticks > 0 and last.obs.shape[0] == 2568 and int((last.obs != 0).sum()) > 0
    print("SPIKE OK" if ok else "SPIKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
