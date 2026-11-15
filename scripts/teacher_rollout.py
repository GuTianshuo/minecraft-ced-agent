"""跑教师，检查它在 Craftium 上能不能到达目标。

跑法：python scripts/teacher_rollout.py --episodes 5 --max-decisions 120
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import craftium  # noqa: F401,E402
import craftium_env  # noqa: E402
from schema_v0 import OFF_TASK, validate_obs  # noqa: E402
from teachers import goto_action  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-decisions", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    env = craftium_env.CraftiumCedEnv(seed=args.seed0,
                                      **({"render_mode": "human"} if args.render else {}))
    wins = 0
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        obs = env.reset(seed=seed)
        validate_obs(obs)
        d0 = math.hypot(obs[OFF_TASK + 1], obs[OFF_TASK + 2]) * 32
        done = False
        dec = 0
        for dec in range(1, args.max_decisions + 1):
            act = goto_action(obs)
            obs, reward, term, trunc, info = env.step(act)
            if term or trunc:
                done = bool(term)
                break
        dist = math.hypot(obs[OFF_TASK + 1], obs[OFF_TASK + 2]) * 32
        wins += int(done)
        print(f"ep {ep} seed {seed}: {'REACHED' if done else 'timeout'} "
              f"decisions={dec} start_dist={d0:.1f} final_dist={dist:.2f}")
    env.close()
    print(f"teacher reached {wins}/{args.episodes}")
    return 0 if wins else 1


if __name__ == "__main__":
    raise SystemExit(main())
