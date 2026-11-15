"""按 0001 的口径评测一个 checkpoint，并支持观测消融。

跑法：
  python scripts/eval_policy.py --ckpt runs/.../best.pt --model gru --split test
  python scripts/eval_policy.py --ckpt ... --ablate rays     # 射线全清零
  python scripts/eval_policy.py --ckpt ... --ablate task     # 目标向量清零
  python scripts/eval_policy.py --ckpt ... --ablate selfpos  # 自身位置/速度清零

为什么要消融：成功率只说明"它到了"，不说明"它靠什么到的"。
把某一段观测清零后看成功率掉多少，才知道模型是真的用了射线，还是只靠 task 段。

0001 的统计口径：3 个种子起步，种子间成功率标准差 > 0.10 视为结果不可用。
本脚本按 split 取种（seed % 10 == 8 为 val，== 9 为 test），逐种子报一行。
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import craftium  # noqa: F401,E402
import craftium_env  # noqa: E402
import shards  # noqa: E402
from runlib import make_model, policy_actions  # noqa: E402
from schema_v0 import OFF_SELF, OFF_TASK  # noqa: E402

MAX_CTX = 32


def seeds_of(split: str, limit: int = 8) -> list[int]:
    rem = {"train": range(0, 8), "val": (8,), "test": (9,)}[split]
    seeds = [s for s in range(80) if s % 10 in rem]
    return seeds[:limit]


def ablate(obs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return obs
    obs = obs.copy()
    if mode == "rays":
        obs[:OFF_SELF] = 0.0
    elif mode == "task":
        obs[OFF_TASK:] = 0.0
    elif mode == "selfpos":
        # 位置与速度清零，朝向与自身状态保留
        obs[OFF_SELF + 0:OFF_SELF + 3] = 0.0
        obs[OFF_SELF + 7:OFF_SELF + 10] = 0.0
    else:
        raise ValueError(mode)
    return obs


def run_episode(model, env, seed: int, max_decisions: int, mode: str):
    obs = env.reset(seed=seed)
    buf = [ablate(obs, mode)]
    success, dec = False, 0
    for dec in range(1, max_decisions + 1):
        act = policy_actions(model, np.stack(buf[-MAX_CTX:]))
        obs, reward, term, trunc, info = env.step(act)
        buf.append(ablate(obs, mode))
        if term or trunc:
            success = bool(term)
            break
    return success, dec, math.hypot(obs[OFF_TASK + 1], obs[OFF_TASK + 2]) * 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="gru", choices=["gru", "tf", "ced"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--ablate", default="none",
                    choices=["none", "rays", "task", "selfpos"])
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--env", default="room")
    ap.add_argument("--pmul", type=int, default=1)
    ap.add_argument("--goal-min", type=float, default=5.0)
    ap.add_argument("--goal-max", type=float, default=8.0)
    args = ap.parse_args()

    model = make_model(args.model)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()

    env = craftium_env.CraftiumCedEnv(
        seed=0, env_name=args.env, pmul=args.pmul,
        minetest_conf={"ced_goal_min": args.goal_min, "ced_goal_max": args.goal_max})

    seeds = seeds_of(args.split, args.seeds)
    rows = []
    print(f"split={args.split} ablate={args.ablate} seeds={seeds}")
    for s in seeds:
        ok, dec, dist = run_episode(model, env, s, args.max_decisions, args.ablate)
        rows.append((ok, dec, dist))
        print(f"  seed {s:3d}  {'REACHED' if ok else 'failed '}  decisions {dec:3d}  "
              f"final_dist {dist:5.2f}")
    env.close()

    succ = np.array([r[0] for r in rows], dtype=float)
    mean, std = float(succ.mean()), float(succ.std())
    print(f"success {mean:.1%}  std {std:.3f}  "
          f"decisions {np.mean([r[1] for r in rows]):.0f}  "
          f"final_dist {np.mean([r[2] for r in rows]):.2f}  "
          f"{'usable' if std <= 0.10 else 'UNUSABLE (std > 0.10)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
