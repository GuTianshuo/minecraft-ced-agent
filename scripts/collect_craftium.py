"""在 Craftium 上用随机策略采数据，写成分片（S10 前半段）。

跑法：python scripts/collect_craftium.py --episodes 3 --decisions 150 --out data_craftium

这一步只验证「真引擎 → 分片 → 训练」的链路，不验证学习：策略是均匀随机的，
任务奖励还没接（见 0006 的待办）。分片格式与 collect.py 完全一致，
所以 bc.py / shards.py 不用改。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import craftium  # noqa: F401,E402
import craftium_env  # noqa: E402
import shards  # noqa: E402
from schema_v0 import ACTION_NCLASSES  # noqa: E402


def random_tokens(rng):
    return np.array([int(rng.integers(0, int(c))) for c in ACTION_NCLASSES], dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--decisions", type=int, default=150)
    ap.add_argument("--out", default="data_craftium")
    ap.add_argument("--task", default="T0")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    env = craftium_env.CraftiumCedEnv(seed=args.seed0)
    paths = []
    for ep in range(args.episodes):
        seed = args.seed0 + ep
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=seed)
        obs_seq, act_seq = [obs], []
        for _ in range(args.decisions):
            a = random_tokens(rng)
            obs, reward, term, trunc, info = env.step(a)
            obs_seq.append(obs)
            act_seq.append(a)
            if term or trunc:
                break
        # 与 rollout_teacher 一致：最后一帧只作为动作目标，不作为输入
        obs_arr, act_arr = np.stack(obs_seq[:-1]), np.stack(act_seq)
        p = shards.write_episode(args.out, args.task, seed, obs_arr, act_arr, "random")
        paths.append(p)
        print(f"episode {ep}: {len(act_arr)} decisions, {info['frames']} frames/decision, "
              f"tick={info['tick']} -> {os.path.basename(p)}")
    env.close()

    store = shards.EpisodeStore(args.out, args.task)
    s = store.stats()
    print(f"wrote {len(paths)} episodes, steps={s['steps']}, "
          f"{s['bytes']/2**20:.1f} MiB, {s['bytes_per_step']:.0f} bytes/step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
