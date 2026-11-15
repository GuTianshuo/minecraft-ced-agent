"""采集 episode 到分片。

  python collect.py --task T0 --episodes 200 --out data
  python collect.py --task T0 --episodes 20  --out data --policy random
  python collect.py --task T0 --out data --stats      # 只看现有数据

采完会打印真实 bytes/step 与按此推算的日增量 —— 磁盘账本必须由实测得出，
不能用估算值直接排期。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import tasks
import shards
from envs import rollout_teacher
from schema_v0 import ACTION_NCLASSES


def random_policy(rng):
    """均匀随机但合法的动作，用于验证管线而不是验证学习。"""
    return np.array([int(rng.integers(0, int(c))) for c in ACTION_NCLASSES],
                    dtype=np.int64)


def collect(out: str, task_id: str, episodes: int, policy: str = "teacher",
            seed0: int = 0, verbose: bool = True) -> list[str]:
    paths = []
    for i in range(episodes):
        seed = seed0 + i
        env = tasks.make_task(task_id, seed=seed)
        if policy == "teacher":
            obs, act = rollout_teacher(env, max_steps=env.max_steps, seed=seed)
        elif policy == "random":
            obs, act = _rollout_random(env, seed)
        else:
            raise ValueError(policy)
        if len(act) < 2:
            continue
        paths.append(shards.write_episode(out, task_id, seed, obs, act, policy))
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i+1}/{episodes} episodes")
    return paths


def _rollout_random(env, seed):
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=seed)
    obs_seq, act_seq = [obs], []
    while True:
        a = random_policy(rng)
        obs, _, term, trunc, _ = env.step(a)
        obs_seq.append(obs)
        act_seq.append(a)
        if term or trunc or len(act_seq) >= env.max_steps:
            break
    return np.stack(obs_seq[:-1]), np.stack(act_seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T0")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default="data")
    ap.add_argument("--policy", default="teacher", choices=["teacher", "random"])
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--stats", action="store_true", help="只打印现有数据统计")
    ap.add_argument("--max-gb", type=float, default=80.0)
    args = ap.parse_args()

    if args.stats:
        store = shards.EpisodeStore(args.out, args.task)
        s = store.stats()
        print(s)
        print(f"按 8 环境 × 20 步/s 推算：{s['bytes_per_step']*8*20*3600/2**30:.2f} GiB/h")
        return

    paths = collect(args.out, args.task, args.episodes, args.policy, args.seed0)
    store = shards.EpisodeStore(args.out, args.task)
    s = store.stats()
    splits = {k: len(v) for k, v in store.splits().items()}
    print(f"wrote {len(paths)} episodes -> {splits}")
    print(f"steps={s['steps']}  bytes={s['bytes']/2**20:.1f} MiB  "
          f"bytes/step={s['bytes_per_step']:.0f} "
          f"(未压缩理论值 {s['bytes_per_step_uncompressed']})")

    daily = s["bytes_per_step"] * 8 * 20 * 86400 / 2**30
    print(f"8 环境连续采样的日增量约 {daily:.1f} GiB/天")
    if daily * 14 > args.max_gb:
        print(f"!! 两周就超过 {args.max_gb} GiB 配额：先降采样频率或只存决策步子集",
              file=sys.stderr)


if __name__ == "__main__":
    main()
