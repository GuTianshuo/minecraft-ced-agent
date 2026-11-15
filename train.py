"""P0a 空转自检：合成环境 → 行为克隆 → 贪心评测 → checkpoint。

  python train.py --model gru --steps 300 --smoke
  python train.py --model tf  --steps 300 --smoke
  python train.py --model ced --steps 300 --smoke

它只验证"训练框架是否健康"。真任务的数据管线见 collect.py / bc.py。
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from schema_v0 import OBS_DIM
from envs import DummyNavEnv, rollout_teacher
from runlib import make_model, policy_actions, rss_mb, train_step, n_params

WINDOW = 16
MAX_CTX = 32


def collect(n_episodes: int = 400, seed: int = 0):
    env = DummyNavEnv(seed=seed)
    eps = []
    for i in range(n_episodes):
        obs, act = rollout_teacher(env, seed=seed + i + 1)
        if len(act) >= 2:
            eps.append((obs, act))
    return eps


def sample_batch(eps, batch: int, rng: np.random.Generator):
    obs_b, act_b = [], []
    for _ in range(batch):
        obs, act = eps[rng.integers(len(eps))]
        t = len(act)
        if t < WINDOW:
            pad = WINDOW - t
            obs = np.concatenate([obs, np.repeat(obs[-1:], pad, axis=0)])
            act = np.concatenate([act, np.repeat(act[-1:], pad, axis=0)])
            start = 0
        else:
            start = int(rng.integers(0, t - WINDOW + 1))
        obs_b.append(obs[start:start + WINDOW])
        act_b.append(act[start:start + WINDOW])
    return (torch.as_tensor(np.stack(obs_b), dtype=torch.float32),
            torch.as_tensor(np.stack(act_b), dtype=torch.long))


def evaluate(model, n_ep: int = 20, seed: int = 10_000) -> float:
    env = DummyNavEnv(seed=seed)
    wins = []
    model.eval()
    for ep in range(n_ep):
        obs = env.reset(seed=seed + ep)
        buf = [obs]
        success = False
        for _ in range(env.max_steps):
            act = policy_actions(model, np.stack(buf[-MAX_CTX:]))
            obs, _, term, trunc, _ = env.step(act)
            buf.append(obs)
            success = success or term
            if term or trunc:
                break
        wins.append(success)
    model.train()
    return float(np.mean(wins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru", choices=["gru", "tf", "ced"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = make_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[{args.model}] params={n_params(model)/1e6:.2f}M  obs_dim={OBS_DIM}  "
          f"device={device}")

    eps = collect(args.episodes)
    print(f"collected {len(eps)} episodes, "
          f"{sum(len(a) for _, a in eps)} decision steps")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    rss0 = rss_mb()
    first, losses = None, []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        obs, act = sample_batch(eps, args.batch, rng)
        losses.append(train_step(model, opt, obs, act))
        if first is None and step == 10:
            first = float(np.mean(losses[-10:]))
            # 基线取在这里：前 10 步会把 CUDA 上下文与 cublas/cudnn workspace
            # 一次性分配掉（GRU 约 800MB），它不是泄漏，不能算进增长。
            rss0 = rss_mb()
        if step % max(1, args.steps // 6) == 0 or step == 1:
            print(f"step {step:4d}  loss {np.mean(losses[-20:]):.4f}  "
                  f"rss {rss_mb():.0f}MB  {time.time()-t0:.1f}s")

    tail = float(np.mean(losses[-20:]))
    print(f"loss first10={first:.4f} tail={tail:.4f}  rss+{rss_mb()-rss0:.0f}MB")
    acc = evaluate(model, n_ep=20)
    print(f"greedy success rate (20 eps) = {acc:.2%}")

    if args.ckpt:
        torch.save({"model": model.state_dict(), "name": args.model}, args.ckpt)
        m2 = make_model(args.model)
        m2.load_state_dict(torch.load(args.ckpt)["model"])
        print(f"checkpoint roundtrip ok -> {args.ckpt}")

    if args.smoke:
        assert np.isfinite(tail), "loss is not finite"
        assert tail < first * 0.95, f"loss did not decrease: {first:.3f} -> {tail:.3f}"
        assert rss_mb() - rss0 < 400, "memory grew unexpectedly"
        print("SMOKE OK")


if __name__ == "__main__":
    main()
