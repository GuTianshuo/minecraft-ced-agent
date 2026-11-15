"""从分片做行为克隆训练 + 定期评测 + 门禁检查。

  python bc.py --model gru --task T0 --data data --steps 2000 --eval-every 500

与 train.py 的区别：train.py 是空转自检（内存里合成数据，只验证框架健康），
bc.py 走真实分片、真实切分、真实评测协议，是 P1 起每天用的入口。
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch

import evalx
import shards
import tasks
from runlib import make_model, train_step, rss_mb, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru", choices=["gru", "tf", "ced"])
    ap.add_argument("--task", default="T0")
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--gate", type=float, default=0.85)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or os.path.join("runs", f"{stamp}-{args.model}-{args.task}")
    os.makedirs(out, exist_ok=True)

    store = shards.EpisodeStore(args.data, args.task)
    splits = store.splits()
    counts = {k: len(v) for k, v in splits.items()}
    print(f"episodes {counts}")
    if not splits["train"]:
        raise SystemExit(f"{args.data}/{args.task} 里没有训练 episode，先跑 collect.py")
    if not splits["val"]:
        print("!! 没有 val 划分，评测将不可信：检查种子分布（seed%10==8 为 val）")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = make_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        print(f"resumed from {args.resume} @ step {start_step}")

    print(f"[{args.model}] params={n_params(model)/1e6:.2f}M device={device} "
          f"window={args.window} batch={args.batch}")
    print(f"log -> {out}")

    csv_path = os.path.join(out, "train.csv")
    new_file = not os.path.exists(csv_path)
    f_log = open(csv_path, "a", newline="")
    writer = csv.writer(f_log)
    if new_file:
        writer.writerow(["step", "loss", "rss_mb", "elapsed_s",
                         "success_mean", "success_std", "usable"])

    rss0 = rss_mb()
    t0 = time.time()
    losses = []
    best = -1.0
    last_eval = None

    for step in range(start_step + 1, args.steps + 1):
        obs, act = store.sample_batch(splits["train"], args.batch, args.window, rng)
        losses.append(train_step(model, opt, obs, act))

        if step % args.eval_every == 0 or step == args.steps:
            res = evalx.evaluate(model, args.task,
                                 make_env=lambda s: tasks.make_task(args.task, seed=s),
                                 seeds=(0, 1, 2), episodes=args.eval_episodes)
            last_eval = res
            writer.writerow([step, f"{np.mean(losses[-50:]):.5f}", f"{rss_mb():.0f}",
                             f"{time.time()-t0:.1f}", f"{res['success_mean']:.4f}",
                             f"{res['success_std']:.4f}", res["usable"]])
            f_log.flush()
            print(f"step {step:5d} loss {np.mean(losses[-50:]):.4f} "
                  f"succ {res['success_mean']:.1%}±{res['success_std']:.3f} "
                  f"rss {rss_mb():.0f}MB")
            if res["success_mean"] > best:
                best = res["success_mean"]
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "name": args.model},
                           os.path.join(out, "best.pt"))

    f_log.close()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": args.steps, "name": args.model},
               os.path.join(out, "last.pt"))
    print(f"saved -> {out}/last.pt  (elapsed {time.time()-t0:.0f}s, "
          f"rss+{rss_mb()-rss0:.0f}MB)")

    if last_eval is not None:
        print(evalx.format_table(last_eval))
        ok, why = evalx.check_gate(last_eval, args.gate)
        print(("GATE PASS" if ok else "GATE FAIL") + f": {why}")


if __name__ == "__main__":
    main()
