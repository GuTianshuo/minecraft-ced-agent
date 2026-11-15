"""边训练边看：BC 训练 + 定期在窗口里用当前模型跑评测。

跑法：
  python scripts/watch_train.py --data data_room --model gru --steps 3000 --eval-every 500

窗口里是 agent 的第一视角（Craftium 的 render_mode="human"），标题栏实时显示
step / loss / 成功率。终端同时打印同样的数字。

训练本身不占窗口；窗口只在评测时打开，看到的是"当前模型在真环境里怎么走"，
这是判断它有没有学到东西最直接的方式。
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import craftium  # noqa: F401,E402
import craftium_env  # noqa: E402
import shards  # noqa: E402
from runlib import make_model, policy_actions, train_step, n_params  # noqa: E402
from schema_v0 import OFF_TASK  # noqa: E402

WINDOW = 16
MAX_CTX = 32


def evaluate(model, env, seeds, max_decisions, caption=None):
    model.eval()
    rows = []
    for seed in seeds:
        obs = env.reset(seed=seed)
        buf = [obs]
        success, dec = False, 0
        for dec in range(1, max_decisions + 1):
            act = policy_actions(model, np.stack(buf[-MAX_CTX:]))
            obs, reward, term, trunc, info = env.step(act)
            buf.append(obs)
            if term or trunc:
                success = bool(term)
                break
            if caption is not None:
                _set_caption(f"{caption} | episode {seed} | decision {dec} | "
                             f"dist {math.hypot(obs[OFF_TASK+1], obs[OFF_TASK+2])*32:.1f}")
        dist = math.hypot(obs[OFF_TASK + 1], obs[OFF_TASK + 2]) * 32
        rows.append((success, dec, dist))
    model.train()
    return rows


def _set_caption(text):
    try:
        import pygame
        pygame.display.set_caption(text)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_room")
    ap.add_argument("--task", default="T0")
    ap.add_argument("--model", default="gru", choices=["gru", "tf", "ced"])
    ap.add_argument("--env", default="room")
    ap.add_argument("--goal-min", type=float, default=5.0)
    ap.add_argument("--goal-max", type=float, default=8.0)
    ap.add_argument("--pmul", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=2)
    ap.add_argument("--max-decisions", type=int, default=120)
    ap.add_argument("--window", action="store_true", help="开窗口看评测")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = args.out or os.path.join("runs", f"{time.strftime('%Y%m%d-%H%M%S')}-watch-{args.model}")
    os.makedirs(out, exist_ok=True)

    store = shards.EpisodeStore(args.data, args.task)
    splits = store.splits()
    counts = {k: len(v) for k, v in splits.items()}
    print(f"episodes {counts}")
    if not splits["train"]:
        raise SystemExit(f"{args.data}/{args.task} 没有训练数据")
    if not splits["val"]:
        print("!! 没有 val 划分（seed%10==8 为 val）")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = make_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"[{args.model}] params={n_params(model)/1e6:.2f}M device={device}")

    env_kwargs = dict(
        env_name=args.env,
        pmul=args.pmul,
        minetest_conf={"ced_goal_min": args.goal_min, "ced_goal_max": args.goal_max},
    )
    if args.window:
        env_kwargs["render_mode"] = "human"
    env = craftium_env.CraftiumCedEnv(seed=0, **env_kwargs)

    val_seeds = [8 + 10 * i for i in range(args.eval_episodes)]

    def dashboard(step, loss, rows):
        succ = np.mean([r[0] for r in rows])
        dec = np.mean([r[1] for r in rows])
        dist = np.mean([r[2] for r in rows])
        line = (f"step {step:5d}  loss {loss:.3f}  success {succ:.0%}  "
                f"decisions {dec:.0f}  final_dist {dist:.2f}")
        print(line, flush=True)
        _set_caption(f"{line}   {args.model}/{args.env}")

    t0 = time.time()
    losses = []
    best = -1.0
    for step in range(1, args.steps + 1):
        obs, act = store.sample_batch(splits["train"], args.batch, WINDOW, rng)
        losses.append(train_step(model, opt, obs, act))
        if step % args.eval_every == 0 or step == args.steps:
            rows = evaluate(model, env, val_seeds, args.max_decisions,
                            caption=None if not args.window else f"step {step}")
            dashboard(step, float(np.mean(losses[-50:])), rows)
            succ = float(np.mean([r[0] for r in rows]))
            if succ > best:
                best = succ
                torch.save({"model": model.state_dict(), "name": args.model},
                           os.path.join(out, "best.pt"))

    print(f"done in {time.time()-t0:.0f}s, best success {best:.0%}, ckpt -> {out}/best.pt")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
