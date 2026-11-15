"""边训练边看：BC 训练 + 定期在真环境评测 + 窗口回放。

跑法：
  python scripts/watch_train.py --data data_room --model gru --steps 8000 \
      --eval-every 800 --eval-episodes 2 --pmul 1 --window

为什么是回放而不是 render_mode="human"：Craftium 的 human 模式会渲染每一帧，
物理时间步跟着真实时间走，帧率被拉到 30–60 fps 后玩家每步走出的距离变成训练时的
数倍。实测教师在有窗口时 499 个决策都到不了目标，无窗口 192 个就到。
所以评测保持无渲染（与训练完全一致），把每决策的画面记下来，训练间隙在窗口里回放。
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
from runlib import (make_model, policy_actions, train_step, n_params,  # noqa: E402
                    balanced_weights)
from schema_v0 import OFF_TASK  # noqa: E402

WINDOW = 16
MAX_CTX = 32
WINDOW_SIZE = (640, 640)


class Replay:
    """用一个 pygame 窗口回放评测帧。"""

    def __init__(self):
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "150,150")
        import pygame
        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("ced trainer — 等待第一次评测")

    def play(self, frames, title, fps=50):
        pg = self.pygame
        for px, cap in frames:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    return
            surf = pg.image.frombuffer(px.tobytes(), (px.shape[1], px.shape[0]), "RGB")
            self.screen.blit(pg.transform.scale(surf, WINDOW_SIZE), (0, 0))
            pg.display.set_caption(f"{title} | {cap}")
            pg.display.flip()
            self.clock.tick(fps)
        pg.display.set_caption(f"{title} | 训练中…")

    def close(self):
        self.pygame.quit()


def evaluate(model, env, seeds, max_decisions, record=False):
    model.eval()
    rows, frames = [], []
    for seed in seeds:
        obs = env.reset(seed=seed)
        buf = [obs]
        success, dec = False, 0
        for dec in range(1, max_decisions + 1):
            act = policy_actions(model, np.stack(buf[-MAX_CTX:]))
            obs, reward, term, trunc, info = env.step(act)
            buf.append(obs)
            if record and env.last_pixels is not None:
                d = math.hypot(obs[OFF_TASK + 1], obs[OFF_TASK + 2]) * 32
                frames.append((env.last_pixels, f"ep {seed} decision {dec} dist {d:.1f}"))
            if term or trunc:
                success = bool(term)
                break
        dist = math.hypot(obs[OFF_TASK + 1], obs[OFF_TASK + 2]) * 32
        rows.append((success, dec, dist))
    model.train()
    return rows, frames


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
    ap.add_argument("--start-bias", type=float, default=0.35,
                    help="按概率从 episode 开头附近取窗口，否则学不会开局转向")
    ap.add_argument("--balance-alpha", type=float, default=0.5,
                    help="类别频率加权指数，0 关闭（不加权时模型会退化成永不转向）")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=2)
    ap.add_argument("--max-decisions", type=int, default=300)
    ap.add_argument("--window", action="store_true", help="开窗口回放评测")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = args.out or os.path.join("runs", f"{time.strftime('%Y%m%d-%H%M%S')}-watch-{args.model}")
    os.makedirs(out, exist_ok=True)

    store = shards.EpisodeStore(args.data, args.task)
    splits = store.splits()
    print(f"episodes {{'train': {len(splits['train'])}, 'val': {len(splits['val'])}, "
          f"'test': {len(splits['test'])}}}", flush=True)
    if not splits["train"]:
        raise SystemExit(f"{args.data}/{args.task} 没有训练数据")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = make_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"[{args.model}] params={n_params(model)/1e6:.2f}M device={device}", flush=True)

    env = craftium_env.CraftiumCedEnv(
        seed=0, env_name=args.env, pmul=args.pmul,
        minetest_conf={"ced_goal_min": args.goal_min, "ced_goal_max": args.goal_max})
    replay = Replay() if args.window else None
    val_seeds = [8 + 10 * i for i in range(args.eval_episodes)]  # val：seed % 10 == 8

    class_weights = None
    if args.balance_alpha > 0:
        acts = np.concatenate([shards.read_episode(p)["act"] for p in splits["train"]])
        class_weights = balanced_weights(acts, args.balance_alpha)
        print(f"balanced loss on (alpha={args.balance_alpha}); "
              f"yaw weights: center {class_weights[11][60]:.2f} vs edge {class_weights[11][0]:.2f}",
              flush=True)

    t0 = time.time()
    losses = []
    best = -1.0
    for step in range(1, args.steps + 1):
        obs, act = store.sample_batch(splits["train"], args.batch, WINDOW, rng,
                                      start_bias=args.start_bias)
        losses.append(train_step(model, opt, obs, act, class_weights=class_weights))
        if step % args.eval_every == 0 or step == args.steps:
            loss = float(np.mean(losses[-50:]))
            rows, frames = evaluate(model, env, val_seeds, args.max_decisions,
                                    record=replay is not None)
            succ_arr = np.array([r[0] for r in rows], dtype=float)
            succ = float(succ_arr.mean())
            line = (f"step {step:5d}  loss {loss:.3f}  success {succ:.0%}"
                    f"±{succ_arr.std():.2f}  "
                    f"decisions {np.mean([r[1] for r in rows]):.0f}  "
                    f"final_dist {np.mean([r[2] for r in rows]):.2f}")
            print(line, flush=True)
            if succ > best:
                best = succ
                torch.save({"model": model.state_dict(), "name": args.model},
                           os.path.join(out, "best.pt"))
            if replay is not None:
                replay.play(frames, f"step {step} loss {loss:.3f} success {succ:.0%}")

    print(f"done in {time.time()-t0:.0f}s, best success {best:.0%}, ckpt -> {out}/best.pt")
    if replay is not None:
        replay.close()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
